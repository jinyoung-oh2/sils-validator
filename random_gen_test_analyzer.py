#!/usr/bin/env python3
# File: random_gen_test_analyzer.py

import os
import csv
import concurrent.futures
import shutil
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

# Helper function: 리스트를 batch로 나눔
def batchify(lst, batch_size):
    """주어진 리스트 lst를 batch_size 크기의 배치로 분할하여 반환"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

# ================= Fail 분석 관련 =================

class RandomGenTestAnalyzer(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터를 분석하는 클래스입니다.
    각 파일별로 ca_path_gen_fail 및 isCloseTargetDetected 플래그를 기준으로 
    "N/A (safe target)", "N/A (path gen failure)", "fail (closed target)",
    "fail (path gen failure)" 또는 "success"를 판정하며, 실패 시 해당 이벤트 내 모든 
    target_ships의 CPA, TCPA, SOG 정보를 함께 반환합니다.
    """
    def __init__(self):
        super().__init__()
        self.safty_distance = 0.5 + 250/1852

    def analyze_dataset(self):
        n = len(self.events)
        if n == 0:
            return {"result": "N/A (safe target)"}
        elif n == 1:
            event = self.events[0]
            # ca_path_gen_fail 여부만 확인
            if event.get("ca_path_gen_fail") is True:
                return {"result": "N/A (path gen failure)"}
            else:
                return {"result": "success"}
        else:
            # 한 번 순회하면서 이벤트를 분리함
            close_events = []
            ca_fail_events = []
            for event in self.events:
                if event.get("isCloseTargetDetected") is True:
                    close_events.append(event)
                elif event.get("ca_path_gen_fail") is True:
                    ca_fail_events.append(event)
            # 우선, isCloseTargetDetected가 있는 경우 처리
            if close_events:
                failed_cpa_s = []
                failed_tcpa_s = []
                failed_sogs = []
                for event in close_events:
                    for ship in event.get("target_ships", []):
                        if ship.get("cpa", -1) < self.safty_distance :
                            failed_cpa_s.append(ship.get("cpa"))
                            failed_tcpa_s.append(ship.get("tcpa"))
                            failed_sogs.append(ship.get("sog"))
                return {"result": "fail (closed target)",
                        "failed_cpa": failed_cpa_s,
                        "failed_tcpa": failed_tcpa_s,
                        "failed_sog": failed_sogs}
            # 그 다음 ca_path_gen_fail가 있는 경우 처리
            elif ca_fail_events:
                failed_cpa_s = []
                failed_tcpa_s = []
                failed_sogs = []
                for event in ca_fail_events:
                    for ship in event.get("target_ships", []):
                        if ship.get("cpa", -1) < self.safty_distance :
                            failed_cpa_s.append(ship.get("cpa"))
                            failed_tcpa_s.append(ship.get("tcpa"))
                            failed_sogs.append(ship.get("sog"))
                return {"result": "fail (path gen failure)",
                        "failed_cpa": failed_cpa_s,
                        "failed_tcpa": failed_tcpa_s,
                        "failed_sog": failed_sogs}
            else:
                return {"result": "success"}

    def analyze_file(self, marzip_file):
        self.marzip = marzip_file
        self.run(marzip_file)
        result = self.analyze_dataset()
        return result

def process_single_fail_file(marzip_file):
    """
    단일 마집 파일에 대해 RandomGenTestAnalyzer를 실행합니다.
    결과는 {"Folder": ..., "File": ..., "Result": ..., (선택적)failed_cpa, failed_tcpa, failed_sog} 형태로 반환합니다.
    """
    try:
        analyzer = RandomGenTestAnalyzer()
        analysis = analyzer.analyze_file(marzip_file)
        base_name = os.path.splitext(os.path.basename(marzip_file))[0]
        parent_dir = os.path.dirname(marzip_file)
        grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
        folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
        result_dict = {
            "Folder": folder_info,
            "File": base_name,
            "Result": analysis["result"]
        }
        if analysis["result"].lower().startswith("fail") or "n/a" in analysis["result"].lower():
            result_dict["failed_cpa"] = analysis.get("failed_cpa", [])
            result_dict["failed_tcpa"] = analysis.get("failed_tcpa", [])
            result_dict["failed_sog"] = analysis.get("failed_sog", [])
        return result_dict
    except Exception as e:
        print(f"Error processing {marzip_file}: {e}")
        return None

def write_csv(results, output_csv, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["Folder", "File", "Result", "failed_cpa", "failed_tcpa", "failed_sog"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

class RandomGenTestFailAnalysisRunner:
    """
    이벤트 마집 파일들을 배치 단위로 병렬 처리하여 실패 분석 결과를 CSV 파일로 저장하는 클래스입니다.
    모든 결과는 하나의 CSV 파일에 집계됩니다.
    """
    def __init__(self, event_folder, batch_size=1000):
        self.event_folder = event_folder
        self.batch_size = batch_size
        self.file_manager = FileInputManager(event_folder)

    def run_all_parallel(self):
        marzip_files = self.file_manager.get_all_marzip_files()
        total_files = len(marzip_files)
        print(f"총 {total_files}개의 마집 파일을 찾았습니다.")
        results = []
        batch_num = 0
        start_total = time.time()
        for batch in batchify(marzip_files, self.batch_size):
            batch_num += 1
            start_batch = time.time()
            print(f"Batch {batch_num}: {len(batch)}개 파일 처리 시작...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                batch_results = list(executor.map(process_single_fail_file, batch))
                for res in batch_results:
                    if res is not None:
                        results.append(res)
            end_batch = time.time()
            print(f"Batch {batch_num} 완료. 소요 시간: {end_batch - start_batch:.2f}초")
        end_total = time.time()
        print(f"전체 처리 완료. 총 소요 시간: {end_total - start_total:.2f}초")
        return results

    def run_and_save(self):
        results = self.run_all_parallel()
        common_result_dir = os.path.join(self.event_folder, "result")
        if not os.path.exists(common_result_dir):
            os.makedirs(common_result_dir)
        output_csv = os.path.join(self.event_folder, "all_event_analysis.csv")
        
        count_success = sum(1 for r in results if r["Result"].lower() == "success")
        count_fail_closed = sum(1 for r in results if r["Result"].lower() == "fail (closed target)")
        count_fail_path = sum(1 for r in results if r["Result"].lower() == "fail (path gen failure)")
        count_na_safe = sum(1 for r in results if r["Result"].lower() == "n/a (safe target)")
        count_na_path = sum(1 for r in results if r["Result"].lower() == "n/a (path gen failure)")
        
        fail_total = count_fail_closed + count_fail_path
        na_total = count_na_safe + count_na_path
        total = len(results)
        
        summary_str = (
            f"Total: {total}, Success: {count_success}, "
            f"Fail: {fail_total} (closed: {count_fail_closed}, path gen failure: {count_fail_path}), "
            f"N/A: {na_total} (safe target: {count_na_safe}, path gen failure: {count_na_path})"
        )
        results.append({"Folder": "", "File": "Summary", "Result": summary_str})
        write_csv(results, output_csv)
        print(f"통합 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")
        
        # 실패 이벤트들의 CPA와 TCPA 데이터를 수집하여 히스토그램 생성
        all_failed_cpa = []
        all_failed_tcpa = []
        for r in results:
            if r.get("File") == "Summary":
                continue
            if ("fail" in r["Result"].lower()) or ("n/a" in r["Result"].lower()):
                all_failed_cpa.extend(r.get("failed_cpa", []))
                all_failed_tcpa.extend(r.get("failed_tcpa", []))
        
        # 평탄화: 각 요소가 리스트인 경우를 대비하여 numpy.concatenate 사용
        import numpy as np
        if all_failed_cpa:
            try:
                all_failed_cpa_flat = np.concatenate([np.atleast_1d(x) for x in all_failed_cpa])
            except Exception as e:
                print(f"Flatten CPA error: {e}")
                all_failed_cpa_flat = np.array(all_failed_cpa)
            plt.figure()
            plt.hist(all_failed_cpa_flat, bins=20, color="skyblue", edgecolor="black")
            plt.title("Histogram of Failed CPA")
            plt.xlabel("CPA")
            plt.ylabel("Count")
            cpa_hist_path = os.path.join(common_result_dir, "failed_cpa_histogram.png")
            plt.savefig(cpa_hist_path)
            plt.close()
            print(f"Failed CPA 히스토그램이 {cpa_hist_path}에 저장되었습니다.")
        else:
            print("히스토그램에 표시할 Failed CPA 데이터가 없습니다.")
        
        if all_failed_tcpa:
            try:
                all_failed_tcpa_flat = np.concatenate([np.atleast_1d(x) for x in all_failed_tcpa])
            except Exception as e:
                print(f"Flatten TCPA error: {e}")
                all_failed_tcpa_flat = np.array(all_failed_tcpa)
            plt.figure()
            plt.hist(all_failed_tcpa_flat, bins=20, color="skyblue", edgecolor="black")
            plt.title("Histogram of Failed TCPA")
            plt.xlabel("TCPA")
            plt.ylabel("Count")
            tcpa_hist_path = os.path.join(common_result_dir, "failed_tcpa_histogram.png")
            plt.savefig(tcpa_hist_path)
            plt.close()
            print(f"Failed TCPA 히스토그램이 {tcpa_hist_path}에 저장되었습니다.")
        else:
            print("히스토그램에 표시할 Failed TCPA 데이터가 없습니다.")

# ================= Detailed 분석 관련 =================

class DetailedEventAnalysisRunner:
    """
    이벤트 마집 파일들을 배치 단위로 병렬 처리하여,
    각 파일의 이벤트별 및 ship별 세부 결과를 CSV 파일로 저장하는 클래스입니다.
    """
    def __init__(self, base_folder, batch_size=50):
        self.base_folder = base_folder
        self.batch_size = batch_size
        self.file_manager = FileInputManager(base_folder)

    def analyze_file_deep(self, marzip_file):
        analyzer = RandomGenTestAnalyzer()
        analyzer.run(marzip_file)
        base_name = os.path.splitext(os.path.basename(marzip_file))[0]
        parent_dir = os.path.dirname(marzip_file)
        grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
        folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
        results = []
        for event_index, event in enumerate(analyzer.events):
            target_ships = event.get("target_ships", [])
            for ship_index, ship in enumerate(target_ships):
                row = {
                    "Folder": folder_info,
                    "File": base_name,
                    "EventIndex": event_index,
                    "ShipIndex": ship_index,
                    "ca_path_gen_fail": event.get("ca_path_gen_fail"),
                    "isCloseTargetDetected": event.get("isCloseTargetDetected"),
                    "CPA": ship.get("cpa"),
                    "TCPA": ship.get("tcpa"),
                    "SOG": ship.get("sog")
                }
                results.append(row)
        return results

    def run_all_parallel(self):
        marzip_files = self.file_manager.get_all_marzip_files()
        total_files = len(marzip_files)
        print(f"총 {total_files}개의 마집 파일을 찾았습니다.")
        all_results = []
        batch_num = 0
        start_total = time.time()
        for batch in batchify(marzip_files, self.batch_size):
            batch_num += 1
            start_batch = time.time()
            print(f"Detailed Batch {batch_num}: {len(batch)}개 파일 처리 시작...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                batch_results = list(executor.map(self.analyze_file_deep, batch))
                for res in batch_results:
                    if res:
                        all_results.extend(res)
            end_batch = time.time()
            print(f"Detailed Batch {batch_num} 완료. 소요 시간: {end_batch - start_batch:.2f}초")
        end_total = time.time()
        print(f"전체 Detailed 처리 완료. 총 소요 시간: {end_total - start_total:.2f}초")
        return all_results

    def run_and_save(self):
        results = self.run_all_parallel()
        output_csv = os.path.join(self.base_folder, "detailed_event_analysis.csv")
        fieldnames = ["Folder", "File", "EventIndex", "ShipIndex", "ca_path_gen_fail", 
                      "isCloseTargetDetected", "CPA", "TCPA", "SOG"]
        with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"상세 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")

# ================= Main =================

def main():
    # mode 선택: "fail" 또는 "detailed"
    mode = "fail"  # 변경 가능: "fail" 또는 "detailed"
    base_data_dir = "data/ver014_20250220_colregs_test_2"
    if mode == "fail":
        runner = RandomGenTestFailAnalysisRunner(base_data_dir)
        runner.run_and_save()
    elif mode == "detailed":
        runner = DetailedEventAnalysisRunner(base_data_dir)
        runner.run_and_save()
    else:
        print("알 수 없는 모드입니다.")
    print("DONE")

if __name__ == "__main__":
    main()
