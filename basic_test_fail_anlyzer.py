import os
import csv
import concurrent.futures
import matplotlib.pyplot as plt
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

class BasicTestFailAnalyzer(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터를 분석하는 클래스입니다.
    각 파일별로 ca_path_gen_fail 및 isCloseTargetDetected 플래그를 기준으로 
    "N/A", "fail", 또는 "success"를 판정하며, 실패한 경우 해당 이벤트의
    target_ships[0]에 있는 "cpa"와 "sog" 정보를 함께 반환합니다.
    
    - 이벤트가 0개이면 → "N/A"
    - 이벤트가 1개이면:
          if ca_path_gen_fail가 True → "N/A (path gen fail)"
          else → "success"
    - 이벤트가 2개 이상이면:
          if 하나라도 isCloseTargetDetected가 True → "fail (closed target)"
          elif 하나라도 ca_path_gen_fail가 True → "fail (path gen fail)"
          else → "success"
    """
    def __init__(self):
        super().__init__()

    def analyze_dataset(self):
        n = len(self.events)
        failed_cpa_s = []
        failed_tcpa_s = []
        failed_sogs = []
        
        # 이벤트가 하나도 없으면: N/A (safe target)
        if n == 0:
            return {"result": "N/A (safe target)"}
        
        # 이벤트가 1개이면:
        elif n == 1:
            event = self.events[0]
            # ca_path_gen_fail이 True이면: N/A (path gen fail)
            if event.get("ca_path_gen_fail") is True:
                return {"result": "N/A (path gen fail)"}
            else:
                return {"result": "success"}
        
        # 이벤트가 2개 이상이면:
        else:
            # 하나라도 isCloseTargetDetected가 True이면: fail (closed target)
            if any(event.get("isCloseTargetDetected") is True for event in self.events):
                for event in self.events:
                    if event.get("isCloseTargetDetected") is True:
                        try:
                            ship = event.get("target_ships")[0]
                            failed_cpa_s.append(ship.get("cpa"))
                            failed_tcpa_s.append(ship.get("tcpa"))
                            failed_sogs.append(ship.get("sog"))
                        except Exception as e:
                            print(f"Error extracting ship data: {e}")
                return {"result": "fail (closed target)",
                        "failed_cpa": failed_cpa_s,
                        "failed_tcpa": failed_tcpa_s,
                        "failed_sog": failed_sogs}
            # 그렇지 않고 하나라도 ca_path_gen_fail이 True이면: fail (path gen fail)
            elif any(event.get("ca_path_gen_fail") is True for event in self.events):
                for event in self.events:
                    if event.get("ca_path_gen_fail") is True:
                        try:
                            ship = event.get("target_ships")[0]
                            failed_cpa_s.append(ship.get("cpa"))
                            failed_tcpa_s.append(ship.get("tcpa"))
                            failed_sogs.append(ship.get("sog"))
                        except Exception as e:
                            print(f"Error extracting ship data: {e}")
                return {"result": "fail (path gen fail)",
                        "failed_cpa": failed_cpa_s,
                        "failed_tcpa": failed_tcpa_s,
                        "failed_sog": failed_sogs}
            else:
                return {"result": "success"}

    def analyze_file(self, marzip_file):
        """
        주어진 마집 파일을 처리하여 이벤트 데이터를 추출한 후,
        조건에 따라 결과를 분석하여 반환합니다.
        """
        self.marzip = marzip_file
        self.run(marzip_file)
        result = self.analyze_dataset()
        return result

def process_single_file(marzip_file):
    """
    단일 마집 파일에 대해 BasicTestFailAnalyzer를 실행합니다.
    성공 시 {"Folder": ..., "File": ..., "Result": ..., (선택적)failed_cpa, failed_sog} 형태의 딕셔너리를 반환하고,
    실패하면 None을 반환합니다.
    """
    try:
        analyzer = BasicTestFailAnalyzer()
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
        # 실패한 경우에만 failed_cpa, failed_sog 정보를 추가합니다.
        if analysis["result"].startswith("fail") or "n/a" in analysis["result"].lower():
            result_dict["failed_cpa"] = analysis.get("failed_cpa", [])
            result_dict["failed_tcpa"] = analysis.get("failed_tcpa", [])
            result_dict["failed_sog"] = analysis.get("failed_sog", [])
        return result_dict
    except Exception as e:
        print(f"Error processing {marzip_file}: {e}")
        return None

def write_csv(results, output_csv, fieldnames=None):
    if fieldnames is None:
        # CSV에는 히스토그램 관련 데이터는 저장하지 않고, 기본 정보만 기록합니다.
        fieldnames = ["Folder", "File", "Result"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

class BasicTestFailAnalysisRunner:
    """
    이벤트 마집 파일들을 병렬 처리하여 분석 결과를 CSV 파일로 저장하고,
    실패한 이벤트의 cpa와 sog 값을 히스토그램으로 출력하는 클래스입니다.
    모든 결과를 하나의 CSV 파일에 집계합니다.
    """
    def __init__(self, event_folder):
        """
        :param event_folder: 이벤트 마집 파일들이 저장된 폴더 경로 (예: ".../output")
        """
        self.event_folder = event_folder
        self.file_manager = FileInputManager(event_folder)

    def run_all_parallel(self):
        marzip_files = self.file_manager.get_all_marzip_files()
        print(f"총 {len(marzip_files)}개의 마집 파일을 찾았습니다.")
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(process_single_file, marzip_files):
                if res is not None:
                    results.append(res)
        return results

    def run_and_save(self):
        results = self.run_all_parallel()
        common_result_dir = "analysis_result"
        if not os.path.exists(common_result_dir):
            os.makedirs(common_result_dir)
        output_csv = os.path.join(common_result_dir, "all_event_analysis.csv")
        
        # 각 결과의 상태별 카운트
        count_success = sum(1 for r in results if r["Result"].lower() == "success")
        count_fail_closed = sum(1 for r in results if r["Result"].lower() == "fail (closed target)")
        count_fail_path = sum(1 for r in results if r["Result"].lower() == "fail (path gen fail)")
        count_na_safe = sum(1 for r in results if r["Result"].lower() == "n/a (safe target)")
        count_na_path = sum(1 for r in results if r["Result"].lower() == "n/a (path gen fail)")
        
        fail_total = count_fail_closed + count_fail_path
        na_total = count_na_safe + count_na_path
        total = len(results)
        
        summary_str = (
            f"Total: {total}, Success: {count_success}, "
            f"Fail: {fail_total} (closed: {count_fail_closed}, path gen fail: {count_fail_path}), "
            f"N/A: {na_total} (safe target: {count_na_safe}, path gen fail: {count_na_path})"
        )
        results.append({"Folder": "", "File": "Summary", "Result": summary_str})
        write_csv(results, output_csv)
        print(f"통합 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")
        
        # 실패한 이벤트들의 cpa와 sog 값을 모두 수집하여 히스토그램 생성
        all_failed_cpa_s = []
        all_failed_tcpa_s = []
        all_failed_sogs = []
        for r in results:
            # Summary row는 건너뜁니다.
            if r.get("File") == "Summary":
                continue
            # 결과 문자열에 "fail" 또는 "n/a"가 포함되어 있으면 실패 이벤트라고 가정합니다.
            if ("fail" in r["Result"].lower()) or ("n/a" in r["Result"].lower()):
                all_failed_cpa_s.extend(r.get("failed_cpa", []))
                all_failed_tcpa_s.extend(r.get("failed_tcpa", []))
                all_failed_sogs.extend(r.get("failed_sog", []))

                
        # 만약 값들이 존재하면 히스토그램을 그립니다.
        if all_failed_cpa_s:
            plt.figure()
            plt.hist(all_failed_cpa_s, bins=20, color="skyblue", edgecolor="black")
            plt.title("Histogram of Failed CPA")
            plt.xlabel("CPA")
            plt.ylabel("num")
            cpa_hist_path = os.path.join(common_result_dir, "failed_cpa_histogram.png")
            plt.savefig(cpa_hist_path)
            plt.close()
            print(f"Failed CPA 히스토그램이 {cpa_hist_path}에 저장되었습니다.")
        else:
            print("히스토그램에 표시할 Failed CPA 데이터가 없습니다.")

        # 만약 값들이 존재하면 히스토그램을 그립니다.
        if all_failed_tcpa_s:
            plt.figure()
            plt.hist(all_failed_tcpa_s, bins=20, color="skyblue", edgecolor="black")
            plt.title("Histogram of Failed TCPA")
            plt.xlabel("TCPA")
            plt.ylabel("num")
            tcpa_hist_path = os.path.join(common_result_dir, "failed_tcpa_histogram.png")
            plt.savefig(tcpa_hist_path)
            plt.close()
            print(f"Failed TCPA 히스토그램이 {tcpa_hist_path}에 저장되었습니다.")
        else:
            print("히스토그램에 표시할 Failed TCPA 데이터가 없습니다.")

        if all_failed_sogs:
            plt.figure()
            plt.hist(all_failed_sogs, bins=20, color="salmon", edgecolor="black")
            plt.title("Histogram of Failed SOG")
            plt.xlabel("SOG")
            plt.ylabel("Num")
            sog_hist_path = os.path.join(common_result_dir, "failed_sog_histogram.png")
            plt.savefig(sog_hist_path)
            plt.close()
            print(f"Failed SOG 히스토그램이 {sog_hist_path}에 저장되었습니다.")
        else:
            print("히스토그램에 표시할 Failed SOG 데이터가 없습니다.")

def main():
    base_data_dir = "data/ver014_20250218_colregs_test-2"  # 여러 이벤트 폴더가 있는 최상위 디렉토리
    runner = BasicTestFailAnalysisRunner(base_data_dir)
    runner.run_and_save()
    print("DONE")

if __name__ == "__main__":
    main()
