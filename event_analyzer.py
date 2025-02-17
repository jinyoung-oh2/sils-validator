import os
import csv
import json
import re
import concurrent.futures
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

class EventAnalyzer(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터를 분석하는 클래스입니다.
    각 파일별로 ca_path_gen_fail 플래그를 기준으로 "Success", "Fail", 또는 "NA"를 판정합니다.
    
    - 이벤트가 0개이면 결과는 "NA"
    - 이벤트가 1개이면, 해당 이벤트의 플래그가 True이면 "NA", False이면 "Success"
    - 이벤트가 여러 개이면, 모든 플래그가 False이면 "Success", 하나라도 True가 있으면 "Fail"
    
    analyze_file() 메서드를 통해 파일을 처리합니다.
    """
    def __init__(self):
        super().__init__()
        self.events = []

    def analyze_dataset(self, events):
        flags = [event.get("ca_path_gen_fail") for event in events if event.get("ca_path_gen_fail") is not None]
        count = len(flags)
        if count == 0:
            return "NA"
        elif count == 1:
            return "NA" if flags[0] is True else "Success"
        else:
            return "Success" if all(flag is False for flag in flags) else "Fail"

    def analyze_file(self, marzip_file):
        """
        주어진 마집 파일을 처리하여 이벤트 데이터를 추출한 후,
        ca_path_gen_fail 플래그에 따라 결과를 분석하여 반환합니다.
        """
        self.marzip = marzip_file
        self.run(marzip_file)  
        self.events = self.events
        result = self.analyze_dataset(self.events)
        return {"Result": result}

def process_single_event(marzip_file):
    """
    단일 마집 파일에 대해 EventAnalyzer를 실행합니다.
    성공 시 {"Folder": ..., "File": ..., "Result": ...} 형태의 딕셔너리를 반환하고,
    실패하면 None을 반환합니다.
    """
    try:
        analyzer = EventAnalyzer()
        analysis = analyzer.analyze_file(marzip_file)
        base_name = os.path.splitext(os.path.basename(marzip_file))[0]
        parent_dir = os.path.dirname(marzip_file)
        grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
        folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
        return {"Folder": folder_info, "File": base_name, "Result": analysis["Result"]}
    except Exception as e:
        print(f"Error processing {marzip_file}: {e}")
        return None

def write_csv(results, output_csv, fieldnames=None):
    if fieldnames is None:
        fieldnames = ["Folder", "File", "Result"]
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

class EventAnalysisRunner:
    """
    이벤트 마집 파일들을 병렬 처리하여 분석 결과를 CSV 파일로 저장하는 클래스입니다.
    """
    def __init__(self, event_folder):
        """
        :param event_folder: 이벤트 마집 파일들이 저장된 폴더 경로 (예: ".../output")
        """
        self.event_folder = event_folder
        self.file_manager = FileInputManager(event_folder)

    def run_all_parallel(self):
        # FileInputManager를 사용하여 모든 마집 파일을 재귀적으로 수집
        marzip_files = self.file_manager.get_all_marzip_files()
        print(f"총 {len(marzip_files)}개의 마집 파일을 찾았습니다.")
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(process_single_event, marzip_files):
                if res is not None:
                    results.append(res)
        return results

    def run_and_save(self, aggregate_results=True):
        results = self.run_all_parallel()
        if aggregate_results:
            common_result_dir = "analysis_result"
            if not os.path.exists(common_result_dir):
                os.makedirs(common_result_dir)
            output_csv = os.path.join(common_result_dir, "all_event_analysis.csv")
            count_success = sum(1 for r in results if r["Result"] == "Success")
            count_fail = sum(1 for r in results if r["Result"] == "Fail")
            count_na = sum(1 for r in results if r["Result"] == "NA")
            summary_str = f"Success: {count_success}, Fail: {count_fail}, NA: {count_na}"
            results.append({"Folder": "", "File": "Summary", "Result": summary_str})
            write_csv(results, output_csv)
            print(f"통합 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")
        else:
            # 폴더별로 그룹화하여 CSV 파일 생성
            grouped = {}
            for r in results:
                folder = r["Folder"]
                if folder not in grouped:
                    grouped[folder] = []
                grouped[folder].append(r)
            for folder, group_results in grouped.items():
                result_dir = os.path.join("analysis_result", folder)
                if not os.path.exists(result_dir):
                    os.makedirs(result_dir)
                output_csv = os.path.join(result_dir, "event_analysis.csv")
                count_success = sum(1 for r in group_results if r["Result"] == "Success")
                count_fail = sum(1 for r in group_results if r["Result"] == "Fail")
                count_na = sum(1 for r in group_results if r["Result"] == "NA")
                summary_str = f"Success: {count_success}, Fail: {count_fail}, NA: {count_na}"
                group_results.append({"Folder": "", "File": "Summary", "Result": summary_str})
                write_csv(group_results, output_csv)
                print(f"{folder} 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")

def main(AGGREGATE_RESULTS, base_data_dir):
    runner = EventAnalysisRunner(base_data_dir)
    runner.run_and_save(AGGREGATE_RESULTS)
    print("DONE")

if __name__ == "__main__":
    AGGREGATE_RESULTS = True  # True: 통합 결과, False: 폴더별 결과
    base_data_dir = "data/ver014_20205215_basic_test"  # 여러 이벤트 폴더가 있는 최상위 디렉토리
    analyzer = EventAnalyzer()
    main(AGGREGATE_RESULTS, base_data_dir)
