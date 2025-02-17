import os
import csv
import json
import re
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

class EventAnalyzer(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터를 분석하는 클래스입니다.
    각 파일별로 ca_path_gen_fail 플래그를 기준으로 "Success", "Fail", 또는 "NA"를 판정합니다.
    
    - 이벤트가 0개이면 결과는 "NA"
    - 이벤트가 1개이면, 해당 이벤트의 플래그가 True이면 "NA", False이면 "Success"
    - 이벤트가 여러 개이면, 모든 플래그가 False이면 "Success", 하나라도 True가 있으면 "Fail"
    
    생성자에서는 파일 인자를 받지 않고, analyze_file() 메서드를 통해 파일을 처리합니다.
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
        self.run(marzip_file)  # run() 호출 후 self.sils_events 등 채워짐
        self.events = self.sils_events
        result = self.analyze_dataset(self.events)
        return {"Result": result}


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
    이벤트 마집 파일들을 읽어 분석을 수행하고,
    각각의 결과를 CSV 파일로 저장하는 클래스입니다.
    """
    def __init__(self, event_folder):
        """
        :param event_folder: 이벤트 마집 파일들이 저장된 폴더 경로 (예: ".../output")
        """
        self.event_folder = event_folder
        self.file_manager = FileInputManager(event_folder)

    def run(self):
        event_files = sorted(
            self.file_manager.get_sils_files(mode="marzip"),
            key=lambda f: FileInputManager.natural_sort_key(os.path.basename(f))
        )
        results = []
        for event_file in event_files:
            base_name = os.path.splitext(os.path.basename(event_file))[0]
            print(f"Processing {base_name} ...")
            analyzer = EventAnalyzer()
            analysis = analyzer.analyze_file(event_file)
            # 폴더 정보: 상위 폴더 이름 (예: 해당 이벤트 파일의 상위 폴더 이름)
            folder_info = os.path.basename(os.path.dirname(os.path.dirname(event_file)))
            results.append({"Folder": folder_info, "File": base_name, "Result": analysis["Result"]})
        return results

    def run_and_save(self):
        results = self.run()
        # 결과 폴더: event_folder의 상위 폴더 이름 사용
        parent_dir = os.path.dirname(self.event_folder)
        folder_name = os.path.basename(parent_dir)
        result_dir = os.path.join("analysis_result", folder_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        output_csv = os.path.join(result_dir, "event_analysis.csv")
        # 요약 행 추가
        count_success = sum(1 for r in results if r["Result"] == "Success")
        count_fail = sum(1 for r in results if r["Result"] == "Fail")
        count_na = sum(1 for r in results if r["Result"] == "NA")
        summary_str = f"Success: {count_success}, Fail: {count_fail}, NA: {count_na}"
        results.append({"Folder": "", "File": "Summary", "Result": summary_str})
        write_csv(results, output_csv)
        print(f"이벤트 분석 결과가 {output_csv}에 저장되었습니다.")


# =====================
# main 함수
# =====================
def main(AGGREGATE_RESULTS, base_data_dir):
    # AGGREGATE_RESULTS가 True이면 모든 마집 파일을 한 번에 분석하여 통합 CSV 파일을 생성하고,
    # False이면 각 이벤트 폴더별로 별도의 CSV 파일을 생성합니다.

    # FileInputManager의 재귀 검색 기능으로 모든 마집 파일을 수집
    file_manager = FileInputManager(base_data_dir)
    marzip_files = file_manager.get_all_marzip_files()

    if AGGREGATE_RESULTS:
        all_results = []
        for marzip_file in marzip_files:
            base_name = os.path.splitext(os.path.basename(marzip_file))[0]
            # 상위 폴더의 상위 폴더 이름을 폴더 정보로 사용 (없으면 빈 문자열)
            parent_dir = os.path.dirname(marzip_file)
            grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
            folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
            print(f"Processing: {marzip_file}")
            analyzer = EventAnalyzer()
            analysis = analyzer.analyze_file(marzip_file)
            all_results.append({"Folder": folder_info, "File": base_name, "Result": analysis["Result"]})

        common_result_dir = "analysis_result"
        if not os.path.exists(common_result_dir):
            os.makedirs(common_result_dir)
        output_csv = os.path.join(common_result_dir, "all_event_analysis.csv")
        count_success = sum(1 for r in all_results if r["Result"] == "Success")
        count_fail = sum(1 for r in all_results if r["Result"] == "Fail")
        count_na = sum(1 for r in all_results if r["Result"] == "NA")
        summary_str = f"Success: {count_success}, Fail: {count_fail}, NA: {count_na}"
        all_results.append({"Folder": "", "File": "Summary", "Result": summary_str})
        write_csv(all_results, output_csv)
        print(f"통합 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")
    else:
        # AGGREGATE_RESULTS가 False인 경우, 파일들을 그룹화하여 각 그룹별로 CSV 생성
        grouped = {}
        for marzip_file in marzip_files:
            base_name = os.path.splitext(os.path.basename(marzip_file))[0]
            parent_dir = os.path.dirname(marzip_file)
            grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
            folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
            if folder_info not in grouped:
                grouped[folder_info] = []
            print(f"Processing: {marzip_file}")
            analyzer = EventAnalyzer()
            analysis = analyzer.analyze_file(marzip_file)
            grouped[folder_info].append({"Folder": folder_info, "File": base_name, "Result": analysis["Result"]})

        for folder_info, results in grouped.items():
            result_dir = os.path.join("analysis_result", folder_info)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            output_csv = os.path.join(result_dir, "event_analysis.csv")
            count_success = sum(1 for r in results if r["Result"] == "Success")
            count_fail = sum(1 for r in results if r["Result"] == "Fail")
            count_na = sum(1 for r in results if r["Result"] == "NA")
            summary_str = f"Success: {count_success}, Fail: {count_fail}, NA: {count_na}"
            results.append({"Folder": "", "File": "Summary", "Result": summary_str})
            write_csv(results, output_csv)
            print(f"{folder_info} 이벤트 분석 결과가 {output_csv}에 저장되었습니다.")
    print("DONE")

if __name__ == "__main__":

    AGGREGATE_RESULTS = True # True 통합결과 / False 폴더별 결과
    base_data_dir = "data/ver014_20205214_basic_test"  # 여러 이벤트 폴더가 있는 최상위 디렉토리
    main(AGGREGATE_RESULTS, base_data_dir)