import os
import csv
import concurrent.futures
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

class BasicTestFailAnalyzer(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터를 분석하는 클래스입니다.
    각 파일별로 ca_path_gen_fail 및 isCloseTargetDetected 플래그를 기준으로 
    "N/A", "fail", 또는 "success"를 판정합니다.
    
    - 이벤트가 0개이면 → "N/A"
    - 이벤트가 1개이면:
          if ca_path_gen_fail가 True → "N/A" (path gen fail)
          else → "success"
    - 이벤트가 2개 이상이면:
          if 하나라도 isCloseTargetDetected가 True → "fail" (closed target)
          elif 하나라도 ca_path_gen_fail가 True → "fail" (path gen fail)
          else → "success"
    """
    def __init__(self):
        super().__init__()

    def analyze_dataset(self):
        n = len(self.events)
        # 이벤트가 하나도 없으면: N/A (safe target)
        if n == 0:
            return "N/A (safe target)"
        # 이벤트가 1개이면:
        elif n == 1:
            event = self.events[0]
            # ca_path_gen_fail이 True이면: N/A (path gen fail)
            if event.get("ca_path_gen_fail") is True:
                return "N/A (path gen fail)"
            else:
                return "success"
        # 이벤트가 2개 이상이면:
        else:
            # 하나라도 isCloseTargetDetected가 True이면: fail (closed target)
            if any(event.get("isCloseTargetDetected") is True for event in self.events):
                return "fail (closed target)"
            # 그렇지 않고 하나라도 ca_path_gen_fail이 True이면: fail (path gen fail)
            elif any(event.get("ca_path_gen_fail") is True for event in self.events):
                # 조건에 맞는 첫 번째 이벤트를 찾습니다.
                for event in self.events:
                    if event.get("ca_path_gen_fail") is True:
                        print(event.get("target_ships")[0].get("distanceToTarget"))
                return "fail (path gen fail)"
            else:
                return "success"

    def analyze_file(self, marzip_file):
        """
        주어진 마집 파일을 처리하여 이벤트 데이터를 추출한 후,
        조건에 따라 결과를 분석하여 반환합니다.
        """
        self.marzip = marzip_file
        self.run(marzip_file)
        # 이벤트 목록을 복사하여 사용합니다.
        result = self.analyze_dataset()
        return {"Result": result}

def process_single_file(marzip_file):
    """
    단일 마집 파일에 대해 BasicTestFailAnalyzer를 실행합니다.
    성공 시 {"Folder": ..., "File": ..., "Result": ...} 형태의 딕셔너리를 반환하고,
    실패하면 None을 반환합니다.
    """
    try:
        analyzer = BasicTestFailAnalyzer()
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

class BasicTestFailAnalysisRunner:
    """
    이벤트 마집 파일들을 병렬 처리하여 분석 결과를 CSV 파일로 저장하는 클래스입니다.
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


def main():
    base_data_dir = "data/ver014_20250218_colregs_test"  # 여러 이벤트 폴더가 있는 최상위 디렉토리
    runner = BasicTestFailAnalysisRunner(base_data_dir)
    runner.run_and_save()
    print("DONE")

if __name__ == "__main__":
    main()
