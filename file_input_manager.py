import os
import json
import re
from marzip_extractor import MarzipExtractor

class FileInputManager:
    """
    파일 입력 관련 기능을 제공하는 클래스.
    폴더 내의 파일을 읽어 marzip 또는 json 파일을 로드합니다.
    """
    def __init__(self, file_folder):
        """
        :param file_folder: 파일들이 저장된 폴더 경로.
        """
        if not os.path.exists(file_folder):
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {file_folder}")
        self.file_folder = file_folder

    @staticmethod
    def natural_sort_key(s):
        """
        파일 이름 내 숫자들을 정수로 변환하여 자연 정렬할 수 있는 키를 생성합니다.
        예: "scen_1", "scen_2", "scen_10" -> ["scen_", 1, "scen_", 2, "scen_", 10]
        """
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

    def get_files(self, folder, extension):
        """
        지정한 폴더 내의 특정 확장자를 가진 파일들의 전체 경로 리스트를 반환합니다.
        """
        if not os.path.exists(folder):
            raise FileNotFoundError(f"폴더를 찾을 수 없습니다: {folder}")
        return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(extension)]
    
    def get_all_marzip_files(self):
        """
        self.file_folder 아래의 모든 .marzip 파일을 재귀적으로 검색하여
        전체 경로 리스트를 반환합니다.
        """
        marzip_files = []
        for root, dirs, files in os.walk(self.file_folder):
            batch_count = len([f for f in files if f.endswith('.marzip')])
            print(f"Found {batch_count} .marzip files in {root}")
            for file in files:
                if file.endswith(".marzip"):
                    marzip_files.append(os.path.join(root, file))
        return sorted(marzip_files, key=self.natural_sort_key)

    def get_sils_files(self, mode="marzip"):
        """
        self.file_folder 내의 파일 목록을 반환합니다.
        mode가 "marzip"이면 ".marzip", "json"이면 ".json" 파일을 찾습니다.
        mode가 None이면 빈 리스트를 반환합니다.
        """
        if mode is None:
            return []
        ext = ".marzip" if mode == "marzip" else ".json"
        files = self.get_files(self.file_folder, ext)
        return sorted(files, key=self.natural_sort_key)

    def load_marzip(self, file_path):
        """
        단일 marzip 파일을 로드하여 데이터를 반환합니다.
        에러 발생 시 None을 반환합니다.
        """
        try:
            data = MarzipExtractor(file_path).extract_and_read_marzip()
            return data
        except Exception:
            return None

    def load_json(self, file_path):
        """
        단일 JSON 파일을 로드하여 데이터를 반환합니다.
        에러 발생 시 None을 반환합니다.
        """
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                data = json.load(f)
            return data
        except Exception:
            return None

    def load_data(self, file_path, mode="marzip"):
        """
        :param file_path: 파일 경로.
        :param mode: "marzip"이면 marzip 내부에서, "json"이면 해당 json 파일에서 로드합니다.
        :return: 데이터 (dict) 또는 None.
        """
        if mode is None:
            return None

        if mode == "marzip":
            return self.load_marzip(file_path)
        elif mode == "json":
            return self.load_json(file_path)
        else:
            raise ValueError("mode는 'marzip' 또는 'json'이어야 합니다.")

if __name__ == "__main__":
    # 예시: data/ver013_20250213_6_20250213T104604/output 폴더 내의 모든 마집 파일 검색
    event_folder = "data/ver013_20250213_6_20250213T104604/output"
    file_manager = FileInputManager(event_folder)
    
    marzip_files = file_manager.get_all_marzip_files()
    print("찾은 마집 파일 목록:")
    for f in marzip_files:
        print("  ", f)
    
    # 예시: 첫 번째 파일을 로드
    if marzip_files:
        sample_file = marzip_files[0]
        data = file_manager.load_data(sample_file, mode="marzip")
        if data:
            print("데이터 로드 성공!")
            sim_result = data.get("simulation_result")
            print("simulation_result의 타입:", type(sim_result))
        else:
            print("데이터 로드 실패")
