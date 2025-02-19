import os
import zipfile
import shutil
import pyarrow as pa
import pyarrow.ipc as ipc
import json

class MarzipExtractor:
    def __init__(self, marzip_file=None):
        self.static_dataset = []
        self.timeseries_dataset = []
        self.simulation_result = []

        self.base_route = []
        self.own_ship_static = []
        self.events = []
        self.colreg = []

        if marzip_file is not None:
            self.marzip = marzip_file


    def run(self, marzip=None):
        if marzip is None:
            marzip = self.marzip
            
        data = self.extract_and_read_marzip(marzip)

        self.static_dataset = data["static_dataset"]
        self.timeseries_dataset = data["timeseries_dataset"]
        self.simulation_result = data["simulation_result"]

        self.base_route = self.extract_base_route(self.simulation_result)
        self.own_ship_static = self.extract_own_ship_static_info(self.simulation_result)
        self.events = self.extract_events_info(self.simulation_result)

    def flatten(self, item):
        """
        재귀적으로 리스트를 평탄화하여 모든 중첩을 풀어줍니다.
        예: [[a, b], c, [d, [e, f]]] => [a, b, c, d, e, f]
        """
        if isinstance(item, list):
            result = []
            for sub in item:
                result.extend(self.flatten(sub))
            return result
        else:
            return [item]
        

    def safe_get(self, data, keys, default=None):
        """
        중첩 딕셔너리에서 키 체인을 따라 안전하게 값을 가져옵니다.
        """
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
                if data == default:
                    return default
            else:
                return default
        return data


    def extract_base_route(self, data):
        """
        Base Route 추출 (trafficSituation → ownShip → waypoints)
        """
        return self.safe_get(data, ["trafficSituation", "ownShip", "waypoints"], default=[])

    def extract_own_ship_static_info(self, data):
        """
        Own Ship의 정적(static) 정보 추출 (trafficSituation → ownShip → static)
        """
        return self.safe_get(data, ["trafficSituation", "ownShip", "static"], default={})

    def extract_events_info(self, data):
        """
        이벤트 정보를 통합하여 추출합니다.
        각 이벤트에서 아래 데이터를 딕셔너리 형태로 추출하여 리스트로 반환합니다.
        """
        events = []
        event_data = self.safe_get(data, ["cagaData", "eventData"], default=[])
        if isinstance(event_data, list):
            for event in event_data:
                event_info = {}
                event_info["safe_route"] = self.safe_get(event, ["safe_path_info", "route"])
                
                # targetShips: 리스트가 아닐 경우 리스트로 변환 후 평탄화 적용
                target_ships = self.safe_get(event, ["timeSeriesData", "targetShips"], default=[])
                if not isinstance(target_ships, list):
                    target_ships = [target_ships]
                event_info["target_ships"] = self.flatten(target_ships)
                
                event_info["own_ship_event"] = self.safe_get(event, ["timeSeriesData", "ownShip"])
                event_info["ca_path_gen_fail"] = self.safe_get(event, ["caPathGenFail"])
                event_info["is_near_target"] = self.safe_get(event, ["isNearTarget"])
                events.append(event_info)
        return events



    def _read_arrow_file(self, file_path):
        """
        주어진 Arrow 파일을 읽어 pyarrow Table로 반환.
        IPC File Format이 실패하면 Streaming Format으로 읽음.
        """
        with open(file_path, 'rb') as f:
            try:
                reader = ipc.RecordBatchFileReader(f)
                return reader.read_all()
            except pa.lib.ArrowInvalid:
                # 파일 포인터 재설정 후 스트리밍 포맷 시도
                f.seek(0)
                try:
                    reader = ipc.RecordBatchStreamReader(f)
                    return reader.read_all()
                except pa.lib.ArrowInvalid as e:
                    raise e

    def extract_and_save_simulation_result(self):
        """
        .marzip 파일에서 simulation_result 데이터를 추출한 후,
        원본 파일 이름 (확장자 제거) + ".json" 파일로 저장합니다.
        """
        extracted_files = []
        extract_dir = None

        # 파일이 ZIP 형식인지 확인
        if zipfile.is_zipfile(self.marzip):
            with zipfile.ZipFile(self.marzip, 'r') as zip_ref:
                # .marzip 확장자를 제거한 디렉토리명을 사용합니다.
                extract_dir = os.path.splitext(self.marzip)[0]
                zip_ref.extractall(extract_dir)
                extracted_files = [os.path.join(extract_dir, name) for name in zip_ref.namelist()]
        else:
            raise ValueError("제공된 파일은 올바른 .marzip 아카이브가 아닙니다.")

        simulation_result = {}
        # 압축 해제된 파일 중 .json 파일을 찾아 simulation_result를 읽어옴
        for file in extracted_files:
            if file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        simulation_result = json.load(f)
                    # 첫 번째 JSON 파일만 사용 (필요에 따라 수정 가능)
                    break
                except Exception as e:
                    print(f"JSON 파일 {file} 읽기 오류: {e}")

        if simulation_result:
            output_json_path = os.path.splitext(self.marzip)[0] + ".json"
            try:
                with open(output_json_path, 'w', encoding='utf-8') as f:
                    json.dump(simulation_result, f, ensure_ascii=False, indent=4)
                print(f"Simulation result saved to {output_json_path}")
            except Exception as e:
                print(f"Simulation result 저장 실패: {e}")
        else:
            print("simulation_result 데이터를 찾지 못했습니다.")

        # 압축 해제한 디렉토리 삭제
        if extract_dir and os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                print(f"추출된 디렉토리 삭제 실패: {e}")

    def extract_and_read_marzip(self, marzip):
        """
        .marzip 파일을 압축해제하여 각 파일의 데이터를 읽은 후,
        timeseries 및 static 데이터셋을 반환한다.
        또한 result.json 파일의 내용은 simulation_result에 저장한다.
        
        :return: dict
            {
                "timeseries_dataset": [...],
                "static_dataset": [...],
                "simulation_result": { ... }
            }
        """
        extracted_files = []
        extract_dir = None

        # 파일이 ZIP 형식인지 확인
        if zipfile.is_zipfile(marzip):
            with zipfile.ZipFile(marzip, 'r') as zip_ref:
                # .marzip 확장자를 제거한 디렉토리명을 사용합니다.
                extract_dir = os.path.splitext(marzip)[0]
                zip_ref.extractall(extract_dir)
                extracted_files = [os.path.join(extract_dir, name) for name in zip_ref.namelist()]
        else:
            raise ValueError("제공된 파일은 올바른 .marzip 아카이브가 아닙니다.")

        timeseries_dataset = []
        static_dataset = []
        simulation_result = []

        # 압축 해제된 파일들을 순회하며 데이터 읽기
        for file in extracted_files:
            if file.endswith('timeseries.arrow'):
                try:
                    table = self._read_arrow_file(file)
                    # to_pylist()로 각 row를 dict로 변환하여 리스트에 추가
                    timeseries_dataset.extend(table.to_pylist())
                except Exception as e:
                    print(f"파일 {file} 읽기 오류: {e}")
            elif file.endswith('static.arrow'):
                try:
                    table = self._read_arrow_file(file)
                    static_dataset.extend(table.to_pylist())
                except Exception as e:
                    print(f"파일 {file} 읽기 오류: {e}")
            elif file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as json_file:
                        simulation_result = json.load(json_file)
                except Exception as e:
                    print(f"JSON 파일 {file} 읽기 오류: {e}")

        # 압축 해제한 디렉토리 삭제
        if extract_dir and os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                print(f"추출된 디렉토리 삭제 실패: {e}")

        return {
            "timeseries_dataset": timeseries_dataset,
            "static_dataset": static_dataset,
            "simulation_result": simulation_result
        }
