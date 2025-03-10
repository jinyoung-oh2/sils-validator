#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import zipfile
import shutil
import gc
import pyarrow as pa
import pyarrow.ipc as ipc
import json
from colorama import Fore, Style

class MarzipExtractor:
    def __init__(self, marzip_file=None):
        self.static_dataset = []
        self.timeseries_dataset = []
        self.own_ship_time_series = []  # ownShip이 True인 timeseries 데이터 저장
        self.simulation_result = {}

        self.base_route = []
        self.own_ship_static = {}
        self.events = []
        self.colreg = []
        self.hinas_setup = {}

        if marzip_file is not None:
            self.marzip = marzip_file

    def run(self, marzip=None):
        if marzip is None:
            marzip = self.marzip

        data = self.extract_and_read_marzip(marzip)

        self.static_dataset = data.get("static_dataset", [])
        self.timeseries_dataset = data.get("timeseries_dataset", [])
        self.own_ship_time_series = data.get("own_ship_time_series", [])
        self.simulation_result = data.get("simulation_result", {})

        if not self.simulation_result:
            print("simulation_result가 비어있습니다. 기본 경로 및 이벤트 추출을 건너뜁니다.")
            return

        try:
            self.base_route = self.extract_base_route(self.simulation_result)
            self.hinas_setup = self.extract_hinas_setup(self.simulation_result)
        except Exception as e:
            print(f"base_route 추출 실패: {e}")
            self.base_route = []
            self.hinas_setup = {}

        try:
            self.own_ship_static = self.extract_own_ship_static_info(self.simulation_result)
        except Exception as e:
            print(f"own_ship_static 추출 실패: {e}")
            self.own_ship_static = {}

        try:
            self.events = self.extract_events_info(self.simulation_result)
        except Exception as e:
            print(f"events 추출 실패: {e}")
            self.events = []

    def flatten(self, item):
        """재귀적으로 리스트를 평탄화합니다."""
        if isinstance(item, list):
            result = []
            for sub in item:
                result.extend(self.flatten(sub))
            return result
        else:
            return [item]

    def safe_get(self, data, keys, default=None):
        """중첩 딕셔너리에서 안전하게 값을 가져옵니다."""
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, default)
                if data == default:
                    return default
            else:
                return default
        return data

    def extract_base_route(self, data):
        """trafficSituation → ownShip → waypoints"""
        return self.safe_get(data, ["trafficSituation", "ownShip", "waypoints"], default=[])

    def extract_hinas_setup(self, data):
        return self.safe_get(data, ["cagaData", "caga_configuration", "hinas_setup"], default={})

    def extract_own_ship_static_info(self, data):
        """trafficSituation → ownShip → static"""
        return self.safe_get(data, ["trafficSituation", "ownShip", "static"], default={})

    def extract_events_info(self, data):
        """cagaData → eventData에서 이벤트들을 추출합니다."""
        events = []
        event_data = self.safe_get(data, ["cagaData", "eventData"], default=[])
        if isinstance(event_data, list):
            for event in event_data:
                event_info = {}
                event_info["safe_route"] = self.safe_get(event, ["safe_path_info", "route"], default=[])
                target_ships = self.safe_get(event, ["timeSeriesData", "targetShips"], default=[])
                if not isinstance(target_ships, list):
                    target_ships = [target_ships]
                event_info["target_ships"] = self.flatten(target_ships)
                event_info["own_ship_event"] = self.safe_get(event, ["timeSeriesData", "ownShip"], default={})
                event_info["ca_path_gen_fail"] = self.safe_get(event, ["caPathGenFail"])
                event_info["is_near_target"] = self.safe_get(event, ["isNearTarget"])
                events.append(event_info)
        return events

    def _read_arrow_file(self, file_path):
        """Arrow 파일을 일반 파일 읽기 방식으로 읽습니다."""
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            buffer_reader = pa.BufferReader(data)
            try:
                reader = ipc.RecordBatchFileReader(buffer_reader)
                table = reader.read_all()
            except pa.lib.ArrowInvalid:
                buffer_reader.seek(0)
                reader = ipc.RecordBatchStreamReader(buffer_reader)
                table = reader.read_all()
            gc.collect()  # 메모리 정리
            return table
        except Exception as e:
            print(f"[ERROR] Arrow 파일 읽기 실패 (버퍼 방식): {file_path}, {e}")
            return None
        
    def extract_and_save_simulation_result(self):
        """.marzip 파일 내 simulation_result 데이터를 추출하여 JSON으로 저장합니다."""
        extracted_files = []
        extract_dir = None

        if zipfile.is_zipfile(self.marzip):
            with zipfile.ZipFile(self.marzip, 'r') as zip_ref:
                extract_dir = os.path.splitext(self.marzip)[0]
                zip_ref.extractall(extract_dir)
                extracted_files = [os.path.join(extract_dir, name) for name in zip_ref.namelist()]
        else:
            raise ValueError("제공된 파일은 올바른 .marzip 아카이브가 아닙니다.")

        simulation_result = {}
        for file in extracted_files:
            if file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        simulation_result = json.load(f)
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

        if extract_dir and os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                print(f"추출된 디렉토리 삭제 실패: {e}")

    def extract_and_read_marzip(self, marzip):
        """
        .marzip 파일을 압축해제하여 Arrow 및 JSON 파일의 데이터를 읽어 반환합니다.
        """
        extracted_files = []
        extract_dir = None

        try:
            if zipfile.is_zipfile(marzip):
                with zipfile.ZipFile(marzip, 'r') as zip_ref:
                    extract_dir = os.path.splitext(marzip)[0]
                    zip_ref.extractall(extract_dir)
                    extracted_files = [os.path.join(extract_dir, name) for name in zip_ref.namelist()]
            else:
                print(Fore.RED + f"올바른 .marzip 아카이브가 아닙니다: {marzip}" + Style.RESET_ALL)
                return {}
        except Exception as e:
            print(Fore.RED + f"압축 해제 실패: {marzip} / {e}" + Style.RESET_ALL)
            return {}

        timeseries_dataset = []
        own_ship_time_series = []
        static_dataset = []
        simulation_result = {}

        for file in extracted_files:
            if file.endswith('timeseries.arrow'):
                try:
                    table = self._read_arrow_file(file)
                    if table is None:
                        print(Fore.RED + f"Arrow 파일 읽기 실패: {file}" + Style.RESET_ALL)
                        continue
                    for row in table.to_pylist():
                        if row.get("ownShip", False):
                            own_ship_time_series.append(row)
                        else:
                            timeseries_dataset.append(row)
                    del table
                    gc.collect()
                except Exception as e:
                    print(Fore.RED + f"파일 읽기 오류: {file} / {e}" + Style.RESET_ALL)
            elif file.endswith('static.arrow'):
                try:
                    table = self._read_arrow_file(file)
                    if table is None:
                        print(Fore.RED + f"Arrow 파일 읽기 실패: {file}" + Style.RESET_ALL)
                        continue
                    static_dataset.extend(table.to_pylist())
                    del table
                    gc.collect()
                except Exception as e:
                    print(Fore.RED + f"파일 읽기 오류: {file} / {e}" + Style.RESET_ALL)
            elif file.endswith('.json'):
                try:
                    with open(file, 'r', encoding='utf-8') as json_file:
                        simulation_result = json.load(json_file)
                except Exception as e:
                    print(Fore.RED + f"JSON 파일 읽기 오류: {file} / {e}" + Style.RESET_ALL)

        if extract_dir and os.path.exists(extract_dir):
            try:
                shutil.rmtree(extract_dir)
            except Exception as e:
                print(Fore.RED + f"추출된 디렉토리 삭제 실패: {extract_dir} / {e}" + Style.RESET_ALL)

        return {
            "timeseries_dataset": timeseries_dataset,
            "own_ship_time_series": own_ship_time_series,
            "static_dataset": static_dataset,
            "simulation_result": simulation_result
        }