import os
import concurrent.futures
import pyarrow as pa
import pyarrow.ipc as ipc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import gc
import random
import itertools

from marzip_extractor import MarzipExtractor

class TargetDistribution(MarzipExtractor):
    """
    data 폴더 내의 모든 .marzip 파일을 배치 및 병렬 처리하여,
    각 파일에서 extractor.run()을 통해 타겟 정보를 추출합니다.
    """
    def __init__(self, base_data_dir):
        self.base_data_dir = base_data_dir
        self.all_targets = []         # process_initialize_target_ship 으로 누적된 초기 타겟 데이터
        self.all_event_targets = []   # process_event_target_ship 으로 누적된 이벤트 타겟 데이터
        self.first_extractor = None   # 첫 번째 성공한 마집 파일의 extractor (own_ship, base_route 용)
        self.marzip_files = []        # 모든 마집 파일 경로

    def process_all_files(self):
        # 모든 마집 파일 경로 수집
        marzip_files = []
        for root, dirs, files in os.walk(self.base_data_dir):
            batch_count = len([f for f in files if f.endswith('.marzip')])
            print(f"Found {batch_count} .marzip files in {root}")
            for file in files:
                if file.endswith(".marzip"):
                    marzip_files.append(os.path.join(root, file))
        total_files = len(marzip_files)
        self.marzip_files = marzip_files
        print(f"총 {total_files}개의 마집 파일을 찾았습니다.")

    @staticmethod
    def process_initialize_target_ship(file_path):
        """
        주어진 파일(file_path)에 대해 MarzipExtractor를 실행하고,
        simulation_result의 targetShips 리스트에서 각 타겟의 initial 데이터를 추출합니다.
        """
        try:
            extractor = MarzipExtractor(file_path)
            extractor.run(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return file_path, [], None

        simulation_result = extractor.simulation_result
        if not simulation_result:
            return file_path, [], extractor

        target_ships = simulation_result.get("trafficSituation", {}).get("targetShips", [])
        records = []

        for target in target_ships:
            initial = target.get("initial")
            if not initial:
                continue
            record = {
                "cog": initial.get("cog"),
                "heading": initial.get("heading"),
                "position": initial.get("position"),
                "sog": initial.get("sog"),
            }
            records.append(record)

        return file_path, records, extractor

    @staticmethod
    def process_event_target_ship(file_path):
        """
        주어진 파일(file_path)에 대해 MarzipExtractor를 실행하고,
        event 내의 target_ships 리스트에서 각 타겟의 데이터를 추출합니다.
        caPathGenFail 값도 함께 기록합니다.
        """
        try:
            extractor = MarzipExtractor(file_path)
            extractor.run(file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return file_path, [], None

        events = extractor.events
        records = []

        if not events:
            # 이벤트가 0개인 경우
            print(f"[DEBUG] No events found in file: {file_path}")
            simulation_result = extractor.simulation_result
            if not simulation_result:
                return file_path, [], extractor

            target_ships = simulation_result.get("trafficSituation", {}).get("targetShips", [])
            for target in target_ships:
                initial = target.get("initial")
                if not initial:
                    continue
                record = {
                    "cog": initial.get("cog"),
                    "heading": initial.get("heading"),
                    "position": initial.get("position"),
                    "sog": initial.get("sog"),
                    "caPathGenFail": None
                }
                records.append(record)  # 각 레코드를 for 루프 내에서 추가합니다.
            return file_path, records, extractor

        # 이벤트가 존재하는 경우 (현재는 첫 번째 이벤트만 사용)
        # 디버그: 첫 번째 이벤트의 caPathGenFail 값 확인
        ca_pg_fail = events[0].get("ca_path_gen_fail")

        target_ships = events[0].get("target_ships", [])
        for target in target_ships:
            record = {
                "cog": target.get("cog"),
                "heading": target.get("heading"),
                "position": target.get("position"),
                "sog": target.get("sog"),
                "caPathGenFail": ca_pg_fail,
                "tcpa": target.get("tcpa"),
                "cpa": target.get("cpa"),
                "colreg": target.get("encounterType")

            }
            records.append(record)

        return file_path, records, extractor


    def collect_initial_targets(self, sample_fraction=1.0, max_workers=4, batch_size=1000):
        # 파일 리스트 자체를 샘플링 (전체 파일 중 일부만 처리)
        if sample_fraction < 1.0:
            sampled_files = random.sample(self.marzip_files, int(len(self.marzip_files) * sample_fraction))
        else:
            sampled_files = self.marzip_files
            
        total_files = len(sampled_files)
        aggregated = []  # 모든 초기 타겟을 누적

        for i in range(0, total_files, batch_size):
            batch = sampled_files[i:i + batch_size]
            print(f"\n[Batch {i} ~ {i + len(batch)}] Processing {len(batch)} files for initial targets...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        TargetDistribution.process_initialize_target_ship,
                        batch
                    )
                )

            for file_path, records, extractor in results:
                # 최초 extractor 저장 (own_ship, base_route 정보용)
                if self.first_extractor is None and extractor is not None:
                    self.first_extractor = extractor
                aggregated.extend(records)

            gc.collect()
            print(f"Batch {i} ~ {i + len(batch)} 완료. 누적 초기 타겟 개수: {len(aggregated)}")

        self.all_targets = aggregated
        print(f"\n모든 파일 처리 완료 (initial). 전체 초기 타겟 개수: {len(self.all_targets)}")

    def collect_event_targets(self, sample_fraction=1.0, max_workers=4, batch_size=1000):
        """
        process_event_target_ship 메서드를 사용하여 이벤트 기반의 타겟 데이터를 수집합니다.
        """
        if sample_fraction < 1.0:
            sampled_files = random.sample(self.marzip_files, int(len(self.marzip_files) * sample_fraction))
        else:
            sampled_files = self.marzip_files

        total_files = len(sampled_files)
        aggregated = []  # 모든 이벤트 타겟을 누적

        for i in range(0, total_files, batch_size):
            batch = sampled_files[i:i + batch_size]
            print(f"\n[Batch {i} ~ {i + len(batch)}] Processing {len(batch)} files for event targets...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(
                    executor.map(
                        TargetDistribution.process_event_target_ship,
                        batch
                    )
                )

            for file_path, records, extractor in results:
                if self.first_extractor is None and extractor is not None:
                    self.first_extractor = extractor
                aggregated.extend(records)

            gc.collect()
            print(f"Batch {i} ~ {i + len(batch)} 완료. 누적 이벤트 타겟 개수: {len(aggregated)}")

        self.all_event_targets = aggregated
        print(f"\n모든 파일 처리 완료 (event). 전체 이벤트 타겟 개수: {len(self.all_event_targets)}")

    def _plot_distribution(self, aggregated_targets, output_file, title, convert_to_nm=False):
        """
        초기 타겟 데이터를 기반으로 플롯을 생성합니다.
        (이전과 동일한 방식으로 SOG에 따른 색상 매핑)
        """
        if not aggregated_targets:
            print(f"{title}: 타겟 데이터가 없습니다.")
            return

        if not (self.first_extractor and self.first_extractor.events):
            print("own_ship 데이터가 없습니다.")
            return

        first_event = self.first_extractor.events[0]
        own_ship_event = first_event.get("own_ship_event")
        if not (own_ship_event and "position" in own_ship_event):
            print("own_ship 위치 데이터가 없습니다.")
            return

        ship_lat = own_ship_event["position"].get("latitude")
        ship_lon = own_ship_event["position"].get("longitude")
        if ship_lat is None or ship_lon is None:
            print("own_ship 위치 데이터가 불완전합니다.")
            return

        target_lats = [rec["position"].get("latitude") for rec in aggregated_targets 
                       if rec.get("position") and "latitude" in rec["position"]]
        target_lons = [rec["position"].get("longitude") for rec in aggregated_targets 
                       if rec.get("position") and "longitude" in rec["position"]]
        target_sog = [rec.get("sog", 0) for rec in aggregated_targets if rec.get("sog") is not None]

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if convert_to_nm:
            ship_lat_rad = math.radians(ship_lat)
            target_lats = [(lat - ship_lat) * 60 for lat in target_lats]
            target_lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in target_lons]
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
            base_lat = []
            base_lon = []
            if self.first_extractor.base_route:
                for point in self.first_extractor.base_route:
                    if "position" in point:
                        base_lat.append((point["position"]["latitude"] - ship_lat) * 60)
                        base_lon.append((point["position"]["longitude"] - ship_lon) * 60 * math.cos(ship_lat_rad))
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            base_lat = []
            base_lon = []
            if self.first_extractor.base_route:
                base_lat = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point]
                base_lon = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point]

        sc = ax.scatter(
            target_lons, 
            target_lats, 
            c=target_sog,
            cmap='viridis',
            marker='o', 
            s=10,
            alpha=0.8,
            vmin=0,
            vmax=40,
            label='Target Distribution'
        )
        plt.colorbar(sc, ax=ax, label='SOG (kn)')
        
        if convert_to_nm:
            ax.plot(0, 0, 'r*', markersize=15, label='Own Ship (Reference)')
        else:
            ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        if self.first_extractor.base_route:
            try:
                if base_lat and base_lon:
                    ax.plot(base_lon, base_lat, marker='o', linestyle='-', color='black', label='Base Route')
            except Exception as e:
                print(f"Base Route 그리기 실패: {e}")
        else:
            print("Base Route 데이터가 없습니다.")
        
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"플롯이 {output_file}에 저장되었습니다.")

    def plot_initial_distribution(self, output_file, convert_to_nm=False):
        """
        수집된 초기 타겟 데이터를 기반으로 플롯 생성
        """
        self._plot_distribution(
            aggregated_targets=self.all_targets,
            output_file=output_file,
            title="Initial Target Distribution with Base Route",
            convert_to_nm=convert_to_nm,
        )

    def plot_event_distribution(self, output_file, convert_to_nm=False):
        """
        수집된 이벤트 타겟 데이터를 기반으로 플롯 생성  
        caPathGenFail 값에 따라 색상이 달라집니다.  
        - True: 빨간색 (실패: 선속(sog)와 방향(heading)을 화살표로 표기)  
        - False: 초록색 (성공)  
        - None : 파란색 (N/A)
        """
        aggregated_targets = self.all_event_targets
        if not aggregated_targets:
            print("Event Target Distribution: 타겟 데이터가 없습니다.")
            return

        if not (self.first_extractor and self.first_extractor.events):
            print("own_ship 데이터가 없습니다.")
            return

        first_event = self.first_extractor.events[0]
        own_ship_event = first_event.get("own_ship_event")
        if not (own_ship_event and "position" in own_ship_event):
            print("own_ship 위치 데이터가 없습니다.")
            return

        ship_lat = own_ship_event["position"].get("latitude")
        ship_lon = own_ship_event["position"].get("longitude")
        if ship_lat is None or ship_lon is None:
            print("own_ship 위치 데이터가 불완전합니다.")
            return

        # 그룹별로 좌표 및 추가 정보를 저장 (fail: caPathGenFail True)
        red_lats, red_lons, red_sogs, red_headings = [], [], [], []
        green_lats, green_lons = [], []
        blue_lats, blue_lons = [], []  # caPathGenFail 값이 None 또는 기타인 경우

        for rec in aggregated_targets:
            pos = rec.get("position")
            if not (pos and "latitude" in pos and "longitude" in pos):
                continue
            lat = pos.get("latitude")
            lon = pos.get("longitude")
            cp = rec.get("caPathGenFail")
            if cp is True:
                red_lats.append(lat)
                red_lons.append(lon)
                red_sogs.append(rec.get("sog", 0))
                red_headings.append(rec.get("heading", 0))
            elif cp is False:
                green_lats.append(lat)
                green_lons.append(lon)
            else:
                blue_lats.append(lat)
                blue_lons.append(lon)

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if convert_to_nm:
            ship_lat_rad = math.radians(ship_lat)
            red_lats = [(lat - ship_lat) * 60 for lat in red_lats]
            red_lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in red_lons]
            green_lats = [(lat - ship_lat) * 60 for lat in green_lats]
            green_lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in green_lons]
            blue_lats = [(lat - ship_lat) * 60 for lat in blue_lats]
            blue_lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in blue_lons]
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # 각 그룹별 scatter plot

        
        ax.scatter(green_lons, green_lats, color="green", marker='o', s=10, alpha=0.8, label="Path Gen Success")
        ax.scatter(blue_lons, blue_lats, color="blue", marker='o', s=10, alpha=0.8, label="Path Gen N/A")
        
        # Fail(경로 생성 실패) 케이스: 빨간색 scatter와 함께, SOG와 Heading 정보를 이용한 화살표 표기
        ax.scatter(red_lons, red_lats, color="red", marker='o', s=10, alpha=0.8, label="path gen failure")
        
        if convert_to_nm:
            ax.plot(0, 0, 'r*', markersize=15, label='Own Ship (Reference)')
        else:
            ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        # Base Route 그리기
        if self.first_extractor.base_route:
            base_lat = []
            base_lon = []
            if convert_to_nm:
                for point in self.first_extractor.base_route:
                    if "position" in point:
                        base_lat.append((point["position"]["latitude"] - ship_lat) * 60)
                        base_lon.append((point["position"]["longitude"] - ship_lon) * 60 * math.cos(ship_lat_rad))
            else:
                base_lat = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point]
                base_lon = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point]
            try:
                if base_lat and base_lon:
                    ax.plot(base_lon, base_lat, marker='o', linestyle='-', color='black', label='Base Route')
            except Exception as e:
                print(f"Base Route 그리기 실패: {e}")
        else:
            print("Base Route 데이터가 없습니다.")

        ax.set_title("Event Target Distribution with Base Route (caPathGenFail Colored)")
        ax.legend()
        ax.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"플롯이 {output_file}에 저장되었습니다.")

    def plot_event_distribution_by_tcpa(self, output_file, convert_to_nm=True):
        """
        수집된 이벤트 타겟 데이터를 기반으로 tcpa 값에 따라 색상이 달라지는 플롯 생성  
        - tcpa 값이 낮을수록 (빠른 충돌 위험) 다른 색상으로 표시됩니다.
        """
        aggregated_targets = self.all_event_targets
        if not aggregated_targets:
            print("TCPA Target Distribution: 타겟 데이터가 없습니다.")
            return

        if not (self.first_extractor and self.first_extractor.events):
            print("own_ship 데이터가 없습니다.")
            return

        first_event = self.first_extractor.events[0]
        own_ship_event = first_event.get("own_ship_event")
        if not (own_ship_event and "position" in own_ship_event):
            print("own_ship 위치 데이터가 없습니다.")
            return

        ship_lat = own_ship_event["position"].get("latitude")
        ship_lon = own_ship_event["position"].get("longitude")
        if ship_lat is None or ship_lon is None:
            print("own_ship 위치 데이터가 불완전합니다.")
            return

        # tcpa 값과 좌표를 저장할 리스트 생성
        lats, lons, tcpa_values = [], [], []
        for rec in aggregated_targets:
            pos = rec.get("position")
            if not (pos and "latitude" in pos and "longitude" in pos):
                continue
            lat = pos.get("latitude")
            lon = pos.get("longitude")
            tcpa = rec.get("tcpa")
            if tcpa is None:
                continue  # tcpa 값이 없는 경우는 건너뜁니다.
            lats.append(lat)
            lons.append(lon)
            try:
                tcpa_values.append(float(tcpa))
            except Exception as e:
                print(f"tcpa 변환 오류: {e}")
                tcpa_values.append(0)

        if not lats or not lons or not tcpa_values:
            print("tcpa 데이터를 포함한 타겟이 없습니다.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if convert_to_nm:
            ship_lat_rad = math.radians(ship_lat)
            lats = [(lat - ship_lat) * 60 for lat in lats]
            lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in lons]
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # scatter plot: tcpa 값에 따라 색상 지정 (연속형 컬러맵 사용)
        sc = ax.scatter(lons, lats, c=tcpa_values, cmap="viridis", marker='o', s=10, alpha=0.8)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("TCPA Value")
        
        # own_ship 위치 표시 (참조점)
        if convert_to_nm:
            ax.plot(0, 0, 'r*', markersize=15, label='Own Ship (Reference)')
        else:
            ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        # Base Route 그리기 (있을 경우)
        if self.first_extractor.base_route:
            base_lat = []
            base_lon = []
            if convert_to_nm:
                ship_lat_rad = math.radians(ship_lat)
                for point in self.first_extractor.base_route:
                    if "position" in point:
                        base_lat.append((point["position"]["latitude"] - ship_lat) * 60)
                        base_lon.append((point["position"]["longitude"] - ship_lon) * 60 * math.cos(ship_lat_rad))
            else:
                base_lat = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point]
                base_lon = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point]
            try:
                if base_lat and base_lon:
                    ax.plot(base_lon, base_lat, marker='o', linestyle='-', color='black', label='Base Route')
            except Exception as e:
                print(f"Base Route 그리기 실패: {e}")
        else:
            print("Base Route 데이터가 없습니다.")

        ax.set_title("Event Target Distribution by TCPA")
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"TCPA 기반 타겟 분포 플롯이 {output_file}에 저장되었습니다.")


    def plot_tcpacpa_scatter(self, output_file):
        """
        각 타겟 레코드에서 'tcpa', 'cpa', 'colreg' 값을 추출하여,
        가로축(CPA), 세로축(TCPA) scatter plot을 생성합니다.
        각 타겟은 colreg 값에 따라 두 그룹(그룹 1: "CR-GW", "OT-GW", "HO",
        그룹 2: "OT-SO", "CR-SO")으로 나누어 별도의 플롯으로 표시되며,
        TCPA가 50과 60인 위치에 두께가 늘어난 수평 점선, CPA가 0.5인 위치에 수직 점선을 추가합니다.
        
        그룹별 색상 매핑은 아래와 같이 설정됩니다.
            Group1: green
            Group2: red
        
        Parameters:
            output_file (str): 생성된 플롯 이미지 파일의 기본 경로.
                            예를 들어 "scatter.png"를 전달하면
                            그룹별로 "scatter_Group1.png", "scatter_Group2.png"로 저장됨.
        """
        import os

        # colreg 그룹화 함수: 원래 colreg 값을 두 그룹으로 매핑
        def map_colreg(colreg_val):
            if colreg_val in ["CR-GW", "OT-GW", "HO"]:
                return "Group1"  # 그룹 1
            elif colreg_val in ["OT-SO", "CR-SO"]:
                return "Group2"  # 그룹 2
            else:
                return None  # 관심 그룹이 아닌 경우

        # 그룹별 색상 매핑
        group_color_mapping = {
            "Group1": "blue",
            "Group2": "purple"
        }
        
        # 그룹별 데이터 저장
        groups = {}
        for rec in self.all_event_targets:
            tcpa = rec.get("tcpa")
            cpa = rec.get("cpa")
            orig_colreg = rec.get("colreg", "Unknown")
            group_label = map_colreg(orig_colreg)
            if group_label is None:
                continue
            if tcpa is not None and cpa is not None:
                try:
                    tcpa_val = float(tcpa)
                    cpa_val = float(cpa)
                except Exception as e:
                    print(f"TCPA 또는 CPA 변환 오류: {e}")
                    continue
                if group_label not in groups:
                    groups[group_label] = {"tcpas": [], "cpas": []}
                groups[group_label]["tcpas"].append(tcpa_val)
                groups[group_label]["cpas"].append(cpa_val)
        
        if not groups:
            print("TCPA 또는 CPA 데이터를 포함한 타겟이 없습니다.")
            return

        base, ext = os.path.splitext(output_file)
        # 그룹 1과 그룹 2를 각각 별도의 플롯으로 저장
        for group_label in ["Group1", "Group2"]:
            if group_label not in groups:
                print(f"{group_label}에 해당하는 데이터가 없습니다.")
                continue
            
            fig, ax = plt.subplots(figsize=(10, 8))
            data = groups[group_label]
            color = group_color_mapping.get(group_label, "black")
            count = len(data["tcpas"])
            ax.scatter(data["cpas"], data["tcpas"], color=color, marker='o', s=5, alpha=0.5,
                    label=f"{group_label} ({count})")
            
            # 기준선 추가:
            # TCPA가 50과 60인 위치에 수평 점선, CPA가 0.5인 위치에 수직 점선 (두께 2)
            ax.axhline(y=50, linestyle='--', color='red', linewidth=2, label='TCPA = 50')
            ax.axhline(y=60, linestyle='--', color='green', linewidth=2, label='TCPA = 60')
            ax.axvline(x=0.5, linestyle='--', color='black', linewidth=2, label='CPA = 0.5')
            
            ax.set_xlabel("CPA")
            ax.set_ylabel("TCPA")
            ax.set_title(f"CPA - TCPA Scatter Plot for {group_label}")
            ax.set_xlim(0.0, 0.7)
            ax.set_ylim(0.0, 70)
            ax.legend()
            ax.grid(True)
            
            new_output_file = f"{base}_{group_label}{ext}"
            plt.savefig(new_output_file)
            plt.close()
            print(f"CPA vs TCPA scatter plot for {group_label}이 {new_output_file}에 저장되었습니다.")



    def plot_event_distribution_by_colreg(self, output_file, convert_to_nm=True):
        """
        수집된 이벤트 타겟 데이터를 기반으로, 각 타겟의 'colreg' 값이
        지정된 경우에만(예: colreg_color_mapping에 정의된 값),
        다른 색상으로 분포를 플롯하고, 각 그룹별 타겟 개수를 레전드에 표기합니다.
        
        Parameters:
            output_file (str): 저장할 플롯 이미지 파일 경로.
            convert_to_nm (bool): 좌표를 해리 단위(Nautical Miles)로 변환할지 여부.
        """
        # 기본 색상 매핑 (원하는 값만 포함)
        colreg_color_mapping = {
            "HO": "red",
            "CR-GW": "blue",
            "CR-SO": "green",
            "OT-GW": "orange",
            "OT-SO": "purple"
            
        }
        
        # own_ship 데이터 확보
        if not (self.first_extractor and self.first_extractor.events):
            print("own_ship 데이터가 없습니다.")
            return

        first_event = self.first_extractor.events[0]
        own_ship_event = first_event.get("own_ship_event")
        if not (own_ship_event and "position" in own_ship_event):
            print("own_ship 위치 데이터가 없습니다.")
            return

        ship_lat = own_ship_event["position"].get("latitude")
        ship_lon = own_ship_event["position"].get("longitude")
        if ship_lat is None or ship_lon is None:
            print("own_ship 위치 데이터가 불완전합니다.")
            return

        # 각 타겟 레코드를 colreg 값에 따라 그룹핑 (단, colreg 값이 지정된 경우에만)
        groups = {}
        for rec in self.all_event_targets:
            colreg_val = rec.get("colreg")
            # colreg 값이 없거나, 지정된 매핑에 포함되지 않은 경우 건너뛰기
            if colreg_val is None or colreg_val not in colreg_color_mapping:
                continue
            if colreg_val not in groups:
                groups[colreg_val] = {"lats": [], "lons": []}
            pos = rec.get("position")
            if not (pos and "latitude" in pos and "longitude" in pos):
                continue
            groups[colreg_val]["lats"].append(pos.get("latitude"))
            groups[colreg_val]["lons"].append(pos.get("longitude"))

        if not groups:
            print("지정된 COLREG 값을 포함한 타겟 데이터가 없습니다.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if convert_to_nm:
            ship_lat_rad = math.radians(ship_lat)
            for key in groups:
                groups[key]["lats"] = [(lat - ship_lat) * 60 for lat in groups[key]["lats"]]
                groups[key]["lons"] = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in groups[key]["lons"]]
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
        
        # 각 그룹별로 산점도 플롯 (각 그룹에 포함된 타겟 개수를 레전드에 포함)
        for colreg_val, coords in groups.items():
            color = colreg_color_mapping.get(colreg_val, "black")  # 매핑되지 않은 경우 기본 검정색 사용 (이 경우 발생하지 않음)
            count = len(coords["lats"])
            ax.scatter(coords["lons"], coords["lats"], color=color, marker='o', s=4, alpha=0.5,
                    label=f"COLREG: {colreg_val} ({count})")

        # own_ship 위치 표시
        if convert_to_nm:
            ax.plot(0, 0, 'r*', markersize=15, label='Own Ship (Reference)')
        else:
            ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        # Base Route 그리기 (존재하는 경우)
        if self.first_extractor.base_route:
            base_lat = []
            base_lon = []
            if convert_to_nm:
                for point in self.first_extractor.base_route:
                    if "position" in point:
                        base_lat.append((point["position"]["latitude"] - ship_lat) * 60)
                        base_lon.append((point["position"]["longitude"] - ship_lon) * 60 * math.cos(ship_lat_rad))
            else:
                base_lat = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point]
                base_lon = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point]
            try:
                if base_lat and base_lon:
                    ax.plot(base_lon, base_lat, marker='o', linestyle='-', color='black', label='Base Route')
            except Exception as e:
                print(f"Base Route 그리기 실패: {e}")
        else:
            print("Base Route 데이터가 없습니다.")
        
        ax.set_title("Event Target Distribution by COLREG")
        ax.legend()
        ax.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"COLREG 기반 타겟 분포 플롯이 {output_file}에 저장되었습니다.")


def main():
    base_data_dir = "/media/avikus/One Touch/HinasControlSilsCA/CA_v0.1.4_data/COLREG_TESTING"
    output_dir = "analyze/CA_v0.1.4_data/Colreg/20250227_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 출력 파일명 지정
    output_file_initial = os.path.join(output_dir, "initial_target_distribution.png")
    output_file_event   = os.path.join(output_dir, "event_target_distribution.png")
    output_file_colreg = os.path.join(output_dir, "event_target_distribution_by_colreg.png")
    output_file_event_tcpa   = os.path.join(output_dir, "event_target_distribution_by_tcpa.png")
    output_file_tcpacpa = os.path.join(output_dir, "tcpacpa_scatter_plot.png")


    # 공통 옵션 변수 정의
    sample_fraction = 1
    max_workers = 8
    batch_size = 1000

    aggregator = TargetDistribution(base_data_dir)
    aggregator.process_all_files()
    
    print("\n[Initial Target Collection]")
    aggregator.collect_initial_targets(sample_fraction=sample_fraction, max_workers=max_workers, batch_size=batch_size)
    
    print("\n[Event Target Collection]")
    aggregator.collect_event_targets(sample_fraction=sample_fraction, max_workers=max_workers, batch_size=batch_size)
    
    print(f"\nInitial targets count: {len(aggregator.all_targets)}")
    print(f"Event targets count: {len(aggregator.all_event_targets)}")
    
    # # 초기 타겟 플롯 생성 (기존 방식: SOG에 따라 색상)
    aggregator.plot_initial_distribution(output_file_initial, convert_to_nm=True)

    # # 이벤트 타겟 플롯 생성 (caPathGenFail 값에 따라 빨간색/초록색)
    aggregator.plot_event_distribution(output_file_event, convert_to_nm=True)

    aggregator.plot_event_distribution_by_colreg(output_file_colreg, convert_to_nm=True)
    # TCPA에 따라 플롯 생성
    aggregator.plot_event_distribution_by_tcpa(output_file_event_tcpa, convert_to_nm=True)

    # ## RA Test
    # aggregator.plot_tcpacpa_scatter(output_file_tcpacpa)


if __name__ == "__main__":
    main()
