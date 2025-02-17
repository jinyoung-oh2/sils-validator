import os
import concurrent.futures
import zipfile
import shutil
import pyarrow as pa
import pyarrow.ipc as ipc
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
from functools import partial
import gc

from marzip_extractor import MarzipExtractor

def process_single_marzip(marzip_file, sample_fraction=1.0):
    """
    단일 마집 파일을 처리합니다.
    extractor.run()을 호출하여 타선 데이터를 추출하고,
    각 타겟에 'source' 필드를 추가합니다.
    sample_fraction이 1.0 미만이면, 해당 비율만큼 타겟을 무작위 샘플링합니다.
    성공하면 (marzip_file, targets, extractor)를 반환하고,
    실패하면 빈 리스트와 None을 반환합니다.
    """
    try:
        extractor = MarzipExtractor(marzip_file)
        extractor.run(marzip_file)
        targets = extractor.timeseries_dataset
        if sample_fraction < 1.0 and targets:
            total = len(targets)
            sample_size = int(total * sample_fraction)
            if sample_size < 1:
                sample_size = 1
            indices = np.random.permutation(total)[:sample_size]
            targets = [targets[i] for i in indices]
        for target in targets:
            target["source"] = marzip_file
        return (marzip_file, targets, extractor)
    except Exception as e:
        print(f"Error processing {marzip_file}: {e}")
        return (marzip_file, [], None)

class TargetDistribution(MarzipExtractor):
    """
    data 폴더 내의 모든 .marzip 파일을 배치 및 병렬 처리하여,
    각 파일에서 extractor.run()을 통해 timeseries_dataset(타선 정보)를 추출합니다.
    각 파일에서 추출한 타겟은 (source, id)를 기준으로 바로 집계하여,
    중복된 타겟 중 timeStamp가 가장 작은 레코드만 유지합니다.
    Base Route도 함께 오버레이합니다.
    """
    def __init__(self, base_data_dir):
        self.base_data_dir = base_data_dir
        self.all_targets = []         # 최종 집계된 타겟 데이터
        self.first_extractor = None   # 첫 번째 성공한 마집 파일의 extractor (own_ship, base_route 용)

    def process_all_files(self, sample_fraction=1.0, max_workers=4, batch_size=1000):
        # 모든 마집 파일 경로 수집
        marzip_files = []
        for root, dirs, files in os.walk(self.base_data_dir):
            batch_count = len([f for f in files if f.endswith('.marzip')])
            print(f"Found {batch_count} .marzip files in {root}")
            for file in files:
                if file.endswith(".marzip"):
                    marzip_files.append(os.path.join(root, file))
        total_files = len(marzip_files)
        print(f"총 {total_files}개의 마집 파일을 찾았습니다.")

        aggregated = {}  # (source, id) => record

        # 배치 단위로 처리
        for i in range(0, total_files, batch_size):
            batch = marzip_files[i:i+batch_size]
            print(f"\n[Batch {i} ~ {i+len(batch)}] Processing {len(batch)} files...")
            process_func = partial(process_single_marzip, sample_fraction=sample_fraction)
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(process_func, batch))
            for marzip_file, targets, extractor in results:
                if self.first_extractor is None and extractor is not None:
                    self.first_extractor = extractor
                for record in targets:
                    target_id = record.get("id")
                    timestamp = record.get("timeStamp")
                    source = record.get("source")
                    if target_id is None or timestamp is None or source is None:
                        continue
                    key = (source, target_id)
                    if key not in aggregated or timestamp < aggregated[key]["timeStamp"]:
                        aggregated[key] = record
            gc.collect()
            print(f"Batch {i} ~ {i+len(batch)} 완료. 누적 집계 타겟 개수: {len(aggregated)}")
        
        self.all_targets = list(aggregated.values())
        print(f"\n모든 파일 처리 완료. 전체 집계 타겟 개수: {len(self.all_targets)}")

    def get_aggregated_targets(self):
        return self.all_targets

    def plot_distribution(self, output_file, convert_to_nm=False, sample_fraction=1.0):
        """
        기존 Matplotlib scatter plot을 사용하여 집계된 타선의 초기 위치를 플롯합니다.
        첫 번째 마집 파일에서 추출한 own_ship 기준점과 Base Route를 함께 오버레이합니다.
        sog 값에 따라 색상을 매핑합니다.
        
        convert_to_nm이 True이면, own_ship 기준 상대 좌표를 NM 단위로 변환합니다.
        sample_fraction이 1.0 미만이면, 해당 비율만큼 데이터를 무작위 샘플링합니다.
        """
        aggregated_targets = self.get_aggregated_targets()
        if not aggregated_targets:
            print("타선 데이터가 없습니다.")
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

        target_lats = [rec.get("lat") for rec in aggregated_targets if rec.get("lat") is not None]
        target_lons = [rec.get("lon") for rec in aggregated_targets if rec.get("lon") is not None]
        target_sog = [rec.get("sog", 0) for rec in aggregated_targets if rec.get("sog") is not None]

        if sample_fraction < 1.0:
            total_points = len(target_lats)
            sample_size = int(total_points * sample_fraction)
            indices = np.random.choice(np.arange(total_points), size=sample_size, replace=False)
            target_lats = np.array(target_lats)[indices]
            target_lons = np.array(target_lons)[indices]
            target_sog = np.array(target_sog)[indices]

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
            cmap='viridis',  # 또는 'viridis' 등으로 변경해 보세요.
            marker='o', 
            s=2,           # 마커 크기를 늘림 (예: 5 또는 10)
            alpha=0.8,     # 투명도 조정 (예: 0.8)
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
        
        ax.set_title("Aggregated Target Distribution with Base Route")
        ax.legend()
        ax.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"플롯이 {output_file}에 저장되었습니다.")

def main():
    base_data_dir = "data/ver014_20205215_basic_test"
    output_dir = "plot_result/ver014_20205215_basic_test"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "target_distribution.png")
    
    aggregator = TargetDistribution(base_data_dir)
    aggregator.process_all_files(sample_fraction=0.5, max_workers=4, batch_size=1000)
    aggregator.plot_distribution(output_file, convert_to_nm=True, sample_fraction=0.1)

if __name__ == "__main__":
    main()
