import os
import concurrent.futures
import zipfile
import shutil
import pyarrow as pa
import pyarrow.ipc as ipc
import json
import matplotlib
matplotlib.use("Agg")  # 파일 저장을 위한 backend 설정
import matplotlib.pyplot as plt
import math
import numpy as np

from marzip_extractor import MarzipExtractor

def process_single_marzip(marzip_file):
    """
    단일 마집 파일을 처리합니다.
    extractor.run()을 호출하여 타선 데이터를 추출하고,
    각 타겟에 'source' 필드를 추가합니다.
    성공하면 (marzip_file, targets, extractor)를 반환하고,
    실패하면 빈 리스트와 None을 반환합니다.
    """
    try:
        extractor = MarzipExtractor(marzip_file)
        extractor.run(marzip_file)  # 내부에서 timeseries_dataset, base_route, sils_events 등이 채워짐
        targets = extractor.timeseries_dataset  # 타선 데이터
        for target in targets:
            target["source"] = marzip_file
        return (marzip_file, targets, extractor)
    except Exception as e:
        print(f"Error processing {marzip_file}: {e}")
        return (marzip_file, [], None)

class TargetDistribution(MarzipExtractor):
    """
    data 폴더 내의 모든 .marzip 파일을 병렬로 순회하며,
    각 파일에서 extractor.run()을 통해 timeseries_dataset(타선 정보)를 추출합니다.
    각 마집 파일 내에서는 동일 id가 있을 경우 최초(timeStamp가 가장 작은) 레코드만 취하고,
    여러 마집 파일에서 나온 타겟 정보를 모두 플롯합니다.
    Base Route도 함께 오버레이합니다.
    """
    def __init__(self, base_data_dir):
        self.base_data_dir = base_data_dir
        self.all_targets = []         # 모든 마집 파일에서 추출한 타선 데이터를 저장 (source 필드 추가)
        self.first_extractor = None   # 첫 번째 성공한 마집 파일의 extractor (own_ship, base_route 용)

    def process_all_files(self):
        """
        base_data_dir 아래의 모든 .marzip 파일을 재귀적으로 검색하여,
        각 파일을 병렬로 처리합니다.
        각 결과로부터 타선 데이터를 self.all_targets에 누적하고,
        첫 번째 성공한 extractor를 self.first_extractor로 설정합니다.
        """
        # 모든 마집 파일 경로 수집
        marzip_files = []
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                if file.endswith(".marzip"):
                    marzip_files.append(os.path.join(root, file))
            print(f"Found {len([f for f in files if f.endswith('.marzip')])} .marzip files in {root}")
        print(f"총 {len(marzip_files)}개의 마집 파일을 찾았습니다.")

        # 병렬 처리 (프로세스 풀)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(executor.map(process_single_marzip, marzip_files))

        # 결과 합치기
        for marzip_file, targets, extractor in results:
            self.all_targets.extend(targets)
            if self.first_extractor is None and extractor is not None:
                self.first_extractor = extractor

        print(f"모든 파일 처리 완료. 전체 타겟 개수: {len(self.all_targets)}")

    def get_aggregated_targets(self):
        """
        self.all_targets에서 각 (source, id)별로 가장 빠른(timeStamp가 가장 작은) 레코드를 선택하여 반환합니다.
        동일 마집 파일 내에서는 id 중복 체크를 수행하지만, 다른 마집 파일의 동일 id는 별도로 취급합니다.
        """
        aggregated = {}
        for record in self.all_targets:
            target_id = record.get("id")
            timestamp = record.get("timeStamp")
            source = record.get("source")
            if target_id is None or timestamp is None or source is None:
                continue
            key = (source, target_id)
            if key not in aggregated or timestamp < aggregated[key]["timeStamp"]:
                aggregated[key] = record
        return list(aggregated.values())

    def plot_distribution(self, output_file, convert_to_nm=False, sample_fraction=1.0):
        """
        Aggregated된 타선의 초기 위치를 플롯합니다.
        첫 번째 마집 파일에서 추출한 own_ship 기준점과 함께,
        Base Route도 오버레이합니다.
        sog 값에 따라 색상을 매핑하여 scatter plot을 생성합니다.
        
        convert_to_nm가 True이면, own_ship 기준 상대 좌표를 NM 단위로 변환합니다.
        sample_fraction이 1.0 미만이면, 해당 비율만큼 데이터를 무작위 샘플링하여 플롯합니다.
        """
        aggregated_targets = self.get_aggregated_targets()
        if not aggregated_targets:
            print("타선 데이터가 없습니다.")
            return

        # own_ship 기준점 추출 (첫 번째 extractor의 events 사용)
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

        # aggregated 타겟의 위도, 경도, sog 값 추출
        target_lats = [rec.get("lat") for rec in aggregated_targets if rec.get("lat") is not None]
        target_lons = [rec.get("lon") for rec in aggregated_targets if rec.get("lon") is not None]
        target_sog = [rec.get("sog", 0) for rec in aggregated_targets if rec.get("sog") is not None]

        # 데이터 샘플링 (전체 데이터의 sample_fraction만 사용)
        if sample_fraction < 1.0:
            total_points = len(target_lats)
            sample_size = int(total_points * sample_fraction)
            indices = np.random.choice(np.arange(total_points), size=sample_size, replace=False)
            target_lats = np.array(target_lats)[indices]
            target_lons = np.array(target_lons)[indices]
            target_sog = np.array(target_sog)[indices]

        fig, ax = plt.subplots(figsize=(12, 10))
        
        if convert_to_nm:
            # own_ship을 기준으로 상대 좌표를 NM 단위로 변환 (1도 위도 ≈ 60 NM)
            ship_lat_rad = math.radians(ship_lat)
            target_lats = [(lat - ship_lat) * 60 for lat in target_lats]
            target_lons = [(lon - ship_lon) * 60 * math.cos(ship_lat_rad) for lon in target_lons]
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
            # Base Route 좌표 변환
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
            base_lat = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point] if self.first_extractor.base_route else []
            base_lon = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point] if self.first_extractor.base_route else []

        # sog 값에 따라 색상을 매핑하여 scatter plot 생성
        sc = ax.scatter(
            target_lons, 
            target_lats, 
            c=target_sog,           # sog 값에 따라 색상 결정
            cmap='Blues',           # 낮은 값은 연한 파랑, 높은 값은 짙은 파랑
            marker='o', 
            s=2, 
            alpha=0.6, 
            vmin=0,                 # sog 최소값
            vmax=40,                # sog 최대값
            label='Target Distribution'
        )
        plt.colorbar(sc, ax=ax, label='SOG (kn)')
        
        # own_ship 기준점 플롯
        if convert_to_nm:
            ax.plot(0, 0, 'r*', markersize=15, label='Own Ship (Reference)')
        else:
            ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        # Base Route 그리기 (첫 번째 extractor 사용)
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
    base_data_dir = "data/ver014_20205215_basic_test"      # 최상위 데이터 폴더
    output_dir = "plot_result/ver014_20205215_basic_test"     # 결과 플롯 저장 폴더
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "target_distribution.png")
    
    aggregator = TargetDistribution(base_data_dir)
    aggregator.process_all_files()
    # convert_to_nm=True 옵션과 sample_fraction=0.1 (10% 샘플링) 옵션을 적용합니다.
    aggregator.plot_distribution(output_file, convert_to_nm=True, sample_fraction=0.1)


if __name__ == "__main__":
    main()
