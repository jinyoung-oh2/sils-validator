import os
import zipfile
import shutil
import pyarrow as pa
import pyarrow.ipc as ipc
import json
import matplotlib
matplotlib.use("Agg")  # 파일 저장을 위한 backend 설정
import matplotlib.pyplot as plt

from marzip_extractor import MarzipExtractor

class AggregatedTargetDistribution(MarzipExtractor):
    """
    data 폴더 내의 모든 .marzip 파일을 순회하며,
    각 파일에서 extractor.run()을 통해 timeseries_dataset(타선 정보)를 추출합니다.
    각 마집 파일 내에서는 동일 id가 있을 경우 최초(timeStamp가 가장 작은) 레코드만 취하고,
    여러 마집 파일에서 나온 타겟 정보를 모두 플롯합니다.
    Base Route 함께 오버레이합니다.
    """
    def __init__(self, base_data_dir):
        self.base_data_dir = base_data_dir
        self.all_targets = []         # 모든 마집 파일에서 추출한 타선 데이터를 저장 (source 필드 추가)
        self.first_extractor = None   # 첫 번째 성공한 마집 파일 extractor (own_ship, base_route 용)

    def process_all_files(self):
        """
        base_data_dir 아래의 모든 .marzip 파일을 순회하며,
        각 파일에서 extractor.run()을 호출하여 타선 정보를 추출합니다.
        각 타겟 레코드에 'source' 필드를 추가하여, 동일 id라도 마집 파일별로 구분할 수 있도록 합니다.
        """
        for root, dirs, files in os.walk(self.base_data_dir):
            for file in files:
                if file.endswith(".marzip"):
                    marzip_file = os.path.join(root, file)
                    try:
                        extractor = MarzipExtractor(marzip_file)
                        extractor.run(marzip_file)  # 내부에서 timeseries_dataset, base_route, sils_events 등이 채워짐
                        targets = extractor.timeseries_dataset  # 타선 데이터
                        # 각 타겟에 현재 마집 파일 정보를 추가
                        for target in targets:
                            target["source"] = marzip_file
                        self.all_targets.extend(targets)
                        print(f"Processed {marzip_file} - {len(targets)} records")
                        if self.first_extractor is None:
                            self.first_extractor = extractor
                    except Exception as e:
                        print(f"Error processing {marzip_file}: {e}")

    def get_aggregated_targets(self):
        """
        self.all_targets에서 각 (source, id)별로 가장 빠른(timeStamp가 가장 작은) 레코드를 선택하여 반환합니다.
        즉, 동일 마집 파일 내에서는 id 중복 체크를 수행하지만, 다른 마집 파일의 동일 id는 별도로 취급합니다.
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

    def plot_distribution(self, output_file):
        """
        Aggregated된 타선의 초기 위치를 플롯합니다.
        첫 번째 마집 파일에서 추출한 own_ship 기준점과 함께,
        EventPlotter의 Base Route 그리기 기능(기본 route 오버레이)도 포함합니다.
        """
        aggregated_targets = self.get_aggregated_targets()
        if not aggregated_targets:
            print("타선 데이터가 없습니다.")
            return

        # own_ship 기준점 추출 (첫 번째 extractor의 sils_events 사용)
        if not (self.first_extractor and self.first_extractor.sils_events):
            print("own_ship 데이터가 없습니다.")
            return

        first_event = self.first_extractor.sils_events[0]
        own_ship_event = first_event.get("own_ship_event")
        if not (own_ship_event and "position" in own_ship_event):
            print("own_ship 위치 데이터가 없습니다.")
            return

        ship_lat = own_ship_event["position"].get("latitude")
        ship_lon = own_ship_event["position"].get("longitude")
        if ship_lat is None or ship_lon is None:
            print("own_ship 위치 데이터가 불완전합니다.")
            return

        fig, ax = plt.subplots(figsize=(12, 10))
        
        # 모든 aggregated 타겟 플롯 (절대 좌표)
        target_lats = [rec.get("lat") for rec in aggregated_targets if rec.get("lat") is not None]
        target_lons = [rec.get("lon") for rec in aggregated_targets if rec.get("lon") is not None]
        ax.scatter(target_lons, target_lats, c='blue', marker='o', s=20, alpha=0.6, label='Target Distribution')
        
        # own_ship 기준점 플롯
        ax.plot(ship_lon, ship_lat, 'r*', markersize=15, label='Own Ship (Reference)')
        
        # EventPlotter의 Base Route 그리기 기능
        if self.first_extractor.base_route:
            try:
                lat_base = [point["position"]["latitude"] for point in self.first_extractor.base_route if "position" in point]
                lon_base = [point["position"]["longitude"] for point in self.first_extractor.base_route if "position" in point]
                if lat_base and lon_base:
                    ax.plot(lon_base, lat_base, marker='o', linestyle='-', color='black', label='Base Route')
            except Exception as e:
                print(f"Base Route 그리기 실패: {e}")
        else:
            print("Base Route 데이터가 없습니다.")
        
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title("Aggregated Target Distribution with Base Route")
        ax.legend()
        ax.grid(True)
        plt.savefig(output_file)
        plt.close()
        print(f"플롯이 {output_file}에 저장되었습니다.")


if __name__ == "__main__":
    base_data_dir = "data"      # 최상위 데이터 폴더
    output_dir = "plot_result"  # 결과 플롯 저장 폴더
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "aggregated_target_distribution.png")
    
    aggregator = AggregatedTargetDistribution(base_data_dir)
    aggregator.process_all_files()
    aggregator.plot_distribution(output_file)
