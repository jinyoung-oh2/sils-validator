import math
import os
import matplotlib
matplotlib.use("Agg")  # GUI 백엔드 설정 (파일로 저장)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

class EventPlotter(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터에서
    Safe Path, Base Route, Own Ship, Target Ships를 플롯하는 클래스입니다.
    """
    def __init__(self, marzip_file):
        """
        :param marzip_file: 이벤트 마집 파일의 경로.
        """
        super().__init__()
        self.marzip = marzip_file
        self.run(marzip_file)
        # 기존 MarzipExtractor에서 추출된 이벤트 목록은 self.events에 저장되었으므로,
        # 이를 더 간단히 self.events로 사용합니다.
        self.events = self.events

    def get_safe_path(self, safe_paths, idx):
        """
        주어진 safe_paths 리스트에서 인덱스 idx에 해당하는 safe path가 없으면
        이전 이벤트들의 safe path를 순차적으로 확인하여 처음 만나는 값을 반환합니다.
        """
        while idx >= 0:
            if idx < len(safe_paths) and safe_paths[idx]:
                return safe_paths[idx]
            idx -= 1
        return None

    def get_route_coordinates(self, route):
        """
        주어진 route의 각 point에서 위도와 경도 리스트를 추출합니다.
        """
        latitudes = [point["position"]["latitude"] for point in route]
        longitudes = [point["position"]["longitude"] for point in route]
        return latitudes, longitudes

    def set_axis_limits_based_on_base_route(self, ax):
        """
        Base Route의 좌표를 기준으로 플롯의 x, y 범위를 정사각형으로 고정합니다.
        """
        if not self.base_route:
            return
        try:
            lat_base, lon_base = self.get_route_coordinates(self.base_route)
            lat_min, lat_max = min(lat_base), max(lat_base)
            lon_min, lon_max = min(lon_base), max(lon_base)
            lat_range = lat_max - lat_min
            lon_range = lon_max - lon_min
            max_range = max(lat_range, lon_range)
            lat_mid = (lat_min + lat_max) / 2
            lon_mid = (lon_min + lon_max) / 2
            margin = max_range * 0.1
            half_range = max_range / 2 + margin

            ax.set_xlim(lon_mid - half_range, lon_mid + half_range)
            ax.set_ylim(lat_mid - half_range, lat_mid + half_range)
            ax.set_aspect('equal', adjustable='datalim')
        except Exception as e:
            print(f"Base Route를 기반으로 축 설정 실패: {e}")

    def draw_ship(self, ax, lon, lat, heading, color, ship_length=230, ship_width=30, scale=10.0, shape='star'):
        """
        실제 배의 길이와 폭을 사용하여 위도/경도 단위의 배 모양(별 또는 삼각형)을 그립니다.
        """
        if shape == 'star':
            deg_per_meter = (1 / 111320) * scale
            outer_radius = (ship_length / 2) * deg_per_meter  
            inner_radius = outer_radius * 0.5  
            vertices = []
            for i in range(10):
                angle_deg = i * 36
                r = outer_radius if i % 2 == 0 else inner_radius
                x = r * np.sin(np.radians(angle_deg))
                y = r * np.cos(np.radians(angle_deg))
                vertices.append([x, y])
            vertices = np.array(vertices)
            angle = np.radians(-heading)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            rotated_vertices = vertices.dot(rotation_matrix.T)
            rotated_vertices[:, 0] += lon
            rotated_vertices[:, 1] += lat
            star_polygon = patches.Polygon(
                rotated_vertices, closed=True,
                facecolor=color, edgecolor=color, lw=2, alpha=0.9, zorder=5
            )
            ax.add_patch(star_polygon)
        else:
            deg_per_meter = (1 / 111320) * scale  
            L_deg = ship_length * deg_per_meter  
            W_deg = ship_width * deg_per_meter   
            vertices = np.array([
                [0, L_deg/2 + L_deg/6],
                [-W_deg/2, -L_deg/2 + L_deg/6],
                [W_deg/2, -L_deg/2 + L_deg/6]
            ])
            angle = np.radians(-heading)
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle),  np.cos(angle)]
            ])
            rotated_vertices = vertices.dot(rotation_matrix.T)
            rotated_vertices[:, 0] += lon
            rotated_vertices[:, 1] += lat
            ship_polygon = patches.Polygon(
                rotated_vertices, closed=True,
                facecolor=color, edgecolor='red', lw=2, alpha=0.9, zorder=5
            )
            ax.add_patch(ship_polygon)

    def plot_all(self, output_path_pattern, fail_only = True):
        """
        이벤트별로 플롯을 생성하여 저장합니다.
        
        :param output_path_pattern: 저장 경로 패턴 (예: "result/event_{}_plot.png")
                                    {}에 이벤트 인덱스가 들어갑니다.
        """
        events = self.events if self.events else []
        safe_paths = [event.get("safe_route") for event in events]
        num_events = len(events)
        for idx in range(num_events):
            fig, ax = plt.subplots(figsize=(8, 6))

            # 평가 텍스트 출력 (예: safe_path_gen fail, Near target)
            ca_gen_fail = False
            if idx < len(events):
                ca_gen_fail = events[idx].get("ca_path_gen_fail")

                if ca_gen_fail == False and fail_only == True:
                    continue

                if events[idx].get("ca_path_gen_fail"):
                    ax.text(0.05, 0.95, "safe_path_gen fail", transform=ax.transAxes,
                            fontsize=12, color='red', verticalalignment='top',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                if events[idx].get("is_near_target"):
                    ax.text(0.95, 0.95, "Near target", transform=ax.transAxes,
                            fontsize=12, color='blue', verticalalignment='top',
                            horizontalalignment='right',
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
            
            # Safe Path 그리기
            route = self.get_safe_path(safe_paths, idx)
            if route:
                lat, lon = self.get_route_coordinates(route)
                ax.plot(lon, lat, marker='o', linestyle=':',
                        color='darkorange', label='Safe Path')
            
            # Base Route 그리기
            if self.base_route:
                lat_base, lon_base = self.get_route_coordinates(self.base_route)
                ax.plot(lon_base, lat_base, marker='o', linestyle='-',
                        color='black', label='Base Route')
            
            # Own Ship 그리기
            if idx < len(events):
                event = events[idx]
                own_ship_event = event.get("own_ship_event")
                if own_ship_event:
                    try:
                        ship_length = self.own_ship_static.get("length", 230)
                        ship_width = self.own_ship_static.get("width", 30)
                        lat_pos = own_ship_event["position"]["latitude"]
                        lon_pos = own_ship_event["position"]["longitude"]
                        heading = own_ship_event.get("heading", 0)
                        self.draw_ship(ax, lon_pos, lat_pos, heading, color='crimson',
                                       ship_length=ship_length, ship_width=ship_width)
                    except Exception as e:
                        print(f"Warning: Own ship position error: {e}")
            
            # Target Ships 그리기
            if idx < len(events):
                targets = events[idx].get("target_ships", [])
                for tidx, target in enumerate(self.flatten(targets)):
                    if not isinstance(target, dict):
                        print(f"Warning: Target이 dict가 아님: {target}")
                        continue
                    try:
                        t_lat = target["position"]["latitude"]
                        t_lon = target["position"]["longitude"]
                    except Exception as e:
                        print(f"Warning: Target position error: {e}, 대상: {target}")
                        continue
                    label = "Target Ship" if tidx == 0 else None
                    ax.scatter(t_lon, t_lat, color='firebrick', marker='x', s=80, label=label)
                    if "cog" in target and "sog" in target:
                        try:
                            cog = float(target["cog"])
                            sog = float(target["sog"])
                            max_sog = 20.0
                            arrow_length = (min(max(sog, 0), max_sog) / max_sog) * 0.05
                            angle_rad = math.radians(cog)
                            dx = arrow_length * math.sin(angle_rad)
                            dy = arrow_length * math.cos(angle_rad)
                            ax.annotate('', xy=(t_lon + dx, t_lat + dy), xytext=(t_lon, t_lat),
                                        arrowprops=dict(arrowstyle='->', linestyle='dashed',
                                                        color='firebrick', linewidth=1))
                        except Exception as e:
                            print(f"Warning: Target cog/sog error: {e}, 대상: {target}")
            
            
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_title(f"Event {idx} - Safe Paths, Target Ships & Own Ship")
            self.set_axis_limits_based_on_base_route(ax)
            ax.legend()
            ax.grid(True)
            fig.savefig(output_path_pattern.format(idx))
            plt.close(fig)
            print(f"플롯이 {output_path_pattern.format(idx)}에 저장되었습니다.")


def main():
    base_data_dir = "data/ver014_20250218_colregs_test"           # 원본 데이터가 있는 최상위 폴더
    base_result_dir = "result/ver014_20250218_colregs_test"  # 결과 플롯을 저장할 폴더

    # FileInputManager를 사용하여 base_data_dir 내의 모든 마집 파일을 재귀적으로 검색
    file_manager = FileInputManager(base_data_dir)
    marzip_files = file_manager.get_all_marzip_files()
    
    if not marzip_files:
        print("마집 파일을 찾지 못했습니다.")
        return

    print("찾은 마집 파일:")
    for f in marzip_files:
        print("  ", f)

    # 각 마집 파일에 대해 EventPlotter를 이용하여 플롯 생성
    for marzip_file in marzip_files:
        # base_data_dir 기준 상대 경로 계산 및 "output" 폴더 이름 제거
        rel_path = os.path.relpath(marzip_file, base_data_dir)
        parts = rel_path.split(os.sep)
        filtered_parts = [part for part in parts if part.lower() != "output"]
        result_rel_path = os.path.join(*filtered_parts)
        result_base = os.path.splitext(result_rel_path)[0]

        # 결과 폴더 생성 (base_result_dir 아래 동일한 폴더 구조)
        output_dir = os.path.join(base_result_dir, os.path.dirname(result_base))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 출력 파일 패턴: 파일명_event{idx}.png
        output_pattern = os.path.join(output_dir, f"{os.path.basename(result_base)}_event{{}}.png")

        # 마집 파일 하나에 대해 EventPlotter 인스턴스 생성 후 이벤트별 플롯 생성
        plotter = EventPlotter(marzip_file)
        plotter.plot_all(output_pattern, fail_only = True)

    print("\nDONE")

if __name__ == "__main__":
    main()