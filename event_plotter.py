#!/usr/bin/env python3
# File: random_gen_test_analyzer.py

import os
import csv
import fnmatch
import shutil
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 파일 저장용 백엔드
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import re
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

def batchify(lst, batch_size):
    """리스트를 batch_size 크기의 배치로 분할하여 반환"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

def interpolate_along_route(offsets_x, offsets_y, desired_dist):
    """
    주어진 경로(offsets_x, offsets_y)의 누적 거리를 계산하고,
    원하는 거리(desired_dist, NM)에 해당하는 위치를 선형 보간하여 반환합니다.
    """
    if len(offsets_x) < 2:
        return offsets_x[0], offsets_y[0]
    cumdist = [0]
    for i in range(1, len(offsets_x)):
        d = math.sqrt((offsets_x[i]-offsets_x[i-1])**2 + (offsets_y[i]-offsets_y[i-1])**2)
        cumdist.append(cumdist[-1] + d)
    if desired_dist >= cumdist[-1]:
        return offsets_x[-1], offsets_y[-1]
    for i in range(1, len(cumdist)):
        if cumdist[i] >= desired_dist:
            ratio = (desired_dist - cumdist[i-1]) / (cumdist[i] - cumdist[i-1])
            interp_x = offsets_x[i-1] + ratio * (offsets_x[i] - offsets_x[i-1])
            interp_y = offsets_y[i-1] + ratio * (offsets_y[i] - offsets_y[i-1])
            return interp_x, interp_y
    return offsets_x[-1], offsets_y[-1]

def project_point_onto_polyline(px, py, xs, ys):
    """
    주어진 점 (px, py)를 polyline (xs, ys) 상에 투영하여,
    투영된 점까지의 누적 거리를 반환합니다.
    """
    best_dist_along = None
    best_distance = float('inf')
    cumdist = [0]
    for i in range(1, len(xs)):
        seg_dx = xs[i] - xs[i-1]
        seg_dy = ys[i] - ys[i-1]
        seg_length = math.sqrt(seg_dx**2 + seg_dy**2)
        cumdist.append(cumdist[-1] + seg_length)
        if seg_length == 0:
            continue
        t = ((px - xs[i-1]) * seg_dx + (py - ys[i-1]) * seg_dy) / (seg_length**2)
        t = max(0, min(1, t))
        proj_x = xs[i-1] + t * seg_dx
        proj_y = ys[i-1] + t * seg_dy
        dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        if dist < best_distance:
            best_distance = dist
            best_dist_along = cumdist[i-1] + t * seg_length
    return best_dist_along if best_dist_along is not None else 0

# -------------------- EventPlotter 클래스 --------------------
class EventPlotter(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터에서
    Safe Path, Base Route, Own Ship, Target Ships를 플롯하는 클래스입니다.
    플롯은 기준 좌표(base_route의 첫번째 좌표 또는 own_ship_event의 첫 좌표)를 기준으로
    NM 단위 offset으로 표시되며, X, Y 축은 ±20 NM로 고정됩니다.
    Own Ship은 빨간 별로 표시되고, Target Ship은 'x' 마커로 표시되며,
    target ship은 0분부터 30분까지 예상 위치(화살표와 tick 선)를 보입니다.
    자선(own ship 예상 경로)는 safe path(이벤트 루트)를 사용하여,
    own_ship_event의 현재 위치를 safe path 상에 투영한 후 그 지점부터 선형 보간으로 계산합니다.
    """
    def __init__(self, marzip_file):
        super().__init__()
        self.marzip = marzip_file
        self.run(marzip_file)

    def get_safe_path(self, safe_paths, idx):
        while idx >= 0:
            if idx < len(safe_paths) and safe_paths[idx]:
                return safe_paths[idx]
            idx -= 1
        return None

    def get_route_coordinates(self, route):
        latitudes = [pt["position"]["latitude"] for pt in route]
        longitudes = [pt["position"]["longitude"] for pt in route]
        return latitudes, longitudes

    def set_axis_limits_ownship(self, ax):
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal', adjustable='datalim')

    def draw_ship(self, ax, lon, lat, heading, color, ship_length=230, ship_width=30, scale=10.0, shape='star'):
        # 사용하지 않습니다.
        pass

    def simulate_own_ship_path(self, own_ship_speed, event_index, safe_path, base_path,time_points=[10, 20, 30]):
        """
        자선(Own Ship)의 미래 위치를 safe_path를 따라 선형보간 방식으로 계산합니다.
        모든 좌표는 base_path(또는 base_route) 첫 점(base_ref_lat, base_ref_lon)을 기준으로
        NM 단위로 변환된 후 처리됩니다.

        :param own_ship_speed: 자선 속도 (NM/h) - 주로 own_ship_event["sog"]를 사용
        :param event_index: 현재 이벤트 인덱스
        :param safe_path: 사용 중인 safe_path (각 점에 lat/lon이 있는 리스트)
        :param base_ref_lat: base_path(또는 base_route) 첫 점의 위도 (좌표계 기준)
        :param base_ref_lon: base_path(또는 base_route) 첫 점의 경도 (좌표계 기준)
        :param time_points: 예측할 시간(분) 리스트
        :return: [(pred_x, pred_y), ...] 형식의 NM 오프셋 좌표. base_ref_lat/lon 기준.
        """


        base_ref_lat = self.base_route[0]["position"]["latitude"]
        base_ref_lon = self.base_route[0]["position"]["longitude"]        


        if not self.events or event_index >= len(self.events):
            return [(0,0) for _ in time_points]
        own_event = self.events[event_index].get("own_ship_event")
        if not own_event:
            return [(0,0) for _ in time_points]
        
        # 자선 현재 위치 -> NM 변환
        curr_lat = own_event["position"]["latitude"]
        curr_lon = own_event["position"]["longitude"]
        cos_factor = math.cos(math.radians(base_ref_lat))
        own_x = (curr_lon - base_ref_lon) * 60.0 * cos_factor
        own_y = (curr_lat - base_ref_lat) * 60.0

        # safe_path를 (x, y) (NM) 리스트로 변환
        route_lats_s, route_lons_s = self.get_route_coordinates(safe_path)  # lat[], lon[]
        safe_x = []
        safe_y = []
        for lat_s, lon_s in zip(route_lats_s, route_lons_s):
            sx = (lon_s - base_ref_lon) * 60.0 * cos_factor
            sy = (lat_s - base_ref_lat) * 60.0
            safe_x.append(sx)
            safe_y.append(sy)
        
        # base_path를 (x, y) (NM) 리스트로 변환
        route_lats_b, route_lons_b = self.get_route_coordinates(base_path)
        base_x = []
        base_y = []
        for lat_b, lon_b in zip(route_lats_b, route_lons_b):
            bx = (lon_b - base_ref_lon) * 60.0 * cos_factor
            by = (lat_b - base_ref_lat) * 60.0
            base_x.append(bx)
            base_y.append(by)
        
        # safe_path 총 길이(safe_total_dist) 계산
        safe_total_dist = 0.0
        for i in range(len(safe_x) - 1):
            seg_dx = safe_x[i+1] - safe_x[i]
            seg_dy = safe_y[i+1] - safe_y[i]
            safe_total_dist += math.sqrt(seg_dx**2 + seg_dy**2)
        
        # base_path(또는 safe_path 끝 + base_path) 쪽 누적 길이도 대비하여 계산
        # 필요하다면 미리 base_total_dist를 구해둘 수도 있음
        # 여기선 일단 base_path의 누적 거리를 사용하려면, project & interpolate를 활용
        
        # 현재 자선 위치를 safe_path에 투영
        proj_dist = project_point_onto_polyline(own_x, own_y, safe_x, safe_y)
        
        predicted_positions = []
        for t in time_points:
            travel_dist = own_ship_speed * (t / 60.0)  # 이동 거리 (NM)
            desired_dist = proj_dist + travel_dist
            
            if desired_dist <= safe_total_dist:
                # safe_path 내에서 선형보간
                px, py = interpolate_along_route(safe_x, safe_y, desired_dist)
                predicted_positions.append((px, py))
            else:
                # safe_path 끝까지 이동
                remainder = desired_dist - safe_total_dist
                
                # safe_path 끝 좌표
                safe_end_x = safe_x[-1]
                safe_end_y = safe_y[-1]
                
                # base_path 상의 시작점 찾기
                # (1) safe_path 끝과 base_path가 이어진다고 가정 -> base_path[0] == safe_path[-1] 라면
                #     remainder만큼 base_path를 보간하면 됨
                #
                # (2) 만약 정확히 이어지지 않는 경우, safe_path 끝점을 base_path 폴리라인에 투영
                #     (아래처럼 project_point_onto_polyline() 사용)
                base_start_dist = project_point_onto_polyline(safe_end_x, safe_end_y, base_x, base_y)
                
                # base_path에서 이동해야 할 누적거리
                base_desired_dist = base_start_dist + remainder
                
                px, py = interpolate_along_route(base_x, base_y, base_desired_dist)
                predicted_positions.append((px, py))
        
        return predicted_positions


    def plot_all(self, output_path_pattern, plot_mode="fail", analysis_result=None):
        """
        이벤트 플롯을 생성하여 저장합니다.
        :param output_path_pattern: 저장 경로 패턴 (예: "result/event_{}_plot.png"),
                                    {}에 이벤트 인덱스가 들어갑니다.
        :param plot_mode: "all", "fail", 또는 "na"
                           "all": 모든 이벤트 플롯 생성.
                           "fail": CSV 분석 결과의 첫 단어가 "fail"일 때만 플롯 생성.
                           "na": CSV 분석 결과의 첫 단어가 "n/a"일 때만 플롯 생성.
        :param analysis_result: CSV에서 읽은 분석 결과 문자열 (예: "fail (closed target)" 등)
        """
        if plot_mode == "fail":
            if analysis_result is None or not re.search(r'\bfail\b', analysis_result, flags=re.IGNORECASE):
                print("CSV 결과에 'fail' 단어가 없으므로 플롯 생성 건너뜁니다.")
                plt.close('all')
                return
        elif plot_mode == "na":
            if analysis_result is None or not re.search(r'\bn/a\b', analysis_result, flags=re.IGNORECASE):
                print("CSV 결과에 'n/a' 단어가 없으므로 플롯 생성 건너뜁니다.")
                plt.close('all')
                return

        events = self.events if self.events else []
        safe_paths = [event.get("safe_route") for event in events]
        num_events = len(events)
        event_indices = range(num_events) if plot_mode == "all" else [num_events - 1] if num_events > 0 else []

        # 기준 좌표: base_route의 첫 번째 좌표, 없으면 own_ship_event 사용
        if self.base_route and len(self.base_route) > 0:
            ref_lat = self.base_route[0]["position"]["latitude"]
            ref_lon = self.base_route[0]["position"]["longitude"]
            lat_base, lon_base = self.get_route_coordinates(self.base_route)
            base_lat_offsets = [(lat - ref_lat) * 60 for lat in lat_base]
            base_lon_offsets = [(lon - ref_lon) * 60 * math.cos(math.radians(ref_lat)) for lon in lon_base]
        else:
            print("기준 좌표를 찾지 못하여 플롯 생성 불가.")
            return

        cos_factor = math.cos(math.radians(ref_lat))
        
        for idx in event_indices:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Safe Path 그리기 (NM 변환)
            route = self.get_safe_path(safe_paths, idx)
            if route:
                lat_list, lon_list = self.get_route_coordinates(route)
                lat_offsets = [(lat - ref_lat) * 60 for lat in lat_list]
                lon_offsets = [(lon - ref_lon) * 60 * cos_factor for lon in lon_list]
                ax.plot(lon_offsets, lat_offsets, marker='o', linestyle=':',
                        color='darkorange', label='Safe Path')
            # Base Route 그리기 (NM 변환)
            if self.base_route:
                ax.plot(base_lon_offsets, base_lat_offsets, marker='o', linestyle='-',
                        color='black', label='Base Route')
            # Own Ship 그리기 (빨간 별)
            if idx < num_events and events[idx].get("own_ship_event"):
                own = events[idx].get("own_ship_event")
                own_lat = own["position"]["latitude"]
                own_lon = own["position"]["longitude"]
                own_lat_offset = (own_lat - ref_lat) * 60
                own_lon_offset = (own_lon - ref_lon) * 60 * cos_factor
                ax.scatter(own_lon_offset, own_lat_offset, marker='*', color='crimson', s=200, label="Own Ship")
                # 자선(own ship 예상 경로) 플롯: own_ship의 속도는 own["sog"]를 사용
                own_speed = float(own["sog"])

                predicted_own_positions = self.simulate_own_ship_path(own_speed, idx, route, self.base_route, time_points=[10,20,30])
                if predicted_own_positions:
                    xs_own, ys_own = zip(*predicted_own_positions)
                    ax.plot(xs_own, ys_own, linestyle='dashed', color='green', linewidth=1, label="Predicted Own Path")
                    for pos in predicted_own_positions:
                        ax.scatter(pos[0], pos[1], marker='o', color='green', s=40)
                        
            # Target Ships 및 tcpa 예측 (NM 변환)
            targets = events[idx].get("target_ships", [])
            for tidx, target in enumerate(self.flatten(targets)):
                if not isinstance(target, dict):
                    continue
                try:
                    t_lat = target["position"]["latitude"]
                    t_lon = target["position"]["longitude"]
                    t_lat_offset = (t_lat - ref_lat) * 60
                    t_lon_offset = (t_lon - ref_lon) * 60 * cos_factor
                except Exception:
                    continue
                label = "Target Ship" if tidx == 0 else None
                ax.scatter(t_lon_offset, t_lat_offset, color='firebrick', marker='x', s=80, label=label)
                if "cog" in target and "sog" in target:
                    try:
                        sog = float(target["sog"])  # NM/h
                        if sog < 0.1:
                            continue
                        cog = float(target["cog"])
                        angle_rad = math.radians(cog)
                        # 예측 시간: 0, 10, 20, 30분 (0분은 현재 위치)
                        time_points = [0, 10, 20, 30]
                        predicted_positions = []
                        for t in time_points:
                            travel_dist = sog * (t / 60.0)
                            dx = travel_dist * math.sin(angle_rad)
                            dy = travel_dist * math.cos(angle_rad)
                            predicted_positions.append((t_lon_offset + dx, t_lat_offset + dy))
                        # 예측 경로를 dashed 선으로 그리기
                        xs, ys = zip(*predicted_positions)
                        ax.plot(xs, ys, linestyle='dashed', color='firebrick', linewidth=1)
                        # 최종 위치(30분 후)까지 화살표 그리기
                        ann = ax.annotate(
                            "",
                            xy=predicted_positions[-1],
                            xytext=predicted_positions[0],
                            arrowprops=dict(arrowstyle="->", linestyle='dashed',
                                            color='firebrick', linewidth=1, mutation_scale=15, clip_on=False)
                        )
                        ann.set_clip_on(False)
                        # 10분, 20분에 해당하는 예측 위치에서 tick 선 그리기 (0.5NM 길이)
                        tick_length = 0.5  # NM
                        perp_angle = angle_rad + math.pi/2  # 수직 방향 (필요 시 -math.pi/2로 조정)
                        for pos in predicted_positions[1:-1]:
                            tick_dx = (tick_length/2) * math.sin(perp_angle)
                            tick_dy = (tick_length/2) * math.cos(perp_angle)
                            ax.plot([pos[0] - tick_dx, pos[0] + tick_dx],
                                    [pos[1] - tick_dy, pos[1] + tick_dy],
                                    color='blue', linewidth=1)
                    except Exception:
                        continue   
            self.set_axis_limits_ownship(ax)
            ax.set_xlabel("Longitude Offset (NM)")
            ax.set_ylabel("Latitude Offset (NM)")
            ax.set_title(f"Event {idx} - Relative to Base Reference")
            ax.legend()
            ax.grid(True)
            fig.savefig(output_path_pattern.format(idx))
            plt.close(fig)
            print(f"플롯 저장: {output_path_pattern.format(idx)}")

# -------------------- CSV 분석 결과 읽기 --------------------
def read_analysis_csv(csv_path):
    analysis_dict = {}
    if not os.path.exists(csv_path):
        print(f"CSV 파일 없음: {csv_path}")
        return analysis_dict
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            folder = row.get("Folder", "").strip()
            file_base = row.get("File", "").strip()
            if not folder or not file_base:
                continue
            key = f"{folder}_{file_base}"
            analysis_dict[key] = row.get("Result", "").strip()
    return analysis_dict

# -------------------- Main --------------------
def main():
    base_data_dir = "data/ver014_20250220_colregs_test_2"           # 원본 데이터 폴더
    base_result_dir = "result/ver014_20250220_colregs_test_2"          # 결과 플롯 저장 폴더

    analysis_csv = os.path.join(base_data_dir, "all_event_analysis.csv")
    analysis_results = read_analysis_csv(analysis_csv)
    print(f"CSV 분석 결과 {len(analysis_results)}개 항목 읽음.")
    
    file_manager = FileInputManager(base_data_dir)
    marzip_files = file_manager.get_all_marzip_files()
    if not marzip_files:
        print("마집 파일 없음.")
        return
    print(f"총 {len(marzip_files)}개 마집 파일 발견.")
    
    for marzip_file in marzip_files:
        rel_path = os.path.relpath(marzip_file, base_data_dir)
        parts = rel_path.split(os.sep)
        filtered_parts = [p for p in parts if p.lower() != "output"]
        result_rel_path = os.path.join(*filtered_parts)
        result_base = os.path.splitext(result_rel_path)[0]
        output_dir = os.path.join(base_result_dir, os.path.dirname(result_base))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_pattern = os.path.join(output_dir, f"{os.path.basename(result_base)}_event{{}}.png")
        plotter = EventPlotter(marzip_file)
        folder_key = os.path.basename(os.path.dirname(os.path.dirname(marzip_file)))
        file_key = os.path.splitext(os.path.basename(marzip_file))[0]
        key = f"{folder_key}_{file_key}"
        analysis_result = analysis_results.get(key)
        plot_mode = "fail"  # "fail", "na", 또는 "all"
        plotter.plot_all(output_pattern, plot_mode=plot_mode, analysis_result=analysis_result)
    
    print("DONE")

if __name__ == "__main__":
    main()
