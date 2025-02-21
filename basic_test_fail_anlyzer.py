#!/usr/bin/env python3
# File: random_gen_test_analyzer.py

import os
import csv
import concurrent.futures
import shutil
import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 파일 저장용 백엔드
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager

def batchify(lst, batch_size):
    """리스트를 batch_size 크기의 배치로 분할하여 반환"""
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size]

# -------------------- EventPlotter 클래스 --------------------
class EventPlotter(MarzipExtractor):
    """
    MarzipExtractor를 상속받아 이벤트 데이터에서
    Safe Path, Base Route, Own Ship, Target Ships를 플롯하는 클래스입니다.
    플롯은 기준 좌표(base_route의 첫번째 좌표 또는 own_ship_event의 첫 좌표)를 기준으로
    NM 단위 offset으로 표시되며, X, Y 축은 ±20 NM로 고정됩니다.
    (Own Ship은 빨간 별로 표시합니다.)
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
        latitudes = [point["position"]["latitude"] for point in route]
        longitudes = [point["position"]["longitude"] for point in route]
        return latitudes, longitudes

    def set_axis_limits_ownship(self, ax):
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.set_aspect('equal', adjustable='datalim')

    def draw_ship(self, ax, lon, lat, heading, color, ship_length=230, ship_width=30, scale=10.0, shape='star'):
        # 이 함수는 사용하지 않습니다.
        pass

    def plot_all(self, output_path_pattern, plot_mode="fail", analysis_result=None):
        """
        이벤트 플롯을 생성하여 저장합니다.
        :param output_path_pattern: 저장 경로 패턴 (예: "result/event_{}_plot.png")
        :param plot_mode: "all", "fail", 또는 "na"
                           - "all": 모든 이벤트 플롯 생성.
                           - "fail": CSV 분석 결과의 첫 단어가 "fail"일 때만 플롯 생성.
                           - "na": CSV 분석 결과의 첫 단어가 "n/a"일 때만 플롯 생성.
        :param analysis_result: CSV에서 읽은 분석 결과 문자열
        """
        # 조건 체크: "fail" 모드이면 결과 문자열의 첫 단어가 "fail"이어야 함.
        if plot_mode == "fail":
            if analysis_result is None or analysis_result.strip().split()[0].lower() != "fail":
                print("CSV 결과에 'fail'가 없으므로 플롯 생성 건너뜁니다.")
                return
        elif plot_mode == "na":
            if analysis_result is None or analysis_result.strip().split()[0].lower() != "n/a":
                print("CSV 결과에 'n/a'가 없으므로 플롯 생성 건너뜁니다.")
                return

        events = self.events if self.events else []
        safe_paths = [event.get("safe_route") for event in events]
        num_events = len(events)
        event_indices = range(num_events) if plot_mode == "all" else [num_events - 1] if num_events > 0 else []

        # 기준 좌표: base_route의 첫 번째 좌표, 없으면 own_ship_event 첫 좌표 사용
        if self.base_route and len(self.base_route) > 0:
            ref_lat = self.base_route[0]["position"]["latitude"]
            ref_lon = self.base_route[0]["position"]["longitude"]
        elif num_events > 0 and events[0].get("own_ship_event"):
            own = events[0].get("own_ship_event")
            ref_lat = own["position"]["latitude"]
            ref_lon = own["position"]["longitude"]
        else:
            print("기준 좌표를 찾지 못하여 플롯 생성 불가.")
            return

        cos_factor = math.cos(math.radians(ref_lat))
        
        for idx in event_indices:
            fig, ax = plt.subplots(figsize=(8, 6))
            # Safe Path
            route = self.get_safe_path(safe_paths, idx)
            if route:
                lat_list, lon_list = self.get_route_coordinates(route)
                lat_offsets = [(lat - ref_lat) * 60 for lat in lat_list]
                lon_offsets = [(lon - ref_lon) * 60 * cos_factor for lon in lon_list]
                ax.plot(lon_offsets, lat_offsets, marker='o', linestyle=':',
                        color='darkorange', label='Safe Path')
            # Base Route
            if self.base_route:
                lat_base, lon_base = self.get_route_coordinates(self.base_route)
                lat_base_offsets = [(lat - ref_lat) * 60 for lat in lat_base]
                lon_base_offsets = [(lon - ref_lon) * 60 * cos_factor for lon in lon_base]
                ax.plot(lon_base_offsets, lat_base_offsets, marker='o', linestyle='-',
                        color='black', label='Base Route')
            # Own Ship: 플롯 중심(0,0)로 표시 (빨간 별)
            if idx < num_events and events[idx].get("own_ship_event"):
                own = events[idx].get("own_ship_event")
                own_lat = own["position"]["latitude"]
                own_lon = own["position"]["longitude"]
                own_lat_offset = (own_lat - ref_lat) * 60
                own_lon_offset = (own_lon - ref_lon) * 60 * cos_factor
                ax.scatter(own_lon_offset, own_lat_offset, marker='*', color='crimson', s=200, label="Own Ship")
            # Target Ships
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
                        cog = float(target["cog"])
                        sog = float(target["sog"])
                        max_sog = 40.0
                        arrow_length = (min(max(sog, 0), max_sog) / max_sog) * 10.0
                        angle_rad = math.radians(cog)
                        dx = arrow_length * math.sin(angle_rad)
                        dy = arrow_length * math.cos(angle_rad)
                        ax.annotate("", 
                                    xy=(t_lon_offset + dx, t_lat_offset + dy),
                                    xytext=(t_lon_offset, t_lat_offset),
                                    arrowprops=dict(arrowstyle="->", linestyle='dashed', color='firebrick', linewidth=1))
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
    base_data_dir = "data/ver014_20250220_colregs_test_2"   # 원본 데이터 폴더
    base_result_dir = "result/ver014_20250220_colregs_test_2"  # 결과 플롯 저장 폴더

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
        plot_mode = "fail"  # "fail", "na", 또는 "all" 선택
        plotter.plot_all(output_pattern, plot_mode=plot_mode, analysis_result=analysis_result)
    
    print("DONE")

if __name__ == "__main__":
    main()
