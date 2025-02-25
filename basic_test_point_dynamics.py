#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import math
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
import json

from marzip_extractor import MarzipExtractor
from file_input_manager import FileInputManager
import faulthandler
faulthandler.enable()

# 플롯 옵션: "collision", "na_collision", "none", "ALL"
PLOT_OPTION = "collision"

################################
# CSV 분석 결과 (옵션)
################################
def read_analysis_csv(csv_path):
    analysis_dict = {}
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV 파일 없음: {csv_path}")
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

################################
# 시뮬레이션 결과 구조체
################################
class SimulationResult:
    def __init__(self):
        self.event_index = None
        self.has_path = True
        self.is_fail = False
        self.fail_time_sec = None
        self.min_distance = float('inf')
        self.min_distance_time = 0
        self.times = []
        self.own_positions = []
        self.targets_positions = []
        # 최종 분류:
        # "NA - Safe Target", "NA - No Path", "NA Collision",
        # "Collision", "No Collision"
        self.result_tag = None

################################
# 보조 유틸 (보간/투영)
################################
def interpolate_along_route(offsets_x, offsets_y, desired_dist):
    if len(offsets_x) < 2:
        return offsets_x[0], offsets_y[0]
    cumdist = [0.0]
    for i in range(1, len(offsets_x)):
        dx = offsets_x[i] - offsets_x[i-1]
        dy = offsets_y[i] - offsets_y[i-1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        cumdist.append(cumdist[-1] + seg_len)
    if desired_dist >= cumdist[-1]:
        return offsets_x[-1], offsets_y[-1]
    for i in range(1, len(cumdist)):
        if cumdist[i] >= desired_dist:
            ratio = (desired_dist - cumdist[i-1]) / (cumdist[i] - cumdist[i-1])
            ix = offsets_x[i-1] + ratio*(offsets_x[i] - offsets_x[i-1])
            iy = offsets_y[i-1] + ratio*(offsets_y[i] - offsets_y[i-1])
            return ix, iy
    return offsets_x[-1], offsets_y[-1]

def project_point_onto_polyline(px, py, xs, ys):
    best_dist_along = None
    best_distance = float('inf')
    cumdist = [0]
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i-1]
        dy = ys[i] - ys[i-1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        cumdist.append(cumdist[-1] + seg_len)
        if seg_len == 0:
            continue
        t = ((px - xs[i-1]) * dx + (py - ys[i-1]) * dy) / (seg_len**2)
        t = max(0, min(1, t))
        proj_x = xs[i-1] + t*dx
        proj_y = ys[i-1] + t*dy
        dist = math.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        if dist < best_distance:
            best_distance = dist
            best_dist_along = cumdist[i-1] + t*seg_len
    return best_dist_along if best_dist_along else 0.0

################################
# 시뮬 + 플롯 클래스
################################
class SimpulatePlotter(MarzipExtractor):
    COLLISION_DIST = 0.5 + (250 / 1852)
    SIM_DURATION_SEC = 5 * 60 * 60  # 실제 3시간 = 10800초
    TIME_STEP_SEC = 5            # 5초 간격

    def __init__(self, marzip_file):
        super().__init__()
        self.run(marzip_file)

    def flatten(self, arr):
        if not arr:
            return []
        if isinstance(arr[0], list):
            merged = []
            for sub in arr:
                merged.extend(sub)
            return merged
        return arr

    def simulate_event_with_safe_route(self, event_index, safe_route):
        res = SimulationResult()
        res.event_index = event_index

        if not self.events or event_index < 0 or event_index >= len(self.events):
            res.has_path = False
            return res

        e = self.events[event_index]
        own_ev = e.get("own_ship_event")
        if not own_ev:
            res.has_path = False
            return res

        try:
            own_sog = float(own_ev["sog"])
        except Exception as e:
            print(f"own_sog 변환 실패: {e}")
            own_sog = 0

        start_lat = own_ev["position"]["latitude"]
        start_lon = own_ev["position"]["longitude"]

        if self.base_route and len(self.base_route) > 0:
            ref_lat = self.base_route[0]["position"]["latitude"]
            ref_lon = self.base_route[0]["position"]["longitude"]
        else:
            ref_lat, ref_lon = start_lat, start_lon

        cos_factor = math.cos(math.radians(ref_lat))
        start_x = (start_lon - ref_lon) * 60 * cos_factor
        start_y = (start_lat - ref_lat) * 60

        safe_x, safe_y = [], []
        for pt in safe_route:
            la = pt["position"]["latitude"]
            lo = pt["position"]["longitude"]
            safe_x.append((lo - ref_lon) * 60 * cos_factor)
            safe_y.append((la - ref_lat) * 60)

        base_x, base_y = [], []
        for pt in self.base_route:
            la = pt["position"]["latitude"]
            lo = pt["position"]["longitude"]
            base_x.append((lo - ref_lon) * 60 * cos_factor)
            base_y.append((la - ref_lat) * 60)

        total_safe_dist = sum(math.sqrt((safe_x[i+1]-safe_x[i])**2 + (safe_y[i+1]-safe_y[i])**2) 
                              for i in range(len(safe_x)-1))
        base_total_dist = 0.0
        if len(base_x) > 1:
            base_total_dist = sum(math.sqrt((base_x[i+1]-base_x[i])**2 + (base_y[i+1]-base_y[i])**2)
                                  for i in range(len(base_x)-1))

        proj_dist = 0.0
        if len(safe_x) > 1:
            proj_dist = project_point_onto_polyline(start_x, start_y, safe_x, safe_y)

        flat_targets = self.flatten(e.get("target_ships", []))
        tgt_info = []
        for t in flat_targets:
            if not t.get("position"):
                continue
            la = t["position"]["latitude"]
            lo = t["position"]["longitude"]
            try:
                sog = float(t.get("sog", 0.0))
                cog = float(t.get("cog", 0.0))
            except Exception as e:
                sog, cog = 0, 0
            tx0 = (lo - ref_lon) * 60 * cos_factor
            ty0 = (la - ref_lat) * 60
            arad = math.radians(cog)
            tgt_info.append((tx0, ty0, sog, arad))
        res.targets_positions = [[] for _ in range(len(tgt_info))]

        safe_path_reached_time = None

        for sec in range(0, self.SIM_DURATION_SEC + 1, self.TIME_STEP_SEC):
            res.times.append(sec)
            traveled = own_sog * (sec / 3600.0)
            desired_dist = proj_dist + traveled

            if desired_dist <= total_safe_dist:
                ox, oy = interpolate_along_route(safe_x, safe_y, desired_dist)
            else:
                sx_end, sy_end = safe_x[-1], safe_y[-1]
                if safe_path_reached_time is None:
                    safe_path_reached_time = sec

                if sec - safe_path_reached_time >= self.hinas_setup.get("TCPA_GW", 0) * 60:
                    ox, oy = sx_end, sy_end
                    res.own_positions.append((ox, oy))
                    res.times.append(sec)
                    break

                remain = desired_dist - total_safe_dist
                if len(base_x) > 1:
                    base_st = project_point_onto_polyline(sx_end, sy_end, base_x, base_y)
                    base_des = base_st + remain
                    if base_des >= base_total_dist:
                        ox, oy = base_x[-1], base_y[-1]
                        res.own_positions.append((ox, oy))
                        res.times.append(sec)
                        break
                    else:
                        ox, oy = interpolate_along_route(base_x, base_y, base_des)
                else:
                    ox, oy = sx_end, sy_end

            res.own_positions.append((ox, oy))

            for idx_t, (tx0, ty0, sog_t, arad) in enumerate(tgt_info):
                dt = sog_t * (sec / 3600.0)
                tx_ = tx0 + dt * math.sin(arad)
                ty_ = ty0 + dt * math.cos(arad)
                res.targets_positions[idx_t].append((tx_, ty_))
                dx_ = ox - tx_
                dy_ = oy - ty_
                dd_ = math.sqrt(dx_**2 + dy_**2)
                if dd_ < res.min_distance:
                    res.min_distance = dd_
                    res.min_distance_time = sec
                if dd_ < self.COLLISION_DIST and not res.is_fail:
                    res.is_fail = True
                    res.fail_time_sec = sec

        res.result_tag = "Collision" if res.is_fail else "No Collision"
        return res

    def simulate_event(self, event_index):
        res = SimulationResult()
        res.event_index = event_index

        if not self.events or event_index < 0 or event_index >= len(self.events):
            res.has_path = False
            res.result_tag = "NA - No Path"
            return res

        e = self.events[event_index]
        own_ev = e.get("own_ship_event")
        if not own_ev:
            res.has_path = False
            res.result_tag = "NA - No Path"
            return res

        safe_r = e.get("safe_route")
        if not safe_r or len(safe_r) == 0:
            if event_index == 0:
                res.has_path = False
                res.result_tag = "NA - No Path"
                return res
            else:
                prev_sr = self.events[event_index-1].get("safe_route", [])
                if not prev_sr:
                    res.has_path = False
                    res.result_tag = "NA - No Path"
                    return res
                sub_res = self.simulate_event_with_safe_route(event_index, prev_sr)
                sub_res.result_tag = "NA Collision" if sub_res.is_fail else "NA - No Path"
                return sub_res
        return self.simulate_event_with_safe_route(event_index, safe_r)

    def set_axis_limits_ownship(self, ax):
        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 40)
        ax.set_aspect('equal')

    def plot_collision_event(self, sim_res, out_file):
        PLOT_SKIP = 10
        idx = sim_res.event_index
        fig, ax = plt.subplots(figsize=(8, 6))

        if self.base_route and len(self.base_route) > 0:
            ref_lat = self.base_route[0]["position"]["latitude"]
            ref_lon = self.base_route[0]["position"]["longitude"]
        else:
            ref_lat = self.events[0]["own_ship_event"]["position"]["latitude"]
            ref_lon = self.events[0]["own_ship_event"]["position"]["longitude"]
        cos_factor = math.cos(math.radians(ref_lat))

        bx, by = [], []
        for pt in self.base_route:
            la = pt["position"]["latitude"]
            lo = pt["position"]["longitude"]
            bx.append((lo - ref_lon) * 60 * cos_factor)
            by.append((la - ref_lat) * 60)
        if bx and by:
            ax.plot(bx, by, 'ko-', label='Base Route')

        e = self.events[idx]
        if (sim_res.result_tag in ["NA - No Path", "NA Collision"]) and idx > 0:
            e = self.events[idx-1]
        sr = e.get("safe_route", [])
        if sr:
            sx = [ (pt["position"]["longitude"] - ref_lon)*60*cos_factor for pt in sr ]
            sy = [ (pt["position"]["latitude"] - ref_lat)*60 for pt in sr ]
            ax.plot(sx, sy, 'o--', color='darkorange', label='Safe Route')

        own_pos = sim_res.own_positions
        n_own = len(own_pos)
        for chunk_i in range(0, n_own - 1, PLOT_SKIP):
            nxt_i = min(chunk_i + PLOT_SKIP, n_own - 1)
            x0, y0 = own_pos[chunk_i]
            x1, y1 = own_pos[nxt_i]
            frac = chunk_i / (n_own - 1) if n_own > 1 else 0
            alpha_val = 0.2 + 0.8 * frac
            ax.plot([x0, x1], [y0, y1], color='red', alpha=alpha_val, lw=2)
        if n_own > 0:
            fx, fy = own_pos[-1]
            ax.scatter(fx, fy, marker='*', color='crimson', s=20, label='Own Ship')

        for t_i, tpos in enumerate(sim_res.targets_positions):
            m_ = len(tpos)
            for chunk_j in range(0, m_ - 1, PLOT_SKIP):
                nxt_j = min(chunk_j + PLOT_SKIP, m_ - 1)
                tx0, ty0 = tpos[chunk_j]
                tx1, ty1 = tpos[nxt_j]
                frac_t = chunk_j / (m_ - 1) if m_ > 1 else 0
                alpha_t = 0.2 + 0.8 * frac_t
                ax.plot([tx0, tx1], [ty0, ty1], color='blue', alpha=alpha_t, lw=1.5)
            if m_ > 0:
                ftx, fty = tpos[-1]
                lbl_ = "Target Ship" if t_i == 0 else None
                ax.scatter(ftx, fty, marker='x', color='blue', s=15, label=lbl_)

        if sim_res.is_fail and sim_res.fail_time_sec is not None:
            if sim_res.fail_time_sec in sim_res.times:
                f_idx = sim_res.times.index(sim_res.fail_time_sec)
                cx, cy = own_pos[f_idx]
                ax.scatter(cx, cy, edgecolor='yellow', facecolor='red', s=20, marker='o', lw=2, label='Collision Point', zorder=5)

        tag_txt = sim_res.result_tag or "Collision"
        info_list = [
            f"Event {idx} - {tag_txt}",
            f"Fail={sim_res.is_fail}, fail_time={sim_res.fail_time_sec}",
            f"MinDist={sim_res.min_distance:.3f}NM @t={sim_res.min_distance_time}"
        ]
        ax.set_title(f"Event {idx} - {tag_txt}")
        ax.set_xlabel("Longitude Offset (NM)")
        ax.set_ylabel("Latitude Offset (NM)")
        self.set_axis_limits_ownship(ax)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        fig.tight_layout()
        ax.text(0.02, 0.98, "\n".join(info_list), transform=ax.transAxes, va='top', ha='left', fontsize=9,
                bbox=dict(facecolor='white', alpha=0.7))
        fig.savefig(out_file)
        plt.close(fig)

def load_partial_summary(summary_path):
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                print(f"부분 요약 파일 로드 실패: {e}")
    # 파일이 없으면 기본값 반환
    return {
        "grand_file_count": 0,
        "grand_na_safe_target": 0,
        "grand_na_no_path": 0,
        "grand_na_collision": 0,
        "grand_collision": 0,
        "grand_no_collision": 0,
        "total_fail_files": 0,
        "total_na_files": 0,
        "total_success_files": 0
    }

def save_partial_summary(summary_path, summary_dict):
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, ensure_ascii=False, indent=4)
###############################################
# 메인
###############################################
def main():
    base_data_dir = "/media/avikus/One Touch/HinasControlSilsCA/CA_v0.1.4_data/Random"
    base_result_dir = "analyze/CA_v0.1.4_data"
    checkpoint_path = os.path.join(base_result_dir, "checkpoint.txt")
    summary_path = os.path.join(base_result_dir, "partial_summary.json")
    
    # 이전에 처리한 파일 목록과 누적 통계를 불러옴
    processed_files_set = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as cp:
            for line in cp:
                processed_files_set.add(line.strip())
                
    summary_dict = load_partial_summary(summary_path)
    
    grand_file_count = summary_dict.get("grand_file_count", 0)
    grand_na_safe_target = summary_dict.get("grand_na_safe_target", 0)
    grand_na_no_path = summary_dict.get("grand_na_no_path", 0)
    grand_na_collision = summary_dict.get("grand_na_collision", 0)
    grand_collision = summary_dict.get("grand_collision", 0)
    grand_no_collision = summary_dict.get("grand_no_collision", 0)
    total_fail_files = summary_dict.get("total_fail_files", 0)
    total_na_files = summary_dict.get("total_na_files", 0)
    total_success_files = summary_dict.get("total_success_files", 0)
    
    file_mgr = FileInputManager(base_data_dir)
    marzip_files = file_mgr.get_all_marzip_files()
    if not marzip_files:
        print("마집 파일 없음.")
        return

    analysis_log_path = os.path.join(base_result_dir, "analysis_log.txt")
    log_dir = os.path.dirname(analysis_log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(analysis_log_path, "a", encoding="utf-8") as lf:
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lf.write(f"\n--- Analysis start: {now_str} ---\n")
    
    processed_files = 0
    last_processed_file = ""
    
    for marzip_file in marzip_files:
        if marzip_file in processed_files_set:
            continue

        try:
            processed_files += 1
            grand_file_count += 1
            last_processed_file = marzip_file

            plotter = SimpulatePlotter(marzip_file)
            events_count = len(plotter.events)
            
            # 파일 단위 로컬 통계
            local_na_safe = 0
            local_na_no_path = 0
            local_na_coll = 0
            local_coll = 0
            local_no_coll = 0
            local_detection_count = 0

            if events_count == 0:
                grand_na_safe_target += 1
                total_na_files += 1
                msg = f"[FOLDER_LOG] {marzip_file} => 0 events => Result=NA (NA-SafeTarget), detectionTargets=0\n"
                print(msg, end="")
                with open(analysis_log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
                with open(checkpoint_path, "a", encoding="utf-8") as cp:
                    cp.write(marzip_file + "\n")
                continue
            else:
                for e_idx in range(events_count):
                    e_data = plotter.events[e_idx]
                    tg_raw = e_data.get("target_ships", [])
                    tg_flat = plotter.flatten(tg_raw)
                    local_detection_count += len(tg_flat)
                    
                    sim_res = plotter.simulate_event(e_idx)
                    tag = sim_res.result_tag

                    if tag == "NA - Safe Target":
                        local_na_safe += 1
                    elif tag == "NA - No Path":
                        local_na_no_path += 1
                    elif tag == "NA Collision":
                        local_na_coll += 1
                        if PLOT_OPTION in ("na_collision", "ALL"):
                            rel_path = os.path.relpath(marzip_file, base_data_dir)
                            result_subdir = os.path.join(base_result_dir, os.path.dirname(rel_path))
                            if not os.path.exists(result_subdir):
                                os.makedirs(result_subdir)
                            out_file = os.path.join(result_subdir, f"{os.path.splitext(os.path.basename(marzip_file))[0]}_ev{e_idx}_NAcoll.png")
                            plotter.plot_collision_event(sim_res, out_file)
                    elif tag == "Collision":
                        local_coll += 1
                        if PLOT_OPTION in ("collision", "ALL"):
                            rel_path = os.path.relpath(marzip_file, base_data_dir)
                            result_subdir = os.path.join(base_result_dir, os.path.dirname(rel_path))
                            if not os.path.exists(result_subdir):
                                os.makedirs(result_subdir)
                            out_file = os.path.join(result_subdir, f"{os.path.splitext(os.path.basename(marzip_file))[0]}_ev{e_idx}_Collision.png")
                            plotter.plot_collision_event(sim_res, out_file)
                    elif tag == "No Collision":
                        if PLOT_OPTION in ("ALL"):
                            rel_path = os.path.relpath(marzip_file, base_data_dir)
                            result_subdir = os.path.join(base_result_dir, os.path.dirname(rel_path))
                            if not os.path.exists(result_subdir):
                                os.makedirs(result_subdir)
                            out_file = os.path.join(result_subdir, f"{os.path.splitext(os.path.basename(marzip_file))[0]}_ev{e_idx}_Pass.png")
                            plotter.plot_collision_event(sim_res, out_file)
                        local_no_coll += 1

                msg = (
                    f"[FOLDER_LOG] {marzip_file} => events={events_count}, "
                    f"NA-SafeTarget={local_na_safe}, NA-NoPath={local_na_no_path}, "
                    f"NA-Collision={local_na_coll}, Collision={local_coll}, "
                    f"NoCollision={local_no_coll}, detectionTargets={local_detection_count} => "
                )
                if local_coll > 0:
                    file_result = "FAIL"
                    total_fail_files += 1
                elif (local_na_safe > 0 or local_na_no_path > 0 or local_na_coll > 0):
                    na_details = []
                    if local_na_safe > 0:
                        na_details.append(f"NA-SafeTarget={local_na_safe}")
                    if local_na_no_path > 0:
                        na_details.append(f"NA-NoPath={local_na_no_path}")
                    if local_na_coll > 0:
                        na_details.append(f"NA-Collision={local_na_coll}")
                    file_result = f"NA ({', '.join(na_details)})"
                    total_na_files += 1
                else:
                    file_result = "SUCCESS"
                    total_success_files += 1
                msg += f"Result={file_result}\n"
                print(msg, end="")
                with open(analysis_log_path, "a", encoding="utf-8") as f:
                    f.write(msg)

                grand_na_safe_target += local_na_safe
                grand_na_no_path += local_na_no_path
                grand_na_collision += local_na_coll
                grand_collision += local_coll
                grand_no_collision += local_no_coll

            # 각 파일 처리 후 summary_dict를 업데이트하여 중간 저장
            summary_dict.update({
                "grand_file_count": grand_file_count,
                "grand_na_safe_target": grand_na_safe_target,
                "grand_na_no_path": grand_na_no_path,
                "grand_na_collision": grand_na_collision,
                "grand_collision": grand_collision,
                "grand_no_collision": grand_no_collision,
                "total_fail_files": total_fail_files,
                "total_na_files": total_na_files,
                "total_success_files": total_success_files,
            })
            save_partial_summary(summary_path, summary_dict)
            
            with open(checkpoint_path, "a", encoding="utf-8") as cp:
                cp.write(marzip_file + "\n")
                
        except Exception as e:
            error_msg = f"[ERROR] 파일 처리 중 오류: {marzip_file}, {e}\n"
            print(error_msg)
            with open(analysis_log_path, "a", encoding="utf-8") as f:
                f.write(f"[ERROR] {marzip_file} 처리 중 예외 발생: {e}\n")
    
    total_fail = grand_collision + grand_na_collision
    total_na = grand_na_safe_target + grand_na_no_path + grand_na_collision
    total_success = grand_no_collision
    total_events = total_fail + total_na + total_success

    summary_msg = (
        f"[SUMMARY] (Total Files={grand_file_count}, Total Events={total_events})\n"
        f"  NA-SafeTarget={grand_na_safe_target}\n"
        f"  NA-NoPath={grand_na_no_path}\n"
        f"  NA-Collision={grand_na_collision}\n"
        f"  Collision={grand_collision}\n"
        f"  NoCollision={grand_no_collision}\n\n"
        f"  === File Results ===\n"
        f"  FAIL Files={total_fail_files}\n"
        f"  NA Files={total_na_files}\n"
        f"  SUCCESS Files={total_success_files}\n"
        f"  Total Files={grand_file_count}\n"
        f"마지막 처리 파일: {os.path.basename(last_processed_file) if last_processed_file else '없음'}\n"
    )
    print(summary_msg)
    with open(analysis_log_path, "a", encoding="utf-8") as f:
        f.write(summary_msg)

if __name__ == "__main__":
    main()
