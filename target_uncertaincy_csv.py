import os
import csv
import numpy as np
from marzip_extractor import MarzipExtractor

def angular_diff(value, mean):
    """Returns the minimal difference between two angles (°), result range: [-180, 180)."""
    return (value - mean + 180) % 360 - 180

class TargetShipTimeseriesAnalyzer(MarzipExtractor):
    def analyze_timeseries(self, marzip_file, output_dir, base_to_trim, nm_csv_writer=None, m_csv_writer=None):
        """
        marzip 파일을 분석하여 non‑movement와 movement 타겟의 통계 데이터를
        각각 CSV 행으로 기록하고, 추가로 타겟 개수 및 duration 통계도 계산합니다.
        
        - non‑movement 타겟:
            * COG와 SOG의 평균(mean)과 표준편차(std)를 계산 (COG는 첫번째 값을 기준으로 unwrapping)
            * CSV 칼럼: FilePath, TargetID, cog mean, cog std, sog mean, sog std
        - movement 타겟:
            * 측정값이 1개 이상이면 시작 COG, 종료 COG, 차이(diff_cog), 시작 SOG, 종료 SOG, 차이를 계산
            * CSV 칼럼: FilePath, TargetID, start_cog, end_cog, diff_cog, start_sog, end_sog, diff_sog
        
        추가로 반환:
          - target_counts: { "FilePath": ..., "non_movement_count": ..., "movement_count": ... }
          - duration_stats: { "FilePath": ..., "avg_duration": ..., "min_duration": ..., "max_duration": ... }
        
        CSV에 기록할 때, base_to_trim 이후의 상대경로만 FilePath로 기록합니다.
        """
        # 데이터 추출
        self.run(marzip_file)
        targets = self.simulation_result.get("trafficSituation", {}).get("targetShips", [])
        
        # base_to_trim 이후의 상대경로 사용
        file_path_full = os.path.relpath(os.path.abspath(marzip_file), base_to_trim)
        
        # --- non‑movement 타겟 처리 ---
        non_movement_ids = set()
        for target in targets:
            if not target.get("movement"):
                tid = target.get("static", {}).get("id")
                if tid is not None:
                    non_movement_ids.add(tid)
        
        nm_groups = {}
        for entry in self.timeseries_dataset:
            entry_id = entry.get("id")
            if entry_id not in non_movement_ids:
                continue
            if entry_id not in nm_groups:
                nm_groups[entry_id] = {"sog": [], "cog": []}
            if "sog" in entry:
                try:
                    nm_groups[entry_id]["sog"].append(float(entry["sog"]))
                except Exception:
                    pass
            if "cog" in entry:
                try:
                    nm_groups[entry_id]["cog"].append(float(entry["cog"]))
                except Exception:
                    pass
        
        for target_id, data in sorted(nm_groups.items(), key=lambda x: x[0]):
            if data["sog"]:
                sog_arr = np.array(data["sog"])
                sog_mean = np.mean(sog_arr)
                sog_std = np.std(sog_arr)
            else:
                sog_mean = sog_std = 0.0
            if data["cog"]:
                cog_arr = np.mod(np.array(data["cog"]), 360)
                initial_cog = cog_arr[0]
                adjusted_cog = np.array([initial_cog + angular_diff(v, initial_cog) for v in cog_arr])
                cog_mean = np.mean(adjusted_cog)
                cog_std = np.std(adjusted_cog)
            else:
                cog_mean = cog_std = 0.0
            row = {
                "FilePath": file_path_full,
                "TargetID": target_id,
                "cog mean": f"{cog_mean:.3f}",
                "cog std": f"{cog_std:.3f}",
                "sog mean": f"{sog_mean:.3f}",
                "sog std": f"{sog_std:.3f}"
            }
            if nm_csv_writer is not None:
                nm_csv_writer.writerow(row)
        
        # --- movement 타겟 처리 ---
        movement_ids = set()
        for target in targets:
            if target.get("movement"):
                tid = target.get("static", {}).get("id")
                if tid is not None:
                    movement_ids.add(tid)
                    
        m_groups = {}
        for entry in self.timeseries_dataset:
            entry_id = entry.get("id")
            if entry_id not in movement_ids:
                continue
            if entry_id not in m_groups:
                m_groups[entry_id] = {"time": [], "sog": [], "cog": []}
            if "timeStamp" in entry:
                try:
                    m_groups[entry_id]["time"].append(float(entry["timeStamp"]))
                except Exception:
                    pass
            if "sog" in entry:
                try:
                    m_groups[entry_id]["sog"].append(float(entry["sog"]))
                except Exception:
                    pass
            if "cog" in entry:
                try:
                    m_groups[entry_id]["cog"].append(float(entry["cog"]))
                except Exception:
                    pass
        
        for target_id, data in sorted(m_groups.items(), key=lambda x: x[0]):
            if len(data["time"]) < 1:
                continue
            times = np.array(data["time"])
            sog_vals = np.array(data["sog"]) if data["sog"] else np.array([])
            cog_vals = np.array(data["cog"]) if data["cog"] else np.array([])
            if len(times) == 1:
                start_cog = cog_vals[0] if cog_vals.size > 0 else 0.0
                end_cog = start_cog
                diff_cog = 0.0
                start_sog = sog_vals[0] if sog_vals.size > 0 else 0.0
                end_sog = start_sog
                diff_sog = 0.0
            else:
                sort_idx = np.argsort(times)
                times = times[sort_idx]
                sog_vals = sog_vals[sort_idx]
                cog_vals = cog_vals[sort_idx]
                start_cog = cog_vals[0]
                end_cog = cog_vals[-1]
                diff_cog = angular_diff(end_cog, start_cog)
                start_sog = sog_vals[0]
                end_sog = sog_vals[-1]
                diff_sog = end_sog - start_sog
            row = {
                "FilePath": file_path_full,
                "TargetID": target_id,
                "start_cog": f"{start_cog:.3f}",
                "end_cog": f"{end_cog:.3f}",
                "diff_cog": f"{diff_cog:.3f}",
                "start_sog": f"{start_sog:.3f}",
                "end_sog": f"{end_sog:.3f}",
                "diff_sog": f"{diff_sog:.3f}"
            }
            if m_csv_writer is not None:
                m_csv_writer.writerow(row)
        
        # --- 추가: 타겟 개수 및 timeStamp duration 통계 ---
        target_counts = {
            "FilePath": file_path_full,
            "non_movement_count": len(nm_groups),
            "movement_count": len(m_groups)
        }
        
        # 타겟별 timeStamp duration 계산 (timeStamp가 있는 경우, 단위: ns)
        duration_dict = {}
        for entry in self.timeseries_dataset:
            if "timeStamp" in entry:
                tid = entry.get("id")
                try:
                    ts = float(entry["timeStamp"])
                except Exception:
                    continue
                if tid not in duration_dict:
                    duration_dict[tid] = []
                duration_dict[tid].append(ts)
        durations = []
        for ts_list in duration_dict.values():
            if ts_list:
                durations.append(max(ts_list) - min(ts_list))
        if durations:
            # ns를 s로 변환 (1e9 ns = 1 s)하고 소수점은 제거
            avg_duration = int(np.mean(durations) / 1e9)
            min_duration = int(np.min(durations) / 1e9)
            max_duration = int(np.max(durations) / 1e9)
        else:
            avg_duration = min_duration = max_duration = 0
        
        duration_stats = {
            "FilePath": file_path_full,
            "avg_duration": f"{avg_duration}",
            "min_duration": f"{min_duration}",
            "max_duration": f"{max_duration}"
        }
        
        # 파일 처리 후 요약 메시지 출력
        nm_count = len(nm_groups)
        m_count = len(m_groups)
        print(f"Processed file: {file_path_full} - {nm_count} non-movement targets, {m_count} movement targets.")
        
        # 반환값 (추가 CSV 작성을 위한 정보)
        return {"target_counts": target_counts, "duration_stats": duration_stats}

    def get_all_marzip_files(self, base_dir):
        """모든 하위 폴더의 .marzip 파일을 재귀적으로 반환합니다."""
        all_files = []
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f.endswith('.marzip'):
                    all_files.append(os.path.join(root, f))
        return all_files

def main():
    # 데이터 디렉토리 (모든 폴더 포함)
    base_data_dir = "/media/avikus/One Touch/HinasControlSilsCA/CA_v0.1.4_data/Random/20250226"
    # CSV 저장용 출력 디렉토리
    output_base_dir = "timeseries/CA_v0.1.4_data/Test/20250226"
    os.makedirs(output_base_dir, exist_ok=True)
    
    # base_to_trim은 base_data_dir로 지정합니다.
    base_to_trim = base_data_dir
    
    analyzer = TargetShipTimeseriesAnalyzer()
    marzip_files = analyzer.get_all_marzip_files(base_data_dir)
    
    nm_csv_file = os.path.join(output_base_dir, "aggregated_non_movement_summary.csv")
    m_csv_file = os.path.join(output_base_dir, "aggregated_movement_summary.csv")
    counts_csv_file = os.path.join(output_base_dir, "target_counts_summary.csv")
    duration_csv_file = os.path.join(output_base_dir, "duration_stats_summary.csv")
    
    nm_fieldnames = ["FilePath", "TargetID", "cog mean", "cog std", "sog mean", "sog std"]
    m_fieldnames = ["FilePath", "TargetID", "start_cog", "end_cog", "diff_cog", "start_sog", "end_sog", "diff_sog"]
    counts_fieldnames = ["FilePath", "non_movement_count", "movement_count"]
    duration_fieldnames = ["FilePath", "avg_duration", "min_duration", "max_duration"]
    
    with open(nm_csv_file, "w", newline="") as nm_f, \
         open(m_csv_file, "w", newline="") as m_f, \
         open(counts_csv_file, "w", newline="") as counts_f, \
         open(duration_csv_file, "w", newline="") as duration_f:
        
        nm_writer = csv.DictWriter(nm_f, fieldnames=nm_fieldnames)
        m_writer = csv.DictWriter(m_f, fieldnames=m_fieldnames)
        counts_writer = csv.DictWriter(counts_f, fieldnames=counts_fieldnames)
        duration_writer = csv.DictWriter(duration_f, fieldnames=duration_fieldnames)
        
        nm_writer.writeheader()
        m_writer.writeheader()
        counts_writer.writeheader()
        duration_writer.writeheader()
        
        for file_path in marzip_files:
            try:
                results = analyzer.analyze_timeseries(file_path, output_base_dir, base_to_trim,
                                                      nm_csv_writer=nm_writer, m_csv_writer=m_writer)
                counts_writer.writerow(results["target_counts"])
                duration_writer.writerow(results["duration_stats"])
            except Exception as e:
                print(f"Error processing file ({file_path}): {e}")
    
    print(f"Aggregated non-movement CSV saved to {nm_csv_file}")
    print(f"Aggregated movement CSV saved to {m_csv_file}")
    print(f"Target counts CSV saved to {counts_csv_file}")
    print(f"Duration stats CSV saved to {duration_csv_file}")

if __name__ == "__main__":
    main()
