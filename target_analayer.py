import os
import concurrent.futures
import gc
import matplotlib
matplotlib.use("Agg")  # 결과를 파일로 저장
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
from tqdm import tqdm

from basic_test_fail_anlyzer import BasicTestFailAnalyzer
from file_input_manager import FileInputManager


class BasicTestFailAnalyzerWithExtendedPlot(BasicTestFailAnalyzer):
    """
    BasicTestFailAnalyzer를 상속받아, 여러 마집 파일의 분석 결과와 
    초기 타겟의 sog 값 및 이벤트의 distanceToTarget, cpa, tcpa 값을 
    분류별(ALL, N/A, Success, Fail)로 시각화하는 클래스입니다.
    
    각 파일 처리 결과는 다음과 같은 딕셔너리로 저장됩니다:
      {"Result": ..., "initial": [sog, ...],
       "event": {"distanceToTarget": ..., "cpa": ..., "tcpa": ...} }
    """
    def __init__(self, base_data_dir):
        super().__init__()
        self.base_data_dir = base_data_dir
        self.file_manager = FileInputManager(base_data_dir)
        self.results = []  # 각 파일의 결과 저장

    def _extract_file_info(self, marzip_file):
        base_name = os.path.splitext(os.path.basename(marzip_file))[0]
        parent_dir = os.path.dirname(marzip_file)
        grandparent_dir = os.path.dirname(parent_dir) if parent_dir else ""
        folder_info = os.path.basename(grandparent_dir) if grandparent_dir else ""
        return folder_info, base_name

    def evaluate_single_file(self, marzip_file):
        try:
            analyzer = BasicTestFailAnalyzerWithExtendedPlot(self.base_data_dir)
            analysis = analyzer.analyze_file(marzip_file)
        except Exception as e:
            print(f"Error processing {marzip_file}: {e}")
            return None

        # 초기 타겟의 sog 값 추출 (simulation_result → trafficSituation → targetShips → initial)
        initial = []
        sim_result = analyzer.simulation_result
        if sim_result:
            target_ships = sim_result.get("trafficSituation", {}).get("targetShips", [])
            for target in target_ships:
                init_data = target.get("initial")
                if not init_data:
                    continue
                sog_val = init_data.get("sog")
                if sog_val is None:
                    continue
                try:
                    initial.append(float(sog_val))
                except Exception:
                    pass

        # 첫 번째 이벤트의 첫 번째 target_ship 정보 추출
        event = None
        if analyzer.events:
            first_event = analyzer.events[0]
            target_ships = first_event.get("target_ships")
            if target_ships and isinstance(target_ships, list) and len(target_ships) > 0:
                first_target = target_ships[0]
                event = {
                    "distanceToTarget": first_target.get("distanceToTarget"),
                    "cpa": first_target.get("cpa"),
                    "tcpa": first_target.get("tcpa")
                }

        folder_info, base_name = self._extract_file_info(marzip_file)
        return {
            "Result": analysis["result"],
            "initial": initial,
            "event": event
        }

    def run_all_files(self):
        marzip_files = self.file_manager.get_all_marzip_files()
        total_files = len(marzip_files)
        print(f"총 {total_files}개의 마집 파일을 찾았습니다.")

        results = []
        batch_size = 200  # 배치 크기는 시스템 상황에 따라 조정
        num_batches = (total_files + batch_size - 1) // batch_size

        for batch_index in range(num_batches):
            start = batch_index * batch_size
            end = start + batch_size
            batch = marzip_files[start:end]
            print(f"배치 {batch_index+1}/{num_batches} 처리 중...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                batch_results = list(tqdm(
                    executor.map(self.evaluate_single_file, batch, chunksize=10),
                    total=len(batch),
                    desc=f"Batch {batch_index+1}/{num_batches}"
                ))
            results.extend([r for r in batch_results if r is not None])
            # 배치 처리 후 메모리 해제
            del batch_results, batch
            gc.collect()

        self.results = results

    def plot_histogram_by_classification(self, data_list, parameter_name, output_file, draw_gaussian=False):
        ordered_categories = ["all"]
        categories = [cat for cat in ordered_categories if cat in data_list]
        n_cats = len(categories)
        
        fig, axs = plt.subplots(n_cats, 1, figsize=(10, 4 * n_cats), sharex=True)
        if n_cats == 1:
            axs = [axs]
        
        for ax, cat in zip(axs, categories):
            values = data_list[cat]
            if values:
                counts, bins, patches = ax.hist(values, bins=20, density=True, 
                                                color="skyblue", edgecolor="black", alpha=0.6)
                
                if draw_gaussian:
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 200)
                    pdf = norm.pdf(x, mean_val, std_val)
                    ax.plot(x, pdf, 'r', linewidth=2, label="Gaussian Fit")
                    peak_index = np.argmax(pdf)
                    peak_x = x[peak_index]
                    peak_y = pdf[peak_index]
                    ax.annotate(f'Peak: {peak_x:.2f}', 
                                xy=(peak_x, peak_y), 
                                xytext=(peak_x, peak_y*1.1),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                horizontalalignment='center')
                    
                    ax.axvline(mean_val, color='blue', linestyle='--', linewidth=1, label="Mean")
                    ax.axvline(mean_val + std_val, color='green', linestyle='--', linewidth=1, label="1σ")
                    ax.axvline(mean_val - std_val, color='green', linestyle='--', linewidth=1)
                    ax.axvline(mean_val + 2*std_val, color='orange', linestyle='--', linewidth=1, label="2σ")
                    ax.axvline(mean_val - 2*std_val, color='orange', linestyle='--', linewidth=1)
                    ax.axvline(mean_val + 3*std_val, color='purple', linestyle='--', linewidth=1, label="3σ")
                    ax.axvline(mean_val - 3*std_val, color='purple', linestyle='--', linewidth=1)
                    
                    ax.legend()
            
            ax.set_title(f"{parameter_name.upper()} Histogram for '{cat}'")
            ax.set_xlabel(f"{parameter_name.upper()} Value")
            ax.set_ylabel("Frequency")
        
        plt.tight_layout()
        plt.savefig(output_file)
        plt.close()
        print(f"{parameter_name.upper()} 히스토그램이 {output_file}에 저장되었습니다. (Gaussian Fit: {draw_gaussian})")

    def _plot_event_histogram(self, param, out_dir):
        """하나의 이벤트 파라미터에 대해 히스토그램을 생성하는 헬퍼 함수."""
        event_groups = {"all": []}
        for r in self.results:
            event_data = r.get("event")
            if event_data is None:
                continue
            val = event_data.get(param)
            if val is None:
                continue
            try:
                fval = float(val)
            except Exception:
                continue
            # 그룹 분류 함수
            res = r.get("Result", "").lower()
            event_groups["all"].append(fval)
            if res.startswith("n/a"):
                event_groups.setdefault("n/a", []).append(fval)
            elif res.startswith("success"):
                event_groups.setdefault("success", []).append(fval)
            elif res.startswith("fail"):
                event_groups.setdefault("fail", []).append(fval)
        output_file = os.path.join(out_dir, f"classification_event_{param}_histogram.png")
        self.plot_histogram_by_classification(
            data_list=event_groups,
            parameter_name=param,
            output_file=output_file
        )

    def run_and_save(self, out_dir):
        """
        - 모든 마집 파일을 처리(run_all_files)
        - 초기(initial)은 sog 파라미터 히스토그램 1장
        - 이벤트(event)는 distanceToTarget, cpa, tcpa 히스토그램 3장(병렬 처리)
        """
        self.run_all_files()

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # initial - sog 플롯은 단일 프로세스로 처리
        initial_groups = {"all": []}
        for r in self.results:
            res = r.get("Result", "").lower()
            for sog in r.get("initial", []):
                initial_groups["all"].append(sog)
                if res.startswith("n/a"):
                    initial_groups.setdefault("n/a", []).append(sog)
                elif res.startswith("success"):
                    initial_groups.setdefault("success", []).append(sog)
                elif res.startswith("fail"):
                    initial_groups.setdefault("fail", []).append(sog)
        self.plot_histogram_by_classification(
            data_list=initial_groups,
            parameter_name="sog",
            output_file=os.path.join(out_dir, "classification_initial_sog_histogram.png")
        )

        # 이벤트 플롯(여러 파라미터)을 병렬 처리
        event_params = ["distanceToTarget", "cpa", "tcpa"]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._plot_event_histogram, param, out_dir) for param in event_params]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"이벤트 플롯 생성 중 에러: {e}")


def main():
    base_data_dir = "/media/avikus/One Touch/HinasControlSilsCA/CA_v0.1.4_data/COLREG_TESTING"
    out_dir = "analyze/CA_v0.1.4_data/Colreg/20250227_1"
    plotter = BasicTestFailAnalyzerWithExtendedPlot(base_data_dir)
    plotter.run_and_save(out_dir)
    print("DONE")


if __name__ == "__main__":
    main()
