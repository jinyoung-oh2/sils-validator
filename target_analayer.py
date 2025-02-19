import os
import concurrent.futures
import matplotlib
matplotlib.use("Agg")  # 결과를 파일로 저장
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np



from basic_test_fail_anlyzer import BasicTestFailAnalyzer
from file_input_manager import FileInputManager


class BasicTestFailAnalyzerWithExtendedPlot(BasicTestFailAnalyzer):
    """
    BasicTestFailAnalyzer를 상속받아, 여러 마집 파일의 분석 결과와 
    초기 타겟의 sog 값 및 이벤트의 distanceToTarget, cpa, tcpa 값을 
    분류별(ALL, N/A, Success, Fail)로 시각화하는 클래스입니다.
    
    각 파일 처리 결과는 다음과 같은 딕셔너리로 저장됩니다:
      {"Folder": ..., "File": ..., "Result": ..., "initial": [sog, ...],
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
        print(f"총 {len(marzip_files)}개의 마집 파일을 찾았습니다.")
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for res in executor.map(self.evaluate_single_file, marzip_files):
                if res is not None:
                    results.append(res)
        self.results = results

    def plot_histogram_by_classification(self, data_list, parameter_name, output_file, draw_gaussian=False):
        """
        data_list: (파일마다의) 값 리스트의 딕셔너리 그룹 {"all": [...], "n/a": [...], "success": [...], "fail": [...]}
        parameter_name: 파라미터 이름 (표시용)
        output_file: 저장할 그림 파일 경로
        draw_gaussian: True이면 히스토그램 위에 가우시안 피팅 곡선을 겹쳐 그립니다.
        """
        # 사용할 카테고리 (필요에 따라 확장 가능)
        ordered_categories = ["all", "n/a"]
        categories = [cat for cat in ordered_categories if cat in data_list]
        n_cats = len(categories)
        
        fig, axs = plt.subplots(n_cats, 1, figsize=(10, 4 * n_cats), sharex=True)
        if n_cats == 1:
            axs = [axs]
        
        for ax, cat in zip(axs, categories):
            values = data_list[cat]
            if values:
                # density=True 로 정규화된 히스토그램 그리기
                counts, bins, patches = ax.hist(values, bins=20, density=True, 
                                                color="skyblue", edgecolor="black", alpha=0.6)
                
                if draw_gaussian:
                    # 데이터의 평균과 표준편차 계산
                    mean_val = np.mean(values)
                    std_val = np.std(values)
                    
                    # x 범위 지정
                    xmin, xmax = ax.get_xlim()
                    x = np.linspace(xmin, xmax, 200)
                    
                    # 정규분포 PDF 계산
                    pdf = norm.pdf(x, mean_val, std_val)
                    ax.plot(x, pdf, 'r', linewidth=2, label="Gaussian Fit")
                    
                    # 피크 (PDF 최댓값) 위치 계산 및 표시
                    peak_index = np.argmax(pdf)
                    peak_x = x[peak_index]
                    peak_y = pdf[peak_index]
                    ax.annotate(f'Peak: {peak_x:.2f}', 
                                xy=(peak_x, peak_y), 
                                xytext=(peak_x, peak_y*1.1),
                                arrowprops=dict(facecolor='black', shrink=0.05),
                                horizontalalignment='center')
                    
                    # 평균 및 1σ, 2σ, 3σ 위치에 수직선 그리기
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

    def run_and_save(self, out_dir):
        """
        - 모든 마집 파일을 처리(run_all_files)
        - 초기(initial)은 sog 파라미터 히스토그램 1장
        - 이벤트(event)는 distanceToTarget, cpa, tcpa 히스토그램 3장
        """
        self.run_all_files()

        # 결과를 저장할 폴더 생성
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        # 그룹별 값 추출 함수 (결과의 Result 기준으로 분류)
        def group_values(values, result):
            groups = {"all": []}
            groups["all"].extend(values)
            res = result.lower()
            if res.startswith("n/a"):
                groups.setdefault("n/a", []).extend(values)
            elif res.startswith("success"):
                groups.setdefault("success", []).extend(values)
            elif res.startswith("fail"):
                groups.setdefault("fail", []).extend(values)
            return groups

        # initial - sog
        initial_groups = {"all": []}
        for r in self.results:
            groups = group_values(r.get("initial", []), r.get("Result", ""))
            # 모든 그룹에 대해 값 병합
            for key, vals in groups.items():
                initial_groups.setdefault(key, []).extend(vals)
        self.plot_histogram_by_classification(
            data_list=initial_groups,
            parameter_name="sog (initial)",
            output_file=os.path.join(out_dir, "classification_initial_sog_histogram.png")
        )

        # event - distanceToTarget, cpa, tcpa
        for param in ["distanceToTarget", "cpa", "tcpa"]:
            event_groups = {"all": []}
            for r in self.results:
                event_data = r.get("event", {})
                val = event_data.get(param)
                if val is None:
                    continue
                try:
                    fval = float(val)
                except Exception:
                    continue
                groups = group_values([fval], r.get("Result", ""))
                for key, vals in groups.items():
                    event_groups.setdefault(key, []).extend(vals)
            self.plot_histogram_by_classification(
                data_list=event_groups,
                parameter_name=f"event {param}",
                output_file=os.path.join(out_dir, f"classification_event_{param}_histogram.png")
            )


def main():
    base_data_dir = "data/ver014_20250219_colregs_test"
    out_dir = "result/ver014_20250219_colregs_test"

    plotter = BasicTestFailAnalyzerWithExtendedPlot(base_data_dir)
    plotter.run_and_save(out_dir)
    print("DONE")


if __name__ == "__main__":
    main()
