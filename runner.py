from event_analyzer import main as analysis_main


def run_analysis(aggregate_results, analysis_data_dir):
    """
    이벤트 분석을 실행합니다.
    전달받은 설정값을 이용하여 EventAnalysisRunner의 main() 함수를 호출합니다.
    """
    analysis_main(aggregate_results, analysis_data_dir)

def run_event_plotter(plot_data_dir, result_dir):
    """
    이벤트 플롯을 실행합니다.
    전달받은 설정값을 이용하여 EventPlotter의 main() 함수를 호출합니다.
    """
    from event_plotter import main as event_plotter_main
    event_plotter_main(plot_data_dir, result_dir)

def run_target_distribution(target_data_dir, target_result_dir):
    """
    타겟 분포(Distribution)를 실행합니다.
    전달받은 설정값을 이용하여 TargetDistribution의 main() 함수를 호출합니다.
    """
    from target_distribution import main
    main(target_data_dir, target_result_dir)

def main():
    # 아래 설정값들을 원하는 값으로 수정하세요.
    aggregate_results = True
    analysis_data_dir = "data/ver014_20205215_basic_test"      # 이벤트 분석용 데이터 폴더
    plot_data_dir     = "data/ver014_20205215_basic_test"         # 이벤트 플롯용 데이터 폴더
    result_dir   = "result/ver014_20205215_basic_test"    # 이벤트 플롯 결과 저장 폴더
    target_data_dir   = "data/ver014_20205215_basic_test"           # 타겟 분포용 데이터 폴더
    target_result_dir = "result/ver014_20205215_basic_test"      # 타겟 분포 결과 저장 폴더

    # 실행할 작업들을 리스트에 지정합니다.
    # 가능한 값: "analysis", "eventplot", "targetdist"
    tasks = ["targetdist"]

    for task in tasks:
        print(f"\n========== {task} 시작 ==========")
        if task == "analysis":
            run_analysis(aggregate_results, analysis_data_dir)
        elif task == "eventplot":
            run_event_plotter(plot_data_dir, result_dir)
        elif task == "targetdist":
            run_target_distribution(target_data_dir, target_result_dir)
        else:
            print(f"알 수 없는 작업: {task}")
        print(f"========== {task} 완료 ==========")

if __name__ == "__main__":
    main()
