import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_uncertainty(csv_file):
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # 필요한 컬럼을 숫자형으로 변환
    numeric_cols = ["cog mean", "cog std", "sog mean", "sog std"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # 0.0 (소수점 한 자리까지 반올림)인 데이터는 제외
    for col in numeric_cols:
        df = df[df[col].round(1) != 0.0]
    
    # 산점도: cog mean vs cog std
    plt.figure(figsize=(8,6))
    plt.scatter(df["cog mean"], df["cog std"], alpha=0.5, s=10, color='blue')
    plt.xlabel("COG Mean (°)")
    plt.ylabel("COG Std (°)")
    plt.title("Scatter Plot: COG Mean vs COG Std")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_cog_uncertainty.png")
    plt.show()
    
    # 산점도: sog mean vs sog std
    plt.figure(figsize=(8,6))
    plt.scatter(df["sog mean"], df["sog std"], alpha=0.5, s=10, color='green')
    plt.xlabel("SOG Mean (kn)")
    plt.ylabel("SOG Std (kn)")
    plt.title("Scatter Plot: SOG Mean vs SOG Std")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("scatter_sog_uncertainty.png")
    plt.show()
    
    # 에러바 플롯: 무작위 샘플(예: 100개) 타겟에 대해 평균과 불확실성 표시
    sample_df = df.sample(n=100, random_state=42)
    
    plt.figure(figsize=(10,6))
    plt.errorbar(sample_df["cog mean"], np.zeros(len(sample_df)),
                 xerr=sample_df["cog std"], fmt='o', color='blue',
                 ecolor='lightblue', capsize=3)
    plt.xlabel("COG Mean (°)")
    plt.title("COG Mean with Uncertainty (Error Bars) - Sample Targets")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("errorbar_cog_uncertainty.png")
    plt.show()
    
    plt.figure(figsize=(10,6))
    plt.errorbar(sample_df["sog mean"], np.zeros(len(sample_df)),
                 xerr=sample_df["sog std"], fmt='o', color='green',
                 ecolor='lightgreen', capsize=3)
    plt.xlabel("SOG Mean (kn)")
    plt.title("SOG Mean with Uncertainty (Error Bars) - Sample Targets")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("errorbar_sog_uncertainty.png")
    plt.show()
    
    # 박스플롯: 전체 타겟에 대한 불확실성 분포 (COG std와 SOG std)
    plt.figure(figsize=(6,6))
    plt.boxplot([df["cog std"].dropna(), df["sog std"].dropna()],
                labels=["COG Std (°)", "SOG Std (kn)"],
                patch_artist=True)
    plt.title("Boxplot of Uncertainty (Std)")
    plt.ylabel("Standard Deviation")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("boxplot_uncertainty.png")
    plt.show()


def plot_movement(csv_file):
    csv_file = "~/workspace/timeseries/CA_v0.1.4_data/Test/20250226/aggregated_movement_summary.csv"
    
    # CSV 파일 읽기
    df = pd.read_csv(csv_file)
    
    # diff_cog와 diff_sog를 숫자형으로 변환
    df["diff_cog"] = pd.to_numeric(df["diff_cog"], errors="coerce")
    df["diff_sog"] = pd.to_numeric(df["diff_sog"], errors="coerce")
    
    # 소수점 한 자리까지 반올림 후 0.0인 값 제거
    df_diff_cog = df[df["diff_cog"].round(1) != 0.00]
    df_diff_sog = df[df["diff_sog"].round(1) != 0.00]
    
    # bin 폭 (단위)
    bin_width = 1
    
    # diff_cog의 최소/최대 구간 결정 (bin_edge는 bin_width 단위)
    if not df_diff_cog["diff_cog"].empty:
        min_val_cog = np.floor(df_diff_cog["diff_cog"].min())
        max_val_cog = np.ceil(df_diff_cog["diff_cog"].max())
        bins_cog = np.arange(min_val_cog, max_val_cog + bin_width, bin_width)
    else:
        bins_cog = 20  # 기본값
    
    # diff_sog의 최소/최대 구간 결정
    if not df_diff_sog["diff_sog"].empty:
        min_val_sog = np.floor(df_diff_sog["diff_sog"].min())
        max_val_sog = np.ceil(df_diff_sog["diff_sog"].max())
        bins_sog = np.arange(min_val_sog, max_val_sog + bin_width, bin_width)
    else:
        bins_sog = 20  # 기본값
    
    # diff_cog 히스토그램 그리기
    plt.figure(figsize=(8,6))
    plt.hist(df_diff_cog["diff_cog"], bins=bins_cog, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title("Histogram of cog change movement (excluding 0.0)")
    plt.xlabel("cog change(deg)")
    plt.ylabel("Number")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("hist_diff_cog.png")
    plt.show()
    
    # diff_sog 히스토그램 그리기
    plt.figure(figsize=(8,6))
    plt.hist(df_diff_sog["diff_sog"], bins=bins_sog, color="salmon", edgecolor="black", alpha=0.7)
    plt.title("Histogram of sog change movement (excluding 0.0)")
    plt.xlabel("sog change(kn)")
    plt.ylabel("Number")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("hist_diff_sog.png")
    plt.show()

if __name__ == "__main__":

    csv_file = "~/workspace/timeseries/CA_v0.1.4_data/Test/20250226/aggregated_non_movement_summary.csv"

    plot_uncertainty(csv_file)