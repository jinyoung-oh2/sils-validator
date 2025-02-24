#!/usr/bin/env python3
import os
import shutil
import argparse

def process_log_file(log_path, dest_folder, search_keywords, base_search_path):
    """
    log_path: 로그 파일 경로 (예: analysis_log.txt 또는 절대/상대 경로)
    dest_folder: 문제 파일들을 복사할 대상 폴더 (하위 폴더: NA 또는 fail)
    search_keywords: 검색할 키워드 리스트 (예: ["Collision=1", "NA-Collision=1"])
                     기본 키워드가 "Collision=1"인 경우, NA-Collision=1이 포함된 라인은 무시합니다.
    base_search_path: marzip 파일들의 상위 경로 (복사 대상 파일의 상대 경로 추출에 사용)
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match_found = False
            for keyword in search_keywords:
                if keyword in line:
                    match_found = True
                    break
            if not match_found:
                continue

            # 로그 라인 예시:
            # [FOLDER_LOG] data/경로/.../scen_158.marzip => events=1, ... NA-Collision=1, ...
            try:
                filepath = line.split("]")[1].split("=>")[0].strip()
            except IndexError:
                print(f"파일 경로 추출 실패: {line}")
                continue

            # NA-Collision=1가 있으면 하위 폴더를 NA, 그렇지 않으면 fail로 지정
            subfolder = "NA" if "NA-Collision=1" in line else "fail"
            dest_subfolder = os.path.join(dest_folder, subfolder)
            if not os.path.exists(dest_subfolder):
                os.makedirs(dest_subfolder)
            
            # base_search_path 기준 상대 경로를 이용해 고유한 파일 이름 생성
            try:
                relative_path = os.path.relpath(filepath, base_search_path)
            except ValueError:
                relative_path = os.path.basename(filepath)
            # 경로 구분자를 언더바로 치환
            dest_filename = relative_path.replace(os.sep, '_')
            dest_path = os.path.join(dest_subfolder, dest_filename)
            
            try:
                shutil.copy(filepath, dest_path)
                print(f"복사 성공: {filepath} -> {dest_path}")
            except Exception as e:
                print(f"복사 실패: {filepath}, 에러: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="로그 파일에서 문제 marzip 파일을 검색하여 복사합니다."
    )
    parser.add_argument(
        "--base", 
        default="data/ver014_20250220_colregs_test_5", 
        help="marzip 파일들이 포함된 상위 경로 (기본: data/ver014_20250220_colregs_test_3)"
    )
    parser.add_argument(
        "--log", 
        default="analyzer/ver014_20250220_colregs_test_5/analysis_log.txt", 
        help=("로그 파일 경로. 절대 경로일 수도 있고, 현재 작업 디렉토리 기준의 상대 경로일 수도 있습니다. "
              "예를 들어, analyzer/ver014_20250220_colregs_test_4/analysis_log.txt처럼 base 경로와 다른 위치에 있다면 해당 경로를 그대로 입력하세요.")
    )
    parser.add_argument(
        "--keywords", 
        nargs="+", 
        # default=["Result=FAIL", "Result=NA (NA-Collision=1)"],
        default=["events=2"],
        help="검색할 키워드 리스트 (여러 개 입력 가능). 'Collision=1' 입력시 NA-Collision=1는 제외됩니다."
    )
    
    args = parser.parse_args()
    
    # 먼저 --log로 입력된 경로가 존재하는지 확인 (절대/현재 작업 디렉토리 기준)
    if os.path.exists(args.log):
        log_path = args.log
    else:
        # 없으면 base 경로와 결합한 경로로 시도
        possible_path = os.path.join(args.base, args.log)
        if os.path.exists(possible_path):
            log_path = possible_path
        else:
            print(f"로그 파일을 찾을 수 없습니다: {args.log} 또는 {possible_path}")
            return

    base_search_path = args.base
    base_folder_name = os.path.basename(os.path.normpath(base_search_path))
    dest_folder = os.path.join("result", base_folder_name, "problematic_files")
    
    process_log_file(log_path, dest_folder, args.keywords, base_search_path)
    print("모든 문제 파일 복사 완료.")

if __name__ == "__main__":
    main()
