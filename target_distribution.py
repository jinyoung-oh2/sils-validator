#!/usr/bin/env python3
import os
import shutil
import argparse

def process_log_file(log_path, dest_folder, search_keywords, base_search_path):
    """
    log_path: 로그 파일 경로 (예: analysis_log.txt 또는 절대/상대 경로)
    dest_folder: 문제 파일들을 복사할 대상 폴더 (하위 폴더: NA 또는 fail)
    search_keywords: 검색할 키워드 리스트 (예: ["Collision=1"])
                     단, "Collision=1" 입력 시 NA-Collision=1, NoCollision*는 제외합니다.
    base_search_path: marzip 파일들의 상위 경로 (대상 파일의 상대 경로 추출에 사용)
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            match_found = False
            for keyword in search_keywords:
                if keyword == "Collision=1":
                    # "Collision=1"이 포함되어 있으면서, "NA-Collision=1"이나 "NoCollision"이 포함되지 않은 경우에만 match 처리
                    if keyword in line and "NA-Collision=1" not in line and "NoCollision" not in line:
                        match_found = True
                        break
                else:
                    if keyword in line:
                        match_found = True
                        break
            if not match_found:
                continue

            # 로그 라인이 예상 형식([FOLDER_LOG] ... => ...)인지 확인 후 파일 경로 추출
            if "]" not in line or "=>" not in line:
                print(f"파일 경로 추출 실패(형식 오류): {line.strip()}")
                continue
            try:
                filepath = line.split("]")[1].split("=>")[0].strip()
            except IndexError:
                print(f"파일 경로 추출 실패: {line.strip()}")
                continue

            # 조건에 따라 하위 폴더 결정: NA-Collision=1이면 NA, 아니면 fail
            subfolder = "NA" if "NA-Collision=1" in line else "fail"
            dest_subfolder = os.path.join(dest_folder, subfolder)
            if not os.path.exists(dest_subfolder):
                os.makedirs(dest_subfolder)
            
            # base_search_path 기준 상대 경로를 얻어 파일 이름에 포함 (구분자를 언더바로 치환)
            try:
                relative_path = os.path.relpath(filepath, base_search_path)
            except ValueError:
                relative_path = os.path.basename(filepath)
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
        default="data/ver014_20250220_colregs_test_3", 
        help="marzip 파일들이 포함된 상위 경로 (기본: data/ver014_20250220_colregs_test_3)"
    )
    parser.add_argument(
        "--log", 
        default="analysis_log.txt", 
        help=("로그 파일 경로. 절대 경로 또는 현재 작업 디렉토리 기준의 상대 경로를 사용할 수 있습니다. "
              "예를 들어, analyzer/ver014_20250220_colregs_test_3/analysis_log.txt처럼 base 경로와 다른 위치에 있다면 해당 경로를 입력하세요.")
    )
    parser.add_argument(
        "--keywords", 
        nargs="+", 
        default=["Collision=1"],
        help="검색할 키워드 리스트 (여러 개 입력 가능). 'Collision=1' 입력시 NA-Collision=1, NoCollision*는 제외됩니다."
    )
    
    args = parser.parse_args()
    
    # 먼저 --log 옵션으로 지정된 경로가 존재하는지 확인합니다.
    if os.path.exists(args.log):
        log_path = args.log
    else:
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
