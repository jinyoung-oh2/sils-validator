#!/usr/bin/env python3
import os
import csv
import shutil
import fnmatch

def find_file_with_pattern(expected_pattern, search_root):
    """
    search_root 아래의 모든 하위 폴더를 검색하여, 
    파일명이 expected_pattern과 fnmatch 방식으로 일치하는 파일 경로를 반환합니다.
    (소문자 변환 후 비교)
    없으면 None을 반환합니다.
    """
    expected_pattern = expected_pattern.lower()
    for root, dirs, files in os.walk(search_root):
        for f in files:
            if fnmatch.fnmatch(f.lower(), expected_pattern):
                return os.path.join(root, f)
    return None

def process_csv(csv_path, base_search_path, dest_folder):
    """
    csv_path: CSV 파일 경로 (헤더: Folder, File, Result)
    base_search_path: marzip 파일들을 검색할 base 경로
    dest_folder: 문제 파일들을 복사할 대상 폴더 (여기서 하위 폴더로 N_A와 fail로 분류)
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        problematic_entries = []
        for row in reader:
            folder = row.get("Folder", "").strip()
            file_base = row.get("File", "").strip()
            # Summary 또는 Folder 값이 없는 행은 건너뜁니다.
            if not folder or not file_base or file_base.lower() == "summary":
                continue
            result = row.get("Result", "").lower()
            # "fail" 또는 "n/a"가 결과에 있으면 문제 파일로 판단합니다.
            if "fail" in result or "n/a" in result:
                problematic_entries.append(row)
    
    print(f"CSV 파일에서 문제 항목 {len(problematic_entries)}개를 찾았습니다.")
    
    for row in problematic_entries:
        folder = row.get("Folder", "").strip()
        file_base = row.get("File", "").strip()
        result_text = row.get("Result", "").lower()
        # 문제 결과에 따라 하위 폴더 결정: "n/a"가 있으면 N_A, 아니면 fail
        subfolder = "NA" if "n/a" in result_text else "fail"
        
        # 대상 하위 폴더 생성 (dest_folder 내에)
        dest_subfolder = os.path.join(dest_folder, subfolder)
        if not os.path.exists(dest_subfolder):
            os.makedirs(dest_subfolder)
        
        # 예상 상대 경로: "folder/file_base.marzip"
        expected_relpath = os.path.join(folder, f"{file_base}.marzip")
        print(f"검색 중 (직접): {expected_relpath}")
        direct_path = os.path.join(base_search_path, expected_relpath)
        if os.path.exists(direct_path):
            src_file = direct_path
            print(f"직접 찾음: {direct_path}")
        else:
            # 직접 찾지 못한 경우, 패턴 검색 진행
            pattern = f"*{file_base}*.marzip"
            print(f"직접 찾지 못함. 패턴 검색 시도: Folder='{folder}', Pattern='{pattern}'")
            folder_path = os.path.join(base_search_path, folder)
            if os.path.exists(folder_path):
                src_file = find_file_with_pattern(pattern, folder_path)
            else:
                src_file = find_file_with_pattern(pattern, base_search_path)
        if src_file:
            # 대상 파일명: 폴더와 파일명을 결합 (예: ver014_colreg_testing_6_20250219T140954_scen_193.marzip)
            dest_filename = f"{folder}_{file_base}.marzip"
            dest_path = os.path.join(dest_subfolder, dest_filename)
            try:
                shutil.copy(src_file, dest_path)
                print(f"복사 성공: {src_file} -> {dest_path}")
            except Exception as e:
                print(f"복사 실패: {src_file}, 에러: {e}")
        else:
            print(f"파일을 찾지 못했습니다: {expected_relpath} (패턴: {pattern})")

def main():
    # base_search_path 예시: "data/ver014_20250219_colregs_test"
    base_search_path = "data/ver014_20250219_colregs_test"
    # CSV 파일 경로: base_search_path 내에 있다고 가정
    csv_path = os.path.join(base_search_path, "all_event_analysis.csv")
    # 문제 파일들을 복사할 대상 폴더를 result 하위에, base_search_path 마지막 폴더명으로 구성
    base_folder_name = os.path.basename(os.path.normpath(base_search_path))
    dest_folder = os.path.join("result", base_folder_name, "problematic_files")
    
    process_csv(csv_path, base_search_path, dest_folder)
    print("모든 문제 파일 복사 완료.")

if __name__ == "__main__":
    main()
