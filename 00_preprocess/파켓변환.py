import pandas as pd
import os

# =============================================================================
# 경로 설정 (VPIN 파이프라인과 동일하게 맞춤)
# =============================================================================
BASE_DIR = r"E:\vpin_project_parquet"
OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")

def convert_parquet_to_excel(sub_folder, file_name, row_limit=1000):
    """
    지정된 출력 폴더 내의 Parquet 파일을 Excel 파일로 변환합니다.
    """
    # 원본 Parquet 파일 경로 구성
    parquet_path = os.path.join(OUTPUT_DIR, sub_folder, file_name)
    
    # 저장할 엑셀 파일 경로 구성 (같은 폴더에 .xlsx 확장자로 저장)
    excel_name = file_name.replace('.parquet', '.xlsx')
    excel_path = os.path.join(OUTPUT_DIR, sub_folder, f"VIEW_{excel_name}")

    if not os.path.exists(parquet_path):
        print(f"❌ 오류: 파일을 찾을 수 없습니다.\n경로: {parquet_path}")
        return

    try:
        print(f"📂 '{file_name}' 파일을 읽어오는 중...")
        df = pd.read_parquet(parquet_path)
        
        total_rows = len(df)
        print(f"✅ 총 {total_rows:,}개의 행을 확인했습니다.")
        
        # 엑셀 열림 방지 및 빠른 확인을 위해 데이터 자르기
        if row_limit is not None and total_rows > row_limit:
            print(f"⚠️ 데이터가 너무 큽니다! 구조 확인을 위해 상위 {row_limit:,}줄만 추출합니다.")
            df_to_save = df.head(row_limit)
        else:
            print("전체 데이터를 변환합니다.")
            df_to_save = df
            
        print(f"💾 엑셀 파일 생성 중... (경로: {excel_path})")
        df_to_save.to_excel(excel_path, index=False, engine='openpyxl')
        
        print(f"🎉 변환 완료! 폴더를 확인해 보세요.")
        
    except Exception as e:
        print(f"❌ 변환 중 오류가 발생했습니다: {e}")

# =============================================================================
# 실행부 (이 부분을 수정해서 사용하세요)
# =============================================================================
if __name__ == "__main__":
    # 1. 확인할 데이터가 있는 폴더를 선택하세요.
    #    - 1분봉 데이터를 보려면: "1m_bars"
    #    - 최종 VPIN 결과를 보려면: "vpin_results"
    TARGET_FOLDER = "vpin_results" 
    
    # 2. 구조를 확인할 구체적인 파일명을 적어주세요.
    #    - 예시 (1분봉): "1m_bars_KOR_2021.parquet"
    #    - 예시 (VPIN) : "sym_005930.parquet"
    TARGET_FILE = "sym_005930.parquet" 
    
    # 3. 엑셀로 출력할 최대 행 수 (기본 1000줄)
    #    12년치 데이터 전체를 엑셀로 열면 컴퓨터가 멈출 수 있으니 가급적 유지해 주세요.
    MAX_ROWS = 1000 

    convert_parquet_to_excel(TARGET_FOLDER, TARGET_FILE, row_limit=MAX_ROWS)