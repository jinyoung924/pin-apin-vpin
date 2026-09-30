"""
=============================================================================
[VPIN 정밀 추적기] 파이썬 입력 시간 vs R 출력 시간 1:1 교차 검증
=============================================================================
목적: 1분봉 데이터(Python)에는 주말이 없는데, VPIN 결과(R)에는 주말이 생기는 
원인을 파악하기 위해, 특정 종목의 입력/출력 타임스탬프를 직접 대조합니다.
=============================================================================
"""

import os
import glob
import polars as pl
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY = "CA"
TARGET_SYMBOL = "A"  # 추적할 샘플 종목 (A, AAB 등 결과 리포트에 있던 종목)

# 파일 경로 자동 탐색
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
BAR_1M_PATH = os.path.join(OUTPUT_DIR, "all_1m_bars.parquet")

def find_latest_parquet(directory: str) -> str:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [f for f in sorted(glob.glob(pattern)) if "_sample" not in os.path.basename(f)]
    return files[-1] if files else ""

VPIN_RESULT_PATH = find_latest_parquet(OUTPUT_DIR)

# =============================================================================

def run_cross_check():
    print(f"\n{'='*65}")
    print(f" [정밀 추적] 종목 '{TARGET_SYMBOL}' 타임스탬프 변형 추적")
    print(f"{'='*65}")
    
    if not os.path.exists(BAR_1M_PATH) or not os.path.exists(VPIN_RESULT_PATH):
        print("❌ 1분봉 파일 또는 VPIN 결과 파일을 찾을 수 없습니다.")
        return

    # 1. 파이썬 입력 데이터 (1분봉) 로드
    df_in = pl.scan_parquet(BAR_1M_PATH).filter(pl.col("Symbol") == TARGET_SYMBOL).collect()
    
    # 2. R 출력 데이터 (VPIN 버킷) 로드
    df_out = pl.read_parquet(VPIN_RESULT_PATH).filter(pl.col("Symbol") == TARGET_SYMBOL)

    if df_in.is_empty() or df_out.is_empty():
        print(f"❌ 종목 {TARGET_SYMBOL}의 데이터를 찾을 수 없습니다.")
        return

    print(f"✅ 데이터 로드 완료")
    print(f"  - 입력 1분봉 행 수 : {df_in.height:,}행")
    print(f"  - 출력 VPIN 버킷 수: {df_out.height:,}행")

    # 3. 입력 데이터의 날짜 범위 확인 (절대 주말이 없어야 함)
    in_dates = df_in.with_columns(pl.col("Datetime").dt.date().alias("Date"))["Date"].unique().sort()
    out_dates = df_out["Date"].unique().sort()
    
    print(f"\n[날짜 범위 비교]")
    print(f"  입력 기간 (Python): {in_dates.min()} ~ {in_dates.max()} (총 {len(in_dates)}일)")
    print(f"  출력 기간 (R)     : {out_dates.min()} ~ {out_dates.max()} (총 {len(out_dates)}일)")

    # 4. 출력 결과에서 '주말'에 해당하는 이상한 행 추출
    weekend_out = df_out.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    
    if weekend_out.is_empty():
        print(f"\n✅ 종목 {TARGET_SYMBOL}에는 주말 버킷이 없습니다. 다른 종목으로 테스트해보세요.")
        return
        
    print(f"\n🚨 R 출력 결과에서 주말 버킷 {weekend_out.height:,}행 발견!")
    
    # 이상이 발생한 샘플 버킷 1개 선택
    sample_bucket = weekend_out.row(0, named=True)
    bucket_no = sample_bucket['BucketNo']
    r_start = sample_bucket['Starttime']
    r_end = sample_bucket['Endtime']
    r_date = sample_bucket['Date']
    
    print(f"\n{'='*65}")
    print(f" [단서 포착] BucketNo {bucket_no}의 시간이 오염되었습니다!")
    print(f"{'='*65}")
    print(f"  👉 R이 뱉어낸 시간:")
    print(f"     - Date     : {r_date} (요일: {r_date.weekday() + 1})")
    print(f"     - Starttime: {r_start}")
    print(f"     - Endtime  : {r_end}")
    
    print(f"\n  👉 이 시점에 Python이 R로 넘겨준 원본 1분봉 시간:")
    
    # R의 Starttime 주변의 원본 1분봉 데이터 탐색 (±3일)
    try:
        # Starttime과 Endtime 사이의 원본 데이터 검색
        matched_in = df_in.filter(
            (pl.col("Datetime") >= r_start) & 
            (pl.col("Datetime") <= r_end)
        )
        
        if matched_in.is_empty():
            print("     ⚠️ R이 뱉어낸 시간대에 해당하는 원본 데이터가 아예 존재하지 않습니다!!")
            print("        (결론: R 내부에서 아예 없는 가짜 시간표를 무에서 유로 창조해냈습니다.)")
            
            # 가장 가까운 원본 시간대 찾기
            print("\n     [참고] 이 종목의 진짜 원본 시작/끝 시간은 아래와 같습니다.")
            print(f"     - 첫 거래: {df_in['Datetime'].min()}")
            print(f"     - 끝 거래: {df_in['Datetime'].max()}")
        else:
            print("     [해당 시간대의 원본 데이터 5건]")
            print(matched_in.select(["Datetime", "Volume"]).head(5))
            print("\n     (결론: 원본 데이터의 날짜와 R의 날짜가 일치하지만, 주말이 섞여있습니다. 시간 변환 오프셋 에러입니다.)")
            
    except Exception as e:
        print(f"비교 중 에러: {e}")

if __name__ == "__main__":
    run_cross_check()