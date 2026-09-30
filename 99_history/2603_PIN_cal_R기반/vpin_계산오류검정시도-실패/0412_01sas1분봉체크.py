"""
=============================================================================
[VPIN 원인 추적] 원시 틱 데이터 vs 1분봉 데이터의 요일 분포 비교
=============================================================================
목적: "평일 데이터가 주말로 흩뿌려지는 현상"이 파이프라인의 어느 단계에서 
발생했는지 범인을 색출합니다.
=============================================================================
"""

import os
import glob
import polars as pl
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# ★ 사용자 설정 구역 (기존 경로에 맞게 수정해주세요)
# =============================================================================
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY = "CA"

# 확인할 샘플 원시 틱 데이터 폴더 (예: 2019년 1월 데이터 1개만 확인)
RAW_TICK_FOLDER = os.path.join(BASE_DIR, f"{COUNTRY}_2019")
# 1분봉 집계 파일
BAR_1M_PATH = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin", "all_1m_bars.parquet")

# =============================================================================

def trace_raw_ticks():
    print(f"\n{'='*60}")
    print(f"[1단계] 원시 틱(Raw Tick) 데이터 검증")
    print(f"{'='*60}")
    
    if not os.path.exists(RAW_TICK_FOLDER):
        print(f"❌ 폴더를 찾을 수 없습니다: {RAW_TICK_FOLDER}")
        return

    sample_files = glob.glob(os.path.join(RAW_TICK_FOLDER, "*.parquet"))
    if not sample_files:
        print(f"❌ 원시 parquet 파일을 찾을 수 없습니다.")
        return
        
    sample_file = sample_files[0]
    print(f"  📂 샘플 파일: {os.path.basename(sample_file)}")
    
    # 원시 데이터 스키마 및 샘플 로드
    df_raw = pl.read_parquet(sample_file)
    print(f"  📝 원시 스키마: {df_raw.schema}")
    
    print("\n  [원시 데이터 상위 5행 Date/Time 확인]")
    print(df_raw.select(["Symbol", "Date", "Time"]).head(5))

    # 원시 데이터 요일 분포 확인
    print("\n  [원시 데이터 요일 분포]")
    # Date 컬럼이 문자열인지, Date 타입인지에 따라 캐스팅
    if df_raw["Date"].dtype == pl.Utf8:
        df_raw = df_raw.with_columns(pl.col("Date").str.to_date(strict=False))
        
    weekday_dist = (
        df_raw.with_columns(pl.col("Date").dt.weekday().alias("Weekday"))
        .group_by("Weekday").agg(pl.len().alias("행수"))
        .sort("Weekday")
    )
    
    names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    has_weekend = False
    for r in weekday_dist.iter_rows(named=True):
        if r["Weekday"] in (6, 7):
            has_weekend = True
            print(f"    ❌ {names[r['Weekday']]}: {r['행수']:,}행 (주말 틱 존재!!)")
        else:
            print(f"    ✅ {names[r['Weekday']]}: {r['행수']:,}행")
            
    if has_weekend:
        print("\n  🚨 진단 결론: SAS -> Parquet 변환 과정에서 날짜가 이미 오염되었습니다!")
        print("      (원시 데이터 추출 스크립트의 날짜 파싱/Epoch 로직을 확인해야 합니다.)")
    else:
        print("\n  ✅ 원시 데이터는 정상(주말 없음)입니다. 파이썬 1분봉 집계에서 꼬였습니다.")

def trace_1m_bars():
    print(f"\n{'='*60}")
    print(f"[2단계] 1분봉(all_1m_bars.parquet) 데이터 검증")
    print(f"{'='*60}")
    
    if not os.path.exists(BAR_1M_PATH):
        print(f"❌ 1분봉 파일을 찾을 수 없습니다: {BAR_1M_PATH}")
        return
        
    df_bars = pl.scan_parquet(BAR_1M_PATH).select(["Symbol", "Datetime"]).collect()
    
    weekday_dist = (
        df_bars.with_columns(pl.col("Datetime").dt.weekday().alias("Weekday"))
        .group_by("Weekday").agg(pl.len().alias("행수"))
        .sort("Weekday")
    )
    
    names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    for r in weekday_dist.iter_rows(named=True):
        status = "❌" if r["Weekday"] in (6, 7) else "✅"
        print(f"  {status} {names[r['Weekday']]}: {r['행수']:,}행")

if __name__ == "__main__":
    trace_raw_ticks()
    trace_1m_bars()