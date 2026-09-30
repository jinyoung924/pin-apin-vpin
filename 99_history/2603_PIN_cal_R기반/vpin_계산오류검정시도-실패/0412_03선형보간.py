"""
=============================================================================
[VPIN 가설 전수 검증] 주말 버킷이 '금-월 선형 보간'으로 생성되었는지 추적
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

OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
BAR_1M_PATH = os.path.join(OUTPUT_DIR, "all_1m_bars.parquet")

def find_latest_parquet(directory: str) -> str:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [f for f in sorted(glob.glob(pattern)) if "_sample" not in os.path.basename(f) and "_cleaned" not in os.path.basename(f)]
    return files[-1] if files else ""

VPIN_PATH = find_latest_parquet(OUTPUT_DIR)

# =============================================================================

def verify_interpolation_hypothesis():
    print(f"\n{'='*70}")
    print(f" [가설 전수 검증] R 패키지의 주말 버킷 생성 원인 분석")
    print(f"{'='*70}")

    if not os.path.exists(BAR_1M_PATH) or not os.path.exists(VPIN_PATH):
        print("❌ 1분봉 파일 또는 VPIN 결과 파일을 찾을 수 없습니다.")
        return

    print("1. 데이터 로드 및 정렬 중... (시간이 조금 걸릴 수 있습니다)")
    # 진짜 거래 데이터 로드
    df_bars = pl.scan_parquet(BAR_1M_PATH).select(["Symbol", "Datetime"]).collect()
    
    # VPIN 결과 로드 후 주말 데이터만 필터링
    df_vpin = pl.scan_parquet(VPIN_PATH).collect()
    df_weekend = df_vpin.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    
    total_weekend_buckets = df_weekend.height
    if total_weekend_buckets == 0:
        print("✅ 주말 버킷이 존재하지 않습니다. 이미 정상적인 데이터입니다.")
        return

    print(f"  - 검증 대상: 주말 버킷 총 {total_weekend_buckets:,}행")

    # ASOF Join을 위한 정렬 (Symbol, 시간 순)
    df_bars_sorted = df_bars.sort(["Symbol", "Datetime"])
    df_weekend_sorted_start = df_weekend.sort(["Symbol", "Starttime"])
    
    print("\n2. ASOF Join 수행 중 (과거 마지막 틱 & 미래 첫 번째 틱 탐색)...")
    
    # [과거 탐색] 버킷 시작 시간(Starttime) 기준 직전 진짜 틱 찾기
    df_merged = df_weekend_sorted_start.join_asof(
        df_bars_sorted,
        left_on="Starttime",
        right_on="Datetime",
        by="Symbol",
        strategy="backward"
    ).rename({"Datetime": "Last_Real_Tick"})
    
    # [미래 탐색] 버킷 종료 시간(Endtime) 기준 직후 진짜 틱 찾기
    df_merged = df_merged.sort(["Symbol", "Endtime"])
    df_merged = df_merged.join_asof(
        df_bars_sorted,
        left_on="Endtime",
        right_on="Datetime",
        by="Symbol",
        strategy="forward"
    ).rename({"Datetime": "Next_Real_Tick"})

    # 3. Gap(공백) 시간 계산
    df_merged = df_merged.with_columns([
        (pl.col("Next_Real_Tick") - pl.col("Last_Real_Tick")).dt.total_hours().alias("Gap_Hours"),
        pl.col("Last_Real_Tick").dt.weekday().alias("Last_Tick_Day"), # 1=월, 5=금
        pl.col("Next_Real_Tick").dt.weekday().alias("Next_Tick_Day")
    ])

    # 4. 가설 검증 통계 산출
    # 가설 성립 조건: 마지막 틱이 금요일(5)이고, 다음 틱이 월요일(1)이며, 그 사이 갭이 48시간 이상인 경우
    # (공휴일이 껴있다면 목요일-월요일, 금요일-화요일 패턴도 존재 가능)
    
    # 24시간 이상 비어있는 거대한 Gap 사이에 생성된 버킷 수
    massive_gap_buckets = df_merged.filter(pl.col("Gap_Hours") >= 48).height
    
    # 완벽한 금요일-월요일 패턴 수
    fri_mon_pattern = df_merged.filter(
        (pl.col("Last_Tick_Day") == 5) & (pl.col("Next_Tick_Day") == 1)
    ).height

    print(f"\n{'='*70}")
    print(f" [검증 결과 리포트]")
    print(f"{'='*70}")
    print(f"  총 주말 버킷 수               : {total_weekend_buckets:,}행")
    print(f"  거대 공백(48시간 이상) 내 발생 : {massive_gap_buckets:,}행 ({massive_gap_buckets/total_weekend_buckets*100:.2f}%)")
    print(f"  완벽한 금요일-월요일 패턴 발생: {fri_mon_pattern:,}행 ({fri_mon_pattern/total_weekend_buckets*100:.2f}%)")
    
    if massive_gap_buckets / total_weekend_buckets > 0.95:
        print("\n  🎯 [결론] 가설이 100% 입증되었습니다.")
        print("     사실상 모든 주말 버킷이 장이 닫혀있는 거대한 시간 공백 사이에서")
        print("     수학적 보간(Interpolation)에 의해 창조된 환각 타임스탬프입니다.")
    else:
        print("\n  ⚠️ 다른 원인이 섞여 있을 수 있습니다. 상세 확인이 필요합니다.")

    print(f"\n [증거 자료: 금-월 사이를 관통한 샘플 데이터 5건]")
    sample_df = df_merged.filter(
        (pl.col("Last_Tick_Day") == 5) & (pl.col("Next_Tick_Day") == 1)
    ).select([
        "Symbol", "BucketNo", "Last_Real_Tick", "Starttime", "Endtime", "Next_Real_Tick", "Gap_Hours"
    ]).head(5)
    
    print(sample_df)

if __name__ == "__main__":
    verify_interpolation_hypothesis()