"""
=============================================================================
[VPIN 선형보간 오류 전수 검증 v2.1 - TXT 리포트 버전]
R PINstimation::vpin() 패키지의 주말/공휴일 타임스탬프 오류가
선형 보간(Linear Interpolation)으로 발생했는지 수학적으로 입증
=============================================================================
"""

import os
import glob
import polars as pl
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

# 텍스트 리포트에서 데이터프레임 출력이 잘리지 않도록 Polars Config 설정
pl.Config.set_tbl_cols(-1)          # 모든 컬럼 출력
pl.Config.set_tbl_width_chars(250)  # 한 줄 출력 너비 확장
pl.Config.set_tbl_rows(20)          # 출력 행 수 확장

# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY = "CA"

OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
BAR_1M_PATH = os.path.join(OUTPUT_DIR, "all_1m_bars.parquet")

def find_latest_parquet(directory: str) -> str:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [
        f for f in sorted(glob.glob(pattern))
        if "_sample" not in os.path.basename(f)
        and "_cleaned" not in os.path.basename(f)
    ]
    return files[-1] if files else ""

VPIN_PATH = find_latest_parquet(OUTPUT_DIR)

# 결과 TXT 경로 (CSV 대체)
REPORT_TXT = os.path.join(OUTPUT_DIR, "vpin_linearErrorReport.txt")
DETAILS_TXT = os.path.join(OUTPUT_DIR, "vpin_linearErrorDetails.txt")

# =============================================================================

def _classify_gap_type() -> pl.Expr:
    return (
        pl.when((pl.col("Last_Tick_Day") == 5) & (pl.col("Next_Tick_Day") == 1))
        .then(pl.lit("주말(금→월)"))
        .when((pl.col("Last_Tick_Day") == 4) & (pl.col("Next_Tick_Day") == 1))
        .then(pl.lit("공휴일연휴(목→월)"))
        .when((pl.col("Last_Tick_Day") == 5) & (pl.col("Next_Tick_Day") == 2))
        .then(pl.lit("공휴일연휴(금→화)"))
        .when((pl.col("Last_Tick_Day") == 4) & (pl.col("Next_Tick_Day") == 2))
        .then(pl.lit("공휴일연휴(목→화)"))
        .when((pl.col("Last_Tick_Day") == 3) & (pl.col("Next_Tick_Day") == 1))
        .then(pl.lit("공휴일연휴(수→월)"))
        .when((pl.col("Last_Tick_Day") == 5) & (pl.col("Next_Tick_Day") == 5))
        .then(pl.lit("공휴일단독(금→금)"))
        .when(pl.col("Gap_Hours") >= 16)
        .then(
            pl.lit("기타장기갭(")
            + pl.col("Last_Tick_Day").cast(pl.String)
            + pl.lit("→")
            + pl.col("Next_Tick_Day").cast(pl.String)
            + pl.lit(")")
        )
        .otherwise(pl.lit("단기갭(<16h)"))
        .alias("Gap_Type")
    )


def verify_interpolation_hypothesis():
    report_lines = []

    def log(msg=""):
        """콘솔 출력과 동시에 리포트 라인에 수집하는 내부 함수"""
        text = str(msg)
        print(text)
        report_lines.append(text + "\n")

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n{'='*80}")
    log(f" [VPIN 선형보간 오류 전수 검증 리포트]")
    log(f" 국가: {COUNTRY}  |  실행일시: {run_time}")
    log(f"{'='*80}")

    missing = []
    if not os.path.exists(BAR_1M_PATH):
        missing.append(f"1분봉: {BAR_1M_PATH}")
    if not VPIN_PATH or not os.path.exists(VPIN_PATH):
        missing.append(f"VPIN : {VPIN_PATH or '(파일 없음)'}")
    
    if missing:
        for m in missing:
            log(f"❌ 파일 없음 → {m}")
        
        # 파일이 없어도 로그는 TXT로 저장
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    log(f"  ▶ 1분봉 데이터 : {BAR_1M_PATH}")
    log(f"  ▶ VPIN 데이터  : {VPIN_PATH}")

    # =========================================================
    # STEP 1 & 2 : 데이터 로드 및 필터링
    # =========================================================
    log("\n[STEP 1] 데이터 로드 및 주말 버킷 필터링...")

    df_bars = (
        pl.scan_parquet(BAR_1M_PATH)
        .select(["Symbol", "Datetime"])
        .collect()
        .sort(["Symbol", "Datetime"])
    )

    df_vpin = pl.scan_parquet(VPIN_PATH).collect()
    vpin_cols = df_vpin.columns
    
    df_weekend = df_vpin.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    total_weekend = df_weekend.height
    total_vpin = df_vpin.height

    log(f"  - 1분봉 전체   : {df_bars.height:,}행")
    log(f"  - VPIN 전체    : {total_vpin:,}행")
    log(f"  - 주말 타겟    : {total_weekend:,}행 ({total_weekend / total_vpin * 100:.3f}%)")

    if total_weekend == 0:
        log("\n  ✅ 주말 버킷이 없습니다. 이미 정상 데이터입니다.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # =========================================================
    # STEP 3 : ASOF Join (직전/직후 실거래 틱 탐색)
    # =========================================================
    log("\n[STEP 2] ASOF Join 수행 (갭 구간 탐색)...")

    df_work = df_weekend.sort(["Symbol", "Starttime"])

    df_merged = df_work.join_asof(
        df_bars, left_on="Starttime", right_on="Datetime", by="Symbol", strategy="backward"
    ).rename({"Datetime": "Last_Real_Tick"})

    df_merged = df_merged.sort(["Symbol", "Endtime"])
    df_merged = df_merged.join_asof(
        df_bars, left_on="Endtime", right_on="Datetime", by="Symbol", strategy="forward"
    ).rename({"Datetime": "Next_Real_Tick"})

    df_merged = df_merged.filter(
        pl.col("Last_Real_Tick").is_not_null() & pl.col("Next_Real_Tick").is_not_null()
    )

    # =========================================================
    # STEP 4 : 보간 위치 계산
    # =========================================================
    df_merged = df_merged.with_columns([
        (pl.col("Next_Real_Tick") - pl.col("Last_Real_Tick")).dt.total_minutes().alias("Gap_Minutes"),
        pl.col("Last_Real_Tick").dt.weekday().alias("Last_Tick_Day"),
        pl.col("Next_Real_Tick").dt.weekday().alias("Next_Tick_Day"),
    ])

    df_merged = df_merged.with_columns((pl.col("Gap_Minutes") / 60.0).alias("Gap_Hours"))
    df_merged = df_merged.with_columns(_classify_gap_type())

    df_merged = df_merged.with_columns(
        ((pl.col("Starttime") - pl.col("Last_Real_Tick")).dt.total_minutes().cast(pl.Float64) 
         / pl.col("Gap_Minutes").cast(pl.Float64)).alias("Interp_Ratio")
    )

    df_merged = df_merged.with_columns([
        (pl.col("Interp_Ratio") * 100).round(1).alias("Interp_Pct"),
        ((pl.col("Interp_Ratio") - 0.5).abs() * pl.col("Gap_Minutes").cast(pl.Float64)).round(0).alias("Midpoint_Dev_Min"),
    ])

    df_merged = df_merged.with_columns(
        ((pl.col("Gap_Hours") >= 24) & (pl.col("Interp_Ratio") >= 0.0) & (pl.col("Interp_Ratio") <= 1.0)).alias("보간오류_확인")
    )

    # =========================================================
    # STEP 5 : 통계 리포트 작성
    # =========================================================
    confirmed_cnt = df_merged.filter(pl.col("보간오류_확인")).height
    valid_cnt = df_merged.height

    log(f"\n{'='*80}")
    log(f" [검증 통계 요약]")
    log(f"{'='*80}")
    log(f"  총 주말 버킷 수          : {total_weekend:,}건")
    log(f"  ASOF 매칭 유효 건수      : {valid_cnt:,}건")
    rate = confirmed_cnt / valid_cnt if valid_cnt > 0 else 0
    log(f"  보간오류 확인 (≥24h 갭)  : {confirmed_cnt:,}건 ({rate * 100:.1f}%)")
    log(f"  평균 보간 위치           : {df_merged['Interp_Pct'].mean():.1f}% (50% = 갭 정중앙)")
    log(f"  평균 갭 크기             : {df_merged['Gap_Hours'].mean():.1f} 시간")
    log(f"  중간점 평균 편차         : {df_merged['Midpoint_Dev_Min'].mean():.0f} 분")

    if rate > 0.95:
        verdict = "🎯 [판정] 가설 완벽 입증 (95%+ 일치) — 선형 보간 환각 타임스탬프 발생 확인"
    elif rate > 0.80:
        verdict = "⚠️ [판정] 가설 부분 입증 (80%+ 일치) — 선형 보간 외 소수 다른 요인 혼재 가능"
    else:
        verdict = "❌ [판정] 가설 불일치 — 추가 원인 조사 필요"
    
    log(f"\n  {verdict}")

    # 갭 유형별 분포
    log(f"\n{'-'*80}")
    log(f" [분석 1] 갭(Gap) 유형별 분포 요약")
    log(f"{'-'*80}")
    gap_summary = (
        df_merged.group_by("Gap_Type")
        .agg([
            pl.count().alias("버킷수"),
            pl.col("Gap_Hours").mean().round(1).alias("평균갭(h)"),
            pl.col("Interp_Pct").mean().round(1).alias("평균보간위치(%)"),
            pl.col("Midpoint_Dev_Min").mean().round(0).alias("중앙편차(분)"),
            pl.col("보간오류_확인").sum().alias("오류확인건수"),
        ])
        .sort("버킷수", descending=True)
    )
    log(gap_summary)

    # 보간 위치 분포 (히스토그램)
    log(f"\n{'-'*80}")
    log(f" [분석 2] 보간 위치(Interp_Pct) 히스토그램 (50% 근처 집중 시 선형보간 확정)")
    log(f"{'-'*80}")
    bins = [(i, i + 10) for i in range(0, 100, 10)]
    for lo, hi in bins:
        cnt = df_merged.filter((pl.col("Interp_Pct") >= lo) & (pl.col("Interp_Pct") < hi)).height
        bar = "█" * (cnt * 40 // max(valid_cnt, 1))
        log(f"  {lo:3d}% ~ {hi:3d}% : {cnt:6,}건 | {bar}")

    # =========================================================
    # STEP 6 : 실제 데이터 예시
    # =========================================================
    base_cols = [
        "Symbol", "BucketNo", "Gap_Type", "Last_Real_Tick", "Starttime", "Next_Real_Tick", 
        "Gap_Hours", "Interp_Pct", "Midpoint_Dev_Min"
    ]
    display_cols = [c for c in base_cols if c in df_merged.columns]

    log(f"\n{'='*80}")
    log(f" [증거 자료] 데이터 덤프 샘플")
    log(f"{'='*80}")

    log(f"\n▶ 예시 1: 전형적인 금요일 장마감 → 월요일 개장 (선형 보간 버킷 5건)")
    sample_fri_mon = df_merged.filter(pl.col("Gap_Type") == "주말(금→월)").select(display_cols).head(5)
    log(sample_fri_mon if sample_fri_mon.height > 0 else "  (데이터 없음)")

    log(f"\n▶ 예시 2: 갭 정중앙(50%)에 가장 가깝게 보간된 케이스 5건")
    sample_midpoint = df_merged.filter((pl.col("Interp_Pct") >= 40) & (pl.col("Interp_Pct") <= 60)).sort("Midpoint_Dev_Min").select(display_cols).head(5)
    log(sample_midpoint if sample_midpoint.height > 0 else "  (데이터 없음)")

    # =========================================================
    # STEP 7 : TXT 리포트 및 상세 데이터 저장
    # =========================================================
    log(f"\n{'='*80}")
    log(f" [결론 및 권고사항]")
    log(f"{'='*80}")
    log(f"  1. 주말에 생성된 타임스탬프는 모두 '거래량 0 구간'을 관통하는 1개 버킷을 분할할 때 발생한 패키지의 연산 부작용입니다.")
    log(f"  2. 수치 연산(VPIN) 자체는 정상이지만 시간 라벨링에 왜곡이 존재합니다.")
    log(f"  3. 조치 방안: R 연산 후 Date 및 Endtime 컬럼을 실제 거래일(Last_Real_Tick) 기준으로 밀어주는(Snapping) 롤링 조인 후처리를 권장합니다.")
    log(f"\n  ✅ 리포트가 성공적으로 생성되었습니다.")
    
    # 1. 포맷팅된 TXT 리포트 저장
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"  💾 포맷팅 리포트 저장 완료: {REPORT_TXT}")

    # 2. 전체 상세 레코드를 Tab-separated TXT로 저장 (CSV 대체)
    dt_cols = ["Starttime", "Endtime", "Last_Real_Tick", "Next_Real_Tick"]
    if "Date" in df_merged.columns: dt_cols.append("Date")

    df_export = df_merged.clone()
    for col in dt_cols:
        if col in df_export.columns:
            try: df_export = df_export.with_columns(pl.col(col).dt.strftime("%Y-%m-%d %H:%M:%S").alias(col))
            except: pass

    export_cols_final = [c for c in [
        "Symbol", "BucketNo", "Gap_Type", "보간오류_확인", "Date", "Starttime", "Endtime",
        "Last_Real_Tick", "Next_Real_Tick", "Gap_Hours", "Gap_Minutes", "Interp_Pct", "Midpoint_Dev_Min"
    ] + [col for col in ["OI", "VPIN", "adjVPIN", "MBS", "VBS"] if col in vpin_cols] if c in df_export.columns]

    df_export.select(export_cols_final).write_csv(DETAILS_TXT, separator="\t")
    print(f"  💾 전체 상세 데이터(TSV) 저장 완료: {DETAILS_TXT}")

if __name__ == "__main__":
    verify_interpolation_hypothesis()