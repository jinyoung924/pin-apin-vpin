"""
=============================================================================
[검증 05] 주말 VPIN의 앞뒤 평일 값 대비 선형보간 패턴 검증
=============================================================================
목적:
  주말 VPIN 값이 직전 금요일과 직후 월요일 VPIN의 선형보간으로 생성되었는지
  수학적으로 확인한다.

  선형보간이라면:
    VPIN_주말 ≈ VPIN_금 + ratio × (VPIN_월 - VPIN_금)
  여기서 ratio = (주말시간 - 금요일시간) / (월요일시간 - 금요일시간)

  → 04번 검증에서 "경우 1(보간 추정)"로 판정된 경우,
    이 스크립트로 보간의 정확한 수학적 형태를 추가 확인한다.
=============================================================================
"""

import os
import glob
import polars as pl
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(250)
pl.Config.set_tbl_rows(30)

# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY = "CA"

OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")

SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
REPORT_TXT = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_report.txt")


def find_latest_parquet(directory: str) -> str:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [
        f for f in sorted(glob.glob(pattern))
        if "_sample" not in os.path.basename(f)
        and "_cleaned" not in os.path.basename(f)
    ]
    return files[-1] if files else ""


VPIN_PATH = find_latest_parquet(OUTPUT_DIR)


def run_verification():
    report_lines = []

    def log(msg=""):
        text = str(msg)
        print(text)
        report_lines.append(text + "\n")

    run_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n{'='*90}")
    log(f" [검증 05] 주말 VPIN 선형보간 패턴 수학적 검증")
    log(f" 스크립트: {SCRIPT_NAME}.py  |  국가: {COUNTRY}  |  실행: {run_time}")
    log(f"{'='*90}")

    if not VPIN_PATH or not os.path.exists(VPIN_PATH):
        log(f"❌ VPIN 파일 없음: {VPIN_PATH or '(미탐지)'}")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    log(f"  ▶ VPIN 데이터: {VPIN_PATH}")

    # =================================================================
    # STEP 1: 로드 및 종목별 순서 정렬
    # =================================================================
    log(f"\n[STEP 1] 데이터 로드")
    df = pl.scan_parquet(VPIN_PATH).collect().sort(["Symbol", "Starttime"])
    log(f"  전체 행 수: {df.height:,}")

    vpin_candidates = ["VPIN", "adjVPIN", "vpin"]
    vpin_cols = [c for c in vpin_candidates if c in df.columns]
    if not vpin_cols:
        log("❌ VPIN 컬럼 없음")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    primary_vpin = vpin_cols[0]
    log(f"  분석 대상 VPIN 컬럼: {primary_vpin}")

    # =================================================================
    # STEP 2: 주말 플래그 & 직전/직후 평일 VPIN 연결
    # =================================================================
    log(f"\n[STEP 2] 주말 플래그 및 앞뒤 평일 VPIN 연결")

    df = df.with_columns([
        pl.col("Date").dt.weekday().alias("Weekday"),
        (pl.col("Date").dt.weekday().is_in([6, 7])).alias("Is_Weekend"),
    ])

    # 종목별로 직전/직후 행의 VPIN과 시간을 가져옴
    df = df.with_columns([
        pl.col(primary_vpin).shift(1).over("Symbol").alias("Prev_VPIN"),
        pl.col(primary_vpin).shift(-1).over("Symbol").alias("Next_VPIN"),
        pl.col("Starttime").shift(1).over("Symbol").alias("Prev_Time"),
        pl.col("Starttime").shift(-1).over("Symbol").alias("Next_Time"),
        pl.col("Date").dt.weekday().shift(1).over("Symbol").alias("Prev_Weekday"),
        pl.col("Date").dt.weekday().shift(-1).over("Symbol").alias("Next_Weekday"),
    ])

    # 주말 행만 필터
    df_we = df.filter(pl.col("Is_Weekend"))
    n_weekend = df_we.height
    log(f"  주말 버킷 수: {n_weekend:,}")

    if n_weekend == 0:
        log("  ✅ 주말 버킷 없음. 정상 데이터.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # =================================================================
    # STEP 3: 선형보간 추정값 계산 & 실제값과 비교
    # =================================================================
    log(f"\n[STEP 3] 선형보간 추정값 vs 실제값 비교")

    # 보간 비율 = (주말시간 - 이전시간) / (다음시간 - 이전시간)
    df_we = df_we.with_columns([
        ((pl.col("Starttime") - pl.col("Prev_Time")).dt.total_minutes().cast(pl.Float64)
         / (pl.col("Next_Time") - pl.col("Prev_Time")).dt.total_minutes().cast(pl.Float64)
        ).alias("Time_Ratio"),
    ])

    # 선형보간 추정값 = Prev_VPIN + ratio * (Next_VPIN - Prev_VPIN)
    df_we = df_we.with_columns([
        (pl.col("Prev_VPIN") + pl.col("Time_Ratio") * (pl.col("Next_VPIN") - pl.col("Prev_VPIN"))
        ).alias("Interpolated_VPIN"),
    ])

    # 오차 = 실제 - 추정
    df_we = df_we.with_columns([
        (pl.col(primary_vpin) - pl.col("Interpolated_VPIN")).alias("Error"),
        (pl.col(primary_vpin) - pl.col("Interpolated_VPIN")).abs().alias("Abs_Error"),
    ])

    # 유효한 행만 (null 제거)
    df_valid = df_we.filter(
        pl.col("Interpolated_VPIN").is_not_null()
        & pl.col("Interpolated_VPIN").is_finite()
        & pl.col(primary_vpin).is_not_null()
    )
    n_valid = df_valid.height
    log(f"  유효 비교 건수: {n_valid:,}")

    if n_valid == 0:
        log("  ⚠️ 유효 비교 건수가 0입니다.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # =================================================================
    # STEP 4: 오차 통계
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 4] 선형보간 오차 통계")
    log(f"{'='*90}")

    mean_err = df_valid["Error"].mean()
    mean_abs_err = df_valid["Abs_Error"].mean()
    max_abs_err = df_valid["Abs_Error"].max()
    median_abs_err = df_valid["Abs_Error"].median()
    std_err = df_valid["Error"].std()

    # "거의 완벽 일치" 판정 (절대오차 < 0.0001)
    near_perfect = df_valid.filter(pl.col("Abs_Error") < 0.0001).height
    near_perfect_rate = near_perfect / n_valid

    # 상대 오차 (VPIN 값 대비)
    df_valid = df_valid.with_columns([
        (pl.col("Abs_Error") / pl.col(primary_vpin).abs().clip(lower_bound=1e-10) * 100).alias("Rel_Error_Pct"),
    ])
    mean_rel_err = df_valid["Rel_Error_Pct"].mean()

    log(f"  평균 오차(Error)       : {mean_err:.8f}")
    log(f"  평균 절대오차          : {mean_abs_err:.8f}")
    log(f"  중앙 절대오차          : {median_abs_err:.8f}")
    log(f"  최대 절대오차          : {max_abs_err:.8f}")
    log(f"  오차 표준편차          : {std_err:.8f}")
    log(f"  평균 상대오차          : {mean_rel_err:.4f}%")
    log(f"  거의 완벽 일치(|e|<0.0001): {near_perfect:,}/{n_valid:,} ({near_perfect_rate*100:.1f}%)")

    # =================================================================
    # STEP 5: 판정
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 5] 선형보간 패턴 판정")
    log(f"{'='*90}")

    if near_perfect_rate > 0.90:
        log(f"  🎯 [판정] 주말 VPIN은 앞뒤 평일 값의 선형보간으로 생성된 것이 확실합니다.")
        log(f"     {near_perfect_rate*100:.1f}%의 주말 버킷이 선형보간 공식과 거의 완벽 일치합니다.")
        log(f"     → VPIN 계산 자체는 정상이며, 시간 라벨링만 보간으로 추정된 것")
        log(f"     → 조치: 주말/공휴일 행을 단순 삭제하면 됩니다 (재계산 불필요)")
    elif near_perfect_rate > 0.50:
        log(f"  ⚠️ [판정] 부분적 선형보간 패턴 ({near_perfect_rate*100:.1f}% 일치)")
        log(f"     일부는 선형보간이지만 나머지는 다른 메커니즘 가능성")
    else:
        log(f"  ❌ [판정] 선형보간 패턴 아님 ({near_perfect_rate*100:.1f}% 일치)")
        log(f"     → VPIN 값 자체가 주말 시간 구간에서 별도 계산되었을 가능성")
        log(f"     → 단순 삭제 전에 추가 분석 필요")

    # =================================================================
    # STEP 6: 오차 분포 히스토그램
    # =================================================================
    log(f"\n{'-'*90}")
    log(f" [분석] 절대오차 분포 히스토그램")
    log(f"{'-'*90}")

    thresholds = [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0]
    for t in thresholds:
        cnt = df_valid.filter(pl.col("Abs_Error") < t).height
        pct = cnt / n_valid * 100
        bar = "█" * int(pct / 2.5)
        log(f"  |e| < {t:<10} : {cnt:>8,}건 ({pct:5.1f}%) {bar}")

    # =================================================================
    # STEP 7: 샘플 데이터
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [샘플] 주말 VPIN vs 선형보간 추정값 (처음 15건)")
    log(f"{'='*90}")

    sample_cols = ["Symbol", "BucketNo", "Date", "Starttime",
                   primary_vpin, "Prev_VPIN", "Next_VPIN",
                   "Time_Ratio", "Interpolated_VPIN", "Error", "Abs_Error"]
    sample_cols = [c for c in sample_cols if c in df_valid.columns]
    log(df_valid.select(sample_cols).head(15))

    # 오차가 큰 이상치 샘플
    log(f"\n  ▶ 오차가 가장 큰 케이스 (top 10):")
    log(df_valid.sort("Abs_Error", descending=True).select(sample_cols).head(10))

    # =================================================================
    # 저장
    # =================================================================
    log(f"\n{'='*90}")
    log(f" 리포트 저장: {REPORT_TXT}")
    log(f"{'='*90}")

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.writelines(report_lines)
    print(f"  💾 저장 완료: {REPORT_TXT}")


if __name__ == "__main__":
    run_verification()