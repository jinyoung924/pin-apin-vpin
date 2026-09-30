"""
=============================================================================
[검증 06] 주말 행 제거 전후 데이터 무결성 & 연속성 최종 검증
=============================================================================
목적:
  04번(거래량/VPIN 값 확인), 05번(선형보간 패턴 확인) 검증 이후,
  실제로 주말 행을 삭제해도 데이터가 무결한지 최종 확인한다.

검증 항목:
  1. 종목별 거래량 합산: 전체 vs 주말제거 후 → 차이 비율
  2. 종목별 버킷 번호 연속성: 주말 제거 후 BucketNo 갭 발생 여부
  3. VPIN 시계열 연속성: 주말 제거 전후로 비정상 점프 여부
  4. 최종 판정: 단순 삭제(방법 B) 가능 vs 재계산(방법 A) 필요
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
    log(f" [검증 06] 주말 행 제거 전후 데이터 무결성 & 연속성 최종 검증")
    log(f" 스크립트: {SCRIPT_NAME}.py  |  국가: {COUNTRY}  |  실행: {run_time}")
    log(f"{'='*90}")

    if not VPIN_PATH or not os.path.exists(VPIN_PATH):
        log(f"❌ VPIN 파일 없음: {VPIN_PATH or '(미탐지)'}")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    log(f"  ▶ VPIN 데이터: {VPIN_PATH}")

    # =================================================================
    # STEP 1: 로드
    # =================================================================
    log(f"\n[STEP 1] 데이터 로드")
    df = pl.scan_parquet(VPIN_PATH).collect().sort(["Symbol", "Starttime"])

    df = df.with_columns([
        pl.col("Date").dt.weekday().alias("Weekday"),
        (pl.col("Date").dt.weekday().is_in([6, 7])).alias("Is_Weekend"),
    ])

    df_all = df
    df_clean = df.filter(~pl.col("Is_Weekend"))
    df_weekend = df.filter(pl.col("Is_Weekend"))

    log(f"  전체: {df_all.height:,}  |  평일: {df_clean.height:,}  |  주말: {df_weekend.height:,}")

    if df_weekend.height == 0:
        log("  ✅ 주말 버킷 없음.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # --- 컬럼 탐지 ---
    vol_candidates = ["VBS", "OI", "aVBS", "MBS", "Volume", "volume", "tbuy", "tsell"]
    vpin_candidates = ["VPIN", "adjVPIN", "vpin"]
    vol_cols = [c for c in vol_candidates if c in df.columns]
    vpin_cols = [c for c in vpin_candidates if c in df.columns]
    primary_vpin = vpin_cols[0] if vpin_cols else None

    # =================================================================
    # STEP 2: 종목별 거래량 합산 비교
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 2] 종목별 거래량 합산 비교 (전체 vs 주말제거)")
    log(f"{'='*90}")

    if vol_cols:
        pv = vol_cols[0]
        agg_all = df_all.group_by("Symbol").agg(pl.col(pv).sum().alias("Total_All"))
        agg_clean = df_clean.group_by("Symbol").agg(pl.col(pv).sum().alias("Total_Clean"))
        agg_we = df_weekend.group_by("Symbol").agg(pl.col(pv).sum().alias("Total_Weekend"))

        comp = agg_all.join(agg_clean, on="Symbol").join(agg_we, on="Symbol", how="left")
        comp = comp.with_columns([
            (pl.col("Total_Weekend").fill_null(0) / pl.col("Total_All").clip(lower_bound=1) * 100)
            .round(4).alias("Weekend_Pct"),
        ])

        log(f"\n  거래량 컬럼: {pv}")
        log(comp.sort("Symbol"))

        max_pct = comp["Weekend_Pct"].max()
        mean_pct = comp["Weekend_Pct"].mean()
        log(f"\n  주말 거래량 비중 — 평균: {mean_pct:.4f}%  최대: {max_pct:.4f}%")

        if max_pct < 1.0:
            log(f"  ✅ 모든 종목에서 주말 거래량 비중 < 1% → 삭제 시 손실 무시 가능")
        else:
            log(f"  ⚠️ 일부 종목 주말 거래량 비중 > 1% → 삭제 시 데이터 손실 주의")
    else:
        log(f"  (거래량 컬럼 미탐지 — 생략)")

    # =================================================================
    # STEP 3: 버킷 번호 연속성 검증
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 3] 버킷 번호(BucketNo) 연속성 검증")
    log(f"{'='*90}")

    if "BucketNo" in df.columns:
        # 전체 데이터에서 종목별 BucketNo 갭 체크
        df_all_bn = df_all.sort(["Symbol", "BucketNo"]).with_columns([
            (pl.col("BucketNo") - pl.col("BucketNo").shift(1).over("Symbol")).alias("BN_Diff"),
        ])
        gaps_all = df_all_bn.filter(pl.col("BN_Diff") > 1)
        log(f"  전체 데이터 BucketNo 갭 수: {gaps_all.height:,}")

        # 주말 제거 후 BucketNo 갭 체크
        df_clean_bn = df_clean.sort(["Symbol", "BucketNo"]).with_columns([
            (pl.col("BucketNo") - pl.col("BucketNo").shift(1).over("Symbol")).alias("BN_Diff"),
        ])
        gaps_clean = df_clean_bn.filter(pl.col("BN_Diff") > 1)
        log(f"  주말제거 후 BucketNo 갭 수: {gaps_clean.height:,}")

        new_gaps = gaps_clean.height - gaps_all.height
        log(f"  주말 제거로 인한 추가 갭: {new_gaps:,}건")

        if new_gaps > 0:
            log(f"  → 주말 버킷이 BucketNo 시퀀스 중간에 있었음 (보간 삽입 확인)")
            log(f"\n  ▶ 추가 갭 샘플 (처음 10건):")
            gap_sample_cols = ["Symbol", "BucketNo", "BN_Diff", "Date", "Starttime"]
            gap_sample_cols = [c for c in gap_sample_cols if c in df_clean_bn.columns]
            log(df_clean_bn.filter(pl.col("BN_Diff") > 1).select(gap_sample_cols).head(10))
        else:
            log(f"  ✅ 주말 제거로 인한 추가 갭 없음 → 버킷 시퀀스에 영향 없음")
    else:
        log(f"  (BucketNo 컬럼 없음 — 생략)")

    # =================================================================
    # STEP 4: VPIN 시계열 점프 검증
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 4] VPIN 시계열 연속성 — 주말 제거 전후 점프 비교")
    log(f"{'='*90}")

    if primary_vpin:
        # 전체에서 연속 VPIN 차이
        df_all_ts = df_all.sort(["Symbol", "Starttime"]).with_columns([
            (pl.col(primary_vpin) - pl.col(primary_vpin).shift(1).over("Symbol")).abs().alias("VPIN_Jump"),
        ])
        all_mean_jump = df_all_ts["VPIN_Jump"].mean()
        all_max_jump = df_all_ts["VPIN_Jump"].max()
        all_std_jump = df_all_ts["VPIN_Jump"].std()

        # 주말 제거 후 연속 VPIN 차이
        df_clean_ts = df_clean.sort(["Symbol", "Starttime"]).with_columns([
            (pl.col(primary_vpin) - pl.col(primary_vpin).shift(1).over("Symbol")).abs().alias("VPIN_Jump"),
        ])
        clean_mean_jump = df_clean_ts["VPIN_Jump"].mean()
        clean_max_jump = df_clean_ts["VPIN_Jump"].max()
        clean_std_jump = df_clean_ts["VPIN_Jump"].std()

        log(f"  VPIN 컬럼: {primary_vpin}")
        log(f"")
        log(f"  {'항목':<25s} {'전체(주말포함)':<20s} {'주말제거후':<20s}")
        log(f"  {'-'*65}")
        log(f"  {'평균 점프':<25s} {all_mean_jump:.8f}       {clean_mean_jump:.8f}")
        log(f"  {'최대 점프':<25s} {all_max_jump:.8f}       {clean_max_jump:.8f}")
        log(f"  {'점프 표준편차':<25s} {all_std_jump:.8f}       {clean_std_jump:.8f}")

        jump_ratio = clean_mean_jump / all_mean_jump if all_mean_jump else 0
        log(f"\n  주말제거 후/전 평균 점프 비율: {jump_ratio:.4f}")

        if jump_ratio < 1.5:
            log(f"  ✅ 주말 제거 후에도 VPIN 점프 크기가 유사 → 시계열 연속성 유지")
            log(f"     (보간된 주말 값을 제거해도 금→월 전이가 자연스러움)")
        else:
            log(f"  ⚠️ 주말 제거 후 점프 크기 증가 → 금→월 사이 불연속 가능성")
            log(f"     → 주말 버킷에 실거래 정보가 포함되어 있을 수 있음")

        # 금→월 전이 구간만 별도 분석
        log(f"\n  ── 금요일→월요일 전이 구간 점프 분석 ──")
        df_clean_ts2 = df_clean_ts.with_columns([
            pl.col("Date").dt.weekday().alias("WD"),
            pl.col("Date").dt.weekday().shift(1).over("Symbol").alias("Prev_WD"),
        ])
        # 금→월 전이 = 현재 월요일(1)이고 이전이 금요일(5)
        fri_to_mon = df_clean_ts2.filter(
            (pl.col("WD") == 1) & (pl.col("Prev_WD") == 5)
        )
        if fri_to_mon.height > 0:
            ftm_mean = fri_to_mon["VPIN_Jump"].mean()
            ftm_max = fri_to_mon["VPIN_Jump"].max()
            # 같은 요일 내 전이(예: 월→월)
            same_day = df_clean_ts2.filter(pl.col("WD") == pl.col("Prev_WD"))
            sd_mean = same_day["VPIN_Jump"].mean() if same_day.height > 0 else 0

            log(f"  금→월 전이 건수     : {fri_to_mon.height:,}")
            log(f"  금→월 평균 점프     : {ftm_mean:.8f}")
            log(f"  금→월 최대 점프     : {ftm_max:.8f}")
            log(f"  같은요일 평균 점프  : {sd_mean:.8f}")
            if sd_mean and ftm_mean:
                log(f"  금→월 / 같은요일 비율: {ftm_mean/sd_mean:.2f}배")
        else:
            log(f"  (금→월 전이 건수 0건)")
    else:
        log(f"  (VPIN 컬럼 미탐지 — 생략)")

    # =================================================================
    # STEP 5: 최종 종합 판정
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 5] ★ 최종 종합 판정 ★")
    log(f"{'='*90}")
    log(f"")
    log(f"  본 검증(06)은 04번, 05번 검증 결과와 함께 종합 판단합니다.")
    log(f"")
    log(f"  ┌─────────────────────────────────────────────────────────────────┐")
    log(f"  │ 04번: 주말 VPIN ≠ 0 + 거래량 = 0 → 경우 1(보간 추정)         │")
    log(f"  │ 05번: 앞뒤 평일 값의 선형보간과 일치 → 보간 확정              │")
    log(f"  │ 06번: 삭제 전후 합산/연속성 유지 → 단순 삭제 가능             │")
    log(f"  │                                                               │")
    log(f"  │ → 위 3개 모두 충족 시: 방법 B (주말 행 삭제) 적용 가능        │")
    log(f"  │ → 하나라도 미충족 시: 방법 A (로직 수정 후 재계산) 필요       │")
    log(f"  └─────────────────────────────────────────────────────────────────┘")
    log(f"")
    log(f"  ※ 각 검증 스크립트의 판정 결과를 확인한 뒤 최종 결정하세요.")
    log(f"  ※ 교수님 보고 시: '경우 1(보간 추정)임을 확인했고,")
    log(f"    주말 행 삭제로 처리 가능합니다' 또는")
    log(f"    '경우 2(시간구간 할당 오류)로 확인되어 재계산이 필요합니다'")
    log(f"    형태로 보고하시면 됩니다.")

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