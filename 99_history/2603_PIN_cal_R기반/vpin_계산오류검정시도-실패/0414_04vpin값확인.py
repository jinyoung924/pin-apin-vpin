"""
=============================================================================
[검증 04] 주말 타임스탬프의 VPIN 값 및 거래량 검증
=============================================================================
목적:
  교수님 질문 핵심 — "보간법으로 VPIN 값을 추정한 것인가,
  아니면 주말에 시간 구간을 할당해서 VPIN을 계산한 것인가?"

판별 로직:
  (A) 주말 버킷의 거래량(VBS, OI 등)이 0인데 VPIN ≠ 0 → 보간으로 값 자체를 생성
  (B) 주말 버킷의 거래량이 0이고 VPIN도 0 → 시간만 할당 (교수님: 이러면 프로그램 정상)
  (C) 주말 버킷의 거래량이 0이 아님 → 실거래가 주말로 잘못 배분 (프로그램 오류)
=============================================================================
"""

import os
import sys
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
BAR_1M_PATH = os.path.join(OUTPUT_DIR, "all_1m_bars.parquet")

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
    log(f" [검증 04] 주말 VPIN 값 & 거래량 검증 — 보간 추정 vs 시간구간 할당 판별")
    log(f" 스크립트: {SCRIPT_NAME}.py  |  국가: {COUNTRY}  |  실행: {run_time}")
    log(f"{'='*90}")

    # --- 파일 확인 ---
    if not VPIN_PATH or not os.path.exists(VPIN_PATH):
        log(f"❌ VPIN 파일 없음: {VPIN_PATH or '(미탐지)'}")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    log(f"  ▶ VPIN 데이터: {VPIN_PATH}")

    # =================================================================
    # STEP 1: 데이터 로드 및 컬럼 확인
    # =================================================================
    log(f"\n[STEP 1] 데이터 로드 및 컬럼 구조 파악")
    df = pl.scan_parquet(VPIN_PATH).collect()
    log(f"  전체 행 수: {df.height:,}")
    log(f"  컬럼 목록: {df.columns}")
    log(f"  컬럼 타입:")
    for col_name, dtype in zip(df.columns, df.dtypes):
        log(f"    {col_name:20s} : {dtype}")

    # --- 거래량 관련 컬럼 자동 탐지 ---
    volume_candidates = ["VBS", "OI", "aVBS", "MBS", "Volume", "volume",
                         "tbuy", "tsell", "buy", "sell", "Tbuy", "Tsell"]
    vol_cols = [c for c in volume_candidates if c in df.columns]

    vpin_candidates = ["VPIN", "adjVPIN", "vpin"]
    vpin_cols = [c for c in vpin_candidates if c in df.columns]

    log(f"\n  탐지된 거래량 컬럼: {vol_cols}")
    log(f"  탐지된 VPIN 컬럼 : {vpin_cols}")

    if not vpin_cols:
        log("❌ VPIN 컬럼을 찾을 수 없습니다. 컬럼명을 확인하세요.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # =================================================================
    # STEP 2: 주말 버킷 필터링
    # =================================================================
    log(f"\n[STEP 2] 주말 버킷 필터링 (weekday 6=토, 7=일)")
    df_weekend = df.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    df_weekday = df.filter(~pl.col("Date").dt.weekday().is_in([6, 7]))

    n_weekend = df_weekend.height
    n_weekday = df_weekday.height
    log(f"  주말 버킷: {n_weekend:,}건")
    log(f"  평일 버킷: {n_weekday:,}건")

    if n_weekend == 0:
        log("\n  ✅ 주말 버킷이 없습니다. 데이터 정상.")
        with open(REPORT_TXT, "w", encoding="utf-8") as f:
            f.writelines(report_lines)
        return

    # =================================================================
    # STEP 3: 핵심 판별 — 주말 버킷의 거래량과 VPIN 값 분석
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 3] ★ 핵심 판별: 주말 버킷의 거래량 & VPIN 값 패턴 ★")
    log(f"{'='*90}")

    # --- 3-1: 거래량 컬럼별 통계 ---
    log(f"\n  ── 3-1. 주말 버킷의 거래량 컬럼별 통계 ──")
    for vc in vol_cols:
        col_data = df_weekend[vc]
        zero_cnt = df_weekend.filter(pl.col(vc) == 0).height
        nonzero_cnt = n_weekend - zero_cnt
        log(f"\n    [{vc}]")
        log(f"      총 건수     : {n_weekend:,}")
        log(f"      0인 건수    : {zero_cnt:,} ({zero_cnt/n_weekend*100:.1f}%)")
        log(f"      0 아닌 건수 : {nonzero_cnt:,} ({nonzero_cnt/n_weekend*100:.1f}%)")
        log(f"      평균        : {col_data.mean():.6f}")
        log(f"      최솟값      : {col_data.min()}")
        log(f"      최댓값      : {col_data.max()}")

    # --- 3-2: VPIN 컬럼별 통계 ---
    log(f"\n  ── 3-2. 주말 버킷의 VPIN 값 통계 ──")
    for vc in vpin_cols:
        col_data = df_weekend[vc]
        zero_cnt = df_weekend.filter(pl.col(vc) == 0).height
        nonzero_cnt = n_weekend - zero_cnt
        null_cnt = col_data.null_count()
        log(f"\n    [{vc}]")
        log(f"      총 건수     : {n_weekend:,}")
        log(f"      0인 건수    : {zero_cnt:,} ({zero_cnt/n_weekend*100:.1f}%)")
        log(f"      0 아닌 건수 : {nonzero_cnt:,} ({nonzero_cnt/n_weekend*100:.1f}%)")
        log(f"      NULL 건수   : {null_cnt:,}")
        log(f"      평균        : {col_data.mean():.6f}")
        log(f"      중앙값      : {col_data.median():.6f}")
        log(f"      최솟값      : {col_data.min()}")
        log(f"      최댓값      : {col_data.max()}")

    # --- 3-3: 평일 VPIN과 비교 ---
    log(f"\n  ── 3-3. 평일 vs 주말 VPIN 비교 ──")
    for vc in vpin_cols:
        wd_mean = df_weekday[vc].mean()
        we_mean = df_weekend[vc].mean()
        wd_std = df_weekday[vc].std()
        we_std = df_weekend[vc].std()
        log(f"\n    [{vc}]")
        log(f"      평일 평균: {wd_mean:.6f}  (std: {wd_std:.6f})")
        log(f"      주말 평균: {we_mean:.6f}  (std: {we_std:.6f})")
        if wd_mean and we_mean:
            ratio = we_mean / wd_mean
            log(f"      주말/평일 비율: {ratio:.4f}")
            if abs(ratio - 1.0) < 0.05:
                log(f"      → 주말 VPIN이 평일과 거의 동일 → 보간 추정 가능성 높음")
            elif we_mean < 0.001:
                log(f"      → 주말 VPIN ≈ 0 → 시간만 할당된 경우 (교수님: 정상 작동)")

    # =================================================================
    # STEP 4: 경우 판정
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 4] ★ 최종 판정 ★")
    log(f"{'='*90}")

    # 주말 VPIN이 0인 비율
    primary_vpin = vpin_cols[0]
    vpin_zero_cnt = df_weekend.filter(pl.col(primary_vpin) == 0).height
    vpin_nonzero_cnt = n_weekend - vpin_zero_cnt
    vpin_zero_rate = vpin_zero_cnt / n_weekend

    # 주말 거래량이 0인 비율 (탐지된 첫 번째 거래량 컬럼 기준)
    if vol_cols:
        primary_vol = vol_cols[0]
        vol_zero_cnt = df_weekend.filter(pl.col(primary_vol) == 0).height
        vol_nonzero_cnt = n_weekend - vol_zero_cnt
        vol_zero_rate = vol_zero_cnt / n_weekend
    else:
        vol_zero_cnt = -1
        vol_nonzero_cnt = -1
        vol_zero_rate = -1

    log(f"\n  기준 VPIN 컬럼: {primary_vpin}")
    log(f"  기준 거래량 컬럼: {vol_cols[0] if vol_cols else '(없음)'}")
    log(f"  주말 VPIN=0 비율: {vpin_zero_rate*100:.1f}%")
    if vol_cols:
        log(f"  주말 거래량=0 비율: {vol_zero_rate*100:.1f}%")

    log(f"\n  ─── 판정 결과 ───")

    if vpin_nonzero_cnt > 0 and (vol_zero_rate > 0.95 or not vol_cols):
        log(f"  🔴 경우 1 해당: 거래량은 0인데 VPIN은 0이 아닌 값 존재")
        log(f"     → 패키지가 보간법으로 VPIN '값 자체'를 추정 생성한 것")
        log(f"     → 교수님 의견: 주말 행을 삭제하면 되고, 재계산 불필요")
        log(f"")
        log(f"     ✅ 조치 방안: 주말/공휴일 행을 단순 삭제(Drop)하여 사용")
    elif vpin_zero_rate > 0.95:
        log(f"  🟡 경우 2-정상: 거래량 0, VPIN도 0")
        log(f"     → 시간만 할당했고 VPIN은 0으로 계산됨 (교수님: 정상)")
        log(f"     → 주말 행 삭제 후 사용 가능")
    elif vol_cols and vol_zero_rate < 0.50:
        log(f"  🔴 경우 2-오류: 주말 버킷에 거래량이 0이 아닌 값 다수 존재")
        log(f"     → 실제 거래(금/월)가 주말 시간대로 잘못 배분된 것")
        log(f"     → 교수님 의견: 프로그램에 문제 있음 → 로직 수정 후 재계산 필요")
        log(f"")
        log(f"     ⚠️ 조치 방안: 단순 삭제 불가, 방법 A(로직 수정 후 재계산) 필요")
    else:
        log(f"  🟠 혼합 케이스: 추가 분석 필요")
        log(f"     거래량 0 비율: {vol_zero_rate*100:.1f}%, VPIN 0 비율: {vpin_zero_rate*100:.1f}%")

    # =================================================================
    # STEP 5: 주말 버킷 샘플 데이터 출력
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 5] 주말 버킷 샘플 데이터 (처음 20건)")
    log(f"{'='*90}")

    sample_cols = ["Symbol", "BucketNo", "Date", "Starttime", "Endtime"]
    sample_cols += vol_cols[:3]
    sample_cols += vpin_cols
    sample_cols = [c for c in sample_cols if c in df_weekend.columns]

    log(df_weekend.select(sample_cols).head(20))

    # --- 주말 VPIN ≠ 0인 샘플 ---
    if vpin_nonzero_cnt > 0:
        log(f"\n  ▶ 주말 VPIN ≠ 0인 샘플 (최대 10건):")
        df_nonzero = df_weekend.filter(pl.col(primary_vpin) != 0).select(sample_cols).head(10)
        log(df_nonzero)

    # =================================================================
    # STEP 6: 주말 행 제거 전후 합산 비교
    # =================================================================
    log(f"\n{'='*90}")
    log(f" [STEP 6] 주말 행 제거 전후 합산 비교 (데이터 무결성 검증)")
    log(f"{'='*90}")

    for vc in vol_cols[:3] + vpin_cols:
        total_all = df[vc].sum()
        total_weekday = df_weekday[vc].sum()
        total_weekend_val = df_weekend[vc].sum()
        log(f"\n  [{vc}]")
        log(f"    전체 합계   : {total_all}")
        log(f"    평일 합계   : {total_weekday}")
        log(f"    주말 합계   : {total_weekend_val}")
        if total_all and total_all != 0:
            loss_pct = abs(float(total_weekend_val)) / abs(float(total_all)) * 100
            log(f"    주말 비중   : {loss_pct:.4f}%")
            if loss_pct < 1.0:
                log(f"    → 주말 합계 비중 < 1% → 삭제해도 전체 영향 미미")
            else:
                log(f"    → ⚠️ 주말 비중 {loss_pct:.2f}% → 삭제 시 데이터 손실 주의")

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