"""
=============================================================================
[VPIN 주말 발생 문제 종합 진단기] vpin_diagnose.py
=============================================================================

목적:
  VPIN 결과에 주말(토/일) 날짜가 포함되는 근본 원인을 찾는다.

검증하는 가설:
  [가설 A] 타임존 버그 (R 코드에 tz="Asia/Seoul" 고정)
    - vpin_r_heavy.R 이 CA(캐나다)를 처리할 때도 Asia/Seoul 타임존을 사용
    - 캐나다 ET 시간은 KST보다 14시간(EST) 느림
    - 금요일 오전 10시 ET 이후 거래 → KST 기준 다음날(토요일) 날짜 배정

  [가설 B] PINstimation 선형 보간
    - vpin() 내부에서 금요일 마지막 틱~월요일 첫 번째 틱 사이를
      비어있는 시간대로 보고 중간 버킷에 주말 타임스탬프 할당

사용법:
  pip install polars exchange-calendars
  python vpin_diagnose.py

출력:
  콘솔 + vpin_diagnosis_report.txt (NEW_PARQUET 파일과 같은 폴더)
=============================================================================
"""

import os
import sys
import glob
import warnings
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Tuple
from zoneinfo import ZoneInfo

import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

try:
    import exchange_calendars as xcals
    HAS_XCALS = True
except ImportError:
    HAS_XCALS = False

# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY  = "CA"

# VPIN 결과 parquet (0409 코드에 기록된 파일)
VPIN_PARQUET: Optional[str] = (
    r"E:\vpin_project_parquet\processing_data_2019_2021"
    r"\R_output\CA\vpin\vpin_CA_20260409_0202.parquet"
)

# 1분봉 집계 파일 (Python 전처리 결과)
BAR_1M_PATH: Optional[str] = (
    r"E:\vpin_project_parquet\processing_data_2019_2021"
    r"\R_output\CA\vpin\all_1m_bars.parquet"
)

# 나라별 exchange_calendars 코드
XCAL_MAP = {
    "KOR": ("XKRX", "Asia/Seoul"),
    "US":  ("XNYS", "America/New_York"),
    "JP":  ("XTKS", "Asia/Tokyo"),
    "CA":  ("XTSE", "America/Toronto"),
    "FR":  ("XPAR", "Europe/Paris"),
    "GR":  ("XFRA", "Europe/Berlin"),
    "HK":  ("XHKG", "Asia/Hong_Kong"),
    "IT":  ("XMIL", "Europe/Rome"),
    "UK":  ("XLON", "Europe/London"),
}

# R 코드에 하드코딩된 (잘못된) 타임존
R_HARDCODED_TZ = "Asia/Seoul"

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

OUTPUT_DIR  = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
REPORT_PATH = os.path.join(OUTPUT_DIR, "vpin_diagnosis_report.txt")

lines: List[str] = []

def log(msg: str = ""):
    print(msg)
    lines.append(msg)

def sep(char: str = "=", n: int = 65):
    log(char * n)

def find_latest_parquet(directory: str) -> Optional[str]:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [f for f in sorted(glob.glob(pattern))
             if "_sample" not in os.path.basename(f)
             and "_cleaned" not in os.path.basename(f)]
    return files[-1] if files else None


# =============================================================================
# [0] 파일 로드 유틸
# =============================================================================

def load_vpin(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "Date" in df.columns and df["Date"].dtype != pl.Date:
        df = df.with_columns(pl.col("Date").cast(pl.Date))
    return df


# =============================================================================
# [1] 기초 통계 — 날짜/요일 분포
# =============================================================================

def section_basic(df: pl.DataFrame):
    sep()
    log("[1] 기초 통계 — 날짜/요일 분포")
    sep()

    day_names = {1: "월요일", 2: "화요일", 3: "수요일", 4: "목요일",
                 5: "금요일", 6: "토요일", 7: "일요일"}

    dist = (
        df.with_columns(pl.col("Date").dt.weekday().alias("WD"))
        .group_by("WD")
        .agg([pl.len().alias("Rows"), pl.col("Date").n_unique().alias("Days")])
        .sort("WD")
    )

    log(f"  총 행 수  : {df.height:,}")
    log(f"  종목 수   : {df['Symbol'].n_unique():,}")
    log(f"  날짜 범위 : {df['Date'].min()} ~ {df['Date'].max()}")
    log(f"\n  {'요일':<8} {'날짜 수':>8} {'행 수':>12} {'이상':>4}")
    log(f"  {'─'*8} {'─'*8} {'─'*12} {'─'*4}")
    for r in dist.iter_rows(named=True):
        flag = " ← ✗" if r["WD"] in (6, 7) else ""
        log(f"  {day_names[r['WD']]:<8} {r['Days']:>8,} {r['Rows']:>12,}{flag}")

    n_wk = df.filter(pl.col("Date").dt.weekday().is_in([6, 7])).height
    log(f"\n  주말 행 합계: {n_wk:,}행")
    return n_wk


# =============================================================================
# [2] 가설 A: 타임존 버그 진단
#
# R 코드의 핵심 버그:
#   attr(ts_raw, "tzone") <- "Asia/Seoul"   ← 모든 나라에 Seoul 적용
#   date_vals <- as.Date(et_vals, tz = "Asia/Seoul")  ← 날짜 추출도 Seoul 기준
#
# CA(캐나다) 데이터의 타임스탬프는 현지 ET 시간으로 저장되어 있음.
# Arrow가 이를 파이낸셜로 읽을 때 UTC로 간주하고, R은 Seoul 라벨만 붙임.
# → endtime에서 as.Date(tz="Asia/Seoul")을 하면 날짜가 밀림.
#
# 예시 (EST = UTC-5, KST = UTC+9):
#   실제 거래: 2019-01-04 (금) 10:00 AM ET
#   parquet 저장값: 2019-01-04 10:00:00 (naive → Arrow가 UTC 취급)
#   R이 Seoul 기준 날짜 추출: UTC 10:00 = KST 19:00 → 날짜: 2019-01-04 (금) OK
#
#   실제 거래: 2019-01-04 (금) 10:01 AM ET
#   parquet 저장값: 2019-01-04 10:01:00 (naive → Arrow가 UTC 취급)
#   R이 Seoul 기준 날짜 추출: UTC 10:01 = KST 19:01 → 날짜: 2019-01-04 (금) OK
#
#   실제 거래: 2019-01-04 (금) 15:00 PM ET
#   parquet 저장값: 2019-01-04 15:00:00 (naive → UTC 취급)
#   R이 Seoul 기준 날짜 추출: UTC 15:00 = KST 00:00 (자정, 다음날!) → 2019-01-05 (토)  ← 버그!
#
# 결론: ET 오전 10시 ~ 14:59:59 사이 버킷은 날짜가 1일 앞당겨짐
#       → 금요일 오전 10시 이후 거래의 Date가 토요일로 기록됨
# =============================================================================

def section_timezone_bug(df: pl.DataFrame):
    sep()
    log("[2] 가설 A: 타임존 버그 (tz='Asia/Seoul' 고정)")
    sep()

    xcal_entry = XCAL_MAP.get(COUNTRY)
    if xcal_entry is None:
        log(f"  ⚠ XCAL_MAP에 '{COUNTRY}' 없음 → 타임존 분석 스킵")
        return

    _, local_tz_str = xcal_entry
    KST = ZoneInfo("Asia/Seoul")
    LOCAL_TZ = ZoneInfo(local_tz_str)

    # UTC-KST 오프셋 차이: KST는 UTC+9
    # 로컬 시각이 UTC로 잘못 해석될 때의 날짜 오차 계산
    log(f"\n  COUNTRY      : {COUNTRY}")
    log(f"  현지 타임존  : {local_tz_str}")
    log(f"  R 코드 타임존: {R_HARDCODED_TZ} (잘못된 값!)")

    # 현지 타임존과 KST의 UTC 오프셋 계산 (겨울 기준)
    sample_dt_naive = datetime(2019, 1, 15, 12, 0, 0)  # 1월 겨울 기준
    sample_dt_local = sample_dt_naive.replace(tzinfo=LOCAL_TZ)
    sample_dt_kst   = sample_dt_naive.replace(tzinfo=ZoneInfo("UTC")).astimezone(KST)

    local_utc_offset = sample_dt_local.utcoffset()
    kst_utc_offset   = timedelta(hours=9)

    log(f"\n  현지 UTC 오프셋 (겨울): {local_utc_offset}")
    log(f"  KST  UTC 오프셋      : +09:00")

    # R이 하는 짓: naive 타임스탬프를 UTC로 읽고 KST 날짜를 계산
    # 현지 시각이 UTC로 오해받을 때, KST 기준으로 날짜가 달라지는 시간 경계 계산
    # KST = UTC+9, so UTC 15:00 = KST 00:00 (midnight) → next day
    # 즉, naive timestamp값이 15:00:00 이상이면 KST 날짜가 다음날로 바뀜
    shift_threshold_utc_hour = 15  # UTC 15시 = KST 자정
    # 하지만 현지 시각의 나이브 값이 UTC처럼 취급되므로
    # 현지 오전 15시 이상 저장값 → KST 자정 넘어 → 날짜 1일 증가
    # 실제로는 (24 - KST_offset_hours) = 24 - 9 = 15시가 경계

    log(f"\n  [날짜 오류 발생 경계]")
    log(f"  naive timestamp 값 15:00:00 이상이면 KST 기준으로 다음날 날짜 배정됨")
    log(f"  {COUNTRY} 현지 시각 기준: {local_tz_str} 15:00 이상 → KST 날짜 1일 증가")

    # 현지 KST 오프셋 차이로 경계 계산
    if local_utc_offset is not None:
        local_utc_hours = local_utc_offset.total_seconds() / 3600
        # naive dt가 UTC처럼 취급 → 실제 UTC 변환 없이 raw값이 UTC로 사용
        # raw 15:00:00 = KST 00:00 → next day
        # 현지 시각 x를 naive로 저장 → R이 UTC x로 취급 → KST x+9
        # x+9 >= 24 → x >= 15 → naive값 15:00이상이면 날짜 증가
        log(f"  따라서 저장된 naive 시각 15:00 이상인 버킷의 Date가 1일 밀림")
        log(f"  {COUNTRY} 장 마감 시각(현지 naive) 기준:")
        # CA 장 시간: 9:30 ~ 16:00 ET
        # ET = UTC-5 (겨울) → 저장 naive값은 ET 시각과 동일
        # 15:00 이상 → KST 날짜 틀림 → 장 중 10:00(나이브) ~ 16:00(나이브) 모두 영향!
        if COUNTRY == "CA":
            log(f"    - CA 장 시간: 09:30 ~ 16:00 (naive ET 기준 저장)")
            log(f"    - naive 15:00 이상이면 KST 날짜 +1")
            log(f"    - 따라서 장 중 15:00~16:00 구간의 버킷은 모두 날짜가 1일 밀림")
            log(f"    - 금요일 15:00~16:00 버킷 → 토요일 날짜 배정!")
        elif COUNTRY == "US":
            log(f"    - US 장 시간: 09:30 ~ 16:00 (naive ET 기준 저장)")
            log(f"    - 동일하게 15:00~16:00 버킷 날짜 1일 밀림")

    # 실제 데이터 검증: endtime의 Hour 분포 확인
    log(f"\n  [실제 VPIN 결과에서 Hour 분포 분석]")
    if "Starttime" in df.columns:
        ts_col = "Starttime"
    elif "Endtime" in df.columns:
        ts_col = "Endtime"
    else:
        log("  ⚠ Starttime/Endtime 컬럼 없음 → 시간 분석 불가")
        return

    df_with_hour = df.with_columns([
        pl.col(ts_col).dt.hour().alias("Hour"),
        pl.col("Date").dt.weekday().alias("WD"),
    ])

    # 주말 버킷의 Hour 분포
    weekend_df = df_with_hour.filter(pl.col("WD").is_in([6, 7]))
    if weekend_df.is_empty():
        log("  ✓ 주말 버킷 없음 — 타임존 버그 미발생 (또는 이미 수정됨)")
        return

    hour_dist = (
        weekend_df.group_by("Hour").agg(pl.len().alias("Count")).sort("Hour")
    )
    log(f"  주말 버킷 총 {weekend_df.height:,}행의 Hour 분포:")
    log(f"  {'Hour':>5} | {'Count':>8}")
    log(f"  {'─'*5}+{'─'*9}")
    peak_hours = []
    for r in hour_dist.iter_rows(named=True):
        log(f"  {r['Hour']:>5}h | {r['Count']:>8,}")
        peak_hours.append(r["Hour"])

    # 해석
    log(f"\n  [해석]")
    early_morning = [h for h in peak_hours if h < 6]
    mid_day = [h for h in peak_hours if 9 <= h <= 16]
    if early_morning:
        log(f"  🎯 새벽 시간대({early_morning})에 집중 → 타임존 시프트 버그 강력 의심")
        log(f"     예: ET 15:00 저장값 → UTC로 오해 → KST 00:00 → 다음날 날짜")
    if mid_day:
        log(f"  ⚠ 장중 시간대({mid_day})에도 발생 → 보간 버그 가능성도 있음")


    # 타임존 버그 정량 검증
    # "만약 endtime을 올바른 현지 타임존으로 변환했다면 날짜는?" 비교
    log(f"\n  [타임존 버그 정량 검증]")
    log(f"  주말 버킷의 {ts_col} 값을 올바른 타임존({local_tz_str})으로")
    log(f"  해석했을 때와 잘못된 타임존(Asia/Seoul)으로 해석했을 때의 날짜 비교:")

    if df[ts_col].dtype in (pl.Datetime, pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")):
        # naive datetime 또는 UTC datetime인 경우
        sample_wknd = weekend_df.select([ts_col, "Date"]).head(10)
        log(f"\n  {'저장값 (naive/UTC)':>30} | {'R 날짜 (KST)':>12} | {'올바른 날짜 (ET)':>15} | {'일치?':>4}")
        log(f"  {'─'*30}+{'─'*12}+{'─'*15}+{'─'*4}")

        for r in sample_wknd.iter_rows(named=True):
            raw_ts = r[ts_col]
            r_date = r["Date"]  # R이 Seoul 기준으로 만든 날짜

            if raw_ts is not None:
                # R이 하는 짓: naive ts → UTC 취급 → KST 날짜
                kst_date = (raw_ts.replace(tzinfo=timezone.utc)
                            .astimezone(ZoneInfo("Asia/Seoul"))
                            .date())
                # 올바른 날짜: naive ts → 현지(ET) 취급 → 날짜
                correct_date = (raw_ts.replace(tzinfo=timezone.utc)
                                .astimezone(LOCAL_TZ)
                                .date())
                # 하지만 실제로는 naive ts 값이 이미 현지 시각
                # 즉, 저장값 자체가 ET 시각이므로, 그냥 naive의 date()가 올바른 날짜
                naive_date = raw_ts.date()
                match = "✓" if kst_date == r_date else "?"
                log(f"  {str(raw_ts):>30} | {str(kst_date):>12} | {str(naive_date):>15} | {match}")
    else:
        log(f"  {ts_col} dtype={df[ts_col].dtype} — 직접 변환 스킵")


    # 가장 중요한 발견: R 코드에서 날짜가 달라지는 경계
    log(f"\n  [결론]")
    log(f"  R 코드 (vpin_r_heavy.R) 에서:")
    log(f"    attr(ts_raw, 'tzone') <- 'Asia/Seoul'   ← 모든 국가에 동일 적용")
    log(f"    date_vals <- as.Date(et_vals, tz = 'Asia/Seoul')  ← 날짜 추출 오류")
    log(f"")
    log(f"  {COUNTRY} 데이터 naive 타임스탬프 15:00 이상이면 KST 기준 다음날 날짜 배정")
    log(f"  → 금요일 15:00~16:00 거래 버킷 → 토요일 날짜로 기록")
    log(f"  → 이것이 주말 VPIN 발생의 핵심 원인!")


# =============================================================================
# [3] 가설 B: PINstimation 선형 보간 (금-월 갭 사이 버킷 생성)
# =============================================================================

def section_interpolation_bug(df: pl.DataFrame, bar_path: Optional[str]):
    sep()
    log("[3] 가설 B: PINstimation 선형 보간 (금→월 갭 내 버킷 생성)")
    sep()

    if "Starttime" not in df.columns or "Endtime" not in df.columns:
        log("  ⚠ Starttime/Endtime 컬럼 없음 → 보간 검증 스킵")
        return

    weekend_df = df.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    if weekend_df.is_empty():
        log("  ✓ 주말 버킷 없음 → 보간 검증 불필요")
        return

    n_wknd = weekend_df.height

    if bar_path and os.path.exists(bar_path):
        log(f"  1분봉 파일 로드 중: {bar_path}")
        df_bars = pl.scan_parquet(bar_path).select(["Symbol", "Datetime"]).collect()

        df_bars_sorted = df_bars.sort(["Symbol", "Datetime"])
        df_wknd_sorted = weekend_df.sort(["Symbol", "Starttime"])

        # 주말 버킷 시작 직전의 실제 틱 시간 (backward join)
        merged = df_wknd_sorted.join_asof(
            df_bars_sorted,
            left_on="Starttime",
            right_on="Datetime",
            by="Symbol",
            strategy="backward"
        ).rename({"Datetime": "Last_Real_Tick"})

        # 주말 버킷 종료 직후의 실제 틱 시간 (forward join)
        merged = merged.sort(["Symbol", "Endtime"])
        merged = merged.join_asof(
            df_bars_sorted,
            left_on="Endtime",
            right_on="Datetime",
            by="Symbol",
            strategy="forward"
        ).rename({"Datetime": "Next_Real_Tick"})

        merged = merged.with_columns([
            (pl.col("Next_Real_Tick") - pl.col("Last_Real_Tick"))
            .dt.total_hours().alias("Gap_Hours"),
            pl.col("Last_Real_Tick").dt.weekday().alias("Last_WD"),
            pl.col("Next_Real_Tick").dt.weekday().alias("Next_WD"),
        ])

        # 금-월 패턴 (완전한 주말 갭)
        fri_mon = merged.filter(
            (pl.col("Last_WD") == 5) & (pl.col("Next_WD") == 1)
        ).height

        # 48시간 이상 갭 (주말 포함)
        big_gap = merged.filter(pl.col("Gap_Hours") >= 48).height

        log(f"\n  총 주말 버킷              : {n_wknd:,}행")
        log(f"  실제 금→월 갭 내 버킷     : {fri_mon:,}행  ({fri_mon/n_wknd*100:.1f}%)")
        log(f"  48시간 이상 갭 내 버킷    : {big_gap:,}행  ({big_gap/n_wknd*100:.1f}%)")

        if n_wknd > 0:
            if big_gap / n_wknd > 0.90:
                log(f"\n  🎯 90% 이상이 금-월 큰 갭 안에 있음 → 보간 가설 유력")
            elif fri_mon / n_wknd > 0.80:
                log(f"\n  🎯 80% 이상이 금요일 마지막 틱~월요일 첫 틱 사이 → 보간 가설 유력")
            else:
                log(f"\n  ⚠ 보간 패턴이 뚜렷하지 않음 → 타임존 버그가 주요 원인일 가능성 높음")

        # 샘플 5개 출력
        sample = merged.filter(
            (pl.col("Last_WD") == 5) & (pl.col("Next_WD") == 1)
        ).select([
            "Symbol", "BucketNo", "Last_Real_Tick",
            "Starttime", "Endtime", "Next_Real_Tick", "Gap_Hours"
        ]).head(5)

        if not sample.is_empty():
            log(f"\n  [금→월 갭 내 주말 버킷 샘플 5건]")
            for r in sample.iter_rows(named=True):
                log(f"    Symbol={r['Symbol']}, BucketNo={r['BucketNo']}")
                log(f"      Last Real Tick : {r['Last_Real_Tick']}")
                log(f"      Bucket Start   : {r['Starttime']}")
                log(f"      Bucket End     : {r['Endtime']}")
                log(f"      Next Real Tick : {r['Next_Real_Tick']}")
                log(f"      Gap            : {r['Gap_Hours']:.1f}h")
                log("")
    else:
        log(f"  ⚠ 1분봉 파일 없음 → 보간 검증 스킵")
        log(f"  (BAR_1M_PATH를 설정하거나 파일을 확인하세요)")

    # 보간 없이 타임존 버그만으로도 설명 가능한지 체크
    log(f"\n  [타임존 버그로만 주말 발생이 설명되는지 검토]")
    log(f"  Starttime의 naive 시각이 15:00 이상인 주말 버킷 비율:")
    if "Starttime" in df.columns:
        weekend_hour_check = (
            weekend_df
            .with_columns(pl.col("Starttime").dt.hour().alias("Hour"))
            .filter(pl.col("Hour") >= 15)
        ).height
        pct = weekend_hour_check / n_wknd * 100 if n_wknd > 0 else 0
        log(f"    Starttime naive >= 15:00 → {weekend_hour_check:,}행 ({pct:.1f}%)")
        if pct > 60:
            log(f"  → 타임존 버그로 대부분 설명 가능 (가설 A 확정)")
        else:
            log(f"  → 타임존 버그만으로는 불충분, 보간 버그도 함께 작용 중 (가설 A+B)")


# =============================================================================
# [4] 두 가설의 상호 배타성 검토
# =============================================================================

def section_root_cause(df: pl.DataFrame):
    sep()
    log("[4] 근본 원인 특정 — 코드 레벨 분석")
    sep()

    log("""
  R 코드 (vpin_r_heavy.R) 의 핵심 문제 라인:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ # ── POSIXct tzone 설정 ──                                         │
  │ ts_raw <- sym_df$Datetime                                           │
  │ if (inherits(ts_raw, "POSIXct")) {                                  │
  │     attr(ts_raw, "tzone") <- "Asia/Seoul"   ← ★ 버그 1             │
  │ } else if (is.numeric(ts_raw)) {                                    │
  │     ts_raw <- as.POSIXct(ts_raw, origin="1970-01-01",              │
  │                           tz="Asia/Seoul")  ← ★ 버그 2             │
  │ } else {                                                            │
  │     ts_raw <- as.POSIXct(as.character(ts_raw),                     │
  │                           tz="Asia/Seoul")  ← ★ 버그 3             │
  │ }                                                                   │
  │                                                                     │
  │ (중략)                                                              │
  │                                                                     │
  │ # ★ 날짜 추출: endtime 기준                                         │
  │ date_vals <- as.Date(et_vals, tz = "Asia/Seoul")  ← ★ 버그 4       │
  └─────────────────────────────────────────────────────────────────────┘

  문제:
    모든 국가의 데이터를 "Asia/Seoul" 타임존으로 처리함.
    KOR(한국)에서는 올바르지만, CA/US/UK/JP/FR 등은 모두 잘못됨.

  CA(캐나다) 에서의 영향:
    1. Python이 저장: ET 시각을 naive datetime으로 저장
       예) 금요일 15:30 ET → 2019-01-04 15:30:00 (naive)

    2. Arrow가 R로 로드: naive datetime → UTC POSIXct로 읽힘
       → 2019-01-04 15:30:00 UTC (epoch=1546617000)

    3. R이 tzone 라벨만 교체: 값(epoch) 변경 없이 라벨만 "Asia/Seoul"로 변경
       → 내부적으로는 여전히 epoch=1546617000이지만 KST로 표시됨
       → 2019-01-05 00:30:00 KST (자정 이후 → 토요일!)

    4. as.Date(tz="Asia/Seoul"): epoch를 KST 기준 날짜로 변환
       → 2019-01-05 (토요일) ← 원래는 2019-01-04 (금요일)이어야 함!

  계산:
    ET와 KST 차이: KST = ET + 14시간 (EST 기준, 겨울)
    naive값 15:00 이상 → KST 00:00 이상 → 날짜 1일 증가
    CA 장 시간 (ET): 09:30 ~ 16:00
    따라서 15:00 ~ 16:00 (30분 구간) 거래 버킷 → 날짜 1일 밀림
    → 금요일 15:00 ~ 16:00 거래 버킷 → 토요일 날짜로 기록

  수정 방법 (R 코드에서):
    # COUNTRY별 올바른 타임존 매핑 추가
    tz_map <- list(
      KOR = "Asia/Seoul", US = "America/New_York", CA = "America/Toronto",
      UK = "Europe/London", JP = "Asia/Tokyo", FR = "Europe/Paris",
      GR = "Europe/Berlin", HK = "Asia/Hong_Kong", IT = "Europe/Rome"
    )
    local_tz <- tz_map[[COUNTRY]]  # 또는 function argument로 전달

    # 타임존 적용 시:
    attr(ts_raw, "tzone") <- local_tz  # "Asia/Seoul" 대신

    # 날짜 추출 시:
    date_vals <- as.Date(et_vals, tz = local_tz)  # "Asia/Seoul" 대신
    """)

    log("""
  Python 전처리 (vpin_pre_heavy.py) 쪽은 문제 없음:
    dt.combine(Date, Time) → naive datetime 저장
    이것 자체는 정상 동작, 문제는 R이 이를 잘못 해석하는 것
    """)


# =============================================================================
# [5] 영업일 기반 최종 검증
# =============================================================================

def section_calendar(df: pl.DataFrame):
    sep()
    log("[5] 영업일 기반 검증 (exchange_calendars)")
    sep()

    if not HAS_XCALS:
        log("  ⚠ exchange_calendars 미설치 → pip install exchange-calendars")
        return

    xcal_entry = XCAL_MAP.get(COUNTRY)
    if xcal_entry is None:
        log(f"  ⚠ XCAL_MAP에 '{COUNTRY}' 없음")
        return

    xcal_code, _ = xcal_entry
    dates = df["Date"].unique().sort()
    date_min = str(dates.min())
    date_max = str(dates.max())

    try:
        import exchange_calendars as xcals
        cal = xcals.get_calendar(xcal_code)
        sessions = set(s.date() for s in cal.sessions_in_range(date_min, date_max))
    except Exception as e:
        log(f"  ⚠ exchange_calendars 오류: {e}")
        return

    log(f"  캘린더: {xcal_code}")
    log(f"  데이터 기간: {date_min} ~ {date_max}")
    log(f"  실제 영업일 수: {len(sessions):,}일")

    data_dates = set(dates.to_list())
    extra_dates = data_dates - sessions  # 영업일이 아닌 날짜

    weekends    = [d for d in extra_dates if d.weekday() >= 5]
    hol_weekday = [d for d in extra_dates if d.weekday() < 5]

    log(f"\n  VPIN에 있는 날짜 중 영업일 아닌 날:")
    log(f"    주말 날짜  : {len(weekends):,}일")
    log(f"    주중 휴장일: {len(hol_weekday):,}일")

    if weekends:
        wknd_rows = df.filter(pl.col("Date").is_in(weekends)).height
        log(f"\n  주말 날짜 상세 (최대 10개):")
        for d in sorted(weekends)[:10]:
            cnt = df.filter(pl.col("Date") == d).height
            wday = ["월", "화", "수", "목", "금", "토", "일"][d.weekday()]
            log(f"    {d} ({wday}) : {cnt:,}행")
        if len(weekends) > 10:
            log(f"    ... 외 {len(weekends) - 10}일")
        log(f"\n  주말 VPIN 총 행 수: {wknd_rows:,}행")

    if not weekends and not hol_weekday:
        log(f"\n  ✓ 모든 날짜가 영업일 — 정상")


# =============================================================================
# [6] 전후 파일 비교 (OLD vs NEW)
# =============================================================================

def section_compare(old_path: Optional[str], new_path: str):
    sep()
    log("[6] OLD vs NEW 파일 비교")
    sep()

    if not old_path or not os.path.exists(old_path):
        log(f"  OLD 파일 미지정 또는 없음 → 비교 스킵")
        return

    old_df = load_vpin(old_path)
    new_df = load_vpin(new_path)

    log(f"  OLD: {os.path.basename(old_path)}")
    log(f"  NEW: {os.path.basename(new_path)}")
    log(f"\n  {'항목':<20} {'OLD':>12} {'NEW':>12}")
    log(f"  {'─'*20} {'─'*12} {'─'*12}")
    log(f"  {'총 행 수':<20} {old_df.height:>12,} {new_df.height:>12,}")
    log(f"  {'종목 수':<20} {old_df['Symbol'].n_unique():>12,} {new_df['Symbol'].n_unique():>12,}")
    log(f"  {'고유 날짜 수':<20} {old_df['Date'].n_unique():>12,} {new_df['Date'].n_unique():>12,}")

    old_wk = old_df.filter(pl.col("Date").dt.weekday().is_in([6, 7])).height
    new_wk = new_df.filter(pl.col("Date").dt.weekday().is_in([6, 7])).height
    log(f"  {'주말 행 수':<20} {old_wk:>12,} {new_wk:>12,}")

    old_vpin = old_df["VPIN"].drop_nulls()
    new_vpin = new_df["VPIN"].drop_nulls()
    log(f"  {'VPIN 평균':<20} {old_vpin.mean():>12.6f} {new_vpin.mean():>12.6f}")

    if old_wk > 0 and new_wk == 0:
        log(f"\n  ✓ 주말 VPIN 해결됨: {old_wk:,}행 → 0행")
    elif old_wk > 0 and new_wk > 0:
        log(f"\n  ✗ 여전히 주말 VPIN 존재: {old_wk:,}행 → {new_wk:,}행")
    elif old_wk == 0 and new_wk > 0:
        log(f"\n  ✗ 새 파일에서 주말 VPIN 발생: {new_wk:,}행")
    else:
        log(f"\n  ✓ 두 파일 모두 주말 VPIN 없음")


# =============================================================================
# [7] 요약 및 수정 가이드
# =============================================================================

def section_summary(n_weekend: int):
    sep()
    log("[7] 진단 요약 및 수정 가이드")
    sep()

    if n_weekend == 0:
        log("""
  ✅ 현재 파일에는 주말 VPIN이 없습니다.
     이전 파일과 비교할 때 문제가 해결된 것으로 보입니다.
     OLD 파일에 주말 VPIN이 있었다면, 아래 수정 사항이 적용된 것입니다.
        """)
    else:
        log(f"""
  ❌ 주말 VPIN {n_weekend:,}행 발견.

  [근본 원인]
  vpin_r_heavy.R 에서 tz="Asia/Seoul"이 모든 국가에 하드코딩되어 있음.
  {COUNTRY} 데이터의 경우:
    - 현지 ET 시각을 naive datetime으로 저장
    - Arrow가 R로 읽을 때 UTC로 해석
    - R이 KST 기준으로 날짜 추출
    - → 오후 15:00 이후 버킷의 날짜가 1일 밀림
    - → 금요일 15:00~16:00 거래 → 토요일로 기록

  [수정 방법 - vpin_r_heavy.R]

  1) 상단 설정 구역에 타임존 맵 추가:
     ─────────────────────────────────────────────
     tz_map <- list(
       KOR = "Asia/Seoul",    US = "America/New_York",
       CA  = "America/Toronto", JP = "Asia/Tokyo",
       UK  = "Europe/London",  FR = "Europe/Paris",
       GR  = "Europe/Berlin", HK = "Asia/Hong_Kong",
       IT  = "Europe/Rome"
     )
     LOCAL_TZ <- tz_map[[COUNTRY]]
     ─────────────────────────────────────────────

  2) worker_fn 내부에서 "Asia/Seoul" → LOCAL_TZ 로 교체:
     ─────────────────────────────────────────────
     # 변경 전:
     attr(ts_raw, "tzone") <- "Asia/Seoul"
     ts_raw <- as.POSIXct(ts_raw, origin="1970-01-01", tz="Asia/Seoul")

     # 변경 후:
     attr(ts_raw, "tzone") <- local_tz   # args에서 받아야 함
     ts_raw <- as.POSIXct(ts_raw, origin="1970-01-01", tz=local_tz)
     ─────────────────────────────────────────────

  3) Date 추출 라인:
     ─────────────────────────────────────────────
     # 변경 전:
     date_vals <- as.Date(et_vals, tz = "Asia/Seoul")

     # 변경 후:
     date_vals <- as.Date(et_vals, tz = local_tz)
     ─────────────────────────────────────────────

  4) symbol_args 생성 시 local_tz 전달:
     ─────────────────────────────────────────────
     list(
       ...
       local_tz = LOCAL_TZ   # ← 추가
     )
     ─────────────────────────────────────────────

  [주의]
  - vpin_pre_heavy.py (Python 전처리)는 수정 불필요
  - 기존 체크포인트(.rds) 삭제 후 R 재실행 필요
  - 또는 FORCE_REPROCESS 옵션으로 재처리
        """)


# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()

    sep()
    log(f"  VPIN 주말 발생 문제 종합 진단 리포트")
    log(f"  실행 시각: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  국가: {COUNTRY}")
    sep()

    # 파일 탐색
    vpin_path = VPIN_PARQUET or find_latest_parquet(OUTPUT_DIR)
    if not vpin_path or not os.path.exists(vpin_path):
        log(f"[Error] VPIN parquet 파일을 찾을 수 없습니다.")
        log(f"  탐색 경로: {OUTPUT_DIR}")
        log(f"  VPIN_PARQUET 경로를 직접 설정하세요.")
        sys.exit(1)

    log(f"\n  대상 파일: {os.path.basename(vpin_path)}")

    # 파일 로드
    df = load_vpin(vpin_path)
    log(f"  로드 완료: {df.height:,}행, {df['Symbol'].n_unique():,}종목")

    # 검증 실행
    n_weekend = section_basic(df)
    section_timezone_bug(df)
    section_interpolation_bug(df, BAR_1M_PATH)
    section_root_cause(df)
    section_calendar(df)

    # OLD vs NEW 비교
    old_path = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\vpin_CA_20260402_2323.parquet"
    section_compare(old_path if os.path.exists(old_path) else None, vpin_path)

    section_summary(n_weekend)

    # 소요 시간
    sep()
    log(f"  총 소요 시간: {datetime.now() - start}")
    sep()

    # 리포트 저장
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  📄 리포트 저장: {REPORT_PATH}")
