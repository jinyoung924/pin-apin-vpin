"""
=============================================================================
[VPIN 검증] 수정 전/후 parquet 비교 + 영업일 검증
=============================================================================

사용법:
  python vpin_validate.py

기능:
  1) 수정 후 결과에서 영업일이 아닌 날(주말/공휴일)에 VPIN이 산출되었는지 검증
  2) 수정 전/후 parquet이 모두 있으면 비교 분석 (날짜 분포 변화, VPIN 통계 비교)

출력:
  콘솔 리포트 + {OUTPUT_DIR}/vpin_validation_report.txt

의존:
  pip install polars exchange-calendars
=============================================================================
"""

import os
import sys
import glob
import polars as pl
import warnings
from datetime import datetime
from typing import Optional, List

try:
    import exchange_calendars as xcals
    HAS_XCALS = True
except ImportError:
    HAS_XCALS = False

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"
COUNTRY = "CA"

# 수정 전 parquet 경로 (없으면 None → 비교 스킵)
OLD_PARQUET: Optional[str] = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\vpin_CA_20260402_2323.parquet"  # 예: r"E:\...\vpin_UK_20250101_1200.parquet"

# 수정 후 parquet 경로 (None이면 자동 탐색: 가장 최신 파일)
NEW_PARQUET: Optional[str] = None
# r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\vpin_CA_20260409_0202.parquet"

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

# ── 나라별 exchange_calendars 코드 매핑 ──
XCAL_MAP = {
    "KOR": "XKRX",
    "US":  "XNYS",
    "JP":  "XTKS",
    "CA":  "XTSE",
    "FR":  "XPAR",
    "GR":  "XFRA",
    "HK":  "XHKG",
    "IT":  "XMIL",
    "UK":  "XLON",
}

OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
REPORT_PATH = os.path.join(OUTPUT_DIR, "vpin_validation_report.txt")

lines: List[str] = []


def log(msg: str = ""):
    print(msg)
    lines.append(msg)


def find_latest_parquet(directory: str, prefix: str = "vpin_") -> Optional[str]:
    """디렉토리에서 prefix로 시작하는 가장 최신 parquet 파일을 찾는다."""
    pattern = os.path.join(directory, f"{prefix}{COUNTRY}_*.parquet")
    files = sorted(glob.glob(pattern))
    # _sample 파일 제외
    files = [f for f in files if "_sample" not in os.path.basename(f)]
    return files[-1] if files else None


def load_parquet(path: str) -> pl.DataFrame:
    log(f"  파일: {path}")
    df = pl.read_parquet(path)
    log(f"  행 수: {df.height:,}  |  종목 수: {df['Symbol'].n_unique():,}")

    # Date 열 확인 및 변환
    if "Date" in df.columns:
        if df["Date"].dtype == pl.Date:
            pass
        elif df["Date"].dtype == pl.Datetime:
            df = df.with_columns(pl.col("Date").cast(pl.Date))
        elif df["Date"].dtype == pl.Utf8:
            df = df.with_columns(pl.col("Date").str.to_date())
        else:
            df = df.with_columns(pl.col("Date").cast(pl.Date))

    return df


def get_trading_calendar(country: str, start: str, end: str):
    """exchange_calendars를 이용해 영업일 세트를 반환한다."""
    xcal_code = XCAL_MAP.get(country)
    if not xcal_code:
        return None
    cal = xcals.get_calendar(xcal_code)
    sessions = cal.sessions_in_range(start, end)
    return set(s.date() for s in sessions)


# =============================================================================
# 검증 함수들
# =============================================================================

def check_weekends(df: pl.DataFrame, label: str) -> int:
    """주말(토·일)에 VPIN이 산출된 행 수를 확인한다."""
    log(f"\n{'─'*50}")
    log(f"[주말 검증] {label}")
    log(f"{'─'*50}")

    weekend = df.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    n_weekend = weekend.height

    if n_weekend == 0:
        log("  ✓ 주말(토·일) VPIN 없음 — 정상")
    else:
        log(f"  ✗ 주말 VPIN {n_weekend:,}행 발견!")
        # 요일별 분포
        weekend_dist = (
            weekend
            .with_columns(pl.col("Date").dt.weekday().alias("Weekday"))
            .group_by("Weekday")
            .agg(pl.len().alias("Count"))
            .sort("Weekday")
        )
        for row in weekend_dist.iter_rows(named=True):
            day_name = {6: "토요일", 7: "일요일"}.get(row["Weekday"], str(row["Weekday"]))
            log(f"    {day_name}: {row['Count']:,}행")

        # 주말 날짜 샘플
        weekend_dates = sorted(weekend["Date"].unique().to_list())
        sample_dates = weekend_dates[:10]
        log(f"  주말 날짜 샘플 ({len(weekend_dates)}개 중 최대 10개):")
        for d in sample_dates:
            log(f"    {d}")

    return n_weekend


def check_holidays(df: pl.DataFrame, label: str, country: str) -> int:
    """exchange_calendars 기반으로 공휴일에 VPIN이 산출된 행 수를 확인한다."""
    log(f"\n{'─'*50}")
    log(f"[공휴일 검증] {label}")
    log(f"{'─'*50}")

    if not HAS_XCALS:
        log("  ⚠ exchange_calendars 미설치 → 공휴일 검증 스킵")
        log("    설치: pip install exchange-calendars")
        return -1

    dates = df["Date"].unique().sort()
    date_min = str(dates.min())
    date_max = str(dates.max())
    log(f"  데이터 기간: {date_min} ~ {date_max}")

    trading_days = get_trading_calendar(country, date_min, date_max)
    if trading_days is None:
        log(f"  ⚠ XCAL_MAP에 '{country}' 없음 → 공휴일 검증 스킵")
        return -1

    log(f"  거래소 영업일 수: {len(trading_days):,}일")

    # 데이터에 있는 날짜 중 영업일이 아닌 날짜 찾기
    data_dates = set(d for d in dates.to_list())
    non_trading = data_dates - trading_days
    # 주말은 이미 위에서 검사했으므로 여기서는 주중 공휴일만
    weekday_holidays = set()
    for d in non_trading:
        if d.weekday() < 5:  # 월~금
            weekday_holidays.add(d)

    n_holiday_rows = 0
    if weekday_holidays:
        holiday_dates_list = sorted(weekday_holidays)
        holiday_filter = df.filter(
            pl.col("Date").is_in(holiday_dates_list)
        )
        n_holiday_rows = holiday_filter.height
        log(f"  ✗ 주중 공휴일 {len(weekday_holidays)}일에 VPIN {n_holiday_rows:,}행 발견!")
        for d in holiday_dates_list[:15]:
            cnt = holiday_filter.filter(pl.col("Date") == d).height
            log(f"    {d} ({cnt:,}행)")
        if len(weekday_holidays) > 15:
            log(f"    ... 외 {len(weekday_holidays) - 15}일")
    else:
        log("  ✓ 주중 공휴일 VPIN 없음 — 정상")

    return n_holiday_rows


def print_date_stats(df: pl.DataFrame, label: str):
    """날짜 분포 통계를 출력한다."""
    log(f"\n{'─'*50}")
    log(f"[날짜 분포] {label}")
    log(f"{'─'*50}")

    dates = df["Date"].unique().sort()
    log(f"  고유 날짜 수: {dates.len():,}")
    log(f"  기간: {dates.min()} ~ {dates.max()}")

    # 요일별 분포
    weekday_dist = (
        df
        .with_columns(pl.col("Date").dt.weekday().alias("Weekday"))
        .group_by("Weekday")
        .agg([
            pl.len().alias("행수"),
            pl.col("Date").n_unique().alias("날짜수"),
        ])
        .sort("Weekday")
    )
    day_names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    log(f"\n  {'요일':>4}  {'행 수':>12}  {'날짜 수':>8}")
    log(f"  {'─'*4}  {'─'*12}  {'─'*8}")
    for row in weekday_dist.iter_rows(named=True):
        name = day_names.get(row["Weekday"], "?")
        marker = " ← ✗" if row["Weekday"] in (6, 7) else ""
        log(f"  {name:>4}  {row['행수']:>12,}  {row['날짜수']:>8,}{marker}")


def print_vpin_stats(df: pl.DataFrame, label: str):
    """VPIN 기초 통계를 출력한다."""
    log(f"\n{'─'*50}")
    log(f"[VPIN 통계] {label}")
    log(f"{'─'*50}")

    vpin = df["VPIN"].drop_nulls().drop_nans()
    if vpin.is_empty():
        log("  VPIN 값 없음")
        return

    log(f"  유효 VPIN 수: {vpin.len():,}")
    log(f"  평균: {vpin.mean():.6f}")
    log(f"  중앙: {vpin.median():.6f}")
    log(f"  표준편차: {vpin.std():.6f}")
    log(f"  최소: {vpin.min():.6f}")
    log(f"  최대: {vpin.max():.6f}")
    log(f"  Q25: {vpin.quantile(0.25):.6f}")
    log(f"  Q75: {vpin.quantile(0.75):.6f}")


def compare_old_new(old_df: pl.DataFrame, new_df: pl.DataFrame):
    """수정 전/후 결과를 비교한다."""
    log(f"\n{'='*60}")
    log(f"[수정 전/후 비교]")
    log(f"{'='*60}")

    log(f"\n  {'항목':<20} {'수정 전':>15} {'수정 후':>15}")
    log(f"  {'─'*20} {'─'*15} {'─'*15}")
    log(f"  {'전체 행':.<20} {old_df.height:>15,} {new_df.height:>15,}")
    log(f"  {'종목 수':.<20} {old_df['Symbol'].n_unique():>15,} {new_df['Symbol'].n_unique():>15,}")
    log(f"  {'고유 날짜':.<20} {old_df['Date'].n_unique():>15,} {new_df['Date'].n_unique():>15,}")

    old_vpin = old_df["VPIN"].drop_nulls().drop_nans()
    new_vpin = new_df["VPIN"].drop_nulls().drop_nans()
    log(f"  {'VPIN 유효 행':.<20} {old_vpin.len():>15,} {new_vpin.len():>15,}")
    log(f"  {'VPIN 평균':.<20} {old_vpin.mean():>15.6f} {new_vpin.mean():>15.6f}")
    log(f"  {'VPIN 중앙':.<20} {old_vpin.median():>15.6f} {new_vpin.median():>15.6f}")
    log(f"  {'VPIN 표준편차':.<20} {old_vpin.std():>15.6f} {new_vpin.std():>15.6f}")

    # 주말 행 비교
    old_wk = old_df.filter(pl.col("Date").dt.weekday().is_in([6, 7])).height
    new_wk = new_df.filter(pl.col("Date").dt.weekday().is_in([6, 7])).height
    log(f"  {'주말 행':.<20} {old_wk:>15,} {new_wk:>15,}")

    if old_wk > 0 and new_wk == 0:
        log(f"\n  ✓ 주말 VPIN 문제 해결됨 ({old_wk:,}행 → 0행)")
    elif old_wk == 0 and new_wk == 0:
        log(f"\n  ✓ 수정 전/후 모두 주말 VPIN 없음")
    elif new_wk > 0:
        log(f"\n  ✗ 수정 후에도 주말 VPIN {new_wk:,}행 존재")

    # 공통 종목의 VPIN 차이 분석
    common_syms = set(old_df["Symbol"].unique().to_list()) & set(new_df["Symbol"].unique().to_list())
    log(f"\n  공통 종목: {len(common_syms):,}개")

    if len(common_syms) > 0:
        # 종목별 평균 VPIN 비교 (샘플)
        sample_syms = sorted(common_syms)[:5]
        log(f"\n  종목별 평균 VPIN 비교 (샘플 {len(sample_syms)}개):")
        log(f"  {'종목':<15} {'수정 전':>12} {'수정 후':>12} {'차이':>12}")
        log(f"  {'─'*15} {'─'*12} {'─'*12} {'─'*12}")
        for sym in sample_syms:
            old_mean = old_df.filter(pl.col("Symbol") == sym)["VPIN"].drop_nulls().mean()
            new_mean = new_df.filter(pl.col("Symbol") == sym)["VPIN"].drop_nulls().mean()
            if old_mean is not None and new_mean is not None:
                diff = new_mean - old_mean
                log(f"  {sym:<15} {old_mean:>12.6f} {new_mean:>12.6f} {diff:>+12.6f}")


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()

    log(f"{'='*60}")
    log(f"[VPIN 검증] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"{'='*60}")
    log(f"  나라코드: {COUNTRY}")
    log(f"  BASE_DIR: {BASE_DIR}")
    log(f"  exchange_calendars: {'사용 가능' if HAS_XCALS else '미설치'}")

    # ── 파일 탐색 ──
    new_path = NEW_PARQUET or find_latest_parquet(OUTPUT_DIR)
    if new_path is None or not os.path.exists(new_path):
        log(f"\n[Error] 수정 후 parquet을 찾을 수 없습니다.")
        log(f"  탐색 경로: {OUTPUT_DIR}")
        log(f"  NEW_PARQUET을 직접 설정하거나 vpin_r.R을 먼저 실행하세요.")
        sys.exit(1)

    # ── 수정 후 결과 검증 ──
    log(f"\n{'='*60}")
    log(f"[수정 후 결과 로드]")
    log(f"{'='*60}")
    new_df = load_parquet(new_path)

    print_date_stats(new_df, "수정 후")
    print_vpin_stats(new_df, "수정 후")
    n_weekend = check_weekends(new_df, "수정 후")
    n_holiday = check_holidays(new_df, "수정 후", COUNTRY)

    # ── 수정 전 결과 비교 (있으면) ──
    if OLD_PARQUET and os.path.exists(OLD_PARQUET):
        log(f"\n{'='*60}")
        log(f"[수정 전 결과 로드]")
        log(f"{'='*60}")
        old_df = load_parquet(OLD_PARQUET)

        print_date_stats(old_df, "수정 전")
        print_vpin_stats(old_df, "수정 전")
        check_weekends(old_df, "수정 전")
        check_holidays(old_df, "수정 전", COUNTRY)

        compare_old_new(old_df, new_df)
    else:
        log(f"\n  수정 전 parquet 미지정 → 비교 스킵")
        log(f"  비교하려면 OLD_PARQUET 경로를 설정하세요.")

    # ── 최종 판정 ──
    log(f"\n{'='*60}")
    log(f"[최종 판정]")
    log(f"{'='*60}")

    all_pass = True
    if n_weekend > 0:
        log(f"  ✗ 주말 VPIN 존재: {n_weekend:,}행")
        all_pass = False
    else:
        log(f"  ✓ 주말 검증 통과")

    if n_holiday > 0:
        log(f"  ✗ 공휴일 VPIN 존재: {n_holiday:,}행")
        all_pass = False
    elif n_holiday == 0:
        log(f"  ✓ 공휴일 검증 통과")
    else:
        log(f"  ⚠ 공휴일 검증 스킵 (exchange_calendars 필요)")

    if all_pass:
        log(f"\n  ★ 전체 검증 통과 — 영업일 외 VPIN 없음")
    else:
        log(f"\n  ★ 검증 실패 — 위 항목 확인 필요")

    # ── 리포트 저장 ──
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    log(f"\n  리포트 저장: {REPORT_PATH}")
    log(f"  소요 시간: {datetime.now() - start}")