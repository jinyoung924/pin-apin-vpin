"""
=============================================================================
[VPIN 검증] 수정 전/후 parquet 비교 + 영업일 및 시간대 정밀 진단
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
OLD_PARQUET: Optional[str] = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\vpin_CA_20260402_2323.parquet"
NEW_PARQUET: Optional[str] = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\vpin_CA_20260409_0202.parquet"

# =============================================================================

XCAL_MAP = {"KOR": "XKRX", "US": "XNYS", "JP": "XTKS", "CA": "XTSE", "FR": "XPAR", "GR": "XFRA", "HK": "XHKG", "IT": "XMIL", "UK": "XLON"}
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
REPORT_PATH = os.path.join(OUTPUT_DIR, "vpin_validation_report.txt")
lines: List[str] = []

def log(msg: str = ""):
    print(msg)
    lines.append(msg)

def find_latest_parquet(directory: str, prefix: str = "vpin_") -> Optional[str]:
    pattern = os.path.join(directory, f"{prefix}{COUNTRY}_*.parquet")
    files = [f for f in sorted(glob.glob(pattern)) if "_sample" not in os.path.basename(f)]
    return files[-1] if files else None

def load_parquet(path: str) -> pl.DataFrame:
    df = pl.read_parquet(path)
    if "Date" in df.columns:
        if df["Date"].dtype != pl.Date:
            df = df.with_columns(pl.col("Date").cast(pl.Date))
    return df

def get_trading_calendar(country: str, start: str, end: str):
    xcal_code = XCAL_MAP.get(country)
    if not xcal_code: return None
    cal = xcals.get_calendar(xcal_code)
    sessions = cal.sessions_in_range(start, end)
    return set(s.date() for s in sessions)

def check_anomaly_hours(anomaly_df: pl.DataFrame, title: str):
    """비정상 날짜(주말/공휴일)의 발생 시간대(Hour) 분포를 분석하여 원인을 진단합니다."""
    if anomaly_df.is_empty():
        return
    
    # Starttime에서 시간(Hour) 추출
    hour_dist = (
        anomaly_df
        .with_columns(pl.col("Starttime").dt.hour().alias("Hour"))
        .group_by("Hour")
        .agg(pl.len().alias("Count"))
        .sort("Hour")
    )
    
    log(f"    [🔍 {title} 발생 시간대 분포 - 타임존 버그 vs 데이터 노이즈 판별]")
    log(f"    {'Hour':>4}시 | {'발생 건수':>12}")
    log(f"    {'─'*6}+{'─'*13}")
    for row in hour_dist.iter_rows(named=True):
        log(f"    {row['Hour']:>4}시 | {row['Count']:>12,}")
    log("    💡 해석: 새벽에 몰려있으면 타임존 문제, 장중(09~16시)에 퍼져있으면 원시 데이터 노이즈입니다.\n")

def check_weekends(df: pl.DataFrame) -> int:
    log(f"\n[ 1. 주말(토/일) 데이터 검증 ]")
    log(f"{'='*60}")
    
    weekend = df.filter(pl.col("Date").dt.weekday().is_in([6, 7]))
    n_weekend = weekend.height

    if n_weekend == 0:
        log("  ✅ 주말 데이터 없음 (정상)")
    else:
        dates = weekend["Date"].unique().to_list()
        log(f"  ❌ 주말 VPIN 발생! (총 {len(dates):,}일 / {n_weekend:,}행)")
        
        # 발생 시간대 진단
        check_anomaly_hours(weekend, "주말 데이터")

    return n_weekend

def check_holidays(df: pl.DataFrame, country: str) -> int:
    log(f"\n[ 2. 주중 공휴일 데이터 검증 ]")
    log(f"{'='*60}")

    if not HAS_XCALS:
        log("  ⚠️ exchange_calendars 패키지가 없어 검증을 스킵합니다.")
        return -1

    dates = df["Date"].unique().sort()
    trading_days = get_trading_calendar(country, str(dates.min()), str(dates.max()))
    
    if trading_days is None:
        log(f"  ⚠️ 지원하지 않는 국가 코드입니다. ({country})")
        return -1

    non_trading_weekdays = [d for d in set(dates.to_list()) - trading_days if d.weekday() < 5]
    
    if not non_trading_weekdays:
        log("  ✅ 주중 공휴일 데이터 없음 (정상)")
        return 0

    holiday_df = df.filter(pl.col("Date").is_in(non_trading_weekdays))
    n_holiday = holiday_df.height
    
    log(f"  ❌ 주중 공휴일 VPIN 발생! (총 {len(non_trading_weekdays):,}일 / {n_holiday:,}행)")
    
    # 상위 5개 휴일 샘플
    log("    [발견된 주요 공휴일 샘플]")
    for d in sorted(non_trading_weekdays)[:5]:
        cnt = holiday_df.filter(pl.col("Date") == d).height
        log(f"      - {d}: {cnt:,}행")
    log("")
    
    # 발생 시간대 진단
    check_anomaly_hours(holiday_df, "공휴일 데이터")

    return n_holiday

def print_date_stats(df: pl.DataFrame):
    log(f"\n[ 3. 요일별 VPIN 발생 분포 ]")
    log(f"{'='*60}")
    weekday_dist = (
        df.with_columns(pl.col("Date").dt.weekday().alias("Weekday"))
        .group_by("Weekday").agg([pl.len().alias("Rows"), pl.col("Date").n_unique().alias("Days")])
        .sort("Weekday")
    )
    names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    
    log(f"  {'요일':<4} | {'영업 일수':>10} | {'VPIN 행 수':>15} | {'상태':>4}")
    log(f"  {'─'*5}+{'─'*12}+{'─'*17}+{'─'*6}")
    for r in weekday_dist.iter_rows(named=True):
        status = "❌" if r["Weekday"] in (6, 7) else "✅"
        log(f"  {names[r['Weekday']]:<4} | {r['Days']:>10,} | {r['Rows']:>15,} |  {status}")

if __name__ == "__main__":
    start = datetime.now()
    log(f"============================================================")
    log(f"  VPIN 무결성 검증 리포트 ({start.strftime('%Y-%m-%d %H:%M')})")
    log(f"  국가: {COUNTRY} / 검증 대상: {NEW_PARQUET.split('/')[-1] if NEW_PARQUET else '최신 파일'}")
    log(f"============================================================")

    new_path = NEW_PARQUET or find_latest_parquet(OUTPUT_DIR)
    if not new_path or not os.path.exists(new_path):
        log("파일을 찾을 수 없습니다."); sys.exit(1)

    df = load_parquet(new_path)
    
    print_date_stats(df)
    check_weekends(df)
    check_holidays(df, COUNTRY)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n✅ 리포트 생성 완료: {REPORT_PATH}")