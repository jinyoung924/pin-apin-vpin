"""
=============================================================================
[VPIN 파이프라인 - Step 1] Python 전처리
틱 데이터 ({나라코드}_{연도} 폴더들) → 1분봉 집계 → all_1m_bars.parquet
=============================================================================

PINstimation::vpin() 은 틱 레벨 데이터 {timestamp, price, volume} 을 받아서
내부에서 timebarsize 초 간격으로 재집계한 뒤:
    dp  = last(price) - first(price)   ← 바 내 가격 변동
    tbv = sum(volume)                  ← 바 내 총 거래량
를 계산한다. (model_vpin.R line 614-627 참조)

★ dp=0 문제:
  1분봉 1행당 가격이 1개(last)만 있으면 → R에서 dp = last - first = 0
  해결: Open(= 1분 내 첫 틱 가격) + Close(= 1분 내 마지막 틱 가격) 모두 저장.
  R에서 바 1개당 pseudo-tick 2개로 확장하면 dp = Close - Open ≠ 0 이 된다.

입력 폴더 형식: {BASE_DIR}/{나라코드}_{연도}/
  예) E:\\vpin_project_parquet\\KOR_2019\\KOR_201901.parquet

출력 경로: {BASE_DIR}/R_output/{COUNTRY}/vpin/
  all_1m_bars.parquet
    스키마: Symbol(Utf8), Datetime(Datetime),
            Open(Float64), Close(Float64), Volume(Float64)

실행 순서:
  1) python vpin_pre.py
  2) Rscript vpin_r.R

★ 타임존 수정 (v2):
  원시 틱 데이터는 각 나라의 현지 시간으로 기록되어 있으나 tz 정보가 없다(naive).
  naive Datetime을 그대로 parquet에 쓰면 R의 arrow::read_parquet()가 UTC로 해석한 뒤
  attr(tzone) <- "Asia/Seoul" 등으로 라벨만 교체할 경우 시간이 밀려
  영업일이 아닌 날짜(주말 등)에 VPIN이 산출되는 버그가 발생한다.
  → TZ_MAP을 통해 나라별 현지 타임존을 Datetime에 부여한 뒤 parquet에 저장.
    R에서 read_parquet() 시 올바른 타임존의 POSIXct로 자동 복원된다.

★ 정리 (v3):
  - VALID_COUNTRIES 제거: TZ_MAP 키셋이 곧 유효 나라코드이므로 중복 관리 불필요.
=============================================================================
"""

import os
import re
import sys
import glob
import polars as pl
import warnings
from datetime import datetime
from typing import List

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"
COUNTRY = "CA"
FORCE_REPROCESS = False

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")

# ── 나라별 현지 타임존 매핑 (유효 나라코드 역할 겸용) ──
TZ_MAP = {
    "KOR": "Asia/Seoul",
    "US":  "America/New_York",
    "JP":  "Asia/Tokyo",
    "CA":  "America/Toronto",
    "FR":  "Europe/Paris",
    "GR":  "Europe/Berlin",
    "HK":  "Asia/Hong_Kong",
    "IT":  "Europe/Rome",
    "UK":  "Europe/London",
}

if COUNTRY not in TZ_MAP:
    raise ValueError(
        f"COUNTRY 설정 오류: '{COUNTRY}'\n"
        f"  지원 나라코드: {', '.join(sorted(TZ_MAP))}"
    )

TIMEZONE = TZ_MAP[COUNTRY]
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")


def get_country_folders(base_dir: str, country: str) -> List[str]:
    folders = []
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if entry.is_dir():
            m = _YEAR_FOLDER_RE.match(entry.name)
            if m and m.group(1) == country:
                folders.append(entry.path)
    return folders


def get_parquet_files(base_dir: str, country: str) -> List[str]:
    folders = get_country_folders(base_dir, country)
    if not folders:
        raise RuntimeError(
            f"[Error] '{country}_YYYY' 패턴의 폴더를 찾을 수 없습니다: {base_dir}"
        )
    print(f"\n[파일 탐색] {base_dir}  (나라코드: {country})")
    print(f"  연도 폴더 {len(folders)}개: {[os.path.basename(f) for f in folders]}")

    files = []
    for folder in folders:
        found = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        print(f"  {os.path.basename(folder)}: parquet {len(found)}개")
        files.extend(found)
    print(f"  합계: parquet {len(files)}개")
    return files


# =============================================================================
# Step 1: 틱 데이터 → 1분봉 (Open + Close)
# =============================================================================
#
# 원시 틱 데이터를 1분 간격으로 집계할 때:
#   Open  = 해당 1분 구간 내 첫 번째 틱의 가격  (Price.first)
#   Close = 해당 1분 구간 내 마지막 틱의 가격    (Price.last)
#   Volume = 해당 1분 구간 내 모든 틱의 거래량 합산
#
# 이 두 가격은 서로 다른 틱에서 오는 별개의 값이다.
# 예) 09:00:03 가격=50,000  09:00:47 가격=50,100 이면
#     Open=50,000, Close=50,100 → dp = +100
#
# R의 vpin()은 tick data를 받아 timebarsize초 바로 재집계하므로,
# R에서 bar 1개를 pseudo-tick 2개 (Open→Close) 로 복원하면
# dp = Close - Open 이 올바르게 계산된다.
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """틱 parquet 파일 1개를 읽어 1분봉으로 집계한다."""
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "Time", "Price", "Volume"])
        .filter(pl.col("Volume") > 0)
        .with_columns(pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime"))
        .drop(["Date", "Time"])
        .sort(["Symbol", "Datetime"])
        .collect()
    )

    if df.is_empty():
        print("0 봉")
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime(time_zone=TIMEZONE),
            "Open": pl.Float64, "Close": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            # ── 같은 1분 구간 내에서 서로 다른 틱으로부터 추출 ──
            pl.col("Price").first().alias("Open"),     # 첫 번째 틱 가격 (시가)
            pl.col("Price").last().alias("Close"),     # 마지막 틱 가격 (종가)
            pl.col("Volume").sum().alias("Volume"),    # 총 거래량
        ])
        # ── naive Datetime → 현지 타임존 부여 ──
        .with_columns(
            pl.col("Datetime").dt.replace_time_zone(TIMEZONE)
        )
        .select(["Symbol", "Datetime", "Open", "Close", "Volume"])
        .sort(["Symbol", "Datetime"])
    )
    print(f"{bars.height:,} 봉")
    return bars


def run_preprocessing(base_dir: str, country: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_1m_bars.parquet")

    if not FORCE_REPROCESS and os.path.exists(output_path):
        n = pl.scan_parquet(output_path).select(pl.len()).collect()[0, 0]
        print(f"\n[Step 1 스킵] 기존 파일 재사용: {output_path}  ({n:,} 행)")
        return output_path

    parquet_files = get_parquet_files(base_dir, country)

    print(f"\n{'='*65}")
    print(f"[Step 1] 틱 데이터 → 1분봉 집계  [{country}]")
    print(f"  타임존   : {TIMEZONE}")
    print(f"  출력 스키마: Symbol, Datetime, Open, Close, Volume")
    print(f"  출력 경로  : {output_path}")
    print(f"{'='*65}\n")

    all_bars: List[pl.DataFrame] = []
    for path in parquet_files:
        bars = process_file_to_1m_bars(path)
        if not bars.is_empty():
            all_bars.append(bars)

    if not all_bars:
        raise RuntimeError("[Error] 유효한 봉 데이터가 없습니다.")

    print(f"\n  파일 {len(all_bars)}개 병합 중...")
    full_bars = (
        pl.concat(all_bars, how="vertical")
        .sort(["Symbol", "Datetime"])
    )
    full_bars.write_parquet(output_path, compression="zstd")

    print(f"\n[Step 1 완료] {output_path}")
    print(f"  총 봉 수 : {full_bars.height:,}")
    print(f"  종목 수  : {full_bars['Symbol'].n_unique():,}")
    print(f"  시간 범위: {full_bars['Datetime'].min()} ~ {full_bars['Datetime'].max()}")

    # dp 검증
    n_nonzero = full_bars.filter(pl.col("Open") != pl.col("Close")).height
    pct = n_nonzero / full_bars.height * 100 if full_bars.height > 0 else 0
    print(f"  dp≠0 봉  : {n_nonzero:,} / {full_bars.height:,} ({pct:.1f}%)")
    n_same = full_bars.height - n_nonzero
    print(f"  dp=0 봉  : {n_same:,} ({100-pct:.1f}%)  ← 1분 내 가격 변동 없는 봉 (정상)")

    return output_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 전처리] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드 : {COUNTRY}")
    print(f"  타임존   : {TIMEZONE}")
    print(f"  입력 루트: {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  출력     : {OUTPUT_DIR}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        exit(1)

    run_preprocessing(BASE_DIR, COUNTRY, OUTPUT_DIR)

    print(f"\n[완료] 소요 시간: {datetime.now() - start}")
    print(f"\n다음 단계: Rscript vpin_r.R")