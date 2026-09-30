"""
=============================================================================
[VPIN 파이프라인 - Step 1] Python 전처리
틱 데이터 ({나라코드}_{연도} 폴더들) → 1분봉 집계 → all_1m_bars.parquet
=============================================================================

PIN/APIN과 달리 원시 가격·거래량이 필요하므로 별도 전처리를 수행한다.

입력 폴더 형식: {BASE_DIR}/{나라코드}_{연도}/  ← SAS→parquet 출력 구조와 동일
  예) E:\\vpin_project_parquet\\KOR_2019\\KOR_201901.parquet
      E:\\vpin_project_parquet\\KOR_2020\\KOR_202001.parquet

COUNTRY 설정만 바꾸면 해당 나라의 모든 연도 폴더를 자동으로 스캔한다.

지원 나라코드: KOR  US  JP  CA  FR  GR  HK  IT  UK

출력 경로: {BASE_DIR}/R_output/{COUNTRY}/vpin/
  all_1m_bars.parquet  ← R 입력

★ 수정사항 (dp=0 문제 해결):
  기존: Price(=last price만 저장) → R에서 dp = last - first = 0
  수정: Open(=first price) + Close(=last price) 모두 저장
        → R에서 2개의 pseudo-tick으로 확장하여 dp = Close - Open ≠ 0

실행 순서:
  1) python 02vpin/01_preprocess.py
  2) Rscript 02vpin/02_r_vpin.R

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

# 틱 parquet 루트 폴더
BASE_DIR = r"E:\vpin_project_parquet/processing_data"

# 처리할 나라코드 — 이 코드로 시작하는 모든 연도 폴더를 자동 스캔
# 예) "KOR" → KOR_2019, KOR_2020, KOR_2021 ... 자동 탐색
COUNTRY = "KOR"

# True → 기존 all_1m_bars.parquet 가 있어도 강제 재생성
FORCE_REPROCESS = False

# =============================================================================
# (이하 수정 불필요)
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK"}
_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")

if COUNTRY not in VALID_COUNTRIES:
    raise ValueError(
        f"COUNTRY 설정 오류: '{COUNTRY}'\n"
        f"  지원 나라코드: {', '.join(sorted(VALID_COUNTRIES))}"
    )

OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")


def get_country_folders(base_dir: str, country: str) -> List[str]:
    """base_dir 하위에서 {country}_{연도} 패턴의 폴더를 모두 찾아 정렬 반환."""
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
# Step 1: 틱 데이터 → 1분봉 (Open + Close 가격 모두 저장)
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """틱 parquet 파일 1개를 읽어 1분봉으로 집계한다.
    
    ★ 핵심 수정: Open(first)과 Close(last) 가격을 모두 저장하여
       R에서 dp = Close - Open 을 올바르게 계산할 수 있게 한다.
    """
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
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Open": pl.Float64, "Close": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            pl.col("Price").first().alias("Open"),    # ★ 시가 (first price)
            pl.col("Price").last().alias("Close"),     # ★ 종가 (last price)
            pl.col("Volume").sum().alias("Volume"),
        ])
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
    print(f"  ★ Open(시가) + Close(종가) 모두 저장 (dp=0 문제 해결)")
    print(f"  출력 경로: {output_path}")
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
    
    # ★ dp 검증: Open ≠ Close 인 봉의 비율 출력
    n_nonzero_dp = full_bars.filter(pl.col("Open") != pl.col("Close")).height
    pct = n_nonzero_dp / full_bars.height * 100 if full_bars.height > 0 else 0
    print(f"  dp≠0 봉  : {n_nonzero_dp:,} / {full_bars.height:,} ({pct:.1f}%)")
    
    return output_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 전처리] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드 : {COUNTRY}")
    print(f"  입력 루트: {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  출력     : {OUTPUT_DIR}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        exit(1)

    run_preprocessing(BASE_DIR, COUNTRY, OUTPUT_DIR)

    print(f"\n[완료] 소요 시간: {datetime.now() - start}")
    print(f"\n다음 단계: Rscript 02vpin/02_r_vpin.R")