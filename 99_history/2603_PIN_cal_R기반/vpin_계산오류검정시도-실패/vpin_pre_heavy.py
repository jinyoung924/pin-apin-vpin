"""
=============================================================================
[VPIN 파이프라인 - Step 1] Python 전처리
틱 데이터 ({나라코드}_{연도} 폴더들) → 1분봉 집계 → 종목별 parquet
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

출력 경로: {BASE_DIR}/R_output/{COUNTRY}/vpin/bars_by_symbol/
  종목별 parquet 파일들 (예: 005930.parquet, 000660.parquet ...)
    스키마: Symbol(Utf8), Datetime(Datetime),
            Open(Float64), Close(Float64), Volume(Float64)

★ 메모리 절약 전략:
  Phase 1: 원본 파일별 → 임시 1분봉 parquet (파일 1개만 메모리에)
  Phase 2: 임시 파일들 lazy scan → 종목별 parquet로 분배
           (배치 단위로 filter+collect → 소량만 RAM에)

★ 중단 복구:
  Phase 1: 임시 파일 존재 시 스킵
  Phase 2: 종목별 파일 존재 시 스킵

실행 순서:
  1) python vpin_pre.py
  2) Rscript vpin_r.R
=============================================================================
"""

import os
import re
import sys
import glob
import gc
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

BASE_DIR = r"E:\vpin_project_parquet/processing_data"
COUNTRY = "CA"
FORCE_REPROCESS = False

# Phase 2에서 한 번에 처리할 종목 수 (RAM에 따라 조절)
# 예) 4000종목, SYMBOLS_PER_BATCH=200 → 20번 반복
SYMBOLS_PER_BATCH = 200

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

OUTPUT_DIR      = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
TEMP_DIR        = os.path.join(OUTPUT_DIR, "_temp_bars")          # Phase 1 임시
SYMBOL_OUT_DIR  = os.path.join(OUTPUT_DIR, "bars_by_symbol")      # Phase 2 최종


# =============================================================================
# 유틸리티
# =============================================================================

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


def _temp_name(parquet_path: str) -> str:
    """원본 경로 → 충돌 없는 임시 파일명"""
    parent = os.path.basename(os.path.dirname(parquet_path))
    fname = os.path.basename(parquet_path)
    return f"{parent}__{fname}"


def safe_filename(symbol: str) -> str:
    """종목코드를 파일명으로 쓸 수 있게 치환 (예: '/' → '_')"""
    return re.sub(r'[<>:"/\\|?*]', '_', symbol)


# =============================================================================
# Phase 1: 틱 데이터 → 파일별 1분봉 (임시 저장)
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """틱 parquet 파일 1개 → 1분봉 DataFrame"""
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
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Open": pl.Float64, "Close": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            pl.col("Price").first().alias("Open"),
            pl.col("Price").last().alias("Close"),
            pl.col("Volume").sum().alias("Volume"),
        ])
        .select(["Symbol", "Datetime", "Open", "Close", "Volume"])
        .sort(["Symbol", "Datetime"])
    )
    return bars


def run_phase1(parquet_files: List[str]) -> List[str]:
    """Phase 1: 원본 파일별 → 임시 1분봉 parquet"""
    os.makedirs(TEMP_DIR, exist_ok=True)

    total = len(parquet_files)
    skipped = 0
    processed = 0

    print(f"\n{'='*65}")
    print(f"[Phase 1] 틱 → 1분봉 (파일 단위, 임시 저장)")
    print(f"  임시 폴더: {TEMP_DIR}")
    print(f"{'='*65}\n")

    for i, path in enumerate(parquet_files, 1):
        fname = os.path.basename(path)
        temp_out = os.path.join(TEMP_DIR, _temp_name(path))

        if not FORCE_REPROCESS and os.path.exists(temp_out):
            skipped += 1
            print(f"  [{i}/{total}] {fname}  → 스킵")
            continue

        print(f"  [{i}/{total}] {fname} ...", end=" ", flush=True)
        bars = process_file_to_1m_bars(path)
        bars.write_parquet(temp_out, compression="zstd")
        cnt = bars.height
        del bars
        gc.collect()
        processed += 1
        print(f"{cnt:,} 봉")

    print(f"\n  Phase 1 완료: 전체 {total} | 신규 {processed} | 스킵 {skipped}")
    return sorted(glob.glob(os.path.join(TEMP_DIR, "*.parquet")))


# =============================================================================
# Phase 2: 임시 파일들 → 종목별 parquet 분배
# =============================================================================

def run_phase2(temp_files: List[str]):
    """
    Phase 2: 임시 파일들을 lazy scan하여 종목별 parquet로 분배한다.

    작동 원리:
      1) 모든 임시 파일을 scan_parquet (메타데이터만, 데이터 안 읽음)
      2) Symbol 고유값 목록만 먼저 추출
      3) SYMBOLS_PER_BATCH개씩 묶어서:
         - 해당 종목만 filter → collect (배치분만 RAM)
         - 종목별로 slice → 개별 parquet 저장
         - del + gc.collect()로 메모리 해제
      4) 결과: bars_by_symbol/{종목코드}.parquet
    """
    os.makedirs(SYMBOL_OUT_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Phase 2] 임시 파일 → 종목별 parquet 분배")
    print(f"  입력: {TEMP_DIR}  ({len(temp_files)}개)")
    print(f"  출력: {SYMBOL_OUT_DIR}")
    print(f"  배치 크기: {SYMBOLS_PER_BATCH}종목씩")
    print(f"{'='*65}\n")

    # (1) 전체 종목 목록 추출 (Symbol 컬럼만 lazy로)
    print("  종목 목록 추출 중...", end=" ", flush=True)
    all_symbols = (
        pl.scan_parquet(temp_files)
        .select("Symbol")
        .unique()
        .collect()
        .get_column("Symbol")
        .sort()
        .to_list()
    )
    print(f"{len(all_symbols):,}종목")

    # 이미 처리된 종목 스킵 (중단 복구)
    if not FORCE_REPROCESS:
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(SYMBOL_OUT_DIR)
            if f.endswith(".parquet")
        }
        remaining = [s for s in all_symbols if safe_filename(s) not in existing]
        print(f"  이미 완료: {len(all_symbols) - len(remaining)}종목, 남은 종목: {len(remaining)}")
    else:
        remaining = all_symbols

    if not remaining:
        print("  모든 종목 처리 완료!")
        return

    # (2) 배치 단위로 처리
    batches = [
        remaining[i:i + SYMBOLS_PER_BATCH]
        for i in range(0, len(remaining), SYMBOLS_PER_BATCH)
    ]

    total_written = 0
    for b_idx, batch_symbols in enumerate(batches, 1):
        print(f"\n  배치 {b_idx}/{len(batches)} ({len(batch_symbols)}종목) 로딩...", end=" ", flush=True)

        # 이 배치에 해당하는 종목만 filter → collect
        batch_df = (
            pl.scan_parquet(temp_files)
            .filter(pl.col("Symbol").is_in(batch_symbols))
            .sort(["Symbol", "Datetime"])
            .collect()
        )
        print(f"{batch_df.height:,}행")

        # 종목별로 쪼개서 저장
        for symbol in batch_symbols:
            sym_df = batch_df.filter(pl.col("Symbol") == symbol)
            if sym_df.is_empty():
                continue
            out_path = os.path.join(SYMBOL_OUT_DIR, f"{safe_filename(symbol)}.parquet")
            sym_df.write_parquet(out_path, compression="zstd")
            total_written += 1

        del batch_df
        gc.collect()

    print(f"\n  Phase 2 완료: {total_written}종목 저장됨 → {SYMBOL_OUT_DIR}")


# =============================================================================
# 통계 출력
# =============================================================================

def print_summary():
    """최종 결과 통계"""
    sym_files = sorted(glob.glob(os.path.join(SYMBOL_OUT_DIR, "*.parquet")))
    if not sym_files:
        print("\n[경고] 종목별 파일이 없습니다.")
        return

    print(f"\n{'='*65}")
    print(f"[최종 결과 요약]")
    print(f"{'='*65}")
    print(f"  종목별 파일 경로: {SYMBOL_OUT_DIR}")
    print(f"  종목 수: {len(sym_files):,}")

    # 전체 통계 (lazy scan — 집계만 하므로 메모리 최소)
    stats = (
        pl.scan_parquet(sym_files)
        .select([
            pl.len().alias("total_rows"),
            pl.col("Datetime").min().alias("dt_min"),
            pl.col("Datetime").max().alias("dt_max"),
            (pl.col("Open") != pl.col("Close")).sum().alias("dp_nonzero"),
        ])
        .collect()
    )
    total = stats[0, "total_rows"]
    dt_min = stats[0, "dt_min"]
    dt_max = stats[0, "dt_max"]
    dp_nz = stats[0, "dp_nonzero"]

    print(f"  총 봉 수 : {total:,}")
    print(f"  시간 범위: {dt_min} ~ {dt_max}")
    pct = dp_nz / total * 100 if total > 0 else 0
    print(f"  dp≠0 봉  : {dp_nz:,} / {total:,} ({pct:.1f}%)")
    print(f"  dp=0 봉  : {total - dp_nz:,} ({100 - pct:.1f}%)")

    print(f"\n  R에서 읽기:")
    print(f'    ds <- arrow::open_dataset("{SYMBOL_OUT_DIR}")')
    print(f'    df <- ds |> filter(Symbol == "005930") |> collect()')


# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 전처리] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드 : {COUNTRY}")
    print(f"  입력 루트: {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  Phase 1  : {TEMP_DIR}")
    print(f"  Phase 2  : {SYMBOL_OUT_DIR}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        exit(1)

    # Phase 1: 틱 → 파일별 1분봉
    parquet_files = get_parquet_files(BASE_DIR, COUNTRY)
    temp_files = run_phase1(parquet_files)

    # Phase 2: 파일별 1분봉 → 종목별 parquet
    run_phase2(temp_files)

    # 통계
    print_summary()

    print(f"\n[완료] 소요 시간: {datetime.now() - start}")
    print(f"\n다음 단계: Rscript vpin_r.R")