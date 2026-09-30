"""
=============================================================================
공통 유틸리티 — PIN, APIN, VPIN 모듈이 공유하는 함수들
=============================================================================
"""

import os
import glob
import numpy as np
import polars as pl

import config


# =============================================================================
# 파일 탐색
# =============================================================================

def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """
    base_dir 아래의 KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다.
    year_folders가 None이면 KOR_로 시작하는 모든 폴더를 자동으로 탐색한다.
    """
    if year_folders is None:
        year_folders = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("KOR_")
        ])

    parquet_files = []
    for yf in year_folders:
        folder_path = os.path.join(base_dir, yf)
        if not os.path.exists(folder_path):
            print(f"  [Warning] 폴더 없음: {folder_path}")
            continue
        files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        parquet_files.extend(files)

    print(f"\n[파일 탐색]")
    print(f"  기준 경로   : {base_dir}")
    print(f"  대상 연도   : {year_folders if year_folders else 'All'}")
    print(f"  parquet 파일: {len(parquet_files)}개")
    if parquet_files:
        print(f"  첫 번째     : {os.path.basename(parquet_files[0])}")
        print(f"  마지막      : {os.path.basename(parquet_files[-1])}")

    return parquet_files


# =============================================================================
# [Step 1] 전처리: 틱 데이터 → 일별 B/S 집계 (PIN / APIN 공통)
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일(월별 틱 데이터)을 읽어 일별 매수/매도 건수로 집계한다.

    원본 컬럼 LR의 값: 1 → 매수(Buy), -1 → 매도(Sell)
    반환 스키마: Symbol (Utf8), Date (pl.Date), B (UInt32), S (UInt32)
    """
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    aggregated_df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "LR"])
        .filter(pl.col("LR").is_in([1, -1]))
        .with_columns([
            (pl.col("LR") == 1).cast(pl.UInt32).alias("is_Buy"),
            (pl.col("LR") == -1).cast(pl.UInt32).alias("is_Sell"),
        ])
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("is_Buy").sum().alias("B"),
            pl.col("is_Sell").sum().alias("S"),
        ])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))
        .sort(["Symbol", "Date"])
        .collect()
    )

    if aggregated_df["Date"].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(pl.col("Date").cast(pl.Date))

    print(f"{aggregated_df.height:,} records")
    return aggregated_df


def run_bs_preprocessing(base_dir: str, year_folders: list | None,
                         output_dir: str) -> str:
    """
    모든 월별 parquet를 순회해 일별 B/S를 집계하고,
    결과를 all_daily_bs.parquet 한 파일로 저장한다.
    이미 파일이 존재하고 FORCE_REPROCESS_STEP1=False이면 스킵.

    반환값: 저장된 all_daily_bs.parquet의 전체 경로 (실패 시 빈 문자열)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_daily_bs.parquet")

    if not config.FORCE_REPROCESS_STEP1 and os.path.exists(output_path):
        print(f"\n{'='*65}")
        print(f"[Step 1 스킵] 기존 전처리 파일이 존재합니다: {output_path}")
        print(f"  (다시 생성하려면 config.FORCE_REPROCESS_STEP1 = True 로 설정하세요)")
        print(f"{'='*65}\n")
        return output_path

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("\n[Error] parquet 파일을 찾을 수 없습니다.")
        return ""

    print(f"\n{'='*65}")
    print("[Step 1] 전처리 시작: 틱 데이터 → 일별 B/S 집계")
    print(f"{'='*65}\n")

    all_dfs = []
    skipped = 0

    for i, path in enumerate(parquet_files, 1):
        year_tag = next(
            (part for part in path.replace("\\", "/").split("/")
             if part.startswith("KOR_")), ""
        )
        if i == 1 or year_tag != next(
            (p for p in parquet_files[i - 2].replace("\\", "/").split("/")
             if p.startswith("KOR_")), ""
        ):
            print(f"\n[{year_tag}]")

        df = preprocess_trade_data_polars(path)
        if df.is_empty():
            print(f"    → [Warning] 데이터 없음, 건너뜀")
            skipped += 1
            continue
        all_dfs.append(df)

    if not all_dfs:
        print("\n[Error] 유효한 데이터가 없습니다.")
        return ""

    print(f"\n{'='*65}")
    print("전체 데이터 합산 및 정렬 중...")
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .group_by(["Symbol", "Date"])
        .agg([pl.col("B").sum(), pl.col("S").sum()])
        .sort(["Symbol", "Date"])
    )

    full_df.write_parquet(output_path, compression="zstd")

    print(f"\n{'='*65}")
    print("[Step 1 완료]")
    print(f"{'='*65}")
    print(f"  저장 경로  : {output_path}")
    print(f"  전체 행 수 : {full_df.height:,}")
    print(f"  종목 수    : {full_df['Symbol'].n_unique():,}")
    print(f"  날짜 범위  : {full_df['Date'].min()} ~ {full_df['Date'].max()}")
    print(f"  처리 파일  : {len(parquet_files) - skipped}개  (건너뜀: {skipped}개)")
    print(f"{'='*65}\n")

    return output_path


# =============================================================================
# 영업일 캘린더 구축 및 종목별 B/S 정렬 (PIN / APIN 공통)
# =============================================================================

def build_market_calendar(daily_bs: pl.DataFrame) -> pl.Series:
    """전체 데이터에서 '시장 공통 영업일 캘린더'를 추출한다."""
    return (
        daily_bs.select("Date").unique().sort("Date").get_column("Date")
    )


def align_symbol_to_calendar(
    sym_df: pl.DataFrame,
    calendar: pl.Series,
) -> tuple[np.ndarray, np.ndarray, list]:
    """단일 종목의 B/S 시계열을 영업일 캘린더에 맞춰 정렬한다."""
    aligned = (
        pl.DataFrame({"Date": calendar})
        .join(sym_df.select(["Date", "B", "S"]), on="Date", how="left")
        .with_columns([
            pl.col("B").fill_null(0).cast(pl.Int32),
            pl.col("S").fill_null(0).cast(pl.Int32),
        ])
    )
    return aligned["B"].to_numpy(), aligned["S"].to_numpy(), aligned["Date"].to_list()


# =============================================================================
# 체크포인트 유틸리티 (PIN / APIN 공통)
# =============================================================================

def save_checkpoint(estimates: list[dict], checkpoint_idx: int,
                    session_dir: str, prefix: str,
                    columns: list[str]) -> str:
    """
    추정 결과를 세션 전용 폴더에 체크포인트 파일로 저장한다.

    Args:
        estimates     : dict 리스트
        checkpoint_idx: 체크포인트 번호
        session_dir   : 세션 폴더 경로
        prefix        : 파일 접두사 ("pin" 또는 "apin")
        columns       : 선택할 컬럼 리스트
    """
    os.makedirs(session_dir, exist_ok=True)

    if not estimates:
        df = pl.DataFrame()
    else:
        df = (
            pl.DataFrame(estimates)
            .with_columns(pl.col("Date").cast(pl.Date))
            .select(columns)
        )

    path = os.path.join(session_dir, f"{prefix}_checkpoint_{checkpoint_idx:04d}.parquet")
    df.write_parquet(path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] {df.height:,} records → {os.path.basename(path)}")
    return path


def load_already_done_symbols(session_dir: str, prefix: str) -> set[str]:
    """
    중단 후 재실행 시 세션 폴더의 체크포인트에서 이미 완료된 종목 목록을 복원한다.

    Args:
        session_dir: 세션 폴더 경로
        prefix     : 파일 접두사 ("pin" 또는 "apin")
    """
    if not os.path.exists(session_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(session_dir) if f.startswith(f"{prefix}_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols = set()
    for fname in checkpoint_files:
        path = os.path.join(session_dir, fname)
        df = pl.read_parquet(path, columns=["Symbol"])
        done_symbols.update(df["Symbol"].unique().to_list())

    print(f"[재개 모드] 기존 체크포인트 {len(checkpoint_files)}개 발견 → "
          f"완료된 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


# =============================================================================
# 종목 분리 유틸리티 (PIN / APIN 공통)
# =============================================================================

def prepare_symbol_data(daily_bs: pl.DataFrame,
                        remaining_symbols: list[str]) -> list[tuple]:
    """종목별 DataFrame을 분리하여 (sym, df) 튜플 리스트로 반환한다."""
    symbol_partitions = daily_bs.partition_by("Symbol", maintain_order=False)
    symbol_dfs = {}
    for part in symbol_partitions:
        sym = part["Symbol"][0]
        if sym in remaining_symbols:
            symbol_dfs[sym] = part.sort("Date")

    return [(sym, symbol_dfs[sym]) for sym in remaining_symbols if sym in symbol_dfs]
