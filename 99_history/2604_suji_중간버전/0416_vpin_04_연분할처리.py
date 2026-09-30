
"""
=============================================================================
[VPIN 파이프라인] tick parquet → 월별 1분봉 캐시 → 기간별 stateful VPIN 계산

메모리 절약 흐름
  월별 parquet
    → Step 1: lazy/streaming으로 월별 1분봉 변환 → monthly_1m_bars/*.parquet
    → Step 2: 종목별 1분봉을 기간 단위로 읽고 VPIN state를 이어서 계산
              → checkpoints/sym_{Symbol}.parquet
              → states/sym_{Symbol}.pkl
    → Step 3: checkpoint lazy scan 병합 → 최종 parquet
    → Step 4: 간단 검증 리포트 생성

중요
  · VPIN의 BVC, volume bucketing, rolling VPIN 산식은 기존 calculate_vpin_for_single_symbol
    로직과 같게 유지하되, bucket/rolling 상태를 기간별로 이어받도록 변경했다.
  · sigma_p와 bucket_size_v는 종목 전체 1분봉 기준으로 먼저 계산한다.
  · state를 저장하므로 중단 후 재개가 가능하다.
=============================================================================
"""

import os
import re
import sys
import glob
import math
import pickle
import shutil
import warnings
import multiprocessing
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# 사용자 설정
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"
COUNTRY = "CN"

FORCE_REPROCESS = False
FORCE_RECOMPUTE_VPIN = False

ROLLING_WINDOW   = 50
BUCKETS_PER_DAY  = 50
T_DIST_DF        = 0.25

# 32GB RAM 기준: 1~2 권장
NUM_WORKERS = 2

# 종목별 1분봉을 몇 년 단위로 끊어 읽을지. state를 넘기므로 overlap은 필요 없다.
VPIN_CHUNK_YEARS = 2

SAVE_STATE = True

REPORT_EXTREME_VPIN_THRESHOLD = 0.50


# =============================================================================
# 경로/상수
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK", "CN"}
_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")
_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

if COUNTRY not in VALID_COUNTRIES:
    raise ValueError(f"COUNTRY 설정 오류: {COUNTRY}")

OUTPUT_ROOT      = os.path.join(BASE_DIR, "0416_output", COUNTRY, "vpin")
MONTHLY_BARS_DIR = os.path.join(OUTPUT_ROOT, "monthly_1m_bars")
CHECKPOINT_DIR   = os.path.join(OUTPUT_ROOT, "checkpoints")
STATE_DIR        = os.path.join(OUTPUT_ROOT, "states")


# =============================================================================
# 유틸리티
# =============================================================================

def safe_filename(value: object) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(value)).strip(" .")
    return cleaned if cleaned else "_EMPTY_"


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
        raise RuntimeError(f"'{country}_YYYY' 패턴 폴더를 찾을 수 없습니다: {base_dir}")

    print(f"\n[파일 탐색] {base_dir}  (나라코드: {country})")
    files = []
    for folder in folders:
        found = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        print(f"  {os.path.basename(folder)}: parquet {len(found)}개")
        files.extend(found)
    print(f"  합계: parquet {len(files)}개")
    return sorted(files)


def list_monthly_bar_files() -> List[str]:
    return sorted(glob.glob(os.path.join(MONTHLY_BARS_DIR, "*.parquet")))


def get_completed_symbols() -> set:
    completed = set()
    for f in glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet")):
        name = os.path.splitext(os.path.basename(f))[0]
        completed.add(name[4:] if name.startswith("sym_") else name)
    return completed


# =============================================================================
# Step 1: 월별 tick parquet → 월별 Symbol별 1분봉 parquet
# =============================================================================

def _build_datetime_expr() -> pl.Expr:
    return pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")


def monthly_file_to_1m_bars(path: str) -> pl.DataFrame:
    lf = (
        pl.scan_parquet(path)
        .select(["Date", "Time", "Symbol", "Price", "Volume"])
        .with_columns(_build_datetime_expr())
        .select(["Datetime", "Symbol", "Price", "Volume"])
        .filter(
            pl.col("Datetime").is_not_null()
            & pl.col("Symbol").is_not_null()
            & pl.col("Price").is_not_null()
            & pl.col("Volume").is_not_null()
        )
        .sort(["Symbol", "Datetime"])
        .group_by_dynamic(
            index_column="Datetime",
            every="1m",
            group_by="Symbol",
            closed="left",
            include_boundaries=False,
        )
        .agg([
            pl.col("Price").last().cast(pl.Float64).alias("Price"),
            pl.col("Volume").sum().cast(pl.Float64).alias("Volume"),
        ])
        .select([
            pl.col("Datetime").cast(pl.Datetime),
            pl.col("Symbol").cast(pl.String),
            pl.col("Price").cast(pl.Float64),
            pl.col("Volume").cast(pl.Float64),
        ])
        .sort(["Symbol", "Datetime"])
    )
    return lf.collect(streaming=True)


def run_step1(base_dir: str, country: str) -> List[str]:
    os.makedirs(MONTHLY_BARS_DIR, exist_ok=True)

    if FORCE_REPROCESS and os.path.isdir(MONTHLY_BARS_DIR):
        print(f"\n[Step 1] 기존 1분봉 캐시 삭제: {MONTHLY_BARS_DIR}")
        shutil.rmtree(MONTHLY_BARS_DIR)
        os.makedirs(MONTHLY_BARS_DIR, exist_ok=True)

    existing = list_monthly_bar_files()
    if not FORCE_REPROCESS and existing:
        print(f"\n[Step 1 스킵] 기존 월별 1분봉 캐시 재사용: {MONTHLY_BARS_DIR}")
        print(f"  월별 1분봉 파일 {len(existing):,}개")
        return existing

    parquet_files = get_parquet_files(base_dir, country)

    print(f"\n{'='*65}")
    print(f"[Step 1] 월별 tick parquet → 월별 Symbol별 1분봉 캐시 [{country}]")
    print(f"{'='*65}\n")

    output_files = []
    total_bar_rows = 0

    for path in tqdm(parquet_files, desc="1분봉 변환"):
        stem = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(MONTHLY_BARS_DIR, f"{safe_filename(stem)}_1m.parquet")
        try:
            bars = monthly_file_to_1m_bars(path)
            bars.write_parquet(out_path, compression="zstd")
            total_bar_rows += bars.height
            output_files.append(out_path)
            del bars
        except Exception as e:
            print(f"  [Error] 1분봉 변환 실패: {os.path.basename(path)}: {e}")

    print(f"\n[Step 1 완료] {MONTHLY_BARS_DIR}")
    print(f"  월별 1분봉 파일 수: {len(output_files):,}")
    print(f"  총 1분봉 행 수    : {total_bar_rows:,}")
    return sorted(output_files)


# =============================================================================
# 종목/기간/파라미터
# =============================================================================

def get_symbols_from_monthly_bars(monthly_bar_files: List[str]) -> List[str]:
    if not monthly_bar_files:
        return []
    symbols = (
        pl.scan_parquet(monthly_bar_files)
        .select(pl.col("Symbol").cast(pl.String))
        .unique()
        .collect(streaming=True)
        .get_column("Symbol")
        .drop_nulls()
        .to_list()
    )
    return sorted(str(s) for s in symbols)


def get_dataset_year_range(monthly_bar_files: List[str]) -> Tuple[int, int]:
    stats = (
        pl.scan_parquet(monthly_bar_files)
        .select([
            pl.col("Datetime").min().alias("min_dt"),
            pl.col("Datetime").max().alias("max_dt"),
        ])
        .collect(streaming=True)
        .row(0, named=True)
    )
    min_dt, max_dt = stats["min_dt"], stats["max_dt"]
    if min_dt is None or max_dt is None:
        raise RuntimeError("monthly_1m_bars에서 Datetime 범위를 계산할 수 없습니다.")
    return int(min_dt.year), int(max_dt.year)


def build_year_periods(min_year: int, max_year: int, chunk_years: int) -> List[Tuple[datetime, datetime]]:
    if chunk_years < 1:
        raise ValueError("VPIN_CHUNK_YEARS는 1 이상이어야 합니다.")
    periods = []
    y = min_year
    while y <= max_year:
        start = datetime(y, 1, 1)
        end_year = min(y + chunk_years, max_year + 1)
        end = datetime(end_year, 1, 1)
        periods.append((start, end))
        y = end_year
    return periods


def load_bars_for_symbol_period(
    symbol: str,
    monthly_bar_files: List[str],
    start_dt: datetime,
    end_dt: datetime,
) -> pl.DataFrame:
    return (
        pl.scan_parquet(monthly_bar_files)
        .filter(
            (pl.col("Symbol") == symbol)
            & (pl.col("Datetime") >= pl.lit(start_dt))
            & (pl.col("Datetime") < pl.lit(end_dt))
        )
        .select(["Datetime", "Symbol", "Price", "Volume"])
        .sort("Datetime")
        .collect(streaming=True)
    )


def compute_symbol_vpin_params(
    symbol: str,
    monthly_bar_files: List[str],
    buckets_per_day: int,
) -> Optional[Dict[str, float]]:
    """
    기존 calculate_vpin_for_single_symbol 내부의 sigma_p, bucket_size_v 계산을
    종목 전체 1분봉 기준으로 수행한다.
    """
    lf = (
        pl.scan_parquet(monthly_bar_files)
        .filter(pl.col("Symbol") == symbol)
        .select(["Datetime", "Price", "Volume"])
        .sort("Datetime")
    )

    stats = (
        lf.with_columns([
            (pl.col("Price") - pl.col("Price").shift(1)).fill_null(0.0).alias("delta_p"),
            pl.when(pl.col("Volume") > 0).then(pl.col("Datetime").dt.date()).otherwise(None).alias("trading_date"),
        ])
        .select([
            pl.len().alias("n_rows"),
            pl.col("delta_p").std().alias("sigma_p"),
            pl.col("Volume").sum().alias("total_volume"),
            pl.col("trading_date").n_unique().alias("n_trading_days"),
        ])
        .collect(streaming=True)
        .row(0, named=True)
    )

    n_rows = int(stats["n_rows"] or 0)
    if n_rows < 2:
        return None

    n_trading_days = int(stats["n_trading_days"] or 0)
    total_volume = float(stats["total_volume"] or 0.0)
    sigma_p = float(stats["sigma_p"] or 0.0) + 1e-9

    if n_trading_days == 0:
        return None

    adv = total_volume / n_trading_days
    bucket_size_v = adv / buckets_per_day
    if bucket_size_v <= 1e-5:
        return None

    return {
        "sigma_p": sigma_p,
        "bucket_size_v": float(bucket_size_v),
        "n_rows": float(n_rows),
        "n_trading_days": float(n_trading_days),
        "total_volume": float(total_volume),
    }


# =============================================================================
# Stateful VPIN 계산
# =============================================================================

@dataclass
class VpinState:
    current_bucket_buy: float = 0.0
    current_bucket_sell: float = 0.0
    current_filled: float = 0.0
    bucket_no: int = 0
    rolling_ois: list = None
    prev_price: Optional[float] = None

    def __post_init__(self):
        if self.rolling_ois is None:
            self.rolling_ois = []


def save_state(state: VpinState, path: str, meta: Dict[str, Any]) -> None:
    with open(path, "wb") as f:
        pickle.dump({"state": state, "meta": meta}, f, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_vpin_for_symbol_period_stateful(
    bar_df: pl.DataFrame,
    symbol: str,
    state: VpinState,
    sigma_p: float,
    bucket_size_v: float,
    rolling_window: int = ROLLING_WINDOW,
    t_df: float = T_DIST_DF,
) -> Tuple[pl.DataFrame, VpinState]:
    """
    기존 VPIN 내부 로직을 기간 chunk 단위로 수행한다.

    기존 로직 유지:
      - delta_p / sigma_p → t_dist.cdf → prob_buy
      - volume bucketing
      - oi = abs(buy - sell)
      - VPIN = rolling sum(oi) / (rolling_window * bucket_size_v)

    변경:
      - bucket 상태와 rolling_ois를 state로 받아 다음 기간에 이어준다.
      - sigma_p, bucket_size_v는 종목 전체 기준으로 미리 계산된 값을 사용한다.
      - prev_price도 state로 넘겨 chunk 경계의 첫 delta_p를 자연스럽게 계산한다.
    """
    schema_def = {
        "Datetime": pl.Datetime,
        "Symbol": pl.String,
        "BucketNo": pl.Int64,
        "VPIN": pl.Float64,
    }
    empty_res = pl.DataFrame(schema=schema_def)

    if bar_df.height == 0:
        return empty_res, state

    times_dt = bar_df["Datetime"].to_list()
    prices = bar_df["Price"].to_numpy()
    volumes = bar_df["Volume"].to_numpy()

    # 원래 np.diff(prices, prepend=prices[0])였지만, chunk 경계에서는 직전 chunk의 마지막 가격을 사용.
    if state.prev_price is None:
        delta_p = np.diff(prices, prepend=prices[0])
    else:
        delta_p = np.empty_like(prices, dtype=float)
        delta_p[0] = prices[0] - state.prev_price
        if len(prices) > 1:
            delta_p[1:] = np.diff(prices)
    state.prev_price = float(prices[-1])

    z_scores = delta_p / sigma_p
    prob_buy = t_dist.cdf(z_scores, df=t_df)

    bucket_records = []

    for i in range(len(volumes)):
        rem_vol = float(volumes[i])
        p_b = float(prob_buy[i])
        curr_time = times_dt[i]

        if rem_vol <= 0:
            continue

        while rem_vol > 0:
            needed = bucket_size_v - state.current_filled
            alloc_vol = min(needed, rem_vol)

            state.current_bucket_buy  += alloc_vol * p_b
            state.current_bucket_sell += alloc_vol * (1.0 - p_b)
            state.current_filled      += alloc_vol
            rem_vol                   -= alloc_vol

            if state.current_filled >= bucket_size_v - 1e-9:
                oi = abs(state.current_bucket_buy - state.current_bucket_sell)
                state.rolling_ois.append(float(oi))
                if len(state.rolling_ois) > rolling_window:
                    state.rolling_ois = state.rolling_ois[-rolling_window:]

                if len(state.rolling_ois) >= rolling_window:
                    vpin = float(sum(state.rolling_ois) / (rolling_window * bucket_size_v))
                else:
                    vpin = np.nan

                bucket_records.append((curr_time, state.bucket_no, vpin))

                state.bucket_no += 1
                state.current_bucket_buy = 0.0
                state.current_bucket_sell = 0.0
                state.current_filled = 0.0

    if not bucket_records:
        return empty_res, state

    result_df = pl.DataFrame({
        "Datetime": [r[0] for r in bucket_records],
        "Symbol": [symbol] * len(bucket_records),
        "BucketNo": [r[1] for r in bucket_records],
        "VPIN": [r[2] for r in bucket_records],
    }).with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64),
    ])
    return result_df, state


# =============================================================================
# Step 2: 종목별 기간 chunk VPIN 계산
# =============================================================================

def worker_process_symbol_stateful(args):
    symbol, monthly_bar_files, periods, checkpoint_path, state_path, rolling_window, buckets_per_day, t_df = args

    try:
        params = compute_symbol_vpin_params(symbol, monthly_bar_files, buckets_per_day)
        empty_schema = {
            "Datetime": pl.Datetime,
            "Symbol": pl.String,
            "BucketNo": pl.Int64,
            "VPIN": pl.Float64,
        }
        if params is None:
            pl.DataFrame(schema=empty_schema).write_parquet(checkpoint_path, compression="zstd")
            return {"symbol": symbol, "status": "empty_params", "n_rows": 0}

        state = VpinState()
        parts = []

        for start_dt, end_dt in periods:
            bars_df = load_bars_for_symbol_period(symbol, monthly_bar_files, start_dt, end_dt)
            if bars_df.is_empty():
                continue

            vpin_part, state = calculate_vpin_for_symbol_period_stateful(
                bars_df,
                symbol=symbol,
                state=state,
                sigma_p=params["sigma_p"],
                bucket_size_v=params["bucket_size_v"],
                rolling_window=rolling_window,
                t_df=t_df,
            )
            if not vpin_part.is_empty():
                parts.append(vpin_part)
            del bars_df

        if parts:
            vpin_df = pl.concat(parts, how="vertical").sort("Datetime")
        else:
            vpin_df = pl.DataFrame(schema=empty_schema)

        vpin_df.write_parquet(checkpoint_path, compression="zstd")

        if SAVE_STATE:
            meta = {
                "symbol": symbol,
                "params": params,
                "rolling_window": rolling_window,
                "buckets_per_day": buckets_per_day,
                "t_df": t_df,
                "periods": [(s.isoformat(), e.isoformat()) for s, e in periods],
                "created_at": datetime.now().isoformat(),
            }
            save_state(state, state_path, meta)

        return {"symbol": symbol, "status": "success", "n_rows": vpin_df.height, "bucket_no": state.bucket_no}

    except Exception as e:
        import traceback
        return {
            "symbol": symbol,
            "status": "error",
            "n_rows": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def run_step2(monthly_bar_files: List[str]) -> Tuple[List[str], int]:
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(STATE_DIR, exist_ok=True)

    if FORCE_REPROCESS or FORCE_RECOMPUTE_VPIN:
        if os.path.isdir(CHECKPOINT_DIR):
            print(f"\n[Step 2] 기존 체크포인트 삭제: {CHECKPOINT_DIR}")
            shutil.rmtree(CHECKPOINT_DIR)
        if os.path.isdir(STATE_DIR):
            print(f"[Step 2] 기존 state 삭제: {STATE_DIR}")
            shutil.rmtree(STATE_DIR)
        os.makedirs(CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(STATE_DIR, exist_ok=True)

    print(f"\n[Step 2 준비] Symbol 목록과 기간 chunk 생성...")
    all_symbols = get_symbols_from_monthly_bars(monthly_bar_files)
    n_total_symbols = len(all_symbols)
    min_year, max_year = get_dataset_year_range(monthly_bar_files)
    periods = build_year_periods(min_year, max_year, VPIN_CHUNK_YEARS)
    completed = get_completed_symbols()

    tasks = []
    used_checkpoint_names: Dict[str, str] = {}

    for symbol in all_symbols:
        checkpoint_symbol_name = safe_filename(symbol)
        base_name = checkpoint_symbol_name
        i = 1
        while checkpoint_symbol_name.lower() in used_checkpoint_names and used_checkpoint_names[checkpoint_symbol_name.lower()] != symbol:
            checkpoint_symbol_name = f"{base_name}__{i}"
            i += 1
        used_checkpoint_names[checkpoint_symbol_name.lower()] = symbol

        if checkpoint_symbol_name in completed:
            continue

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sym_{checkpoint_symbol_name}.parquet")
        state_path = os.path.join(STATE_DIR, f"sym_{checkpoint_symbol_name}.pkl")
        tasks.append((
            symbol, monthly_bar_files, periods, checkpoint_path, state_path,
            ROLLING_WINDOW, BUCKETS_PER_DAY, T_DIST_DF,
        ))

    n_remaining = len(tasks)
    n_completed = n_total_symbols - n_remaining

    print(f"\n{'='*65}")
    print(f"[Step 2] 종목별 1분봉 → 기간별 stateful VPIN 계산 [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  전체 종목      : {n_total_symbols:,}개")
    print(f"  기완료 (재개)  : {n_completed:,}개")
    print(f"  처리 예정      : {n_remaining:,}개")
    print(f"  기간 chunk     : {len(periods)}개 / {VPIN_CHUNK_YEARS}년 단위")
    for s, e in periods:
        print(f"    - {s.date()} ~ {e.date()}")

    if n_remaining == 0:
        return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet"))), n_total_symbols

    n_workers = max(1, NUM_WORKERS if NUM_WORKERS is not None else multiprocessing.cpu_count() - 1)
    n_workers = min(n_workers, n_remaining)

    print(f"  워커 수        : {n_workers}개")
    print(f"  VPIN 파라미터  : rolling_window={ROLLING_WINDOW}, buckets_per_day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"  state 저장     : {STATE_DIR if SAVE_STATE else 'False'}")
    print(f"{'-'*65}\n")

    start = datetime.now()
    success_count = empty_count = error_count = 0
    error_log = []

    with multiprocessing.Pool(processes=n_workers) as pool:
        iterator = pool.imap_unordered(worker_process_symbol_stateful, tasks)
        for res in tqdm(iterator, total=n_remaining, desc="Stateful VPIN 계산"):
            if res is None:
                continue
            status = res.get("status")
            if status == "success":
                success_count += 1
            elif status in {"empty_params", "empty_bars"}:
                empty_count += 1
            elif status == "error":
                error_count += 1
                error_log.append((res.get("symbol"), res.get("error", "")))

    elapsed = datetime.now() - start
    print(f"\n  [Step 2 완료] 소요 시간: {elapsed}")
    print(f"    성공       : {success_count:,}")
    print(f"    빈 결과    : {empty_count:,}")
    print(f"    에러       : {error_count:,}")

    if error_log:
        print("\n  [에러 발생 종목 목록 (최대 10개)]")
        for sym, msg in error_log[:10]:
            print(f"    - {sym}: {msg.splitlines()[0] if msg else ''}")

    return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet"))), n_total_symbols


# =============================================================================
# Step 3: checkpoint 병합
# =============================================================================

def run_step3(checkpoint_files: List[str]) -> Tuple[Optional[str], Optional[str]]:
    print(f"\n{'='*65}")
    print(f"[Step 3] 체크포인트 lazy 병합 → 최종 결과 [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  체크포인트 파일 수: {len(checkpoint_files):,}")

    if not checkpoint_files:
        print("[Warning] 병합할 체크포인트가 없습니다.")
        return None, None

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"vpin_{COUNTRY}_stateful_{run_id}"
    parquet_path = os.path.join(OUTPUT_ROOT, f"{stem}.parquet")
    csv_path = os.path.join(OUTPUT_ROOT, f"{stem}_sample1000.csv")

    lf = (
        pl.scan_parquet(checkpoint_files)
        .select(["Datetime", "Symbol", "BucketNo", "VPIN"])
        .sort(["Symbol", "Datetime"])
    )
    try:
        lf.sink_parquet(parquet_path, compression="zstd")
    except TypeError:
        lf.collect(streaming=True).write_parquet(parquet_path, compression="zstd")

    sample_df = pl.scan_parquet(parquet_path).head(1000).collect(streaming=True)
    sample_df.write_csv(csv_path)

    summary = pl.scan_parquet(parquet_path).select([
        pl.len().alias("n_rows"),
        pl.col("Symbol").n_unique().alias("n_symbols"),
        pl.col("Datetime").min().alias("date_min"),
        pl.col("Datetime").max().alias("date_max"),
    ]).collect(streaming=True).row(0, named=True)

    print("\n  [Step 3 완료]")
    print(f"    전체 parquet: {parquet_path}")
    print(f"    샘플 CSV    : {csv_path}")
    print(f"    전체 행 수  : {summary['n_rows']:,}")
    print(f"    종목 수     : {summary['n_symbols']:,}")
    print(f"    기간        : {summary['date_min']} ~ {summary['date_max']}")
    return parquet_path, stem


# =============================================================================
# Step 4: 간단 검증 리포트
# =============================================================================

def _fmt_num(x, digits=6):
    if x is None:
        return "N/A"
    try:
        if math.isnan(float(x)):
            return "NaN"
    except (TypeError, ValueError):
        return str(x)
    return f"{x:.{digits}f}"


def run_step4_report(final_parquet_path: str, stem: str, n_symbols_input: int, checkpoint_files: List[str]) -> str:
    print(f"\n{'='*65}")
    print(f"[Step 4] 검증 리포트 생성 [{COUNTRY}]")
    print(f"{'='*65}")

    report_path = os.path.join(OUTPUT_ROOT, f"{stem}_report.txt")
    lf = pl.scan_parquet(final_parquet_path)
    valid = pl.col("VPIN").is_not_null() & pl.col("VPIN").is_not_nan()

    basic = lf.select([
        pl.len().alias("n_total_buckets"),
        pl.col("Symbol").n_unique().alias("n_symbols_out"),
        valid.sum().alias("n_valid_vpin"),
        pl.col("Datetime").min().alias("date_min"),
        pl.col("Datetime").max().alias("date_max"),
    ]).collect(streaming=True).row(0, named=True)

    n_total_buckets = basic["n_total_buckets"]
    n_valid_vpin = basic["n_valid_vpin"]
    n_nan_vpin = n_total_buckets - n_valid_vpin

    if n_valid_vpin > 0:
        stats = lf.filter(valid).select([
            pl.col("VPIN").min().alias("min"),
            pl.col("VPIN").quantile(0.01).alias("p01"),
            pl.col("VPIN").median().alias("median"),
            pl.col("VPIN").mean().alias("mean"),
            pl.col("VPIN").quantile(0.99).alias("p99"),
            pl.col("VPIN").max().alias("max"),
            pl.col("VPIN").std().alias("std"),
        ]).collect(streaming=True).row(0, named=True)
        out_of_range = lf.filter(valid & ((pl.col("VPIN") < 0) | (pl.col("VPIN") > 1))).select(pl.len()).collect(streaming=True).item()
        n_extreme = lf.filter(valid & (pl.col("VPIN") > REPORT_EXTREME_VPIN_THRESHOLD)).select(pl.len()).collect(streaming=True).item()
    else:
        stats = {}
        out_of_range = 0
        n_extreme = 0

    bucket_integrity = (
        lf.group_by("Symbol")
        .agg([
            pl.col("BucketNo").min().alias("bmin"),
            pl.col("BucketNo").max().alias("bmax"),
            pl.col("BucketNo").n_unique().alias("bnunique"),
            pl.len().alias("n"),
        ])
        .collect(streaming=True)
    )
    bad_bucket = bucket_integrity.filter(
        (pl.col("bmin") != 0) |
        (pl.col("bmax") + 1 != pl.col("n")) |
        (pl.col("bnunique") != pl.col("n"))
    )

    lines = []
    add = lines.append
    add("=" * 75)
    add(f"[VPIN Stateful 검증 리포트] {COUNTRY}")
    add(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"대상 파일: {stem}.parquet")
    add("=" * 75)
    add("")
    add("[A] 기본 요약")
    add(f"  전체 종목 수(1분봉 캐시): {n_symbols_input:,}")
    add(f"  결과 종목 수            : {basic['n_symbols_out']:,}")
    add(f"  체크포인트 수           : {len(checkpoint_files):,}")
    add(f"  전체 버킷 수            : {n_total_buckets:,}")
    add(f"  유효 VPIN 수            : {n_valid_vpin:,}")
    add(f"  NaN VPIN 수             : {n_nan_vpin:,}")
    add(f"  기간                    : {basic['date_min']} ~ {basic['date_max']}")
    add(f"  기간 chunk              : {VPIN_CHUNK_YEARS}년 단위, state 이어받기")
    add("")
    add("[B] VPIN 분포")
    if n_valid_vpin > 0:
        for k in ["min", "p01", "median", "mean", "p99", "max", "std"]:
            add(f"  {k:<8}: {_fmt_num(stats.get(k), 6)}")
    else:
        add("  유효 VPIN 값이 없습니다.")
    add("")
    add("[C] 정합성")
    add(f"  VPIN 범위 이탈 [0,1]      : {out_of_range:,}건")
    add(f"  BucketNo 불연속/중복 종목 : {bad_bucket.height:,}개")
    if bad_bucket.height > 0:
        add(f"    예시: {bad_bucket['Symbol'].head(10).to_list()}")
    if n_valid_vpin > 0:
        extreme_ratio = n_extreme / n_valid_vpin
        add(f"  VPIN > {REPORT_EXTREME_VPIN_THRESHOLD:.2f}: {n_extreme:,}건 ({extreme_ratio*100:.3f}%)")
    add("")
    add("[D] State 저장 위치")
    add(f"  {STATE_DIR}")
    add("")
    add("=" * 75)
    add("[리포트 끝]")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [Step 4 완료] 리포트: {report_path}")
    return report_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN Stateful 파이프라인] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드       : {COUNTRY}")
    print(f"  입력 루트      : {BASE_DIR}")
    print(f"  출력 루트      : {OUTPUT_ROOT}")
    print(f"  1분봉 캐시     : {MONTHLY_BARS_DIR}")
    print(f"  체크포인트     : {CHECKPOINT_DIR}")
    print(f"  state 저장     : {STATE_DIR}")
    print(f"  워커 수 설정   : {NUM_WORKERS}")
    print(f"  기간 chunk     : {VPIN_CHUNK_YEARS}년")
    print(f"  VPIN 파라미터  : window={ROLLING_WINDOW}, buckets/day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        sys.exit(1)

    monthly_bar_files = run_step1(BASE_DIR, COUNTRY)
    if not monthly_bar_files:
        print("[Error] Step 1에서 월별 1분봉 파일을 생성하지 못했습니다.")
        sys.exit(1)

    checkpoint_files, n_symbols_input = run_step2(monthly_bar_files)
    if not checkpoint_files:
        print("[Error] Step 2에서 체크포인트가 생성되지 않았습니다.")
        sys.exit(1)

    final_parquet_path, stem = run_step3(checkpoint_files)
    if final_parquet_path is None:
        print("[Error] Step 3에서 최종 결과가 생성되지 않았습니다.")
        sys.exit(1)

    report_path = run_step4_report(
        final_parquet_path,
        stem,
        n_symbols_input=n_symbols_input,
        checkpoint_files=checkpoint_files,
    )

    elapsed = datetime.now() - start
    print(f"\n{'='*65}")
    print(f"[파이프라인 완료] 소요 시간: {elapsed}")
    print(f"  최종 parquet: {final_parquet_path}")
    print(f"  리포트      : {report_path}")
    print(f"{'='*65}\n")
