"""
=============================================================================
[VPIN 파이프라인] 틱 데이터 → 1분봉 캐시 → 종목별 VPIN 계산 (32GB RAM 대응)

변경된 처리 흐름
  월별 parquet
    → Step 1: lazy/streaming으로 월별 1분봉 변환 → monthly_1m_bars/*.parquet
    → Step 2: 종목별 1분봉만 읽어서 VPIN 계산 → checkpoints/sym_{Symbol}.parquet
    → Step 3: checkpoint lazy scan 병합 → vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet
    → Step 4: 검증 리포트 생성

주의
  · VPIN 계산 내부 로직(calculate_vpin_for_single_symbol)은 기존 코드와 동일하게 유지.
  · 기존 symbol_ticks 캐시는 만들지 않음. tick 원본을 종목별로 다시 저장하지 않고,
    월별 1분봉 parquet만 저장하여 RAM/디스크 사용량을 줄임.
  · 32GB RAM 환경에서는 NUM_WORKERS=1~2 권장.
=============================================================================
"""

import os
import re
import sys
import glob
import math
import shutil
import warnings
import multiprocessing
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"
COUNTRY = "CN"

# True → Step 1 1분봉 캐시와 Step 2 체크포인트 모두 강제 재생성
FORCE_REPROCESS = False

# True → Step 1 1분봉 캐시는 유지하고 Step 2 체크포인트만 재계산
FORCE_RECOMPUTE_VPIN = False

# ── VPIN 파라미터 ────────────────────────────────────────────────────────────
ROLLING_WINDOW   = 50
BUCKETS_PER_DAY  = 50
T_DIST_DF        = 0.25

# ── 32GB RAM 권장 설정 ───────────────────────────────────────────────────────
# 중국 16년치 tick 데이터는 1~2 권장. 안정성 우선이면 1.
NUM_WORKERS = 1

# Step 2에서 종목별 1분봉 데이터를 읽을 때 한 번에 처리할 종목 수
# 너무 크게 잡으면 filter 결과 collect 시 메모리 피크가 커질 수 있음.
SYMBOL_BATCH_SIZE = 200

# ── 검증 리포트 임계값 ───────────────────────────────────────────────────────
REPORT_NAN_RATIO_THRESHOLD     = 0.05
REPORT_EXTREME_VPIN_THRESHOLD  = 0.50
REPORT_EXTREME_RATIO_THRESHOLD = 0.20


# =============================================================================
# 경로/상수
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK", "CN"}
_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")
_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

if COUNTRY not in VALID_COUNTRIES:
    raise ValueError(
        f"COUNTRY 설정 오류: '{COUNTRY}'\n"
        f"  지원 나라코드: {', '.join(sorted(VALID_COUNTRIES))}"
    )

OUTPUT_ROOT        = os.path.join(BASE_DIR, "0416_output", COUNTRY, "vpin")
MONTHLY_BARS_DIR   = os.path.join(OUTPUT_ROOT, "monthly_1m_bars")
CHECKPOINT_DIR     = os.path.join(OUTPUT_ROOT, "checkpoints")


# =============================================================================
# 파일/심볼 유틸리티
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
    return sorted(files)


def safe_filename(value: object) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(value)).strip(" .")
    return cleaned if cleaned else "_EMPTY_"


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
    """
    Date + Time 컬럼으로 Datetime 생성.
    기존 코드와 동일하게 Date.dt.combine(Time)을 우선 사용한다.
    """
    return pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")


def monthly_file_to_1m_bars(path: str) -> pl.DataFrame:
    """
    월별 tick parquet 하나를 lazy/streaming으로 Symbol별 1분봉으로 압축.
    반환 스키마: Datetime, Symbol, Price, Volume
    """
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

    # Polars 버전별 호환: collect(streaming=True)가 deprecated인 버전도 있으나 동작함.
    return lf.collect(streaming=True)


def run_step1(base_dir: str, country: str) -> List[str]:
    """월별 parquet들을 1분봉 parquet으로 변환하여 캐시."""
    os.makedirs(MONTHLY_BARS_DIR, exist_ok=True)

    if FORCE_REPROCESS and os.path.isdir(MONTHLY_BARS_DIR):
        print(f"\n[Step 1] FORCE_REPROCESS=True → 기존 1분봉 캐시 삭제: {MONTHLY_BARS_DIR}")
        shutil.rmtree(MONTHLY_BARS_DIR)
        os.makedirs(MONTHLY_BARS_DIR, exist_ok=True)

    existing = list_monthly_bar_files()
    if not FORCE_REPROCESS and existing:
        print(f"\n[Step 1 스킵] 기존 월별 1분봉 캐시 재사용: {MONTHLY_BARS_DIR}")
        print(f"  월별 1분봉 파일 {len(existing):,}개")
        return existing

    parquet_files = get_parquet_files(base_dir, country)

    print(f"\n{'='*65}")
    print(f"[Step 1] 월별 tick parquet → 월별 Symbol별 1분봉 캐시  [{country}]")
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


def get_symbols_from_monthly_bars(monthly_bar_files: List[str]) -> List[str]:
    """월별 1분봉 캐시에서 전체 Symbol 목록만 lazy로 추출."""
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


def load_bars_for_symbol(symbol: str, monthly_bar_files: List[str]) -> pl.DataFrame:
    """월별 1분봉 캐시에서 특정 종목의 1분봉만 읽기."""
    return (
        pl.scan_parquet(monthly_bar_files)
        .filter(pl.col("Symbol") == symbol)
        .select(["Datetime", "Symbol", "Price", "Volume"])
        .sort("Datetime")
        .collect(streaming=True)
    )


# =============================================================================
# VPIN 계산 함수 — 내부 로직 원본 유지
# =============================================================================

def convert_ticks_to_symbol_1m_bars(tick_df: pl.DataFrame) -> pl.DataFrame:
    """기존 호환용 함수. 새 파이프라인에서는 Step 1에서 이미 1분봉을 생성하므로 사용하지 않음."""
    if tick_df.is_empty():
        return pl.DataFrame(schema={
            "Datetime": pl.Datetime, "Symbol": pl.String,
            "Price": pl.Float64, "Volume": pl.Float64,
        })

    df = (
        tick_df
        .with_columns(
            pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")
        )
        .sort("Datetime")
    )

    symbol_val = df["Symbol"][0]

    agg_df = (
        df.group_by_dynamic(
            "Datetime",
            every="1m",
            include_boundaries=False,
            closed="left",
        )
        .agg([
            pl.col("Price").last().alias("Price"),
            pl.col("Volume").sum().alias("Volume"),
        ])
    )

    final_df = (
        agg_df
        .with_columns(pl.lit(symbol_val).alias("Symbol"))
        .select(["Datetime", "Symbol", "Price", "Volume"])
    )
    return final_df


def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int = ROLLING_WINDOW,
    buckets_per_day: int = BUCKETS_PER_DAY,
    t_df: float = T_DIST_DF,
) -> pl.DataFrame:
    """1분봉 → BVC 기반 VPIN 계산. 기존 코드 로직 유지."""

    schema_def = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64, "VPIN": pl.Float64,
    }
    empty_res = pl.DataFrame(schema=schema_def)

    if bar_df.height < 2:
        return empty_res

    times_dt = bar_df["Datetime"].to_list()
    prices = bar_df["Price"].to_numpy()
    volumes = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    # BVC
    delta_p = np.diff(prices, prepend=prices[0])
    sigma_p = np.std(delta_p) + 1e-9
    z_scores = delta_p / sigma_p
    prob_buy = t_dist.cdf(z_scores, df=t_df)

    # Bucket size V = ADV / buckets_per_day
    trading_dates = (
        bar_df
        .filter(pl.col("Volume") > 0)
        .select(pl.col("Datetime").dt.date())
        .unique()
    )
    n_trading_days = trading_dates.height
    total_volume = np.sum(volumes)

    if n_trading_days == 0:
        return empty_res

    adv = total_volume / n_trading_days
    bucket_size_v = adv / buckets_per_day

    if bucket_size_v <= 1e-5:
        return empty_res

    # Volume Bucketing
    bucket_records = []
    current_bucket_buy = 0.0
    current_bucket_sell = 0.0
    current_filled = 0.0
    bucket_no = 0

    for i in range(len(volumes)):
        rem_vol = float(volumes[i])
        p_b = float(prob_buy[i])
        curr_time = times_dt[i]

        if rem_vol <= 0:
            continue

        while rem_vol > 0:
            needed = bucket_size_v - current_filled
            alloc_vol = min(needed, rem_vol)

            current_bucket_buy  += alloc_vol * p_b
            current_bucket_sell += alloc_vol * (1.0 - p_b)
            current_filled      += alloc_vol
            rem_vol             -= alloc_vol

            if current_filled >= bucket_size_v - 1e-9:
                oi = abs(current_bucket_buy - current_bucket_sell)
                bucket_records.append((curr_time, bucket_no, oi))

                bucket_no += 1
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                current_filled = 0.0

    if not bucket_records:
        return empty_res

    # Rolling VPIN
    res_times = [r[0] for r in bucket_records]
    res_bucket_nos = [r[1] for r in bucket_records]
    res_ois = np.array([r[2] for r in bucket_records])

    final_vpin_values = np.full(len(res_ois), np.nan)

    if len(res_ois) >= rolling_window:
        kernel = np.ones(rolling_window)
        oi_sums = np.convolve(res_ois, kernel, mode="valid")
        vpin_calced = oi_sums / (rolling_window * bucket_size_v)
        final_vpin_values[rolling_window - 1:] = vpin_calced

    result_df = pl.DataFrame({
        "Datetime": res_times,
        "Symbol": [symbol_str] * len(res_times),
        "BucketNo": res_bucket_nos,
        "VPIN": final_vpin_values,
    }).with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64),
    ])
    return result_df


# =============================================================================
# Step 2: 종목별 1분봉만 읽어서 VPIN 계산
# =============================================================================

def worker_process_symbol(args):
    """워커 프로세스: 특정 Symbol의 1분봉만 읽어 VPIN 계산 후 체크포인트 저장."""
    symbol, monthly_bar_files, checkpoint_path, rolling_window, buckets_per_day, t_df = args

    try:
        bars_df = load_bars_for_symbol(symbol, monthly_bar_files)

        if bars_df.is_empty():
            empty_schema = {
                "Datetime": pl.Datetime, "Symbol": pl.String,
                "BucketNo": pl.Int64, "VPIN": pl.Float64,
            }
            pl.DataFrame(schema=empty_schema).write_parquet(
                checkpoint_path, compression="zstd"
            )
            return {"symbol": symbol, "status": "empty_bars", "n_rows": 0}

        vpin_df = calculate_vpin_for_single_symbol(
            bars_df,
            rolling_window=rolling_window,
            buckets_per_day=buckets_per_day,
            t_df=t_df,
        )

        vpin_df.write_parquet(checkpoint_path, compression="zstd")
        return {"symbol": symbol, "status": "success", "n_rows": vpin_df.height}

    except Exception as e:
        import traceback
        return {
            "symbol": symbol,
            "status": "error",
            "n_rows": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def run_step2(monthly_bar_files: List[str]) -> Tuple[List[str], int]:
    """종목별 VPIN을 계산하고 체크포인트에 저장."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if FORCE_REPROCESS or FORCE_RECOMPUTE_VPIN:
        if os.path.isdir(CHECKPOINT_DIR):
            flag = "FORCE_REPROCESS" if FORCE_REPROCESS else "FORCE_RECOMPUTE_VPIN"
            print(f"\n[Step 2] {flag}=True → 체크포인트 삭제: {CHECKPOINT_DIR}")
            shutil.rmtree(CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\n[Step 2 준비] 월별 1분봉 캐시에서 Symbol 목록 추출...")
    all_symbols = get_symbols_from_monthly_bars(monthly_bar_files)
    n_total_symbols = len(all_symbols)

    completed = get_completed_symbols()

    tasks = []
    used_checkpoint_names: Dict[str, str] = {}

    for symbol in all_symbols:
        checkpoint_symbol_name = safe_filename(symbol)
        # 파일명 충돌 방어. 중국 symbol은 보통 충돌이 없지만, 특수문자/대소문자 충돌 대비.
        base_name = checkpoint_symbol_name
        i = 1
        while checkpoint_symbol_name.lower() in used_checkpoint_names and used_checkpoint_names[checkpoint_symbol_name.lower()] != symbol:
            checkpoint_symbol_name = f"{base_name}__{i}"
            i += 1
        used_checkpoint_names[checkpoint_symbol_name.lower()] = symbol

        if checkpoint_symbol_name in completed:
            continue

        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sym_{checkpoint_symbol_name}.parquet")
        tasks.append((
            symbol, monthly_bar_files, checkpoint_path,
            ROLLING_WINDOW, BUCKETS_PER_DAY, T_DIST_DF,
        ))

    n_remaining = len(tasks)
    n_completed = n_total_symbols - n_remaining

    print(f"\n{'='*65}")
    print(f"[Step 2] 종목별 1분봉 → VPIN 계산  [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  전체 종목      : {n_total_symbols:,}개")
    print(f"  기완료 (재개)  : {n_completed:,}개")
    print(f"  처리 예정      : {n_remaining:,}개")

    if n_remaining == 0:
        print("\n  모든 종목 처리 완료. Step 3으로 건너뜁니다.")
        return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet"))), n_total_symbols

    if NUM_WORKERS is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_workers = max(1, NUM_WORKERS)
    n_workers = min(n_workers, n_remaining)

    print(f"  워커 수        : {n_workers}개")
    print(f"  VPIN 파라미터  : rolling_window={ROLLING_WINDOW}, "
          f"buckets_per_day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"  입력 방식      : 월별 1분봉 parquet lazy filter by Symbol")
    print(f"{'-'*65}\n")

    start = datetime.now()
    success_count = 0
    empty_count   = 0
    error_count   = 0
    error_log     = []

    with multiprocessing.Pool(processes=n_workers) as pool:
        iterator = pool.imap_unordered(worker_process_symbol, tasks)
        for res in tqdm(iterator, total=n_remaining, desc="VPIN 계산"):
            if res is None:
                continue
            status = res.get("status")
            if status == "success":
                success_count += 1
            elif status == "empty_bars":
                empty_count += 1
            elif status == "error":
                error_count += 1
                error_log.append((res["symbol"], res.get("error", "")))

    elapsed = datetime.now() - start
    print(f"\n  [Step 2 완료] 소요 시간: {elapsed}")
    print(f"    성공       : {success_count:,}")
    print(f"    빈 결과    : {empty_count:,}")
    print(f"    에러       : {error_count:,}")

    if error_log:
        print("\n  [에러 발생 종목 목록 (최대 10개)]")
        for sym, msg in error_log[:10]:
            first_line = msg.split("\n")[0] if msg else ""
            print(f"    - {sym}: {first_line}")

    return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet"))), n_total_symbols


# =============================================================================
# Step 3: 체크포인트 lazy scan 병합 → 최종 결과
# =============================================================================

def run_step3(checkpoint_files: List[str]) -> Tuple[Optional[str], Optional[str], Optional[pl.DataFrame]]:
    """체크포인트를 lazy scan으로 병합하여 최종 parquet과 샘플 CSV 저장."""
    print(f"\n{'='*65}")
    print(f"[Step 3] 체크포인트 lazy 병합 → 최종 결과  [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  체크포인트 파일 수: {len(checkpoint_files):,}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if not checkpoint_files:
        print("\n[Warning] 병합할 체크포인트가 없습니다.")
        return None, None, None

    run_id       = datetime.now().strftime("%Y%m%d_%H%M")
    stem         = f"vpin_{COUNTRY}_{run_id}"
    parquet_path = os.path.join(OUTPUT_ROOT, f"{stem}.parquet")
    csv_path     = os.path.join(OUTPUT_ROOT, f"{stem}_sample1000.csv")

    lf = (
        pl.scan_parquet(checkpoint_files)
        .select(["Datetime", "Symbol", "BucketNo", "VPIN"])
        .sort(["Symbol", "Datetime"])
    )

    # 최종 parquet은 lazy sink로 저장해 전체 concat 메모리 피크를 줄인다.
    try:
        lf.sink_parquet(parquet_path, compression="zstd")
    except TypeError:
        # 구버전 Polars 호환용
        lf.collect(streaming=True).write_parquet(parquet_path, compression="zstd")

    # 리포트/샘플용으로 최종 결과를 읽는다. 결과가 너무 크면 여기서도 메모리가 부담될 수 있으므로
    # Step 4는 final_df가 아닌 parquet_path 기반 lazy 집계로 바꿔둔다.
    sample_df = pl.scan_parquet(parquet_path).head(1000).collect(streaming=True)
    sample_df.write_csv(csv_path)

    summary_df = pl.scan_parquet(parquet_path).select([
        pl.len().alias("n_rows"),
        pl.col("Symbol").n_unique().alias("n_symbols"),
    ]).collect(streaming=True)
    n_rows, n_symbols = summary_df.row(0)

    print("\n  [Step 3 완료]")
    print(f"    전체 parquet: {parquet_path}")
    print(f"    샘플 CSV    : {csv_path}")
    print(f"    전체 행 수  : {n_rows:,}")
    print(f"    종목 수     : {n_symbols:,}")

    return parquet_path, stem, sample_df


# =============================================================================
# Step 4: 검증 리포트 생성 — 최종 parquet lazy scan 기반
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


def _pass_fail(ok: bool, warn: bool = False) -> str:
    if ok:
        return "[PASS]"
    if warn:
        return "[WARN]"
    return "[FAIL]"


def run_step4_report(
    final_parquet_path: str,
    stem: str,
    n_symbols_input: int,
    checkpoint_files: List[str],
) -> str:
    """검증 리포트 생성. 대형 결과를 고려해 대부분 lazy 집계로 수행."""
    print(f"\n{'='*65}")
    print(f"[Step 4] 검증 리포트 생성  [{COUNTRY}]")
    print(f"{'='*65}")

    report_path = os.path.join(OUTPUT_ROOT, f"{stem}_report.txt")
    lf = pl.scan_parquet(final_parquet_path)
    valid = pl.col("VPIN").is_not_null() & pl.col("VPIN").is_not_nan()

    lines = []
    add = lines.append

    add("=" * 75)
    add(f"[VPIN 검증 리포트]  {COUNTRY}")
    add(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"대상 파일: {stem}.parquet")
    add("=" * 75)
    add("")

    add("─" * 75)
    add("[A] 기본 통계 요약")
    add("─" * 75)

    basic = lf.select([
        pl.len().alias("n_total_buckets"),
        pl.col("Symbol").n_unique().alias("n_symbols_out"),
        valid.sum().alias("n_valid_vpin"),
        pl.col("Datetime").min().alias("date_min"),
        pl.col("Datetime").max().alias("date_max"),
    ]).collect(streaming=True).row(0, named=True)

    n_total_buckets = basic["n_total_buckets"]
    n_symbols_out = basic["n_symbols_out"]
    n_valid_vpin = basic["n_valid_vpin"]
    n_nan_vpin = n_total_buckets - n_valid_vpin
    n_checkpoints = len(checkpoint_files)
    n_failed = n_symbols_input - n_checkpoints
    n_empty_sym = n_checkpoints - n_symbols_out

    add(f"  전체 종목 수 (1분봉 캐시)     : {n_symbols_input:,}")
    add(f"  처리 성공 종목 수            : {n_symbols_out:,}")
    add(f"  빈 결과 종목 수 (버킷 불가)  : {n_empty_sym:,}")
    add(f"  실패/미처리 종목 수          : {n_failed:,}")
    add("")
    add(f"  전체 버킷 수                 : {n_total_buckets:,}")
    add(f"  유효 VPIN 행 수 (non-NaN)    : {n_valid_vpin:,}")
    add(f"  NaN VPIN 행 수 (초기 49행 등): {n_nan_vpin:,}")
    add("")
    add(f"  Datetime 범위                : {basic['date_min']}  ~  {basic['date_max']}")
    avg_buckets = n_total_buckets / n_symbols_out if n_symbols_out > 0 else 0
    add(f"  종목당 평균 버킷 수          : {avg_buckets:,.1f}")
    add("")

    if n_valid_vpin > 0:
        vpin_stats = lf.filter(valid).select([
            pl.col("VPIN").min().alias("min"),
            pl.col("VPIN").quantile(0.01).alias("p01"),
            pl.col("VPIN").quantile(0.25).alias("p25"),
            pl.col("VPIN").median().alias("median"),
            pl.col("VPIN").mean().alias("mean"),
            pl.col("VPIN").quantile(0.75).alias("p75"),
            pl.col("VPIN").quantile(0.99).alias("p99"),
            pl.col("VPIN").max().alias("max"),
            pl.col("VPIN").std().alias("std"),
        ]).collect(streaming=True).row(0, named=True)

        add("  VPIN 분포 (유효 값 기준):")
        for k in ["min", "p01", "p25", "median", "mean", "p75", "p99", "max", "std"]:
            add(f"    {k:<6} = {_fmt_num(vpin_stats[k])}")
        v_min = vpin_stats["min"]
        v_max = vpin_stats["max"]
    else:
        v_min = v_max = None
        add("  [Warning] 유효 VPIN 값이 없습니다.")
    add("")

    add("─" * 75)
    add("[B] 요일별 VPIN 분포")
    add("─" * 75)

    weekday_names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
    wd_df = (
        lf.filter(valid)
        .with_columns(pl.col("Datetime").dt.weekday().alias("wd"))
        .group_by("wd")
        .agg([
            pl.len().alias("count"),
            pl.col("VPIN").mean().alias("mean_vpin"),
            pl.col("VPIN").std().alias("std_vpin"),
        ])
        .sort("wd")
        .collect(streaming=True)
    )
    total_valid = wd_df["count"].sum() if wd_df.height > 0 else 0
    add(f"  {'요일':<6} {'개수':>12} {'비율':>8} {'평균 VPIN':>14} {'표준편차':>14}")
    add(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*14} {'-'*14}")
    for row in wd_df.iter_rows(named=True):
        wd_num = row["wd"]
        wd_name = weekday_names.get(wd_num, f"?{wd_num}")
        cnt = row["count"]
        ratio = cnt / total_valid * 100 if total_valid > 0 else 0
        add(f"  {wd_name:<6} {cnt:>12,} {ratio:>7.2f}% "
            f"{_fmt_num(row['mean_vpin'], 6):>14} {_fmt_num(row['std_vpin'], 6):>14}")
    add(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*14} {'-'*14}")
    add(f"  {'합계':<6} {total_valid:>12,} {100.00:>7.2f}%")
    add("")

    add("─" * 75)
    add("[C] 정합성 검증 테스트")
    add("─" * 75)

    test_results = []

    if n_valid_vpin > 0:
        out_of_range = lf.filter(valid & ((pl.col("VPIN") < 0) | (pl.col("VPIN") > 1))).select(pl.len()).collect(streaming=True).item()
        status = _pass_fail(out_of_range == 0)
        detail = (f"범위 이탈 {out_of_range:,}건 / 유효 {n_valid_vpin:,}건  "
                  f"(min={_fmt_num(v_min,6)}, max={_fmt_num(v_max,6)})")
    else:
        status = "[WARN]"
        detail = "유효 VPIN 값 없음"
    test_results.append(("Test 1: VPIN 값이 [0, 1] 범위 내인가", status, detail))

    n_post = lf.filter(pl.col("BucketNo") >= (ROLLING_WINDOW - 1)).select(pl.len()).collect(streaming=True).item()
    if n_post > 0:
        n_post_nan = lf.filter(
            (pl.col("BucketNo") >= (ROLLING_WINDOW - 1))
            & (pl.col("VPIN").is_null() | pl.col("VPIN").is_nan())
        ).select(pl.len()).collect(streaming=True).item()
        nan_ratio = n_post_nan / n_post
        status = _pass_fail(nan_ratio <= REPORT_NAN_RATIO_THRESHOLD)
        detail = (f"롤링 완성 이후 NaN {n_post_nan:,} / {n_post:,} "
                  f"= {nan_ratio*100:.3f}%  "
                  f"(임계 {REPORT_NAN_RATIO_THRESHOLD*100:.1f}%)")
    else:
        status = "[WARN]"
        detail = f"BucketNo >= {ROLLING_WINDOW - 1} 인 버킷이 없음"
    test_results.append((f"Test 2: 롤링 윈도우(>= 버킷 {ROLLING_WINDOW-1}) 이후 NaN 비율 정상", status, detail))

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
    bad_rows = bucket_integrity.filter(
        (pl.col("bmin") != 0) |
        (pl.col("bmax") + 1 != pl.col("n")) |
        (pl.col("bnunique") != pl.col("n"))
    )
    status = _pass_fail(bad_rows.height == 0)
    detail = f"불량 종목 {bad_rows.height:,} / 전체 {bucket_integrity.height:,}"
    if bad_rows.height > 0:
        detail += f"  (예시: {bad_rows['Symbol'].head(5).to_list()})"
    test_results.append(("Test 3: BucketNo 연속성 (종목별 0~N-1, 중복 없음)", status, detail))

    dt_violations = (
        lf.sort(["Symbol", "BucketNo"])
        .with_columns(pl.col("Datetime").shift(1).over("Symbol").alias("prev_dt"))
        .filter(pl.col("prev_dt").is_not_null() & (pl.col("Datetime") < pl.col("prev_dt")))
        .select(pl.len())
        .collect(streaming=True)
        .item()
    )
    status = _pass_fail(dt_violations == 0)
    detail = f"역행 발생 {dt_violations:,}건 / 전체 {n_total_buckets:,}건"
    test_results.append(("Test 4: Datetime 단조증가 (종목별)", status, detail))

    n_eligible = bucket_integrity.filter(pl.col("n") >= ROLLING_WINDOW).height
    if n_eligible > 0:
        # 전체 파일에서 직접 검사하므로 eligible symbol list가 큰 경우에도 리스트 필터를 피함.
        early_bad = lf.filter(
            (pl.col("BucketNo") < ROLLING_WINDOW - 1)
            & ~(pl.col("VPIN").is_null() | pl.col("VPIN").is_nan())
        ).select(pl.len()).collect(streaming=True).item()
        late_bad = lf.filter(
            (pl.col("BucketNo") >= ROLLING_WINDOW - 1)
            & (pl.col("VPIN").is_null() | pl.col("VPIN").is_nan())
        ).select(pl.len()).collect(streaming=True).item()
        status = _pass_fail(early_bad == 0 and late_bad == 0)
        detail = (f"초기 {ROLLING_WINDOW-1}개 중 유효값 {early_bad:,}건, "
                  f"완성 구간 무효값 {late_bad:,}건  "
                  f"(검사 대상 종목 {n_eligible:,})")
    else:
        status = "[WARN]"
        detail = f"버킷 >= {ROLLING_WINDOW} 인 종목이 없음"
    test_results.append((f"Test 5: 롤링 정합성 (초기 {ROLLING_WINDOW-1}개 NaN, 이후 값)", status, detail))

    if n_valid_vpin > 0:
        n_extreme = lf.filter(valid & (pl.col("VPIN") > REPORT_EXTREME_VPIN_THRESHOLD)).select(pl.len()).collect(streaming=True).item()
        ext_ratio = n_extreme / n_valid_vpin
        status = _pass_fail(ext_ratio <= REPORT_EXTREME_RATIO_THRESHOLD)
        detail = (f"VPIN > {REPORT_EXTREME_VPIN_THRESHOLD:.2f} 비율: "
                  f"{n_extreme:,} / {n_valid_vpin:,} = {ext_ratio*100:.3f}%  "
                  f"(임계 {REPORT_EXTREME_RATIO_THRESHOLD*100:.1f}%)")
    else:
        status = "[WARN]"
        detail = "유효 VPIN 값 없음"
    test_results.append((f"Test 6: VPIN 극단값(>{REPORT_EXTREME_VPIN_THRESHOLD:.2f}) 비율 정상", status, detail))

    under_rolling = bucket_integrity.filter(pl.col("n") < ROLLING_WINDOW)
    n_under = under_rolling.height
    n_syms_total = bucket_integrity.height
    under_ratio = n_under / n_syms_total if n_syms_total > 0 else 0
    test_results.append((
        f"Test 7: 종목별 최소 버킷 수 (>= {ROLLING_WINDOW}이어야 VPIN 생성)",
        "[INFO]",
        f"버킷 < {ROLLING_WINDOW} (롤링 불가) 종목: {n_under:,} / {n_syms_total:,} = {under_ratio*100:.2f}%",
    ))

    for name, status, detail in test_results:
        add(f"  {status}  {name}")
        add(f"         └ {detail}")
        add("")

    n_pass = sum(1 for _, s, _ in test_results if s == "[PASS]")
    n_fail = sum(1 for _, s, _ in test_results if s == "[FAIL]")
    n_warn = sum(1 for _, s, _ in test_results if s == "[WARN]")
    n_info = sum(1 for _, s, _ in test_results if s == "[INFO]")
    add(f"  [테스트 요약]  PASS {n_pass}  |  FAIL {n_fail}  |  WARN {n_warn}  |  INFO {n_info}")
    add("")

    add("─" * 75)
    add("[D] 상위/하위 10종목 (VPIN 평균 기준)")
    add("─" * 75)

    if n_valid_vpin > 0:
        sym_stats = (
            lf.filter(valid)
            .group_by("Symbol")
            .agg([
                pl.len().alias("n_vpin"),
                pl.col("VPIN").mean().alias("mean_vpin"),
                pl.col("VPIN").std().alias("std_vpin"),
            ])
            .filter(pl.col("n_vpin") >= 10)
            .sort("mean_vpin", descending=True)
            .collect(streaming=True)
        )
        add(f"  (유효 VPIN 10개 이상 종목 대상, 총 {sym_stats.height:,}종목)")
        add("")
        add(f"  [Top 10]   {'Symbol':<16} {'n_vpin':>10} {'mean VPIN':>14} {'std':>12}")
        for row in sym_stats.head(10).iter_rows(named=True):
            add(f"             {str(row['Symbol']):<16} {row['n_vpin']:>10,} {_fmt_num(row['mean_vpin'], 6):>14} {_fmt_num(row['std_vpin'], 6):>12}")
        add("")
        add(f"  [Bottom 10] {'Symbol':<16} {'n_vpin':>10} {'mean VPIN':>14} {'std':>12}")
        for row in sym_stats.tail(10).reverse().iter_rows(named=True):
            add(f"             {str(row['Symbol']):<16} {row['n_vpin']:>10,} {_fmt_num(row['mean_vpin'], 6):>14} {_fmt_num(row['std_vpin'], 6):>12}")
    else:
        add("  [Warning] 유효 VPIN 값이 없어 상/하위 종목을 계산할 수 없습니다.")
    add("")

    add("=" * 75)
    add("[리포트 끝]")
    add("=" * 75)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [Step 4 완료] 리포트: {report_path}")
    print(f"    테스트 결과: PASS {n_pass} | FAIL {n_fail} | WARN {n_warn} | INFO {n_info}")
    return report_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 파이프라인] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드       : {COUNTRY}")
    print(f"  입력 루트      : {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  출력 루트      : {OUTPUT_ROOT}")
    print(f"  1분봉 캐시     : {MONTHLY_BARS_DIR}")
    print(f"  체크포인트     : {CHECKPOINT_DIR}")
    print(f"  워커 수 설정   : {NUM_WORKERS}")
    print(f"  VPIN 파라미터  : window={ROLLING_WINDOW}, buckets/day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        sys.exit(1)

    # Step 1: 월별 tick → 월별 Symbol별 1분봉 캐시
    monthly_bar_files = run_step1(BASE_DIR, COUNTRY)
    if not monthly_bar_files:
        print("[Error] Step 1에서 월별 1분봉 파일을 생성하지 못했습니다.")
        sys.exit(1)

    # Step 2: 종목별 1분봉 → VPIN 체크포인트
    checkpoint_files, n_symbols_input = run_step2(monthly_bar_files)
    if not checkpoint_files:
        print("[Error] Step 2에서 체크포인트가 생성되지 않았습니다.")
        sys.exit(1)

    # Step 3: 체크포인트 lazy 병합
    final_parquet_path, stem, _sample_df = run_step3(checkpoint_files)
    if final_parquet_path is None:
        print("[Error] Step 3에서 최종 결과가 생성되지 않았습니다.")
        sys.exit(1)

    # Step 4: 검증 리포트
    report_path = run_step4_report(
        final_parquet_path,
        stem,
        n_symbols_input=n_symbols_input,
        checkpoint_files=checkpoint_files,
    )

    elapsed = datetime.now() - start
    print(f"\n{'='*65}")
    print(f"[파이프라인 완료]  소요 시간: {elapsed}")
    print(f"  최종 parquet: {final_parquet_path}")
    print(f"  리포트      : {report_path}")
    print(f"{'='*65}\n")
