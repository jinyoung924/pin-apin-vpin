"""
=============================================================================
VPIN (Volume-Synchronized PIN) 계산 모듈
=============================================================================

[Step 1] 틱 데이터 → 연도별 1분봉 파일
[Step 2] 1분봉 → 종목별 VPIN 파일

BVC: ProbBuy = CDF_t(ΔP/σ; df=0.25)
버킷 크기 V = ADV(해당 연도) / BUCKETS_PER_DAY
VPIN = Σ|V_buy - V_sell| / (ROLLING_WINDOW × V)
"""

import os
import glob
import multiprocessing
import warnings

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

import config
import common

warnings.filterwarnings("ignore")


# =============================================================================
# [Step 1] 틱 데이터 → 연도별 1분봉 파일
# =============================================================================

def _process_one_month(parquet_path: str) -> pl.DataFrame:
    """단일 월 parquet를 읽어 Symbol·분 단위 1분봉으로 집계한다."""
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
        print("0 bars")
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Price": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            pl.col("Price").last().alias("Price"),
            pl.col("Volume").sum().alias("Volume"),
        ])
        .select(["Symbol", "Datetime", "Price", "Volume"])
        .sort(["Symbol", "Datetime"])
    )
    print(f"{bars.height:,} bars")
    return bars


def run_preprocessing(base_dir: str, year_folders: list | None,
                      output_dir: str) -> str:
    """
    모든 월별 parquet를 연도 단위로 묶어 1분봉 집계 후 저장한다.
    반환: 1m_bars 폴더 경로 (실패 시 빈 문자열)
    """
    bars_dir = os.path.join(output_dir, "vpin", "1m_bars")
    os.makedirs(bars_dir, exist_ok=True)

    parquet_files = common.get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("[Error] parquet 파일 없음")
        return ""

    def _year_tag(path: str) -> str:
        return next(
            (p for p in path.replace("\\", "/").split("/") if p.startswith("KOR_")),
            "KOR_UNKNOWN",
        )

    year_groups: dict[str, list[str]] = {}
    for path in parquet_files:
        year_groups.setdefault(_year_tag(path), []).append(path)

    print(f"\n{'='*65}")
    print("[Step 1] 전처리 시작: 틱 데이터 → 1분봉 집계")
    print(f"  저장 폴더 : {bars_dir}")
    print(f"{'='*65}\n")

    produced_files: list[str] = []
    total_skipped = 0

    for year_tag in sorted(year_groups):
        out_path = os.path.join(bars_dir, f"1m_bars_{year_tag}.parquet")

        if not config.VPIN_FORCE_REPROCESS_STEP1 and os.path.exists(out_path):
            n = pl.scan_parquet(out_path).select(pl.len()).collect()[0, 0]
            print(f"[{year_tag}] 스킵 (기존: {n:,}행)")
            produced_files.append(out_path)
            continue

        paths = year_groups[year_tag]
        print(f"\n[{year_tag}] ({len(paths)}개 파일)")

        month_dfs: list[pl.DataFrame] = []
        for path in paths:
            df = _process_one_month(path)
            if df.is_empty():
                print(f"    → [Warning] 데이터 없음, 건너뜀")
                total_skipped += 1
            else:
                month_dfs.append(df)

        if not month_dfs:
            print(f"  [{year_tag}] 유효 데이터 없음, 건너뜀")
            continue

        print(f"  [{year_tag}] 정렬 및 저장 중...", end=" ", flush=True)
        year_df = pl.concat(month_dfs, how="vertical").sort(["Symbol", "Datetime"])
        del month_dfs
        year_df.write_parquet(out_path, compression="zstd")
        produced_files.append(out_path)
        print(f"{year_df.height:,}행 → {os.path.basename(out_path)}")
        del year_df

    if not produced_files:
        print("[Error] 유효한 데이터가 없습니다.")
        return ""

    stats = (
        pl.scan_parquet(produced_files)
        .select([
            pl.len().alias("rows"),
            pl.col("Symbol").n_unique().alias("symbols"),
            pl.col("Datetime").min().alias("dt_min"),
            pl.col("Datetime").max().alias("dt_max"),
        ])
        .collect()
    )
    print(f"\n{'='*65}")
    print("[Step 1 완료]")
    print(f"{'='*65}")
    print(f"  저장 폴더  : {bars_dir}")
    print(f"  연도 파일  : {len(produced_files)}개")
    for f in produced_files:
        n = pl.scan_parquet(f).select(pl.len()).collect()[0, 0]
        print(f"    {os.path.basename(f)}: {n:,}행")
    print(f"  전체 행 수 : {stats['rows'][0]:,}")
    print(f"  종목 수    : {stats['symbols'][0]:,}")
    print(f"  시간 범위  : {stats['dt_min'][0]} ~ {stats['dt_max'][0]}")
    print(f"{'='*65}\n")

    return bars_dir


# =============================================================================
# [Step 2] VPIN 계산 핵심 함수
# =============================================================================

def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int | None = None,
    buckets_per_day: int | None = None,
) -> pl.DataFrame:
    """단일 종목의 1분봉 데이터를 받아 VPIN을 계산한다."""
    rolling_window  = rolling_window  or config.VPIN_ROLLING_WINDOW
    buckets_per_day = buckets_per_day or config.VPIN_BUCKETS_PER_DAY

    schema = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64,    "VPIN":   pl.Float64,
    }
    empty = pl.DataFrame(schema=schema)

    if bar_df.height < 2:
        return empty

    times_dt   = bar_df["Datetime"].to_list()
    prices     = bar_df["Price"].to_numpy()
    volumes    = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    year_stats = (
        bar_df
        .with_columns(pl.col("Datetime").dt.year().alias("Year"))
        .group_by("Year")
        .agg([
            pl.col("Volume").sum().alias("total_vol"),
            pl.col("Datetime").dt.date().n_unique().alias("n_days"),
        ])
    )
    year_bucket_map: dict[int, float] = {
        row["Year"]: (row["total_vol"] / row["n_days"] / buckets_per_day
                      if row["n_days"] > 0 else 1e-5)
        for row in year_stats.iter_rows(named=True)
    }
    if not year_bucket_map:
        return empty

    delta_p  = np.diff(prices, prepend=prices[0])
    prob_buy = t_dist.cdf(delta_p / (np.std(delta_p) + 1e-9), df=0.25)

    records:       list[tuple] = []
    current_buy    = 0.0
    current_sell   = 0.0
    current_filled = 0.0
    bucket_no      = 0
    last_year      = None
    bucket_size_v  = 0.0

    for i in range(len(volumes)):
        rem_vol   = float(volumes[i])
        p_b       = float(prob_buy[i])
        curr_time = times_dt[i]
        curr_year = curr_time.year

        if rem_vol <= 0:
            continue

        if curr_year != last_year:
            bucket_size_v = year_bucket_map.get(curr_year, 1e-5)
            last_year     = curr_year

        if bucket_size_v <= 1e-5:
            continue

        while rem_vol > 0:
            needed    = bucket_size_v - current_filled
            alloc     = min(needed, rem_vol)
            current_buy    += alloc * p_b
            current_sell   += alloc * (1.0 - p_b)
            current_filled += alloc
            rem_vol        -= alloc

            if current_filled >= bucket_size_v - 1e-9:
                records.append((curr_time, bucket_no,
                                 abs(current_buy - current_sell)))
                bucket_no      += 1
                current_buy     = 0.0
                current_sell    = 0.0
                current_filled  = 0.0

    if not records:
        return empty

    res_times  = [r[0] for r in records]
    res_nos    = [r[1] for r in records]
    res_ois    = np.array([r[2] for r in records])
    bucket_bsv = np.array([year_bucket_map.get(t.year, 1e-5) for t in res_times])

    final_vpin = np.full(len(res_ois), np.nan)
    if len(res_ois) >= rolling_window:
        oi_sums  = np.convolve(res_ois, np.ones(rolling_window), mode="valid")
        last_bsv = bucket_bsv[rolling_window - 1:]
        final_vpin[rolling_window - 1:] = oi_sums / (rolling_window * last_bsv)

    return pl.DataFrame({
        "Datetime": res_times,
        "Symbol":   [symbol_str] * len(res_times),
        "BucketNo": res_nos,
        "VPIN":     final_vpin,
    }).with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64),
    ])


# =============================================================================
# 멀티프로세싱 워커
# =============================================================================

_ROLLING_WINDOW:  int = 50
_BUCKETS_PER_DAY: int = 50


def init_worker(rolling_window: int, buckets_per_day: int) -> None:
    global _ROLLING_WINDOW, _BUCKETS_PER_DAY
    _ROLLING_WINDOW  = rolling_window
    _BUCKETS_PER_DAY = buckets_per_day


def process_symbol_worker(args: tuple) -> str | None:
    """sym_input 읽기 → VPIN 계산 → sym_result 저장."""
    symbol, input_path, output_path = args
    try:
        bar_df = pl.read_parquet(input_path)
        if bar_df.is_empty():
            return None
        vpin_df = calculate_vpin_for_single_symbol(
            bar_df, _ROLLING_WINDOW, _BUCKETS_PER_DAY
        )
        if vpin_df.is_empty():
            return None
        vpin_df.write_parquet(output_path, compression="zstd")
        return output_path
    except Exception as e:
        import traceback
        print(f"\n  [Error] {symbol}: {e}\n{traceback.format_exc()}")
        return None


# =============================================================================
# 유틸리티
# =============================================================================

def load_done_symbols(results_dir: str) -> set[str]:
    """vpin_results/ 폴더에서 완료된 종목 목록을 반환한다."""
    if not os.path.exists(results_dir):
        return set()
    done = {
        fname[len("sym_"):-len(".parquet")]
        for fname in os.listdir(results_dir)
        if fname.startswith("sym_") and fname.endswith(".parquet")
    }
    if done:
        print(f"[재개 모드] 완료 종목 {len(done):,}개 발견 → 건너뜀")
    return done


def move_to_results(result_paths: list[str], results_dir: str) -> int:
    """워커 결과 파일을 vpin_results/ 로 rename한다."""
    moved = 0
    for src in result_paths:
        fname = os.path.basename(src)
        dst   = os.path.join(results_dir, fname)
        try:
            os.replace(src, dst)
            moved += 1
        except Exception as e:
            print(f"\n  [Warning] 이동 실패: {fname} ({e})")
    return moved


# =============================================================================
# 메인 VPIN 계산 함수
# =============================================================================

def run(bars_dir: str, output_dir: str, run_id: str,
        year_filter: list[int] | None = None,
        checkpoint_n: int | None = None) -> str:
    """
    1m_bars/ 연도별 파일을 읽어 종목별 VPIN을 계산하고
    vpin_results/{sym}.parquet로 저장한다.

    반환: vpin_results 폴더 경로 (실패 시 빈 문자열)
    """
    checkpoint_n = checkpoint_n or config.CHECKPOINT_N

    results_dir    = os.path.join(output_dir, "vpin", "results")
    sym_input_dir  = os.path.join(output_dir, "vpin", "sessions", run_id, "sym_input")
    sym_result_dir = os.path.join(output_dir, "vpin", "sessions", run_id, "sym_result")

    for d in [results_dir, sym_input_dir, sym_result_dir]:
        os.makedirs(d, exist_ok=True)

    bars_files = sorted(glob.glob(os.path.join(bars_dir, "1m_bars_KOR_*.parquet")))
    if not bars_files:
        print(f"[Error] 1m_bars 파일 없음: {bars_dir}")
        return ""

    if year_filter:
        bars_files = [
            f for f in bars_files
            if any(str(y) in os.path.basename(f) for y in year_filter)
        ]

    print(f"\n{'='*65}")
    print("[Step 2] VPIN 계산 시작")
    print(f"{'='*65}")
    print(f"  1m_bars 폴더: {bars_dir}")
    print(f"  사용 파일   : {len(bars_files)}개")
    for f in bars_files:
        print(f"    {os.path.basename(f)}")
    print(f"  결과 폴더   : {results_dir}")
    print(f"  RUN_ID      : {run_id}")

    print(f"\n[파일 메타 확인]")
    meta = (
        pl.scan_parquet(bars_files)
        .select([
            pl.len().alias("rows"),
            pl.col("Symbol").n_unique().alias("symbols"),
            pl.col("Datetime").min().alias("dt_min"),
            pl.col("Datetime").max().alias("dt_max"),
        ])
        .collect()
    )
    print(f"  전체 행 수  : {meta['rows'][0]:,}")
    print(f"  종목 수     : {meta['symbols'][0]:,}")
    print(f"  시간 범위   : {meta['dt_min'][0]} ~ {meta['dt_max'][0]}")

    num_cores = multiprocessing.cpu_count()
    print(f"\n  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {config.IMAP_CHUNKSIZE} 종목/청크")
    print(f"  롤링 버킷   : {config.VPIN_ROLLING_WINDOW}개")
    print(f"  버킷/일     : {config.VPIN_BUCKETS_PER_DAY}개")

    done_symbols   = load_done_symbols(results_dir)
    all_symbols    = (
        pl.scan_parquet(bars_files)
        .select("Symbol").unique().collect()["Symbol"].sort().to_list()
    )
    remaining_list = [s for s in all_symbols if s not in done_symbols]

    print(f"\n  전체 종목   : {len(all_symbols):,}개")
    print(f"  완료 종목   : {len(done_symbols):,}개")
    print(f"  처리 대상   : {len(remaining_list):,}개\n")

    if not remaining_list:
        print("  모든 종목이 완료됐습니다.")
        return results_dir

    # sym_input 파일 생성
    need_write    = set(s for s in remaining_list
                        if not os.path.exists(
                            os.path.join(sym_input_dir, f"sym_{s}.parquet")))
    already_split = [s for s in remaining_list if s not in need_write]
    tasks: list[tuple] = []

    for sym in already_split:
        tasks.append((sym,
                      os.path.join(sym_input_dir,  f"sym_{sym}.parquet"),
                      os.path.join(sym_result_dir, f"sym_{sym}.parquet")))

    if need_write:
        print(f"[종목 분할] {len(need_write)}개 종목 — "
              f"연도 파일 {len(bars_files)}개 순차 1회 읽기\n")

        sym_bufs: dict[str, list[pl.DataFrame]] = {s: [] for s in need_write}

        for bar_file in bars_files:
            print(f"  읽기: {os.path.basename(bar_file)} ...", end=" ", flush=True)
            year_df = (
                pl.scan_parquet(bar_file)
                .filter(pl.col("Symbol").is_in(need_write))
                .collect()
            )
            if year_df.is_empty():
                print("0행, 건너뜀")
                del year_df
                continue

            for part in year_df.partition_by("Symbol", maintain_order=False):
                sym = part["Symbol"][0]
                if sym in sym_bufs:
                    sym_bufs[sym].append(part.sort("Datetime"))

            del year_df
            print("분배 완료")

        print(f"\n  종목별 sym_input 저장 중...")
        saved = 0
        for sym, dfs in sym_bufs.items():
            if not dfs:
                continue
            inp = os.path.join(sym_input_dir,  f"sym_{sym}.parquet")
            out = os.path.join(sym_result_dir, f"sym_{sym}.parquet")
            pl.concat(dfs, how="vertical").write_parquet(inp, compression="zstd")
            tasks.append((sym, inp, out))
            saved += 1

        sym_bufs.clear()
        print(f"  저장 완료: {saved}개 종목\n")

    print(f"  분할 완료: 신규 {len(need_write)}개 / 재사용 {len(already_split)}개\n")

    # 병렬 VPIN 계산
    print(f"{'='*65}")
    print("VPIN 계산 시작...")
    print(f"{'='*65}\n")

    pending_paths:  list[str] = []
    processed_count = 0
    moved_total     = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(config.VPIN_ROLLING_WINDOW, config.VPIN_BUCKETS_PER_DAY),
    ) as pool:
        for path in tqdm(
            pool.imap_unordered(
                process_symbol_worker, tasks, chunksize=config.IMAP_CHUNKSIZE
            ),
            total=len(tasks),
            desc="  Estimating",
        ):
            if path is not None:
                pending_paths.append(path)
            processed_count += 1

            if processed_count % checkpoint_n == 0 and pending_paths:
                moved = move_to_results(pending_paths, results_dir)
                moved_total += moved
                print(f"\n  [이동] {moved}개 → vpin_results/  (누적 {moved_total}개)")
                pending_paths = []

    if pending_paths:
        moved = move_to_results(pending_paths, results_dir)
        moved_total += moved
        print(f"\n  [이동] {moved}개 → vpin_results/  (누적 {moved_total}개)")

    n_files = len([f for f in os.listdir(results_dir)
                   if f.startswith("sym_") and f.endswith(".parquet")])
    print(f"\n{'='*65}")
    print(f"[Step 2 완료] vpin_results/ 에 {n_files:,}개 종목 파일 저장")
    print(f"{'='*65}")

    return results_dir
