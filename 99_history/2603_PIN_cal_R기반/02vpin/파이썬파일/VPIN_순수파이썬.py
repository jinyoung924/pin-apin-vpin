"""
=============================================================================
VPIN (Volume-Synchronized PIN) — 순수 Python 단일 파일
=============================================================================

Phase 1: 틱 데이터 ({COUNTRY}_{YYYY} 폴더) → 파일별 1분봉 (임시 저장)
Phase 2: 임시 1분봉 → 종목별 parquet 분배 (bars_by_symbol/)
Step  3: 종목별 parquet → VPIN 계산 → results/

BVC 모델 : ProbBuy = CDF_t(ΔP/σ; df=0.25)
버킷 크기: V = ADV(해당 연도) / BUCKETS_PER_DAY
VPIN     : Σ|V_buy - V_sell| / (ROLLING_WINDOW × V)

폴더 구조 (vpin_pre_heavy.py 방식):
  입력: {BASE_DIR}/{COUNTRY}_{YYYY}/*.parquet          ← 틱 데이터
  임시: {BASE_DIR}/R_output/{COUNTRY}/vpin/_temp_bars/  ← Phase 1 중간물
  분배: {BASE_DIR}/R_output/{COUNTRY}/vpin/bars_by_symbol/  ← Phase 2
  결과: {BASE_DIR}/R_output/{COUNTRY}/vpin/results/     ← VPIN 결과

실행: python vpin_pure_python.py
=============================================================================
"""

import os
import re
import gc
import glob
import sys
import multiprocessing
import warnings
from datetime import datetime
from typing import List

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

BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY  = "CA"

# ── 재처리 옵션 ─────────────────────────────────────────────────────────────
FORCE_REPROCESS_PHASE1 = False   # True면 Phase 1 임시 파일을 강제 재생성
FORCE_REPROCESS_PHASE2 = False   # True면 Phase 2 종목별 파일을 강제 재생성
FORCE_REPROCESS_VPIN   = False   # True면 VPIN 결과를 강제 재계산

# ── VPIN 파라미터 ───────────────────────────────────────────────────────────
VPIN_ROLLING_WINDOW = 50         # 롤링 윈도우 버킷 수 (samplength)
VPIN_BUCKETS_PER_DAY = 50        # 하루당 버킷 수

# ── 처리 설정 ───────────────────────────────────────────────────────────────
SYMBOLS_PER_BATCH = 200          # Phase 2 배치당 종목 수 (RAM에 따라 조절)
CHECKPOINT_N      = 500          # VPIN 계산 중 체크포인트 간격 (종목 수)
IMAP_CHUNKSIZE    = 4            # imap_unordered 청크 크기
MIN_ROWS          = 500          # 종목당 최소 행 수 (미만 스킵)


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

# ── 경로 설정 (vpin_pre_heavy.py 방식) ──────────────────────────────────────
OUTPUT_DIR     = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
TEMP_DIR       = os.path.join(OUTPUT_DIR, "_temp_bars")
SYMBOL_OUT_DIR = os.path.join(OUTPUT_DIR, "bars_by_symbol")
RESULTS_DIR    = os.path.join(OUTPUT_DIR, "results")


# =============================================================================
# 유틸리티 함수
# =============================================================================

def safe_filename(symbol: str) -> str:
    """종목코드를 파일명으로 쓸 수 있게 치환 (예: '/' → '_')"""
    return re.sub(r'[<>:"/\\|?*]', '_', symbol)


def get_country_folders(base_dir: str, country: str) -> List[str]:
    """BASE_DIR 아래에서 {COUNTRY}_{YYYY} 패턴의 폴더를 찾는다."""
    folders = []
    if not os.path.isdir(base_dir):
        return folders
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if entry.is_dir():
            m = _YEAR_FOLDER_RE.match(entry.name)
            if m and m.group(1) == country:
                folders.append(entry.path)
    return folders


def get_parquet_files(base_dir: str, country: str) -> List[str]:
    """나라코드에 해당하는 모든 틱 parquet 파일 경로를 반환한다."""
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
    fname  = os.path.basename(parquet_path)
    return f"{parent}__{fname}"


# =============================================================================
# Phase 1: 틱 데이터 → 파일별 1분봉 (임시 저장)
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """틱 parquet 파일 1개 → 1분봉 DataFrame (Open, Close, Volume)"""
    empty_schema = {
        "Symbol": pl.String, "Datetime": pl.Datetime,
        "Open": pl.Float64, "Close": pl.Float64, "Volume": pl.Float64,
    }

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
        return pl.DataFrame(schema=empty_schema)

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
    """Phase 1: 원본 틱 파일별 → 임시 1분봉 parquet"""
    os.makedirs(TEMP_DIR, exist_ok=True)

    total     = len(parquet_files)
    skipped   = 0
    processed = 0

    print(f"\n{'='*65}")
    print(f"[Phase 1] 틱 → 1분봉 (파일 단위, 임시 저장)")
    print(f"  임시 폴더: {TEMP_DIR}")
    print(f"{'='*65}\n")

    for i, path in enumerate(parquet_files, 1):
        fname    = os.path.basename(path)
        temp_out = os.path.join(TEMP_DIR, _temp_name(path))

        if not FORCE_REPROCESS_PHASE1 and os.path.exists(temp_out):
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
    Phase 2: 임시 1분봉 파일들을 lazy scan하여 종목별 parquet로 분배한다.

    작동 원리:
      1) 모든 임시 파일을 scan_parquet (메타데이터만)
      2) Symbol 고유값 목록 추출
      3) SYMBOLS_PER_BATCH 단위로 filter → collect → 개별 저장
      4) 결과: bars_by_symbol/{종목코드}.parquet
    """
    os.makedirs(SYMBOL_OUT_DIR, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Phase 2] 임시 파일 → 종목별 parquet 분배")
    print(f"  입력: {TEMP_DIR}  ({len(temp_files)}개)")
    print(f"  출력: {SYMBOL_OUT_DIR}")
    print(f"  배치 크기: {SYMBOLS_PER_BATCH}종목씩")
    print(f"{'='*65}\n")

    # (1) 전체 종목 목록 추출
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
    if not FORCE_REPROCESS_PHASE2:
        existing = {
            os.path.splitext(f)[0]
            for f in os.listdir(SYMBOL_OUT_DIR)
            if f.endswith(".parquet")
        }
        remaining = [s for s in all_symbols if safe_filename(s) not in existing]
        print(f"  이미 완료: {len(all_symbols) - len(remaining)}종목, "
              f"남은 종목: {len(remaining)}")
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
        print(f"\n  배치 {b_idx}/{len(batches)} ({len(batch_symbols)}종목) "
              f"로딩...", end=" ", flush=True)

        batch_df = (
            pl.scan_parquet(temp_files)
            .filter(pl.col("Symbol").is_in(batch_symbols))
            .sort(["Symbol", "Datetime"])
            .collect()
        )
        print(f"{batch_df.height:,}행")

        for symbol in batch_symbols:
            sym_df = batch_df.filter(pl.col("Symbol") == symbol)
            if sym_df.is_empty():
                continue
            out_path = os.path.join(
                SYMBOL_OUT_DIR, f"{safe_filename(symbol)}.parquet"
            )
            sym_df.write_parquet(out_path, compression="zstd")
            total_written += 1

        del batch_df
        gc.collect()

    print(f"\n  Phase 2 완료: {total_written}종목 저장됨 → {SYMBOL_OUT_DIR}")


# =============================================================================
# Step 3: VPIN 계산 핵심 함수
# =============================================================================

def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int,
    buckets_per_day: int,
) -> pl.DataFrame:
    """
    단일 종목의 1분봉 데이터를 받아 VPIN을 계산한다.

    BVC 모델: ProbBuy = CDF_t(ΔP/σ; df=0.25)
    버킷 크기 V = ADV(해당 연도) / buckets_per_day
    VPIN = rolling_sum(|V_buy - V_sell|) / (rolling_window × V)
    """
    schema = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64,    "VPIN":   pl.Float64,
    }
    empty = pl.DataFrame(schema=schema)

    if bar_df.height < 2:
        return empty

    times_dt   = bar_df["Datetime"].to_list()
    prices     = bar_df["Close"].to_numpy()       # Close 가격 사용
    volumes    = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    # ── 연도별 ADV 기반 버킷 크기 계산 ──
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

    # ── BVC: 가격 변동 → 매수 확률 ──
    delta_p  = np.diff(prices, prepend=prices[0])
    std_dp   = np.std(delta_p) + 1e-9
    prob_buy = t_dist.cdf(delta_p / std_dp, df=0.25)

    # ── 볼륨 버킷 채우기 ──
    records:        list[tuple] = []
    current_buy     = 0.0
    current_sell    = 0.0
    current_filled  = 0.0
    bucket_no       = 0
    last_year       = None
    bucket_size_v   = 0.0

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
            needed = bucket_size_v - current_filled
            alloc  = min(needed, rem_vol)
            current_buy    += alloc * p_b
            current_sell   += alloc * (1.0 - p_b)
            current_filled += alloc
            rem_vol        -= alloc

            if current_filled >= bucket_size_v - 1e-9:
                records.append((curr_time, bucket_no,
                                abs(current_buy - current_sell)))
                bucket_no       += 1
                current_buy      = 0.0
                current_sell     = 0.0
                current_filled   = 0.0

    if not records:
        return empty

    # ── 롤링 VPIN 계산 ──
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
    """종목별 parquet 읽기 → VPIN 계산 → 결과 저장"""
    symbol, input_path, output_path = args
    try:
        bar_df = pl.read_parquet(input_path)
        if bar_df.is_empty() or bar_df.height < MIN_ROWS:
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
# Step 3: VPIN 계산 메인 함수
# =============================================================================

def load_done_symbols(results_dir: str) -> set[str]:
    """results/ 폴더에서 완료된 종목 목록을 반환한다."""
    if not os.path.exists(results_dir):
        return set()
    done = {
        fname[len("sym_"):-len(".parquet")]
        for fname in os.listdir(results_dir)
        if fname.startswith("sym_") and fname.endswith(".parquet")
    }
    if done:
        print(f"  [재개 모드] 완료 종목 {len(done):,}개 발견 → 건너뜀")
    return done


def move_to_results(result_paths: list[str], results_dir: str) -> int:
    """워커 결과 파일을 results/ 로 rename한다."""
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


def run_vpin_calculation():
    """
    bars_by_symbol/ 의 종목별 파일을 읽어 VPIN을 계산하고
    results/sym_{Symbol}.parquet 로 저장한다.
    """
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    sym_result_dir = os.path.join(OUTPUT_DIR, "sessions", run_id, "sym_result")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(sym_result_dir, exist_ok=True)

    # ── 종목별 파일 목록 수집 ──
    sym_files = sorted(glob.glob(os.path.join(SYMBOL_OUT_DIR, "*.parquet")))
    if not sym_files:
        print(f"[Error] 종목별 파일 없음: {SYMBOL_OUT_DIR}")
        print("  Phase 1, Phase 2를 먼저 실행하세요.")
        return

    print(f"\n{'='*65}")
    print(f"[Step 3] VPIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 폴더   : {SYMBOL_OUT_DIR}")
    print(f"  파일 수     : {len(sym_files):,}개")
    print(f"  결과 폴더   : {RESULTS_DIR}")
    print(f"  RUN_ID      : {run_id}")

    num_cores = multiprocessing.cpu_count()
    print(f"\n  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  롤링 버킷   : {VPIN_ROLLING_WINDOW}개")
    print(f"  버킷/일     : {VPIN_BUCKETS_PER_DAY}개")

    # ── 완료된 종목 확인 (재개 지원) ──
    if FORCE_REPROCESS_VPIN:
        done_symbols = set()
    else:
        done_symbols = load_done_symbols(RESULTS_DIR)

    # ── 종목명 → 파일 경로 매핑 ──
    all_symbols_map: dict[str, str] = {}
    for f in sym_files:
        sym_name = os.path.splitext(os.path.basename(f))[0]
        all_symbols_map[sym_name] = f

    remaining_list = [s for s in sorted(all_symbols_map) if s not in done_symbols]

    print(f"\n  전체 종목   : {len(all_symbols_map):,}개")
    print(f"  완료 종목   : {len(done_symbols):,}개")
    print(f"  처리 대상   : {len(remaining_list):,}개\n")

    if not remaining_list:
        print("  모든 종목이 완료됐습니다.")
        return

    # ── 작업 목록 생성 ──
    tasks: list[tuple] = []
    for sym in remaining_list:
        input_path  = all_symbols_map[sym]
        output_path = os.path.join(sym_result_dir, f"sym_{sym}.parquet")
        tasks.append((sym, input_path, output_path))

    # ── 병렬 VPIN 계산 ──
    print(f"{'='*65}")
    print("VPIN 계산 시작...")
    print(f"{'='*65}\n")

    pending_paths:   list[str] = []
    processed_count = 0
    moved_total     = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(VPIN_ROLLING_WINDOW, VPIN_BUCKETS_PER_DAY),
    ) as pool:
        for path in tqdm(
            pool.imap_unordered(
                process_symbol_worker, tasks, chunksize=IMAP_CHUNKSIZE
            ),
            total=len(tasks),
            desc="  Estimating",
        ):
            if path is not None:
                pending_paths.append(path)
            processed_count += 1

            if processed_count % CHECKPOINT_N == 0 and pending_paths:
                moved = move_to_results(pending_paths, RESULTS_DIR)
                moved_total += moved
                print(f"\n  [체크포인트] {moved}개 → results/  "
                      f"(누적 {moved_total}개)")
                pending_paths = []

    if pending_paths:
        moved = move_to_results(pending_paths, RESULTS_DIR)
        moved_total += moved
        print(f"\n  [체크포인트] {moved}개 → results/  (누적 {moved_total}개)")

    n_files = len([
        f for f in os.listdir(RESULTS_DIR)
        if f.startswith("sym_") and f.endswith(".parquet")
    ])
    print(f"\n{'='*65}")
    print(f"[Step 3 완료] results/ 에 {n_files:,}개 종목 파일 저장")
    print(f"{'='*65}")


# =============================================================================
# 전처리 통계 출력
# =============================================================================

def print_phase_summary():
    """Phase 2 결과(bars_by_symbol) 통계"""
    sym_files = sorted(glob.glob(os.path.join(SYMBOL_OUT_DIR, "*.parquet")))
    if not sym_files:
        print("\n[경고] 종목별 파일이 없습니다.")
        return

    print(f"\n{'='*65}")
    print(f"[전처리 결과 요약]")
    print(f"{'='*65}")
    print(f"  종목별 파일 경로: {SYMBOL_OUT_DIR}")
    print(f"  종목 수: {len(sym_files):,}")

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
    total  = stats[0, "total_rows"]
    dt_min = stats[0, "dt_min"]
    dt_max = stats[0, "dt_max"]
    dp_nz  = stats[0, "dp_nonzero"]

    print(f"  총 봉 수 : {total:,}")
    print(f"  시간 범위: {dt_min} ~ {dt_max}")
    pct = dp_nz / total * 100 if total > 0 else 0
    print(f"  dp≠0 봉  : {dp_nz:,} / {total:,} ({pct:.1f}%)")
    print(f"  dp=0 봉  : {total - dp_nz:,} ({100 - pct:.1f}%)")


def print_vpin_summary():
    """VPIN 결과(results/) 통계"""
    result_files = sorted(glob.glob(os.path.join(RESULTS_DIR, "sym_*.parquet")))
    if not result_files:
        print("\n[경고] VPIN 결과 파일이 없습니다.")
        return

    print(f"\n{'='*65}")
    print(f"[VPIN 결과 요약]")
    print(f"{'='*65}")
    print(f"  결과 경로    : {RESULTS_DIR}")
    print(f"  종목 수      : {len(result_files):,}")

    stats = (
        pl.scan_parquet(result_files)
        .select([
            pl.len().alias("total_buckets"),
            pl.col("Symbol").n_unique().alias("n_symbols"),
            pl.col("VPIN").filter(pl.col("VPIN").is_not_nan()).min().alias("vpin_min"),
            pl.col("VPIN").filter(pl.col("VPIN").is_not_nan()).max().alias("vpin_max"),
            pl.col("VPIN").filter(pl.col("VPIN").is_not_nan()).mean().alias("vpin_mean"),
            pl.col("Datetime").min().alias("dt_min"),
            pl.col("Datetime").max().alias("dt_max"),
        ])
        .collect()
    )
    print(f"  총 버킷 수   : {stats[0, 'total_buckets']:,}")
    print(f"  유효 종목 수  : {stats[0, 'n_symbols']:,}")
    print(f"  시간 범위     : {stats[0, 'dt_min']} ~ {stats[0, 'dt_max']}")
    print(f"  VPIN 범위     : {stats[0, 'vpin_min']:.6f} ~ {stats[0, 'vpin_max']:.6f}")
    print(f"  VPIN 평균     : {stats[0, 'vpin_mean']:.6f}")
    print(f"{'='*65}")


# =============================================================================
# 메인 실행부
# =============================================================================

if __name__ == "__main__":
    start = datetime.now()

    print(f"\n{'='*65}")
    print(f"[VPIN 순수 Python] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"  나라코드       : {COUNTRY}")
    print(f"  입력 루트      : {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  출력 루트      : {OUTPUT_DIR}")
    print(f"  Phase 1 (임시) : {TEMP_DIR}")
    print(f"  Phase 2 (종목) : {SYMBOL_OUT_DIR}")
    print(f"  Step 3 (결과)  : {RESULTS_DIR}")
    print(f"  롤링 윈도우    : {VPIN_ROLLING_WINDOW}개 버킷")
    print(f"  버킷/일        : {VPIN_BUCKETS_PER_DAY}개")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        exit(1)

    # ── Phase 1: 틱 → 파일별 1분봉 ──
    parquet_files = get_parquet_files(BASE_DIR, COUNTRY)
    temp_files    = run_phase1(parquet_files)

    # ── Phase 2: 파일별 1분봉 → 종목별 parquet ──
    run_phase2(temp_files)

    # ── 전처리 통계 ──
    print_phase_summary()

    # ── Step 3: VPIN 계산 ──
    run_vpin_calculation()

    # ── VPIN 결과 통계 ──
    print_vpin_summary()

    elapsed = datetime.now() - start
    print(f"\n[완료] 총 소요 시간: {elapsed}")