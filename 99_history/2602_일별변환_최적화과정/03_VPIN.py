"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 VPIN (Volume-Synchronized PIN) 계산
=============================================================================

■ 전체 흐름

  [Step 1] 틱 데이터 → 연도별 1분봉 파일
    output/1m_bars/
      1m_bars_KOR_2010.parquet
      1m_bars_KOR_2011.parquet
      ...
      1m_bars_KOR_2021.parquet
    - 월별 parquet 를 연도 단위로 묶어 1분봉 집계 후 저장
    - RAM 피크 = 가장 큰 연도(2021) 1개 분량
    - 이미 존재하는 연도 파일은 스킵 (FORCE_REPROCESS_STEP1=False)

  [Step 2] 1분봉 → 종목별 VPIN 파일
    output/vpin_results/
      sym_005930.parquet   ← 005930 종목 2010~2021 전체 VPIN
      sym_000660.parquet
      ...
    ① 연도 파일을 1회씩만 읽어 종목별 sym_input 파일 생성
       (배치 반복 스캔 없음 → 디스크 I/O 최소)
    ② 워커: sym_input 읽기 → VPIN 계산 → sym_result 직접 저장
       (DataFrame 을 메인 프로세스로 반환 안 함 → RAM 누적 없음)
    ③ CHECKPOINT_N 마다 sym_result 파일을 vpin_results/ 로 rename
       (읽기·쓰기 없음 → 비용 ≈ 0)
    ④ 최종 병합 단계 없음 → OOM 위험 없음

■ RAM 안전성 요약

  단계                  RAM 피크
  Step 1 월별 집계      단일 월 (~1GB)
  Step 1 연도 concat    가장 큰 연도 (~5GB, 2021)
  Step 2 sym_input 생성 가장 큰 연도 1개 (연도별 1회 읽기)
  Step 2 VPIN 계산      워커당 단일 종목 12년치 (~수십 MB)
  Step 2 결과 이동      경로 문자열만 (rename 이므로 데이터 이동 없음)
  최종 통계             scan_parquet 집계 (전체 로드 없음)

■ 재개
  vpin_results/{sym}.parquet 존재 여부로 완료 판단
  sym_input/{sym}.parquet 존재 여부로 분할 완료 판단
  → RESUME_RUN_ID 를 이전 RUN_ID 로 설정하면 이어 쓰기 가능

■ VPIN 계산
  BVC: ProbBuy = CDF_t(ΔP/σ; df=0.25)
  버킷 크기 V = ADV(해당 연도) / BUCKETS_PER_DAY  ← 연도별 동적 산출
  VPIN = Σ|V_buy - V_sell| / (ROLLING_WINDOW × V)
  버킷은 연도 경계를 넘어 연속 → 초기 NaN 은 종목 역사 전체에서 딱 1회

■ 설정값
  BASE_DIR               : 원본 틱 parquet 루트 (KOR_YYYY 폴더 포함)
  YEAR_FOLDERS           : 처리 연도 (None → KOR_* 전체 자동 탐색)
  OUTPUT_DIR             : 결과 저장 루트
  FORCE_REPROCESS_STEP1  : True → 기존 1m_bars 파일 무시하고 재생성
  CHECKPOINT_N           : N개 종목마다 vpin_results/ 로 이동 (기본 100)
  IMAP_CHUNKSIZE         : 워커 IPC 청크 크기 (기본 10)
  ROLLING_WINDOW         : VPIN 롤링 버킷 수 (기본 50)
  BUCKETS_PER_DAY        : V = ADV / 이 값 (기본 50)
  RESUME_RUN_ID          : 재개 세션 RUN_ID (None → 새 세션)

=============================================================================
"""

import os
import glob
import tempfile
import shutil
import multiprocessing
import warnings
from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 전역 설정
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = None        # None → KOR_* 폴더 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

FORCE_REPROCESS_STEP1 = False       # True → 기존 1m_bars 파일 강제 재생성

CHECKPOINT_N          = 100         # N개 종목마다 vpin_results/ 로 이동
IMAP_CHUNKSIZE        = 10          # 워커 IPC 청크 크기

ROLLING_WINDOW        = 50          # VPIN 롤링 버킷 수
BUCKETS_PER_DAY       = 50          # V = ADV / 이 값

RESUME_RUN_ID         = None        # None → 새 세션 / "20260305_1018" → 재개


# =============================================================================
# 공통 유틸리티
# =============================================================================

def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다."""
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
        parquet_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.parquet"))))

    print(f"\n[파일 탐색]")
    print(f"  기준 경로   : {base_dir}")
    print(f"  대상 연도   : {year_folders}")
    print(f"  parquet 파일: {len(parquet_files)}개")
    if parquet_files:
        print(f"  첫 번째     : {os.path.basename(parquet_files[0])}")
        print(f"  마지막      : {os.path.basename(parquet_files[-1])}")
    return parquet_files


def load_done_symbols(results_dir: str) -> set[str]:
    """
    vpin_results/ 폴더에서 완료된 종목 목록을 반환한다.
    sym_005930.parquet 존재 → "005930" 완료로 간주.
    """
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
    """
    워커 결과 파일(sym_result/{sym}.parquet)을
    vpin_results/{sym}.parquet 로 rename 한다.

    os.replace 는 같은 드라이브 내에서 rename 수준의 비용(읽기·쓰기 없음).
    반환: 성공 이동 수
    """
    moved = 0
    for src in result_paths:
        fname = os.path.basename(src)          # "sym_005930.parquet"
        dst   = os.path.join(results_dir, fname)
        try:
            os.replace(src, dst)
            moved += 1
        except Exception as e:
            print(f"\n  [Warning] 이동 실패: {fname} ({e})")
    return moved


# =============================================================================
# [Step 1] 틱 데이터 → 연도별 1분봉 파일
# =============================================================================

def _process_one_month(parquet_path: str) -> pl.DataFrame:
    """
    단일 월 parquet 를 읽어 Symbol·분 단위 1분봉으로 집계한다.
    출력: Symbol, Datetime, Price(종가), Volume(합계)
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
        print("0 bars")
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Price": pl.Float64,  "Volume": pl.Float64,
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


def run_preprocessing(base_dir: str, year_folders: list | None, output_dir: str) -> str:
    """
    모든 월별 parquet 를 연도 단위로 묶어 1분봉 집계 후
    output/1m_bars/1m_bars_KOR_YYYY.parquet 로 영구 저장한다.

    RAM 전략:
      한 연도의 월별 DataFrame 을 concat + sort → 즉시 저장 → del
      피크 = 가장 큰 연도(2021) 1개 분량

    스킵:
      FORCE_REPROCESS_STEP1=False 이면 기존 연도 파일은 건너뜀.
      특정 연도만 재처리하려면 해당 파일만 삭제 후 실행.

    반환: 1m_bars 폴더 경로 (실패 시 빈 문자열)
    """
    bars_dir = os.path.join(output_dir, "1m_bars")
    os.makedirs(bars_dir, exist_ok=True)

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("[Error] parquet 파일 없음")
        return ""

    # 연도 태그 추출 함수
    def _year_tag(path: str) -> str:
        return next(
            (p for p in path.replace("\\", "/").split("/") if p.startswith("KOR_")),
            "KOR_UNKNOWN",
        )

    # 파일을 연도별로 그룹화
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

        # 기존 파일 스킵
        if not FORCE_REPROCESS_STEP1 and os.path.exists(out_path):
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

    # 전체 통계 출력
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
    rolling_window: int = 50,
    buckets_per_day: int = 50,
) -> pl.DataFrame:
    """
    단일 종목의 1분봉 데이터(2010~2021 연속)를 받아 VPIN 을 계산한다.

    연도별 동적 버킷 크기:
      ADV = 해당 연도 거래량 / 해당 연도 영업일 수
      V   = ADV / buckets_per_day
      → 매년 "하루 평균 buckets_per_day 개 버킷" 스케일 유지
      → 연도 경계에서 current_filled 초기화 없음 → 버킷 연속성 보장
      → 초기 NaN 은 종목 역사 전체에서 딱 1회
    """
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

    # 연도별 버킷 크기 매핑
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

    # BVC 매수 확률 추정
    delta_p  = np.diff(prices, prepend=prices[0])
    prob_buy = t_dist.cdf(delta_p / (np.std(delta_p) + 1e-9), df=0.25)

    # Volume Bucketing (연도별 버킷 크기 동적 적용)
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

    # VPIN 롤링 계산
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
# [Step 2] 멀티프로세싱 워커
# =============================================================================

_ROLLING_WINDOW:  int = 50
_BUCKETS_PER_DAY: int = 50


def init_worker(rolling_window: int, buckets_per_day: int) -> None:
    global _ROLLING_WINDOW, _BUCKETS_PER_DAY
    _ROLLING_WINDOW  = rolling_window
    _BUCKETS_PER_DAY = buckets_per_day


def process_symbol_worker(args: tuple) -> str | None:
    """
    sym_input/{sym}.parquet 읽기 → VPIN 계산 → sym_result/{sym}.parquet 저장.
    메인 프로세스로 DataFrame 반환 안 함 → IPC = 경로 문자열만.
    """
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
# [Step 2] VPIN 계산 통합 실행 함수
# =============================================================================

def run_vpin_calculation(
    bars_dir:     str,
    output_dir:   str,
    run_id:       str,
    year_filter:  list[int] | None,
    checkpoint_n: int,
) -> str:
    """
    1m_bars/ 연도별 파일을 읽어 종목별 VPIN 을 계산하고
    vpin_results/{sym}.parquet 로 저장한다.

    최종 병합 단계 없음:
      워커 완료 → sym_result/{sym}.parquet → os.replace → vpin_results/{sym}.parquet
      rename 이므로 읽기·쓰기 비용 ≈ 0, RAM 에 전체 결과 로드 없음

    반환: vpin_results 폴더 경로 (실패 시 빈 문자열)
    """
    results_dir    = os.path.join(output_dir, "vpin_results")
    sym_input_dir  = os.path.join(output_dir, "intermediate",
                                  f"session_{run_id}", "sym_input")
    sym_result_dir = os.path.join(output_dir, "intermediate",
                                  f"session_{run_id}", "sym_result")

    for d in [results_dir, sym_input_dir, sym_result_dir]:
        os.makedirs(d, exist_ok=True)

    # 연도 파일 수집
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

    # ── 1. 메타 확인 ─────────────────────────────────────────────────────────
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
    print(f"  IPC 청크    : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  롤링 버킷   : {ROLLING_WINDOW}개")
    print(f"  버킷/일     : {BUCKETS_PER_DAY}개")

    # ── 2. 완료 종목 확인 ────────────────────────────────────────────────────
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

    # ── 3. sym_input 파일 생성 (연도별 1회 읽기) ─────────────────────────────
    # 연도 파일 12개를 각 1회씩만 읽어 전체 종목에 배분
    # → 디스크 I/O = 전체 데이터 × 1회 (배치 반복 스캔 없음)
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

        # 버퍼: {sym: [연도별 DataFrame, ...]}
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

    # ── 4. 병렬 VPIN 계산 + vpin_results/ 로 rename ──────────────────────────
    # rename 이므로 읽기·쓰기 없음 → RAM 에 전체 결과 로드 안 됨
    print(f"{'='*65}")
    print("VPIN 계산 시작...")
    print(f"{'='*65}\n")

    pending_paths:  list[str] = []
    processed_count = 0
    moved_total     = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(ROLLING_WINDOW, BUCKETS_PER_DAY),
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


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    run_id = RESUME_RUN_ID if RESUME_RUN_ID else datetime.now().strftime("%Y%m%d_%H%M")
    start  = datetime.now()
    print(f"\n{'[재개 모드]' if RESUME_RUN_ID else '[새 실행]  '} RUN_ID = {run_id}")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    bars_dir = run_preprocessing(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR,
    )
    if not bars_dir or not os.path.exists(bars_dir):
        print("[Error] 전처리 결과 폴더가 없어 종료합니다.")
        exit(1)

    year_filter = (
        [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
        if YEAR_FOLDERS else None
    )

    # ── Step 2 ───────────────────────────────────────────────────────────────
    results_dir = run_vpin_calculation(
        bars_dir=bars_dir,
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    # ── 완료 통계 (scan 집계 — 전체 로드 없음) ────────────────────────────────
    if results_dir and os.path.exists(results_dir):
        result_files = sorted(
            f for f in os.listdir(results_dir)
            if f.startswith("sym_") and f.endswith(".parquet")
        )
        print(f"\n[완료 통계]")
        print(f"  결과 폴더   : {results_dir}")
        print(f"  종목 파일   : {len(result_files)}개")

        if result_files:
            all_paths = [os.path.join(results_dir, f) for f in result_files]
            stats = (
                pl.scan_parquet(all_paths)
                .select([
                    pl.len().alias("total_buckets"),
                    pl.col("VPIN").is_not_null().sum().alias("vpin_ok"),
                    pl.col("VPIN").min().alias("vpin_min"),
                    pl.col("VPIN").max().alias("vpin_max"),
                    pl.col("Datetime").min().alias("dt_min"),
                    pl.col("Datetime").max().alias("dt_max"),
                ])
                .collect()
            )
            print(f"  전체 버킷   : {stats['total_buckets'][0]:,}")
            print(f"  VPIN 유효   : {stats['vpin_ok'][0]:,}")
            print(f"  VPIN NaN    : "
                  f"{stats['total_buckets'][0] - stats['vpin_ok'][0]:,}"
                  f"  (초기 윈도우 미충족)")
            print(f"  VPIN 범위   : "
                  f"{stats['vpin_min'][0]:.6f} ~ {stats['vpin_max'][0]:.6f}")
            print(f"  시간 범위   : {stats['dt_min'][0]} ~ {stats['dt_max'][0]}")

            # 샘플 미리보기 (첫 번째 종목)
            sample_df = pl.read_parquet(all_paths[0])
            print(f"\n[샘플 미리보기: {result_files[0]}]")
            print(sample_df.head(10))
    else:
        print("\n[Warning] 결과 폴더가 없습니다.")

    print(f"\n총 소요 시간: {datetime.now() - start}")
