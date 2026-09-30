"""
=============================================================================
[VPIN 파이프라인] 틱 데이터 → 종목별 VPIN 계산 (대용량 데이터 처리용)
틱 데이터 ({나라코드}_{연도} 폴더들)
  → Step 1: 종목별 틱 데이터 통합 → symbol_ticks/{Symbol}.parquet  (캐시)
  → Step 2: 종목별 VPIN 병렬 계산 → checkpoints/sym_{Symbol}.parquet (체크포인트)
  → Step 3: 체크포인트 병합        → vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet
  → Step 4: 검증 리포트 생성       → vpin_{COUNTRY}_{YYYYMMDD_HHMM}_report.txt
=============================================================================

APIN 파이프라인과 동일한 구조로 대용량 데이터 처리에 최적화.
  · 체크포인트 기반 재개: 중단 후 재실행 시 남은 종목만 처리
  · Load-Balanced 병렬 (imap_unordered): 느린 종목이 다른 워커를 막지 않음
  · 나라코드 자동 스캔: {COUNTRY}_{YYYY} 폴더들을 정규식으로 탐색

입력 폴더 형식: {BASE_DIR}/{나라코드}_{연도}/
  예) E:\\vpin_project_parquet\\KOR_2019\\KOR_201901.parquet

COUNTRY 설정만 바꾸면 해당 나라의 모든 연도 폴더를 자동으로 스캔한다.

지원 나라코드: KOR  US  JP  CA  FR  GR  HK  IT  UK

출력 경로: {BASE_DIR}/R_output/{COUNTRY}/vpin/
  symbol_ticks/{Symbol}.parquet           ← Step 1 캐시 (종목별 통합 틱)
  checkpoints/sym_{Symbol}.parquet        ← Step 2 체크포인트
  vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet  ← 최종 결과
  vpin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv
  vpin_{COUNTRY}_{YYYYMMDD_HHMM}_report.txt  ← 검증 리포트

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
from typing import List

import numpy as np
import polars as pl
import pyarrow.parquet as pq
from scipy.stats import t as t_dist
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

# 틱 parquet 루트 폴더
BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"

# ssu  : "E:\vpin_project_parquet\processing_data"
# suji : "C:\vpin_project/vpin_project_parquet/processing_data"

# 처리할 나라코드 — 이 코드로 시작하는 모든 연도 폴더를 자동 스캔
# 예) "KOR" → KOR_2019, KOR_2020, KOR_2021 ... 자동 탐색
COUNTRY = "CN"

# True → Step 1 캐시(symbol_ticks/)와 Step 2 체크포인트 모두 강제 재생성
FORCE_REPROCESS = False

# True → Step 1 캐시는 유지하고 Step 2 체크포인트만 재계산
#        (VPIN 파라미터만 바꿔 다시 계산할 때 유용)
FORCE_RECOMPUTE_VPIN = False

# ── VPIN 파라미터 ────────────────────────────────────────────────────────────
ROLLING_WINDOW   = 50      # VPIN 롤링 윈도우 (버킷 수)
BUCKETS_PER_DAY  = 50      # 하루당 버킷 수 → V = ADV / BUCKETS_PER_DAY
T_DIST_DF        = 0.25    # BVC에 사용되는 Student's t-분포 자유도

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
# None이면 (CPU 코어 수 - 1)로 자동 설정
NUM_WORKERS = 8

# ── 검증 리포트 임계값 ───────────────────────────────────────────────────────
REPORT_NAN_RATIO_THRESHOLD     = 0.05   # 롤링 윈도우 완성 이후 NaN 비율 허용치
REPORT_EXTREME_VPIN_THRESHOLD  = 0.50   # 극단값 판정 기준 (VPIN > 0.5)
REPORT_EXTREME_RATIO_THRESHOLD = 0.20   # 극단값 비율 허용치


# =============================================================================
# (이하 수정 불필요)
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK", "CN"}
_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")

if COUNTRY not in VALID_COUNTRIES:
    raise ValueError(
        f"COUNTRY 설정 오류: '{COUNTRY}'\n"
        f"  지원 나라코드: {', '.join(sorted(VALID_COUNTRIES))}"
    )

OUTPUT_ROOT        = os.path.join(BASE_DIR, "0416_output", COUNTRY, "vpin")
SYMBOL_TICKS_DIR   = os.path.join(OUTPUT_ROOT, "symbol_ticks")
CHECKPOINT_DIR     = os.path.join(OUTPUT_ROOT, "checkpoints")


# =============================================================================
# 파일 탐색 유틸리티 (APIN과 동일)
# =============================================================================

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
# Step 1: 틱 데이터 → 종목별 통합 파일 (공유 캐시)
# -----------------------------------------------------------------------------
# 여러 해의 틱 데이터를 종목별로 누적 저장한다.
# 전략: 청크 방식으로 종목 그룹을 순회하며 메모리 관리
#   - 연도 parquet을 순차 로드
#   - 종목별 partition 후 symbol_ticks/{Symbol}.parquet에 append
#   - PyArrow ParquetWriter로 스트리밍 쓰기 → 메모리 피크 최소화
# =============================================================================

def run_step1(base_dir: str, country: str) -> List[str]:
    """
    {COUNTRY}_{YYYY} 폴더들의 틱 데이터를 종목별로 누적 저장.
    반환: symbol_ticks/ 디렉토리 내 종목별 parquet 파일 경로 리스트
    """
    os.makedirs(SYMBOL_TICKS_DIR, exist_ok=True)

    # FORCE_REPROCESS가 True면 기존 캐시 전부 삭제
    if FORCE_REPROCESS and os.path.isdir(SYMBOL_TICKS_DIR):
        print(f"\n[Step 1] FORCE_REPROCESS=True → 기존 캐시 삭제: {SYMBOL_TICKS_DIR}")
        shutil.rmtree(SYMBOL_TICKS_DIR)
        os.makedirs(SYMBOL_TICKS_DIR, exist_ok=True)

    # 이미 캐시가 존재하고 FORCE가 아니면 스킵
    existing = sorted(glob.glob(os.path.join(SYMBOL_TICKS_DIR, "*.parquet")))
    if not FORCE_REPROCESS and existing:
        print(f"\n[Step 1 스킵] 기존 캐시 재사용: {SYMBOL_TICKS_DIR}")
        print(f"  종목별 파일 {len(existing):,}개")
        return existing

    parquet_files = get_parquet_files(base_dir, country)

    print(f"\n{'='*65}")
    print(f"[Step 1] 틱 데이터 → 종목별 통합 파일  [{country}]")
    print(f"{'='*65}\n")

    # 종목별 ParquetWriter를 보관 (append 쓰기용)
    writers = {}   # Symbol(원본) -> ParquetWriter
    n_rows_per_sym = {}  # Symbol(원본) -> 누적 행수
    symbol_to_path = {}  # Symbol(원본) -> 실제 저장 파일 경로
    used_paths = {}  # 정규화된(lower) 파일경로 -> Symbol(원본)  (대소문자 충돌 감지용)

    # Windows 등 case-insensitive 파일시스템에서의 파일명 충돌을 방지하기 위한 헬퍼
    # - 파일명에 사용할 수 없는 특수문자를 '_'로 치환
    # - 대소문자만 다른 심볼이 충돌하면 suffix를 붙여 고유한 경로를 만든다
    _INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')

    def _safe_filename(sym: str) -> str:
        # 특수문자 치환 + 양끝 공백/점 제거 (Windows에서 금지)
        cleaned = _INVALID_FILENAME_CHARS.sub("_", str(sym)).strip(" .")
        if not cleaned:
            cleaned = "_EMPTY_"
        return cleaned

    def _resolve_out_path(sym: str) -> str:
        """
        심볼에 대한 출력 경로를 결정한다.
        같은 심볼이면 기존 경로를 재사용하고,
        대소문자만 다른 다른 심볼이 이미 그 경로를 쓰고 있으면 suffix를 붙여 새 경로를 만든다.
        """
        if sym in symbol_to_path:
            return symbol_to_path[sym]

        base = _safe_filename(sym)
        candidate = os.path.join(SYMBOL_TICKS_DIR, f"{base}.parquet")
        key = candidate.lower()  # case-insensitive FS 기준 충돌 키

        if key not in used_paths:
            used_paths[key] = sym
            symbol_to_path[sym] = candidate
            return candidate

        # 대소문자 충돌 발생: suffix로 유일화
        idx = 1
        while True:
            candidate = os.path.join(SYMBOL_TICKS_DIR, f"{base}__{idx}.parquet")
            key = candidate.lower()
            if key not in used_paths:
                used_paths[key] = sym
                symbol_to_path[sym] = candidate
                print(f"    [경고] 파일명 충돌 해결: Symbol={sym!r} → {os.path.basename(candidate)}")
                return candidate
            idx += 1

    try:
        for path in parquet_files:
            print(f"  Loading: {os.path.basename(path)} ...", end=" ", flush=True)
            df = pl.read_parquet(
                path,
                columns=["Date", "Time", "Symbol", "Price", "Volume"],
            )
            print(f"{df.height:,} rows", end=" ")

            # 종목별 partition
            partitions = df.partition_by("Symbol", as_dict=True)
            del df

            print(f"({len(partitions)} symbols)", flush=True)

            for key, sym_df in partitions.items():
                # partition_by 반환 key는 tuple 형태
                symbol = key[0] if isinstance(key, tuple) else key
                # None/NaN 심볼은 스킵
                if symbol is None:
                    del sym_df
                    continue

                out_path = _resolve_out_path(symbol)
                table = sym_df.to_arrow()

                if symbol not in writers:
                    # 새 종목이면 ParquetWriter가 기존 파일을 덮어쓴다
                    # (os.remove를 호출하지 않음 — 만약 동일 경로를 이미 현재
                    #  프로세스에서 열고 있다면 PermissionError가 발생하기 때문)
                    writers[symbol] = pq.ParquetWriter(
                        out_path, table.schema, compression="zstd"
                    )
                    n_rows_per_sym[symbol] = 0

                writers[symbol].write_table(table)
                n_rows_per_sym[symbol] += table.num_rows

                del sym_df, table

            del partitions

    finally:
        # 모든 Writer를 반드시 닫음
        for w in writers.values():
            try:
                w.close()
            except Exception:
                pass

    result_files = sorted(glob.glob(os.path.join(SYMBOL_TICKS_DIR, "*.parquet")))
    total_rows = sum(n_rows_per_sym.values())

    print(f"\n[Step 1 완료] {SYMBOL_TICKS_DIR}")
    print(f"  종목 수: {len(result_files):,}")
    print(f"  총 행 수: {total_rows:,}")
    return result_files


# =============================================================================
# VPIN 계산 함수 (원본 그대로 유지)
# =============================================================================

def convert_ticks_to_symbol_1m_bars(tick_df: pl.DataFrame) -> pl.DataFrame:
    """틱 데이터를 1분 단위 타임바로 변환."""
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
    """1분봉 → BVC 기반 VPIN 계산."""

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
# Step 2: 종목별 VPIN 병렬 계산 (체크포인트 기반)
# =============================================================================

def worker_process_symbol(args):
    """워커 프로세스: 종목 1개를 읽어 VPIN 계산 후 체크포인트에 저장."""
    symbol, input_path, checkpoint_path, rolling_window, buckets_per_day, t_df = args

    try:
        tick_df = pl.read_parquet(input_path)

        bars_df = convert_ticks_to_symbol_1m_bars(tick_df)
        if bars_df.is_empty():
            # 빈 결과여도 체크포인트는 남겨야 재실행 시 스킵됨
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


def run_step2(symbol_tick_files: List[str]) -> List[str]:
    """종목별 VPIN을 병렬로 계산하고 체크포인트에 저장."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if FORCE_REPROCESS or FORCE_RECOMPUTE_VPIN:
        if os.path.isdir(CHECKPOINT_DIR):
            flag = "FORCE_REPROCESS" if FORCE_REPROCESS else "FORCE_RECOMPUTE_VPIN"
            print(f"\n[Step 2] {flag}=True → 체크포인트 삭제: {CHECKPOINT_DIR}")
            shutil.rmtree(CHECKPOINT_DIR)
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 이미 완료된 종목 확인 (재개용)
    completed = set()
    for f in glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet")):
        name = os.path.splitext(os.path.basename(f))[0]
        # "sym_XXX" -> "XXX"
        completed.add(name[4:] if name.startswith("sym_") else name)

    # 처리 대상 태스크 구성
    tasks = []
    for tick_path in symbol_tick_files:
        symbol = os.path.splitext(os.path.basename(tick_path))[0]
        if symbol in completed:
            continue
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"sym_{symbol}.parquet")
        tasks.append((
            symbol, tick_path, checkpoint_path,
            ROLLING_WINDOW, BUCKETS_PER_DAY, T_DIST_DF,
        ))

    n_total = len(symbol_tick_files)
    n_remaining = len(tasks)
    n_completed = n_total - n_remaining

    print(f"\n{'='*65}")
    print(f"[Step 2] 종목별 VPIN 병렬 계산  [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  전체 종목      : {n_total:,}개")
    print(f"  기완료 (재개)  : {n_completed:,}개")
    print(f"  처리 예정      : {n_remaining:,}개")

    if n_remaining == 0:
        print(f"\n  모든 종목 처리 완료. Step 3으로 건너뜁니다.")
        return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet")))

    # 워커 수 결정
    if NUM_WORKERS is None:
        n_workers = max(1, multiprocessing.cpu_count() - 1)
    else:
        n_workers = max(1, NUM_WORKERS)
    n_workers = min(n_workers, n_remaining)

    print(f"  워커 수        : {n_workers}개")
    print(f"  VPIN 파라미터  : rolling_window={ROLLING_WINDOW}, "
          f"buckets_per_day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"  병렬 방식      : imap_unordered (동적 할당)")
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
        print(f"\n  [에러 발생 종목 목록 (최대 10개)]")
        for sym, msg in error_log[:10]:
            first_line = msg.split("\n")[0] if msg else ""
            print(f"    - {sym}: {first_line}")

    return sorted(glob.glob(os.path.join(CHECKPOINT_DIR, "sym_*.parquet")))


# =============================================================================
# Step 3: 체크포인트 병합 → 최종 결과
# =============================================================================

def run_step3(checkpoint_files: List[str]) -> tuple:
    """체크포인트를 병합하여 최종 parquet과 샘플 CSV 저장."""
    print(f"\n{'='*65}")
    print(f"[Step 3] 체크포인트 병합 → 최종 결과  [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  체크포인트 파일 수: {len(checkpoint_files):,}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    dfs = []
    for f in tqdm(checkpoint_files, desc="병합"):
        try:
            df = pl.read_parquet(f)
            if not df.is_empty():
                dfs.append(df)
        except Exception as e:
            print(f"  [Warning] 읽기 실패: {os.path.basename(f)}: {e}")

    if not dfs:
        print("\n[Warning] 병합할 결과가 없습니다.")
        return None, None

    final_df = pl.concat(dfs).sort(["Symbol", "Datetime"])

    run_id       = datetime.now().strftime("%Y%m%d_%H%M")
    stem         = f"vpin_{COUNTRY}_{run_id}"
    parquet_path = os.path.join(OUTPUT_ROOT, f"{stem}.parquet")
    csv_path     = os.path.join(OUTPUT_ROOT, f"{stem}_sample1000.csv")

    final_df.write_parquet(parquet_path, compression="zstd")
    final_df.head(1000).write_csv(csv_path)

    print(f"\n  [Step 3 완료]")
    print(f"    전체 parquet: {parquet_path}")
    print(f"    샘플 CSV    : {csv_path}")
    print(f"    전체 행 수  : {final_df.height:,}")
    print(f"    종목 수     : {final_df['Symbol'].n_unique():,}")

    return final_df, stem


# =============================================================================
# Step 4: 검증 리포트 생성
# -----------------------------------------------------------------------------
# 포함 내용:
#   A. 기본 통계 요약
#   B. 요일별 VPIN 개수
#   C. 정합성 검증 테스트 (7가지)
#   D. 상위/하위 10종목
# =============================================================================

def _fmt_num(x, digits=6):
    """숫자 포맷 (NaN 안전)."""
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
    final_df: pl.DataFrame,
    stem: str,
    n_symbol_ticks: int,
    checkpoint_files: List[str],
) -> str:
    """검증 리포트 생성."""
    print(f"\n{'='*65}")
    print(f"[Step 4] 검증 리포트 생성  [{COUNTRY}]")
    print(f"{'='*65}")

    report_path = os.path.join(OUTPUT_ROOT, f"{stem}_report.txt")

    lines = []
    add = lines.append

    add("=" * 75)
    add(f"[VPIN 검증 리포트]  {COUNTRY}")
    add(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"대상 파일: {stem}.parquet")
    add("=" * 75)
    add("")

    # ==========================================================================
    # A. 기본 통계 요약
    # ==========================================================================
    add("─" * 75)
    add("[A] 기본 통계 요약")
    add("─" * 75)

    n_total_buckets = final_df.height
    n_symbols_out   = final_df["Symbol"].n_unique()

    # VPIN 유효값: null도 아니고 NaN도 아닌 것
    # (원본 VPIN 알고리즘은 초기 49개 버킷을 np.nan으로 채우는데,
    #  Polars에서는 NaN과 null이 구분되므로 둘 다 제외해야 한다.)
    valid_mask = pl.col("VPIN").is_not_null() & pl.col("VPIN").is_not_nan()
    n_valid_vpin = final_df.filter(valid_mask).height
    n_nan_vpin   = n_total_buckets - n_valid_vpin

    # 처리 통계
    n_checkpoints = len(checkpoint_files)
    n_failed      = n_symbol_ticks - n_checkpoints  # 체크포인트조차 없는 종목
    n_empty_sym   = n_checkpoints - n_symbols_out   # 체크포인트는 있지만 VPIN 결과 0행

    add(f"  전체 종목 수 (Step 1 캐시)    : {n_symbol_ticks:,}")
    add(f"  처리 성공 종목 수            : {n_symbols_out:,}")
    add(f"  빈 결과 종목 수 (버킷 불가)  : {n_empty_sym:,}")
    add(f"  실패/미처리 종목 수          : {n_failed:,}")
    add("")
    add(f"  전체 버킷 수                 : {n_total_buckets:,}")
    add(f"  유효 VPIN 행 수 (non-NaN)    : {n_valid_vpin:,}")
    add(f"  NaN VPIN 행 수 (초기 49행 등): {n_nan_vpin:,}")
    add("")

    # 날짜 범위
    date_min = final_df["Datetime"].min()
    date_max = final_df["Datetime"].max()
    add(f"  Datetime 범위                : {date_min}  ~  {date_max}")

    # 종목당 평균 버킷 수
    avg_buckets = n_total_buckets / n_symbols_out if n_symbols_out > 0 else 0
    add(f"  종목당 평균 버킷 수          : {avg_buckets:,.1f}")
    add("")

    # VPIN 분포
    valid_df = final_df.filter(valid_mask)
    if valid_df.height > 0:
        vpin_stats = valid_df.select([
            pl.col("VPIN").min().alias("min"),
            pl.col("VPIN").quantile(0.01).alias("p01"),
            pl.col("VPIN").quantile(0.25).alias("p25"),
            pl.col("VPIN").median().alias("median"),
            pl.col("VPIN").mean().alias("mean"),
            pl.col("VPIN").quantile(0.75).alias("p75"),
            pl.col("VPIN").quantile(0.99).alias("p99"),
            pl.col("VPIN").max().alias("max"),
            pl.col("VPIN").std().alias("std"),
        ]).row(0)
        (v_min, v_p01, v_p25, v_med,
         v_mean, v_p75, v_p99, v_max, v_std) = vpin_stats

        add("  VPIN 분포 (유효 값 기준):")
        add(f"    min    = {_fmt_num(v_min)}")
        add(f"    p01    = {_fmt_num(v_p01)}")
        add(f"    p25    = {_fmt_num(v_p25)}")
        add(f"    median = {_fmt_num(v_med)}")
        add(f"    mean   = {_fmt_num(v_mean)}")
        add(f"    p75    = {_fmt_num(v_p75)}")
        add(f"    p99    = {_fmt_num(v_p99)}")
        add(f"    max    = {_fmt_num(v_max)}")
        add(f"    std    = {_fmt_num(v_std)}")
    else:
        v_min = v_max = None
        add("  [Warning] 유효 VPIN 값이 없습니다.")
    add("")

    # ==========================================================================
    # B. 요일별 VPIN 개수
    # ==========================================================================
    add("─" * 75)
    add("[B] 요일별 VPIN 분포")
    add("─" * 75)

    # Polars의 weekday(): 1=월, 7=일
    weekday_names = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}

    wd_df = (
        final_df
        .filter(valid_mask)
        .with_columns(pl.col("Datetime").dt.weekday().alias("wd"))
        .group_by("wd")
        .agg([
            pl.len().alias("count"),
            pl.col("VPIN").mean().alias("mean_vpin"),
            pl.col("VPIN").std().alias("std_vpin"),
        ])
        .sort("wd")
    )

    total_valid = wd_df["count"].sum() if wd_df.height > 0 else 0

    add(f"  {'요일':<6} {'개수':>12} {'비율':>8} {'평균 VPIN':>14} {'표준편차':>14}")
    add(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*14} {'-'*14}")

    for row in wd_df.iter_rows(named=True):
        wd_num  = row["wd"]
        wd_name = weekday_names.get(wd_num, f"?{wd_num}")
        cnt     = row["count"]
        ratio   = cnt / total_valid * 100 if total_valid > 0 else 0
        mean_v  = row["mean_vpin"]
        std_v   = row["std_vpin"]
        add(f"  {wd_name:<6} {cnt:>12,} {ratio:>7.2f}% "
            f"{_fmt_num(mean_v, 6):>14} {_fmt_num(std_v, 6):>14}")

    add(f"  {'-'*6} {'-'*12} {'-'*8} {'-'*14} {'-'*14}")
    add(f"  {'합계':<6} {total_valid:>12,} {100.00:>7.2f}%")
    add("")

    # 주말에 데이터가 많으면 의심스러움 (참고 정보)
    weekend_cnt = (
        wd_df.filter(pl.col("wd").is_in([6, 7]))["count"].sum()
        if wd_df.height > 0 else 0
    )
    if total_valid > 0:
        weekend_ratio = weekend_cnt / total_valid * 100
        if weekend_ratio > 1.0:
            add(f"  [Note] 주말 비율이 {weekend_ratio:.2f}% 로 높습니다. "
                f"해외 시장이거나 데이터 오류 가능성 점검.")
    add("")

    # ==========================================================================
    # C. 정합성 검증 테스트 (7가지)
    # ==========================================================================
    add("─" * 75)
    add("[C] 정합성 검증 테스트")
    add("─" * 75)

    test_results = []  # (name, status_str, detail)

    # ── Test 1: VPIN 범위 [0, 1] ─────────────────────────────────────────────
    if valid_df.height > 0:
        out_of_range = valid_df.filter(
            (pl.col("VPIN") < 0) | (pl.col("VPIN") > 1)
        ).height
        ok = (out_of_range == 0)
        status = _pass_fail(ok)
        detail = (f"범위 이탈 {out_of_range:,}건 / 유효 {valid_df.height:,}건  "
                  f"(min={_fmt_num(v_min,6)}, max={_fmt_num(v_max,6)})")
    else:
        status = "[WARN]"
        detail = "유효 VPIN 값 없음"
    test_results.append(("Test 1: VPIN 값이 [0, 1] 범위 내인가", status, detail))

    # ── Test 2: 롤링 윈도우 완성 이후 NaN 비율 ──────────────────────────────
    # 각 종목별로 BucketNo >= (ROLLING_WINDOW - 1)인 행 중 NaN/null 비율을 봄
    post_rolling = final_df.filter(pl.col("BucketNo") >= (ROLLING_WINDOW - 1))
    if post_rolling.height > 0:
        n_post      = post_rolling.height
        n_post_nan  = post_rolling.filter(
            pl.col("VPIN").is_null() | pl.col("VPIN").is_nan()
        ).height
        nan_ratio   = n_post_nan / n_post
        ok          = (nan_ratio <= REPORT_NAN_RATIO_THRESHOLD)
        status      = _pass_fail(ok, warn=False)
        detail      = (f"롤링 완성 이후 NaN {n_post_nan:,} / {n_post:,} "
                       f"= {nan_ratio*100:.3f}%  "
                       f"(임계 {REPORT_NAN_RATIO_THRESHOLD*100:.1f}%)")
    else:
        status = "[WARN]"
        detail = f"BucketNo >= {ROLLING_WINDOW - 1} 인 버킷이 없음"
    test_results.append((
        f"Test 2: 롤링 윈도우(>= 버킷 {ROLLING_WINDOW-1}) 이후 NaN 비율 정상",
        status, detail,
    ))

    # ── Test 3: BucketNo 연속성 (종목별 0부터 1씩 증가) ──────────────────────
    # 종목별 (max(BucketNo) + 1) == count(BucketNo) 인지 + min == 0
    bucket_integrity = (
        final_df.group_by("Symbol").agg([
            pl.col("BucketNo").min().alias("bmin"),
            pl.col("BucketNo").max().alias("bmax"),
            pl.col("BucketNo").n_unique().alias("bnunique"),
            pl.len().alias("n"),
        ])
    )
    bad_rows = bucket_integrity.filter(
        (pl.col("bmin") != 0) |
        (pl.col("bmax") + 1 != pl.col("n")) |
        (pl.col("bnunique") != pl.col("n"))
    )
    ok = (bad_rows.height == 0)
    status = _pass_fail(ok)
    detail = f"불량 종목 {bad_rows.height:,} / 전체 {bucket_integrity.height:,}"
    if bad_rows.height > 0:
        sample_syms = bad_rows["Symbol"].head(5).to_list()
        detail += f"  (예시: {sample_syms})"
    test_results.append(("Test 3: BucketNo 연속성 (종목별 0~N-1, 중복 없음)",
                         status, detail))

    # ── Test 4: Datetime 단조증가 (종목별) ───────────────────────────────────
    # shift 후 비교하여 역행 있는지
    dt_check = (
        final_df
        .sort(["Symbol", "BucketNo"])
        .with_columns([
            pl.col("Datetime").shift(1).over("Symbol").alias("prev_dt"),
        ])
        .filter(
            pl.col("prev_dt").is_not_null() &
            (pl.col("Datetime") < pl.col("prev_dt"))
        )
    )
    n_violations = dt_check.height
    ok = (n_violations == 0)
    status = _pass_fail(ok)
    detail = f"역행 발생 {n_violations:,}건 / 전체 {final_df.height:,}건"
    if n_violations > 0:
        bad_syms = dt_check["Symbol"].unique().head(5).to_list()
        detail += f"  (예시: {bad_syms})"
    test_results.append(("Test 4: Datetime 단조증가 (종목별)", status, detail))

    # ── Test 5: 롤링 윈도우 정합성 ───────────────────────────────────────────
    # 종목별로 BucketNo 0..ROLLING_WINDOW-2는 NaN, ROLLING_WINDOW-1부터 non-NaN (단 전체 버킷 >= ROLLING_WINDOW 인 종목만)
    eligible = bucket_integrity.filter(pl.col("n") >= ROLLING_WINDOW)
    n_eligible = eligible.height

    if n_eligible > 0:
        eligible_syms = eligible["Symbol"].to_list()
        sub = final_df.filter(pl.col("Symbol").is_in(eligible_syms))

        # "invalid"(= null 또는 NaN)을 기준으로 판정
        is_invalid = pl.col("VPIN").is_null() | pl.col("VPIN").is_nan()

        # 초기 구간(BucketNo < ROLLING_WINDOW-1)에서 유효값이 있으면 위반
        early_bad = sub.filter(
            (pl.col("BucketNo") < ROLLING_WINDOW - 1) &
            ~is_invalid
        ).height

        # 완성 구간(BucketNo >= ROLLING_WINDOW-1)에서 invalid가 있으면 위반
        late_bad = sub.filter(
            (pl.col("BucketNo") >= ROLLING_WINDOW - 1) &
            is_invalid
        ).height

        ok = (early_bad == 0 and late_bad == 0)
        status = _pass_fail(ok)
        detail = (f"초기 {ROLLING_WINDOW-1}개 중 유효값 {early_bad:,}건, "
                  f"완성 구간 무효값 {late_bad:,}건  "
                  f"(검사 대상 종목 {n_eligible:,})")
    else:
        status = "[WARN]"
        detail = f"버킷 >= {ROLLING_WINDOW} 인 종목이 없음"
    test_results.append((
        f"Test 5: 롤링 정합성 (초기 {ROLLING_WINDOW-1}개 NaN, 이후 값)",
        status, detail,
    ))

    # ── Test 6: VPIN 극단값 비율 ─────────────────────────────────────────────
    if valid_df.height > 0:
        n_extreme = valid_df.filter(
            pl.col("VPIN") > REPORT_EXTREME_VPIN_THRESHOLD
        ).height
        ext_ratio = n_extreme / valid_df.height
        ok = (ext_ratio <= REPORT_EXTREME_RATIO_THRESHOLD)
        status = _pass_fail(ok, warn=False)
        detail = (f"VPIN > {REPORT_EXTREME_VPIN_THRESHOLD:.2f} 비율: "
                  f"{n_extreme:,} / {valid_df.height:,} = {ext_ratio*100:.3f}%  "
                  f"(임계 {REPORT_EXTREME_RATIO_THRESHOLD*100:.1f}%)")
    else:
        status = "[WARN]"
        detail = "유효 VPIN 값 없음"
    test_results.append((
        f"Test 6: VPIN 극단값(>{REPORT_EXTREME_VPIN_THRESHOLD:.2f}) 비율 정상",
        status, detail,
    ))

    # ── Test 7: 종목별 최소 버킷 수 (롤링 완성 가능 여부) ────────────────────
    under_rolling = bucket_integrity.filter(pl.col("n") < ROLLING_WINDOW)
    n_under = under_rolling.height
    n_syms_total = bucket_integrity.height
    under_ratio = n_under / n_syms_total if n_syms_total > 0 else 0
    # 20% 초과면 WARN 성격이지만 리포트 항목이라 정보성 PASS로 표시
    ok = True
    status = "[INFO]"
    detail = (f"버킷 < {ROLLING_WINDOW} (롤링 불가) 종목: "
              f"{n_under:,} / {n_syms_total:,} = {under_ratio*100:.2f}%")
    test_results.append((
        f"Test 7: 종목별 최소 버킷 수 (>= {ROLLING_WINDOW}이어야 VPIN 생성)",
        status, detail,
    ))

    # 출력
    for name, status, detail in test_results:
        add(f"  {status}  {name}")
        add(f"         └ {detail}")
        add("")

    # 요약
    n_pass = sum(1 for _, s, _ in test_results if s == "[PASS]")
    n_fail = sum(1 for _, s, _ in test_results if s == "[FAIL]")
    n_warn = sum(1 for _, s, _ in test_results if s == "[WARN]")
    n_info = sum(1 for _, s, _ in test_results if s == "[INFO]")
    add(f"  [테스트 요약]  "
        f"PASS {n_pass}  |  FAIL {n_fail}  |  WARN {n_warn}  |  INFO {n_info}")
    add("")

    # ==========================================================================
    # D. 상위/하위 10종목
    # ==========================================================================
    add("─" * 75)
    add("[D] 상위/하위 10종목 (VPIN 평균 기준)")
    add("─" * 75)

    if valid_df.height > 0:
        sym_stats = (
            valid_df
            .group_by("Symbol")
            .agg([
                pl.len().alias("n_vpin"),
                pl.col("VPIN").mean().alias("mean_vpin"),
                pl.col("VPIN").std().alias("std_vpin"),
            ])
            .filter(pl.col("n_vpin") >= 10)  # 너무 적은 종목 배제
            .sort("mean_vpin", descending=True)
        )

        add(f"  (유효 VPIN 10개 이상 종목 대상, 총 {sym_stats.height:,}종목)")
        add("")
        add(f"  [Top 10]   {'Symbol':<16} {'n_vpin':>10} "
            f"{'mean VPIN':>14} {'std':>12}")
        for row in sym_stats.head(10).iter_rows(named=True):
            add(f"             {str(row['Symbol']):<16} "
                f"{row['n_vpin']:>10,} "
                f"{_fmt_num(row['mean_vpin'], 6):>14} "
                f"{_fmt_num(row['std_vpin'], 6):>12}")
        add("")
        add(f"  [Bottom 10] {'Symbol':<16} {'n_vpin':>10} "
            f"{'mean VPIN':>14} {'std':>12}")
        for row in sym_stats.tail(10).reverse().iter_rows(named=True):
            add(f"             {str(row['Symbol']):<16} "
                f"{row['n_vpin']:>10,} "
                f"{_fmt_num(row['mean_vpin'], 6):>14} "
                f"{_fmt_num(row['std_vpin'], 6):>12}")
    else:
        add("  [Warning] 유효 VPIN 값이 없어 상/하위 종목을 계산할 수 없습니다.")
    add("")

    add("=" * 75)
    add("[리포트 끝]")
    add("=" * 75)

    # 파일로 저장
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  [Step 4 완료] 리포트: {report_path}")
    print(f"    테스트 결과: PASS {n_pass} | FAIL {n_fail} | "
          f"WARN {n_warn} | INFO {n_info}")
    return report_path


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[VPIN 파이프라인] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드     : {COUNTRY}")
    print(f"  입력 루트    : {BASE_DIR}  ({COUNTRY}_YYYY 폴더 자동 스캔)")
    print(f"  출력 루트    : {OUTPUT_ROOT}")
    print(f"  Step 1 캐시  : {SYMBOL_TICKS_DIR}")
    print(f"  체크포인트   : {CHECKPOINT_DIR}")
    print(f"  VPIN 파라미터: window={ROLLING_WINDOW}, "
          f"buckets/day={BUCKETS_PER_DAY}, t_df={T_DIST_DF}")
    print(f"{'='*65}")

    if not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        sys.exit(1)

    # Step 1: 틱 → 종목별 통합 파일
    symbol_tick_files = run_step1(BASE_DIR, COUNTRY)

    if not symbol_tick_files:
        print("[Error] Step 1에서 종목 파일을 생성하지 못했습니다.")
        sys.exit(1)

    # Step 2: 종목별 VPIN 병렬 계산
    checkpoint_files = run_step2(symbol_tick_files)

    if not checkpoint_files:
        print("[Error] Step 2에서 체크포인트가 생성되지 않았습니다.")
        sys.exit(1)

    # Step 3: 체크포인트 병합
    final_df, stem = run_step3(checkpoint_files)

    if final_df is None or final_df.is_empty():
        print("[Error] Step 3에서 최종 결과가 비어있습니다.")
        sys.exit(1)

    # Step 4: 검증 리포트
    report_path = run_step4_report(
        final_df, stem,
        n_symbol_ticks=len(symbol_tick_files),
        checkpoint_files=checkpoint_files,
    )

    elapsed = datetime.now() - start
    print(f"\n{'='*65}")
    print(f"[파이프라인 완료]  소요 시간: {elapsed}")
    print(f"{'='*65}\n")