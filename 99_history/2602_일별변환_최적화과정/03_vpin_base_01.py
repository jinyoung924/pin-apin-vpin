"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 VPIN (Volume-Synchronized PIN) 계산
=============================================================================

■ 설계 원칙: 두 코드의 장점만 결합
  ┌─────────────────────────┬────────────────────────────────────────────┐
  │ VPIN Base 에서 유지      │ PIN 코드 에서 이식                         │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ calculate_vpin_for_      │ 다중 연도 폴더(KOR_YYYY) 탐색              │
  │   single_symbol 로직     │ Step 1 결과 영구 저장 + 스킵 지원          │
  │ 파일 기반 워커            │ RUN_ID 세션 격리 체크포인트                │
  │ (결과를 디스크에 직접 저장)│ 중단 재개(RESUME_RUN_ID)                 │
  │ → RAM 누적 없음           │ init_worker + imap_unordered + chunksize  │
  └─────────────────────────┴────────────────────────────────────────────┘

■ 전체 흐름 요약

  [Step 1] 원시 틱 데이터 → 전체 1분봉 집계 (all_1m_bars.parquet)
    - 월별 .parquet 파일을 순회하며 Symbol·분 단위로 Price(종가)·Volume(합계) 집계
    - 결과를 all_1m_bars.parquet 한 파일로 영구 저장
    - 이미 파일이 존재하면 자동으로 스킵 (FORCE_REPROCESS_STEP1 = False)

  [Step 2] 1분봉 → 종목별 VPIN 계산
    ① 1분봉 데이터를 종목별 파일로 분할 → 세션 폴더(sym_input/)에 저장
       - 파일이 이미 존재하면 재사용 (재개 시 재분할 불필요)
    ② 워커(process_symbol_worker)가 sym_input/{sym}.parquet 를 읽어 VPIN 계산
       - 결과를 sym_result/{sym}.parquet 에 직접 저장하고 경로(문자열)만 반환
       - 메인 프로세스에 DataFrame 이 전달되지 않으므로 RAM 누적 없음
    ③ CHECKPOINT_N 종목마다: sym_result 파일들을 vpin_checkpoint_NNNN.parquet 로
       통합 저장 후 개별 sym_result 파일 삭제 (디스크 절약)
    ④ 최종: 모든 체크포인트 병합 → 결과 반환

■ 세션 폴더 구조
  intermediate/
    session_{RUN_ID}/
      sym_input/                     ← 종목별 1분봉 파일 (재개 시 재사용)
        sym_005930.parquet
        sym_000660.parquet
        ...
      sym_result/                    ← 워커 결과 파일 (체크포인트 후 삭제)
        sym_005930.parquet           ← 체크포인트 전까지만 존재
        ...
      vpin_checkpoint_0000.parquet   ← 종목 1 ~ CHECKPOINT_N 통합
      vpin_checkpoint_0001.parquet   ← 종목 CHECKPOINT_N+1 ~ 2N 통합
      ...

■ RAM 안전성
  워커가 반환하는 값: 파일 경로 문자열(~30 bytes)
  메인 프로세스 누적:  문자열 리스트 → CHECKPOINT_N 개마다 초기화
  → 종목 수·데이터 크기와 무관하게 메인 프로세스 RAM 사용량이 일정하게 유지됨

■ VPIN 계산 방식
  BVC(Bulk Volume Classification):
    ProbBuy = CDF_t(ΔP / σ_ΔP;  df=0.25)   ← 논문 기준 t-분포 자유도 0.25
  Volume Bucketing:
    버킷 크기 V = ADV / buckets_per_day
    1분봉 거래량을 버킷에 채우며 Spillover 처리
  VPIN = Σ|V_buy - V_sell| / (rolling_window × V)   ← rolling_window 개 버킷 롤링

■ 체크포인트·재개
  - CHECKPOINT_N 종목마다 sym_result 파일들을 체크포인트로 통합
  - 중단 후 재개: RESUME_RUN_ID = "20240115_0930" 으로 이전 세션 이어 쓰기
    완료 후에는 None 으로 복원
  - 세션 폴더가 RUN_ID 별로 격리되므로 다른 실행의 데이터와 섞이지 않음

■ 설정값 안내
  BASE_DIR               : 월별 .parquet 파일이 들어있는 루트 경로
  YEAR_FOLDERS           : 처리할 연도 폴더 (None 이면 KOR_* 전체 자동 탐색)
  OUTPUT_DIR             : 출력 결과 저장 경로
  FORCE_REPROCESS_STEP1  : True 면 all_1m_bars.parquet 강제 재생성
  CHECKPOINT_N           : 종목 N개마다 체크포인트 저장 (기본 100)
  IMAP_CHUNKSIZE         : 워커에 한 번에 넘기는 종목 묶음 크기 (기본 10)
  ROLLING_WINDOW         : VPIN 롤링 버킷 수 (기본 50)
  BUCKETS_PER_DAY        : ADV 를 나눌 버킷 수 → 버킷 크기 V = ADV / 이 값 (기본 50)
  RESUME_RUN_ID          : 이전 실행 재개용 RUN_ID 문자열. None 이면 새 세션 시작.

=============================================================================
"""

import os
import glob
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
YEAR_FOLDERS          = None   # ['KOR_2017']  # None 이면 KOR_* 폴더 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

# Step 1 설정
FORCE_REPROCESS_STEP1 = False  # True: 기존 all_1m_bars.parquet 가 있어도 재생성

# Step 2 설정
CHECKPOINT_N          = 100    # 종목 N개 처리마다 체크포인트 저장
IMAP_CHUNKSIZE        = 10     # 워커에 한 번에 보내는 종목 묶음 크기
                               # 종목당 처리 시간이 짧으면 늘리고(20~50),
                               # 길면 줄여서(5~10) 워커 부하 균형 조정

# VPIN 파라미터
ROLLING_WINDOW        = 50     # VPIN 롤링 버킷 수 (논문 기본값: 50)
BUCKETS_PER_DAY       = 50     # ADV 를 나눌 버킷 수 (버킷 크기 V = ADV / 50)

# 중단된 세션 재개 설정
# - None    : 현재 시각으로 새 RUN_ID 를 만들어 새 세션으로 시작한다.
# - 문자열  : 지정한 RUN_ID 의 세션 폴더를 이어 쓴다.
#             예) RESUME_RUN_ID = "20240115_0930"
#             완료 후에는 다시 None 으로 돌려놓을 것.
RESUME_RUN_ID         = None


# =============================================================================
# 공통 유틸리티 함수
# =============================================================================

def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """
    base_dir 아래의 KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다.
    year_folders 가 None 이면 KOR_ 로 시작하는 모든 폴더를 자동으로 탐색한다.
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
            print(f"[Warning] 폴더 없음: {folder_path}")
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


def load_already_done_symbols(session_dir: str) -> set[str]:
    """
    중단 후 재실행 시 세션 폴더의 체크포인트에서 이미 완료된 종목 목록을 복원한다.

    세션 폴더(intermediate/session_<RUN_ID>/) 를 직접 받으므로
    다른 RUN_ID 의 체크포인트는 참조하지 않는다.
    폴더가 없거나 체크포인트가 없으면 빈 set 을 반환해 처음부터 시작한다.
    """
    if not os.path.exists(session_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(session_dir) if f.startswith("vpin_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols = set()
    for fname in checkpoint_files:
        path = os.path.join(session_dir, fname)
        try:
            df = pl.read_parquet(path, columns=["Symbol"])
            done_symbols.update(df["Symbol"].unique().to_list())
        except Exception as e:
            print(f"[Warning] 체크포인트 읽기 실패: {fname} ({e})")

    print(f"[재개 모드] 기존 체크포인트 {len(checkpoint_files)}개 발견 → "
          f"완료된 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


def flush_to_checkpoint(
    result_paths: list[str],
    checkpoint_idx: int,
    session_dir: str,
) -> None:
    """
    완료된 sym_result 파일 목록을 하나의 체크포인트 parquet 로 통합 저장하고,
    통합이 끝난 개별 sym_result 파일은 삭제한다.

    ─ 통합 저장:  vpin_checkpoint_NNNN.parquet (세션 폴더 루트에 저장)
    ─ 개별 삭제:  sym_result/{sym}.parquet 제거 → 디스크 절약

    개별 파일 읽기에 실패한 파일은 경고만 출력하고 건너뛴다.
    (해당 종목 결과는 누락되지만 전체 파이프라인이 중단되지는 않는다.)
    """
    if not result_paths:
        return

    dfs               = []
    successfully_read = []

    for path in result_paths:
        try:
            df = pl.read_parquet(path)
            if not df.is_empty():
                dfs.append(df)
                successfully_read.append(path)
        except Exception as e:
            print(f"\n[Warning] 결과 파일 읽기 실패: {os.path.basename(path)} ({e})")

    if not dfs:
        return

    # 통합 체크포인트 저장
    combined  = pl.concat(dfs, how="vertical")
    ckpt_path = os.path.join(session_dir, f"vpin_checkpoint_{checkpoint_idx:04d}.parquet")
    combined.write_parquet(ckpt_path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] "
          f"{len(successfully_read)}개 종목 / {combined.height:,} 버킷 "
          f"→ {os.path.basename(ckpt_path)}")

    # 통합된 개별 파일 삭제 (디스크 절약)
    for path in successfully_read:
        try:
            os.remove(path)
        except Exception:
            pass   # 삭제 실패는 무시 (중복 처리 방지에는 영향 없음)


# =============================================================================
# [Step 1] 전처리: 틱 데이터 → 1분봉 집계
# =============================================================================

def preprocess_tick_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일(월별 틱 데이터)을 읽어 Symbol·분 단위 1분봉으로 집계한다.

    입력 컬럼  : Symbol, Date (pl.Date), Time (pl.Time), Price, Volume
    출력 스키마: Symbol (Utf8), Datetime (pl.Datetime), Price (Float64), Volume (Float64)
      - Price  : 해당 1분의 마지막 체결가 (종가/Last) ← sort 후 last() 보장
      - Volume : 해당 1분의 누적 거래량 합계

    group_by_dynamic 의 group_by 파라미터를 사용해 Symbol 별로 독립적으로 집계하므로
    Symbol 경계를 넘는 잘못된 시간 그루핑이 발생하지 않는다.
    Volume = 0 인 행은 집계 전에 제거한다 (Volume Clock 특성상 거래 없는 구간은 불필요).
    """
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    # Lazy 로드 + 필터 + Datetime 생성 후 collect
    # sort 는 group_by_dynamic 의 정렬 요건을 충족하기 위해 필수
    df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "Time", "Price", "Volume"])
        .filter(pl.col("Volume") > 0)
        .with_columns(
            pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")
        )
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

    # group_by_dynamic(group_by="Symbol"):
    #   Symbol 별로 독립적인 1분 윈도우 적용
    #   closed="left": [t, t+1m) 구간 → 분봉 시작 시각이 인덱스
    #   last()  : sort 이후 호출이므로 해당 분의 마지막 체결가(종가) 보장
    bars = (
        df
        .group_by_dynamic(
            "Datetime",
            every="1m",
            group_by="Symbol",
            closed="left",
        )
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
    모든 월별 parquet 를 순회해 1분봉으로 집계하고,
    결과를 all_1m_bars.parquet 한 파일로 영구 저장한다.

    이미 파일이 존재하고 FORCE_REPROCESS_STEP1=False 이면 집계를 건너뛰고
    기존 파일 경로를 그대로 반환한다.

    동일한 Symbol+Datetime 이 여러 파일에 걸쳐 있으면
    Volume 은 합산하고 Price 는 마지막 값으로 처리해 중복을 제거한다.

    반환값: 저장된 all_1m_bars.parquet 의 전체 경로 (실패 시 빈 문자열)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_1m_bars.parquet")

    if not FORCE_REPROCESS_STEP1 and os.path.exists(output_path):
        print(f"\n{'='*65}")
        print(f"[Step 1 스킵] 기존 전처리 파일이 존재합니다: {output_path}")
        print(f"  (다시 생성하려면 FORCE_REPROCESS_STEP1 = True 로 설정하세요)")
        print(f"{'='*65}\n")
        return output_path

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("\n[Error] parquet 파일을 찾을 수 없습니다.")
        return ""

    print(f"\n{'='*65}")
    print("[Step 1] 전처리 시작: 틱 데이터 → 1분봉 집계")
    print(f"{'='*65}\n")

    all_dfs = []
    skipped = 0

    for i, path in enumerate(parquet_files, 1):
        # 연도 폴더가 바뀔 때마다 구분 출력
        year_tag = next(
            (part for part in path.replace("\\", "/").split("/") if part.startswith("KOR_")),
            ""
        )
        prev_tag = next(
            (p for p in parquet_files[i - 2].replace("\\", "/").split("/") if p.startswith("KOR_")),
            ""
        ) if i > 1 else ""

        if i == 1 or year_tag != prev_tag:
            print(f"\n[{year_tag}]")

        df = preprocess_tick_to_1m_bars(path)

        if df.is_empty():
            print(f"    → [Warning] 데이터 없음, 건너뜀")
            skipped += 1
            continue

        all_dfs.append(df)

    if not all_dfs:
        print("\n[Error] 유효한 데이터가 없습니다.")
        return ""

    # 모든 월별 데이터를 수직으로 합친 뒤, Symbol+Datetime 이 겹치는 경우 재집계
    # (동일 종목·분이 두 파일에 걸쳐 있을 때 Volume 합산, Price 는 마지막 값 유지)
    print(f"\n{'='*65}")
    print("전체 데이터 합산 및 정렬 중...")
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .sort(["Symbol", "Datetime"])
        .group_by(["Symbol", "Datetime"])
        .agg([
            pl.col("Price").last(),
            pl.col("Volume").sum(),
        ])
        .sort(["Symbol", "Datetime"])
    )

    full_df.write_parquet(output_path, compression="zstd")

    print(f"\n{'='*65}")
    print("[Step 1 완료]")
    print(f"{'='*65}")
    print(f"  저장 경로  : {output_path}")
    print(f"  전체 행 수 : {full_df.height:,}")
    print(f"  종목 수    : {full_df['Symbol'].n_unique():,}")
    print(f"  시간 범위  : {full_df['Datetime'].min()} ~ {full_df['Datetime'].max()}")
    print(f"  처리 파일  : {len(parquet_files) - skipped}개  (건너뜀: {skipped}개)")
    print(f"{'='*65}\n")

    return output_path


# =============================================================================
# [Step 2] VPIN 계산 핵심 함수 (VPIN Base 원본 그대로 유지)
# =============================================================================

def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int = 50,
    buckets_per_day: int = 50,
) -> pl.DataFrame:
    """
    단일 종목의 1분봉 데이터를 받아 BVC 기반 VPIN 을 계산한다.
    VPIN Base 원본 로직을 그대로 유지한다.

    계산 절차:
      1. 가격 변화량(ΔP) 과 표준편차(σ) 계산
      2. t-분포(df=0.25) CDF 로 매수 확률(ProbBuy) 추정 (논문 기준 자유도)
      3. ADV / buckets_per_day 로 버킷 크기(V) 산출
      4. Volume Bucketing: 1분봉을 쪼개거나 합쳐서 정확히 V 만큼 채움 (Spillover 처리)
      5. rolling_window 개 버킷 슬라이딩 VPIN 계산

    Returns:
      성공: (Datetime, Symbol, BucketNo, VPIN) DataFrame
      실패: 동일 스키마의 빈 DataFrame
    """
    schema_def = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64,    "VPIN":   pl.Float64,
    }
    empty_res = pl.DataFrame(schema=schema_def)

    if bar_df.height < 2:
        return empty_res

    # ── 1. 기본 변수 추출 ────────────────────────────────────────────────────
    times_dt   = bar_df["Datetime"].to_list()
    prices     = bar_df["Price"].to_numpy()
    volumes    = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    # ── 2. BVC 매수/매도 거래량 추정 ─────────────────────────────────────────
    # [2-1] 가격 변화량 (Delta P): P_t - P_{t-1}
    delta_p = np.diff(prices, prepend=prices[0])

    # [2-2] 표준편차: 전체 기간의 ΔP 표준편차 (0 나누기 방지용 eps 추가)
    sigma_p = np.std(delta_p) + 1e-9

    # [2-3] Z-score 표준화 후 t-분포(df=0.25) CDF 로 매수 확률 추정
    z_scores = delta_p / sigma_p
    prob_buy  = t_dist.cdf(z_scores, df=0.25)

    # ── 3. 버킷 크기(V) 계산 ─────────────────────────────────────────────────
    # 실제 거래가 발생한 영업일 수로 ADV 를 계산 (거래 없는 날 제외)
    n_trading_days = (
        bar_df
        .filter(pl.col("Volume") > 0)
        .select(pl.col("Datetime").dt.date())
        .unique()
        .height
    )
    total_volume = np.sum(volumes)

    if n_trading_days == 0:
        return empty_res

    adv           = total_volume / n_trading_days
    bucket_size_v = adv / buckets_per_day

    if bucket_size_v <= 1e-5:
        return empty_res

    # ── 4. Volume Bucketing ───────────────────────────────────────────────────
    # 1분봉을 순서대로 버킷에 채운다.
    # [Case A] 남은 거래량 ≤ 버킷 잔여 공간 → 전량 투입, 다음 분봉으로 이동
    # [Case B] 남은 거래량 > 버킷 잔여 공간 → 버킷 채운 뒤 남은 양으로 다음 버킷 시작
    bucket_records = []
    current_buy    = 0.0
    current_sell   = 0.0
    current_filled = 0.0
    bucket_no      = 0

    for i in range(len(volumes)):
        rem_vol   = float(volumes[i])
        p_b       = float(prob_buy[i])
        curr_time = times_dt[i]

        if rem_vol <= 0:
            continue

        while rem_vol > 0:
            needed    = bucket_size_v - current_filled
            alloc_vol = min(needed, rem_vol)

            current_buy    += alloc_vol * p_b
            current_sell   += alloc_vol * (1.0 - p_b)
            current_filled += alloc_vol
            rem_vol        -= alloc_vol

            if current_filled >= bucket_size_v - 1e-9:
                oi = abs(current_buy - current_sell)
                bucket_records.append((curr_time, bucket_no, oi))
                bucket_no      += 1
                current_buy     = 0.0
                current_sell    = 0.0
                current_filled  = 0.0

    if not bucket_records:
        return empty_res

    # ── 5. VPIN 롤링 계산 ────────────────────────────────────────────────────
    res_times      = [r[0] for r in bucket_records]
    res_bucket_nos = [r[1] for r in bucket_records]
    res_ois        = np.array([r[2] for r in bucket_records])

    final_vpin = np.full(len(res_ois), np.nan)

    if len(res_ois) >= rolling_window:
        kernel    = np.ones(rolling_window)
        oi_sums   = np.convolve(res_ois, kernel, mode="valid")
        vpin_vals = oi_sums / (rolling_window * bucket_size_v)
        # 초기 rolling_window-1 개 버킷은 NaN 유지 (윈도우 미충족)
        final_vpin[rolling_window - 1:] = vpin_vals

    # ── 6. DataFrame 반환 ────────────────────────────────────────────────────
    return pl.DataFrame({
        "Datetime": res_times,
        "Symbol":   [symbol_str] * len(res_times),
        "BucketNo": res_bucket_nos,
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

# 워커 프로세스 간 공유할 VPIN 파라미터 (전역 변수)
# init_worker 가 각 프로세스 시작 시 한 번만 설정한다.
_ROLLING_WINDOW:  int = 50
_BUCKETS_PER_DAY: int = 50


def init_worker(rolling_window: int, buckets_per_day: int) -> None:
    """
    Pool 생성 시 각 워커 프로세스에서 한 번만 실행되는 초기화 함수.

    VPIN 파라미터를 전역 변수에 저장해 두면
    이후 process_symbol_worker 가 호출될 때마다 인수로 넘기지 않아도 된다.
    (매 호출마다 직렬화·역직렬화하는 IPC 비용 제거)
    """
    global _ROLLING_WINDOW, _BUCKETS_PER_DAY
    _ROLLING_WINDOW  = rolling_window
    _BUCKETS_PER_DAY = buckets_per_day


def process_symbol_worker(args: tuple) -> str | None:
    """
    단일 종목의 1분봉 파일을 읽어 VPIN 을 계산하고 결과를 디스크에 직접 저장한다.

    ─ 입력: (symbol, input_path, output_path) 튜플
    ─ 처리: sym_input/{sym}.parquet 읽기 → VPIN 계산 → sym_result/{sym}.parquet 저장
    ─ 반환: 저장된 output_path (문자열) 또는 None (빈 결과·오류)

    [핵심 설계 의도]
    메인 프로세스로 DataFrame 을 반환하지 않고 디스크에 직접 저장한다.
    IPC(프로세스 간 통신)로 이동하는 데이터가 경로 문자열(~30 bytes)뿐이므로
    종목 수·데이터 크기와 무관하게 메인 프로세스 RAM 사용량이 일정하게 유지된다.
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

        # 결과를 메인 프로세스로 보내지 않고 디스크에 직접 저장
        vpin_df.write_parquet(output_path, compression="zstd")
        return output_path

    except Exception as e:
        import traceback
        print(f"\n[Error] {symbol}: {e}\n{traceback.format_exc()}")
        return None


# =============================================================================
# [Step 2] VPIN 계산 통합 실행 함수
# =============================================================================

def run_vpin_calculation(
    bars_path:    str,
    output_dir:   str,
    run_id:       str,
    year_filter:  list[int] | None,
    checkpoint_n: int,
) -> pl.DataFrame:
    """
    all_1m_bars.parquet 를 읽어 종목별 VPIN 을 병렬로 계산하고 결과를 반환한다.

    처리 흐름:
      1. parquet 로드 → 연도 필터 적용
      2. 세션 폴더 구성 + 체크포인트에서 완료 종목 로드 (재개 지원)
      3. 미완료 종목만 sym_input/{sym}.parquet 로 분할 저장
         (이미 파일이 존재하면 재사용 → 재개 시 재분할 불필요)
      4. bars_df 메모리 해제 (이후 단계는 sym_input 파일만 사용)
      5. multiprocessing.Pool 로 종목 단위 병렬 처리
         - init_worker 에 VPIN 파라미터 전달 (워커당 1회)
         - 워커는 sym_result/{sym}.parquet 에 직접 저장하고 경로만 반환
      6. checkpoint_n 마다: sym_result 파일들 → 체크포인트 통합 저장 + 개별 삭제
      7. 모든 체크포인트 병합 → 최종 DataFrame 반환

    세션 폴더 격리:
      체크포인트는 intermediate/session_<run_id>/ 에만 저장된다.
      run_id 가 다르면 폴더 자체가 달라지므로 인풋이 바뀐 새 실행과
      이전 체크포인트가 절대 섞이지 않는다.

    Returns:
        Schema: Datetime, Symbol, BucketNo, VPIN  (실패 시 빈 DataFrame)
    """
    # ── 세션 폴더 구성 ────────────────────────────────────────────────────────
    intermediate_dir = os.path.join(output_dir, "intermediate")
    session_dir      = os.path.join(intermediate_dir, f"session_{run_id}")
    sym_input_dir    = os.path.join(session_dir, "sym_input")   # 종목별 1분봉 입력
    sym_result_dir   = os.path.join(session_dir, "sym_result")  # 워커 결과 출력

    for d in [output_dir, session_dir, sym_input_dir, sym_result_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*65}")
    print("[Step 2] VPIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일   : {bars_path}")
    print(f"  RUN_ID      : {run_id}")
    print(f"  세션 폴더   : {session_dir}")

    # ── 1. 1분봉 데이터 로드 ─────────────────────────────────────────────────
    print(f"\n[1분봉 데이터 로드]")
    bars_df = pl.read_parquet(bars_path)

    if year_filter:
        bars_df = bars_df.filter(pl.col("Datetime").dt.year().is_in(year_filter))
        print(f"  연도 필터   : {year_filter}")

    print(f"  전체 행 수  : {bars_df.height:,}")
    print(f"  종목 수     : {bars_df['Symbol'].n_unique():,}")
    print(f"  시간 범위   : {bars_df['Datetime'].min()} ~ {bars_df['Datetime'].max()}")

    num_cores = multiprocessing.cpu_count()
    print(f"\n  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  롤링 버킷   : {ROLLING_WINDOW}개")
    print(f"  버킷/일     : {BUCKETS_PER_DAY}개")

    # ── 2. 완료 종목 확인 (재개 지원) ────────────────────────────────────────
    done_symbols      = load_already_done_symbols(session_dir)
    all_symbols       = bars_df["Symbol"].unique().sort().to_list()
    remaining_symbols = set(s for s in all_symbols if s not in done_symbols)
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    if not remaining_symbols:
        print("  모든 종목이 이미 완료되었습니다. 체크포인트 병합으로 진행합니다.")

    # ── 3. 미완료 종목 분할 → sym_input 파일 저장 ────────────────────────────
    # partition_by 로 한 번에 분할한 뒤 파일이 없는 종목만 저장
    # 이미 존재하는 파일은 재사용 → 재개 시 재분할 비용 없음
    print("[종목 분할 중...]")
    symbol_partitions = bars_df.partition_by("Symbol", maintain_order=False)

    tasks       = []
    write_count = 0
    reuse_count = 0

    for part in symbol_partitions:
        sym = part["Symbol"][0]
        if sym not in remaining_symbols:
            continue

        input_path  = os.path.join(sym_input_dir,  f"sym_{sym}.parquet")
        output_path = os.path.join(sym_result_dir, f"sym_{sym}.parquet")

        if not os.path.exists(input_path):
            part.sort("Datetime").write_parquet(input_path, compression="zstd")
            write_count += 1
        else:
            reuse_count += 1

        tasks.append((sym, input_path, output_path))

    print(f"  신규 저장: {write_count}개  /  재사용: {reuse_count}개\n")

    # ── 4. bars_df 메모리 해제 ───────────────────────────────────────────────
    # 이후 단계는 sym_input 파일을 직접 읽으므로 전체 DataFrame 불필요
    del bars_df, symbol_partitions

    print(f"{'='*65}")
    print("VPIN 계산 시작...")
    print(f"{'='*65}\n")

    # ── 5. 체크포인트 카운터: 기존 파일 수에서 이어받음 ─────────────────────
    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("vpin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)

    result_paths    = []   # 완료된 sym_result 파일 경로 누적 (문자열만)
    processed_count = 0

    # ── 6. 병렬 처리 ─────────────────────────────────────────────────────────
    # init_worker : VPIN 파라미터를 각 워커에 1회만 직렬화해 전달 (IPC 비용 절감)
    # imap_unordered: 완료 순서 상관없이 결과를 받아 진행바 업데이트
    # chunksize   : 워커에 IMAP_CHUNKSIZE 개 종목을 묶어 전달 (IPC 횟수 감소)
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
                result_paths.append(path)  # 경로 문자열만 누적 (DataFrame 없음)
            processed_count += 1

            # ── 7. 체크포인트: sym_result 파일 통합 → 개별 파일 삭제 ─────────
            if processed_count % checkpoint_n == 0:
                flush_to_checkpoint(result_paths, next_checkpoint_idx, session_dir)
                next_checkpoint_idx += 1
                result_paths = []   # 경로 목록 초기화 (데이터는 이미 디스크에)

    # 마지막 배치 저장 (checkpoint_n 으로 딱 떨어지지 않는 나머지)
    if result_paths:
        flush_to_checkpoint(result_paths, next_checkpoint_idx, session_dir)

    # ── 8. 이 세션의 체크포인트만 병합 ──────────────────────────────────────
    print(f"\n{'='*65}")
    all_session_checkpoints = sorted(
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("vpin_checkpoint_")
    )
    print(f"체크포인트 {len(all_session_checkpoints)}개 병합 중... (세션: {run_id})")

    if not all_session_checkpoints:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    final_df = (
        pl.concat(
            [pl.read_parquet(p) for p in all_session_checkpoints],
            how="vertical",
        )
        .sort(["Symbol", "Datetime"])
    )

    return final_df


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()   # Windows 멀티프로세싱 안전장치

    # ── RUN_ID 결정 ────────────────────────────────────────────────────────────
    # 중단 재개: RESUME_RUN_ID 에 이전 RUN_ID 문자열을 지정한다.
    # 새 실행  : RESUME_RUN_ID = None 이면 현재 시각을 RUN_ID 로 사용한다.
    # 최종 결과 파일명과 세션 폴더명에 동일한 RUN_ID 가 들어가므로
    # "어떤 결과 파일이 어느 체크포인트에서 만들어졌는가" 를 파일명만으로 추적 가능.
    run_id = RESUME_RUN_ID if RESUME_RUN_ID else datetime.now().strftime("%Y%m%d_%H%M")
    start  = datetime.now()

    if RESUME_RUN_ID:
        print(f"\n[재개 모드] RUN_ID = {run_id}")
    else:
        print(f"\n[새 실행]   RUN_ID = {run_id}")

    # Step 1: 틱 데이터 전처리
    bars_path = run_preprocessing(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR,
    )

    if not bars_path or not os.path.exists(bars_path):
        print("\n[Error] 전처리 결과 파일이 없어 VPIN 계산을 종료합니다.")
        exit(1)

    # 폴더명 'KOR_2017' → 연도 정수 2017 로 변환
    year_filter = (
        [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
        if YEAR_FOLDERS else None
    )

    # Step 2: VPIN 계산
    result = run_vpin_calculation(
        bars_path=bars_path,
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    # ── 결과 저장 및 요약 출력 ────────────────────────────────────────────────
    if not result.is_empty():
        # 최종 파일명에 RUN_ID (시작 타임스탬프) 를 사용
        # → 체크포인트 세션 폴더명(session_<RUN_ID>) 과 1:1 대응
        year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename = f"vpin_rolling_{year_tag}_{run_id}"

        # 전체 결과: parquet (ZSTD 압축)
        parquet_path = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result.write_parquet(parquet_path, compression="zstd")
        print(f"\n[저장 완료] {parquet_path}")

        # 확인용 샘플: 상위 1000행 CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result.head(1000).write_csv(csv_path)
        print(f"[샘플 저장] {csv_path}")

        print("\n[미리보기]")
        print(result.head(20))

        print("\n[통계]")
        print(f"  전체 버킷 수      : {result.height:,}")
        print(f"  처리 종목 수      : {result['Symbol'].n_unique():,}")
        vpin_ok = result.filter(pl.col("VPIN").is_not_null()).height
        print(f"  VPIN 유효 버킷    : {vpin_ok:,}")
        print(f"  VPIN null 버킷    : {result.height - vpin_ok:,}"
              f"  (롤링 윈도우 미충족 초기 버킷)")
        print(f"  VPIN 범위         : "
              f"{result['VPIN'].min():.6f} ~ {result['VPIN'].max():.6f}")

        # 종목별 버킷 수 및 평균 VPIN 요약
        sym_stats = (
            result
            .group_by("Symbol")
            .agg([
                pl.len().alias("total_buckets"),
                pl.col("VPIN").is_not_null().sum().alias("vpin_buckets"),
                pl.col("VPIN").mean().alias("mean_vpin"),
            ])
            .with_columns(
                (pl.col("vpin_buckets") / pl.col("total_buckets") * 100)
                .alias("coverage_pct")
            )
            .sort("total_buckets", descending=True)
        )
        print("\n[종목별 통계 샘플]")
        print(sym_stats.head(10))

    else:
        print("\n[Warning] 결과 DataFrame 이 비어 있습니다.")

    elapsed = datetime.now() - start
    print(f"\n총 소요 시간: {elapsed}")
