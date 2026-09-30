"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 일별 롤링 PIN 계산 (2-Stage Pipeline)
=============================================================================

[파이프라인 구조]
  ▶ [Step 1] 전처리 (틱 데이터 → 일별 B/S 요약)
    - 모든 월별 .parquet 틱 데이터를 순회하며 일별 매수(B)/매도(S) 건수 집계
    - 결과를 단 하나의 파일(all_daily_bs.parquet)로 병합 및 저장
    - 매우 빠르고 가벼움. 복잡한 버퍼 관리 로직 제거

  ▶ [Step 2] PIN 추정 (일별 B/S 요약 → PIN 계산)
    - [Step 1]이 생성한 all_daily_bs.parquet 하나만 읽어 계산
    - 종목별 20일 롤링 윈도우로 PIN 계산 (MLE)
    - 멀티프로세싱 및 종목 N개 단위 체크포인트(중간 저장) 기능 지원

[최적화 변경사항 - 원본 대비]
  ① [방안 1,2] scipy.stats.poisson.pmf → log-space 직접 계산
     - 변경 전: poisson.pmf(k, λ)  →  scipy 객체 생성 + Python dispatch 오버헤드
     - 변경 후: log P(k|λ) = k·ln(λ) - λ - lgamma(k+1)  →  순수 NumPy/scipy.special
     - 적용 위치:
         · estimate_pin_parameters() 내 그리드 탐색 (브로드캐스팅 전체 벡터화)
         · negative_log_likelihood() 콜백 (minimize 반복 호출마다 적용)
     - 추가: logsumexp 트릭으로 수치 안정성 향상
         · 변경 전: log(likelihood + 1e-300)  →  언더플로우 시 정보 손실
         · 변경 후: log(w0·L0 + w1·L1 + w2·L2)를 log-space에서 안전하게 계산

  ② [방안 4] imap_unordered chunksize 조정
     - 변경 전: chunksize 미지정 (기본값 1) → 종목 1개씩 IPC 왕복
     - 변경 후: chunksize=IMAP_CHUNKSIZE (기본 10) → 프로세스간 통신 횟수 감소
     - 효과: IPC 오버헤드 감소, 워커 유휴 시간 감소

[버그 수정 사항 - fixed 버전]
  ① [Fix 1] Date_int 변환 로직 제거 (🔴 날짜 오염 위험 수정)
     - 변경 전: Date → cast(pl.Int32) → numpy → dict 저장 → cast(pl.Date) 역변환
               Polars 버전에 따라 epoch days 해석이 달라질 수 있어 날짜 오염 가능
     - 변경 후: pl.Date 타입의 Python date 객체를 그대로 dict에 저장
               build_results_df에서 불필요한 Date_int 역변환 로직 제거

  ② [Fix 2] 20행 윈도우 캘린더 연속성 검사 추가 (🔴 PIN 왜곡 위험 수정)
     - 변경 전: 인덱스 기준 무조건 20행 사용 → 결측 영업일이 많으면
               수개월에 걸친 데이터가 하나의 윈도우로 묶일 수 있음
     - 변경 후: 윈도우 첫날~마지막날의 캘린더 일수가 MAX_WINDOW_CALENDAR_DAYS(40일)
               초과 시 해당 윈도우 스킵 → 비연속 데이터 자동 제외
     - 기준값: 20 영업일 ≈ 4주 ≈ 28일. 여유분 포함 40일로 설정
               (공휴일·연휴 등 최대 2주 연속 휴장을 커버)

  ③ [Fix 3] 윈도우 내 최소 유효 거래일 조건 추가 (🟡 희소 데이터 신뢰성 수정)
     - 변경 전: 20행이면 무조건 추정 → B/S 모두 0에 가까운 날이 다수여도 진행
     - 변경 후: 윈도우 내 B+S > 0인 유효 거래일이 MIN_VALID_DAYS(10일) 이상일 때만
               추정 실행 → 희소 데이터 종목의 신뢰성 없는 PIN 추정 방지
     - 기준값: 주별 코드의 `np.sum(mask) > 10` 조건과 동일한 수준으로 통일

  ④ [Fix 4] run_pin_calculation의 B/S Join 연도 필터 로직 수정 (🟡 코드 취약성 수정)
     - 변경 전: year_filter 존재 시 Symbol만 필터링 → 연도 필터가 적용된
               daily_bs를 또 한 번 Symbol로만 필터링하는 불필요한 분기 존재
               (현재는 무해하나 daily_bs가 재활용될 경우 버그 잠재)
     - 변경 후: year_filter 분기 제거, 항상 현재 메모리의 daily_bs(이미 필터 완료)를
               그대로 사용 → 로직을 단순·명확하게 통일

[설정]
  BASE_DIR              : 월별 .parquet 파일이 들어있는 루트 경로
  YEAR_FOLDERS          : 처리할 연도 폴더 목록 (None이면 전체 자동 탐색)
  OUTPUT_DIR            : 출력 결과 저장 경로
  CHECKPOINT_N          : 종목 N개마다 중간 저장 (중단 후 재개 지원)
  FORCE_REPROCESS_STEP1 : True면 기존 전처리 파일이 있어도 다시 집계
  IMAP_CHUNKSIZE        : imap_unordered 청크 크기 (종목 수·코어 수에 따라 조정)
  MAX_WINDOW_CALENDAR_DAYS : 20일 롤링 윈도우의 최대 허용 캘린더 일수 [Fix 2]
  MIN_VALID_DAYS        : 윈도우 내 유효 거래일 최소 개수 [Fix 3]

=============================================================================
"""

import os
import glob
import math
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
import warnings
import multiprocessing
import itertools
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")


# =============================================================================
# [공통 설정]
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = ['KOR_2017', 'KOR_2018']   # None이면 KOR_* 폴더 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

# Step 1 설정
FORCE_REPROCESS_STEP1 = False  # False: 이미 all_daily_bs.parquet가 있으면 Step 1 스킵

# Step 2 설정
CHECKPOINT_N          = 500    # 종목 500개마다 중간 저장
IMAP_CHUNKSIZE        = 10     # [방안 4] imap_unordered 청크 크기
                               # 종목당 처리 시간이 짧으면 늘리고(20~50),
                               # 길면 줄여서(5~10) 워커 부하 균형 조정

# [Fix 2] 캘린더 연속성 기준: 20 영업일 ≈ 4주 = 28일. 연휴 여유분 포함 40일
MAX_WINDOW_CALENDAR_DAYS = 40

# [Fix 3] 윈도우 내 유효 거래일(B+S > 0인 날) 최소 개수
# 주별 코드의 `np.sum(mask) > 10` 조건과 동일한 기준
MIN_VALID_DAYS        = 10


# =============================================================================
# [Step 1] 전처리 전용 함수들  (변경 없음)
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일에서 틱 데이터를 읽어 일별 매수/매도 건수로 집계합니다.
    Returns:
        Schema: Symbol (Utf8), Date (pl.Date), B (UInt32), S (UInt32)
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

    # Date 타입 보정
    if aggregated_df["Date"].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(pl.col("Date").cast(pl.Date))

    print(f"{aggregated_df.height:,} records")
    return aggregated_df


def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """
    base_dir 아래에서 KOR_YYYY 폴더 구조의 parquet 파일을 시간순으로 반환합니다.
    year_folders가 None이면 KOR_ 로 시작하는 모든 폴더를 자동 탐색합니다.
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


def run_preprocessing(base_dir: str, year_folders: list | None, output_dir: str) -> str:
    """
    모든 parquet 파일을 순회하며 일별 B/S 집계 후 단일 parquet으로 저장합니다.
    저장된 파일의 전체 경로를 반환합니다.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_daily_bs.parquet")

    # 강제 재실행이 아니고, 이미 파일이 존재하면 스킵
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
    print("[Step 1] 전처리 시작: 틱 데이터 → 일별 B/S 집계")
    print(f"{'='*65}\n")

    all_dfs = []
    skipped = 0

    for i, path in enumerate(parquet_files, 1):
        # 연도 폴더 구분 출력
        year_tag = next(
            (part for part in path.replace("\\", "/").split("/") if part.startswith("KOR_")),
            ""
        )
        if i == 1 or year_tag != next(
            (p for p in parquet_files[i - 2].replace("\\", "/").split("/") if p.startswith("KOR_")),
            ""
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

    # 전체 기간 합치기
    print(f"\n{'='*65}")
    print("전체 데이터 합산 및 정렬 중...")
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .sort(["Symbol", "Date"])
    )

    # 중복 제거 (같은 Symbol+Date가 여러 파일에 걸쳐 있을 경우 합산)
    full_df = (
        full_df
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("B").sum(),
            pl.col("S").sum(),
        ])
        .sort(["Symbol", "Date"])
    )

    # 저장
    full_df.write_parquet(output_path, compression="zstd")

    # 결과 요약
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
# [Step 2] PIN 계산 전용 함수들
# =============================================================================
#
# ── 핵심 변경: poisson.pmf → log-space 직접 계산 ─────────────────────────────
#
# [방안 1] negative_log_likelihood 콜백 최적화
#
#   변경 전 (원본):
#     prob = poisson.pmf(B, eb) * poisson.pmf(S, es)   # scipy 객체 dispatch ×4
#     likelihood = (1-α)·p0 + α·δ·p1 + α·(1-δ)·p2
#     return -np.sum(np.log(likelihood + 1e-300))       # 언더플로우 시 정보 손실
#
#   변경 후:
#     log P(k|λ) = k·ln(λ) - λ - lgamma(k+1)           # 순수 NumPy 연산
#     log-sum-exp 트릭으로 log(w0·L0 + w1·L1 + w2·L2) 안전하게 계산
#     → scipy 객체 오버헤드 제거 + 언더플로우 방지
#
# [방안 2] 그리드 탐색 (estimate_pin_parameters 상단) 동일하게 최적화
#
#   변경 전 (원본):
#     poisson.pmf(B_row, eb_g)  ← (243,1) 브로드캐스팅이지만 scipy dispatch 반복
#
#   변경 후:
#     _log_poisson_grid()로 완전 벡터화된 (243×20) log-pmf 행렬 한 번에 계산
#     동일한 logsumexp 트릭 적용
#
# =============================================================================

# ── [방안 1,2 공통] log-Poisson PMF 헬퍼 ────────────────────────────────────

def _log_poisson(k: np.ndarray, lam: float) -> np.ndarray:
    """
    log P(k | λ) = k·ln(λ) - λ - lgamma(k+1)  벡터 버전 (1D 배열 입력).

    원본 poisson.pmf(k, lam) 대체:
      - scipy 객체 생성/dispatch 없이 순수 NumPy + scipy.special.gammaln
      - gammaln(k+1) = log(k!)  →  정수 k에 대해 정확히 일치
      - lam=0 방어: k=0이면 log P = 0, k>0이면 -inf(-1e300)
    """
    safe_lam = max(lam, 1e-300)                     # log(0) 방어
    return k * math.log(safe_lam) - safe_lam - gammaln(k + 1.0)


def _log_poisson_grid(k_row: np.ndarray, lam_col: np.ndarray) -> np.ndarray:
    """
    log P(k | λ) 브로드캐스팅 버전 (그리드 탐색 전용).

    Args:
        k_row  : shape (1, N)   — 윈도우 관측값 (B 또는 S)
        lam_col: shape (G, 1)   — 그리드의 rate 파라미터
    Returns:
        shape (G, N)            — 각 그리드 × 관측값 조합의 log-pmf
    """
    safe_lam = np.where(lam_col > 0, lam_col, 1e-300)
    lgk = gammaln(k_row + 1.0)                      # (1, N) — 한 번만 계산
    return k_row * np.log(safe_lam) - safe_lam - lgk


# ── [방안 2] 그리드 탐색 최적화 ─────────────────────────────────────────────

def _grid_search(grid: np.ndarray, B: np.ndarray, S: np.ndarray) -> int:
    """
    243개 그리드 후보 중 NLL이 가장 낮은 인덱스를 반환합니다.

    변경 전 (원본):
      p1 = (1-a)·poisson.pmf(B, eb)·poisson.pmf(S, es)   ← scipy dispatch
      total_lik = p1 + p2 + p3
      nll = -np.sum(np.log(total_lik), axis=1)             ← 언더플로우 위험

    변경 후:
      log-pmf를 브로드캐스팅으로 (G×N) 행렬로 계산 → poisson.pmf 완전 제거
      logsumexp 트릭으로 log(w0·L0 + w1·L1 + w2·L2) 수치 안정적으로 계산
    """
    a_g  = grid[:, 0:1]          # (G, 1)
    d_g  = grid[:, 1:2]
    mu_g = grid[:, 2:3]
    eb_g = grid[:, 3:4]
    es_g = grid[:, 4:5]

    B_row = B.reshape(1, -1)     # (1, N)
    S_row = S.reshape(1, -1)

    # log-pmf 행렬: (G, N)
    lp_B_eb   = _log_poisson_grid(B_row, eb_g)
    lp_S_es   = _log_poisson_grid(S_row, es_g)
    lp_S_mues = _log_poisson_grid(S_row, mu_g + es_g)
    lp_B_mueb = _log_poisson_grid(B_row, mu_g + eb_g)

    # 각 시나리오의 log-likelihood: (G, N)
    l0 = lp_B_eb + lp_S_es        # 무정보
    l1 = lp_B_eb + lp_S_mues      # 나쁜 뉴스
    l2 = lp_B_mueb + lp_S_es      # 좋은 뉴스

    # 가중치를 log-space에 흡수: log(wk·Lk) = log(wk) + lk
    # → logsumexp([log_w0+l0, log_w1+l1, log_w2+l2]) 가 올바른 log-sum-exp 적용
    w0 = np.clip(1.0 - a_g,       1e-300, None)   # (G, 1)
    w1 = np.clip(a_g * d_g,       1e-300, None)
    w2 = np.clip(a_g * (1 - d_g), 1e-300, None)

    log_w0 = np.log(w0)           # (G, 1)
    log_w1 = np.log(w1)
    log_w2 = np.log(w2)

    # (3, G, N) 스택 → logsumexp over axis=0 → (G, N)
    log_terms = np.stack([log_w0 + l0,
                          log_w1 + l1,
                          log_w2 + l2], axis=0)
    log_lik_per_day = logsumexp(log_terms, axis=0)   # (G, N)

    # NLL per grid = -sum over days
    nll_scores = -np.sum(log_lik_per_day, axis=1)    # (G,)
    return int(np.argmin(nll_scores))


# ── [방안 1] MLE 콜백 최적화 ─────────────────────────────────────────────────

def _make_nll(B: np.ndarray, S: np.ndarray):
    """
    클로저로 negative_log_likelihood 생성.
    B, S는 float64 배열로 미리 변환해 콜백 내 불필요한 캐스팅 제거.

    변경 전 (원본):
      poisson.pmf(B, eb)  →  scipy 객체 dispatch, Python 레벨 검증 반복
      log(likelihood + 1e-300)  →  언더플로우 시 1e-300으로 뭉개져 정보 손실

    변경 후:
      _log_poisson(k, λ)  →  순수 NumPy 연산 (gammaln + 벡터 산술)
      logsumexp 트릭  →  가중치를 log-space에 흡수 후 안전하게 합산
        log(w0·L0 + w1·L1 + w2·L2)
        = logsumexp([log_w0+log_L0, log_w1+log_L1, log_w2+log_L2])
    """
    B_f = B.astype(np.float64)
    S_f = S.astype(np.float64)
    lgk_B = gammaln(B_f + 1.0)    # log(B!) — 콜백 밖에서 한 번만 계산
    lgk_S = gammaln(S_f + 1.0)    # log(S!)

    def negative_log_likelihood(params):
        alpha, delta, mu, eb, es = params
        if not (0 <= alpha <= 1 and 0 <= delta <= 1
                and mu >= 0 and eb >= 0 and es >= 0):
            return np.inf

        # ── log P(k | λ) = k·ln(λ) - λ - lgamma(k+1) ───────────────────────
        safe_eb     = max(eb,      1e-300)
        safe_es     = max(es,      1e-300)
        safe_mues   = max(mu + es, 1e-300)
        safe_mueb   = max(mu + eb, 1e-300)

        log_pB_eb   = B_f * math.log(safe_eb)   - safe_eb   - lgk_B
        log_pS_es   = S_f * math.log(safe_es)   - safe_es   - lgk_S
        log_pS_mues = S_f * math.log(safe_mues) - safe_mues - lgk_S
        log_pB_mueb = B_f * math.log(safe_mueb) - safe_mueb - lgk_B

        # 각 시나리오의 log-likelihood (per day)
        l0 = log_pB_eb + log_pS_es        # 무정보
        l1 = log_pB_eb + log_pS_mues      # 나쁜 뉴스
        l2 = log_pB_mueb + log_pS_es      # 좋은 뉴스

        # 가중치를 log-space에 흡수 후 logsumexp
        w0 = max(1.0 - alpha,             1e-300)
        w1 = max(alpha * delta,           1e-300)
        w2 = max(alpha * (1.0 - delta),   1e-300)

        log_terms = np.stack([
            math.log(w0) + l0,
            math.log(w1) + l1,
            math.log(w2) + l2,
        ], axis=0)                        # (3, N)
        log_lik = logsumexp(log_terms, axis=0)   # (N,)

        return -np.sum(log_lik)

    return negative_log_likelihood


# ── 통합 추정 함수 ────────────────────────────────────────────────────────────

def estimate_pin_parameters(B_array: np.ndarray, S_array: np.ndarray,
                             grid_combinations: np.ndarray) -> dict:
    """
    EKOP(1996) 모델 기반 MLE로 파라미터(alpha, delta, mu, epsilon_b, epsilon_s)를 추정합니다.

    [변경 요약]
      - 그리드 탐색: poisson.pmf 브로드캐스팅 → _grid_search() (log-space 벡터화)
      - MLE 콜백:   poisson.pmf ×4 호출 → _make_nll() (log-space, gammaln 사전 계산)
    """
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    # ── 1단계: 그리드 탐색으로 최적 초기값 선택 ──────────────────────────
    best_idx           = _grid_search(grid_combinations, B, S)
    best_initial_guess = grid_combinations[best_idx]

    # ── 2단계: L-BFGS-B로 MLE 정밀 최적화 ────────────────────────────────
    nll_fn = _make_nll(B, S)      # lgamma(k!) 사전 계산 후 클로저 생성

    try:
        result = minimize(
            nll_fn,
            x0=best_initial_guess,
            bounds=[(0, 1), (0, 1), (0, None), (0, None), (0, None)],
            method="L-BFGS-B",
        )
    except Exception:
        return {"converged": False}

    if not result.success:
        return {"converged": False}

    alpha, delta, mu, eb, es = result.x
    denom = alpha * mu + eb + es
    pin   = (alpha * mu) / (denom if denom != 0 else 1e-9)

    return {
        "alpha": alpha, "delta": delta, "mu": mu,
        "eb": eb, "es": es, "PIN": pin, "converged": True,
    }


# =============================================================================
# 워커 프로세스 초기화 및 종목 처리
# =============================================================================

GLOBAL_GRID = None

def init_worker(grid: np.ndarray) -> None:
    global GLOBAL_GRID
    GLOBAL_GRID = grid


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """
    단일 종목의 전체 시계열 데이터를 받아 20일 롤링 윈도우로 PIN을 계산합니다.

    [Fix 1] Date 저장 방식 변경
      - 변경 전: Date → cast(pl.Int32) → numpy int → dict 저장
                 → build_results_df에서 cast(pl.Date)로 역변환
                 → Polars 버전에 따라 epoch 기준이 달라질 수 있어 날짜 오염 위험
      - 변경 후: sym_df["Date"][i]로 pl.Date의 Python date 객체를 직접 저장
                 → 타입 변환 없이 원본 날짜 그대로 유지

    [Fix 2] 캘린더 연속성 검사 추가
      - 변경 전: 인덱스 기준 직전 20행을 무조건 사용
      - 변경 후: 윈도우 첫날~마지막날의 캘린더 일수가 MAX_WINDOW_CALENDAR_DAYS 초과 시 스킵
                 → 상장폐지·재상장·장기 거래중단 종목의 비연속 윈도우 자동 제외

    [Fix 3] 윈도우 내 유효 거래일 최소 조건 추가
      - 변경 전: 20행이면 무조건 추정
      - 변경 후: B+S > 0인 유효 거래일이 MIN_VALID_DAYS 미만이면 스킵
                 → 거래가 극히 드문 날이 많은 윈도우의 신뢰성 없는 추정 방지
    """
    sym, sym_df = data_tuple

    # 종목 전체 데이터가 윈도우 크기(20)보다 작으면 스킵
    if sym_df.height < 20:
        return []

    Bs    = sym_df["B"].to_numpy()
    Ss    = sym_df["S"].to_numpy()
    # [Fix 1] cast(pl.Int32) 제거 → pl.Date 타입의 Python date 객체 리스트로 직접 추출
    dates = sym_df["Date"].to_list()   # list[datetime.date]

    results = []
    for i in range(19, len(dates)):
        window_B = Bs[i - 19 : i + 1]
        window_S = Ss[i - 19 : i + 1]

        if len(window_B) != 20:
            continue

        # [Fix 2] 캘린더 연속성 검사
        # 윈도우 첫날과 마지막날의 실제 경과 일수가 기준을 초과하면 스킵
        calendar_span = (dates[i] - dates[i - 19]).days
        if calendar_span > MAX_WINDOW_CALENDAR_DAYS:
            continue

        # [Fix 3] 유효 거래일 최소 조건 검사
        # B+S > 0인 날만 카운트하여 MIN_VALID_DAYS 미만이면 스킵
        valid_days = int(np.sum((window_B + window_S) > 0))
        if valid_days < MIN_VALID_DAYS:
            continue

        est = estimate_pin_parameters(window_B, window_S, GLOBAL_GRID)

        if est["converged"]:
            results.append({
                "Symbol": sym,
                # [Fix 1] date 객체를 그대로 저장 (Int32 변환 없음)
                "Date":   dates[i],
                "a":      est["alpha"],
                "d":      est["delta"],
                "u":      est["mu"],
                "eb":     est["eb"],
                "es":     est["es"],
                "PIN":    est["PIN"],
            })

    return results


def build_results_df(estimates: list[dict]) -> pl.DataFrame:
    """
    추정 결과 dict 리스트를 Polars DataFrame으로 변환합니다.

    [Fix 1] Date_int → Date 역변환 로직 제거
      - 변경 전: pl.col("Date_int").cast(pl.Date).alias("Date") + .drop("Date_int")
                 → cast(pl.Int32) → cast(pl.Date) 왕복 변환으로 버전 의존성 존재
      - 변경 후: dict에 date 객체가 그대로 저장되어 있으므로 별도 변환 불필요
                 Polars가 Python date 객체를 pl.Date로 자동 인식
    """
    if not estimates:
        return pl.DataFrame()

    return (
        pl.DataFrame(estimates)
        # Date 컬럼이 Python date 객체로 저장됐으므로 cast만으로 타입 보정
        .with_columns(pl.col("Date").cast(pl.Date))
        .select(["Symbol", "Date", "a", "d", "u", "eb", "es", "PIN"])
    )


def save_checkpoint(estimates: list[dict], checkpoint_idx: int, intermediate_dir: str) -> str:
    """처리된 추정 결과를 배치 단위로 중간 저장합니다."""
    os.makedirs(intermediate_dir, exist_ok=True)
    df   = build_results_df(estimates)
    path = os.path.join(intermediate_dir, f"pin_checkpoint_{checkpoint_idx:04d}.parquet")
    df.write_parquet(path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] {df.height:,} records → {os.path.basename(path)}")
    return path


def load_already_done_symbols(intermediate_dir: str) -> set[str]:
    """중단 후 재개 시 이미 처리된 종목 목록을 체크포인트에서 복원합니다."""
    if not os.path.exists(intermediate_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(intermediate_dir) if f.startswith("pin_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols = set()
    for fname in checkpoint_files:
        path = os.path.join(intermediate_dir, fname)
        df   = pl.read_parquet(path, columns=["Symbol"])
        done_symbols.update(df["Symbol"].unique().to_list())

    print(f"[재개 모드] 기존 체크포인트 {len(checkpoint_files)}개 발견 → "
          f"완료된 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


def run_pin_calculation(daily_bs_path: str, output_dir: str,
                        year_filter: list[int] | None = None,
                        checkpoint_n: int = 500) -> pl.DataFrame:
    """
    all_daily_bs.parquet에서 종목별 20일 롤링 PIN을 계산합니다.

    [Fix 4] B/S Join의 연도 필터 분기 제거
      - 변경 전:
          daily_bs.filter(pl.col("Symbol").is_in(...)) if year_filter else daily_bs
          → year_filter가 있을 때 Symbol 필터만 적용하는 불명확한 분기
          → 함수 상단에서 이미 year_filter로 daily_bs를 필터링했음에도
             마지막 Join에서 또 다른 조건 분기가 존재해 코드 의도가 불명확
          → daily_bs가 외부에서 재사용될 경우 연도 필터 누락 버그 잠재
      - 변경 후:
          항상 daily_bs(이미 상단에서 연도 필터 완료된 상태)를 그대로 사용
          → 로직 단순화, 의도 명확화, 잠재 버그 제거
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] PIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일: {daily_bs_path}")

    daily_bs = pl.read_parquet(daily_bs_path)

    if year_filter:
        daily_bs = daily_bs.filter(pl.col("Date").dt.year().is_in(year_filter))
        print(f"  연도 필터: {year_filter}")

    print(f"  전체 행 수: {daily_bs.height:,}")
    print(f"  종목 수   : {daily_bs['Symbol'].n_unique():,}")
    print(f"  날짜 범위 : {daily_bs['Date'].min()} ~ {daily_bs['Date'].max()}")

    print("\n[Grid 초기값 생성 중...]")
    grid_matrix = np.array(
        list(itertools.product(
            [0.1, 0.5, 0.9],   # alpha
            [0.1, 0.5, 0.9],   # delta
            [20, 200, 2000],   # mu
            [20, 200, 2000],   # epsilon_b
            [20, 200, 2000],   # epsilon_s
        )),
        dtype=np.float64,
    )
    print(f"  Grid 크기 : {grid_matrix.shape[0]} combinations")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어  : {num_cores}개")
    print(f"  IPC 청크  : {IMAP_CHUNKSIZE} 종목/청크  [방안 4]")
    print(f"  캘린더 상한: {MAX_WINDOW_CALENDAR_DAYS}일  [Fix 2]")
    print(f"  최소 유효일: {MIN_VALID_DAYS}일  [Fix 3]")

    done_symbols = load_already_done_symbols(intermediate_dir)

    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    # partition_by로 종목별 분할
    symbol_partitions = daily_bs.partition_by("Symbol", maintain_order=False)
    symbol_dfs = {}
    for part in symbol_partitions:
        sym = part["Symbol"][0]
        if sym in remaining_symbols:
            symbol_dfs[sym] = part.sort("Date")

    grouped_data = [(sym, symbol_dfs[sym]) for sym in remaining_symbols if sym in symbol_dfs]

    print(f"{'='*65}")
    print("PIN 추정 시작...")
    print(f"{'='*65}\n")

    checkpoint_files = sorted(
        os.path.join(intermediate_dir, f)
        for f in os.listdir(intermediate_dir)
        if f.startswith("pin_checkpoint_")
    )
    next_checkpoint_idx = len(checkpoint_files)

    batch_estimates = []
    processed_count = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(grid_matrix,),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_symbol, grouped_data,
                                chunksize=IMAP_CHUNKSIZE),
            total=len(grouped_data),
            desc="  Estimating",
        ):
            batch_estimates.extend(res)
            processed_count += 1

            if processed_count % checkpoint_n == 0:
                save_checkpoint(batch_estimates, next_checkpoint_idx, intermediate_dir)
                checkpoint_files.append(
                    os.path.join(intermediate_dir, f"pin_checkpoint_{next_checkpoint_idx:04d}.parquet")
                )
                next_checkpoint_idx += 1
                batch_estimates = []

    if batch_estimates:
        save_checkpoint(batch_estimates, next_checkpoint_idx, intermediate_dir)
        checkpoint_files.append(
            os.path.join(intermediate_dir, f"pin_checkpoint_{next_checkpoint_idx:04d}.parquet")
        )

    print(f"\n{'='*65}")
    print(f"체크포인트 {len(checkpoint_files)}개 병합 중...")

    all_checkpoint_files = sorted(
        os.path.join(intermediate_dir, f)
        for f in os.listdir(intermediate_dir)
        if f.startswith("pin_checkpoint_")
    )

    if not all_checkpoint_files:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    final_df = (
        pl.concat([pl.read_parquet(p) for p in all_checkpoint_files], how="vertical")
        .sort(["Symbol", "Date"])
    )

    # B/S 원본 컬럼 합치기
    # [Fix 4] year_filter 분기 제거: daily_bs는 이미 상단에서 연도 필터가 완료된 상태
    #         → 항상 동일한 daily_bs를 사용하여 로직 단순화
    final_df = (
        final_df
        .join(
            daily_bs,   # 이미 연도 필터 적용 완료된 daily_bs 그대로 사용
            on=["Symbol", "Date"],
            how="left",
        )
        .select(["Symbol", "Date", "B", "S", "a", "d", "u", "eb", "es", "PIN"])
        .sort(["Symbol", "Date"])
    )

    return final_df


# =============================================================================
# [통합 파이프라인 실행부]
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()
    start = datetime.now()

    # 1. 전처리 실행 (Step 1)
    daily_bs_path = run_preprocessing(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] 전처리 결과 파일이 없어 PIN 계산을 종료합니다.")
        exit(1)

    # 2. 파라미터 변환 (KOR_YYYY -> YYYY)
    if YEAR_FOLDERS:
        year_filter = [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
    else:
        year_filter = None

    # 3. PIN 추정 실행 (Step 2)
    result = run_pin_calculation(
        daily_bs_path=daily_bs_path,
        output_dir=OUTPUT_DIR,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    # 4. 최종 결과물 저장
    if not result.is_empty():
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M")
        year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename = f"pin_daily_rolling_{year_tag}_{timestamp}"

        parquet_path = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result.write_parquet(parquet_path, compression="zstd")
        print(f"\n[저장 완료] {parquet_path}")

        csv_path = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result.head(1000).write_csv(csv_path)
        print(f"[샘플 저장] {csv_path}")

        print("\n[미리보기]")
        print(result.head(20))

        print("\n[통계]")
        print(f"  전체 레코드       : {result.height:,}")
        pin_ok = result.filter(pl.col("PIN").is_not_null()).height
        print(f"  PIN 추정 성공     : {pin_ok:,}")
        print(f"  PIN 추정 실패/없음: {result.height - pin_ok:,}")

        sym_stats = (
            result
            .group_by("Symbol")
            .agg([
                pl.len().alias("total_days"),
                pl.col("PIN").is_not_null().sum().alias("pin_days"),
            ])
            .with_columns(
                (pl.col("pin_days") / pl.col("total_days") * 100).alias("coverage_pct")
            )
        )
        print("\n[종목별 커버리지 샘플]")
        print(sym_stats.head(10))

    else:
        print("\n[Warning] 결과 DataFrame이 비어 있습니다.")

    elapsed = datetime.now() - start
    print(f"\n총 소요 시간: {elapsed}")
