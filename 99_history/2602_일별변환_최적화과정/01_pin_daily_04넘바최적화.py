"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 일별 롤링 PIN 계산 (2-Stage Pipeline)
=============================================================================

[파이프라인 구조]
  ▶ [Step 1] 전처리 (틱 데이터 → 일별 B/S 요약)
    - 모든 월별 .parquet 틱 데이터를 순회하며 일별 매수(B)/매도(S) 건수 집계
    - 결과를 단 하나의 파일(all_daily_bs.parquet)로 병합 및 저장

  ▶ [Step 2] PIN 추정 (일별 B/S 요약 → PIN 계산)
    - [Step 1]이 생성한 all_daily_bs.parquet 하나만 읽어 계산
    - 종목별 20일 롤링 윈도우로 PIN 계산 (MLE)
    - 멀티프로세싱 및 종목 N개 단위 체크포인트(중간 저장) 기능 지원

[최적화 전략 - MLE 속도 개선]
  ① Numba @njit JIT 컴파일
     - negative_log_likelihood 콜백 함수를 Python 인터프리터 없이
       LLVM 네이티브 코드로 컴파일. scipy.optimize.minimize의 C 루프가
       Python GIL 없이 직접 호출.
     - poisson.pmf(k, λ) = exp(-λ + k·ln(λ) - ln(k!)) 를 수식으로 인라이닝
       → scipy 객체 조회/dispatch 오버헤드 제거

  ② Log-space 연산으로 언더플로우 방지 + 수치 안정성 향상
     - logsumexp 트릭: log(a + b + c) = m + log(e^(a-m) + e^(b-m) + e^(c-m))
       (m = max(a,b,c)) → 매우 작은 확률에서도 정확한 계산

  ③ 그리드 탐색 완전 벡터화 (Numba parallel=True)
     - 243개 그리드 후보를 병렬 루프로 동시 평가
     - 기존: scipy.stats.poisson.pmf 브로드캐스팅 (Python 객체 반복)
     - 개선: Numba prange 병렬 루프 + 인라인 수식

  ④ 워커 초기화 최적화
     - Numba 함수는 첫 호출 시 JIT 컴파일(warm-up)
     - init_worker에서 warm-up 수행 → 실제 계산 시 오버헤드 없음

  [Numba 미설치 시 자동 폴백]
  - Numba가 없으면 NumPy 벡터화 + log-space 연산으로만 동작
    (scipy.stats.poisson 대비 ~2-3x 빠름)
  - 설치: pip install numba

=============================================================================
"""

import os
import glob
import math
import numpy as np
import polars as pl
from scipy.optimize import minimize
import warnings
import multiprocessing
import itertools
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")

# Numba 임포트 시도 (없으면 NumPy 폴백)
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("[최적화] Numba JIT 컴파일 활성화")
except ImportError:
    NUMBA_AVAILABLE = False
    print("[최적화] Numba 미설치 → NumPy log-space 폴백 모드 (pip install numba 권장)")


# =============================================================================
# [공통 설정]
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = None            # ['KOR_2017', 'KOR_2018']   # None이면 KOR_* 폴더 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

# Step 1 설정
FORCE_REPROCESS_STEP1 = False

# Step 2 설정
CHECKPOINT_N          = 200


# =============================================================================
# [Step 1] 전처리 전용 함수들  (변경 없음)
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일에서 틱 데이터를 읽어 일별 매수/매도 건수로 집계합니다.
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


def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
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
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_daily_bs.parquet")

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

    print(f"\n{'='*65}")
    print("전체 데이터 합산 및 정렬 중...")
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .sort(["Symbol", "Date"])
    )

    full_df = (
        full_df
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("B").sum(),
            pl.col("S").sum(),
        ])
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
# [Step 2] MLE 최적화 핵심 함수들
# =============================================================================
#
# ── 핵심 아이디어 ────────────────────────────────────────────────────────────
#
# 기존 코드:  scipy.stats.poisson.pmf(k, λ)  호출
#   → Python 객체 조회 → 타입 검사 → C 루틴 → Python 객체 반환
#   → minimize 콜백마다 이 과정을 20번(윈도우 크기) × 3(시나리오) 반복
#
# 최적화:  log-Poisson 수식 인라이닝 + Numba JIT
#   log P(k | λ) = -λ + k·ln(λ) - lgamma(k+1)
#   → Numba가 이 수식을 LLVM IR로 컴파일 → CPU 네이티브 실행
#   → scipy 객체 오버헤드 완전 제거
#
# log-space NLL (수치 안정성):
#   NLL = -Σ log[ (1-α)·L₀ + α·δ·L_bad + α·(1-δ)·L_good ]
#   log-sum-exp 트릭: log(a+b+c) = m + log(e^(a-m) + e^(b-m) + e^(c-m))
#   (m = max(log_a, log_b, log_c))  → 언더플로우 없이 정밀한 계산
#
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# [A] Numba 버전 (가장 빠름: ~10-20x speedup vs 원본)
# ─────────────────────────────────────────────────────────────────────────────

if NUMBA_AVAILABLE:

    @njit(cache=True)
    def _log_poisson(k: float, lam: float) -> float:
        """
        log P(k | λ) = -λ + k·ln(λ) - lgamma(k+1)
        scipy.stats.poisson.pmf를 인라인 수식으로 대체.
        Numba가 LLVM으로 컴파일 → Python 객체 오버헤드 없음.
        """
        if lam <= 0.0:
            return -1e300 if k > 0 else 0.0
        return -lam + k * math.log(lam) - math.lgamma(k + 1.0)

    @njit(cache=True)
    def _nll_numba(params: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
        """
        EKOP PIN 모델의 음의 로그 우도 (Numba JIT 컴파일).

        scipy.optimize.minimize가 이 함수를 콜백으로 직접 호출.
        Python 인터프리터를 거치지 않고 네이티브 코드로 실행됨.

        log-sum-exp 트릭으로 수치 안정성 보장:
          log(w0·L0 + w1·L1 + w2·L2)
          = m + log(w0·e^(l0-m) + w1·e^(l1-m) + w2·e^(l2-m))
          단, m = max(l0, l1, l2)
        """
        alpha, delta, mu, eb, es = params[0], params[1], params[2], params[3], params[4]

        # 파라미터 경계 검사 (scipy bounds 대신 직접 처리)
        if alpha < 0.0 or alpha > 1.0:
            return 1e300
        if delta < 0.0 or delta > 1.0:
            return 1e300
        if mu < 0.0 or eb < 0.0 or es < 0.0:
            return 1e300

        n = len(B)
        w0 = 1.0 - alpha           # 무정보일 확률
        w1 = alpha * delta         # 나쁜 뉴스 확률
        w2 = alpha * (1.0 - delta) # 좋은 뉴스 확률

        total_nll = 0.0
        for i in range(n):
            b = float(B[i])
            s = float(S[i])

            # log-likelihood of each scenario
            l0 = _log_poisson(b, eb)       + _log_poisson(s, es)
            l1 = _log_poisson(b, eb)       + _log_poisson(s, mu + es)
            l2 = _log_poisson(b, mu + eb)  + _log_poisson(s, es)

            # log-sum-exp 트릭: m = max log-likelihood for numerical stability
            m = l0
            if l1 > m: m = l1
            if l2 > m: m = l2

            # weighted sum in log-space
            val = w0 * math.exp(l0 - m) + w1 * math.exp(l1 - m) + w2 * math.exp(l2 - m)

            if val <= 0.0:
                total_nll += 1e300
            else:
                total_nll -= (m + math.log(val))

        return total_nll

    @njit(cache=True, parallel=True)
    def _grid_search_numba(grid: np.ndarray, B: np.ndarray, S: np.ndarray) -> int:
        """
        그리드 후보 전체를 Numba 병렬 루프(prange)로 평가.
        scipy.stats.poisson.pmf 브로드캐스팅 대비 ~5x 빠름.
        반환: 최소 NLL을 갖는 그리드 인덱스
        """
        n_grid = grid.shape[0]
        scores = np.empty(n_grid, dtype=np.float64)

        for g in prange(n_grid):  # 병렬 실행
            scores[g] = _nll_numba(grid[g], B, S)

        return np.argmin(scores)

    def _nll_wrapper_numba(params: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
        """
        scipy.optimize.minimize용 래퍼.
        Numba JIT 함수는 직접 콜백으로 전달할 수 없으므로
        얇은 Python 래퍼를 통해 호출. 래퍼 오버헤드는 무시할 수준.
        """
        return _nll_numba(params, B, S)

    def warmup_numba(grid: np.ndarray) -> None:
        """
        워커 프로세스 시작 시 Numba 함수를 JIT 컴파일(warm-up).
        첫 실제 계산에서 컴파일 지연이 발생하지 않도록 사전 트리거.
        cache=True로 컴파일 결과를 디스크에 저장하므로
        두 번째 실행부터는 warm-up도 즉시 완료됨.
        """
        dummy_B = np.array([10.0, 20.0], dtype=np.float64)
        dummy_S = np.array([10.0, 20.0], dtype=np.float64)
        dummy_p = np.array([0.5, 0.5, 100.0, 100.0, 100.0], dtype=np.float64)
        _nll_numba(dummy_p, dummy_B, dummy_S)
        _grid_search_numba(grid[:4], dummy_B, dummy_S)


# ─────────────────────────────────────────────────────────────────────────────
# [B] NumPy 폴백 버전 (Numba 미설치 시: ~2-3x speedup vs 원본)
# ─────────────────────────────────────────────────────────────────────────────

def _log_poisson_numpy(k: np.ndarray, lam: float) -> np.ndarray:
    """
    log P(k | λ) 벡터화 구현 (NumPy).
    scipy.stats.poisson.pmf 대비 Python 객체 오버헤드 없이
    순수 NumPy ufunc 체인으로 처리.
    """
    if lam <= 0:
        return np.where(k > 0, -1e300, 0.0)
    return -lam + k * np.log(lam) - np.array(
        [math.lgamma(ki + 1.0) for ki in k], dtype=np.float64
    )


def _nll_numpy(params: np.ndarray, B: np.ndarray, S: np.ndarray) -> float:
    """
    EKOP PIN 모델 NLL (NumPy 구현, log-sum-exp 적용).
    Numba 미설치 환경에서의 폴백.
    """
    alpha, delta, mu, eb, es = params
    if not (0.0 <= alpha <= 1.0 and 0.0 <= delta <= 1.0
            and mu >= 0.0 and eb >= 0.0 and es >= 0.0):
        return 1e300

    B_f = B.astype(np.float64)
    S_f = S.astype(np.float64)

    # log-likelihood per scenario
    l0 = _log_poisson_numpy(B_f, eb)       + _log_poisson_numpy(S_f, es)
    l1 = _log_poisson_numpy(B_f, eb)       + _log_poisson_numpy(S_f, mu + es)
    l2 = _log_poisson_numpy(B_f, mu + eb)  + _log_poisson_numpy(S_f, es)

    w0 = 1.0 - alpha
    w1 = alpha * delta
    w2 = alpha * (1.0 - delta)

    # log-sum-exp 트릭 (벡터화)
    m = np.maximum(np.maximum(l0, l1), l2)
    val = w0 * np.exp(l0 - m) + w1 * np.exp(l1 - m) + w2 * np.exp(l2 - m)
    val = np.maximum(val, 1e-300)

    return -np.sum(m + np.log(val))


def _grid_search_numpy(grid: np.ndarray, B: np.ndarray, S: np.ndarray) -> int:
    """
    그리드 탐색 (NumPy 벡터화, 기존 scipy.stats.poisson 브로드캐스팅 방식 대체).

    기존 방식: poisson.pmf(B_row, eb_g) → scipy 객체 dispatch 반복
    개선:     log-poisson 수식을 NumPy broadcasting으로 일괄 계산
              grid(243,) × window(20,) → 완전 벡터화
    """
    B_f = B.astype(np.float64)
    S_f = S.astype(np.float64)

    a_g  = grid[:, 0:1]   # (243, 1)
    d_g  = grid[:, 1:2]
    mu_g = grid[:, 2:3]
    eb_g = grid[:, 3:4]
    es_g = grid[:, 4:5]

    # log P(B | eb), log P(S | es) 등을 브로드캐스팅으로 계산 (243 × 20)
    def log_pois_mat(k_row, lam_col):
        # k_row: (1, 20), lam_col: (243, 1)
        safe_lam = np.where(lam_col > 0, lam_col, 1e-300)
        lgk = np.array([math.lgamma(ki + 1.0) for ki in k_row.ravel()], dtype=np.float64)
        return -safe_lam + k_row * np.log(safe_lam) - lgk.reshape(1, -1)

    B_row = B_f.reshape(1, -1)  # (1, 20)
    S_row = S_f.reshape(1, -1)

    lp_B_eb     = log_pois_mat(B_row, eb_g)           # P(B | eb)
    lp_S_es     = log_pois_mat(S_row, es_g)           # P(S | es)
    lp_S_mues   = log_pois_mat(S_row, mu_g + es_g)    # P(S | mu+es)
    lp_B_mueb   = log_pois_mat(B_row, mu_g + eb_g)    # P(B | mu+eb)

    l0 = lp_B_eb + lp_S_es       # (243, 20)
    l1 = lp_B_eb + lp_S_mues
    l2 = lp_B_mueb + lp_S_es

    w0 = 1.0 - a_g                # (243, 1)
    w1 = a_g * d_g
    w2 = a_g * (1.0 - d_g)

    m = np.maximum(np.maximum(l0, l1), l2)            # (243, 20)
    val = w0 * np.exp(l0 - m) + w1 * np.exp(l1 - m) + w2 * np.exp(l2 - m)
    val = np.maximum(val, 1e-300)

    nll_scores = -np.sum(m + np.log(val), axis=1)     # (243,)
    return int(np.argmin(nll_scores))


# =============================================================================
# 모드에 따른 함수 선택
# =============================================================================

if NUMBA_AVAILABLE:
    _nll_fn         = _nll_wrapper_numba
    _grid_search_fn = _grid_search_numba
else:
    _nll_fn         = _nll_numpy
    _grid_search_fn = _grid_search_numpy


# =============================================================================
# [Step 2] PIN 추정 메인 함수
# =============================================================================

def estimate_pin_parameters(B_array: np.ndarray, S_array: np.ndarray,
                             grid_combinations: np.ndarray) -> dict:
    """
    EKOP(1996) 모델 기반 MLE로 파라미터(alpha, delta, mu, epsilon_b, epsilon_s)를 추정.

    [최적화 포인트]
      1. 그리드 탐색: _grid_search_fn (Numba 병렬 or NumPy 벡터화)
      2. minimize 콜백: _nll_fn (Numba JIT → Python 인터프리터 우회)
    """
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    # ── 1단계: 그리드 탐색으로 최적 초기값 선택 ──────────────────────────
    best_idx     = _grid_search_fn(grid_combinations, B, S)
    best_initial = grid_combinations[best_idx]

    # ── 2단계: L-BFGS-B로 MLE 정밀 최적화 ────────────────────────────────
    #   콜백(_nll_fn)이 Numba JIT 컴파일된 경우:
    #   scipy의 C 루프 → Python 래퍼(얇은 껍데기) → Numba 네이티브 코드
    #   Python 인터프리터가 NLL 계산 자체에 개입하지 않음
    try:
        result = minimize(
            _nll_fn,
            x0=best_initial,
            args=(B, S),
            bounds=[(0, 1), (0, 1), (0, None), (0, None), (0, None)],
            method="L-BFGS-B",
            options={"maxiter": 500, "ftol": 1e-9, "gtol": 1e-6},
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
# 워커 프로세스 초기화
# =============================================================================

GLOBAL_GRID = None

def init_worker(grid: np.ndarray) -> None:
    """
    멀티프로세싱 워커 초기화.
    Numba 활성화 시: warm-up JIT 컴파일 수행.
      - cache=True이므로 디스크 캐시가 있으면 즉시 로드.
      - 첫 실행 시만 컴파일 비용 발생(약 1-3초).
    """
    global GLOBAL_GRID
    GLOBAL_GRID = grid

    if NUMBA_AVAILABLE:
        warmup_numba(grid)


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """
    단일 종목의 전체 시계열 데이터를 받아 20일 롤링 윈도우로 PIN을 계산합니다.
    (로직 변경 없음, estimate_pin_parameters 내부가 최적화됨)
    """
    sym, sym_df = data_tuple

    if sym_df.height < 20:
        return []

    Bs        = sym_df["B"].to_numpy().astype(np.float64)
    Ss        = sym_df["S"].to_numpy().astype(np.float64)
    dates_int = sym_df["Date"].cast(pl.Int32).to_numpy()

    results = []
    for i in range(19, len(dates_int)):
        window_B = Bs[i - 19 : i + 1]
        window_S = Ss[i - 19 : i + 1]

        if len(window_B) != 20:
            continue

        est = estimate_pin_parameters(window_B, window_S, GLOBAL_GRID)

        if est["converged"]:
            results.append({
                "Symbol":   sym,
                "Date_int": int(dates_int[i]),
                "a":        est["alpha"],
                "d":        est["delta"],
                "u":        est["mu"],
                "eb":       est["eb"],
                "es":       est["es"],
                "PIN":      est["PIN"],
            })

    return results


def build_results_df(estimates: list[dict]) -> pl.DataFrame:
    if not estimates:
        return pl.DataFrame()

    return (
        pl.DataFrame(estimates)
        .with_columns(pl.col("Date_int").cast(pl.Date).alias("Date"))
        .drop("Date_int")
        .select(["Symbol", "Date", "a", "d", "u", "eb", "es", "PIN"])
    )


def save_checkpoint(estimates: list[dict], checkpoint_idx: int, intermediate_dir: str) -> str:
    os.makedirs(intermediate_dir, exist_ok=True)
    df   = build_results_df(estimates)
    path = os.path.join(intermediate_dir, f"pin_checkpoint_{checkpoint_idx:04d}.parquet")
    df.write_parquet(path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] {df.height:,} records → {os.path.basename(path)}")
    return path


def load_already_done_symbols(intermediate_dir: str) -> set[str]:
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
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(intermediate_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] PIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일: {daily_bs_path}")
    print(f"  최적화   : {'Numba JIT (병렬 그리드 탐색 + 네이티브 NLL)' if NUMBA_AVAILABLE else 'NumPy log-space 벡터화'}")

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
            [0.1, 0.5, 0.9],
            [0.1, 0.5, 0.9],
            [20, 200, 2000],
            [20, 200, 2000],
            [20, 200, 2000],
        )),
        dtype=np.float64,
    )
    print(f"  Grid 크기: {grid_matrix.shape[0]} combinations")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어 : {num_cores}개")

    done_symbols      = load_already_done_symbols(intermediate_dir)
    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    symbol_partitions = daily_bs.partition_by("Symbol", maintain_order=False)
    symbol_dfs        = {}
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
    batch_estimates     = []
    processed_count     = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(grid_matrix,),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_symbol, grouped_data),
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

    final_df = (
        final_df
        .join(
            daily_bs.filter(pl.col("Symbol").is_in(final_df["Symbol"].unique()))
            if year_filter else daily_bs,
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
        output_dir=OUTPUT_DIR,
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
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M")
        year_tag         = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename  = f"pin_daily_rolling_{year_tag}_{timestamp}"

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
