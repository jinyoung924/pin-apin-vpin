"""
=============================================================================
PIN (Probability of Informed Trading) 추정 모듈 — EKOP 1996 모델
=============================================================================

60일 롤링 윈도우로 종목별 PIN을 MLE 추정한다.
- 그리드 탐색(243개 후보) → L-BFGS-B 최적화
- 멀티프로세싱으로 종목 단위 병렬화
- 체크포인트 저장/재개 지원
"""

import os
import math
import itertools
import multiprocessing

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from tqdm import tqdm

import config
import common


# =============================================================================
# PIN 추정 핵심 수학 함수
# =============================================================================

def _log_poisson_grid(k_row: np.ndarray, lam_col: np.ndarray) -> np.ndarray:
    """그리드 탐색 전용 log-포아송 PMF 행렬 계산. (G, N) 반환."""
    safe_lam = np.where(lam_col > 0, lam_col, 1e-300)
    lgk = gammaln(k_row + 1.0)
    return k_row * np.log(safe_lam) - safe_lam - lgk


def _grid_search(grid: np.ndarray, B: np.ndarray, S: np.ndarray) -> int:
    """243개 파라미터 후보에서 NLL 최소 인덱스를 반환한다."""
    a_g  = grid[:, 0:1]
    d_g  = grid[:, 1:2]
    mu_g = grid[:, 2:3]
    eb_g = grid[:, 3:4]
    es_g = grid[:, 4:5]

    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)

    lp_B_eb   = _log_poisson_grid(B_row, eb_g)
    lp_S_es   = _log_poisson_grid(S_row, es_g)
    lp_S_mues = _log_poisson_grid(S_row, mu_g + es_g)
    lp_B_mueb = _log_poisson_grid(B_row, mu_g + eb_g)

    l0 = lp_B_eb + lp_S_es
    l1 = lp_B_eb + lp_S_mues
    l2 = lp_B_mueb + lp_S_es

    w0 = np.clip(1.0 - a_g,       1e-300, None)
    w1 = np.clip(a_g * d_g,       1e-300, None)
    w2 = np.clip(a_g * (1 - d_g), 1e-300, None)

    log_terms = np.stack([
        np.log(w0) + l0,
        np.log(w1) + l1,
        np.log(w2) + l2,
    ], axis=0)
    log_lik_per_day = logsumexp(log_terms, axis=0)

    nll_scores = -np.sum(log_lik_per_day, axis=1)
    return int(np.argmin(nll_scores))


def _make_nll(B: np.ndarray, S: np.ndarray):
    """L-BFGS-B 최적화에 전달할 NLL 콜백 함수를 클로저로 생성한다."""
    B_f   = B.astype(np.float64)
    S_f   = S.astype(np.float64)
    lgk_B = gammaln(B_f + 1.0)
    lgk_S = gammaln(S_f + 1.0)

    def negative_log_likelihood(params):
        alpha, delta, mu, eb, es = params

        if not (0 <= alpha <= 1 and 0 <= delta <= 1
                and mu >= 0 and eb >= 0 and es >= 0):
            return np.inf

        safe_eb   = max(eb,      1e-300)
        safe_es   = max(es,      1e-300)
        safe_mues = max(mu + es, 1e-300)
        safe_mueb = max(mu + eb, 1e-300)

        log_pB_eb   = B_f * math.log(safe_eb)   - safe_eb   - lgk_B
        log_pS_es   = S_f * math.log(safe_es)   - safe_es   - lgk_S
        log_pS_mues = S_f * math.log(safe_mues) - safe_mues - lgk_S
        log_pB_mueb = B_f * math.log(safe_mueb) - safe_mueb - lgk_B

        l0 = log_pB_eb + log_pS_es
        l1 = log_pB_eb + log_pS_mues
        l2 = log_pB_mueb + log_pS_es

        w0 = max(1.0 - alpha,           1e-300)
        w1 = max(alpha * delta,         1e-300)
        w2 = max(alpha * (1 - delta),   1e-300)

        log_terms = np.stack([
            math.log(w0) + l0,
            math.log(w1) + l1,
            math.log(w2) + l2,
        ], axis=0)
        log_lik = logsumexp(log_terms, axis=0)

        return -np.sum(log_lik)

    return negative_log_likelihood


def estimate_pin_parameters(B_array: np.ndarray, S_array: np.ndarray,
                             grid_combinations: np.ndarray) -> dict:
    """60일 윈도우의 B, S로 EKOP(1996) MLE PIN 추정."""
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    best_idx           = _grid_search(grid_combinations, B, S)
    best_initial_guess = grid_combinations[best_idx]

    nll_fn = _make_nll(B, S)

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
    if denom < 1e-10:
        return {"converged": False}
    pin = (alpha * mu) / denom

    return {
        "alpha": alpha, "delta": delta, "mu": mu,
        "eb": eb, "es": es, "PIN": pin, "converged": True,
    }


# =============================================================================
# 멀티프로세싱 워커
# =============================================================================

GLOBAL_GRID     = None
GLOBAL_CALENDAR = None


def init_worker(grid: np.ndarray, calendar: pl.Series) -> None:
    global GLOBAL_GRID, GLOBAL_CALENDAR
    GLOBAL_GRID     = grid
    GLOBAL_CALENDAR = calendar


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """단일 종목에 대해 60일 슬라이딩 윈도우로 PIN을 계산한다."""
    sym, sym_df = data_tuple

    Bs, Ss, dates = common.align_symbol_to_calendar(sym_df, GLOBAL_CALENDAR)
    n = len(dates)

    window_size    = config.PIN_WINDOW_SIZE
    min_valid_days = config.PIN_MIN_VALID_DAYS

    if n < window_size:
        return []

    results = []
    for i in range(window_size - 1, n):
        window_B = Bs[i - (window_size - 1) : i + 1]
        window_S = Ss[i - (window_size - 1) : i + 1]

        valid_days = int(np.sum((window_B + window_S) > 0))
        if valid_days < min_valid_days:
            continue

        est = estimate_pin_parameters(window_B, window_S, GLOBAL_GRID)

        if est["converged"]:
            results.append({
                "Symbol": sym,
                "Date":   dates[i],
                "a":      est["alpha"],
                "d":      est["delta"],
                "u":      est["mu"],
                "eb":     est["eb"],
                "es":     est["es"],
                "PIN":    est["PIN"],
            })

    return results


# =============================================================================
# PIN 결과 컬럼 정의
# =============================================================================

PIN_COLUMNS = ["Symbol", "Date", "a", "d", "u", "eb", "es", "PIN"]
PIN_FINAL_COLUMNS = ["Symbol", "Date", "B", "S", "a", "d", "u", "eb", "es", "PIN"]


# =============================================================================
# 메인 PIN 계산 함수
# =============================================================================

def run(daily_bs_path: str, output_dir: str, run_id: str,
        year_filter: list[int] | None = None,
        checkpoint_n: int | None = None) -> pl.DataFrame:
    """
    all_daily_bs.parquet를 읽어 종목별 60일 롤링 PIN을 계산하고 결과를 반환한다.
    """
    checkpoint_n = checkpoint_n or config.CHECKPOINT_N

    session_dir = os.path.join(output_dir, "pin", "checkpoints", run_id)
    os.makedirs(os.path.join(output_dir, "pin"), exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] PIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일   : {daily_bs_path}")
    print(f"  RUN_ID      : {run_id}")
    print(f"  세션 폴더   : {session_dir}")

    # 1. 데이터 로드 및 연도 필터
    daily_bs = pl.read_parquet(daily_bs_path)

    if year_filter:
        daily_bs = daily_bs.filter(pl.col("Date").dt.year().is_in(year_filter))
        print(f"  연도 필터   : {year_filter}")

    print(f"  전체 행 수  : {daily_bs.height:,}")
    print(f"  종목 수     : {daily_bs['Symbol'].n_unique():,}")
    print(f"  날짜 범위   : {daily_bs['Date'].min()} ~ {daily_bs['Date'].max()}")

    # 2. 시장 공통 영업일 캘린더 추출
    market_calendar = common.build_market_calendar(daily_bs)
    print(f"\n  영업일 수   : {len(market_calendar):,}일 "
          f"({market_calendar[0]} ~ {market_calendar[-1]})")

    # 3. 그리드 행렬 생성 (3^5 = 243)
    print("\n[Grid 초기값 생성 중...]")
    grid_matrix = np.array(
        list(itertools.product(
            [0.1, 0.5, 0.9],   # alpha
            [0.1, 0.5, 0.9],   # delta
            [20, 200, 2000],    # mu
            [20, 200, 2000],    # epsilon_b
            [20, 200, 2000],    # epsilon_s
        )),
        dtype=np.float64,
    )
    print(f"  Grid 크기   : {grid_matrix.shape[0]} combinations")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {config.IMAP_CHUNKSIZE} 종목/청크")
    print(f"  윈도우      : {config.PIN_WINDOW_SIZE} 영업일")
    print(f"  최소 유효일 : {config.PIN_MIN_VALID_DAYS}일")

    # 4. 완료된 종목 로드 (재개 지원)
    done_symbols      = common.load_already_done_symbols(session_dir, "pin")
    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    # 5. 종목별 DataFrame 분리
    grouped_data = common.prepare_symbol_data(daily_bs, remaining_symbols)

    print(f"{'='*65}")
    print("PIN 추정 시작...")
    print(f"{'='*65}\n")

    # 체크포인트 카운터
    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("pin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)

    batch_estimates = []
    processed_count = 0

    # 6. 병렬 처리
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(grid_matrix, market_calendar),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_symbol, grouped_data,
                                chunksize=config.IMAP_CHUNKSIZE),
            total=len(grouped_data),
            desc="  Estimating",
        ):
            batch_estimates.extend(res)
            processed_count += 1

            if processed_count % checkpoint_n == 0:
                common.save_checkpoint(batch_estimates, next_checkpoint_idx,
                                       session_dir, "pin", PIN_COLUMNS)
                next_checkpoint_idx += 1
                batch_estimates = []

    if batch_estimates:
        common.save_checkpoint(batch_estimates, next_checkpoint_idx,
                               session_dir, "pin", PIN_COLUMNS)

    # 7. 체크포인트 병합 및 최종 Join
    print(f"\n{'='*65}")
    all_session_checkpoints = sorted(
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("pin_checkpoint_")
    )
    print(f"체크포인트 {len(all_session_checkpoints)}개 병합 중... (세션: {run_id})")

    if not all_session_checkpoints:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    pin_estimates = (
        pl.concat([pl.read_parquet(p) for p in all_session_checkpoints], how="vertical")
    )

    final_df = (
        daily_bs
        .join(pin_estimates, on=["Symbol", "Date"], how="left")
        .select(PIN_FINAL_COLUMNS)
        .sort(["Symbol", "Date"])
    )

    return final_df
