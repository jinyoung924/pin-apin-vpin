"""
=============================================================================
APIN (Adjusted PIN) 추정 모듈 — Duarte & Young 2009 모델
=============================================================================

6가지 시나리오, 10개 파라미터(α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s).
- 고정 그리드(1,024개) + Top-K 멀티 스타트
- 조건부 웜 스타트 + 주기적 리셋
- 멀티프로세싱으로 종목 단위 병렬화
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
# APIN 추정 핵심 수학 함수
# =============================================================================

def _log_poisson_grid(k_row: np.ndarray, lam_col: np.ndarray) -> np.ndarray:
    """그리드 탐색 전용 log-포아송 PMF 행렬 계산. (G, N) 반환."""
    safe_lam = np.where(lam_col > 0, lam_col, 1e-300)
    lgk = gammaln(k_row + 1.0)
    return k_row * np.log(safe_lam) - safe_lam - lgk


def _grid_search_topk(grid: np.ndarray, B: np.ndarray, S: np.ndarray,
                      k: int | None = None) -> np.ndarray:
    """그리드 후보에서 NLL 상위 K개 인덱스를 반환한다."""
    if k is None:
        k = config.APIN_TOP_K

    a_g  = grid[:, 0:1]
    d_g  = grid[:, 1:2]
    t1_g = grid[:, 2:3]
    t2_g = grid[:, 3:4]
    ub_g = grid[:, 4:5]
    us_g = grid[:, 5:6]
    eb_g = grid[:, 6:7]
    es_g = grid[:, 7:8]
    pb_g = grid[:, 8:9]
    ps_g = grid[:, 9:10]

    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)

    lp_B_eb        = _log_poisson_grid(B_row, eb_g)
    lp_B_eb_pb     = _log_poisson_grid(B_row, eb_g + pb_g)
    lp_B_ub_eb     = _log_poisson_grid(B_row, ub_g + eb_g)
    lp_B_ub_eb_pb  = _log_poisson_grid(B_row, ub_g + eb_g + pb_g)

    lp_S_es        = _log_poisson_grid(S_row, es_g)
    lp_S_es_ps     = _log_poisson_grid(S_row, es_g + ps_g)
    lp_S_us_es     = _log_poisson_grid(S_row, us_g + es_g)
    lp_S_us_es_ps  = _log_poisson_grid(S_row, us_g + es_g + ps_g)

    l0 = lp_B_eb       + lp_S_es
    l1 = lp_B_eb_pb    + lp_S_es_ps
    l2 = lp_B_eb       + lp_S_us_es
    l3 = lp_B_eb_pb    + lp_S_us_es_ps
    l4 = lp_B_ub_eb    + lp_S_es
    l5 = lp_B_ub_eb_pb + lp_S_es_ps

    w0 = np.clip((1 - a_g) * (1 - t1_g),       1e-300, None)
    w1 = np.clip((1 - a_g) * t1_g,              1e-300, None)
    w2 = np.clip(a_g * (1 - d_g) * (1 - t2_g), 1e-300, None)
    w3 = np.clip(a_g * (1 - d_g) * t2_g,        1e-300, None)
    w4 = np.clip(a_g * d_g * (1 - t2_g),        1e-300, None)
    w5 = np.clip(a_g * d_g * t2_g,              1e-300, None)

    log_terms = np.stack([
        np.log(w0) + l0, np.log(w1) + l1, np.log(w2) + l2,
        np.log(w3) + l3, np.log(w4) + l4, np.log(w5) + l5,
    ], axis=0)
    log_lik_per_day = logsumexp(log_terms, axis=0)

    nll_scores = -np.sum(log_lik_per_day, axis=1)

    actual_k     = min(k, len(nll_scores))
    top_k_unsort = np.argpartition(nll_scores, actual_k)[:actual_k]
    top_k_sorted = top_k_unsort[np.argsort(nll_scores[top_k_unsort])]

    return top_k_sorted


def _make_nll(B: np.ndarray, S: np.ndarray):
    """L-BFGS-B 최적화에 전달할 NLL 콜백 함수를 클로저로 생성한다."""
    B_f   = B.astype(np.float64)
    S_f   = S.astype(np.float64)
    lgk_B = gammaln(B_f + 1.0)
    lgk_S = gammaln(S_f + 1.0)

    def negative_log_likelihood(params):
        alpha, delta, t1, t2, ub, us, eb, es, pb, ps = params

        if not (0 <= alpha <= 1 and 0 <= delta <= 1
                and 0 <= t1 <= 1 and 0 <= t2 <= 1
                and ub >= 0 and us >= 0 and eb >= 0 and es >= 0
                and pb >= 0 and ps >= 0):
            return np.inf

        safe_eb       = max(eb,            1e-300)
        safe_es       = max(es,            1e-300)
        safe_eb_pb    = max(eb + pb,       1e-300)
        safe_es_ps    = max(es + ps,       1e-300)
        safe_ub_eb    = max(ub + eb,       1e-300)
        safe_us_es    = max(us + es,       1e-300)
        safe_ub_eb_pb = max(ub + eb + pb,  1e-300)
        safe_us_es_ps = max(us + es + ps,  1e-300)

        log_pB_eb       = B_f * math.log(safe_eb)       - safe_eb       - lgk_B
        log_pB_eb_pb    = B_f * math.log(safe_eb_pb)    - safe_eb_pb    - lgk_B
        log_pB_ub_eb    = B_f * math.log(safe_ub_eb)    - safe_ub_eb    - lgk_B
        log_pB_ub_eb_pb = B_f * math.log(safe_ub_eb_pb) - safe_ub_eb_pb - lgk_B

        log_pS_es       = S_f * math.log(safe_es)       - safe_es       - lgk_S
        log_pS_es_ps    = S_f * math.log(safe_es_ps)    - safe_es_ps    - lgk_S
        log_pS_us_es    = S_f * math.log(safe_us_es)    - safe_us_es    - lgk_S
        log_pS_us_es_ps = S_f * math.log(safe_us_es_ps) - safe_us_es_ps - lgk_S

        l0 = log_pB_eb       + log_pS_es
        l1 = log_pB_eb_pb    + log_pS_es_ps
        l2 = log_pB_eb       + log_pS_us_es
        l3 = log_pB_eb_pb    + log_pS_us_es_ps
        l4 = log_pB_ub_eb    + log_pS_es
        l5 = log_pB_ub_eb_pb + log_pS_es_ps

        w0 = max((1 - alpha) * (1 - t1),         1e-300)
        w1 = max((1 - alpha) * t1,                1e-300)
        w2 = max(alpha * (1 - delta) * (1 - t2), 1e-300)
        w3 = max(alpha * (1 - delta) * t2,        1e-300)
        w4 = max(alpha * delta * (1 - t2),        1e-300)
        w5 = max(alpha * delta * t2,              1e-300)

        log_terms = np.stack([
            math.log(w0) + l0, math.log(w1) + l1, math.log(w2) + l2,
            math.log(w3) + l3, math.log(w4) + l4, math.log(w5) + l5,
        ], axis=0)
        log_lik = logsumexp(log_terms, axis=0)

        return -np.sum(log_lik)

    return negative_log_likelihood


# =============================================================================
# 고정 그리드 생성
# =============================================================================

def generate_fixed_grid() -> np.ndarray:
    """2^10 = 1,024개의 초기값 조합을 고정 값으로 생성한다."""
    prob_candidates = [0.3, 0.7]
    rate_candidates = config.APIN_FIXED_GRID_RATE_CANDIDATES

    grid = np.array(
        list(itertools.product(
            prob_candidates,    # α
            prob_candidates,    # δ
            prob_candidates,    # θ₁
            prob_candidates,    # θ₂
            rate_candidates,    # μ_b
            rate_candidates,    # μ_s
            rate_candidates,    # ε_b
            rate_candidates,    # ε_s
            rate_candidates,    # Δ_b
            rate_candidates,    # Δ_s
        )),
        dtype=np.float64,
    )
    return grid


# =============================================================================
# APIN 추정 통합 함수 — Top-K 멀티 스타트 + 웜 스타트
# =============================================================================

_PARAM_BOUNDS = [
    (0, 1), (0, 1), (0, 1), (0, 1),
    (0, None), (0, None), (0, None), (0, None),
    (0, None), (0, None),
]


def _run_single_lbfgsb(nll_fn, x0: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """단일 초기값에서 L-BFGS-B 최적화 1회 수행."""
    try:
        result = minimize(nll_fn, x0=x0, bounds=_PARAM_BOUNDS, method="L-BFGS-B")
    except Exception:
        return False, np.inf, np.array([])

    if not result.success:
        return False, np.inf, np.array([])

    return True, float(result.fun), result.x


def estimate_apin_parameters_topk(
    B_array: np.ndarray,
    S_array: np.ndarray,
    top_k_x0s: list[np.ndarray],
) -> dict:
    """Top-K 초기값으로 L-BFGS-B를 독립 수행, NLL 최소 채택."""
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    nll_fn = _make_nll(B, S)

    best_nll    = np.inf
    best_params = None

    for x0 in top_k_x0s:
        converged, final_nll, params = _run_single_lbfgsb(nll_fn, x0)
        if converged and final_nll < best_nll:
            best_nll    = final_nll
            best_params = params

    if best_params is None:
        return {"converged": False}

    a, d, t1, t2, ub, us, eb, es, pb, ps = best_params

    informed_flow = a * (d * ub + (1 - d) * us)
    shock_flow    = (pb + ps) * (a * t2 + (1 - a) * t1)
    denom         = informed_flow + shock_flow + eb + es

    if denom < 1e-10:
        return {"converged": False}

    return {
        "a": a, "d": d, "t1": t1, "t2": t2,
        "ub": ub, "us": us, "eb": eb, "es": es, "pb": pb, "ps": ps,
        "APIN": informed_flow / denom,
        "PSOS": shock_flow / denom,
        "converged": True,
        "params": best_params,
        "nll": best_nll,
    }


# =============================================================================
# 멀티프로세싱 워커
# =============================================================================

GLOBAL_CALENDAR = None


def init_worker(calendar: pl.Series) -> None:
    global GLOBAL_CALENDAR
    GLOBAL_CALENDAR = calendar


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """단일 종목에 대해 60일 슬라이딩 윈도우로 APIN을 계산한다."""
    sym, sym_df = data_tuple

    Bs, Ss, dates = common.align_symbol_to_calendar(sym_df, GLOBAL_CALENDAR)
    n = len(dates)

    window_size        = config.APIN_WINDOW_SIZE
    min_valid_days     = config.APIN_MIN_VALID_DAYS
    warm_reset_interval = config.APIN_WARM_RESET_INTERVAL

    if n < window_size:
        return []

    results = []

    last_optimized_params = None
    warm_streak           = 0

    for i in range(window_size - 1, n):
        window_B = Bs[i - (window_size - 1) : i + 1]
        window_S = Ss[i - (window_size - 1) : i + 1]

        valid_days = int(np.sum((window_B + window_S) > 0))
        if valid_days < min_valid_days:
            last_optimized_params = None
            warm_streak           = 0
            continue

        use_warm_start = (
            last_optimized_params is not None
            and warm_streak < warm_reset_interval
        )

        if use_warm_start:
            top_k_x0s = [last_optimized_params]
        else:
            fixed_grid = generate_fixed_grid()
            top_k_indices = _grid_search_topk(fixed_grid, window_B, window_S,
                                              k=config.APIN_TOP_K)
            top_k_x0s = [fixed_grid[idx] for idx in top_k_indices]
            warm_streak = 0

        est = estimate_apin_parameters_topk(window_B, window_S, top_k_x0s)

        if est["converged"]:
            last_optimized_params = est["params"]
            warm_streak          += 1

            results.append({
                "Symbol": sym,
                "Date":   dates[i],
                "a":      est["a"],
                "d":      est["d"],
                "t1":     est["t1"],
                "t2":     est["t2"],
                "ub":     est["ub"],
                "us":     est["us"],
                "eb":     est["eb"],
                "es":     est["es"],
                "pb":     est["pb"],
                "ps":     est["ps"],
                "APIN":   est["APIN"],
                "PSOS":   est["PSOS"],
            })
        else:
            last_optimized_params = None
            warm_streak           = 0

    return results


# =============================================================================
# APIN 결과 컬럼 정의
# =============================================================================

APIN_COLUMNS = [
    "Symbol", "Date",
    "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
    "APIN", "PSOS",
]
APIN_FINAL_COLUMNS = [
    "Symbol", "Date", "B", "S",
    "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
    "APIN", "PSOS",
]


# =============================================================================
# 메인 APIN 계산 함수
# =============================================================================

def run(daily_bs_path: str, output_dir: str, run_id: str,
        year_filter: list[int] | None = None,
        checkpoint_n: int | None = None) -> pl.DataFrame:
    """
    all_daily_bs.parquet를 읽어 종목별 60일 롤링 APIN을 계산하고 결과를 반환한다.
    """
    checkpoint_n = checkpoint_n or config.CHECKPOINT_N

    session_dir = os.path.join(output_dir, "apin", "checkpoints", run_id)
    os.makedirs(os.path.join(output_dir, "apin"), exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] APIN 계산 시작 (고정 그리드 + Top-{config.APIN_TOP_K} + 웜 스타트)")
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

    print(f"\n[최적화 모드]")
    print(f"  그리드 크기          : 고정 2^10 = 1,024개")
    print(f"  멀티 스타트          : Top-{config.APIN_TOP_K}")
    print(f"  웜 스타트 리셋 간격  : {config.APIN_WARM_RESET_INTERVAL}회")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어             : {num_cores}개")
    print(f"  IPC 청크             : {config.IMAP_CHUNKSIZE} 종목/청크")
    print(f"  윈도우               : {config.APIN_WINDOW_SIZE} 영업일")
    print(f"  최소 유효일          : {config.APIN_MIN_VALID_DAYS}일")

    # 3. 완료된 종목 로드 (재개 지원)
    done_symbols      = common.load_already_done_symbols(session_dir, "apin")
    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    # 4. 종목별 DataFrame 분리
    grouped_data = common.prepare_symbol_data(daily_bs, remaining_symbols)

    print(f"{'='*65}")
    print("APIN 추정 시작...")
    print(f"{'='*65}\n")

    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("apin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)

    batch_estimates = []
    processed_count = 0

    # 5. 병렬 처리
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(market_calendar,),
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
                                       session_dir, "apin", APIN_COLUMNS)
                next_checkpoint_idx += 1
                batch_estimates = []

    if batch_estimates:
        common.save_checkpoint(batch_estimates, next_checkpoint_idx,
                               session_dir, "apin", APIN_COLUMNS)

    # 6. 체크포인트 병합 및 최종 Join
    print(f"\n{'='*65}")
    all_session_checkpoints = sorted(
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("apin_checkpoint_")
    )
    print(f"체크포인트 {len(all_session_checkpoints)}개 병합 중... (세션: {run_id})")

    if not all_session_checkpoints:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    apin_estimates = (
        pl.concat([pl.read_parquet(p) for p in all_session_checkpoints], how="vertical")
    )

    final_df = (
        daily_bs
        .join(apin_estimates, on=["Symbol", "Date"], how="left")
        .select(APIN_FINAL_COLUMNS)
        .sort(["Symbol", "Date"])
    )

    return final_df
