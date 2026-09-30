"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 일별 롤링 APIN 계산 — v04 고정 그리드
=============================================================================

■ v03(동적 그리드) 대비 변경 사항 요약

  [변경 1] 동적 그리드 → 고정 그리드
    - 기존(v03): 윈도우 B/S 분위수 + Multiplier 방어막으로 윈도우마다 후보값을
                 동적 생성하여 2^10 = 1,024개 조합을 만들었음.
    - 변경(v04): 동적 생성 로직을 완전히 제거하고,
                 μ(뮤)/ε(엡실론)/Δ(델타)의 초기값 후보를 [100, 1000]으로 고정.
                 확률 파라미터(α, δ, θ₁, θ₂)는 기존과 동일하게 [0.3, 0.7] 고정.
    - 결과: 그리드 크기는 동일하게 2^10 = 1,024개이나,
            매 윈도우마다 분위수를 계산하는 오버헤드가 제거됨.

  [변경 2 ~ 변경 3, 공통]
    - Top-3 멀티 스타트, 조건부 웜 스타트 + 주기적 리셋은 v03과 동일.

  ※ 그 외 Step 1 전처리, 영업일 캘린더, 체크포인트, 최종 Join 로직은 동일

■ 전체 흐름 요약
  [Step 1] 원시 틱 데이터 → 일별 매수/매도 건수(B/S) 집계
  [Step 2] 일별 B/S → 종목별 60일 롤링 APIN 추정 (Duarte & Young 2009 모델)

■ APIN 모델 (Duarte & Young 2009)
  6가지 시나리오, 10개 파라미터(α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s).

■ MLE 추정 방식 (v04 기준)
  [1단계-A] 고정 그리드 + Top-3 서치 (그리드 모드)
    [100, 1000] 고정 후보로 2^10 = 1,024개 생성 후
    NLL 상위 3개를 선택하여 각각 L-BFGS-B 최적화 → 최종 NLL 최소 채택

  [1단계-B] 웜 스타트 (웜 모드)
    이전 최적 파라미터 1개로 L-BFGS-B 1회만 수행

  [2단계] L-BFGS-B 최적화 (공통)
    파라미터 범위 제약: α,δ,θ₁,θ₂ ∈ [0,1], μ_b,μ_s,ε_b,ε_s,Δ_b,Δ_s ≥ 0

■ 설정값 안내
  BASE_DIR                 : 월별 .parquet 파일이 들어있는 루트 경로
  YEAR_FOLDERS             : 처리할 연도 폴더 (None이면 KOR_* 전체 자동 탐색)
  OUTPUT_DIR               : 출력 결과 저장 경로
  FORCE_REPROCESS_STEP1    : True면 all_daily_bs.parquet를 강제로 다시 생성
  CHECKPOINT_N             : 종목 N개마다 중간 저장 (기본 100)
  IMAP_CHUNKSIZE           : 워커에 한 번에 넘기는 종목 수 (기본 10)
  WINDOW_SIZE              : 롤링 윈도우 크기 (영업일 기준, 기본 60)
  MIN_VALID_DAYS           : 윈도우 내 유효 거래일 최소 개수 (기본 30)
  TOP_K                    : 그리드 서치에서 선정할 초기값 개수 (기본 3)
  WARM_RESET_INTERVAL      : 연속 웜 스타트 N회마다 강제 리셋 (기본 20)
  FIXED_GRID_RATE_CANDIDATES: μ/ε/Δ 초기값 후보 (기본 [100, 1000])
  RESUME_RUN_ID            : 중단된 세션을 재개할 때 이전 RUN_ID

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
# 전역 설정
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = None
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

# Step 1 설정
FORCE_REPROCESS_STEP1 = False

# Step 2 설정
CHECKPOINT_N          = 100
IMAP_CHUNKSIZE        = 10

# 중단된 세션 재개 설정
RESUME_RUN_ID         = None

# 60일 롤링 윈도우 품질 기준
WINDOW_SIZE    = 60
MIN_VALID_DAYS = 30

# ── v03 최적화 전용 설정 ─────────────────────────────────────────────────────

# Top-K 멀티 스타트: 그리드 서치에서 NLL 상위 K개를 선정하여 각각 L-BFGS-B 수행
TOP_K = 3

# 웜 스타트 주기적 리셋 간격
# 연속으로 웜 스타트 성공한 횟수가 이 값에 도달하면 강제로 그리드+Top-K 수행
WARM_RESET_INTERVAL = 20

# 고정 그리드 초기값: μ(뮤), ε(엡실론), Δ(델타) 공통 후보값
# 한국 주식 시장 틱 데이터 스케일에서 100은 소형주, 1000은 중형주 수준에 해당하며
# 넓은 탐색 범위를 제공한다.
FIXED_GRID_RATE_CANDIDATES = [100.0, 1000.0]


# =============================================================================
# [Step 1] 전처리: 틱 데이터 → 일별 B/S 집계
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일(월별 틱 데이터)을 읽어 일별 매수/매도 건수로 집계한다.

    원본 컬럼 LR의 값: 1 → 매수(Buy), -1 → 매도(Sell)
    반환 스키마: Symbol (Utf8), Date (pl.Date), B (UInt32), S (UInt32)
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
    """base_dir 아래의 KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다."""
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
    모든 월별 parquet를 순회해 일별 B/S를 집계하고,
    결과를 all_daily_bs.parquet 한 파일로 저장한다.
    이미 파일이 존재하고 FORCE_REPROCESS_STEP1=False이면 스킵.
    """
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
            (part for part in path.replace("\\", "/").split("/")
             if part.startswith("KOR_")), ""
        )
        if i == 1 or year_tag != next(
            (p for p in parquet_files[i - 2].replace("\\", "/").split("/")
             if p.startswith("KOR_")), ""
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
        .group_by(["Symbol", "Date"])
        .agg([pl.col("B").sum(), pl.col("S").sum()])
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
# [Step 2-전처리] 영업일 캘린더 구축 및 종목별 B/S 정렬
# =============================================================================

def build_market_calendar(daily_bs: pl.DataFrame) -> pl.Series:
    """전체 데이터에서 '시장 공통 영업일 캘린더'를 추출한다."""
    return (
        daily_bs.select("Date").unique().sort("Date").get_column("Date")
    )


def align_symbol_to_calendar(
    sym_df: pl.DataFrame,
    calendar: pl.Series,
) -> tuple[np.ndarray, np.ndarray, list]:
    """단일 종목의 B/S 시계열을 영업일 캘린더에 맞춰 정렬한다."""
    aligned = (
        pl.DataFrame({"Date": calendar})
        .join(sym_df.select(["Date", "B", "S"]), on="Date", how="left")
        .with_columns([
            pl.col("B").fill_null(0).cast(pl.Int32),
            pl.col("S").fill_null(0).cast(pl.Int32),
        ])
    )
    return aligned["B"].to_numpy(), aligned["S"].to_numpy(), aligned["Date"].to_list()


# =============================================================================
# [Step 2] APIN 추정 핵심 수학 함수
# =============================================================================

def _log_poisson_grid(k_row: np.ndarray, lam_col: np.ndarray) -> np.ndarray:
    """
    그리드 탐색 전용 log-포아송 PMF 행렬 계산.
    log P(k|λ) = k·ln(λ) - λ - ln(k!)
    k_row: (1, N),  lam_col: (G, 1)  →  결과: (G, N)
    """
    safe_lam = np.where(lam_col > 0, lam_col, 1e-300)
    lgk = gammaln(k_row + 1.0)
    return k_row * np.log(safe_lam) - safe_lam - lgk


def _grid_search_topk(grid: np.ndarray, B: np.ndarray, S: np.ndarray,
                      k: int = TOP_K) -> np.ndarray:
    """
    그리드 후보 전체에 대해 NLL을 동시에 계산하고,
    NLL이 가장 낮은 상위 K개 후보의 인덱스를 반환한다.

    ■ v03 변경사항 (기존 _grid_search 대비)
      - 반환값: int 1개 → np.ndarray shape (K,)
      - np.argpartition으로 O(G) 시간에 Top-K 선정 후 정렬
      - K개 출발점으로 L-BFGS-B를 독립 실행하여 지역 최적해 회피

    Args:
        grid : shape (G, 10) — [α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s]
        B    : shape (N,) — 윈도우 매수 건수
        S    : shape (N,) — 윈도우 매도 건수
        k    : 반환할 상위 후보 개수 (기본 TOP_K=3)
    Returns:
        NLL이 가장 낮은 K개 그리드 행의 인덱스 배열 (NLL 오름차순 정렬)
    """
    # 각 파라미터를 열벡터(G, 1)로 분리 → 브로드캐스팅 준비
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

    B_row = B.reshape(1, -1)   # (1, N)
    S_row = S.reshape(1, -1)

    # 시나리오별 log-PMF 행렬: 각각 shape (G, N)
    lp_B_eb        = _log_poisson_grid(B_row, eb_g)
    lp_B_eb_pb     = _log_poisson_grid(B_row, eb_g + pb_g)
    lp_B_ub_eb     = _log_poisson_grid(B_row, ub_g + eb_g)
    lp_B_ub_eb_pb  = _log_poisson_grid(B_row, ub_g + eb_g + pb_g)

    lp_S_es        = _log_poisson_grid(S_row, es_g)
    lp_S_es_ps     = _log_poisson_grid(S_row, es_g + ps_g)
    lp_S_us_es     = _log_poisson_grid(S_row, us_g + es_g)
    lp_S_us_es_ps  = _log_poisson_grid(S_row, us_g + es_g + ps_g)

    # 6가지 시나리오 하루 log-우도 (B·S 독립이므로 log-PMF 합산)
    l0 = lp_B_eb       + lp_S_es           # 1) 무정보 + 무충격
    l1 = lp_B_eb_pb    + lp_S_es_ps        # 2) 무정보 + 충격
    l2 = lp_B_eb       + lp_S_us_es        # 3) 악재   + 무충격
    l3 = lp_B_eb_pb    + lp_S_us_es_ps     # 4) 악재   + 충격
    l4 = lp_B_ub_eb    + lp_S_es           # 5) 호재   + 무충격
    l5 = lp_B_ub_eb_pb + lp_S_es_ps        # 6) 호재   + 충격

    # 가중치 (0 방어 후 ln 변환)
    w0 = np.clip((1 - a_g) * (1 - t1_g),       1e-300, None)
    w1 = np.clip((1 - a_g) * t1_g,              1e-300, None)
    w2 = np.clip(a_g * (1 - d_g) * (1 - t2_g), 1e-300, None)
    w3 = np.clip(a_g * (1 - d_g) * t2_g,        1e-300, None)
    w4 = np.clip(a_g * d_g * (1 - t2_g),        1e-300, None)
    w5 = np.clip(a_g * d_g * t2_g,              1e-300, None)

    # logsumexp 트릭: (6, G, N) → axis=0 → (G, N)
    log_terms = np.stack([
        np.log(w0) + l0, np.log(w1) + l1, np.log(w2) + l2,
        np.log(w3) + l3, np.log(w4) + l4, np.log(w5) + l5,
    ], axis=0)
    log_lik_per_day = logsumexp(log_terms, axis=0)   # (G, N)

    # 후보별 NLL: shape (G,)
    nll_scores = -np.sum(log_lik_per_day, axis=1)

    # Top-K 선정: argpartition은 O(G)으로 상위 K개를 찾고, 이를 NLL 오름차순 정렬
    # (그리드 크기 G=1,024이면 argsort와 차이가 미미하지만, 확장성을 위해 partition 사용)
    actual_k     = min(k, len(nll_scores))
    top_k_unsort = np.argpartition(nll_scores, actual_k)[:actual_k]
    top_k_sorted = top_k_unsort[np.argsort(nll_scores[top_k_unsort])]

    return top_k_sorted


def _make_nll(B: np.ndarray, S: np.ndarray):
    """
    L-BFGS-B 최적화에 전달할 NLL 콜백 함수를 클로저로 생성한다.

    ln(B!)와 ln(S!)은 파라미터 변화와 무관한 상수이므로
    클로저 외부에서 한 번만 계산하여 반복 호출 시 재계산을 방지한다.
    """
    B_f   = B.astype(np.float64)
    S_f   = S.astype(np.float64)
    lgk_B = gammaln(B_f + 1.0)
    lgk_S = gammaln(S_f + 1.0)

    def negative_log_likelihood(params):
        alpha, delta, t1, t2, ub, us, eb, es, pb, ps = params

        # 파라미터 범위 이중 방어
        if not (0 <= alpha <= 1 and 0 <= delta <= 1
                and 0 <= t1 <= 1 and 0 <= t2 <= 1
                and ub >= 0 and us >= 0 and eb >= 0 and es >= 0
                and pb >= 0 and ps >= 0):
            return np.inf

        # ln(0) 방어
        safe_eb       = max(eb,            1e-300)
        safe_es       = max(es,            1e-300)
        safe_eb_pb    = max(eb + pb,       1e-300)
        safe_es_ps    = max(es + ps,       1e-300)
        safe_ub_eb    = max(ub + eb,       1e-300)
        safe_us_es    = max(us + es,       1e-300)
        safe_ub_eb_pb = max(ub + eb + pb,  1e-300)
        safe_us_es_ps = max(us + es + ps,  1e-300)

        # 시나리오별 log-PMF
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
# [Step 2] 고정 그리드 생성 — μ/ε/Δ 초기값 [100, 1000] 고정
# =============================================================================

def generate_fixed_grid() -> np.ndarray:
    """
    2^10 = 1,024개의 초기값 조합을 고정 값으로 생성한다.

    ■ 설계 원리
      확률 파라미터 (α, δ, θ₁, θ₂):
        [0.3, 0.7] 고정. L-BFGS-B가 경계 근처까지 자유롭게 탐색하므로
        출발점은 중간 대역이 효율적.

      뮤(μ_b, μ_s), 엡실론(ε_b, ε_s), 델타(Δ_b, Δ_s):
        [100, 1000] 고정.
        100은 소형주 평균, 1000은 중형주 평균에 해당하는 스케일로
        넓은 탐색 범위를 제공한다.

    Returns:
        shape (1024, 10) — [α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s]
    """
    prob_candidates = [0.3, 0.7]
    rate_candidates = FIXED_GRID_RATE_CANDIDATES   # [100.0, 1000.0]

    grid = np.array(
        list(itertools.product(
            prob_candidates,    # α   (2)
            prob_candidates,    # δ   (2)
            prob_candidates,    # θ₁  (2)
            prob_candidates,    # θ₂  (2)
            rate_candidates,    # μ_b (2)
            rate_candidates,    # μ_s (2)
            rate_candidates,    # ε_b (2)
            rate_candidates,    # ε_s (2)
            rate_candidates,    # Δ_b (2)
            rate_candidates,    # Δ_s (2)
        )),
        dtype=np.float64,
    )

    return grid   # shape (1024, 10)


# =============================================================================
# [Step 2] APIN 추정 통합 함수 — v03: Top-K 멀티 스타트 + 웜 스타트 x0 지원
# =============================================================================

# L-BFGS-B 파라미터 범위 제약 (10개 파라미터에 대한 bounds)
# 한 번만 정의하여 반복 생성 비용 제거
_PARAM_BOUNDS = [
    (0, 1), (0, 1), (0, 1), (0, 1),                     # α, δ, θ₁, θ₂
    (0, None), (0, None), (0, None), (0, None),          # μ_b, μ_s, ε_b, ε_s
    (0, None), (0, None),                                # Δ_b, Δ_s
]


def _run_single_lbfgsb(nll_fn, x0: np.ndarray) -> tuple[bool, float, np.ndarray]:
    """
    단일 초기값 x0에서 L-BFGS-B 최적화를 1회 수행하고 결과를 반환한다.

    estimate_apin_parameters에서 Top-K 멀티 스타트와 웜 스타트 모두에서
    공통으로 사용하기 위해 분리한 헬퍼 함수.

    Returns:
        (converged, final_nll, params)
        - converged : 수렴 성공 여부
        - final_nll : 최적화 후 NLL 값 (실패 시 np.inf)
        - params    : 최적화된 10개 파라미터 배열 (실패 시 빈 배열)
    """
    try:
        result = minimize(
            nll_fn,
            x0=x0,
            bounds=_PARAM_BOUNDS,
            method="L-BFGS-B",
        )
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
    """
    Top-K 초기값 각각에 대해 L-BFGS-B를 독립 수행하고,
    수렴한 결과 중 최종 NLL이 가장 낮은 것을 채택한다.

    ■ v03 핵심 변경사항 (기존 estimate_apin_parameters 대비)
      - 입력: grid_combinations 1개 → top_k_x0s 리스트 (길이 K 또는 1)
      - 웜 스타트 시: [이전_파라미터] 1개짜리 리스트 → L-BFGS-B 1회만 수행
      - 그리드 모드 시: Top-3 초기값 리스트 → L-BFGS-B 3회 독립 수행
      - NLL 비교 후 최종 채택 → 지역 최적해 회피

    Args:
        B_array   : shape (N,) — 윈도우 매수 건수
        S_array   : shape (N,) — 윈도우 매도 건수
        top_k_x0s : 초기값 리스트 (각 원소: shape (10,) np.ndarray)

    Returns:
        수렴 성공: {"a", ..., "APIN", "PSOS", "converged": True,
                    "params": np.array(10,), "nll": float}
        전부 실패: {"converged": False}
    """
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    # NLL 콜백 함수를 1회만 생성 (lgk_B, lgk_S 사전 계산은 1번만)
    nll_fn = _make_nll(B, S)

    # ── Top-K 초기값 각각에 대해 L-BFGS-B 독립 수행 ──────────────────────────
    best_nll    = np.inf
    best_params = None

    for x0 in top_k_x0s:
        converged, final_nll, params = _run_single_lbfgsb(nll_fn, x0)
        if converged and final_nll < best_nll:
            best_nll    = final_nll
            best_params = params

    # ── 모든 시도가 실패한 경우 ──────────────────────────────────────────────
    if best_params is None:
        return {"converged": False}

    # ── APIN / PSOS 계산 (Duarte & Young 2009) ──────────────────────────────
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
        "params": best_params,    # 웜 스타트를 위해 보존
        "nll": best_nll,          # 디버깅 및 품질 모니터링용
    }


# =============================================================================
# [Step 2] 멀티프로세싱 워커 함수 — v03: GLOBAL_GRID 제거, 캘린더만 전달
# =============================================================================

GLOBAL_CALENDAR = None   # 시장 공통 영업일 Series (워커 간 공유)


def init_worker(calendar: pl.Series) -> None:
    """
    Pool 생성 시 각 워커에서 한 번만 실행되는 초기화 함수.
    v03: grid 인자 제거 (동적 그리드로 대체되어 전역 공유 불필요).
    """
    global GLOBAL_CALENDAR
    GLOBAL_CALENDAR = calendar


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """
    단일 종목의 거래 데이터를 받아 60일 슬라이딩 윈도우로 APIN을 계산한다.

    ■ v03 최적화 전략 — 3단계 하이브리드

      각 윈도우에서 아래 3가지 모드 중 하나가 자동 선택된다.

      [모드 A] 고정 그리드 + Top-3 멀티 스타트 (= 풀 서치)
        적용 조건:
          ① 종목의 첫 번째 유효 윈도우 (이전 파라미터 없음)
          ② 이전 윈도우에서 수렴 실패 또는 유효일 부족 스킵 직후
          ③ 연속 웜 스타트가 WARM_RESET_INTERVAL(20)회에 도달 (주기적 리셋)
        동작:
          generate_fixed_grid로 1,024개 후보 생성 →
          _grid_search_topk로 NLL Top-3 선정 →
          estimate_apin_parameters_topk에서 L-BFGS-B 3회 독립 수행 →
          최종 NLL 최소 채택
        비용: 그리드 NLL 계산 1회(벡터화) + L-BFGS-B 3회

      [모드 B] 조건부 웜 스타트 (= 고속 추적)
        적용 조건:
          이전 윈도우 수렴 성공 + 연속 성공 < WARM_RESET_INTERVAL
        동작:
          이전 최적 파라미터 1개를 x0로 바로 투입 →
          L-BFGS-B 1회만 수행
        비용: L-BFGS-B 1회 (그리드 연산 0)

      [모드 C] 에러 리셋
        적용 조건:
          유효일 부족(valid_days < MIN_VALID_DAYS) 또는 수렴 실패
        동작:
          last_optimized_params = None, warm_streak = 0으로 초기화
          → 다음 유효 윈도우가 반드시 모드 A로 진입

    ■ 속도 시뮬레이션 (종목당 윈도우 ~1,200개 가정)
      기존: 1,200 × (59,049 그리드 + L-BFGS-B 1회) = 1,200 풀 서치
      v03 : ~60 풀 서치(첫 윈도우 + 20회마다 리셋) + ~1,140 웜 스타트
            풀 서치 비용도 59,049 → 1,024 + Top-3으로 대폭 감소
            → 전체 약 10~20배 속도 향상 기대
    """
    sym, sym_df = data_tuple

    Bs, Ss, dates = align_symbol_to_calendar(sym_df, GLOBAL_CALENDAR)
    n = len(dates)

    if n < WINDOW_SIZE:
        return []

    results = []

    # ── 웜 스타트 상태 변수 ──────────────────────────────────────────────────
    last_optimized_params = None   # 이전 윈도우의 최적화 성공 파라미터 (shape (10,))
    warm_streak           = 0     # 연속 웜 스타트 성공 횟수

    for i in range(WINDOW_SIZE - 1, n):
        window_B = Bs[i - (WINDOW_SIZE - 1) : i + 1]
        window_S = Ss[i - (WINDOW_SIZE - 1) : i + 1]

        # ── 유효 거래일 검사 ─────────────────────────────────────────────────
        valid_days = int(np.sum((window_B + window_S) > 0))
        if valid_days < MIN_VALID_DAYS:
            # [모드 C] 유효일 부족 스킵 → 웜 스타트 즉시 초기화
            # 스킵된 구간 이후 데이터 특성이 달라질 수 있으므로
            # 이전 파라미터를 그대로 사용하면 부적절한 초기값이 될 위험
            last_optimized_params = None
            warm_streak           = 0
            continue

        # ── 초기값(x0) 결정: 웜 스타트 vs 풀 서치 ──────────────────────────
        use_warm_start = (
            last_optimized_params is not None
            and warm_streak < WARM_RESET_INTERVAL
        )

        if use_warm_start:
            # ▶ [모드 B] 웜 스타트: 그리드 + Top-K 완전 생략
            # 인접 윈도우는 59/60일이 겹치므로 최적 파라미터가 거의 유사하여
            # 이전 결과를 x0로 사용하면 L-BFGS-B가 1~2번 반복만에 수렴
            top_k_x0s = [last_optimized_params]
        else:
            # ▶ [모드 A] 풀 서치: 고정 그리드 생성 + Top-3 멀티 스타트
            # (주기적 리셋이었다면 streak 카운터도 0으로 초기화)
            fixed_grid = generate_fixed_grid()
            top_k_indices = _grid_search_topk(fixed_grid, window_B, window_S,
                                              k=TOP_K)
            top_k_x0s = [fixed_grid[idx] for idx in top_k_indices]
            warm_streak = 0

        # ── L-BFGS-B 최적화 (모드에 따라 1회 또는 K회) ──────────────────────
        est = estimate_apin_parameters_topk(window_B, window_S, top_k_x0s)

        if est["converged"]:
            # 성공: 최적 파라미터를 보존하여 다음 윈도우의 웜 스타트에 활용
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
            # [모드 C] 수렴 실패 → 웜 스타트 즉시 초기화
            # 실패한 파라미터를 다음 윈도우에 넘기면 연쇄 실패 위험
            last_optimized_params = None
            warm_streak           = 0

    return results


# =============================================================================
# [Step 2] 결과 저장 및 재개 지원 함수
# =============================================================================

def build_results_df(estimates: list[dict]) -> pl.DataFrame:
    """process_single_symbol이 반환한 dict 리스트를 Polars DataFrame으로 변환한다."""
    if not estimates:
        return pl.DataFrame()
    return (
        pl.DataFrame(estimates)
        .with_columns(pl.col("Date").cast(pl.Date))
        .select([
            "Symbol", "Date",
            "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
            "APIN", "PSOS",
        ])
    )


def save_checkpoint(estimates: list[dict], checkpoint_idx: int, session_dir: str) -> str:
    """현재까지 누적된 추정 결과를 세션 전용 폴더에 체크포인트 파일로 저장한다."""
    os.makedirs(session_dir, exist_ok=True)
    df   = build_results_df(estimates)
    path = os.path.join(session_dir, f"apin_checkpoint_{checkpoint_idx:04d}.parquet")
    df.write_parquet(path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] {df.height:,} records → {os.path.basename(path)}")
    return path


def load_already_done_symbols(session_dir: str) -> set[str]:
    """중단 후 재실행 시 세션 폴더의 체크포인트에서 이미 완료된 종목 목록을 복원한다."""
    if not os.path.exists(session_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(session_dir) if f.startswith("apin_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols = set()
    for fname in checkpoint_files:
        path = os.path.join(session_dir, fname)
        df   = pl.read_parquet(path, columns=["Symbol"])
        done_symbols.update(df["Symbol"].unique().to_list())

    print(f"[재개 모드] 기존 체크포인트 {len(checkpoint_files)}개 발견 → "
          f"완료된 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


# =============================================================================
# [Step 2] 메인 APIN 계산 함수 — v03: 전역 그리드 제거
# =============================================================================

def run_apin_calculation(daily_bs_path: str, output_dir: str,
                         run_id: str,
                         year_filter: list[int] | None = None,
                         checkpoint_n: int = 100) -> pl.DataFrame:
    """
    all_daily_bs.parquet를 읽어 종목별 60일 롤링 APIN을 계산하고 결과를 반환한다.

    v04 변경사항:
      - 동적 그리드 제거 → generate_fixed_grid([100, 1000] 고정) 사용
      - init_worker에 calendar만 전달 (grid 불필요)
      - 고정 그리드 + Top-K + 웜 스타트는 process_single_symbol 내부에서 처리
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    session_dir      = os.path.join(intermediate_dir, f"apin_session_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] APIN 계산 시작 (고정 그리드 [100,1000] + Top-{TOP_K} + 웜 스타트)")
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
    market_calendar = build_market_calendar(daily_bs)
    print(f"\n  영업일 수   : {len(market_calendar):,}일 "
          f"({market_calendar[0]} ~ {market_calendar[-1]})")

    # v03: 고정 그리드 사용 → μ/ε/Δ 초기값 [100, 1000] 고정
    print(f"\n[최적화 모드]")
    print(f"  그리드 크기          : 고정 2^10 = 1,024개  (μ/ε/Δ 초기값 100, 1000)")
    print(f"  멀티 스타트          : Top-{TOP_K}")
    print(f"  웜 스타트 리셋 간격  : {WARM_RESET_INTERVAL}회")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어             : {num_cores}개")
    print(f"  IPC 청크             : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  윈도우               : {WINDOW_SIZE} 영업일")
    print(f"  최소 유효일          : {MIN_VALID_DAYS}일")

    # 3. 세션 폴더에서 완료된 종목 로드 (재개 지원)
    done_symbols      = load_already_done_symbols(session_dir)
    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    # 4. 종목별 DataFrame 분리
    symbol_partitions = daily_bs.partition_by("Symbol", maintain_order=False)
    symbol_dfs = {}
    for part in symbol_partitions:
        sym = part["Symbol"][0]
        if sym in remaining_symbols:
            symbol_dfs[sym] = part.sort("Date")

    grouped_data = [(sym, symbol_dfs[sym]) for sym in remaining_symbols if sym in symbol_dfs]

    print(f"{'='*65}")
    print("APIN 추정 시작...")
    print(f"{'='*65}\n")

    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("apin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)

    batch_estimates = []
    processed_count = 0

    # 5. 병렬 처리 — init_worker에 calendar만 전달 (grid 제거)
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(market_calendar,),
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
                save_checkpoint(batch_estimates, next_checkpoint_idx, session_dir)
                next_checkpoint_idx += 1
                batch_estimates = []

    if batch_estimates:
        save_checkpoint(batch_estimates, next_checkpoint_idx, session_dir)

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
        .select([
            "Symbol", "Date", "B", "S",
            "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
            "APIN", "PSOS",
        ])
        .sort(["Symbol", "Date"])
    )

    return final_df


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    run_id = RESUME_RUN_ID if RESUME_RUN_ID else datetime.now().strftime("%Y%m%d_%H%M")
    start  = datetime.now()

    if RESUME_RUN_ID:
        print(f"\n[재개 모드] RUN_ID = {run_id}")
    else:
        print(f"\n[새 실행]   RUN_ID = {run_id}")

    # Step 1: 틱 데이터 전처리
    daily_bs_path = run_preprocessing(
        base_dir=BASE_DIR, year_folders=YEAR_FOLDERS, output_dir=OUTPUT_DIR
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] 전처리 결과 파일이 없어 APIN 계산을 종료합니다.")
        exit(1)

    if YEAR_FOLDERS:
        year_filter = [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
    else:
        year_filter = None

    # Step 2: APIN 추정
    result = run_apin_calculation(
        daily_bs_path=daily_bs_path,
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    if not result.is_empty():
        year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename = f"apin_daily_rolling_{year_tag}_{run_id}"

        parquet_path = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result.write_parquet(parquet_path, compression="zstd")
        print(f"\n[저장 완료] {parquet_path}")

        csv_path = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result.head(1000).write_csv(csv_path)
        print(f"[샘플 저장] {csv_path}")

        print("\n[미리보기]")
        print(result.head(20))

        print("\n[통계]")
        print(f"  전체 레코드        : {result.height:,}")
        apin_ok = result.filter(pl.col("APIN").is_not_null()).height
        print(f"  APIN 추정 성공     : {apin_ok:,}")
        print(f"  APIN 추정 실패/없음: {result.height - apin_ok:,}")

        sym_stats = (
            result
            .group_by("Symbol")
            .agg([
                pl.len().alias("total_days"),
                pl.col("APIN").is_not_null().sum().alias("apin_days"),
            ])
            .with_columns(
                (pl.col("apin_days") / pl.col("total_days") * 100).alias("coverage_pct")
            )
        )
        print("\n[종목별 커버리지 샘플]")
        print(sym_stats.head(10))

    else:
        print("\n[Warning] 결과 DataFrame이 비어 있습니다.")

    elapsed = datetime.now() - start
    print(f"\n총 소요 시간: {elapsed}")