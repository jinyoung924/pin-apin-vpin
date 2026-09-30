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

예외처리 참조: R 패키지 PINstimation (Zaghouani, 2024)
  - vpin() 함수의 입력 검증, 수치 안정성, 경계 조건 처리 로직을 Python으로 이식
  - estimate.vpin 클래스의 success/errorMessage 패턴 반영
  - Ke & Lin (2017) 개선 BVC 모델의 유효성 검증 반영

실행: python VPIN_순수파이썬.py
=============================================================================
"""

import os
import re
import gc
import glob
import sys
import multiprocessing
import warnings
import logging
from datetime import datetime
from typing import List, Optional, Tuple, Dict

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

# ── PINstimation 호환 상수 ──────────────────────────────────────────────────
MAX_BUCKET_COUNT       = 10_000_000   # 단일 종목 최대 버킷 수 (무한 루프 방지)
MIN_TRADING_DAYS       = 2            # 최소 거래일 수
MIN_NONZERO_DP_RATIO   = 0.01         # 가격 변동이 있는 봉의 최소 비율 (1%)
VPIN_UPPER_BOUND       = 1.0          # VPIN 이론 상한
VPIN_LOWER_BOUND       = 0.0          # VPIN 이론 하한

# ── 비유동 종목 필터 상수 ────────────────────────────────────────────────────
MIN_MEDIAN_DAILY_BUCKETS   = 2        # 일별 추정 버킷 수의 중앙값 최소 기준
MAX_ZERO_BUCKET_DAY_RATIO  = 0.5      # 버킷 0개 생성일 비율 상한 (50%)


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

# ── 로깅 설정 ──────────────────────────────────────────────────────────────
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")


def _setup_logger() -> logging.Logger:
    """
    VPIN 처리 전용 로거를 설정한다.

    PINstimation 참조:
        PINstimation은 show() 메서드를 통해 추정 결과와 오류를 구조화하여 출력한다.
        이를 참고하여, 파이썬에서는 logging 모듈로 콘솔+파일 이중 로깅을 구현한다.
        스킵된 종목, 경고, 오류를 모두 기록하여 사후 진단이 가능하도록 한다.
    """
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("vpin")
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(
        os.path.join(LOG_DIR, f"vpin_{datetime.now():%Y%m%d_%H%M%S}.log"),
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"
    ))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("  [%(levelname)s] %(message)s"))
    logger.addHandler(ch)
    return logger


logger = _setup_logger()


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


def build_business_day_calendar(symbol_out_dir: str) -> Dict[int, int]:
    """
    전체 종목 데이터를 스캔하여 글로벌 영업일 캘린더를 생성한다.

    목적:
        VPIN의 ADV(Average Daily Volume) 계산과 비유동 종목 필터에서
        "하루"의 기준을 종목 개별이 아닌 시장 전체로 통일한다.

        영업일 정의:
            전체 종목 중 하나라도 거래(Volume > 0인 1분봉)가 발생한 날.
            어떤 종목도 거래하지 않은 날은 비영업일(주말·공휴일)이다.

        왜 글로벌 캘린더가 필요한가:
            종목별로 거래일이 다르면 ADV의 분모(n_days)가 종목마다 달라진다.
            예) 연간 영업일 250일인데, 종목 A는 100일만 거래 발생:
                - 종목별 기준: ADV = total_vol / 100  (과대)
                - 글로벌 기준: ADV = total_vol / 250  (정확)

            종목별 기준으로 ADV를 과대 산출하면 VBS도 커져서,
            거래 발생일에 생성되는 버킷 수가 줄어든다. 이는 비유동 종목의
            VPIN 롤링 윈도우가 과도하게 긴 기간을 커버하게 만든다.

            글로벌 기준을 사용하면 모든 종목이 동일한 "하루" 정의를 공유하여,
            종목 간 VPIN의 시간 해상도가 일관된다.

    구현:
        bars_by_symbol/ 폴더의 모든 parquet 파일을 lazy scan하여
        전체 유니크 날짜를 수집한 뒤, 연도별 영업일 수를 집계한다.

    Args:
        symbol_out_dir: bars_by_symbol/ 폴더 경로

    Returns:
        {연도(int): 영업일 수(int)} 딕셔너리
        예) {2019: 251, 2020: 253, 2021: 252}
    """
    sym_files = sorted(glob.glob(os.path.join(symbol_out_dir, "*.parquet")))
    if not sym_files:
        logger.error(
            f"[캘린더] bars_by_symbol 폴더에 파일이 없습니다: {symbol_out_dir}"
        )
        return {}

    print("  영업일 캘린더 생성 중...", end=" ", flush=True)

    # 전체 종목의 모든 유니크 날짜 수집
    all_dates = (
        pl.scan_parquet(sym_files)
        .select(pl.col("Datetime").dt.date().alias("date"))
        .unique()
        .collect()
        .get_column("date")
    )

    # 연도별 영업일 수 집계
    calendar: Dict[int, int] = {}
    for d in all_dates:
        y = d.year
        calendar[y] = calendar.get(y, 0) + 1

    for year in sorted(calendar):
        print(f"{year}={calendar[year]}일", end="  ")
    print()

    logger.info(
        f"[캘린더] 글로벌 영업일: "
        + ", ".join(f"{y}={n}일" for y, n in sorted(calendar.items()))
        + f"  (총 {sum(calendar.values())}일, "
        + f"종목 파일 {len(sym_files):,}개 스캔)"
    )

    return calendar


# =============================================================================
# 입력 데이터 검증 함수 (PINstimation .check_args / .xcheck 패턴 이식)
# =============================================================================

def validate_vpin_parameters(
    rolling_window: int,
    buckets_per_day: int,
) -> None:
    """
    VPIN 추정 파라미터의 유효성을 검사한다.

    PINstimation 참조:
        R 패키지 vpin() 함수는 내부적으로 .check_args()를 호출하여 아래와 같은
        파라미터 검증을 수행한다:

        1) samplength (rolling_window) ≥ 1 인지 확인
           - 0이면 롤링 계산이 불가하므로 즉시 에러를 발생시킨다.

        2) buckets (buckets_per_day) ≥ 1 인지 확인
           - 0이면 VBS(Volume Bucket Size)가 무한대가 되어 버킷이 생성되지 않는다.

        3) samplength ≤ 합리적 범위 인지 확인
           - 너무 큰 rolling_window는 의미 있는 VPIN을 생성할 수 없다.
           - PINstimation은 데이터 크기 대비 samplength를 경고한다.

    적용 방어 로직:
        - rolling_window < 1: ValueError 발생
        - buckets_per_day < 1: ValueError 발생
        - 비정수 입력: TypeError 발생

    Raises:
        TypeError: 파라미터가 정수가 아닌 경우
        ValueError: 파라미터가 허용 범위를 벗어난 경우
    """
    if not isinstance(rolling_window, int) or not isinstance(buckets_per_day, int):
        raise TypeError(
            f"rolling_window과 buckets_per_day는 정수여야 합니다. "
            f"현재: rolling_window={type(rolling_window).__name__}, "
            f"buckets_per_day={type(buckets_per_day).__name__}"
        )
    if rolling_window < 1:
        raise ValueError(
            f"rolling_window(samplength)은 1 이상이어야 합니다. 현재: {rolling_window}"
        )
    if buckets_per_day < 1:
        raise ValueError(
            f"buckets_per_day는 1 이상이어야 합니다. 현재: {buckets_per_day}"
        )
    if rolling_window > 1000:
        logger.warning(
            f"rolling_window={rolling_window}은 매우 큽니다. "
            f"충분한 버킷이 생성되지 않으면 NaN만 반환될 수 있습니다."
        )


def validate_bar_dataframe(
    bar_df: pl.DataFrame,
    symbol_hint: str = "unknown",
) -> Tuple[bool, str]:
    """
    1분봉 DataFrame의 구조적·통계적 유효성을 검사한다.

    PINstimation 참조:
        R 패키지 vpin() 함수는 입력 데이터에 대해 아래 검증을 수행한다:

        1) 필수 컬럼 존재 여부:
           - PINstimation은 'price'와 'volume' 컬럼을 요구한다.
           - 여기서는 "Datetime", "Symbol", "Open", "Close", "Volume"이 필수이다.

        2) 데이터 타입 호환성:
           - Price, Volume이 numeric인지 확인한다.
           - PINstimation은 is.numeric() 검사를 수행한다.

        3) 결측치(NA/NaN) 검사:
           - PINstimation은 NA가 포함된 행을 자동으로 제거하며 경고를 출력한다.
           - 모든 행이 NA이면 estimation을 중단한다.

        4) 비양수 가격/거래량 검사:
           - 가격 ≤ 0 또는 거래량 ≤ 0 인 행은 비정상 데이터이다.
           - PINstimation은 이런 행을 자동 필터링하고 경고한다.

        5) 최소 거래일 수 검사:
           - 2일 미만의 데이터는 ADV(Average Daily Volume) 계산에 신뢰성이 없다.
           - 또한 가격 변동 표준편차가 의미가 없다.

        6) 최소 가격 변동 비율 검사:
           - 모든 봉의 가격이 동일(ΔP=0)하면 BVC 분류가 50:50이 되어
             VPIN이 항상 0에 수렴한다. 이는 정보 비대칭 측정이 불가능한 상태이다.
           - PINstimation은 이런 경우 경고 또는 에러를 발생시킨다.

    적용 방어 로직:
        - 필수 컬럼 누락 → (False, 사유)
        - NaN/Inf 행 → 자동 제거 후 로그 기록
        - 가격/거래량 ≤ 0 → 자동 제거 후 로그 기록
        - 거래일 < MIN_TRADING_DAYS → (False, 사유)
        - 가격 변동 0% 비율 > (1-MIN_NONZERO_DP_RATIO) → (False, 사유)

    Args:
        bar_df: 검증할 1분봉 DataFrame
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        (is_valid, reason): 유효성 여부와 사유 메시지
    """
    # ── (1) 필수 컬럼 존재 확인 ──
    required_cols = {"Datetime", "Symbol", "Open", "Close", "Volume"}
    missing = required_cols - set(bar_df.columns)
    if missing:
        msg = f"[{symbol_hint}] 필수 컬럼 누락: {missing}"
        logger.error(msg)
        return False, msg

    # ── (2) 빈 데이터프레임 ──
    if bar_df.is_empty():
        msg = f"[{symbol_hint}] 빈 데이터프레임"
        logger.info(msg)
        return False, msg

    n_original = bar_df.height

    # ── (3) NaN/Inf 검사 및 제거 ──
    numeric_cols = ["Open", "Close", "Volume"]
    nan_inf_mask = pl.lit(False)
    for col in numeric_cols:
        nan_inf_mask = nan_inf_mask | pl.col(col).is_null() | pl.col(col).is_nan()
        if bar_df[col].dtype in (pl.Float32, pl.Float64):
            nan_inf_mask = nan_inf_mask | pl.col(col).is_infinite()

    n_bad = bar_df.filter(nan_inf_mask).height
    if n_bad > 0:
        logger.warning(
            f"[{symbol_hint}] NaN/Inf/Null 행 {n_bad:,}개 제거 "
            f"({n_bad/n_original*100:.1f}%)"
        )

    # ── (4) 비양수 가격/거래량 검사 ──
    nonpositive_mask = (
        (pl.col("Close") <= 0) | (pl.col("Open") <= 0) | (pl.col("Volume") <= 0)
    )
    n_nonpos = bar_df.filter(nonpositive_mask).height
    if n_nonpos > 0:
        logger.warning(
            f"[{symbol_hint}] 비양수 가격/거래량 행 {n_nonpos:,}개 발견 "
            f"({n_nonpos/n_original*100:.1f}%)"
        )

    # 결합 필터 적용 후 남는 행 수 확인
    valid_mask = ~nan_inf_mask & ~nonpositive_mask
    n_valid = bar_df.filter(valid_mask).height

    if n_valid < MIN_ROWS:
        msg = (
            f"[{symbol_hint}] 유효 행 부족: {n_valid:,}행 < 최소 {MIN_ROWS}행 "
            f"(원본 {n_original:,}행, NaN/Inf {n_bad:,}행, "
            f"비양수 {n_nonpos:,}행 제거)"
        )
        logger.info(msg)
        return False, msg

    # ── (5) 최소 거래일 수 검사 ──
    n_days = bar_df.filter(valid_mask).select(
        pl.col("Datetime").dt.date().n_unique()
    ).item()

    if n_days < MIN_TRADING_DAYS:
        msg = (
            f"[{symbol_hint}] 거래일 부족: {n_days}일 < 최소 {MIN_TRADING_DAYS}일. "
            f"ADV 계산에 충분한 데이터가 없습니다."
        )
        logger.info(msg)
        return False, msg

    # ── (6) 가격 변동 비율 검사 (상수 가격 종목 탐지) ──
    close_arr = bar_df.filter(valid_mask)["Close"].to_numpy()
    delta_p = np.diff(close_arr)
    n_nonzero_dp = np.count_nonzero(delta_p)
    dp_ratio = n_nonzero_dp / max(len(delta_p), 1)

    if dp_ratio < MIN_NONZERO_DP_RATIO:
        msg = (
            f"[{symbol_hint}] 가격 변동 부족: ΔP≠0 비율 {dp_ratio:.4f} "
            f"< 최소 {MIN_NONZERO_DP_RATIO}. "
            f"BVC 분류가 50:50에 수렴하여 VPIN 추정이 무의미합니다. "
            f"(PINstimation: 'The proportion of zero price changes is too high')"
        )
        logger.warning(msg)
        return False, msg

    return True, "OK"


def sanitize_bar_dataframe(
    bar_df: pl.DataFrame,
    symbol_hint: str = "unknown",
) -> pl.DataFrame:
    """
    1분봉 DataFrame에서 비정상 행을 제거하고 정제된 데이터를 반환한다.

    PINstimation 참조:
        R 패키지는 vpin() 내부에서 데이터를 정제한 뒤 추정을 수행한다:

        1) NA 행 자동 제거:
           - na.omit(data) 패턴으로 결측치를 제거한다.
           - 제거된 행 수를 경고로 출력한다.

        2) 비양수 필터링:
           - price > 0 & volume > 0 조건으로 필터링한다.

        3) 정렬 보장:
           - 시간순 정렬이 보장되지 않으면 버킷 경계가 잘못 계산된다.
           - PINstimation은 내부적으로 timestamp 기준 정렬을 수행한다.

        4) 중복 타임스탬프 처리:
           - 동일 시간대에 여러 봉이 있으면 첫 번째를 사용하거나
             거래량 가중 평균을 사용한다.
           - 여기서는 group_by_dynamic으로 이미 집계되어 있지만,
             혹시 남아있는 중복은 마지막 값을 유지한다.

    적용 방어 로직:
        - NaN/Inf/Null → 행 제거
        - 가격 ≤ 0 또는 거래량 ≤ 0 → 행 제거
        - 시간순 정렬 보장
        - 중복 타임스탬프 → 마지막 행 유지 (최신 데이터 우선)

    Args:
        bar_df: 원본 1분봉 DataFrame
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        정제된 DataFrame
    """
    n_original = bar_df.height

    # NaN / Null / Inf 제거
    clean = bar_df.drop_nulls(subset=["Open", "Close", "Volume", "Datetime"])

    # Float 컬럼의 NaN/Inf 필터링
    for col in ["Open", "Close", "Volume"]:
        if clean[col].dtype in (pl.Float32, pl.Float64):
            clean = clean.filter(
                pl.col(col).is_finite() & ~pl.col(col).is_nan()
            )

    # 비양수 가격/거래량 제거
    clean = clean.filter(
        (pl.col("Close") > 0) & (pl.col("Open") > 0) & (pl.col("Volume") > 0)
    )

    # 시간순 정렬
    clean = clean.sort("Datetime")

    # 중복 타임스탬프 제거 (마지막 행 유지)
    n_before_dedup = clean.height
    clean = clean.unique(subset=["Datetime"], keep="last")
    n_dedup = n_before_dedup - clean.height

    n_removed = n_original - clean.height
    if n_removed > 0:
        logger.debug(
            f"[{symbol_hint}] 데이터 정제: {n_original:,} → {clean.height:,}행 "
            f"({n_removed:,}행 제거, 중복 {n_dedup:,}행)"
        )

    return clean


# =============================================================================
# Phase 1: 틱 데이터 → 파일별 1분봉 (임시 저장)
# =============================================================================

def process_file_to_1m_bars(parquet_path: str) -> pl.DataFrame:
    """
    틱 parquet 파일 1개 → 1분봉 DataFrame (Open, Close, Volume)

    PINstimation 참조:
        PINstimation의 vpin() 함수는 timebar 크기 (tbSize, 초 단위)를 받아
        raw 틱 데이터를 time bar로 집계한다:

        1) 컬럼 존재 확인:
           - 필요한 컬럼(price, volume, timestamp)이 없으면 즉시 에러를 반환한다.
           - estimate.vpin 객체의 success=FALSE, errorMessage에 사유를 기록한다.

        2) 빈 파일 / 읽기 실패 방어:
           - 파일이 비었거나 읽기에 실패하면 빈 DataFrame을 반환한다.
           - 호출자가 빈 DataFrame을 확인하고 해당 파일을 스킵한다.

        3) 거래량 0인 틱 필터링:
           - Volume=0인 틱은 체결이 아닌 호가 변경이므로 BVC에 무의미하다.
           - PINstimation은 이를 필터링한 뒤 timebar를 구성한다.

    적용 방어 로직:
        - 파일 읽기 실패 → 빈 DataFrame 반환, 에러 로그
        - 필수 컬럼 누락 → 빈 DataFrame 반환, 에러 로그
        - Volume=0 필터링 (기존 동일)
        - 집계 결과 빈 → 빈 DataFrame 반환
    """
    empty_schema = {
        "Symbol": pl.String, "Datetime": pl.Datetime,
        "Open": pl.Float64, "Close": pl.Float64, "Volume": pl.Float64,
    }
    fname = os.path.basename(parquet_path)

    # ── 파일 읽기 방어 ──
    try:
        raw_scan = pl.scan_parquet(parquet_path)
        available_cols = raw_scan.collect_schema().names()
    except Exception as e:
        logger.error(f"[Phase1] 파일 읽기 실패: {fname} → {e}")
        return pl.DataFrame(schema=empty_schema)

    # ── 필수 컬럼 확인 ──
    required = {"Symbol", "Date", "Time", "Price", "Volume"}
    missing = required - set(available_cols)
    if missing:
        logger.error(f"[Phase1] 필수 컬럼 누락 ({fname}): {missing}")
        return pl.DataFrame(schema=empty_schema)

    # ── 데이터 로드 및 집계 ──
    try:
        df = (
            raw_scan
            .select(["Symbol", "Date", "Time", "Price", "Volume"])
            .filter(pl.col("Volume") > 0)
            .with_columns(
                pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")
            )
            .drop(["Date", "Time"])
            .sort(["Symbol", "Datetime"])
            .collect()
        )
    except Exception as e:
        logger.error(f"[Phase1] 데이터 처리 실패 ({fname}): {e}")
        return pl.DataFrame(schema=empty_schema)

    if df.is_empty():
        logger.debug(f"[Phase1] 빈 데이터: {fname}")
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
# Step 3: VPIN 계산 핵심 함수 (PINstimation 예외처리 이식)
# =============================================================================

def _check_liquidity(
    bar_df: pl.DataFrame,
    year_bucket_map: Dict[int, float],
    business_day_calendar: Dict[int, int] = None,
    symbol_hint: str = "unknown",
) -> Tuple[bool, str]:
    """
    비유동 종목을 사전에 걸러내는 필터.

    문제 배경:
        VBS = ADV / buckets_per_day 이므로, "평균적으로" 하루 buckets_per_day개의
        버킷이 생성된다. 그러나 이것은 평균일 뿐이다. 비유동 종목은 거래량이
        소수의 날에 집중되고 나머지 대부분의 날에는 거래가 거의 없다.
        이 경우:

        - 거래량 0 또는 극소인 날에는 버킷이 아예 생성되지 않는다.
        - 거래량이 몰린 날에만 수십~수백 개 버킷이 한꺼번에 생성된다.
        - rolling_window=50 버킷이 며칠이 아닌 수주~수개월에 걸치게 되어
          VPIN의 "시의성(timeliness)"이 완전히 사라진다.
        - 이런 VPIN은 정보 비대칭의 실시간 척도로서 의미가 없다.

    필터 설계 원리:
        두 가지 기준을 동시에 적용하여, 서로 다른 유형의 비유동성을 포착한다.

        기준 1 — 일별 추정 버킷 수의 중앙값 (MIN_MEDIAN_DAILY_BUCKETS):
            글로벌 영업일 캘린더의 모든 영업일에 대해 추정 버킷 수를 계산한다.
            이 종목이 거래하지 않은 영업일은 버킷 0개로 포함된다.

            이 값의 중앙값이 MIN_MEDIAN_DAILY_BUCKETS(기본 2) 미만이면
            "대부분의 시장 영업일에 버킷이 거의 안 만들어지는 종목"으로 판단한다.

            왜 글로벌 영업일 기준인가:
                종목 자체의 거래일만 보면, 거래가 있었던 날은 당연히 버킷이
                생성되므로 유동성이 괜찮아 보인다. 하지만 그 종목이 250 영업일 중
                50일만 거래되었다면, 나머지 200일은 버킷이 0개이다.
                종목 거래일만 보면 이 사실이 드러나지 않는다.

            중앙값을 사용하는 이유:
                평균은 소수의 고거래량 날에 의해 왜곡된다. 중앙값은 "전형적인
                시장 영업일"의 버킷 생성 능력을 반영하므로, 거래량 편중이 심한
                종목을 정확하게 탐지한다.

            임계값=2의 의미:
                하루 버킷 2개 미만이면, rolling_window=50을 채우는 데
                최소 25 영업일(약 5주)이 필요하다. 이는 VPIN의 시의성 관점에서
                지나치게 긴 윈도우이다.

        기준 2 — 버킷 0개 생성일 비율 (MAX_ZERO_BUCKET_DAY_RATIO):
            글로벌 영업일 중 VBS보다 일 거래량이 적은 날(미거래일 포함)의 비율.
            MAX_ZERO_BUCKET_DAY_RATIO(기본 50%)를 초과하면 스킵한다.

            50% 임계값의 의미:
                절반 이상의 시장 영업일에 버킷이 0개면, 해당 종목의 VPIN 시계열에
                거대한 시간적 공백이 존재하여 연속적인 모니터링이 불가능하다.

    Args:
        bar_df: 정제된 1분봉 DataFrame (Datetime, Volume 컬럼 필수)
        year_bucket_map: {연도: VBS} 매핑 (이미 계산된 것)
        business_day_calendar: {연도: 영업일 수} 글로벌 캘린더.
                               None이면 종목 자체의 거래일만 사용.
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        (is_liquid, reason): 유동성 충분 여부와 사유 메시지
    """
    # ── 종목의 일별 거래량 집계 (거래가 있었던 날만) ──
    daily = (
        bar_df
        .with_columns([
            pl.col("Datetime").dt.date().alias("date"),
            pl.col("Datetime").dt.year().alias("year"),
        ])
        .group_by(["date", "year"])
        .agg(pl.col("Volume").sum().alias("daily_vol"))
    )

    # ── 종목 거래일의 연도별 집계 ──
    sym_days_by_year: Dict[int, int] = {}
    for row in daily.select("year").unique().iter_rows():
        y = row[0]
        sym_days_by_year[y] = daily.filter(pl.col("year") == y).height

    # ── 각 거래일의 추정 버킷 수 계산 ──
    est_buckets_list = []
    for row in daily.iter_rows(named=True):
        vbs = year_bucket_map.get(row["year"], 0.0)
        if vbs > 0:
            est_buckets_list.append(row["daily_vol"] / vbs)
        else:
            est_buckets_list.append(0.0)

    # ── 글로벌 캘린더 기반: 미거래 영업일을 버킷 0개로 추가 ──
    n_zero_from_missing = 0
    if business_day_calendar:
        for year, market_days in business_day_calendar.items():
            sym_days = sym_days_by_year.get(year, 0)
            missing_days = market_days - sym_days
            if missing_days > 0:
                # 이 종목이 거래하지 않은 영업일 = 버킷 0개
                est_buckets_list.extend([0.0] * missing_days)
                n_zero_from_missing += missing_days

    if not est_buckets_list:
        msg = f"[{symbol_hint}] 유동성 검사 실패: 유효 영업일 0일"
        logger.info(msg)
        return False, msg

    est_buckets = np.array(est_buckets_list)
    n_total_days = len(est_buckets)
    median_daily_buckets = float(np.median(est_buckets))
    zero_bucket_days = int(np.sum(est_buckets < 1.0))
    zero_bucket_ratio = zero_bucket_days / n_total_days

    if n_zero_from_missing > 0:
        logger.debug(
            f"[{symbol_hint}] 유동성 검사: 시장 영업일 {n_total_days}일 중 "
            f"종목 미거래일 {n_zero_from_missing}일 (버킷 0개로 포함)"
        )

    # ── 기준 1: 중앙값 검사 ──
    if median_daily_buckets < MIN_MEDIAN_DAILY_BUCKETS:
        msg = (
            f"[{symbol_hint}] 비유동 종목 스킵: "
            f"일별 추정 버킷 수 중앙값 {median_daily_buckets:.1f} "
            f"< 최소 {MIN_MEDIAN_DAILY_BUCKETS}. "
            f"(시장 영업일 {n_total_days}일 기준, "
            f"미거래일 {n_zero_from_missing}일 포함, "
            f"rolling_window={VPIN_ROLLING_WINDOW}을 채우는 데 "
            f"약 {VPIN_ROLLING_WINDOW / max(median_daily_buckets, 0.01):.0f}일 소요)"
        )
        logger.info(msg)
        return False, msg

    # ── 기준 2: 빈 날 비율 검사 ──
    if zero_bucket_ratio > MAX_ZERO_BUCKET_DAY_RATIO:
        msg = (
            f"[{symbol_hint}] 비유동 종목 스킵: "
            f"버킷 0개 생성일 {zero_bucket_days}/{n_total_days}일 "
            f"({zero_bucket_ratio:.1%}) > 상한 {MAX_ZERO_BUCKET_DAY_RATIO:.0%}. "
            f"(미거래 영업일 {n_zero_from_missing}일 포함) "
            f"VPIN 시계열에 과도한 시간 공백이 발생합니다."
        )
        logger.info(msg)
        return False, msg

    logger.debug(
        f"[{symbol_hint}] 유동성 통과: "
        f"중앙값 {median_daily_buckets:.1f} 버킷/일, "
        f"빈 날 {zero_bucket_ratio:.1%} "
        f"(시장 {n_total_days}일 기준)"
    )
    return True, "OK"

def _compute_bvc_probabilities(
    prices: np.ndarray,
    datetimes: list,
    symbol_hint: str = "unknown",
) -> Optional[np.ndarray]:
    """
    BVC (Bulk Volume Classification) 모델로 각 봉의 매수 확률을 계산한다.

    PINstimation 참조:
        PINstimation은 BVC 확률 계산 시 아래와 같은 수치 안정성 방어를 수행한다:

        1) 가격 변동 표준편차 = 0 방어:
           - 모든 봉의 가격이 동일하면 std(ΔP) = 0이 된다.
           - 이 경우 ΔP/σ = 0/0 → NaN이 전파되어 전체 VPIN이 NaN이 된다.
           - PINstimation은 이 경우를 사전에 감지하고 경고를 출력한다.
           - improved=TRUE(Ke & Lin, 2017) 모델에서도 동일한 검증을 수행한다.
           - 여기서는 validate_bar_dataframe의 MIN_NONZERO_DP_RATIO로 사전 차단하되,
             혹시 통과한 경우 σ에 머신 엡실론 대신 median(|ΔP|)의 일정 비율을 사용한다.

        2) t-분포 자유도 검증:
           - BVC 모델은 CDF_t(ΔP/σ; df) 를 사용한다.
           - Easley et al.(2012)은 df를 자유 파라미터로 두지만,
             실무적으로 df=0.25가 표준이다.
           - df ≤ 0이면 t-분포가 정의되지 않으므로 검증이 필요하다.

        3) CDF 출력 범위 검증:
           - t_dist.cdf()는 이론적으로 [0, 1] 범위이지만,
             극단적 입력(|ΔP/σ| > 1e15 등)에서 부동소수점 오차로
             0 또는 1을 정확히 반환하지 못할 수 있다.
           - np.clip(prob_buy, 0, 1)로 방어한다.

    야간 갭(Overnight Gap) 처리:
        영업일 경계(첫 번째 봉 또는 날짜가 바뀌는 시점)의 ΔP를 0으로 설정한다.

        문제:
            연속된 1분봉의 ΔP = Close[i] - Close[i-1] 을 구하면,
            금요일 마지막 봉과 월요일 첫 봉 사이(또는 공휴일 전후)의 ΔP에
            야간/주말 수익률이 포함된다. 이 overnight return은 장중
            정보거래(informed trading)와 무관한 뉴스 반영이므로 BVC 분류에
            노이즈를 유발한다. 또한, 큰 overnight gap이 σ(ΔP)를 부풀려
            장중 ΔP/σ 전체를 축소시키고, 모든 봉의 prob_buy를 0.5 방향으로
            밀어 VPIN을 체계적으로 과소추정하게 만든다.

        해결:
            - 영업일 경계를 인식: 날짜(date)가 바뀌는 첫 봉을 탐지한다.
            - 경계 봉의 ΔP = 0으로 설정 → CDF_t(0/σ) = 0.5 (중립 분류).
            - σ(ΔP) 계산에서 경계 봉을 제외: 순수 장중 변동성만 사용한다.

        이 방식은 모든 종목에 대해 거래 데이터가 있는 날을 영업일로
        간주하므로, 국가별 공휴일 캘린더 없이도 정확하게 동작한다.

    적용 방어 로직:
        - 영업일 경계 ΔP → 0 (중립 분류, σ 계산에서 제외)
        - σ(ΔP) ≈ 0 → MAD(Median Absolute Deviation) 기반 대체 또는 None 반환
        - prob_buy를 [0, 1]로 클리핑
        - NaN 전파 방지를 위한 np.nan_to_num 적용

    Args:
        prices: Close 가격 배열 (N,)
        datetimes: 타임스탬프 리스트 (N,) — 영업일 경계 탐지에 사용
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        매수 확률 배열 (N,) 또는 None (계산 불가 시)
    """
    if len(prices) < 2:
        return None

    delta_p = np.diff(prices, prepend=prices[0])

    # ── 영업일 경계 탐지 및 야간 갭 제거 ──
    # 날짜(date)가 바뀌는 봉 = 새 영업일의 첫 봉 → ΔP를 0으로 설정
    is_day_boundary = np.zeros(len(datetimes), dtype=bool)
    is_day_boundary[0] = True  # 첫 봉은 항상 경계
    prev_date = datetimes[0].date()
    for i in range(1, len(datetimes)):
        curr_date = datetimes[i].date()
        if curr_date != prev_date:
            is_day_boundary[i] = True
            prev_date = curr_date

    n_boundaries = is_day_boundary.sum()
    delta_p[is_day_boundary] = 0.0

    logger.debug(
        f"[{symbol_hint}] 영업일 경계 {n_boundaries}개 탐지 → "
        f"해당 봉의 ΔP=0 처리 (야간 갭 제거)"
    )

    # ── σ(ΔP) 계산: 영업일 경계 봉을 제외한 순수 장중 변동성 ──
    intraday_dp = delta_p[~is_day_boundary]

    if len(intraday_dp) < 2:
        logger.warning(
            f"[{symbol_hint}] 장중 ΔP가 2개 미만 (영업일 경계 제외 후). "
            f"BVC 분류 불가."
        )
        return None

    std_dp = np.std(intraday_dp)

    # ── σ ≈ 0 방어: MAD 기반 대체 시도 ──
    if std_dp < np.finfo(np.float64).eps * 100:
        # MAD (Median Absolute Deviation) 기반 robust 대체
        mad = np.median(np.abs(intraday_dp - np.median(intraday_dp)))
        # MAD → σ 변환 (정규분포 가정: σ ≈ 1.4826 * MAD)
        std_dp = 1.4826 * mad

        if std_dp < np.finfo(np.float64).eps * 100:
            logger.warning(
                f"[{symbol_hint}] σ(ΔP)=0: 모든 장중 가격이 동일합니다. "
                f"BVC 분류 불가 → 종목 스킵"
            )
            return None

        logger.debug(
            f"[{symbol_hint}] σ(ΔP) → MAD 기반 대체: σ={std_dp:.6e}"
        )

    # ── BVC 확률 계산 ──
    z_scores = delta_p / std_dp

    # 극단값 방어: |z| > 1e10 이면 CDF가 정확히 0 또는 1
    z_scores = np.clip(z_scores, -1e10, 1e10)

    prob_buy = t_dist.cdf(z_scores, df=0.25)

    # ── 출력 범위 보장 ──
    prob_buy = np.clip(prob_buy, 0.0, 1.0)

    # NaN이 남아있으면 0.5로 대체 (중립적 분류)
    nan_count = np.isnan(prob_buy).sum()
    if nan_count > 0:
        logger.warning(
            f"[{symbol_hint}] BVC prob_buy에 NaN {nan_count}개 발견 → 0.5로 대체"
        )
        prob_buy = np.nan_to_num(prob_buy, nan=0.5)

    return prob_buy


def _compute_volume_buckets(
    times_dt: list,
    volumes: np.ndarray,
    prob_buy: np.ndarray,
    year_bucket_map: Dict[int, float],
    symbol_hint: str = "unknown",
) -> Optional[List[Tuple]]:
    """
    거래량 기반 버킷을 생성한다 (Volume Bucketing).

    PINstimation 참조:
        PINstimation의 버킷 생성 로직에는 아래와 같은 방어 장치가 있다:

        1) VBS (Volume Bucket Size) = 0 방어:
           - ADV가 0이면 VBS = ADV / buckets_per_day = 0이 된다.
           - 이 경우 버킷이 절대 채워지지 않아 무한 루프가 발생한다.
           - PINstimation은 VBS ≤ 0이면 해당 연도를 건너뛴다.

        2) 버킷 수 상한 (MAX_BUCKET_COUNT):
           - VBS가 매우 작으면 수백만 개의 버킷이 생성될 수 있다.
           - 이는 메모리 부족과 처리 시간 폭발을 유발한다.
           - PINstimation은 합리적인 상한을 설정하여 방어한다.

        3) 연도 경계에서의 VBS 전환:
           - 데이터가 여러 연도에 걸쳐 있을 때 연도가 바뀌면 VBS도 변경된다.
           - PINstimation은 현재 진행 중인 버킷의 VBS를 유지하고,
             새 버킷부터 새 VBS를 적용한다 (기존 코드와 동일한 방식).

        4) 미완성 버킷 처리:
           - 마지막 봉을 처리한 후 버킷이 완전히 채워지지 않았을 수 있다.
           - PINstimation은 미완성 버킷을 버린다 (보수적 접근).
           - 이는 불완전한 정보 기반의 VPIN 편향을 방지한다.

        5) 단일 봉의 거래량이 VBS보다 큰 경우:
           - 한 봉의 거래량이 매우 커서 여러 버킷을 채울 수 있다.
           - 이 경우 while 루프에서 올바르게 처리해야 하며,
             PINstimation은 이를 재귀적으로 분배한다.
           - 기존 코드의 while 루프가 이를 처리하지만,
             무한 루프 방지를 위해 MAX_BUCKET_COUNT 검사를 추가한다.

    적용 방어 로직:
        - VBS ≤ 0 → 해당 연도 건너뜀
        - 버킷 수 > MAX_BUCKET_COUNT → 조기 종료 + 경고
        - 미완성 버킷 폐기 (기존 동일)
        - rem_vol > 0 무한루프 방어 (alloc ≤ 0 시 break)

    Args:
        times_dt: 타임스탬프 리스트
        volumes: 거래량 배열
        prob_buy: 매수 확률 배열
        year_bucket_map: {연도: VBS} 매핑
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        [(timestamp, bucket_no, |V_buy - V_sell|), ...] 리스트 또는 None
    """
    records: list[tuple] = []
    current_buy     = 0.0
    current_sell    = 0.0
    current_filled  = 0.0
    bucket_no       = 0
    last_year: Optional[int] = None
    bucket_size_v   = 0.0

    for i in range(len(volumes)):
        rem_vol   = float(volumes[i])
        p_b       = float(prob_buy[i])
        curr_time = times_dt[i]
        curr_year = curr_time.year

        if rem_vol <= 0:
            continue

        # ── 연도 경계 VBS 업데이트 ──
        if curr_year != last_year:
            new_vbs = year_bucket_map.get(curr_year, 0.0)
            if new_vbs <= 0:
                logger.warning(
                    f"[{symbol_hint}] 연도 {curr_year}: VBS={new_vbs:.2f} ≤ 0. "
                    f"해당 연도 데이터를 건너뜁니다."
                )
                last_year = curr_year
                bucket_size_v = 0.0
                continue
            bucket_size_v = new_vbs
            last_year = curr_year

        if bucket_size_v <= 0:
            continue

        # ── 거래량 → 버킷 분배 ──
        loop_guard = 0
        while rem_vol > 1e-12:
            needed = bucket_size_v - current_filled
            alloc  = min(needed, rem_vol)

            # 무한 루프 방어: alloc이 0이면 부동소수점 한계
            if alloc <= 1e-15:
                logger.debug(
                    f"[{symbol_hint}] 부동소수점 한계로 alloc≈0 → break "
                    f"(rem_vol={rem_vol:.2e}, needed={needed:.2e})"
                )
                break

            current_buy    += alloc * p_b
            current_sell   += alloc * (1.0 - p_b)
            current_filled += alloc
            rem_vol        -= alloc

            if current_filled >= bucket_size_v - 1e-9:
                records.append((
                    curr_time,
                    bucket_no,
                    abs(current_buy - current_sell),
                ))
                bucket_no       += 1
                current_buy      = 0.0
                current_sell     = 0.0
                current_filled   = 0.0

                # ── 버킷 수 상한 검사 ──
                if bucket_no >= MAX_BUCKET_COUNT:
                    logger.warning(
                        f"[{symbol_hint}] 버킷 수 상한 도달: {MAX_BUCKET_COUNT:,}. "
                        f"VBS가 너무 작거나 데이터가 매우 큽니다. "
                        f"buckets_per_day 값을 줄이는 것을 권장합니다."
                    )
                    return records if records else None

            loop_guard += 1
            if loop_guard > 10_000:
                logger.error(
                    f"[{symbol_hint}] 단일 봉 내 루프 10000회 초과 → break"
                )
                break

    return records if records else None


def _compute_rolling_vpin(
    records: List[Tuple],
    year_bucket_map: Dict[int, float],
    rolling_window: int,
    symbol_hint: str = "unknown",
) -> Optional[pl.DataFrame]:
    """
    버킷 레코드로부터 롤링 VPIN을 계산한다.

    PINstimation 참조:
        PINstimation의 VPIN 계산 최종 단계에서 수행하는 검증:

        1) 롤링 윈도우 > 총 버킷 수 검사:
           - samplength보다 생성된 버킷이 적으면 유효한 VPIN이 하나도 없다.
           - PINstimation은 이 경우 경고를 출력하고 모든 VPIN을 NA로 설정한다.
           - 여기서는 롤링 계산은 수행하되, 유효 VPIN이 0개이면 로그에 기록한다.

        2) VPIN 범위 검증 [0, 1]:
           - VPIN은 이론적으로 [0, 1] 범위이다.
           - |V_buy - V_sell| ≤ V 이므로, 정상적으로는 1을 초과할 수 없다.
           - 그러나 연도 경계에서 VBS가 변경되면 분모의 bucket_size가
             분자 계산 시의 bucket_size와 불일치하여 1을 초과할 수 있다.
           - PINstimation은 VPIN > 1 인 값을 1로 클리핑하고 경고한다.
           - VPIN < 0 은 이론적으로 불가하지만, 부동소수점 오차로 발생할 수 있다.

        3) 전체 NaN 검사:
           - 모든 VPIN이 NaN이면 추정이 실패한 것이다.
           - PINstimation의 estimate.vpin 객체에서 success=FALSE로 표시한다.

    적용 방어 로직:
        - 유효 버킷 < rolling_window → 경고 (NaN만 있는 결과)
        - VPIN 값을 [0, 1]로 클리핑
        - 전체 NaN → None 반환
        - 비정상 VPIN(>1) 비율 로깅

    Args:
        records: [(timestamp, bucket_no, order_imbalance), ...]
        year_bucket_map: {연도: VBS} 매핑
        rolling_window: 롤링 윈도우 크기
        symbol_hint: 로그에 표시할 종목코드

    Returns:
        VPIN DataFrame 또는 None
    """
    symbol_str = symbol_hint

    res_times = [r[0] for r in records]
    res_nos   = [r[1] for r in records]
    res_ois   = np.array([r[2] for r in records])
    bucket_bsv = np.array([
        year_bucket_map.get(t.year, 1e-5) for t in res_times
    ])

    n_buckets = len(res_ois)

    # ── 롤링 윈도우 vs 총 버킷 수 검사 ──
    if n_buckets < rolling_window:
        logger.info(
            f"[{symbol_hint}] 총 버킷 수({n_buckets}) < "
            f"rolling_window({rolling_window}). "
            f"유효한 VPIN이 생성되지 않습니다."
        )
        # 그래도 버킷 정보는 반환 (NaN VPIN)
        return pl.DataFrame({
            "Datetime": res_times,
            "Symbol":   [symbol_str] * n_buckets,
            "BucketNo": res_nos,
            "VPIN":     [np.nan] * n_buckets,
        }).with_columns([
            pl.col("Datetime").cast(pl.Datetime),
            pl.col("Symbol").cast(pl.String),
            pl.col("BucketNo").cast(pl.Int64),
            pl.col("VPIN").cast(pl.Float64),
        ])

    # ── 롤링 VPIN 계산 ──
    final_vpin = np.full(n_buckets, np.nan)
    oi_sums  = np.convolve(res_ois, np.ones(rolling_window), mode="valid")
    last_bsv = bucket_bsv[rolling_window - 1:]

    # 분모 = 0 방어
    denominator = rolling_window * last_bsv
    safe_denom = np.where(denominator > 1e-15, denominator, np.nan)
    raw_vpin = oi_sums / safe_denom

    final_vpin[rolling_window - 1:] = raw_vpin

    # ── VPIN 범위 클리핑 [0, 1] ──
    valid_mask = ~np.isnan(final_vpin)
    n_above_1 = np.sum(final_vpin[valid_mask] > VPIN_UPPER_BOUND)
    n_below_0 = np.sum(final_vpin[valid_mask] < VPIN_LOWER_BOUND)

    if n_above_1 > 0:
        ratio = n_above_1 / valid_mask.sum() * 100
        logger.warning(
            f"[{symbol_hint}] VPIN > 1.0: {n_above_1}개 ({ratio:.1f}%). "
            f"연도 경계 VBS 불일치 가능성. 1.0으로 클리핑합니다."
        )
    if n_below_0 > 0:
        logger.warning(
            f"[{symbol_hint}] VPIN < 0.0: {n_below_0}개. "
            f"부동소수점 오차. 0.0으로 클리핑합니다."
        )

    final_vpin = np.clip(final_vpin, VPIN_LOWER_BOUND, VPIN_UPPER_BOUND)
    # NaN은 clip에 영향받지 않으므로 안전

    # ── 전체 NaN 검사 ──
    n_valid_vpin = np.sum(~np.isnan(final_vpin))
    if n_valid_vpin == 0:
        logger.warning(
            f"[{symbol_hint}] 모든 VPIN이 NaN입니다. "
            f"(PINstimation: success=FALSE)"
        )
        return None

    return pl.DataFrame({
        "Datetime": res_times,
        "Symbol":   [symbol_str] * n_buckets,
        "BucketNo": res_nos,
        "VPIN":     final_vpin,
    }).with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64),
    ])


def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int,
    buckets_per_day: int,
    business_day_calendar: Dict[int, int] = None,
) -> pl.DataFrame:
    """
    단일 종목의 1분봉 데이터를 받아 VPIN을 계산한다.

    PINstimation 전체 파이프라인 대응:
        이 함수는 PINstimation R 패키지의 vpin() 함수 전체 흐름을 대응한다.
        PINstimation의 내부 처리 순서와 예외처리를 그대로 이식했다:

        ┌─────────────────────────────────────────────────────────────────┐
        │ PINstimation vpin() 흐름         │ 이 함수의 대응              │
        ├─────────────────────────────────────────────────────────────────┤
        │ 1. .check_args()                 │ validate_vpin_parameters()  │
        │    - samplength, buckets 검증     │   (메인에서 1회 호출)       │
        │                                                                │
        │ 2. .xcheck(data)                 │ validate_bar_dataframe()    │
        │    - 컬럼, 타입, NA 검사          │ sanitize_bar_dataframe()   │
        │    - 비양수 데이터 필터링          │                            │
        │                                                                │
        │ 3. timebar 집계                  │ Phase 1에서 완료            │
        │    - tbSize 기반 OHLCV            │                            │
        │                                                                │
        │ 4. ADV / VBS 계산                │ year_stats → year_bucket_map│
        │    - 연도별 평균 일거래량          │   ※ 글로벌 영업일 캘린더    │
        │    - VBS = ADV / buckets          │     기반 n_days 사용       │
        │                                                                │
        │ 5. BVC 확률 계산                 │ _compute_bvc_probabilities()│
        │    - ΔP/σ → CDF_t(df=0.25)       │   야간갭 제거, σ≈0 방어    │
        │                                                                │
        │ 6. Volume Bucketing              │ _compute_volume_buckets()   │
        │    - 버킷 생성 + OI 계산          │   VBS=0, 무한루프, 상한 방어│
        │                                                                │
        │ 7. Rolling VPIN                  │ _compute_rolling_vpin()     │
        │    - Σ|OI| / (n × VBS)            │   범위[0,1], 전체NaN 방어   │
        │                                                                │
        │ 8. estimate.vpin 객체 생성       │ pl.DataFrame 반환           │
        │    - success, errorMessage 설정   │   빈 DataFrame = 실패      │
        └─────────────────────────────────────────────────────────────────┘

    추가 방어 (PINstimation에 없지만 실무적으로 필요한 것):
        - 연도별 ADV 계산 시 n_days=0 방어 (기존 코드의 1e-5 대체 → 명시적 스킵)
        - 단일 봉 거래량 >> VBS인 경우 multi-bucket 분배의 무한 루프 방어
        - 로그에 스킵 사유를 상세히 기록하여 사후 진단 가능

    BVC 모델: ProbBuy = CDF_t(ΔP/σ; df=0.25)
    버킷 크기 V = ADV(해당 연도) / buckets_per_day
    VPIN = rolling_sum(|V_buy - V_sell|) / (rolling_window × V)

    글로벌 영업일 캘린더 적용:
        ADV 계산 시 분모(n_days)에 이 종목의 거래일 수가 아닌
        글로벌 영업일 캘린더의 영업일 수를 사용한다.

        이유:
            종목 A가 연간 250 영업일 중 100일만 거래되었을 때,
            - 종목별 기준: ADV = total_vol / 100  → VBS 과대 → 버킷 과소
            - 글로벌 기준: ADV = total_vol / 250  → VBS 정확 → 버킷 적정

            글로벌 기준이 "시장이 열린 하루 동안 이 종목에 기대되는 평균
            거래량"을 더 정확하게 반영한다. 종목별 기준은 "거래가 있었던
            날만의 평균"이므로, 비유동 종목의 ADV를 과대평가하여 VBS가 커지고
            활성 거래일에도 버킷이 충분히 생성되지 않는 문제가 있다.

    Args:
        bar_df: 단일 종목의 1분봉 DataFrame
                필수 컬럼: Datetime, Symbol, Open, Close, Volume
        rolling_window: 롤링 윈도우 버킷 수 (PINstimation의 samplength)
        buckets_per_day: 하루당 버킷 수 (PINstimation의 buckets)
        business_day_calendar: {연도: 영업일 수} 글로벌 캘린더.
                               None이면 종목 자체의 거래일 수를 사용 (하위 호환).

    Returns:
        VPIN DataFrame (Datetime, Symbol, BucketNo, VPIN)
        실패 시 빈 DataFrame 반환
    """
    schema = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64,    "VPIN":   pl.Float64,
    }
    empty = pl.DataFrame(schema=schema)

    # ── (1) 종목코드 추출 ──
    if bar_df.is_empty():
        return empty
    symbol_str = str(bar_df["Symbol"][0])

    # ── (2) 입력 데이터 검증 (PINstimation .xcheck 대응) ──
    is_valid, reason = validate_bar_dataframe(bar_df, symbol_hint=symbol_str)
    if not is_valid:
        return empty

    # ── (3) 데이터 정제 ──
    bar_df = sanitize_bar_dataframe(bar_df, symbol_hint=symbol_str)
    if bar_df.height < 2:
        logger.info(f"[{symbol_str}] 정제 후 행 부족: {bar_df.height}행")
        return empty

    times_dt = bar_df["Datetime"].to_list()
    prices   = bar_df["Close"].to_numpy()
    volumes  = bar_df["Volume"].to_numpy()

    # ── (4) 연도별 ADV 기반 VBS 계산 (글로벌 영업일 캘린더 적용) ──
    year_stats = (
        bar_df
        .with_columns(pl.col("Datetime").dt.year().alias("Year"))
        .group_by("Year")
        .agg([
            pl.col("Volume").sum().alias("total_vol"),
            pl.col("Datetime").dt.date().n_unique().alias("sym_trading_days"),
        ])
    )

    year_bucket_map: Dict[int, float] = {}
    for row in year_stats.iter_rows(named=True):
        year = row["Year"]
        sym_trading_days = row["sym_trading_days"]
        total_vol = row["total_vol"]

        # 글로벌 캘린더가 있으면 시장 영업일 수 사용, 없으면 종목 거래일 사용
        if business_day_calendar and year in business_day_calendar:
            n_days = business_day_calendar[year]
        else:
            n_days = sym_trading_days

        if n_days <= 0 or total_vol <= 0:
            logger.warning(
                f"[{symbol_str}] 연도 {year}: n_days={n_days}, "
                f"total_vol={total_vol:.0f} → VBS 계산 불가, 해당 연도 스킵"
            )
            year_bucket_map[year] = 0.0
            continue

        adv = total_vol / n_days
        vbs = adv / buckets_per_day

        if business_day_calendar and sym_trading_days < n_days:
            logger.debug(
                f"[{symbol_str}] 연도 {year}: 종목 거래일 {sym_trading_days}일 / "
                f"시장 영업일 {n_days}일 "
                f"(미거래일 {n_days - sym_trading_days}일, "
                f"거래일 비율 {sym_trading_days/n_days:.1%})"
            )

        if vbs < 1.0:
            logger.warning(
                f"[{symbol_str}] 연도 {year}: VBS={vbs:.2f} < 1.0. "
                f"ADV={adv:.0f}, buckets_per_day={buckets_per_day}. "
                f"버킷이 과다 생성될 수 있습니다."
            )

        year_bucket_map[year] = vbs

    if not year_bucket_map or all(v <= 0 for v in year_bucket_map.values()):
        logger.info(f"[{symbol_str}] 모든 연도의 VBS가 0 이하 → 스킵")
        return empty

    # ── (4b) 비유동 종목 필터 ──
    is_liquid, liquidity_msg = _check_liquidity(
        bar_df, year_bucket_map,
        business_day_calendar=business_day_calendar,
        symbol_hint=symbol_str,
    )
    if not is_liquid:
        return empty

    # ── (5) BVC 확률 계산 ──
    prob_buy = _compute_bvc_probabilities(
        prices, datetimes=times_dt, symbol_hint=symbol_str,
    )
    if prob_buy is None:
        return empty

    # ── (6) 볼륨 버킷 생성 ──
    records = _compute_volume_buckets(
        times_dt, volumes, prob_buy, year_bucket_map,
        symbol_hint=symbol_str,
    )
    if records is None:
        logger.info(f"[{symbol_str}] 버킷 생성 실패: 완성된 버킷 0개")
        return empty

    # ── (7) 롤링 VPIN 계산 ──
    result = _compute_rolling_vpin(
        records, year_bucket_map, rolling_window,
        symbol_hint=symbol_str,
    )
    if result is None:
        return empty

    return result


# =============================================================================
# 멀티프로세싱 워커
# =============================================================================

_ROLLING_WINDOW:  int = 50
_BUCKETS_PER_DAY: int = 50
_BUSINESS_DAY_CALENDAR: Dict[int, int] = {}


def init_worker(
    rolling_window: int,
    buckets_per_day: int,
    business_day_calendar: Dict[int, int],
) -> None:
    """
    멀티프로세싱 워커 프로세스 초기화.

    각 워커 프로세스가 생성될 때 한 번 호출되어, VPIN 계산에 필요한
    파라미터와 글로벌 영업일 캘린더를 프로세스 전역 변수로 설정한다.

    business_day_calendar는 메인 프로세스에서 build_business_day_calendar()로
    생성한 {연도: 영업일 수} 딕셔너리이며, pickle 직렬화를 통해 각 워커에
    전달된다. 이 캘린더는 읽기 전용이므로 프로세스 간 동기화가 불필요하다.
    """
    global _ROLLING_WINDOW, _BUCKETS_PER_DAY, _BUSINESS_DAY_CALENDAR
    _ROLLING_WINDOW        = rolling_window
    _BUCKETS_PER_DAY       = buckets_per_day
    _BUSINESS_DAY_CALENDAR = business_day_calendar


def process_symbol_worker(args: tuple) -> str | None:
    """
    종목별 parquet 읽기 → VPIN 계산 → 결과 저장

    PINstimation 참조:
        PINstimation의 estimate.vpin 클래스는 각 추정 결과에 대해
        success (logical)와 errorMessage (character) 슬롯을 제공한다.
        추정이 실패하면 success=FALSE, errorMessage에 사유가 기록된다.

        이를 참고하여, 파이썬 워커에서는:
        - 성공 시: 결과 파일 경로를 반환
        - 실패 시: None을 반환하고 로거에 사유를 기록

    적용 방어 로직:
        - 파일 읽기 실패 → None + 에러 로그
        - 빈 데이터 / 행 부족 → None + 디버그 로그
        - VPIN 계산 실패 → None + 에러 로그 (스택트레이스 포함)
        - 결과 쓰기 실패 → None + 에러 로그
    """
    symbol, input_path, output_path = args
    try:
        # ── 파일 읽기 ──
        try:
            bar_df = pl.read_parquet(input_path)
        except Exception as e:
            logger.error(f"[Worker] 파일 읽기 실패 ({symbol}): {e}")
            return None

        if bar_df.is_empty() or bar_df.height < MIN_ROWS:
            logger.debug(
                f"[Worker] 행 부족 ({symbol}): {bar_df.height}행 < {MIN_ROWS}"
            )
            return None

        # ── VPIN 계산 ──
        vpin_df = calculate_vpin_for_single_symbol(
            bar_df, _ROLLING_WINDOW, _BUCKETS_PER_DAY, _BUSINESS_DAY_CALENDAR
        )
        if vpin_df.is_empty():
            return None

        # ── 결과 저장 ──
        try:
            vpin_df.write_parquet(output_path, compression="zstd")
        except Exception as e:
            logger.error(f"[Worker] 결과 저장 실패 ({symbol}): {e}")
            return None

        return output_path

    except Exception as e:
        import traceback
        logger.error(
            f"[Worker] 예기치 않은 오류 ({symbol}): {e}\n"
            f"{traceback.format_exc()}"
        )
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
            logger.warning(f"결과 이동 실패: {fname} ({e})")
    return moved


def run_vpin_calculation():
    """
    bars_by_symbol/ 의 종목별 파일을 읽어 VPIN을 계산하고
    results/sym_{Symbol}.parquet 로 저장한다.

    PINstimation 참조:
        PINstimation은 추정 시작 전 .check_args()로 파라미터를 검증한다.
        여기서도 동일하게 validate_vpin_parameters()를 호출한다.
    """
    # ── 파라미터 사전 검증 (PINstimation .check_args 대응) ──
    validate_vpin_parameters(VPIN_ROLLING_WINDOW, VPIN_BUCKETS_PER_DAY)

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

    # ── 글로벌 영업일 캘린더 생성 ──
    business_day_calendar = build_business_day_calendar(SYMBOL_OUT_DIR)
    if not business_day_calendar:
        print("[Error] 영업일 캘린더 생성 실패")
        return

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
        initargs=(VPIN_ROLLING_WINDOW, VPIN_BUCKETS_PER_DAY, business_day_calendar),
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