"""
=============================================================================
[PIN/APIN 파이프라인] tick parquet → 일별 B/S 캐시 → 종목별 weekly PIN/APIN 체크포인트

목적
  기존 01_pin_weekly.py / 02_apin_weekly.py의 계산 로직은 유지하고,
  0507_vpin_05_한번에계산.py의 파일 입출력 구조와 중단점 재개 방식을 이식한다.

메모리 절약/재개 흐름
  원천 parquet
    → Step 1: 파일별 lazy/streaming 일별 B/S 집계 캐시
              output/.../pin_apin/daily_counts/*.parquet
    → Step 2: 종목별 weekly PIN/APIN 계산
              output/.../pin_apin/{pin|apin}/checkpoints/sym_{Symbol}.parquet
              output/.../pin_apin/{pin|apin}/meta/sym_{Symbol}.json
    → Step 3: 체크포인트 lazy 병합 → 최종 parquet + sample csv
    → Step 4: 간단 검증 리포트 생성

중요
  · PIN/APIN의 MLE, grid search, 4주 rolling window, 최종 산식은 원본 코드와 같게 유지했다.
  · PIN/APIN은 VPIN처럼 bucket state를 이어받는 모델이 아니므로, 중단점은 '종목별 체크포인트'다.
  · 재실행 시 이미 정상 생성된 일별 캐시와 종목별 체크포인트는 스킵한다.
=============================================================================
"""

import os
import re
import sys
import glob
import gc
import json
import math
import shutil
import warnings
import traceback
import multiprocessing
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import itertools
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
from tqdm import tqdm

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.filterwarnings("ignore")


# =============================================================================
# 사용자 설정
# =============================================================================

# VPIN 코드와 같은 입력 루트 구조를 기본으로 사용한다.
# 예: BASE_DIR 아래에 CN_2019, CN_2020, ... 폴더가 있고 각 폴더에 parquet이 있는 구조.
BASE_DIR = r"E:\vpin_project_parquet/processing_data_2019_2021"
COUNTRY = "CN"

# 둘 다 계산하려면 ["PIN", "APIN"], 하나만 계산하려면 ["PIN"] 또는 ["APIN"]
MODELS_TO_RUN = ["PIN", "APIN"]

# 단일/임의 파일만 처리해야 하면 여기에 parquet 경로를 직접 넣는다.
# 비워두면 BASE_DIR/COUNTRY_YYYY/*.parquet을 자동 탐색한다.
INPUT_FILES: List[str] = []

# Step 1 일별 B/S 캐시를 강제로 다시 만들지 여부
FORCE_REPROCESS_DAILY_CACHE = False

# Step 2 모델별 종목 체크포인트를 강제로 다시 계산할지 여부
FORCE_RECOMPUTE_MODEL = False

# Step 1 오류가 하나라도 있으면 중단할지 여부
FAIL_ON_STEP1_ERROR = True

# Step 2 일부 종목 오류가 있으면 중단할지 여부
# False이면 성공한 종목 체크포인트만 병합하고 리포트에 오류를 남긴다.
FAIL_ON_STEP2_ERROR = False

# 32GB RAM 기준 APIN은 grid가 59,049개라 워커를 과도하게 높이지 않는 것을 권장.
# None이면 cpu_count()-1을 사용한다.
NUM_WORKERS: Optional[int] = None

# 원본 코드와 동일: target week + 과거 3주 = 총 4주 window, 유효 일수 > 10일이면 추정
MIN_OBS_DAYS = 10
ROLLING_LOOKBACK_WEEKS = 3

# 검증 리포트에서 표시할 극단값 기준
REPORT_EXTREME_THRESHOLD = 0.50


# =============================================================================
# 경로/상수
# =============================================================================

VALID_COUNTRIES = {"KOR", "US", "JP", "CA", "FR", "GR", "HK", "IT", "UK", "CN"}
_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")
_INVALID_FILENAME_CHARS = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_VALID_MODELS = {"PIN", "APIN"}


def output_root() -> str:
    return os.path.join(BASE_DIR, "0416_output", COUNTRY, "pin_apin")


def daily_counts_dir() -> str:
    return os.path.join(output_root(), "daily_counts")


def model_output_root(model: str) -> str:
    return os.path.join(output_root(), model.lower())


def checkpoint_dir(model: str) -> str:
    return os.path.join(model_output_root(model), "checkpoints")


def meta_dir(model: str) -> str:
    return os.path.join(model_output_root(model), "meta")


# =============================================================================
# 공통 유틸리티
# =============================================================================


def safe_filename(value: object) -> str:
    cleaned = _INVALID_FILENAME_CHARS.sub("_", str(value)).strip(" .")
    return cleaned if cleaned else "_EMPTY_"


def normalize_model(model: str) -> str:
    model = str(model).upper().strip()
    if model not in _VALID_MODELS:
        raise ValueError(f"지원하지 않는 모델입니다: {model}. 사용 가능: {sorted(_VALID_MODELS)}")
    return model


def write_parquet_atomic(df: pl.DataFrame, out_path: str) -> None:
    """중단/오류 시 깨진 parquet이 완료 파일명으로 남지 않도록 임시 파일 후 교체."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp_{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    df.write_parquet(tmp_path, compression="zstd")
    os.replace(tmp_path, out_path)


def write_json_atomic(data: Dict[str, Any], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp_path = f"{out_path}.tmp_{os.getpid()}"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp_path, out_path)


def _fmt_num(x: Any, digits: int = 6) -> str:
    if x is None:
        return "N/A"
    try:
        xf = float(x)
        if math.isnan(xf):
            return "NaN"
        return f"{xf:.{digits}f}"
    except (TypeError, ValueError):
        return str(x)


def get_country_folders(base_dir: str, country: str) -> List[str]:
    folders = []
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if entry.is_dir():
            m = _YEAR_FOLDER_RE.match(entry.name)
            if m and m.group(1) == country:
                folders.append(entry.path)
    return folders


def get_parquet_files(base_dir: str, country: str) -> List[str]:
    if INPUT_FILES:
        files = [os.path.abspath(p) for p in INPUT_FILES]
        missing = [p for p in files if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError("INPUT_FILES에 없는 파일이 있습니다:\n" + "\n".join(missing[:20]))
        print(f"\n[파일 탐색] INPUT_FILES 직접 지정: parquet {len(files):,}개")
        return sorted(files)

    folders = get_country_folders(base_dir, country)
    if not folders:
        raise RuntimeError(f"'{country}_YYYY' 패턴 폴더를 찾을 수 없습니다: {base_dir}")

    print(f"\n[파일 탐색] {base_dir}  (나라코드: {country})")
    files: List[str] = []
    for folder in folders:
        found = sorted(glob.glob(os.path.join(folder, "*.parquet")))
        print(f"  {os.path.basename(folder)}: parquet {len(found):,}개")
        files.extend(found)
    print(f"  합계: parquet {len(files):,}개")
    return sorted(files)


# =============================================================================
# Step 1: 원천 parquet → 파일별 일별 B/S 캐시
# =============================================================================


def daily_count_out_path(input_path: str) -> str:
    """
    입력 parquet 경로를 일별 B/S 캐시 경로로 변환한다.
    부모 폴더명(CN_2010 등)을 포함해 서로 다른 연도 폴더의 동일 파일명 충돌을 방지한다.
    """
    folder = safe_filename(os.path.basename(os.path.dirname(input_path)))
    stem = safe_filename(os.path.splitext(os.path.basename(input_path))[0])
    return os.path.join(daily_counts_dir(), f"{folder}__{stem}_daily_bs.parquet")


def is_valid_daily_count_file(path: str) -> bool:
    """일별 B/S 캐시 parquet이 읽히고 필요한 컬럼이 있으면 True."""
    if not os.path.exists(path):
        return False
    try:
        (
            pl.scan_parquet(path)
            .select(["Symbol", "Date", "B", "S", "Week_Start"])
            .head(1)
            .collect(streaming=True)
        )
        return True
    except Exception:
        return False


def file_to_daily_counts(path: str) -> pl.DataFrame:
    """
    원본 PIN/APIN의 preprocess_trade_data_polars 로직을 파일 단위 캐시 생성으로 분리.
      - LR == 1  → B
      - LR == -1 → S
      - Symbol-Date 기준 집계
      - Week_Start = Date가 속한 주의 월요일
    """
    lf = (
        pl.scan_parquet(path)
        .select(["Symbol", "Date", "LR"])
        .with_columns([
            pl.col("Symbol").cast(pl.String).alias("Symbol"),
            pl.col("Date").cast(pl.Date, strict=False).alias("Date"),
            pl.col("LR").cast(pl.Int8, strict=False).alias("LR"),
        ])
        .filter(
            pl.col("Symbol").is_not_null()
            & pl.col("Date").is_not_null()
            & pl.col("LR").is_in([1, -1])
        )
        .with_columns([
            (pl.col("LR") == 1).cast(pl.UInt64).alias("is_Buy"),
            (pl.col("LR") == -1).cast(pl.UInt64).alias("is_Sell"),
        ])
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("is_Buy").sum().alias("B"),
            pl.col("is_Sell").sum().alias("S"),
        ])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))
        .with_columns(pl.col("Date").dt.truncate("1w").alias("Week_Start"))
        .select([
            pl.col("Symbol").cast(pl.String),
            pl.col("Date").cast(pl.Date),
            pl.col("B").cast(pl.UInt64),
            pl.col("S").cast(pl.UInt64),
            pl.col("Week_Start").cast(pl.Date),
        ])
        .sort(["Symbol", "Date"])
    )
    return lf.collect(streaming=True)


def run_step1(base_dir: str, country: str) -> List[str]:
    os.makedirs(daily_counts_dir(), exist_ok=True)

    if FORCE_REPROCESS_DAILY_CACHE and os.path.isdir(daily_counts_dir()):
        print(f"\n[Step 1] 기존 일별 B/S 캐시 삭제: {daily_counts_dir()}")
        shutil.rmtree(daily_counts_dir())
        os.makedirs(daily_counts_dir(), exist_ok=True)

    parquet_files = get_parquet_files(base_dir, country)
    expected_pairs = [(path, daily_count_out_path(path)) for path in parquet_files]

    valid_set = {out for _, out in expected_pairs if is_valid_daily_count_file(out)}
    missing_or_invalid = [(path, out) for path, out in expected_pairs if out not in valid_set]

    print(f"\n{'='*65}")
    print(f"[Step 1] 원천 parquet → 일별 B/S 캐시 [{country}]")
    print(f"{'='*65}")
    print(f"  입력 parquet 수      : {len(parquet_files):,}")
    print(f"  기존 정상 캐시       : {len(valid_set):,}")
    print(f"  생성/재생성 대상     : {len(missing_or_invalid):,}")

    total_new_rows = 0
    errors: List[Tuple[str, str]] = []

    for path, out_path in tqdm(missing_or_invalid, desc="일별 B/S 변환/재개"):
        try:
            if os.path.exists(out_path) and not is_valid_daily_count_file(out_path):
                os.remove(out_path)
            df = file_to_daily_counts(path)
            write_parquet_atomic(df, out_path)
            total_new_rows += df.height
            del df
            gc.collect()
        except Exception as e:
            errors.append((path, str(e)))
            print(f"  [Error] 일별 B/S 변환 실패: {os.path.basename(path)}: {e}")

    final_files = [out for _, out in expected_pairs if is_valid_daily_count_file(out)]
    final_set = set(final_files)
    missing = [out for _, out in expected_pairs if out not in final_set]

    print(f"\n[Step 1 완료/재개]")
    print(f"  일별 B/S 캐시 폴더 : {daily_counts_dir()}")
    print(f"  정상 캐시 파일 수  : {len(final_files):,} / {len(expected_pairs):,}")
    print(f"  새로 생성한 행 수  : {total_new_rows:,}")

    if errors or missing:
        msg = ["Step 1 캐시 생성이 완전하지 않습니다."]
        if errors:
            msg.append(f"오류 파일 수: {len(errors):,}")
            for path, err in errors[:10]:
                msg.append(f"  - {path}: {err}")
        if missing:
            msg.append(f"누락/손상 캐시 수: {len(missing):,}")
            for out in missing[:10]:
                msg.append(f"  - {out}")
        if FAIL_ON_STEP1_ERROR:
            raise RuntimeError("\n".join(msg))
        print("\n".join(msg))

    return sorted(final_files)


# =============================================================================
# PIN 계산 로직: 원본 01_pin_weekly.py 유지
# =============================================================================


def estimate_pin_parameters(B_array: np.ndarray, S_array: np.ndarray, grid_combinations: np.ndarray) -> Dict[str, Any]:
    """
    [파라미터 추정 함수 - MLE]
    EKOP(1996) 모델 기반 PIN 추정.

    계산 로직은 원본 01_pin_weekly.py의 estimate_pin_parameters와 같게 유지했다.
    """
    B = B_array
    S = S_array

    a_g = grid_combinations[:, 0:1]
    d_g = grid_combinations[:, 1:2]
    mu_g = grid_combinations[:, 2:3]
    eb_g = grid_combinations[:, 3:4]
    es_g = grid_combinations[:, 4:5]

    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)

    p1 = (1 - a_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, es_g)
    p2 = a_g * d_g * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, mu_g + es_g)
    p3 = a_g * (1 - d_g) * poisson.pmf(B_row, mu_g + eb_g) * poisson.pmf(S_row, es_g)

    total_lik = p1 + p2 + p3
    total_lik[total_lik <= 0] = 1e-300

    nll_scores = -np.sum(np.log(total_lik), axis=1)
    best_idx = np.argmin(nll_scores)
    best_initial_guess = grid_combinations[best_idx]

    def negative_log_likelihood(params: Sequence[float]) -> float:
        alpha, delta, mu, eb, es = params

        if not (0 <= alpha <= 1 and 0 <= delta <= 1 and mu >= 0 and eb >= 0 and es >= 0):
            return np.inf

        epsilon_float = 1e-300

        prob_no_news = poisson.pmf(B, eb) * poisson.pmf(S, es)
        prob_bad_news = poisson.pmf(B, eb) * poisson.pmf(S, mu + es)
        prob_good_news = poisson.pmf(B, mu + eb) * poisson.pmf(S, es)

        likelihood = (1 - alpha) * prob_no_news + \
                     (alpha * delta) * prob_bad_news + \
                     (alpha * (1 - delta)) * prob_good_news

        log_lik = np.log(likelihood + epsilon_float)
        return float(-np.sum(log_lik))

    try:
        best_result = minimize(
            negative_log_likelihood,
            x0=best_initial_guess,
            bounds=[(0, 1), (0, 1), (0, None), (0, None), (0, None)],
            method="L-BFGS-B",
        )
    except Exception:
        return {"converged": False}

    if not best_result.success:
        return {"converged": False}

    alpha, delta, mu, eb, es = best_result.x

    denom = alpha * mu + eb + es
    if denom == 0:
        denom = 1e-9
    pin = (alpha * mu) / denom

    return {
        "alpha": alpha,
        "delta": delta,
        "mu": mu,
        "eb": eb,
        "es": es,
        "PIN": pin,
        "converged": True,
    }


# =============================================================================
# APIN 계산 로직: 원본 02_apin_weekly.py 유지
# =============================================================================


def estimate_apin_parameters(B_array: np.ndarray, S_array: np.ndarray, grid_combinations: np.ndarray) -> Dict[str, Any]:
    """
    [파라미터 추정 함수 - MLE]
    Duarte & Young (2009) APIN 모델.

    계산 로직은 원본 02_apin_weekly.py의 estimate_apin_parameters와 같게 유지했다.
    """
    B = B_array
    S = S_array

    a_g = grid_combinations[:, 0:1]
    d_g = grid_combinations[:, 1:2]
    t1_g = grid_combinations[:, 2:3]
    t2_g = grid_combinations[:, 3:4]
    ub_g = grid_combinations[:, 4:5]
    us_g = grid_combinations[:, 5:6]
    eb_g = grid_combinations[:, 6:7]
    es_g = grid_combinations[:, 7:8]
    pb_g = grid_combinations[:, 8:9]
    ps_g = grid_combinations[:, 9:10]

    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)

    p1 = (1 - a_g) * (1 - t1_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, es_g)
    p2 = (1 - a_g) * t1_g * poisson.pmf(B_row, eb_g + pb_g) * poisson.pmf(S_row, es_g + ps_g)
    p3 = a_g * (1 - t2_g) * (1 - d_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, us_g + es_g)
    p4 = a_g * t2_g * (1 - d_g) * poisson.pmf(B_row, eb_g + pb_g) * poisson.pmf(S_row, us_g + es_g + ps_g)
    p5 = a_g * (1 - t2_g) * d_g * poisson.pmf(B_row, ub_g + eb_g) * poisson.pmf(S_row, es_g)
    p6 = a_g * t1_g * d_g * poisson.pmf(B_row, ub_g + eb_g + pb_g) * poisson.pmf(S_row, es_g + ps_g)

    total_lik = p1 + p2 + p3 + p4 + p5 + p6
    total_lik[total_lik <= 0] = 1e-300

    nll_scores = -np.sum(np.log(total_lik), axis=1)
    best_idx = np.argmin(nll_scores)
    best_initial_guess = grid_combinations[best_idx]

    def negative_log_likelihood(params: Sequence[float]) -> float:
        a, d, t1, t2, ub, us, eb, es, pb, ps = params

        if not (0 <= a <= 1 and 0 <= d <= 1 and 0 <= t1 <= 1 and 0 <= t2 <= 1):
            return np.inf
        if any(p < 0 for p in [ub, us, eb, es, pb, ps]):
            return np.inf

        epsilon = 1e-300

        prob1 = (1 - a) * (1 - t1) * poisson.pmf(B, eb) * poisson.pmf(S, es)
        prob2 = (1 - a) * t1 * poisson.pmf(B, eb + pb) * poisson.pmf(S, es + ps)
        prob3 = a * (1 - t2) * (1 - d) * poisson.pmf(B, eb) * poisson.pmf(S, us + es)
        prob4 = a * t2 * (1 - d) * poisson.pmf(B, eb + pb) * poisson.pmf(S, us + es + ps)
        prob5 = a * (1 - t2) * d * poisson.pmf(B, ub + eb) * poisson.pmf(S, es)
        prob6 = a * t1 * d * poisson.pmf(B, ub + eb + pb) * poisson.pmf(S, es + ps)

        lik = prob1 + prob2 + prob3 + prob4 + prob5 + prob6
        return float(-np.sum(np.log(lik + epsilon)))

    try:
        bounds = [(0, 1)] * 4 + [(0, None)] * 6
        best_result = minimize(
            negative_log_likelihood,
            x0=best_initial_guess,
            bounds=bounds,
            method="L-BFGS-B",
        )
    except Exception:
        return {"converged": False}

    if not best_result.success:
        return {"converged": False}

    a, d, t1, t2, ub, us, eb, es, pb, ps = best_result.x

    denom = (a * (d * ub + (1 - d) * us)) + \
            ((pb + ps) * (a * t2 + (1 - a) * t1)) + \
            es + eb
    if denom == 0:
        denom = 1e-9

    num_apin = a * (d * ub + (1 - d) * us)
    apin = num_apin / denom

    num_psos = (pb + ps) * (a * t2 + (1 - a) * t1)
    psos = num_psos / denom

    return {
        "a": a,
        "d": d,
        "t1": t1,
        "t2": t2,
        "ub": ub,
        "us": us,
        "eb": eb,
        "es": es,
        "pb": pb,
        "ps": ps,
        "APIN": apin,
        "PSOS": psos,
        "converged": True,
    }


# =============================================================================
# 모델별 grid/schema
# =============================================================================


def build_grid(model: str) -> np.ndarray:
    model = normalize_model(model)
    if model == "PIN":
        grid_a = [0.1, 0.5, 0.9]
        grid_d = [0.1, 0.5, 0.9]
        grid_u = [20, 200, 2000]
        grid_eb = [20, 200, 2000]
        grid_es = [20, 200, 2000]
        grid_iter = itertools.product(grid_a, grid_d, grid_u, grid_eb, grid_es)
        return np.array(list(grid_iter), dtype=np.float64)

    probs = [0.1, 0.5, 0.9]
    rates = [20, 200, 2000]
    grid_iter = itertools.product(
        probs, probs, probs, probs,
        rates, rates, rates, rates, rates, rates,
    )
    return np.array(list(grid_iter), dtype=np.float64)


def result_schema(model: str) -> Dict[str, pl.DataType]:
    model = normalize_model(model)
    base: Dict[str, pl.DataType] = {
        "Symbol": pl.String,
        "Date": pl.Date,
        "B": pl.UInt64,
        "S": pl.UInt64,
        "Week": pl.Date,
    }
    if model == "PIN":
        base.update({
            "a": pl.Float64,
            "d": pl.Float64,
            "u": pl.Float64,
            "eb": pl.Float64,
            "es": pl.Float64,
            "PIN": pl.Float64,
        })
    else:
        base.update({
            "APIN": pl.Float64,
            "PSOS": pl.Float64,
            "a": pl.Float64,
            "d": pl.Float64,
            "t1": pl.Float64,
            "t2": pl.Float64,
            "ub": pl.Float64,
            "us": pl.Float64,
            "eb": pl.Float64,
            "es": pl.Float64,
            "pb": pl.Float64,
            "ps": pl.Float64,
        })
    return base


def result_columns(model: str) -> List[Any]:
    model = normalize_model(model)
    if model == "PIN":
        return ["Symbol", "Date", "B", "S", "Week", "a", "d", "u", "eb", "es", "PIN"]
    return ["Symbol", "Date", "B", "S", "Week", "APIN", "PSOS", "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps"]


def metric_columns(model: str) -> List[str]:
    model = normalize_model(model)
    return ["PIN"] if model == "PIN" else ["APIN", "PSOS"]


# =============================================================================
# Step 2 준비: 종목 목록/데이터 로드
# =============================================================================


def get_symbols_from_daily_counts(daily_count_files: List[str]) -> List[str]:
    if not daily_count_files:
        return []
    symbols = (
        pl.scan_parquet(daily_count_files)
        .select(pl.col("Symbol").cast(pl.String))
        .unique()
        .collect(streaming=True)
        .get_column("Symbol")
        .drop_nulls()
        .to_list()
    )
    return sorted(str(s) for s in symbols)


def load_daily_for_symbol(symbol: str, daily_count_files: List[str]) -> pl.DataFrame:
    """
    파일별 일별 캐시를 종목 단위로 읽는다.
    같은 Symbol-Date가 여러 캐시에 나뉘어 있을 수 있으므로 다시 합산한다.
    """
    lf = (
        pl.scan_parquet(daily_count_files)
        .filter(pl.col("Symbol") == symbol)
        .select(["Symbol", "Date", "B", "S"])
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("B").sum().alias("B"),
            pl.col("S").sum().alias("S"),
        ])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))
        .with_columns(pl.col("Date").dt.truncate("1w").alias("Week_Start"))
        .select([
            pl.col("Symbol").cast(pl.String),
            pl.col("Date").cast(pl.Date),
            pl.col("B").cast(pl.UInt64),
            pl.col("S").cast(pl.UInt64),
            pl.col("Week_Start").cast(pl.Date),
        ])
        .sort("Date")
    )
    return lf.collect(streaming=True)


def is_valid_checkpoint_file(path: str, model: str) -> bool:
    if not os.path.exists(path):
        return False
    try:
        cols = result_columns(model)
        (
            pl.scan_parquet(path)
            .select(cols)
            .head(1)
            .collect(streaming=True)
        )
        return True
    except Exception:
        return False


def get_completed_symbols(model: str) -> set:
    model = normalize_model(model)
    completed = set()
    cdir = checkpoint_dir(model)
    for f in glob.glob(os.path.join(cdir, "sym_*.parquet")):
        if not is_valid_checkpoint_file(f, model):
            continue
        name = os.path.splitext(os.path.basename(f))[0]
        completed.add(name[4:] if name.startswith("sym_") else name)
    return completed


# =============================================================================
# Step 2: 종목별 weekly PIN/APIN 계산 + 체크포인트
# =============================================================================

GLOBAL_DAILY_FILES: Optional[List[str]] = None
GLOBAL_GRID: Optional[np.ndarray] = None
GLOBAL_MODEL: Optional[str] = None


def init_worker(daily_files: List[str], grid: np.ndarray, model: str) -> None:
    global GLOBAL_DAILY_FILES, GLOBAL_GRID, GLOBAL_MODEL
    GLOBAL_DAILY_FILES = daily_files
    GLOBAL_GRID = grid
    GLOBAL_MODEL = normalize_model(model)


def estimate_for_week(model: str, B: np.ndarray, S: np.ndarray, grid: np.ndarray) -> Dict[str, Any]:
    model = normalize_model(model)
    if model == "PIN":
        return estimate_pin_parameters(B, S, grid)
    return estimate_apin_parameters(B, S, grid)


def calculate_symbol_model_result(sym_df: pl.DataFrame, symbol: str, model: str, grid: np.ndarray) -> Tuple[pl.DataFrame, Dict[str, int]]:
    """
    원본 process_single_symbol의 4주 rolling window 추정 로직을 유지하되,
    종목별 최종 daily 결과를 바로 만들어 checkpoint로 저장할 수 있게 한다.
    """
    model = normalize_model(model)
    schema = result_schema(model)

    if sym_df.height < MIN_OBS_DAYS:
        return pl.DataFrame(schema=schema), {"n_daily_rows": sym_df.height, "n_weeks": 0, "n_estimated_weeks": 0}

    results: List[Dict[str, Any]] = []
    unique_weeks = sym_df["Week_Start"].unique().sort().to_list()

    dates = sym_df["Date"].to_numpy()
    Bs = sym_df["B"].to_numpy()
    Ss = sym_df["S"].to_numpy()

    n_estimated = 0
    for current_week_start in unique_weeks:
        window_end = current_week_start + timedelta(days=6)
        window_start = current_week_start - timedelta(weeks=ROLLING_LOOKBACK_WEEKS)

        mask = (dates >= window_start) & (dates <= window_end)
        if int(np.sum(mask)) <= MIN_OBS_DAYS:
            continue

        est = estimate_for_week(model, Bs[mask], Ss[mask], grid)
        if not est.get("converged", False):
            continue

        if model == "PIN":
            results.append({
                "Symbol": symbol,
                "Week_Start": current_week_start,
                "a": est["alpha"],
                "d": est["delta"],
                "u": est["mu"],
                "eb": est["eb"],
                "es": est["es"],
                "PIN": est["PIN"],
            })
        else:
            row = {"Symbol": symbol, "Week_Start": current_week_start}
            row.update(est)
            row.pop("converged", None)
            results.append(row)
        n_estimated += 1

    if not results:
        return pl.DataFrame(schema=schema), {
            "n_daily_rows": sym_df.height,
            "n_weeks": len(unique_weeks),
            "n_estimated_weeks": 0,
        }

    week_params_df = pl.DataFrame(results)
    final_df = (
        sym_df.join(week_params_df, on=["Symbol", "Week_Start"], how="inner")
        .sort(["Symbol", "Date"])
    )

    if model == "PIN":
        selected = final_df.select([
            "Symbol", "Date", "B", "S",
            pl.col("Week_Start").alias("Week"),
            "a", "d", "u", "eb", "es", "PIN",
        ])
    else:
        selected = final_df.select([
            "Symbol", "Date", "B", "S",
            pl.col("Week_Start").alias("Week"),
            "APIN", "PSOS",
            "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
        ])

    return selected, {
        "n_daily_rows": sym_df.height,
        "n_weeks": len(unique_weeks),
        "n_estimated_weeks": n_estimated,
    }


def worker_process_symbol(args: Tuple[str, str, str]) -> Dict[str, Any]:
    symbol, ckp_path, meta_path = args

    try:
        if GLOBAL_DAILY_FILES is None or GLOBAL_GRID is None or GLOBAL_MODEL is None:
            raise RuntimeError("Worker global variables are not initialized.")

        sym_df = load_daily_for_symbol(symbol, GLOBAL_DAILY_FILES)
        result_df, stats = calculate_symbol_model_result(sym_df, symbol, GLOBAL_MODEL, GLOBAL_GRID)

        write_parquet_atomic(result_df, ckp_path)
        write_json_atomic({
            "symbol": symbol,
            "model": GLOBAL_MODEL,
            "status": "success" if result_df.height > 0 else "empty_result",
            "checkpoint_path": ckp_path,
            "n_output_rows": result_df.height,
            **stats,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }, meta_path)

        return {
            "symbol": symbol,
            "status": "success" if result_df.height > 0 else "empty_result",
            "n_rows": result_df.height,
            **stats,
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "status": "error",
            "n_rows": 0,
            "error": f"{e}\n{traceback.format_exc()}",
        }


def run_step2_model(model: str, daily_count_files: List[str]) -> Tuple[List[str], int, List[Tuple[str, str]]]:
    model = normalize_model(model)
    cdir = checkpoint_dir(model)
    mdir = meta_dir(model)
    os.makedirs(cdir, exist_ok=True)
    os.makedirs(mdir, exist_ok=True)

    if FORCE_RECOMPUTE_MODEL:
        if os.path.isdir(cdir):
            print(f"\n[Step 2:{model}] 기존 체크포인트 삭제: {cdir}")
            shutil.rmtree(cdir)
        if os.path.isdir(mdir):
            print(f"[Step 2:{model}] 기존 meta 삭제: {mdir}")
            shutil.rmtree(mdir)
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(mdir, exist_ok=True)

    print(f"\n[Step 2 준비:{model}] Symbol 목록과 Grid 생성...")
    all_symbols = get_symbols_from_daily_counts(daily_count_files)
    n_total_symbols = len(all_symbols)
    completed = get_completed_symbols(model)
    grid = build_grid(model)

    tasks: List[Tuple[str, str, str]] = []
    used_checkpoint_names: Dict[str, str] = {}

    for symbol in all_symbols:
        checkpoint_symbol_name = safe_filename(symbol)
        base_name = checkpoint_symbol_name
        i = 1
        while (
            checkpoint_symbol_name.lower() in used_checkpoint_names
            and used_checkpoint_names[checkpoint_symbol_name.lower()] != symbol
        ):
            checkpoint_symbol_name = f"{base_name}__{i}"
            i += 1
        used_checkpoint_names[checkpoint_symbol_name.lower()] = symbol

        if checkpoint_symbol_name in completed:
            continue

        ckp_path = os.path.join(cdir, f"sym_{checkpoint_symbol_name}.parquet")
        mpath = os.path.join(mdir, f"sym_{checkpoint_symbol_name}.json")
        tasks.append((symbol, ckp_path, mpath))

    n_remaining = len(tasks)
    n_completed = n_total_symbols - n_remaining

    print(f"\n{'='*65}")
    print(f"[Step 2] 종목별 weekly {model} 계산 [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  전체 종목         : {n_total_symbols:,}개")
    print(f"  기완료 (재개)     : {n_completed:,}개")
    print(f"  처리 예정         : {n_remaining:,}개")
    print(f"  체크포인트 폴더   : {cdir}")
    print(f"  Grid size         : {grid.shape[0]:,} combinations")
    print(f"  Window            : target week + past {ROLLING_LOOKBACK_WEEKS} weeks, obs > {MIN_OBS_DAYS}")

    if n_remaining == 0:
        return sorted(glob.glob(os.path.join(cdir, "sym_*.parquet"))), n_total_symbols, []

    n_workers = NUM_WORKERS if NUM_WORKERS is not None else max(1, multiprocessing.cpu_count() - 1)
    n_workers = max(1, min(int(n_workers), n_remaining))
    print(f"  워커 수           : {n_workers}개")
    print(f"{'-'*65}\n")

    start = datetime.now()
    success_count = empty_count = error_count = 0
    error_log: List[Tuple[str, str]] = []

    with multiprocessing.Pool(
        processes=n_workers,
        initializer=init_worker,
        initargs=(daily_count_files, grid, model),
    ) as pool:
        iterator = pool.imap_unordered(worker_process_symbol, tasks)
        for res in tqdm(iterator, total=n_remaining, desc=f"{model} 계산/재개"):
            if res is None:
                continue
            status = res.get("status")
            if status == "success":
                success_count += 1
            elif status == "empty_result":
                empty_count += 1
            elif status == "error":
                error_count += 1
                error_log.append((str(res.get("symbol")), str(res.get("error", ""))))

    elapsed = datetime.now() - start
    print(f"\n  [Step 2:{model} 완료] 소요 시간: {elapsed}")
    print(f"    성공       : {success_count:,}")
    print(f"    빈 결과    : {empty_count:,}")
    print(f"    에러       : {error_count:,}")

    if error_log:
        print("\n  [에러 발생 종목 목록 (최대 10개)]")
        for sym, msg in error_log[:10]:
            first_line = msg.splitlines()[0] if msg else ""
            print(f"    - {sym}: {first_line}")
        if FAIL_ON_STEP2_ERROR:
            raise RuntimeError(f"Step 2:{model}에서 에러가 발생했습니다. 첫 에러: {error_log[0]}")

    return sorted(glob.glob(os.path.join(cdir, "sym_*.parquet"))), n_total_symbols, error_log


# =============================================================================
# Step 3: checkpoint 병합
# =============================================================================


def run_step3_model(model: str, checkpoint_files: List[str]) -> Tuple[Optional[str], Optional[str]]:
    model = normalize_model(model)
    print(f"\n{'='*65}")
    print(f"[Step 3] {model} 체크포인트 lazy 병합 → 최종 결과 [{COUNTRY}]")
    print(f"{'='*65}")
    print(f"  체크포인트 파일 수: {len(checkpoint_files):,}")

    valid_files = [f for f in checkpoint_files if is_valid_checkpoint_file(f, model)]
    if not valid_files:
        print("[Warning] 병합할 정상 체크포인트가 없습니다.")
        return None, None

    os.makedirs(model_output_root(model), exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    stem = f"{model.lower()}_{COUNTRY}_weekly_checkpoint_{run_id}"
    parquet_path = os.path.join(model_output_root(model), f"{stem}.parquet")
    csv_path = os.path.join(model_output_root(model), f"{stem}_sample1000.csv")

    lf = (
        pl.scan_parquet(valid_files)
        .select(result_columns(model))
        .sort(["Symbol", "Date"])
    )
    try:
        lf.sink_parquet(parquet_path, compression="zstd")
    except TypeError:
        lf.collect(streaming=True).write_parquet(parquet_path, compression="zstd")

    sample_df = pl.scan_parquet(parquet_path).head(1000).collect(streaming=True)
    sample_df.write_csv(csv_path)

    summary = pl.scan_parquet(parquet_path).select([
        pl.len().alias("n_rows"),
        pl.col("Symbol").n_unique().alias("n_symbols"),
        pl.col("Date").min().alias("date_min"),
        pl.col("Date").max().alias("date_max"),
    ]).collect(streaming=True).row(0, named=True)

    print("\n  [Step 3 완료]")
    print(f"    전체 parquet: {parquet_path}")
    print(f"    샘플 CSV    : {csv_path}")
    print(f"    전체 행 수  : {summary['n_rows']:,}")
    print(f"    종목 수     : {summary['n_symbols']:,}")
    print(f"    기간        : {summary['date_min']} ~ {summary['date_max']}")
    return parquet_path, stem


# =============================================================================
# Step 4: 검증 리포트
# =============================================================================


def run_step4_report(
    model: str,
    final_parquet_path: str,
    stem: str,
    n_symbols_input: int,
    checkpoint_files: List[str],
    error_log: List[Tuple[str, str]],
) -> str:
    model = normalize_model(model)
    print(f"\n{'='*65}")
    print(f"[Step 4] {model} 검증 리포트 생성 [{COUNTRY}]")
    print(f"{'='*65}")

    report_path = os.path.join(model_output_root(model), f"{stem}_report.txt")
    lf = pl.scan_parquet(final_parquet_path)
    metrics = metric_columns(model)
    primary_metric = metrics[0]
    valid = pl.col(primary_metric).is_not_null() & pl.col(primary_metric).is_not_nan()

    basic = lf.select([
        pl.len().alias("n_total_rows"),
        pl.col("Symbol").n_unique().alias("n_symbols_out"),
        pl.col("Week").n_unique().alias("n_weeks_out"),
        valid.sum().alias("n_valid_primary_metric"),
        pl.col("Date").min().alias("date_min"),
        pl.col("Date").max().alias("date_max"),
    ]).collect(streaming=True).row(0, named=True)

    n_total_rows = int(basic["n_total_rows"] or 0)
    n_valid = int(basic["n_valid_primary_metric"] or 0)
    n_nan = n_total_rows - n_valid

    dist_lines: List[str] = []
    range_lines: List[str] = []
    for m in metrics:
        m_valid = pl.col(m).is_not_null() & pl.col(m).is_not_nan()
        m_count = lf.filter(m_valid).select(pl.len()).collect(streaming=True).item()
        if m_count > 0:
            stats = lf.filter(m_valid).select([
                pl.col(m).min().alias("min"),
                pl.col(m).quantile(0.01).alias("p01"),
                pl.col(m).median().alias("median"),
                pl.col(m).mean().alias("mean"),
                pl.col(m).quantile(0.99).alias("p99"),
                pl.col(m).max().alias("max"),
                pl.col(m).std().alias("std"),
            ]).collect(streaming=True).row(0, named=True)
            dist_lines.append(f"  [{m}]")
            for k in ["min", "p01", "median", "mean", "p99", "max", "std"]:
                dist_lines.append(f"    {k:<8}: {_fmt_num(stats.get(k), 6)}")
            out_of_range = lf.filter(m_valid & ((pl.col(m) < 0) | (pl.col(m) > 1))).select(pl.len()).collect(streaming=True).item()
            n_extreme = lf.filter(m_valid & (pl.col(m) > REPORT_EXTREME_THRESHOLD)).select(pl.len()).collect(streaming=True).item()
            extreme_ratio = n_extreme / m_count if m_count else 0.0
            range_lines.append(f"  {m} 범위 이탈 [0,1]      : {out_of_range:,}건")
            range_lines.append(f"  {m} > {REPORT_EXTREME_THRESHOLD:.2f}: {n_extreme:,}건 ({extreme_ratio*100:.3f}%)")
        else:
            dist_lines.append(f"  [{m}] 유효 값이 없습니다.")
            range_lines.append(f"  {m} 유효 값 없음")

    symbol_integrity = (
        lf.group_by("Symbol")
        .agg([
            pl.len().alias("n_rows"),
            pl.col("Date").min().alias("date_min"),
            pl.col("Date").max().alias("date_max"),
            pl.col("Week").n_unique().alias("n_weeks"),
        ])
        .sort("Symbol")
        .collect(streaming=True)
    )

    lines: List[str] = []
    add = lines.append
    add("=" * 75)
    add(f"[{model} Weekly 체크포인트 검증 리포트] {COUNTRY}")
    add(f"생성 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"대상 파일: {stem}.parquet")
    add("=" * 75)
    add("")
    add("[A] 기본 요약")
    add(f"  전체 종목 수(일별 캐시): {n_symbols_input:,}")
    add(f"  결과 종목 수          : {basic['n_symbols_out']:,}")
    add(f"  결과 Week 수          : {basic['n_weeks_out']:,}")
    add(f"  체크포인트 수         : {len(checkpoint_files):,}")
    add(f"  전체 결과 행 수       : {n_total_rows:,}")
    add(f"  유효 {primary_metric} 수       : {n_valid:,}")
    add(f"  NaN/Null {primary_metric} 수   : {n_nan:,}")
    add(f"  기간                  : {basic['date_min']} ~ {basic['date_max']}")
    add(f"  추정 window           : target week + past {ROLLING_LOOKBACK_WEEKS} weeks, obs > {MIN_OBS_DAYS}")
    add("")
    add("[B] 분포")
    add("\n".join(dist_lines))
    add("")
    add("[C] 정합성")
    add("\n".join(range_lines))
    add(f"  Step 2 오류 종목 수    : {len(error_log):,}")
    if error_log:
        add("  오류 예시:")
        for sym, msg in error_log[:10]:
            first = msg.splitlines()[0] if msg else ""
            add(f"    - {sym}: {first}")
    add("")
    add("[D] 종목별 결과 요약 예시")
    if symbol_integrity.height > 0:
        for row in symbol_integrity.head(10).iter_rows(named=True):
            add(
                f"  {row['Symbol']}: rows={row['n_rows']:,}, "
                f"weeks={row['n_weeks']:,}, {row['date_min']}~{row['date_max']}"
            )
    else:
        add("  결과 종목이 없습니다.")
    add("")
    add("[E] 저장 위치")
    add(f"  output root : {model_output_root(model)}")
    add(f"  checkpoints : {checkpoint_dir(model)}")
    add(f"  meta        : {meta_dir(model)}")
    add("")
    add("=" * 75)
    add("[리포트 끝]")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"  [Step 4 완료] 리포트: {report_path}")
    return report_path


# =============================================================================
# 실행부
# =============================================================================


def main() -> None:
    multiprocessing.freeze_support()

    global COUNTRY
    COUNTRY = COUNTRY.upper().strip()
    if COUNTRY not in VALID_COUNTRIES:
        raise ValueError(f"COUNTRY 설정 오류: {COUNTRY}. 사용 가능: {sorted(VALID_COUNTRIES)}")

    models = [normalize_model(m) for m in MODELS_TO_RUN]
    if not models:
        raise ValueError("MODELS_TO_RUN이 비어 있습니다. 예: ['PIN'] 또는 ['PIN', 'APIN']")

    start = datetime.now()
    print(f"\n{'='*65}")
    print(f"[PIN/APIN Stateful-style 파이프라인] {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  나라코드        : {COUNTRY}")
    print(f"  입력 루트       : {BASE_DIR}")
    print(f"  직접 입력 파일  : {len(INPUT_FILES):,}개" if INPUT_FILES else "  직접 입력 파일  : 사용 안 함")
    print(f"  출력 루트       : {output_root()}")
    print(f"  일별 B/S 캐시   : {daily_counts_dir()}")
    print(f"  실행 모델       : {models}")
    print(f"  워커 수 설정    : {NUM_WORKERS}")
    print(f"  Window          : target week + past {ROLLING_LOOKBACK_WEEKS} weeks, obs > {MIN_OBS_DAYS}")
    print(f"{'='*65}")

    if not INPUT_FILES and not os.path.isdir(BASE_DIR):
        print(f"[Error] BASE_DIR이 없습니다: {BASE_DIR}")
        sys.exit(1)

    daily_files = run_step1(BASE_DIR, COUNTRY)
    if not daily_files:
        print("[Error] Step 1에서 일별 B/S 캐시를 생성하지 못했습니다.")
        sys.exit(1)

    final_outputs = []
    for model in models:
        checkpoint_files, n_symbols_input, error_log = run_step2_model(model, daily_files)
        if not checkpoint_files:
            print(f"[Warning] Step 2:{model}에서 체크포인트가 생성되지 않았습니다. 다음 모델로 넘어갑니다.")
            continue

        final_parquet_path, stem = run_step3_model(model, checkpoint_files)
        if final_parquet_path is None or stem is None:
            print(f"[Warning] Step 3:{model}에서 최종 결과가 생성되지 않았습니다. 다음 모델로 넘어갑니다.")
            continue

        report_path = run_step4_report(
            model,
            final_parquet_path,
            stem,
            n_symbols_input=n_symbols_input,
            checkpoint_files=checkpoint_files,
            error_log=error_log,
        )
        final_outputs.append((model, final_parquet_path, report_path))

    elapsed = datetime.now() - start
    print(f"\n{'='*65}")
    print(f"[파이프라인 완료] 소요 시간: {elapsed}")
    for model, parquet_path, report_path in final_outputs:
        print(f"  [{model}] 최종 parquet: {parquet_path}")
        print(f"  [{model}] 리포트      : {report_path}")
    if not final_outputs:
        print("  생성된 최종 결과가 없습니다.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
