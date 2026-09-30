"""
=============================================================================
일별 롤링 윈도우 PIN 계산 (Lookback Buffer 방식)
=============================================================================

[핵심 개념]
- 20영업일 롤링 윈도우로 매일 PIN 계산
- 파일 간 연속성 보장을 위한 Lookback Buffer 활용
- 종목별 독립적인 시계열 연속성 유지
- *** [개선] 연도(KOR_YYYY) 처리 완료 시마다 중간 저장 → 중단 후 재개 가능 ***

[버퍼 전략: 직전 2개 파일 롤링 방식]

  문제 상황:
    10월 파일의 거래일이 16일 → 버퍼로 19일 확보 불가
    → 11월 월초 3일의 PIN 계산 누락 발생

  해결 전략: 직전 2개 파일(전달 + 전전달) 원본 데이터 전체를 버퍼로 유지
    lookback_buffer = [전전달 전체] + [전달 전체]

  보장 근거:
    한국 주식시장 월 최소 거래일 ≈ 14~15일 (추석·설 등 연휴 최대치)
    2개월 합산 ≥ 28일 이상 → 19일 보장 항상 충족
    극단적 단월(10일 수준)도 2개월 합산으로 안전하게 커버

  메모리 영향:
    월별 파일 종목×거래일 행 수 기준 2개 파일치 원본 보유
    일별 집계 후 데이터이므로 실제 크기는 매우 작음 (수천~수만 행)
    → 실질적 메모리 부담 없음

[Date 타입 보존 전략]
  원본 Date: Polars pl.Date 타입
  numpy 변환 시 pl.Date → epoch days(int32) 정수로 변환하여 비교
  결과 저장 시 Date_int(int)로 보관 → build_params_df()에서 pl.Date로 복원
  → join 시 타입 완전 일치 보장

[중간 저장 구조]
  output_data/
    ├── intermediate/
    │   ├── pin_daily_KOR_2017_partial.parquet
    │   ├── pin_daily_KOR_2018_partial.parquet
    │   └── ...
    └── pin_daily_rolling_2017_2018_YYYYMMDD_HHMM.parquet

=============================================================================
"""

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
import os
from datetime import datetime
import multiprocessing
import itertools
from tqdm import tqdm
import glob

warnings.filterwarnings('ignore')


# =============================================================================
# 데이터 전처리
# =============================================================================

def preprocess_trade_data_polars(parquet_path):
    """
    Parquet 파일에서 거래 데이터를 로드하고 일별 매수(B)/매도(S) 건수를 집계합니다.
    Date 컬럼은 pl.Date 타입으로 보존됩니다.
    """
    print(f"Loading data from {os.path.basename(parquet_path)}...")
    q = pl.scan_parquet(parquet_path)

    aggregated_df = (
        q.select(['Symbol', 'Date', 'LR'])
        .filter(pl.col('LR').is_in([1, -1]))
        .with_columns([
            (pl.col('LR') == 1).cast(pl.UInt32).alias('is_Buy'),
            (pl.col('LR') == -1).cast(pl.UInt32).alias('is_Sell')
        ])
        .group_by(['Symbol', 'Date'])
        .agg([
            pl.col('is_Buy').sum().alias('B'),
            pl.col('is_Sell').sum().alias('S')
        ])
        .filter((pl.col('B') > 0) | (pl.col('S') > 0))
        .sort(['Symbol', 'Date'])
    ).collect()

    # Date 컬럼 타입 보정
    if aggregated_df['Date'].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(pl.col('Date').cast(pl.Date))

    return aggregated_df


# =============================================================================
# PIN MLE 추정
# =============================================================================

def estimate_pin_parameters(B_array, S_array, grid_combinations):
    """
    EKOP(1996) 모델 기반 MLE로 파라미터(alpha, delta, mu, epsilon_b, epsilon_s)를 추정합니다.
    """
    B = B_array
    S = S_array

    a_g  = grid_combinations[:, 0:1]
    d_g  = grid_combinations[:, 1:2]
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

    def negative_log_likelihood(params):
        alpha, delta, mu, eb, es = params
        if not (0 <= alpha <= 1 and 0 <= delta <= 1 and mu >= 0 and eb >= 0 and es >= 0):
            return np.inf
        epsilon_float = 1e-300
        prob_no_news   = poisson.pmf(B, eb) * poisson.pmf(S, es)
        prob_bad_news  = poisson.pmf(B, eb) * poisson.pmf(S, mu + es)
        prob_good_news = poisson.pmf(B, mu + eb) * poisson.pmf(S, es)
        likelihood = (1 - alpha) * prob_no_news + \
                     (alpha * delta) * prob_bad_news + \
                     (alpha * (1 - delta)) * prob_good_news
        log_lik = np.log(likelihood + epsilon_float)
        return -np.sum(log_lik)

    try:
        best_result = minimize(
            negative_log_likelihood,
            x0=best_initial_guess,
            bounds=[(0, 1), (0, 1), (0, None), (0, None), (0, None)],
            method='L-BFGS-B'
        )
    except Exception:
        return {'converged': False}

    if not best_result.success:
        return {'converged': False}

    alpha, delta, mu, eb, es = best_result.x
    denom = (alpha * mu + eb + es)
    if denom == 0:
        denom = 1e-9
    pin = (alpha * mu) / denom

    return {
        'alpha': alpha, 'delta': delta, 'mu': mu,
        'eb': eb, 'es': es, 'PIN': pin, 'converged': True
    }


# =============================================================================
# Multiprocessing
# =============================================================================

GLOBAL_GRID = None

def init_worker(grid):
    global GLOBAL_GRID
    GLOBAL_GRID = grid


def process_single_symbol(data_tuple):
    """
    개별 종목의 데이터를 받아 20일 롤링 윈도우로 모든 거래일의 PIN을 계산합니다.

    [Date 타입 처리]
    - sym_df['Date'].cast(pl.Int32).to_numpy() : epoch days 정수 배열로 변환
    - current_period_start_int : 정수로 전달받아 직접 비교
    - 결과 딕셔너리에 'Date_int'(int) 저장 → build_params_df()에서 pl.Date 복원
    """
    sym, sym_df, current_period_start_int = data_tuple

    if sym_df.height < 20:
        return []

    results = []
    Bs        = sym_df['B'].to_numpy()
    Ss        = sym_df['S'].to_numpy()
    dates_int = sym_df['Date'].cast(pl.Int32).to_numpy()  # epoch days 정수 배열

    start_idx = 19 if len(dates_int) >= 20 else len(dates_int)

    for i in range(start_idx, len(dates_int)):
        current_date_int = dates_int[i]

        # lookback buffer 영역(current_period 이전)은 계산에만 사용, 결과 제외
        if current_date_int < current_period_start_int:
            continue

        window_B = Bs[i - 19 : i + 1]
        window_S = Ss[i - 19 : i + 1]

        if len(window_B) != 20:
            continue

        est = estimate_pin_parameters(window_B, window_S, GLOBAL_GRID)

        if est['converged']:
            results.append({
                'Symbol':   sym,
                'Date_int': int(current_date_int),  # epoch days 정수로 저장
                'a':        est['alpha'],
                'd':        est['delta'],
                'u':        est['mu'],
                'eb':       est['eb'],
                'es':       est['es'],
                'PIN':      est['PIN']
            })

    return results


# =============================================================================
# 버퍼 관리 (핵심 개선)
# =============================================================================

def update_rolling_file_buffer(rolling_file_buffer, current_df, max_files=2):
    """
    *** [핵심 개선] 직전 N개 파일 원본 데이터를 롤링으로 유지하는 버퍼 업데이트 ***

    [동작 원리]
    rolling_file_buffer는 최근 max_files(=2)개의 파일 원본 DataFrame을 담는 deque입니다.
    새 파일이 추가되면 가장 오래된 파일을 제거하여 항상 최대 2개만 유지합니다.

    버퍼 크기가 2인 이유:
    - 월 최소 거래일 ≈ 14~15일 (명절 집중 월 기준)
    - 1개월치만 보관 시 해당 월이 16일 미만이면 19일 미확보 가능
    - 2개월치 보관 시 합산 ≥ 28일 이상 → 19일 항상 보장
    - 메모리 비용: 일별 집계 데이터 2개월치 ≈ 수천~수만 행 수준으로 무시 가능

    Parameters:
    rolling_file_buffer (list): 이전 파일들의 DataFrame 리스트 (최대 max_files개)
    current_df (pl.DataFrame):  방금 처리 완료된 파일의 일별 집계 데이터
    max_files (int):            유지할 최대 파일 수 (기본값 2)

    Returns:
    list: 업데이트된 rolling_file_buffer
    """
    rolling_file_buffer.append(current_df)
    # max_files 초과 시 가장 오래된 파일 제거
    if len(rolling_file_buffer) > max_files:
        rolling_file_buffer.pop(0)
    return rolling_file_buffer


def build_lookback_buffer_from_rolling(rolling_file_buffer, buffer_days=19):
    """
    *** [핵심 개선] 롤링 파일 버퍼에서 종목별 최근 N일을 추출하여 lookback buffer 생성 ***

    [동작 원리]
    rolling_file_buffer에 담긴 최대 2개 파일의 데이터를 concat한 뒤
    종목별로 마지막 buffer_days(=19)일을 추출합니다.

    예시 (10월이 16거래일인 경우):
      rolling_file_buffer = [9월 전체(21일), 10월 전체(16일)]
      concat 후 종목별 정렬 → tail(19) 추출
      → 9월 말 3일 + 10월 전체 16일 = 19일 확보 ✓

    Parameters:
    rolling_file_buffer (list): 이전 파일 DataFrame 리스트 (최대 2개)
    buffer_days (int):          추출할 일 수 (기본값 19)

    Returns:
    pl.DataFrame: 종목별 최근 N일 데이터 (Symbol, Date, B, S), pl.Date 타입 보존
    """
    if not rolling_file_buffer:
        return pl.DataFrame()

    # 버퍼 내 모든 파일 데이터 합산
    combined = (
        pl.concat(rolling_file_buffer, how='vertical')
        .sort(['Symbol', 'Date'])
    )

    # 종목별 마지막 N일 추출
    buffer_df = (
        combined
        .group_by('Symbol', maintain_order=True)
        .tail(buffer_days)
        .sort(['Symbol', 'Date'])
    )

    return buffer_df


# =============================================================================
# 결과 처리
# =============================================================================

def build_params_df(estimates):
    """
    추정 결과 리스트 → Polars DataFrame 변환 (Date 타입 복원)

    Date_int(int32 epoch days) → pl.Date 타입으로 복원하여
    원본 DataFrame과 join 시 타입 불일치 방지
    """
    if not estimates:
        return pl.DataFrame()

    params_df = (
        pl.DataFrame(estimates)
        .with_columns(
            pl.col('Date_int').cast(pl.Date).alias('Date')
        )
        .drop('Date_int')
        .select(['Symbol', 'Date', 'a', 'd', 'u', 'eb', 'es', 'PIN'])
    )
    return params_df


def save_yearly_intermediate_result(year_estimates, year_daily_df, intermediate_dir, year_folder):
    """
    한 연도(KOR_YYYY) 전체 처리 완료 후 결과를 intermediate 디렉토리에 저장합니다.

    [Date 타입 흐름]
    year_estimates의 'Date_int' → build_params_df()에서 pl.Date로 복원
    → year_daily_df의 'Date'(pl.Date)와 join → 타입 완전 일치
    → 저장된 parquet의 Date 컬럼도 pl.Date 타입 유지
    """
    os.makedirs(intermediate_dir, exist_ok=True)

    if not year_estimates:
        partial_df = year_daily_df.with_columns([
            pl.lit(None).cast(pl.Float64).alias('a'),
            pl.lit(None).cast(pl.Float64).alias('d'),
            pl.lit(None).cast(pl.Float64).alias('u'),
            pl.lit(None).cast(pl.Float64).alias('eb'),
            pl.lit(None).cast(pl.Float64).alias('es'),
            pl.lit(None).cast(pl.Float64).alias('PIN'),
        ]).select(['Symbol', 'Date', 'B', 'S', 'a', 'd', 'u', 'eb', 'es', 'PIN'])
    else:
        params_df = build_params_df(year_estimates)
        partial_df = (
            year_daily_df
            .join(params_df, on=['Symbol', 'Date'], how='left')
            .sort(['Symbol', 'Date'])
            .select(['Symbol', 'Date', 'B', 'S', 'a', 'd', 'u', 'eb', 'es', 'PIN'])
        )

    save_path = os.path.join(intermediate_dir, f"pin_daily_{year_folder}_partial.parquet")
    partial_df.write_parquet(save_path, compression='zstd')

    pin_count   = partial_df.filter(pl.col('PIN').is_not_null()).height
    total_count = partial_df.height
    print(f"\n  [Yearly Intermediate Saved] {os.path.basename(save_path)}")
    print(f"    Records: {total_count:,}  |  PIN estimated: {pin_count:,}  |  No PIN: {total_count - pin_count:,}")

    return save_path


# =============================================================================
# 메인 처리 함수
# =============================================================================

def run_daily_pin_rolling_sequential(parquet_files, output_dir, year_folders):
    """
    여러 parquet 파일을 순차 처리하며 롤링 2파일 lookback buffer와 연도별 중간 저장을 적용합니다.

    [rolling_file_buffer 동작 흐름 예시]

    파일 1 (9월, 21거래일) 처리:
      lookback_buffer = [] (없음, 첫 파일)
      처리 완료 후: rolling_file_buffer = [9월 DF]

    파일 2 (10월, 16거래일) 처리:
      lookback_buffer = build_lookback_buffer_from_rolling([9월 DF])
        → 9월 마지막 19일 추출
      처리 완료 후: rolling_file_buffer = [9월 DF, 10월 DF]

    파일 3 (11월) 처리:  ← 핵심 케이스
      lookback_buffer = build_lookback_buffer_from_rolling([9월 DF, 10월 DF])
        → concat 후 종목별 tail(19)
        → 9월 말 3일 + 10월 전체 16일 = 19일 확보 ✓
      처리 완료 후: rolling_file_buffer = [10월 DF, 11월 DF]  (9월은 제거)

    파일 4 (12월) 처리:
      lookback_buffer = build_lookback_buffer_from_rolling([10월 DF, 11월 DF])
        → 11월이 충분하면 11월 마지막 19일만 사용
        → 11월이 부족하면 10월 말 데이터도 포함하여 19일 확보 ✓
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    print(f"Intermediate results will be saved to: {intermediate_dir}\n")

    # Grid Combinations 생성
    print("Generating Grid Combinations...")
    grid_a  = [0.1, 0.5, 0.9]
    grid_d  = [0.1, 0.5, 0.9]
    grid_u  = [20, 200, 2000]
    grid_eb = [20, 200, 2000]
    grid_es = [20, 200, 2000]

    grid_iter   = itertools.product(grid_a, grid_d, grid_u, grid_eb, grid_es)
    grid_matrix = np.array(list(grid_iter), dtype=np.float64)
    print(f"Grid Size: {grid_matrix.shape[0]} combinations.")

    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing.\n")

    # -------------------------------------------------------------------------
    # parquet_files를 연도(KOR_YYYY) 단위로 그룹핑
    # -------------------------------------------------------------------------
    if year_folders is None:
        seen = []
        for f in parquet_files:
            for p in f.replace('\\', '/').split('/'):
                if p.startswith('KOR_') and p not in seen:
                    seen.append(p)
        year_folders_ordered = seen
    else:
        year_folders_ordered = year_folders

    year_to_files = {yf: [] for yf in year_folders_ordered}
    for f in parquet_files:
        for yf in year_folders_ordered:
            if yf in f.replace('\\', '/'):
                year_to_files[yf].append(f)
                break

    print("[Year → File Mapping]")
    for yf, files in year_to_files.items():
        print(f"  {yf}: {len(files)} files")
    print()

    # -------------------------------------------------------------------------
    # 연도별 순차 처리 루프
    # -------------------------------------------------------------------------
    saved_intermediate_paths = []

    # *** [핵심] 직전 2개 파일 원본 데이터를 유지하는 롤링 버퍼 ***
    # 연도 경계를 넘어서도 연속성 유지 (예: 12월 → 1월)
    rolling_file_buffer = []  # list of pl.DataFrame, 최대 2개

    for year_folder in year_folders_ordered:
        files_in_year = year_to_files.get(year_folder, [])
        if not files_in_year:
            print(f"[Warning] No files found for {year_folder}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Processing year: {year_folder}  ({len(files_in_year)} files)")
        print(f"{'='*70}")

        # 이미 처리된 연도 스킵 (재시작 지원)
        intermediate_path = os.path.join(intermediate_dir, f"pin_daily_{year_folder}_partial.parquet")
        if os.path.exists(intermediate_path):
            print(f"  [Skip] Already processed: {os.path.basename(intermediate_path)}")
            print(f"         Delete it to reprocess this year.")
            saved_intermediate_paths.append(intermediate_path)

            # *** 스킵 시 rolling_file_buffer 복원 ***
            # 이 연도의 마지막 2개 파일을 로드하여 다음 연도 버퍼로 사용
            restore_files = files_in_year[-2:]  # 마지막 2개 파일
            print(f"  Restoring rolling buffer from last {len(restore_files)} file(s)...")
            rolling_file_buffer = []
            for rf in restore_files:
                buf_df = preprocess_trade_data_polars(rf)
                if not buf_df.is_empty():
                    rolling_file_buffer = update_rolling_file_buffer(rolling_file_buffer, buf_df)

            # 복원된 버퍼 상태 출력
            if rolling_file_buffer:
                combined_buf = pl.concat(rolling_file_buffer, how='vertical')
                print(f"  Rolling buffer restored: {len(rolling_file_buffer)} file(s), "
                      f"{combined_buf.height} records, "
                      f"{combined_buf['Date'].min()} ~ {combined_buf['Date'].max()}")
            continue

        year_estimates = []
        year_daily_dfs = []

        for file_idx, parquet_file in enumerate(files_in_year):
            print(f"\n  --- File {file_idx + 1}/{len(files_in_year)}: {os.path.basename(parquet_file)} ---")

            current_df = preprocess_trade_data_polars(parquet_file)

            if current_df.is_empty():
                print(f"  [Warning] No data, skipping...")
                continue

            year_daily_dfs.append(current_df)

            # current_period_start를 epoch days 정수로 변환
            current_period_start_int = int(
                current_df.select(pl.col('Date').cast(pl.Int32).min()).item()
            )
            current_period_start_date = current_df['Date'].min()
            print(f"  Current period starts: {current_period_start_date} "
                  f"(epoch days: {current_period_start_int})")

            # *** [핵심] 롤링 버퍼에서 lookback buffer 생성 ***
            lookback_buffer = build_lookback_buffer_from_rolling(rolling_file_buffer, buffer_days=19)

            if not lookback_buffer.is_empty():
                # 버퍼 상태 출력 (종목별 실제 확보 일수 확인)
                buf_days_per_sym = (
                    lookback_buffer
                    .group_by('Symbol')
                    .agg(pl.count().alias('buf_days'))
                )
                min_buf = buf_days_per_sym['buf_days'].min()
                max_buf = buf_days_per_sym['buf_days'].max()
                print(f"  Lookback buffer: {lookback_buffer.height} records, "
                      f"{lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()}, "
                      f"per-symbol days: min={min_buf}, max={max_buf}")

                if min_buf < 19:
                    print(f"  [Warning] Some symbols have < 19 buffer days (min={min_buf}). "
                          f"PIN may be missing for early dates of these symbols.")

                combined_df = pl.concat([lookback_buffer, current_df], how='vertical')
                combined_df = combined_df.sort(['Symbol', 'Date'])
            else:
                print(f"  No lookback buffer (first file)")
                combined_df = current_df

            print(f"  Combined data records: {combined_df.height}")

            # 종목별 데이터 분할 (병렬 처리 준비)
            unique_symbols = combined_df['Symbol'].unique().to_list()
            grouped_data = []
            for sym in unique_symbols:
                sub_df = combined_df.filter(pl.col('Symbol') == sym)
                if not sub_df['Date'].is_sorted():
                    sub_df = sub_df.sort('Date')
                grouped_data.append((sym, sub_df, current_period_start_int))

            print(f"  Symbols to process: {len(grouped_data)}")

            # 병렬 처리
            file_estimates = []
            with multiprocessing.Pool(
                processes=num_cores,
                initializer=init_worker,
                initargs=(grid_matrix,)
            ) as pool:
                for res in tqdm(
                    pool.imap_unordered(process_single_symbol, grouped_data),
                    total=len(grouped_data),
                    desc="  Estimating"
                ):
                    file_estimates.extend(res)

            print(f"  Estimated records: {len(file_estimates)}")
            year_estimates.extend(file_estimates)

            # *** [핵심] 현재 파일을 롤링 버퍼에 추가 (최대 2개 유지) ***
            rolling_file_buffer = update_rolling_file_buffer(rolling_file_buffer, current_df)
            print(f"  Rolling buffer updated: {len(rolling_file_buffer)} file(s) buffered "
                  f"({sum(df.height for df in rolling_file_buffer)} total records)")

        # 연도 처리 완료 → 중간 저장
        if year_daily_dfs:
            year_daily_df = (
                pl.concat(year_daily_dfs, how='vertical')
                .sort(['Symbol', 'Date'])
            )
            saved_path = save_yearly_intermediate_result(
                year_estimates, year_daily_df, intermediate_dir, year_folder
            )
            saved_intermediate_paths.append(saved_path)
        else:
            print(f"  [Warning] No data collected for {year_folder}.")

    # -------------------------------------------------------------------------
    # 중간 저장 파일들 병합 → 최종 결과
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("All years processed. Merging intermediate results...")
    print(f"{'='*70}")

    if not saved_intermediate_paths:
        print("No intermediate files found.")
        return pl.DataFrame()

    print(f"Merging {len(saved_intermediate_paths)} intermediate files...")
    partial_dfs = []
    for path in saved_intermediate_paths:
        if os.path.exists(path):
            partial_dfs.append(pl.read_parquet(path))
            print(f"  Loaded: {os.path.basename(path)}")
        else:
            print(f"  [Warning] Missing: {path}")

    if not partial_dfs:
        return pl.DataFrame()

    final_pl = (
        pl.concat(partial_dfs, how='vertical')
        .sort(['Symbol', 'Date'])
    )
    return final_pl


# =============================================================================
# 파일 탐색
# =============================================================================

def get_parquet_files_from_directory(base_dir, year_folders=None):
    """
    디렉토리 구조에서 parquet 파일 목록을 시간순으로 가져옵니다.
    """
    parquet_files = []

    if year_folders is None:
        year_folders = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('KOR_')
        ])

    for year_folder in year_folders:
        folder_path = os.path.join(base_dir, year_folder)
        if not os.path.exists(folder_path):
            print(f"[Warning] Folder not found: {folder_path}")
            continue
        files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        parquet_files.extend(files)

    print(f"\n[File Discovery]")
    print(f"Base directory: {base_dir}")
    print(f"Year folders: {year_folders if year_folders else 'All'}")
    print(f"Total parquet files found: {len(parquet_files)}")
    if parquet_files:
        print(f"First file: {os.path.basename(parquet_files[0])}")
        print(f"Last file:  {os.path.basename(parquet_files[-1])}")

    return parquet_files


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    BASE_DIR = r"E:\vpin_project_parquet"

    # 옵션 A: 특정 연도만 처리
    YEAR_FOLDERS = ['KOR_2017', 'KOR_2018']

    # 옵션 B: 모든 연도 처리
    # YEAR_FOLDERS = None

    OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n{'='*70}")
    print("PIN Calculation with Rolling Window (2-File Rolling Lookback Buffer)")
    print(f"{'='*70}")

    parquet_files = get_parquet_files_from_directory(BASE_DIR, YEAR_FOLDERS)

    if not parquet_files:
        print("\n[Error] No parquet files found.")
        exit(1)

    result_pl = run_daily_pin_rolling_sequential(
        parquet_files,
        output_dir=OUTPUT_DIR,
        year_folders=YEAR_FOLDERS
    )

    if not result_pl.is_empty():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        year_info = "_".join([f.replace('KOR_', '') for f in YEAR_FOLDERS]) if YEAR_FOLDERS else "ALL"
        output_filename = f"pin_daily_rolling_{year_info}_{timestamp}"

        full_path_parquet = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result_pl.write_parquet(full_path_parquet, compression='zstd')
        print(f"\n[Saved Full Data] {full_path_parquet}")

        full_path_csv = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result_pl.head(1000).write_csv(full_path_csv)
        print(f"[Saved Sample]    {full_path_csv}")

        print("\n[Preview]")
        print(result_pl.head(30))

        print("\n[Data Statistics]")
        print(f"Total records:        {result_pl.height:,}")
        print(f"Records with PIN:     {result_pl.filter(pl.col('PIN').is_not_null()).height:,}")
        print(f"Records without PIN:  {result_pl.filter(pl.col('PIN').is_null()).height:,}")

        symbol_stats = (
            result_pl
            .group_by('Symbol')
            .agg([
                pl.count().alias('total_days'),
                pl.col('PIN').is_not_null().sum().alias('pin_days')
            ])
            .with_columns([
                (pl.col('pin_days') / pl.col('total_days') * 100).alias('coverage_pct')
            ])
        )
        print("\n[Symbol Coverage Sample]")
        print(symbol_stats.head(10))

    else:
        print("\n[Warning] Result DataFrame is empty.")