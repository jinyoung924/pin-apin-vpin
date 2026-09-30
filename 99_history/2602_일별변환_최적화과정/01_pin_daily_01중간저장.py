"""
=============================================================================
일별 롤링 윈도우 PIN 계산 (Lookback Buffer 방식)
=============================================================================

[핵심 개념]
- 20영업일 롤링 윈도우로 매일 PIN 계산
- 파일 간 연속성 보장을 위한 Lookback Buffer (19일) 활용
- 종목별 독립적인 시계열 연속성 유지
- *** [개선] 연도(KOR_YYYY) 처리 완료 시마다 중간 저장 → 중단 후 재개 가능 ***

[Date 타입 보존 전략]
- 원본 Date 컬럼: Polars date 타입 (예: 2017-01-02)
- process_single_symbol 내부에서 Date를 numpy로 꺼낼 때
  .to_numpy()가 date → object 또는 datetime64로 변환되어 join 시 타입 불일치 발생
- 해결책: Date를 epoch days(정수, int32)로 변환하여 numpy 배열로 다루고,
  결과 딕셔너리에는 정수값(Date_int)을 저장 → DataFrame 생성 시 pl.Int32로 인식
  → .cast(pl.Date)로 원래 date 타입으로 복원 → join 타입 일치 보장

[원본 데이터 컬럼]
  Price    : f64
  Volume   : f64
  Symbol   : str
  Date     : date  (2017-01-02 형태)
  Time     : time  (나노초 정밀도, 10:00:09.239359254 형태)
  MidPoint : f64
  QSpread  : f64
  LR       : i8

[중간 저장 구조]
  output_data/
    ├── intermediate/
    │   ├── pin_daily_KOR_2017_partial.parquet   ← 연도별 중간 결과
    │   ├── pin_daily_KOR_2018_partial.parquet
    │   └── ...
    └── pin_daily_rolling_2017_2018_YYYYMMDD_HHMM.parquet  ← 최종 결과

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


def preprocess_trade_data_polars(parquet_path):
    """
    Parquet 파일에서 거래 데이터를 로드하고 일별 매수(B)/매도(S) 건수를 집계합니다.

    [Date 타입 보존]
    - 원본 Date(date 타입)를 그대로 유지
    - group_by/sort 모두 Polars 내부에서 처리하므로 타입 변환 없음
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

    # Date 컬럼이 date 타입인지 확인 후 보정
    if aggregated_df['Date'].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(
            pl.col('Date').cast(pl.Date)
        )

    return aggregated_df


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


# -------------------------------------------------------------------------
# Multiprocessing Shared Memory 설정
# -------------------------------------------------------------------------
GLOBAL_GRID = None

def init_worker(grid):
    global GLOBAL_GRID
    GLOBAL_GRID = grid


def process_single_symbol(data_tuple):
    """
    개별 종목의 데이터를 받아 20일 롤링 윈도우로 모든 거래일의 PIN을 계산합니다.

    [Date 타입 보존 핵심]
    기존 방식의 문제:
      dates = sym_df['Date'].to_numpy()
      → numpy가 pl.Date를 object 또는 datetime64[ns]로 변환
      → 딕셔너리에 저장된 날짜가 numpy 타입
      → pl.DataFrame() 생성 시 object로 추론됨
      → join 시 date vs object 타입 불일치 에러 발생

    수정 방식:
      dates_int = sym_df['Date'].cast(pl.Int32).to_numpy()
      → pl.Date를 epoch days(1970-01-01 기준 정수)로 변환
      → numpy int32 배열은 완전히 안전하게 비교 가능
      → current_period_start_int도 정수로 변환하여 비교
      → 결과 딕셔너리에 'Date_int'(정수) 저장
      → build_params_df()에서 .cast(pl.Date)로 원래 date 타입으로 복원
      → join 타입 완전 일치 보장
    """
    sym, sym_df, current_period_start_int = data_tuple

    if sym_df.height < 20:
        return []

    results = []
    Bs = sym_df['B'].to_numpy()
    Ss = sym_df['S'].to_numpy()

    # *** [핵심 수정] Date를 epoch days 정수로 변환 ***
    # pl.Date → pl.Int32: 1970-01-01 기준 경과 일수
    # 예) 2017-01-02 → 17168 (정수)
    dates_int = sym_df['Date'].cast(pl.Int32).to_numpy()

    start_idx = 19 if len(dates_int) >= 20 else len(dates_int)

    for i in range(start_idx, len(dates_int)):
        current_date_int = dates_int[i]

        # current_period 이전 날짜는 결과에서 제외 (lookback 영역)
        if current_date_int < current_period_start_int:
            continue

        window_start_idx = i - 19
        window_end_idx   = i + 1
        window_B = Bs[window_start_idx:window_end_idx]
        window_S = Ss[window_start_idx:window_end_idx]

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


def extract_lookback_buffer(daily_df, buffer_days=19):
    """
    각 종목별 마지막 N일 데이터를 추출합니다.
    Date 타입은 pl.Date 그대로 보존됩니다.
    """
    if daily_df.is_empty():
        return pl.DataFrame()

    sorted_df = daily_df.sort(['Symbol', 'Date'])
    buffer_df = (
        sorted_df
        .group_by('Symbol', maintain_order=True)
        .tail(buffer_days)
    )
    buffer_df = buffer_df.sort(['Symbol', 'Date'])
    return buffer_df


def build_params_df(year_estimates):
    """
    *** [신규 함수] 추정 결과 리스트 → Polars DataFrame 변환 (Date 타입 복원) ***

    process_single_symbol이 Date를 epoch days 정수(Date_int)로 저장했으므로:
    1. pl.DataFrame()으로 생성하면 Date_int는 pl.Int32로 안전하게 인식됨
    2. .cast(pl.Date)로 epoch days → pl.Date 타입으로 복원
    3. 컬럼명 Date_int → Date로 변경
    → 원본 current_df의 Date(pl.Date)와 타입이 완전히 일치 → join 에러 없음

    Parameters:
    year_estimates (list): process_single_symbol이 반환한 딕셔너리 리스트
                           각 딕셔너리에 'Date_int'(int) 키 포함

    Returns:
    pl.DataFrame: Symbol, Date(pl.Date), a, d, u, eb, es, PIN 컬럼
    """
    if not year_estimates:
        return pl.DataFrame()

    params_df = (
        pl.DataFrame(year_estimates)
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


def run_daily_pin_rolling_sequential(parquet_files, output_dir, year_folders):
    """
    여러 parquet 파일을 순차적으로 처리하며 lookback buffer와 연도별 중간 저장을 적용합니다.

    [current_period_start 전달 방식 변경]
    기존: current_period_start_date (Python date 객체)
          → multiprocessing pickling 시 타입 불안정 가능성
    수정: current_period_start_int (epoch days 정수)
          → int는 pickling이 완전히 안전
          → numpy int32 배열과 직접 비교 가능
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    print(f"Intermediate results will be saved to: {intermediate_dir}\n")

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
            parts = f.replace('\\', '/').split('/')
            for p in parts:
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
    lookback_buffer = pl.DataFrame()

    for year_folder in year_folders_ordered:
        files_in_year = year_to_files.get(year_folder, [])
        if not files_in_year:
            print(f"[Warning] No files found for {year_folder}, skipping...")
            continue

        print(f"\n{'='*70}")
        print(f"Processing year: {year_folder}  ({len(files_in_year)} files)")
        print(f"{'='*70}")

        intermediate_path = os.path.join(intermediate_dir, f"pin_daily_{year_folder}_partial.parquet")
        if os.path.exists(intermediate_path):
            print(f"  [Skip] Already processed: {os.path.basename(intermediate_path)}")
            print(f"         Delete it to reprocess this year.")
            saved_intermediate_paths.append(intermediate_path)

            last_file = files_in_year[-1]
            print(f"  Restoring lookback buffer from: {os.path.basename(last_file)}")
            buf_df = preprocess_trade_data_polars(last_file)
            if not buf_df.is_empty():
                lookback_buffer = extract_lookback_buffer(buf_df, buffer_days=19)
                print(f"  Buffer restored: {lookback_buffer.height} records, "
                      f"{lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()}")
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

            # *** [핵심] current_period_start를 epoch days 정수로 변환 ***
            # pl.Date.min() → Python datetime.date → pickling은 되지만
            # numpy int 배열과 직접 비교 불가
            # .cast(pl.Int32).min()으로 정수 추출 → numpy와 직접 비교 가능
            current_period_start_int = int(
                current_df.select(pl.col('Date').cast(pl.Int32).min()).item()
            )
            current_period_start_date = current_df['Date'].min()  # 로그 출력용
            print(f"  Current period starts: {current_period_start_date} "
                  f"(epoch days: {current_period_start_int})")

            # lookback_buffer와 현재 데이터 결합
            if not lookback_buffer.is_empty():
                print(f"  Appending lookback buffer: "
                      f"{lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()} "
                      f"({lookback_buffer.height} records, "
                      f"{lookback_buffer['Symbol'].n_unique()} symbols)")
                combined_df = pl.concat([lookback_buffer, current_df], how='vertical')
                combined_df = combined_df.sort(['Symbol', 'Date'])
            else:
                print(f"  No lookback buffer (first file of first year)")
                combined_df = current_df

            print(f"  Combined data records: {combined_df.height}")

            # 종목별 데이터 분할 (병렬 처리 준비)
            unique_symbols = combined_df['Symbol'].unique().to_list()
            grouped_data = []
            for sym in unique_symbols:
                sub_df = combined_df.filter(pl.col('Symbol') == sym)
                if not sub_df['Date'].is_sorted():
                    sub_df = sub_df.sort('Date')
                # *** current_period_start_int(정수) 전달 ***
                grouped_data.append((sym, sub_df, current_period_start_int))

            print(f"  Symbols to process: {len(grouped_data)}")

            # 병렬 처리로 PIN 계산
            file_estimates = []
            with multiprocessing.Pool(
                processes=num_cores,
                initializer=init_worker,
                initargs=(grid_matrix,)
            ) as pool:
                results_iterator = pool.imap_unordered(process_single_symbol, grouped_data)
                for res in tqdm(results_iterator, total=len(grouped_data), desc="  Estimating"):
                    file_estimates.extend(res)

            print(f"  Estimated records: {len(file_estimates)}")
            year_estimates.extend(file_estimates)

            # 다음 파일을 위한 lookback buffer 갱신
            lookback_buffer = extract_lookback_buffer(current_df, buffer_days=19)
            if not lookback_buffer.is_empty():
                print(f"  Lookback buffer updated: "
                      f"{lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()}")

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
    # 중간 저장 파일들을 읽어서 최종 결과 병합
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
        pattern = os.path.join(folder_path, "*.parquet")
        files = sorted(glob.glob(pattern))
        parquet_files.extend(files)

    print(f"\n[File Discovery]")
    print(f"Base directory: {base_dir}")
    print(f"Year folders: {year_folders if year_folders else 'All'}")
    print(f"Total parquet files found: {len(parquet_files)}")
    if parquet_files:
        print(f"First file: {os.path.basename(parquet_files[0])}")
        print(f"Last file:  {os.path.basename(parquet_files[-1])}")

    return parquet_files


# -------------------------------------------------------------------------
# 실행부
# -------------------------------------------------------------------------
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
    print("PIN Calculation with Rolling Window (Yearly Intermediate Save)")
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
