import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta # 월 단위 연산을 위해 추가
import os
from datetime import datetime
import multiprocessing # 병렬 처리를 위한 모듈
import itertools # mle 계산에서 그리드서치를 위해 사용
from tqdm import tqdm # 진행바 추가

# 경고 메시지 무시
warnings.filterwarnings('ignore')

def preprocess_trade_data_polars(parquet_path):
    """
    [데이터 전처리 함수]
    1. Parquet 파일 로드 (Date 컬럼은 이미 date 타입이라고 가정)
    2. 'LR' 컬럼 (1: 매수, -1: 매도) 기준 집계
    3. Month_Start: SAS 로직(해당 월 1일) 생성
    4. SAS의 특정 날짜(2020-08-01) 이전 데이터만 필터링

    Parameters:
    parquet_path (str): .parquet 파일 경로
    
    Returns:
    pl.DataFrame: Symbol, Date, B, S, Month_Start 포함
    """

    print(f"Loading and aggregating data from {parquet_path}...")
    
    #--cutoff_date : 이 날짜 이전의 데이터만 계산하도록 명시.
    cutoff_date = date(2020, 8, 1)

    # 1. Parquet 스캔 (Lazy Loading)
    q = pl.scan_parquet(parquet_path)
    
    # 2. 전처리 파이프라인
    aggregated_df = (
        q
        # 필요한 컬럼만 선택 (Date는 이미 date 타입이므로 변환 불필요)
        .select(['Symbol', 'Date', 'LR'])
        
        # 3. 유효한 거래 방향 필터링 (LR: 1, -1)
        .filter(pl.col('LR').is_in([1, -1]))
        
        # 매수/매도 플래그 생성
        .with_columns([
            (pl.col('LR') == 1).cast(pl.UInt32).alias('is_Buy'),
            (pl.col('LR') == -1).cast(pl.UInt32).alias('is_Sell')
        ])
        
        # 4. 일별(Daily) 집계
        .group_by(['Symbol', 'Date'])
        .agg([
            pl.col('is_Buy').sum().alias('B'),
            pl.col('is_Sell').sum().alias('S')
        ])
        
        # 거래가 있는 날만 남김
        .filter((pl.col('B') > 0) | (pl.col('S') > 0))
        
        # 5. 월(Month) 변수 생성 (SAS Logic: intnx("Month", Date, 0, 'b'))
        # 해당 날짜가 속한 달의 1일로 truncate
        .with_columns(
            pl.col('Date').dt.truncate("1mo").alias("Month_Start")
        )
        
        # 6. [SAS Logic Difference] 날짜 필터링 (Month < '01AUG2020'd)
        .filter(pl.col('Month_Start') < cutoff_date)
        
        .sort(['Symbol', 'Date'])
    ).collect()
    
    print(f"Data loaded. Total daily records: {aggregated_df.height}")
    return aggregated_df

def estimate_apin_parameters(B_array, S_array, grid_combinations):
    """
    [파라미터 추정 함수 - MLE]
    Duarte & Young (2009) APIN 모델 (10개 파라미터)
    """
    
    B = B_array
    S = S_array
    T = len(B)
    
    # -------------------------------------------------------------------------
    # 1단계: Grid Search (Vectorized)
    # -------------------------------------------------------------------------
    
    # Grid 파라미터 언패킹 (N x 1)
    a_g  = grid_combinations[:, 0:1]
    d_g  = grid_combinations[:, 1:2]
    t1_g = grid_combinations[:, 2:3]
    t2_g = grid_combinations[:, 3:4]
    ub_g = grid_combinations[:, 4:5]
    us_g = grid_combinations[:, 5:6]
    eb_g = grid_combinations[:, 6:7]
    es_g = grid_combinations[:, 7:8]
    pb_g = grid_combinations[:, 8:9]
    ps_g = grid_combinations[:, 9:10]

    # 데이터 언패킹 (1 x T)
    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)

    # 6가지 시나리오 (Duarte & Young, 2009 eq.8)
    p1 = (1-a_g)*(1-t1_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, es_g)
    p2 = (1-a_g)*t1_g     * poisson.pmf(B_row, eb_g+pb_g) * poisson.pmf(S_row, es_g+ps_g)
    p3 = a_g*(1-t2_g)*(1-d_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, us_g+es_g)
    p4 = a_g*t2_g*(1-d_g)     * poisson.pmf(B_row, eb_g+pb_g) * poisson.pmf(S_row, us_g+es_g+ps_g)
    p5 = a_g*(1-t2_g)*d_g     * poisson.pmf(B_row, ub_g+eb_g) * poisson.pmf(S_row, es_g)
    p6 = a_g*t1_g*d_g         * poisson.pmf(B_row, ub_g+eb_g+pb_g) * poisson.pmf(S_row, es_g+ps_g)

    total_lik = p1 + p2 + p3 + p4 + p5 + p6
    total_lik[total_lik <= 0] = 1e-300 # underflow 방지
    
    # Negative Log Likelihood
    nll_scores = -np.sum(np.log(total_lik), axis=1)
    
    best_idx = np.argmin(nll_scores)
    best_initial_guess = grid_combinations[best_idx]

    # -------------------------------------------------------------------------
    # 2단계: 정밀 최적화 (Minimize)
    # -------------------------------------------------------------------------
    
    def negative_log_likelihood(params):
        a, d, t1, t2, ub, us, eb, es, pb, ps = params
        
        # Bounds Check
        if not (0 <= a <= 1 and 0 <= d <= 1 and 0 <= t1 <= 1 and 0 <= t2 <= 1):
            return np.inf
        if any(p < 0 for p in [ub, us, eb, es, pb, ps]):
            return np.inf

        epsilon = 1e-300
        
        prob1 = (1-a)*(1-t1) * poisson.pmf(B, eb) * poisson.pmf(S, es)
        prob2 = (1-a)*t1     * poisson.pmf(B, eb+pb) * poisson.pmf(S, es+ps)
        prob3 = a*(1-t2)*(1-d) * poisson.pmf(B, eb) * poisson.pmf(S, us+es)
        prob4 = a*t2*(1-d)     * poisson.pmf(B, eb+pb) * poisson.pmf(S, us+es+ps)
        prob5 = a*(1-t2)*d     * poisson.pmf(B, ub+eb) * poisson.pmf(S, es)
        prob6 = a*t1*d         * poisson.pmf(B, ub+eb+pb) * poisson.pmf(S, es+ps)
        
        lik = prob1 + prob2 + prob3 + prob4 + prob5 + prob6
        return -np.sum(np.log(lik + epsilon))

    try:
        bounds = [(0, 1)]*4 + [(0, None)]*6
        best_result = minimize(
            negative_log_likelihood, 
            x0=best_initial_guess, 
            bounds=bounds,
            method='L-BFGS-B'
        )
    except Exception:
        return {'converged': False}

    if not best_result.success:
        return {'converged': False}

    # 결과 추출
    a, d, t1, t2, ub, us, eb, es, pb, ps = best_result.x
    
    return {
        'a': a, 'd': d, 't1': t1, 't2': t2,
        'ub': ub, 'us': us, 'eb': eb, 'es': es, 'pb': pb, 'ps': ps,
        'converged': True
    }

# -------------------------------------------------------------------------
# Multiprocessing 관련 설정
# -------------------------------------------------------------------------
GLOBAL_GRID = None

def init_worker(grid):
    """각 프로세스 초기화 시 Grid Matrix 공유"""
    global GLOBAL_GRID
    GLOBAL_GRID = grid

def process_single_symbol(data_tuple):
    """
    Worker 함수: 한 종목의 데이터를 받아 3개월 롤링 윈도우로 APIN 계산
    [SAS Logic Difference]: 
    1. Window: 해당 월 포함 과거 3개월 (-2 ~ 0)
    2. Min Obs: > 20 days
    """
    sym, sym_df = data_tuple
    
    if sym_df.height < 20: # [SAS Logic] 최소 관측치 20개 미만이면 제외
        return []

    results = []
    unique_months = sym_df['Month_Start'].unique().sort().to_list()
    
    # Numpy 변환 (인덱싱 속도 최적화)
    dates = sym_df['Date'].to_numpy()
    Bs = sym_df['B'].to_numpy()
    Ss = sym_df['S'].to_numpy()
    
    for current_month_start in unique_months:
        # [SAS Logic] Target Month 기준 과거 2개월 포함 (총 3개월)
        # intck("Month", a.Month, c.Date) between -2 and 0
        
        # 윈도우 시작: 현재 월 - 2개월
        window_start = current_month_start - relativedelta(months=2)
        
        # 윈도우 종료: 현재 월의 마지막 날
        # (다음달 1일 - 1일)
        window_end = current_month_start + relativedelta(months=1) - timedelta(days=1)
        
        # Boolean Masking
        mask = (dates >= window_start) & (dates <= window_end)
        
        # 데이터가 충분하지 않으면 스킵 (SAS 기준 > 20일)
        if np.sum(mask) <= 20: 
            continue
            
        est = estimate_apin_parameters(
            Bs[mask], 
            Ss[mask],
            GLOBAL_GRID
        )
        
        if est['converged']:
            row = {'Symbol': sym, 'Month_Start': current_month_start}
            row.update(est)
            del row['converged']
            results.append(row)
    
    return results

def run_monthly_apin_parallel(parquet_path):
    """
    메인 실행 함수
    """
    # 1. 데이터 로드 및 전처리
    daily_df = preprocess_trade_data_polars(parquet_path)
    
    if daily_df.height == 0:
        print("No valid data found after filtering.")
        return pl.DataFrame()

    # 2. Grid 생성 (SAS와 동일한 설정 유지)
    print("Generating Grid Combinations...")
    probs = [0.1, 0.5, 0.9]
    rates = [20, 200, 2000]
    
    # 10개 파라미터 조합 (3^10 = 59,049개)
    grid_iter = itertools.product(
        probs, probs, probs, probs,       
        rates, rates, rates, rates, rates, rates 
    )
    grid_matrix = np.array(list(grid_iter), dtype=np.float64)
    print(f"Grid Size: {grid_matrix.shape[0]} combinations.")

    # 3. 데이터 그룹화 (Symbol 기준)
    unique_symbols = daily_df['Symbol'].unique().to_list()
    grouped_data = []
    for sym in unique_symbols:
        sub_df = daily_df.filter(pl.col('Symbol') == sym)
        grouped_data.append((sym, sub_df))

    # 4. 병렬 처리 시작
    num_cores = multiprocessing.cpu_count()
    print(f"Starting Parallel Estimation for {len(grouped_data)} symbols using {num_cores} cores...")
    
    all_estimates = []
    
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(grid_matrix,)) as pool:
        results_iterator = pool.imap_unordered(process_single_symbol, grouped_data)
        
        for res in tqdm(results_iterator, total=len(grouped_data), desc="Processing"):
            all_estimates.extend(res)

    print("Estimation finished.")

    # 5. 결과 병합 및 후처리
    if not all_estimates:
        print("No valid estimates found.")
        return pl.DataFrame()

    month_params_df = pl.DataFrame(all_estimates)
    
    # [SAS Logic Difference] Join Logic
    # SAS: intck("month", a.Date, b.Month) = -1
    # 의미: Trade Date(a)가 속한 월이 Estimate Month(b)보다 1개월 뒤여야 함
    # 즉, Trade Month = Estimate Month + 1 Month
    # 따라서 Trade Data에 'Prev_Month'를 만들어 Estimate의 'Month_Start'와 조인
    
    daily_df_for_join = daily_df.with_columns(
        pl.col('Month_Start').dt.offset_by("-1mo").alias('Match_Month')
    )
    
    final_df = (
        daily_df_for_join.join(
            month_params_df,
            left_on=['Symbol', 'Match_Month'],
            right_on=['Symbol', 'Month_Start'],
            how='inner'
        )
    )
    
    # [SAS Logic Difference] APIN 및 PSOS 계산 (후처리 단계에서 수행)
    # SAS Code: APIN = ( a*(d*ub+(1-d)*us) ) / ...
    
    # 계산 편의를 위해 파라미터 컬럼 참조
    a = pl.col('a')
    d = pl.col('d')
    t1 = pl.col('t1')
    t2 = pl.col('t2')
    ub = pl.col('ub')
    us = pl.col('us')
    eb = pl.col('eb')
    es = pl.col('es')
    pb = pl.col('pb')
    ps = pl.col('ps')

    final_df = final_df.with_columns([
        # 분모 (Total Flow)
        ((a * (d*ub + (1-d)*us)) + ((pb + ps) * (a*t2 + (1-a)*t1)) + es + eb).alias('denom')
    ]).with_columns([
        # APIN
        ((a * (d*ub + (1-d)*us)) / pl.col('denom')).alias('APIN'),
        # PSOS
        (((pb + ps) * (a*t2 + (1-a)*t1)) / pl.col('denom')).alias('PSOS'),
        
        # [SAS Logic Difference] Month 포맷팅 (해당 월의 말일로 설정)
        # SAS: Month=intnx("Month", Date, 0, 'e');
        pl.col('Date').dt.month_end().alias('Month_End')
    ]).sort(['Symbol', 'Date'])

    # 최종 컬럼 정리
    cols_to_select = [
        'Symbol', 'Date', 'B', 'S', 
        pl.col('Month_End').alias('Month'), # SAS 결과와 맞추기 위해 Month_End를 Month로 명명
        'APIN', 'PSOS',
        'a', 'd', 't1', 't2', 'ub', 'us', 'eb', 'es', 'pb', 'ps'
    ]
    
    return final_df.select(cols_to_select)

# -------------------------------------------------------------------------
# 실행부
# -------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 입력 파일명 (확장자 제외)
    target_filename = "kor_201001" 
    input_file_path = os.path.join(project_root, "input_data", f"{target_filename}.parquet")
    output_dir = os.path.join(project_root, "output_data")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Target Input Path: {input_file_path}")

    if os.path.exists(input_file_path):
        result_pl = run_monthly_apin_parallel(input_file_path)
        
        if not result_pl.is_empty():
            # 결과 파일이름 동적으로 생성
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_filename = f"{script_name}_result_{timestamp}"
            
            # (A) 전체 데이터: Polars의 write_parquet 사용
            output_parquet = f"{output_filename}.parquet"
            full_path_parquet = os.path.join(output_dir, output_parquet)
            
            result_pl.write_parquet(full_path_parquet, compression='zstd')
            print(f"\n[Saved Full Data] {full_path_parquet}")
            print(" -> Format: Parquet (ZSTD compressed)")

            # (B) 확인용 샘플: Polars의 write_csv 사용
            output_csv_sample = f"{output_filename}_SAMPLE.csv"
            full_path_csv = os.path.join(output_dir, output_csv_sample)
            
            result_pl.head(1000).write_csv(full_path_csv)
            print(f"[Saved Sample] {full_path_csv}")
            print(" -> Format: CSV (Top 1000 rows only)")
            
            # 미리보기 출력
            print("\n[Preview]")
            print(result_pl.head())
        else:
            print("\n[Warning] Result DataFrame is empty.")
    else:
        print(f"\n[Error] File not found: {input_file_path}")
        print("Please ensure the parquet file exists in 'input_data' folder.")