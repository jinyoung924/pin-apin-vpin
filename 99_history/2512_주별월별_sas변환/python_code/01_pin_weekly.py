import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
from datetime import timedelta
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
    1. Parquet 파일에서 거래 데이터를 로드
    2. LR 컬럼을 기준으로 일별로 매수(B)와 매도(S) 건수를 집계
    3. SAS 코드의 'Week' 개념(주차 식별자)을 생성합니다.

    Parameters:
    parquet_path (str): .parquet 파일의 경로
    
    Returns:
    pl.DataFrame: Symbol, Date, B, S 컬럼을 가진 폴라즈 데이터프레임
    """

    # 1. parquet 파일 로드
    print(f"Loading and aggregating data from {parquet_path}...")
    q = pl.scan_parquet(parquet_path)
    
    # 2. 매수(B)와 매도(S) 건수를 일별로 집계
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
        
        # 3. Week 변수 생성
        # [SAS] Week="&WEEK"d; 부분 대응
        # Python: 해당 날짜가 속한 주의 '월요일'을 기준일로 설정하여 그룹화 키로 사용
        # (Date가 이미 날짜 객체이므로 바로 truncate 사용 가능)
        .with_columns(
            pl.col('Date').dt.truncate("1w").alias("Week_Start")
        )
        .sort(['Symbol', 'Date'])
    ).collect()
    
    print(f"Data loaded. Total daily records: {aggregated_df.height}")
    return aggregated_df

def estimate_pin_parameters(B_array, S_array, grid_combinations):
    """
    [파라미터 추정 함수 - MLE]
    EKOP(1996) 모델에 기반하여 최대우도추정법(MLE)으로 
    파라미터(alpha, delta, mu, epsilon_b, epsilon_s)를 추정합니다.
    SAS의 PROC NLMIXED 프로시저 로직을 Scipy로 구현했습니다.

    Parameters:
    B_array (numpy array): 추정 기간 동안의 일별 매수 횟수 배열
    S_array (numpy array): 추정 기간 동안의 일별 매도 횟수 배열
    grid_combinations (numpy array): 초기값 탐색을 위한 파라미터 조합 행렬

    Returns:
    dict: 추정된 파라미터 값(alpha, delta, mu, eb, es), PIN 값, 수렴 여부(converged)
    """
    
    # [논문] B와 S는 해당 날짜의 거래 데이터를 요약하는 충분 통계량(Sufficient Statistic)
    B = B_array
    S = S_array
    
    # -------------------------------------------------------------------------
    # 1단계: Vectorized Grid Search (행렬 연산)
    # -------------------------------------------------------------------------
    
    # Grid 파라미터 언패킹 --> 파라미터 종류에 따라 '열벡터'를 만듬
    a_g  = grid_combinations[:, 0:1]
    d_g  = grid_combinations[:, 1:2]
    mu_g = grid_combinations[:, 2:3]
    eb_g = grid_combinations[:, 3:4]
    es_g = grid_combinations[:, 4:5]

    # 데이터 언패킹 (Broadcasting 준비) --> B, S는 '행벡터로 만들어서 행렬연산
    B_row = B.reshape(1, -1)
    S_row = S.reshape(1, -1)
    
    # [논문] 식 (15): 혼합 포아송 분포(Mixture Poisson)의 로그 우도 함수 정의
    # Vectorized 연산으로 모든 Grid 조합에 대해 동시에 계산
    
    # 1) 정보 없는 날 (No News): 비정보 거래자만 참여
    p1 = (1 - a_g) * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, es_g)
    
    # 2) 악재 발생 (Bad News): 정보 거래자(매도) + 비정보 거래자 참여
    p2 = a_g * d_g * poisson.pmf(B_row, eb_g) * poisson.pmf(S_row, mu_g + es_g)
    
    # 3) 호재 발생 (Good News): 정보 거래자(매수) + 비정보 거래자 참여
    p3 = a_g * (1 - d_g) * poisson.pmf(B_row, mu_g + eb_g) * poisson.pmf(S_row, es_g)
    
    total_lik = p1 + p2 + p3
    total_lik[total_lik <= 0] = 1e-300 # [SAS] if temp=0 then temp=1E-300 대응
    
    # Negative Log Likelihood 계산 (Row-wise sum)
    nll_scores = -np.sum(np.log(total_lik), axis=1)
    
    # 최적의 초기값 선정
    best_idx = np.argmin(nll_scores)
    best_initial_guess = grid_combinations[best_idx]

    # -------------------------------------------------------------------------
    # 2단계: 선정된 1개의 초기값(best_initial_guess)으로 최적화(Minimize) 1회 수행
    # -------------------------------------------------------------------------
    
    def negative_log_likelihood(params):
        alpha, delta, mu, eb, es = params
        
        # [SAS] 파라미터 경계 조건 설정 (bounds 문 대응)
        # 확률은 0~1 사이, 도착률(Rate)은 0보다 커야 함
        if not (0 <= alpha <= 1 and 0 <= delta <= 1 and mu >= 0 and eb >= 0 and es >= 0):
            return np.inf

        # [SAS] log(0) 계산 방지용 아주 작은 값 (if temp=0 then temp=1E-300 대응)
        epsilon_float = 1e-300
        
        # 1) 정보 없는 날 (No News): 비정보 거래자만 참여
        prob_no_news = poisson.pmf(B, eb) * poisson.pmf(S, es)
        
        # 2) 악재 발생 (Bad News): 정보 거래자(매도) + 비정보 거래자 참여
        prob_bad_news = poisson.pmf(B, eb) * poisson.pmf(S, mu + es)
        
        # 3) 호재 발생 (Good News): 정보 거래자(매수) + 비정보 거래자 참여
        prob_good_news = poisson.pmf(B, mu + eb) * poisson.pmf(S, es)
        
        # [논문] 전체 우도 계산: 각 시나리오별 확률의 가중 합
        likelihood = (1 - alpha) * prob_no_news + \
                     (alpha * delta) * prob_bad_news + \
                     (alpha * (1 - delta)) * prob_good_news
        
        # [SAS] 로그 우도 계산 (model B~general(loglik) 대응)
        log_lik = np.log(likelihood + epsilon_float)
        
        # 전체 데이터셋의 우도는 각 거래일 우도의 합 (독립성 가정)
        # minimize 함수를 쓰기 위해 음수로 반환
        return -np.sum(log_lik)

    try:
        # L-BFGS-B (SAS의 technique=quanew update=bfgs 와 유사하며 경계조건 지원)
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
    
    # [논문] PIN(Probability of Informed Trading) 계산 식
    # SAS Code: PIN=a*u/(a*u + eb + es);
    denom = (alpha * mu + eb + es)
    if denom == 0: denom = 1e-9
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
    """Worker 프로세스 초기화 및 Grid 데이터(초기값 계산에 필요한 243행렬) 공유"""
    global GLOBAL_GRID
    GLOBAL_GRID = grid

def process_single_symbol(data_tuple):
    """
    개별 종목(Symbol)의 데이터를 받아서 모든 주차(Week)의 PIN을 계산하는 함수
    multiprocessing.Pool에서 호출됩니다.
    """
    sym, sym_df = data_tuple
    
    # 데이터가 너무 적으면 스킵
    if sym_df.height < 10:
        return []

    results = []
    unique_weeks = sym_df['Week_Start'].unique().sort().to_list()
    
    # Numpy 변환 (APIN 스타일 최적화)
    dates = sym_df['Date'].to_numpy()
    Bs = sym_df['B'].to_numpy()
    Ss = sym_df['S'].to_numpy()
    
    for current_week_start in unique_weeks:
        # [SAS] 윈도우 설정 및 필터링 로직 동일
        window_end = current_week_start + timedelta(days=6)
        window_start = current_week_start - timedelta(weeks=3)
        
        # Boolean Masking 사용
        mask = (dates >= window_start) & (dates <= window_end)
        
        if np.sum(mask) <= 10:
            continue
        
        # Global Grid 사용
        est = estimate_pin_parameters(
            Bs[mask],
            Ss[mask],
            GLOBAL_GRID
        )
        
        if est['converged']:
            results.append({
                'Symbol': sym,
                'Week_Start': current_week_start,
                'a': est['alpha'],
                'd': est['delta'],
                'u': est['mu'],
                'eb': est['eb'],
                'es': est['es'],
                'PIN': est['PIN']
            })
    
    return results

def run_weekly_pin_parallel(parquet_path):
    """
    병렬 처리를 적용한 메인 분석 함수
    """
    # 1. 데이터 로드
    daily_df = preprocess_trade_data_polars(parquet_path)
    
    if daily_df.height == 0:
        print("No data found.")
        return pl.DataFrame()

    # Grid Combinations 생성
    print("Generating Grid Combinations...")
    grid_a = [0.1, 0.5, 0.9]
    grid_d = [0.1, 0.5, 0.9]
    grid_u = [20, 200, 2000]
    grid_eb = [20, 200, 2000]
    grid_es = [20, 200, 2000]
    
    # 243개 조합 생성 
    grid_iter = itertools.product(grid_a, grid_d, grid_u, grid_eb, grid_es)
    grid_matrix = np.array(list(grid_iter), dtype=np.float64)
    print(f"Grid Size: {grid_matrix.shape[0]} combinations.")

    print("Grouping data by Symbol for parallel processing...")
    
    # 병렬 처리를 위해 데이터를 종목별 (Symbol, DataFrame) 튜플 리스트로 분할
    unique_symbols = daily_df['Symbol'].unique().to_list()
    grouped_data = []
    
    for sym in unique_symbols:
        # 종목별로 데이터를 쪼개서 리스트에 담음
        sub_df = daily_df.filter(pl.col('Symbol') == sym)
        grouped_data.append((sym, sub_df))

    # CPU 코어 수 확인 및 병렬 처리 설정
    num_cores = multiprocessing.cpu_count() 
    print(f"Starting Parallel Estimation for {len(grouped_data)} symbols...")
    print(f"Using {num_cores} CPU cores.")

    all_estimates = []
    
    # Multiprocessing Pool 실행 (Shared Memory Initializer 적용)
    with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(grid_matrix,)) as pool:
        
        # map 대신 imap_unordered 사용
        results_iterator = pool.imap_unordered(process_single_symbol, grouped_data)
        
        # 2. tqdm으로 감싸서 루프 돌리기
        for res in tqdm(results_iterator, total=len(grouped_data), desc="Processing"):
            all_estimates.extend(res)

    print("Estimation finished.")

    # -------------------------------------------------------------------------
    # [SAS] 결과 병합: 주간 파라미터를 일별 데이터에 결합 (Broadcasting)
    # SAS Code: proc sql ... where intck("WEEK", a.Date, b.Week)=0;
    # -------------------------------------------------------------------------
    
    if not all_estimates:
        print("No valid estimates found.")
        return pl.DataFrame()

    # 주간 추정치 DataFrame 생성
    week_params_df = pl.DataFrame(all_estimates)
    
    # 일별 데이터(daily_df)에 주간 파라미터(week_params_df)를 Join
    final_df = (
        daily_df.join(
            week_params_df,
            on=['Symbol', 'Week_Start'],
            how='inner' # 추정치가 있는 날만 남기려면 inner 사용
        )
        .sort(['Symbol', 'Date'])
    )
    
    # 컬럼 선택 및 정리 (to_pandas() 제거함!)
    final_pl = final_df.select([
        'Symbol', 'Date', 'B', 'S', 
        pl.col('Week_Start').alias('Week'),
        'a', 'd', 'u', 'eb', 'es', 'PIN'
    ])
    
    return final_pl

# -------------------------------------------------------------------------
# 실행부 (고성능 I/O 적용)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Mac/Linux 멀티프로세싱 안전장치
    multiprocessing.freeze_support()
    
    # 1. 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    target_filename = "kor_201001"
    input_file_path = os.path.join(project_root, "input_data", f"{target_filename}.parquet")
    output_dir = os.path.join(project_root, "output_data")
    
    os.makedirs(output_dir, exist_ok=True)

    print(f"Target Input Path: {input_file_path}")

    # 2. 실행
    if os.path.exists(input_file_path):
        # 병렬 처리 함수 호출
        result_pl = run_weekly_pin_parallel(input_file_path)
        
        if not result_pl.is_empty():
            # 결과 파일이름 동적으로 생성
            script_name = os.path.splitext(os.path.basename(__file__))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            output_filename = f"{script_name}_result_{timestamp}"
            
            # (A) 전체 데이터: Polars의 write_parquet 사용 (가장 빠름)
            output_parquet = f"{output_filename}.parquet"
            full_path_parquet = os.path.join(output_dir, output_parquet)
            
            # compression='zstd' 권장 (속도/압축률 밸런스 최상)
            result_pl.write_parquet(full_path_parquet, compression='zstd')
            print(f"\n[Saved Full Data] {full_path_parquet}")
            print(" -> Format: Parquet (ZSTD compressed)")

            # (B) 확인용 샘플: Polars의 write_csv 사용
            output_csv_sample = f"{output_filename}_SAMPLE.csv"
            full_path_csv = os.path.join(output_dir, output_csv_sample)
            
            # 상위 1000개만 잘라서 CSV 저장
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