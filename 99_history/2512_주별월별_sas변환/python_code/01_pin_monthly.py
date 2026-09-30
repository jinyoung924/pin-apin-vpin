import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson
import warnings
from datetime import timedelta, date
import os
from datetime import datetime
import multiprocessing # 병렬 처리를 위한 모듈
import itertools # mle 계산에서 그리드서치 243개로 변경을 위해 사용
from tqdm import tqdm # 진행바 추가
# dateutil은 월 단위 연산을 위해 필요 (없을 경우 pip install python-dateutil)
from dateutil.relativedelta import relativedelta 

# SAS Code: 1. PIN Estimation (Monthly).sas 복제본
# Monthly PIN Estimation Logic

# 경고 메시지 무시
warnings.filterwarnings('ignore')

def preprocess_trade_data_polars(parquet_path):
    """
    [데이터 전처리 함수]
    1. Parquet 파일에서 거래 데이터를 로드
    2. LR 컬럼을 기준으로 일별로 매수(B)와 매도(S) 건수를 집계
    3. SAS 코드의 'Month' 개념(월별 식별자)을 생성합니다.

    Parameters:
    parquet_path (str): .parquet 파일의 경로
    
    Returns:
    pl.DataFrame: Symbol, Date, B, S, Month_Start 컬럼을 가진 폴라즈 데이터프레임
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
        
        # 3. Month 변수 생성
        # 해당 날짜가 속한 달의 '1일'을 기준일로 설정하여 그룹화 키로 사용
        .with_columns(
            pl.col('Date').dt.truncate("1mo").alias("Month_Start")
        )
        .sort(['Symbol', 'Date'])
    ).collect()
    
    print(f"Data loaded. Total daily records: {aggregated_df.height}")
    return aggregated_df

def estimate_pin_parameters(B_array, S_array):
    """
    [파라미터 추정 함수 - MLE]
    EKOP(1996) 모델에 기반하여 최대우도추정법(MLE)으로 
    파라미터(alpha, delta, mu, epsilon_b, epsilon_s)를 추정합니다.
    SAS의 PROC NLMIXED 프로시저 로직을 Scipy로 구현했습니다.

    Parameters:
    B_array (numpy array): 추정 기간 동안의 일별 매수 횟수 배열
    S_array (numpy array): 추정 기간 동안의 일별 매도 횟수 배열

    Returns:
    dict: 추정된 파라미터 값(alpha, delta, mu, eb, es), PIN 값, 수렴 여부(converged)
    """
    
    # [논문] B와 S는 해당 날짜의 거래 데이터를 요약하는 충분 통계량(Sufficient Statistic)
    B = B_array
    S = S_array
    
    def negative_log_likelihood(params):
        # [논문] 식 (15): 혼합 포아송 분포(Mixture Poisson)의 로그 우도 함수 정의
        
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
        return -np.sum(log_lik)

    # [수정] SAS와 동일한 Grid Search 구현
    grid_a = [0.1, 0.5, 0.9]
    grid_d = [0.1, 0.5, 0.9]
    grid_u = [20, 200, 2000]
    grid_eb = [20, 200, 2000]
    grid_es = [20, 200, 2000]
    
    # 243개 조합 생성
    initial_guesses = list(itertools.product(grid_a, grid_d, grid_u, grid_eb, grid_es))

    # -------------------------------------------------------------------------
    # [수정된 로직] 
    # 1단계: 243개 초기값에 대해 단순히 Likelihood 값만 계산하여 '최적의 시작점' 찾기
    # -------------------------------------------------------------------------
    best_initial_guess = None
    best_initial_nll = np.inf

    for guess in initial_guesses:
        # 여기서는 minimize를 하지 않고, 최적 시작점을 찾기 위해서 단순 함수 호출만 합니다.
        nll = negative_log_likelihood(guess)
        if nll < best_initial_nll:
            best_initial_nll = nll
            best_initial_guess = guess

    # -------------------------------------------------------------------------
    # 2단계: 선정된 1개의 초기값(best_initial_guess)으로 최적화(Minimize) 1회 수행
    # -------------------------------------------------------------------------
    if best_initial_guess is None:
        return {'converged': False}

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
    pin = (alpha * mu) / (alpha * mu + eb + es)
    
    return {
        'alpha': alpha, 'delta': delta, 'mu': mu, 
        'eb': eb, 'es': es, 'PIN': pin, 'converged': True
    }

# -------------------------------------------------------------------------
# 병렬 처리를 위한 작업자 함수 (Worker Function)
# -------------------------------------------------------------------------
def process_single_symbol(data_tuple):
    """
    개별 종목(Symbol)의 데이터를 받아서 모든 월(Month)의 PIN을 계산하는 함수
    multiprocessing.Pool에서 호출됩니다.
    
    [SAS Difference]
    Weekly 코드와 달리 Monthly 코드는 '3개월 롤링 윈도우'와 '20일 최소 관측치' 조건을 사용합니다.
    """
    sym, sym_df = data_tuple
    
    # 데이터가 너무 적으면 스킵
    if sym_df.height < 20:
        return []

    results = []
    # 데이터에 존재하는 모든 '월'을 가져옵니다.
    unique_months = sym_df['Month_Start'].unique().sort().to_list()
    
    for current_month_start in unique_months:
        # ---------------------------------------------------------------------
        # [SAS] 롤링 윈도우 설정 (3-Month Rolling Window)
        # SAS Code: intnx("month", "&MONTH"d, -2, 'b') <= Date <= intnx("month", "&MONTH"d, 0, 'e')
        # 의미: 타겟 월(current_month) 포함 과거 2개월 전부터 ~ 타겟 월 말일까지
        # ---------------------------------------------------------------------
        
        # 윈도우 시작일: 타겟 월의 2달 전 1일
        window_start = current_month_start - relativedelta(months=2)
        
        # 윈도우 종료일: 타겟 월의 마지막 날 (다음 달 1일 - 1일)
        window_end = current_month_start + relativedelta(months=1, days=-1)
        
        # 해당 기간 데이터 필터링
        window_df = sym_df.filter(
            (pl.col('Date') >= window_start) & 
            (pl.col('Date') <= window_end)
        )
        
        # ---------------------------------------------------------------------
        # [SAS] 최소 관측치 조건 (Minimum Observation Condition)
        # SAS Code: if NObs>20;
        # ---------------------------------------------------------------------
        if window_df.height <= 20:
            continue
        
        # MLE 추정 수행
        est = estimate_pin_parameters(
            window_df['B'].to_numpy(),
            window_df['S'].to_numpy()
        )
        
        if est['converged']:
            results.append({
                'Symbol': sym,
                'Month_Target': current_month_start, # 추정이 수행된 기준 월 (Target Month)
                'a': est['alpha'],
                'd': est['delta'],
                'u': est['mu'],
                'eb': est['eb'],
                'es': est['es'],
                'PIN': est['PIN']
            })
    
    return results

def run_monthly_pin_parallel(parquet_path):
    """
    병렬 처리를 적용한 메인 분석 함수 (Monthly Version)
    """
    # 1. 데이터 로드
    daily_df = preprocess_trade_data_polars(parquet_path)
    
    if daily_df.height == 0:
        print("No data found.")
        return pl.DataFrame()

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
    print(f"Starting Parallel Estimation (Monthly) for {len(grouped_data)} symbols...")
    print(f"Using {num_cores} CPU cores.")

    all_estimates = []
    
    # Multiprocessing Pool 실행
    with multiprocessing.Pool(processes=num_cores) as pool:
        
        results_iterator = pool.imap_unordered(process_single_symbol, grouped_data)
        
        # tqdm으로 감싸서 진행상황 표시
        for res in tqdm(results_iterator, total=len(grouped_data), desc="Processing"):
            all_estimates.extend(res)

    print("Estimation finished.")

    # -------------------------------------------------------------------------
    # [SAS] 결과 병합 및 후처리
    # SAS Code: where intck("month", a.Date, b.Month)=-1;
    # 의미: 추정된 PIN 값을 1개월 시차를 두고 매칭 (Lagged Matching)
    # 예: 10월(Month_Target)에 추정된 PIN -> 11월 데이터(Date)에 매핑
    # -------------------------------------------------------------------------
    
    if not all_estimates:
        print("No valid estimates found.")
        return pl.DataFrame()

    # 월간 추정치 DataFrame 생성
    month_params_df = pl.DataFrame(all_estimates)
    
    # 매칭을 위한 Key 생성: Target Month + 1개월
    # (SAS의 intck=-1 조건을 구현하기 위해, 추정 월에 1달을 더해서 실제 데이터의 Month_Start와 조인)
    month_params_df = month_params_df.with_columns(
        pl.col("Month_Target").dt.offset_by("1mo").alias("Join_Month")
    )
    
    # 해당 월의 마지막 영업일 계산을 위해 daily_df와 조인
    # 각 월의 실제 거래일 중 가장 마지막 날짜를 찾음
    last_trading_days = (
        daily_df
        .group_by(['Symbol', 'Month_Start'])
        .agg(pl.col('Date').max().alias('Month'))  # 해당 월의 마지막 거래일
    )
    
    # month_params_df에 마지막 영업일 정보 추가
    month_params_df = month_params_df.join(
        last_trading_days,
        left_on=['Symbol', 'Join_Month'],
        right_on=['Symbol', 'Month_Start'],
        how='left'
    )
    
    # 일별 데이터(daily_df)에 월간 파라미터를 Join
    final_df = (
        daily_df.join(
            month_params_df,
            left_on=['Symbol', 'Month_Start'],
            right_on=['Symbol', 'Join_Month'],
            how='inner' # SAS: where a.Symbol=b.Symbol ... (Inner Join 성격)
        )
        .sort(['Symbol', 'Date'])
    )
    
    # -------------------------------------------------------------------------
    # [수정] 출력 데이터 형태 (Output Columns)
    # weekly 파일과 동일하게 모든 파라미터 및 Month 컬럼 포함
    # -------------------------------------------------------------------------
    final_pl = final_df.select([
        'Symbol', 
        'Date',
        'B',
        'S',
        'Month',
        pl.col('a'),
        pl.col('d'),
        pl.col('u'),
        pl.col('eb'),
        pl.col('es'),
        'PIN'
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
        # 병렬 처리 함수 호출 (Monthly)
        result_pl = run_monthly_pin_parallel(input_file_path)
        
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