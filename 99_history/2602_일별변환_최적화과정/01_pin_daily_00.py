"""
=============================================================================
일별 롤링 윈도우 PIN 계산 (Lookback Buffer 방식)
=============================================================================

[핵심 개념]
- 20영업일 롤링 윈도우로 매일 PIN 계산
- 파일 간 연속성 보장을 위한 Lookback Buffer (19일) 활용
- 종목별 독립적인 시계열 연속성 유지

[종목별 버퍼 관리 원리]

파일 구조:
  kor_201101.parquet: A종목(1~3월) / B종목(1~3월) / C종목(1~3월)
  kor_201102.parquet: A종목(4~6월) / B종목(4~6월) / C종목(4~6월)
  kor_201103.parquet: A종목(7~9월) / B종목(7~9월) / C종목(7~9월)

처리 과정:

[파일 1] kor_201101.parquet 처리
  1. 데이터 로드
     - A종목: 1/1 ~ 3/31
     - B종목: 1/1 ~ 3/31
     - C종목: 1/1 ~ 3/31
  
  2. PIN 계산 (20일차부터)
     - A종목: 1/20 ~ 3/31 ✓
     - B종목: 1/20 ~ 3/31 ✓
     - C종목: 1/20 ~ 3/31 ✓
  
  3. Buffer 저장 (각 종목별 마지막 19일)
     - A종목: 3/13 ~ 3/31 → Buffer[A]
     - B종목: 3/13 ~ 3/31 → Buffer[B]
     - C종목: 3/13 ~ 3/31 → Buffer[C]

[파일 2] kor_201102.parquet 처리
  1. 데이터 로드
     - A종목: 4/1 ~ 6/30
     - B종목: 4/1 ~ 6/30
     - C종목: 4/1 ~ 6/30
  
  2. Buffer 결합 (종목별로 매칭)
     - A종목: [Buffer[A]: 3/13~3/31] + [4/1~6/30] → 3/13 ~ 6/30
     - B종목: [Buffer[B]: 3/13~3/31] + [4/1~6/30] → 3/13 ~ 6/30
     - C종목: [Buffer[C]: 3/13~3/31] + [4/1~6/30] → 3/13 ~ 6/30
  
  3. PIN 계산 (Buffer 덕분에 처음부터 가능!)
     - A종목: 4/1 ~ 6/30 ✓ (3/13~3/31 버퍼로 4/1 계산 가능)
     - B종목: 4/1 ~ 6/30 ✓
     - C종목: 4/1 ~ 6/30 ✓
  
  4. Buffer 갱신
     - A종목: 6/12 ~ 6/30 → Buffer[A]
     - B종목: 6/12 ~ 6/30 → Buffer[B]
     - C종목: 6/12 ~ 6/30 → Buffer[C]

[정렬 보장 메커니즘]
1. extract_lookback_buffer(): 
   - Symbol, Date로 정렬 후 종목별 마지막 19일 추출
   - maintain_order=True로 그룹 순서 유지
   - 추출 후 재정렬

2. Buffer + Current 결합:
   - pl.concat()으로 vertical 결합
   - Symbol, Date 기준 재정렬 (필수!)

3. 종목별 분할:
   - filter(Symbol == sym)로 종목별 추출
   - 각 종목 데이터는 Date 정렬 상태 유지
   - assert로 정렬 검증

[결과]
- 첫 파일의 처음 19일만 PIN 없음
- 이후 모든 날짜는 PIN 계산 보장
- 종목별 시계열 연속성 완벽 유지

=============================================================================
"""

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
import glob # 파일 패턴 매칭용


# 경고 메시지 무시
warnings.filterwarnings('ignore')

def preprocess_trade_data_polars(parquet_path):
    """
    [데이터 전처리 함수]
    1. Parquet 파일에서 거래 데이터를 로드
    2. LR 컬럼을 기준으로 일별로 매수(B)와 매도(S) 건수를 집계

    Parameters:
    parquet_path (str): .parquet 파일의 경로
    
    Returns:
    pl.DataFrame: Symbol, Date, B, S 컬럼을 가진 폴라즈 데이터프레임
    """

    # 1. parquet 파일 로드
    print(f"Loading data from {os.path.basename(parquet_path)}...")
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
        .sort(['Symbol', 'Date'])
    ).collect()
    
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
    *** [개선사항] Lookback Buffer를 고려한 일별 롤링 윈도우 방식 ***
    
    개별 종목(Symbol)의 데이터를 받아서 모든 거래일의 PIN을 계산하는 함수
    각 거래일에 대해 해당 날짜를 포함한 최근 20영업일 데이터를 사용하여 파라미터 추정
    
    [입력 데이터 구조 예시]
    sym_df for Symbol "A001":
    ┌─────────────┬─────┬─────┐
    │ Date        │ B   │ S   │
    ├─────────────┼─────┼─────┤
    │ 2011-01-13  │ 100 │ 95  │  ← lookback buffer 시작
    │ 2011-01-14  │ 105 │ 98  │
    │ ...         │ ... │ ... │
    │ 2011-01-31  │ 110 │ 102 │  ← lookback buffer 끝
    │ 2011-02-01  │ 115 │ 105 │  ← current_period_start_date (이 날부터 결과 저장)
    │ 2011-02-02  │ 120 │ 108 │
    │ ...         │ ... │ ... │
    │ 2011-02-28  │ 125 │ 110 │
    └─────────────┴─────┴─────┘
    
    [롤링 윈도우 로직]
    - 각 거래일 t에 대해, t를 포함한 과거 20영업일의 데이터로 MLE 추정
    - lookback_buffer가 있으면 current_period 첫날부터 PIN 계산 가능
    - lookback_buffer가 없으면 20일차부터 계산 시작
    
    [결과 저장 규칙]
    - lookback_buffer 영역(current_period_start_date 이전)은 계산에만 사용
    - current_period_start_date 이후의 날짜만 결과에 포함
    
    Parameters:
    data_tuple: (Symbol, DataFrame, current_period_start_date)
        - Symbol (str): 종목 코드
        - DataFrame (pl.DataFrame): 해당 종목의 시계열 데이터 (Buffer + Current 결합)
        - current_period_start_date: 현재 파일의 시작 날짜 (결과 필터링 기준)
    
    Returns:
    list: 추정 성공한 날짜들의 파라미터 딕셔너리 리스트
    
    multiprocessing.Pool에서 호출됩니다.
    """
    sym, sym_df, current_period_start_date = data_tuple
    
    # 데이터가 너무 적으면 스킵
    if sym_df.height < 20:
        return []

    results = []
    
    # Numpy 변환 (벡터화 연산 최적화)
    # 이 시점에서 sym_df는 이미 Symbol, Date로 정렬되어 있음
    dates = sym_df['Date'].to_numpy()
    Bs = sym_df['B'].to_numpy()
    Ss = sym_df['S'].to_numpy()
    
    # *** [개선] lookback_buffer 유무 확인 ***
    # 첫 번째 날짜가 current_period_start_date보다 이전이면 lookback_buffer가 있는 것
    has_lookback = dates[0] < current_period_start_date
    
    if has_lookback:
        # *** [경우 1] lookback_buffer가 있는 경우 ***
        # 예: dates = [1/13, 1/14, ..., 1/31, 2/1, 2/2, ..., 2/28]
        #     current_period_start_date = 2/1
        # → 인덱스 19(20번째 날)부터 계산 시작하면
        #   1/13~2/1 (20일) 윈도우로 2/1의 PIN 계산 가능!
        start_idx = 19 if len(dates) >= 20 else len(dates)
    else:
        # *** [경우 2] lookback_buffer가 없는 경우 (첫 번째 파일) ***
        # 예: dates = [1/1, 1/2, ..., 1/31]
        # → 인덱스 19(20번째 날)부터 계산 시작
        start_idx = 19
    
    # *** [일별 롤링 윈도우 루프] ***
    # start_idx부터 마지막 날까지 반복
    for i in range(start_idx, len(dates)):
        current_date = dates[i]
        
        # *** [중요] 현재 처리 기간(current_period)의 날짜만 결과에 포함 ***
        # lookback_buffer 영역(이전 월 데이터)은 계산에만 사용하고 결과에서 제외
        # 예: current_date가 1/31이고 current_period_start_date가 2/1이면 skip
        if current_date < current_period_start_date:
            continue
        
        # *** [롤링 윈도우 생성] ***
        # 현재 날짜(i) 포함 최근 20일 선택
        # 예: i=19일 때 → [0:20] (인덱스 0~19, 총 20개)
        #     i=25일 때 → [6:26] (인덱스 6~25, 총 20개)
        window_start_idx = i - 19  # 20개를 만들기 위한 시작 인덱스
        window_end_idx = i + 1     # 슬라이싱은 끝 인덱스 미포함이므로 +1
        
        window_B = Bs[window_start_idx:window_end_idx]
        window_S = Ss[window_start_idx:window_end_idx]
        
        # 안전장치: 윈도우 크기 확인 (정확히 20개여야 함)
        if len(window_B) != 20:
            continue
        
        # MLE 파라미터 추정 (Global Grid 사용)
        est = estimate_pin_parameters(
            window_B,
            window_S,
            GLOBAL_GRID
        )
        
        # 수렴한 경우에만 결과 저장
        if est['converged']:
            results.append({
                'Symbol': sym,
                'Date': current_date,
                'a': est['alpha'],
                'd': est['delta'],
                'u': est['mu'],
                'eb': est['eb'],
                'es': est['es'],
                'PIN': est['PIN']
            })
    
    return results

def extract_lookback_buffer(daily_df, buffer_days=19):
    """
    *** [신규 함수] Lookback Buffer 추출 함수 ***
    
    각 종목별로 마지막 N일의 데이터를 추출하여 다음 파일 처리 시 사용
    
    [중요] 종목별 정렬 보장:
    - 각 종목마다 최근 N일의 데이터를 시간순으로 추출
    - 반환되는 DataFrame은 Symbol, Date 기준으로 완전히 정렬됨
    
    Parameters:
    daily_df (pl.DataFrame): 현재 파일의 일별 집계 데이터
    buffer_days (int): 버퍼로 저장할 날짜 수 (기본값: 19일)
    
    Returns:
    pl.DataFrame: 각 종목의 마지막 N일 데이터 (Symbol, Date 정렬 보장)
    """
    if daily_df.is_empty():
        return pl.DataFrame()
    
    # *** [개선] 종목별로 명확하게 마지막 N일 추출 후 재정렬 ***
    # 1단계: Symbol, Date 기준 정렬 (필수)
    sorted_df = daily_df.sort(['Symbol', 'Date'])
    
    # 2단계: 각 종목별로 마지막 N행 추출
    buffer_df = (
        sorted_df
        .group_by('Symbol', maintain_order=True)  # ← maintain_order=True 추가
        .tail(buffer_days)
    )
    
    # 3단계: 추출 후 다시 정렬하여 순서 보장
    # group_by().tail() 결과는 종목 순서가 섞일 수 있으므로 명시적 재정렬
    buffer_df = buffer_df.sort(['Symbol', 'Date'])
    
    return buffer_df

def run_daily_pin_rolling_sequential(parquet_files):
    """
    *** [개선사항] 여러 파켓 파일을 순차적으로 처리하며 lookback buffer 적용 ***
    
    병렬 처리를 적용한 메인 분석 함수
    파일 간 연속성을 보장하기 위해 이전 파일의 마지막 19일을 버퍼로 활용
    
    [처리 흐름 예시]
    
    파일 1: kor_201101.parquet (1월~3월)
    ────────────────────────────────────────────────────────────
    원본 데이터:
      A종목: 1/1 ~ 3/31 (90일)
      B종목: 1/1 ~ 3/31 (90일)
    
    처리 결과:
      - PIN 계산: 1/20 ~ 3/31 (처음 19일은 데이터 부족)
      - Buffer 저장: 3/13 ~ 3/31 (19일) ← 다음 파일을 위해
    
    파일 2: kor_201102.parquet (4월~6월)
    ────────────────────────────────────────────────────────────
    원본 데이터:
      A종목: 4/1 ~ 6/30 (91일)
      B종목: 4/1 ~ 6/30 (91일)
    
    Buffer 결합:
      A종목: [3/13~3/31 (19일)] + [4/1~6/30 (91일)] = 총 110일
      B종목: [3/13~3/31 (19일)] + [4/1~6/30 (91일)] = 총 110일
    
    처리 결과:
      - PIN 계산: 4/1 ~ 6/30 (처음부터 전부 계산 가능! ✓)
      - Buffer 저장: 6/12 ~ 6/30 (19일) ← 다음 파일을 위해
    
    파일 3: kor_201103.parquet (7월~9월)
    ────────────────────────────────────────────────────────────
    원본 데이터:
      A종목: 7/1 ~ 9/30 (92일)
      B종목: 7/1 ~ 9/30 (92일)
    
    Buffer 결합:
      A종목: [6/12~6/30 (19일)] + [7/1~9/30 (92일)] = 총 111일
      B종목: [6/12~6/30 (19일)] + [7/1~9/30 (92일)] = 총 111일
    
    처리 결과:
      - PIN 계산: 7/1 ~ 9/30 (처음부터 전부 계산 가능! ✓)
      - Buffer 저장: 9/12 ~ 9/30 (19일) ← 다음 파일을 위해
    
    [종목별 처리 보장]
    - Buffer는 종목별로 별도 관리됨
    - A종목의 버퍼와 B종목의 버퍼가 섞이지 않음
    - 각 종목은 독립적으로 시간순 연속성 유지
    
    Parameters:
    parquet_files (list): 처리할 parquet 파일 경로 리스트 (시간순 정렬 필요)
    
    Returns:
    pl.DataFrame: 전체 기간의 일별 PIN 결과
    """
    
    # Grid Combinations 생성 (한 번만)
    print("Generating Grid Combinations...")
    grid_a = [0.1, 0.5, 0.9]
    grid_d = [0.1, 0.5, 0.9]
    grid_u = [20, 200, 2000]
    grid_eb = [20, 200, 2000]
    grid_es = [20, 200, 2000]
    
    grid_iter = itertools.product(grid_a, grid_d, grid_u, grid_eb, grid_es)
    grid_matrix = np.array(list(grid_iter), dtype=np.float64)
    print(f"Grid Size: {grid_matrix.shape[0]} combinations.")
    
    # CPU 코어 수 확인
    num_cores = multiprocessing.cpu_count()
    print(f"Using {num_cores} CPU cores for parallel processing.\n")
    
    # *** [신규] 전체 결과를 저장할 리스트 ***
    all_results = []
    
    # *** [신규] 이전 파일의 lookback buffer (종목별 마지막 19일) ***
    lookback_buffer = pl.DataFrame()
    
    # -------------------------------------------------------------------------
    # 파일별 순차 처리 루프
    # -------------------------------------------------------------------------
    for file_idx, parquet_file in enumerate(parquet_files):
        print(f"\n{'='*70}")
        print(f"Processing file {file_idx + 1}/{len(parquet_files)}: {os.path.basename(parquet_file)}")
        print(f"{'='*70}")
        
        # 1. 현재 파일 데이터 로드
        current_df = preprocess_trade_data_polars(parquet_file)
        
        if current_df.is_empty():
            print(f"  [Warning] No data in {os.path.basename(parquet_file)}, skipping...")
            continue
        
        # 2. 현재 파일의 시작 날짜 저장 (결과 필터링에 사용)
        current_period_start_date = current_df['Date'].min()
        print(f"  Current period starts: {current_period_start_date}")
        
        # 3. *** [핵심] lookback_buffer와 현재 데이터 결합 ***
        if not lookback_buffer.is_empty():
            print(f"  Appending lookback buffer from previous file...")
            print(f"    - Buffer records: {lookback_buffer.height}")
            print(f"    - Buffer date range: {lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()}")
            
            # [종목별 버퍼 검증]
            buffer_symbols = lookback_buffer['Symbol'].n_unique()
            print(f"    - Symbols in buffer: {buffer_symbols}")
            
            # 버퍼와 현재 데이터를 결합 (버퍼가 앞에 옴)
            # vertical concat은 단순히 위아래로 붙이는 것
            combined_df = pl.concat([lookback_buffer, current_df], how='vertical')
            
            # *** [중요] 결합 후 반드시 Symbol, Date 기준으로 정렬 ***
            # 이렇게 해야 각 종목의 데이터가 시간순으로 연속됨
            # 예: [A종목 1/13-1/31] + [A종목 2/1-2/28] → [A종목 1/13-2/28]
            combined_df = combined_df.sort(['Symbol', 'Date'])
        else:
            print(f"  No lookback buffer (first file)")
            combined_df = current_df
        
        print(f"  Combined data records: {combined_df.height}")
        
        # 4. *** [종목별 데이터 분할 - 병렬 처리 준비] ***
        # 각 종목을 독립적으로 처리하기 위해 분할
        # 이 시점에서 combined_df는 Symbol, Date로 정렬되어 있으므로
        # 각 종목의 데이터는 시간순으로 연속됨
        unique_symbols = combined_df['Symbol'].unique().to_list()
        grouped_data = []
        
        for sym in unique_symbols:
            # 종목별로 필터링
            sub_df = combined_df.filter(pl.col('Symbol') == sym)
            
            # [검증] 각 종목 데이터가 날짜순으로 정렬되어 있는지 확인
            if not sub_df['Date'].is_sorted():
                print(f"  [Warning] Symbol {sym}: Date not sorted! Re-sorting...")
                sub_df = sub_df.sort('Date')
            
            # *** [중요] current_period_start_date 전달 ***
            # 이 날짜를 기준으로 lookback 영역과 실제 계산 영역을 구분
            grouped_data.append((sym, sub_df, current_period_start_date))
        
        print(f"  Symbols to process: {len(grouped_data)}")
        
        # 5. 병렬 처리로 PIN 계산
        file_estimates = []
        with multiprocessing.Pool(processes=num_cores, initializer=init_worker, initargs=(grid_matrix,)) as pool:
            results_iterator = pool.imap_unordered(process_single_symbol, grouped_data)
            
            for res in tqdm(results_iterator, total=len(grouped_data), desc="  Estimating"):
                file_estimates.extend(res)
        
        print(f"  Estimated records: {len(file_estimates)}")
        
        # 6. 결과 저장
        if file_estimates:
            all_results.extend(file_estimates)
        
        # 7. *** [핵심] 다음 파일을 위한 lookback buffer 생성 ***
        lookback_buffer = extract_lookback_buffer(current_df, buffer_days=19)
        if not lookback_buffer.is_empty():
            print(f"  Lookback buffer created for next file:")
            print(f"    - Records: {lookback_buffer.height}")
            print(f"    - Date range: {lookback_buffer['Date'].min()} ~ {lookback_buffer['Date'].max()}")
    
    # -------------------------------------------------------------------------
    # 전체 결과 병합 및 반환
    # -------------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("All files processed. Finalizing results...")
    print(f"{'='*70}")
    
    if not all_results:
        print("No valid estimates found across all files.")
        return pl.DataFrame()
    
    # 일별 추정치 DataFrame 생성
    daily_params_df = pl.DataFrame(all_results)
    
    # 전체 기간의 원본 데이터 로드 (결과 병합용)
    print("Loading all raw data for final merge...")
    all_daily_dfs = []
    for parquet_file in parquet_files:
        df = preprocess_trade_data_polars(parquet_file)
        if not df.is_empty():
            all_daily_dfs.append(df)
    
    if not all_daily_dfs:
        return pl.DataFrame()
    
    full_daily_df = pl.concat(all_daily_dfs, how='vertical').sort(['Symbol', 'Date'])
    
    # 일별 파라미터와 원본 데이터 조인
    final_df = (
        full_daily_df.join(
            daily_params_df,
            on=['Symbol', 'Date'],
            how='left'
        )
        .sort(['Symbol', 'Date'])
    )
    
    # 컬럼 선택 및 정리
    final_pl = final_df.select([
        'Symbol', 'Date', 'B', 'S', 
        'a', 'd', 'u', 'eb', 'es', 'PIN'
    ])
    
    return final_pl

def get_parquet_files_from_directory(base_dir, year_folders=None):
    """
    *** [신규 함수] 디렉토리 구조에서 parquet 파일 목록을 시간순으로 가져오기 ***
    
    디렉토리 구조:
    D:/vpin_project_parquet/
      ├── KOR_2010/
      │   ├── kor_201001.parquet
      │   ├── kor_201002.parquet
      │   └── kor_201003.parquet
      └── KOR_2011/
          └── ...
    
    Parameters:
    base_dir (str): 최상위 디렉토리 경로 (예: "D:/vpin_project_parquet")
    year_folders (list): 처리할 연도 폴더 리스트 (None이면 전체)
                         예: ['KOR_2010', 'KOR_2011']
    
    Returns:
    list: 시간순으로 정렬된 parquet 파일 경로 리스트
    """
    parquet_files = []
    
    if year_folders is None:
        # 모든 KOR_YYYY 폴더 검색
        year_folders = sorted([d for d in os.listdir(base_dir) 
                              if os.path.isdir(os.path.join(base_dir, d)) 
                              and d.startswith('KOR_')])
    
    for year_folder in year_folders:
        folder_path = os.path.join(base_dir, year_folder)
        
        if not os.path.exists(folder_path):
            print(f"[Warning] Folder not found: {folder_path}")
            continue
        
        # 폴더 내 모든 .parquet 파일 검색
        pattern = os.path.join(folder_path, "*.parquet")
        files = sorted(glob.glob(pattern))
        parquet_files.extend(files)
    
    print(f"\n[File Discovery]")
    print(f"Base directory: {base_dir}")
    print(f"Year folders: {year_folders if year_folders else 'All'}")
    print(f"Total parquet files found: {len(parquet_files)}")
    
    if parquet_files:
        print(f"First file: {os.path.basename(parquet_files[0])}")
        print(f"Last file: {os.path.basename(parquet_files[-1])}")
    
    return parquet_files

# -------------------------------------------------------------------------
# 실행부 (고성능 I/O 적용)
# -------------------------------------------------------------------------
if __name__ == "__main__":
    # Mac/Linux 멀티프로세싱 안전장치
    multiprocessing.freeze_support()
    
    # -------------------------------------------------------------------------
    # *** [사용자 설정 영역] ***
    # -------------------------------------------------------------------------
    
    # 1. 기본 디렉토리 경로 설정
    BASE_DIR = r"E:\vpin_project_parquet"
    
    # 2. 처리할 연도 폴더 선택
    # 옵션 A: 특정 연도만 처리
    YEAR_FOLDERS = ['KOR_2017','KOR_2018']  # 예: 2010년만
    # YEAR_FOLDERS = ['KOR_2010', 'KOR_2011']  # 예: 2010-2011년
    
    # 옵션 B: 모든 연도 처리
    # YEAR_FOLDERS = None  # None으로 설정하면 모든 KOR_YYYY 폴더 처리
    
    # 3. 출력 디렉토리 설정
    OUTPUT_DIR = os.path.join(BASE_DIR, "output_data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 파일 검색 및 실행
    # -------------------------------------------------------------------------
    
    print(f"\n{'='*70}")
    print("PIN Calculation with Rolling Window (Lookback Buffer Method)")
    print(f"{'='*70}")
    
    # 1. Parquet 파일 목록 가져오기
    parquet_files = get_parquet_files_from_directory(BASE_DIR, YEAR_FOLDERS)
    
    if not parquet_files:
        print("\n[Error] No parquet files found.")
        exit(1)
    
    # 2. 순차 처리 실행 (lookback buffer 적용)
    result_pl = run_daily_pin_rolling_sequential(parquet_files)
    
    # 3. 결과 저장
    if not result_pl.is_empty():
        # 결과 파일명 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # 처리한 연도 정보를 파일명에 포함
        if YEAR_FOLDERS:
            year_info = "_".join([f.replace('KOR_', '') for f in YEAR_FOLDERS])
        else:
            year_info = "ALL"
        
        output_filename = f"pin_daily_rolling_{year_info}_{timestamp}"
        
        # (A) 전체 데이터: Parquet 저장
        output_parquet = f"{output_filename}.parquet"
        full_path_parquet = os.path.join(OUTPUT_DIR, output_parquet)
        result_pl.write_parquet(full_path_parquet, compression='zstd')
        print(f"\n[Saved Full Data] {full_path_parquet}")
        
        # (B) 확인용 샘플: CSV 저장
        output_csv_sample = f"{output_filename}_SAMPLE.csv"
        full_path_csv = os.path.join(OUTPUT_DIR, output_csv_sample)
        result_pl.head(1000).write_csv(full_path_csv)
        print(f"[Saved Sample] {full_path_csv}")
        
        # 미리보기 출력
        print("\n[Preview]")
        print(result_pl.head(30))
        
        # 데이터 통계 출력
        print("\n[Data Statistics]")
        print(f"Total records: {result_pl.height:,}")
        print(f"Records with PIN: {result_pl.filter(pl.col('PIN').is_not_null()).height:,}")
        print(f"Records without PIN: {result_pl.filter(pl.col('PIN').is_null()).height:,}")
        
        # 종목별 통계
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