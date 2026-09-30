import os
import multiprocessing
import warnings
from datetime import datetime, time
import tempfile
import shutil

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

# 경고 메시지 무시
warnings.filterwarnings('ignore')

# =============================================================================
# 함수 1. 틱 데이터를 1분봉 데이터로 변환하는 함수
# -----------------------------------------------------------------------------
# [요구사항 반영 내역]
# 1. 입력: 틱 데이터 (Date, Time, Price, Volume)
# 2. 로직:
#    - Date + Time 병합 -> Datetime 생성
#    - Datetime 기준 정렬
#    - 1분 단위 리샘플링 (Price=Close/Last, Volume=Sum)
#    - BVC는 거래량이 중요하므로 거래 없는 구간은 생성하지 않음
# =============================================================================
def convert_ticks_to_symbol_1m_bars(tick_df: pl.DataFrame) -> pl.DataFrame:
    """
    틱 데이터를 입력받아 1분 단위의 타임바(Time Bar)로 변환합니다.
    병렬 처리를 위해 이미 종목별로 분리된 데이터프레임이 입력으로 들어옵니다.
    """
    
    # 데이터 유효성 검사
    if tick_df.is_empty():
        return pl.DataFrame(schema={
            "Datetime": pl.Datetime, "Symbol": pl.String, "Price": pl.Float64, "Volume": pl.Float64
        })

    # -------------------------------------------------------------------------
    # 1. 날짜 및 시간 통합 (Timestamp 생성) & 정렬
    # -------------------------------------------------------------------------
    # Polars의 dt.combine을 사용하여 Date와 Time 컬럼을 합쳐 Datetime 객체 생성
    # 시계열 순서 보장을 위해 정렬 수행
    df = (
        tick_df
        .with_columns(
            pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")
        )
        .sort("Datetime")
    )

    # 종목 코드 추출 (단일 종목 처리 가정)
    symbol_val = df["Symbol"][0]

    # -------------------------------------------------------------------------
    # 2. 1분 단위 집계 (Resampling & Aggregation)
    # -------------------------------------------------------------------------
    # group_by_dynamic을 사용하여 1분(1m) 간격으로 리샘플링
    # - Price: 해당 1분의 마지막 체결가 (Last/Close)
    # - Volume: 해당 1분의 누적 거래량 (Sum)
    # - 거래가 없는 구간은 데이터가 생성되지 않음 (Volume Clock 로직 적합)
    agg_df = (
        df.group_by_dynamic(
            "Datetime", 
            every="1m", 
            include_boundaries=False,
            closed="left"
        )
        .agg([
            pl.col("Price").last().alias("Price"),    # 종가(Close)
            pl.col("Volume").sum().alias("Volume")    # 거래량 합계
        ])
    )

    # -------------------------------------------------------------------------
    # 3. 출력 데이터 구성
    # -------------------------------------------------------------------------
    # Symbol 컬럼을 다시 추가하고 최종 스키마에 맞춰 정렬
    final_df = (
        agg_df
        .with_columns(pl.lit(symbol_val).alias("Symbol"))
        .select(["Datetime", "Symbol", "Price", "Volume"])
    )

    return final_df


# =============================================================================
# 함수 2. 1분봉 -> BVC 기반 VPIN 계산 함수
# -----------------------------------------------------------------------------
# [요구사항 반영 내역]
# 1. 입력: 1분봉 데이터
# 2. 로직:
#    - 가격 변화량(Delta P) 및 표준편차(Sigma P) 계산
#    - t-분포(df=0.25) 기반 매수 확률(ProbBuy) 추정 -> 매수/매도 거래량 분할
#    - 일평균 거래량(ADV) 기반 바구니 크기(V = ADV/50) 산출
#    - Volume Bucketing: 1분봉을 쪼개거나 합쳐서 정확히 V만큼 채움 (Spillover 처리)
#    - VPIN 산출: 50개 버킷 롤링 윈도우 (OI 합 / 총 거래량)
# =============================================================================
def calculate_vpin_for_single_symbol(bar_df: pl.DataFrame, rolling_window=50, buckets_per_day=50) -> pl.DataFrame:
    
    # 결과 스키마 정의
    schema_def = {"Datetime": pl.Datetime, "Symbol": pl.String, "BucketNo": pl.Int64, "VPIN": pl.Float64}
    empty_res = pl.DataFrame(schema=schema_def)

    # 데이터 부족 시 조기 종료
    if bar_df.height < 2:
        return empty_res

    # -------------------------------------------------------------------------
    # 1. 초기 설정 및 변수 선언
    # -------------------------------------------------------------------------
    times_dt = bar_df["Datetime"].to_list()
    prices = bar_df["Price"].to_numpy()
    volumes = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    # -------------------------------------------------------------------------
    # 2. BVC 매수/매도 거래량 추정 (Bulk Volume Classification)
    # -------------------------------------------------------------------------
    
    # [2-1] 가격 변화량 (Delta P) 계산
    # P_t - P_{t-1} 계산 (Numpy diff 사용)
    delta_p = np.diff(prices, prepend=prices[0])

    # [2-2] 표준편차 (Sigma Delta P) 계산
    # 전체 기간의 델타P 표준편차를 사용하여 변동성 척도 마련
    sigma_p = np.std(delta_p) + 1e-9 # 0 나누기 방지

    # [2-3] 표준화된 Z값 (Z-score)
    # 가격 변화량을 변동성으로 나누어 정규화
    z_scores = delta_p / sigma_p

    # [2-4] 매수 확률 (ProbBuy) 계산
    # Student's t-분포 (df=0.25) CDF 값 사용 (논문 기준)
    prob_buy = t_dist.cdf(z_scores, df=0.25)

    # -------------------------------------------------------------------------
    # 3. 바구니 크기 (V) 계산
    # -------------------------------------------------------------------------
    # 일평균 거래량(ADV) 계산을 위해 실제 거래가 발생한 '영업일 수' 카운트
    trading_dates = (
        bar_df
        .filter(pl.col("Volume") > 0)
        .select(pl.col("Datetime").dt.date())
        .unique()
    )
    n_trading_days = trading_dates.height
    total_volume = np.sum(volumes)
    
    if n_trading_days == 0:
        return empty_res
    
    # V = ADV / 50
    adv = total_volume / n_trading_days
    bucket_size_v = adv / buckets_per_day

    if bucket_size_v <= 1e-5:
        return empty_res

    # -------------------------------------------------------------------------
    # 4. 버킷 채우기 (Volume Bucketing Algorithm)
    # -------------------------------------------------------------------------
    bucket_records = []
    
    current_bucket_buy = 0.0  # 현재 버킷의 매수량 누적
    current_bucket_sell = 0.0 # 현재 버킷의 매도량 누적
    current_filled = 0.0      # 현재 버킷의 총 채워진 양
    bucket_no = 0
    
    # 1분봉 데이터를 순회하며 버킷 채우기
    for i in range(len(volumes)):
        rem_vol = float(volumes[i])   # 현재 1분봉의 총 거래량
        p_b = float(prob_buy[i])      # 현재 1분봉의 매수 확률
        curr_time = times_dt[i]       # 현재 시간
        
        if rem_vol <= 0:
            continue
        
        # [Case A & B 처리 루프]
        # 1분봉의 거래량이 버킷 잔여 용량보다 클 경우(Spillover),
        # 쪼개서 현재 버킷을 채우고, 남은 양으로 다음 버킷을 채움
        while rem_vol > 0:
            needed = bucket_size_v - current_filled # 버킷 완성을 위해 필요한 양
            alloc_vol = min(needed, rem_vol)        # 실제로 이번에 할당할 양
            
            # 매수/매도 비율(p_b)을 유지하며 할당
            current_bucket_buy += alloc_vol * p_b
            current_bucket_sell += alloc_vol * (1.0 - p_b)
            current_filled += alloc_vol
            rem_vol -= alloc_vol
            
            # 버킷이 가득 찼다면 (Volume Clock 완성)
            if current_filled >= bucket_size_v - 1e-9:
                # 주문 불균형(OI) 계산: |Buy - Sell|
                oi = abs(current_bucket_buy - current_bucket_sell)
                
                # 완성된 버킷 정보 저장 (시간, 버킷번호, OI)
                bucket_records.append((curr_time, bucket_no, oi))
                
                # 변수 초기화 및 다음 버킷 준비
                bucket_no += 1
                current_bucket_buy = 0.0
                current_bucket_sell = 0.0
                current_filled = 0.0

    if not bucket_records:
        return empty_res

    # -------------------------------------------------------------------------
    # 5. VPIN 롤링 윈도우 계산
    # -------------------------------------------------------------------------
    res_times = [r[0] for r in bucket_records]
    res_bucket_nos = [r[1] for r in bucket_records]
    res_ois = np.array([r[2] for r in bucket_records])
    
    final_vpin_values = np.full(len(res_ois), np.nan)
    
    # 50개 버킷 단위로 Rolling Sum 수행
    if len(res_ois) >= rolling_window:
        kernel = np.ones(rolling_window)
        # OI의 합계 계산 (Convolution 이용)
        oi_sums = np.convolve(res_ois, kernel, mode='valid')
        
        # VPIN 공식: (Sum of OI over 50 buckets) / (50 * Bucket Size V)
        # 50 * Bucket Size V = Total Volume over the window
        vpin_calced = oi_sums / (rolling_window * bucket_size_v)
        
        # 결과 매핑 (초기 49개는 NaN 유지)
        final_vpin_values[rolling_window - 1:] = vpin_calced

    # -------------------------------------------------------------------------
    # 6. 최종 반환 (DataFrame 변환)
    # -------------------------------------------------------------------------
    result_df = pl.DataFrame({
        "Datetime": res_times,
        "Symbol": [symbol_str] * len(res_times),
        "BucketNo": res_bucket_nos,
        "VPIN": final_vpin_values
    })
    
    # 형변환 보장
    result_df = result_df.with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64)
    ])
    
    return result_df


# =============================================================================
# 작업 파이프라인 (Worker Process)
# -----------------------------------------------------------------------------
# 파일 기반 병렬 처리를 위한 Worker 함수
# 각 프로세스는 할당받은 종목의 Parquet 파일을 읽어 독립적으로 VPIN을 계산합니다.
# =============================================================================
def process_symbol_task_file_based(args):
    symbol, input_path, output_path = args
    
    try:
        # 1. 틱 데이터 로드
        tick_df = pl.read_parquet(input_path)
        
        # 2. 함수 1 호출: 틱 -> 1분봉 변환
        bars_df = convert_ticks_to_symbol_1m_bars(tick_df)
        
        if bars_df.is_empty():
            return None
            
        # 3. 함수 2 호출: 1분봉 -> VPIN 계산
        vpin_df = calculate_vpin_for_single_symbol(bars_df)
        
        if vpin_df.is_empty():
            return None
        
        # 4. 결과 저장 (중간 산출물)
        vpin_df.write_parquet(output_path, compression='zstd')
        return output_path
        
    except Exception as e:
        print(f"Error processing symbol {symbol}: {e}")
        import traceback
        print(traceback.format_exc())
        return None


# =============================================================================
# 메인 실행부
# -----------------------------------------------------------------------------
# 1. 대용량 Parquet 파일 로드
# 2. Symbol 기준으로 데이터 파티셔닝 (임시 파일 생성)
# 3. Multiprocessing Pool을 이용한 병렬 VPIN 계산
# 4. 결과 취합 및 저장
# =============================================================================
def run_main_pipeline(input_parquet_path, output_dir_path):
    print(f"Loading data from {input_parquet_path}...")
    
    try:
        # Polars로 대용량 데이터 읽기
        raw_df = pl.read_parquet(input_parquet_path, columns=['Date', 'Time', 'Symbol', 'Price', 'Volume'])
    except Exception as e:
        print(f"Failed to read parquet file. Error: {e}")
        return

    print(f"Total rows: {raw_df.height}")
    print("Partitioning data by Symbol...")

    # 임시 디렉토리 생성 (작업 후 삭제)
    temp_dir = tempfile.mkdtemp(prefix="vpin_temp_")
    temp_input_dir = os.path.join(temp_dir, "input")
    temp_output_dir = os.path.join(temp_dir, "output")
    os.makedirs(temp_input_dir, exist_ok=True)
    os.makedirs(temp_output_dir, exist_ok=True)
    
    try:
        print("Saving temporary symbol files...")
        # [Map 단계] 데이터를 Symbol 별로 분할하여 임시 파일로 저장
        # 이는 메모리 부족을 방지하고 병렬 처리 효율을 높이기 위함
        symbol_groups = raw_df.partition_by("Symbol", as_dict=True)
        
        tasks = []
        for symbol, df in symbol_groups.items():
            if isinstance(symbol, tuple):
                symbol = symbol[0]
            
            input_file = os.path.join(temp_input_dir, f"{symbol}.parquet")
            output_file = os.path.join(temp_output_dir, f"{symbol}.parquet")
            
            df.write_parquet(input_file, compression='zstd')
            tasks.append((symbol, input_file, output_file))
        
        total_symbols = len(tasks)
        print(f"Detected {total_symbols} unique symbols.")
        
        # [Parallel 단계] CPU 코어 수에 맞춰 병렬 처리 수행
        num_cores = min(multiprocessing.cpu_count() - 1, total_symbols)
        num_cores = max(1, num_cores)
        
        print(f"Starting calculation with {num_cores} processes...")
        
        result_files = []
        with multiprocessing.Pool(processes=num_cores) as pool:
            # imap_unordered를 사용하여 먼저 끝나는 작업부터 처리
            iterator = pool.imap_unordered(process_symbol_task_file_based, tasks)
            
            for res in tqdm(iterator, total=total_symbols, desc="Processing VPIN"):
                if res is not None:
                    result_files.append(res)
        
        # [Reduce 단계] 결과 파일 병합 및 저장
        if result_files:
            print("Concatenating results...")
            
            dfs = []
            for fpath in result_files:
                try:
                    df = pl.read_parquet(fpath)
                    if not df.is_empty():
                        dfs.append(df)
                except Exception as e:
                    print(f"Warning: Failed to read {fpath}: {e}")
            
            if dfs:
                final_df = pl.concat(dfs).sort(["Symbol", "Datetime"])
                
                # 동적 파일명 생성 (입력 파일명 + 타임스탬프)
                script_name = os.path.splitext(os.path.basename(__file__))[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                output_name = f"{script_name}_result_{timestamp}"
                
                # Full Parquet 저장
                full_path = os.path.join(output_dir_path, f"{output_name}.parquet")
                final_df.write_parquet(full_path, compression='zstd')
                print(f"Saved Full Parquet: {full_path}")
                
                # Sample CSV 저장 (상위 1000개)
                csv_path = os.path.join(output_dir_path, f"{output_name}_SAMPLE.csv")
                final_df.head(1000).write_csv(csv_path)
                print(f"Saved Sample CSV: {csv_path}")
                
                print("\n[Preview]")
                print(final_df.head())
                
                print(f"\n[Statistics]")
                print(f"Total symbols processed: {final_df['Symbol'].n_unique()}")
                print(f"Total buckets: {final_df.height}")
                print(f"VPIN range: {final_df['VPIN'].min():.6f} ~ {final_df['VPIN'].max():.6f}")
            else:
                print("No valid results generated.")
        else:
            print("No valid results generated.")
    
    finally:
        print("Cleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory: {e}")


if __name__ == "__main__":
    # Windows 환경 멀티프로세싱 지원
    multiprocessing.freeze_support()
    
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 입력/출력 경로 동적 생성
    target_filename = "kor_201001" 
    input_path = os.path.join(project_root, "input_data", f"{target_filename}.parquet")
    output_path = os.path.join(project_root, "output_data")
    
    os.makedirs(output_path, exist_ok=True)
    
    if os.path.exists(input_path):
        run_main_pipeline(input_path, output_path)
    else:
        print(f"File not found: {input_path}")
        print("Please check the 'input_path' variable in the code.")