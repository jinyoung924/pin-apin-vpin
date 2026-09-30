import os
import warnings
from datetime import datetime

import polars as pl

# 경고 메시지 무시
warnings.filterwarnings('ignore')


# =============================================================================
# 월별 VPIN 평균 계산 함수
# -----------------------------------------------------------------------------
# [요구사항]
# 1. 입력: 버킷당 VPIN 데이터 (Datetime, Symbol, BucketNo, VPIN)
# 2. 평균값 계산법: 그 월에 생성된 VPIN의 총합 / 그 월의 생성된 VPIN 수
# 3. 리턴값 형태: {Symbol: string, Date: Date, monthly_VPIN: float64}
# 4. 해당 달에 속한 일자의 Monthly_VPIN은 같은 값을 갖게 됨
# =============================================================================
def calculate_monthly_vpin_average(vpin_df: pl.DataFrame) -> pl.DataFrame:
    """
    버킷 넘버별 VPIN 데이터를 입력받아 월별 VPIN 평균을 계산합니다.
    
    Parameters:
    -----------
    vpin_df : pl.DataFrame
        Schema({'Datetime': Datetime(time_unit='us', time_zone=None), 
                'Symbol': String, 
                'BucketNo': Int64, 
                'VPIN': Float64})
    
    Returns:
    --------
    pl.DataFrame
        Schema({'Symbol': String, 
                'Date': Date, 
                'Month': Date,
                'monthly_VPIN': Float64})
        해당 월에 속한 모든 일자는 같은 monthly_VPIN 값과 Month 값을 가짐
        Month 컬럼은 해당 월의 마지막 영업일(거래가 있는 날) 날짜
    """
    
    # -------------------------------------------------------------------------
    # 1. 데이터 검증 및 전처리
    # -------------------------------------------------------------------------
    if vpin_df.is_empty():
        return pl.DataFrame(schema={
            "Symbol": pl.String,
            "Date": pl.Date,
            "Month": pl.Date,
            "monthly_VPIN": pl.Float64
        })
    
    # NaN 값 제거 (VPIN이 유효한 값만 사용)
    vpin_df = vpin_df.filter(pl.col("VPIN").is_not_nan())
    
    # -------------------------------------------------------------------------
    # 2. Datetime을 Date로 변환 및 연도/월 추출
    # -------------------------------------------------------------------------
    df_with_date = vpin_df.with_columns([
        pl.col("Datetime").cast(pl.Date).alias("Date")
    ])
    
    # 연도와 월 추출
    df_with_month = df_with_date.with_columns([
        pl.col("Date").dt.year().alias("Year"),
        pl.col("Date").dt.month().alias("Month")
    ])
    
    # -------------------------------------------------------------------------
    # 3. 월별 VPIN 평균 계산
    # -------------------------------------------------------------------------
    # Symbol, Year, Month 기준으로 그룹화하여 VPIN 평균 계산
    monthly_vpin = df_with_month.group_by(["Symbol", "Year", "Month"]).agg([
        pl.col("VPIN").mean().alias("monthly_VPIN")
    ])
    
    # -------------------------------------------------------------------------
    # 4. 각 월의 마지막 영업일 계산
    # -------------------------------------------------------------------------
    # Symbol, Year, Month별 마지막 영업일(거래가 있는 날) 찾기
    last_trading_day = df_with_month.group_by(["Symbol", "Year", "Month"]).agg([
        pl.col("Date").max().alias("LastTradingDay")
    ])
    
    # -------------------------------------------------------------------------
    # 5. 각 날짜에 월별 VPIN 값 및 마지막 영업일 매핑
    # -------------------------------------------------------------------------
    # 원본 데이터에서 Symbol과 Date 추출 (중복 제거)
    dates_df = df_with_date.select(["Symbol", "Date"]).unique()
    
    # 날짜에서 Year, Month 추출
    dates_df = dates_df.with_columns([
        pl.col("Date").dt.year().alias("Year"),
        pl.col("Date").dt.month().alias("Month_num")
    ])
    
    # 월별 VPIN 값을 각 날짜에 조인
    result_df = dates_df.join(
        monthly_vpin,
        left_on=["Symbol", "Year", "Month_num"],
        right_on=["Symbol", "Year", "Month"],
        how="left"
    )
    
    # 마지막 영업일 정보 조인
    result_df = result_df.join(
        last_trading_day,
        left_on=["Symbol", "Year", "Month_num"],
        right_on=["Symbol", "Year", "Month"],
        how="left"
    ).select([
        "Symbol",
        "Date",
        pl.col("LastTradingDay").alias("Month"),
        "monthly_VPIN"
    ]).sort(["Symbol", "Date"])
    
    return result_df


# =============================================================================
# 실행부
# =============================================================================
if __name__ == "__main__":
    # 경로 설정
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 앞 단계에서 생성된 VPIN 결과 파일 로드
    input_filename = "03_vpin_Base_result_20260203_0122.parquet" 
    input_path = os.path.join(project_root, "output_data", input_filename)
    output_dir = os.path.join(project_root, "output_data")
    
    if os.path.exists(input_path):
        print(f"Loading VPIN data from: {input_path}")
        vpin_df = pl.read_parquet(input_path)
        
        # 함수 실행
        result_df = calculate_monthly_vpin_average(vpin_df)
        
        # 파일명 동적 생성 (스크립트명 기반)
        script_name = os.path.splitext(os.path.basename(__file__))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_name = f"{script_name}_result_{timestamp}"
        
        # Parquet 저장
        output_parquet = os.path.join(output_dir, f"{output_name}.parquet")
        result_df.write_parquet(output_parquet)
        print(f"Saved Monthly Aggregated VPIN: {output_parquet}")
        
        # 샘플 CSV 저장
        csv_path = os.path.join(output_dir, f"{output_name}_SAMPLE.csv")
        result_df.head(1000).write_csv(csv_path)
        print(f"Saved Sample CSV: {csv_path}")

        # 미리보기
        print("\n[Preview - Monthly VPIN]")
        print(result_df.head(20))
        
        # 통계
        print(f"\n[Statistics]")
        print(f"Total symbols processed: {result_df['Symbol'].n_unique()}")
        print(f"Total date entries: {result_df.height}")
        print(f"Monthly VPIN range: {result_df['monthly_VPIN'].min():.6f} ~ {result_df['monthly_VPIN'].max():.6f}")
        
    else:
        print(f"Input file not found: {input_path}")