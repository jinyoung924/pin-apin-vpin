import os
import warnings
from datetime import datetime

import polars as pl

# 경고 메시지 무시
warnings.filterwarnings('ignore')


# =============================================================================
# 주별 VPIN 평균 계산 함수
# -----------------------------------------------------------------------------
# [요구사항]
# 1. 입력: 버킷당 VPIN 데이터 (Datetime, Symbol, BucketNo, VPIN)
# 2. 평균값 계산법: 그 주에 생성된 VPIN의 총합 / 그 주의 생성된 VPIN 수
# 3. 리턴값 형태: {Symbol: string, Date: Date, Week: Date, weekly_VPIN: float64}
# 4. Week 칼럼: 계산하려는 주의 월요일을 기준으로 작성
# =============================================================================
def calculate_weekly_vpin_average(vpin_df: pl.DataFrame) -> pl.DataFrame:
    """
    버킷 넘버별 VPIN 데이터를 입력받아 주별 VPIN 평균을 계산합니다.
    
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
                'Week': Date, 
                'weekly_VPIN': Float64})
        Week 칼럼은 해당 주의 월요일 날짜를 기준으로 함
    """
    
    # -------------------------------------------------------------------------
    # 1. 데이터 검증 및 전처리
    # -------------------------------------------------------------------------
    if vpin_df.is_empty():
        return pl.DataFrame(schema={
            "Symbol": pl.String,
            "Date": pl.Date,
            "Week": pl.Date,
            "weekly_VPIN": pl.Float64
        })
    
    # NaN 값 제거 (VPIN이 유효한 값만 사용)
    vpin_df = vpin_df.filter(pl.col("VPIN").is_not_nan())
    
    # -------------------------------------------------------------------------
    # 2. Datetime을 Date로 변환 및 주(Week) 계산
    # -------------------------------------------------------------------------
    # Datetime을 Date로 변환하고 해당 주의 월요일 계산
    df_with_week = vpin_df.with_columns([
        pl.col("Datetime").cast(pl.Date).alias("Date")
    ]).with_columns([
        # 요일 번호 계산 (0=월요일, 6=일요일)
        pl.col("Date").dt.weekday().alias("weekday")
    ]).with_columns([
        # 해당 주의 월요일 날짜 계산
        (pl.col("Date") - pl.duration(days=pl.col("weekday")-1)).alias("Week")
    ])
    
    # -------------------------------------------------------------------------
    # 3. 주별 VPIN 평균 계산
    # -------------------------------------------------------------------------
    # Symbol과 Week 기준으로 그룹화하여 VPIN 평균 계산
    weekly_vpin = df_with_week.group_by(["Symbol", "Week"]).agg([
        pl.col("VPIN").mean().alias("weekly_VPIN")
    ])
    
    # -------------------------------------------------------------------------
    # 4. 각 날짜에 주별 VPIN 값 매핑
    # -------------------------------------------------------------------------
    # 원본 데이터에서 Symbol과 Date 추출 (중복 제거)
    dates_df = df_with_week.select(["Symbol", "Date", "Week"]).unique()
    
    # 주별 VPIN 값을 각 날짜에 조인
    result_df = dates_df.join(
        weekly_vpin,
        on=["Symbol", "Week"],
        how="left"
    ).select([
        "Symbol",
        "Date",
        "Week",
        "weekly_VPIN"
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
        result_df = calculate_weekly_vpin_average(vpin_df)
        
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
        print("\n[Preview - Weekly VPIN]")
        print(result_df.head(20))
        
        # 통계
        print(f"\n[Statistics]")
        print(f"Total symbols processed: {result_df['Symbol'].n_unique()}")
        print(f"Total date entries: {result_df.height}")
        print(f"Weekly VPIN range: {result_df['weekly_VPIN'].min():.6f} ~ {result_df['weekly_VPIN'].max():.6f}")
        
    else:
        print(f"Input file not found: {input_path}")