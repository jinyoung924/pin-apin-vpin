import polars as pl

# 경로 앞에 r을 붙여 백슬래시(\)를 일반 문자로 처리합니다.
file_path = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\all_1m_bars.parquet"

bars = pl.read_parquet(file_path)

weekend_bars = bars.filter(
    pl.col("Datetime").dt.weekday().is_in([6, 7])
)

print(f"주말 봉 수: {weekend_bars.height}")
print(weekend_bars.head(20))