import pandas as pd

# 파일 경로를 지정해주세요 (예: 'vpin_result.parquet')
file_path = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\all_1m_bars.parquet"

# pyarrow 엔진을 사용하여 데이터를 데이터프레임으로 불러옵니다.
df = pd.read_parquet(file_path, engine='pyarrow')

# 1. 상위 5개 행 확인 (데이터가 잘 들어왔는지 시각적 확인)
print("=== 데이터 미리보기 ===")
print(df.head())

# 2. 데이터 구조 및 메모리 사용량 확인 (결측치, 데이터 타입 등)
print("\n=== 데이터 정보 ===")
print(df.info())