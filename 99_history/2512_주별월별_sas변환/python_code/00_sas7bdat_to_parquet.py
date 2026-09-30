"""
SAS to Parquet Converter (Polars Optimized)
===========================================

이 스크립트는 대용량 SAS 데이터(.sas7bdat)를 Polars에서 효율적으로 사용할 수 있도록
Parquet 포맷으로 변환하는 도구입니다.

[주요 기능]
1. Chunk 단위 처리: 대용량 파일도 메모리 초과 없이 변환 가능 (기본 100만 행 단위)
2. 데이터 타입 최적화:
   - Date: SAS 날짜 -> Parquet Date32 (YYYY-MM-DD)
   - Time: SAS 시간(초) -> Parquet Time64 (나노초 정밀도 유지)
   - 문자열 및 숫자형 데이터 최적화
3. Polars 호환성: 변환된 파일은 Polars에서 `pl.scan_parquet()`로 즉시 로드 가능

[사용 방법]
1. 아래 `input_sas_file` 변수에 원본 SAS 파일 경로를 지정하세요.
2. `output_parquet_file` 변수에 저장할 Parquet 파일 경로를 지정하세요.
3. 터미널에서 스크립트 실행:
   $ python sas7bdat_to_parquet.py

[필수 라이브러리]
pip install pandas pyarrow pyreadstat polars

작성일: 2026-01-15
"""

import pyreadstat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
import os

# ==========================================
# 사용자 설정 (파일 경로 수정 필요)
# ==========================================
# 입력 파일: 변환할 SAS 데이터 파일 경로
input_sas_file = r'C:\SAS_Project\Data\kor_201001.sas7bdat'

# 출력 파일: 저장될 Parquet 파일 경로
output_parquet_file = r'C:\SAS_Project\Data\kor_201001.parquet'

CHUNK_SIZE = 1_000_000  # 100만 행 (메모리 최적화 설정)

def process_chunk_for_polars(df):
    """
    [Polars 최적화 전처리]
    1. Date: 실제 날짜 객체(Date)로 변환
    2. Time: 나노초 정밀도 정수(Int64)로 변환
    """
    
    # 1. Date 변환: SAS 날짜 -> Pandas Timestamp 객체
    if 'Date' in df.columns:
        df['Date'] = df['Date'].fillna(0)
        # SAS 기준(1960년)으로 날짜 변환
        df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1960-01-01')

    # 2. Time 변환: Float(초) -> Int64(나노초)
    if 'Time' in df.columns:
        df['Time'] = (df['Time'] * 1_000_000_000).round().astype('int64')

    # 3. 기타 최적화
    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.strip()
    if 'LR' in df.columns:
        df['LR'] = df['LR'].fillna(0).astype('int8')
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype('float64')

    return df

def convert_sas_to_parquet_polars():
    start_time = time.time()
    
    # 파일 존재 여부 확인
    if not os.path.exists(input_sas_file):
        print(f"[오류] 입력 파일을 찾을 수 없습니다: {input_sas_file}")
        return

    print(f"Polars용 변환 시작 (Date: 날짜 객체 / Time: 나노초 유지): {input_sas_file}")
    
    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat, 
        input_sas_file, 
        chunksize=CHUNK_SIZE, 
        disable_datetime_conversion=True 
    )
    
    writer = None
    total_rows = 0
    
    try:
        for i, (df, meta) in enumerate(reader):
            chunk_start = time.time()
            
            # 데이터 변환
            df = process_chunk_for_polars(df)
            
            # Pandas -> PyArrow Table 변환
            table = pa.Table.from_pandas(df)
            
            # === [스키마 강제 지정: Date와 Time] ===
            new_schema = table.schema
            
            # (1) Date 컬럼: Timestamp -> Date32 (YYYY-MM-DD)
            date_idx = table.schema.get_field_index('Date')
            if date_idx != -1:
                field = pa.field('Date', pa.date32())
                new_schema = new_schema.set(date_idx, field)

            # (2) Time 컬럼: Int64 -> Time64(ns)
            time_idx = table.schema.get_field_index('Time')
            if time_idx != -1:
                field = pa.field('Time', pa.time64('ns'))
                new_schema = new_schema.set(time_idx, field)
            
            # 변경된 스키마로 데이터 캐스팅
            table = table.cast(new_schema)

            # Writer 생성 및 쓰기
            if writer is None:
                print("\n[Parquet 최종 스키마 확인]")
                for field in table.schema:
                    print(f" - {field.name}: {field.type}")
                print("-" * 50)
                writer = pq.ParquetWriter(output_parquet_file, table.schema, compression='snappy')
            
            writer.write_table(table)
            
            rows = len(df)
            total_rows += rows
            print(f"Chunk {i+1} 처리 완료: {rows:,} 행 ({time.time()-chunk_start:.2f}초)")

    except Exception as e:
        print(f"\n[오류] {e}")
        raise e
    
    finally:
        if writer:
            writer.close()

    end_time = time.time()
    print(f"\n[완료] 총 {total_rows:,} 행 변환됨. (총 소요 시간: {end_time - start_time:.2f}초)")
    print(f"저장된 파일: {output_parquet_file}")

if __name__ == "__main__":
    convert_sas_to_parquet_polars()