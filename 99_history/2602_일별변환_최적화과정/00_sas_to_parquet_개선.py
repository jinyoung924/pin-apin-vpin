"""
SAS7BDAT → Parquet 배치 변환 스크립트
======================================

[개요]
한국 주식 시장 틱 데이터(SAS7BDAT 형식)를 Polars에 최적화된 Parquet 형식으로
청크(Chunk) 단위로 변환합니다. 메모리 효율을 위해 100만 행 단위로 읽고 씁니다.

[원본 데이터 컬럼 구조]
- Price    : float64  — 체결 가격
- Volume   : float64  — 거래량
- Symbol   : str      — 종목 코드 (공백 제거 후 저장)
- Date     : SAS 날짜 숫자 (1960-01-01 기준 경과 일수, float)
- Time     : SAS 시간 숫자 (자정 이후 경과 초, float, 소수점 이하 = 나노초 정밀도)
- MidPoint : float64  — 중간 호가
- QSpread  : float64  — 호가 스프레드
- LR       : int8     — 매수/매도 방향 (-1 or 1)

[핵심 타입 변환 로직]

  ① Date (SAS 날짜 → Polars `pl.Date` / Parquet `date32`)
     - SAS 날짜는 1960-01-01을 기준(origin)으로 경과한 일수를 float으로 저장
     - pd.to_datetime(value, unit='D', origin='1960-01-01')으로 datetime64[ns] 생성 후
       .dt.date로 Python date 객체 배열로 변환
     - PyArrow에 date 객체 리스트로 넘기면 자동으로 date32 타입으로 인식
     - Polars에서 pl.Date 타입으로 로드됨
     ※ timestamp(datetime64)를 그대로 PyArrow에 넘기면 timestamp[ns]로 인식되어
       date32 캐스팅 오류가 발생할 수 있으므로 반드시 .dt.date 변환 필요

  ② Time (SAS 시간 → Polars `pl.Time` / Parquet `time64('ns')`)
     - SAS 시간은 자정(00:00:00)으로부터 경과한 초(seconds)를 float으로 저장
       예) 09:31:40.914478218 → 34300.914478218 (초)
     - 나노초 단위로 변환: round(seconds * 1_000_000_000) → int64
     - pa.array(ns_list, type=pa.time64('ns'))로 직접 생성
     ※ int64 컬럼을 PyArrow Table 생성 후 cast()로 time64로 변환하는 방식은
       PyArrow 버전에 따라 지원되지 않아 오류 발생 가능 → pa.array() 직접 생성 방식 사용

[변환 결과 예시]
  Date       | Time               | Symbol   | Price   | Volume | LR
  -----------|--------------------|----------|---------|--------|----
  2017-01-17 | 09:31:40.914478218 | "258540" | 16950.0 | 2.0    | -1
  2017-01-18 | 09:21:45.197972089 | "258540" | 14450.0 | 1.0    | -1

[파일 구조]
  E:\\vpin_project_sas7bdat\\KOR_{year}\\*.sas7bdat  (입력)
  E:\\vpin_project_parquet\\KOR_{year}\\*.parquet    (출력, Snappy 압축)
"""

import pyreadstat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
from pathlib import Path

# ==========================================
# 사용자 설정 (디렉토리 경로 설정)
# ==========================================
BASE_INPUT_DIR = Path(r"E:\vpin_project_sas7bdat")
BASE_OUTPUT_DIR = Path(r"E:\vpin_project_parquet")

START_YEAR = 2015
END_YEAR = 2016

CHUNK_SIZE = 5_000_000  # 100만 행 단위


def process_chunk_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAS7BDAT에서 읽은 청크 DataFrame을 Polars 호환 Parquet 저장을 위해 변환합니다.

    변환 상세:
    - Date  : SAS 날짜(float, 1960-01-01 기준 경과 일수) → Python date 객체
              .dt.date를 사용해 date32 호환 타입으로 변환 (timestamp 경유 캐스팅 오류 방지)
    - Time  : SAS 시간(float, 자정 이후 경과 초) → 나노초 int64
              build_arrow_table()에서 pa.time64('ns')로 직접 배열 생성
    - Symbol: 문자열 앞뒤 공백 제거
    - LR    : NaN → 0 후 int8 변환 (-1 또는 1)
    - Volume: float64 명시

    Parameters
    ----------
    df : pd.DataFrame
        pyreadstat로 읽은 원본 청크 (disable_datetime_conversion=True 상태)

    Returns
    -------
    pd.DataFrame
        타입 변환이 완료된 청크
    """
    if 'Date' in df.columns:
        # fillna(0) 제거: pd.to_datetime()은 NaN을 NaT로 처리하고,
        # .dt.date는 NaT를 None으로 변환하므로 null이 자연스럽게 보존됨
        df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1960-01-01').dt.date

    if 'Time' in df.columns:
        # SAS 시간(자정 이후 경과 초, float) → 나노초 int64
        # 예) 34300.914478218초 → 34300914478218 나노초
        # NaN이 있는 float에 직접 .astype('int64')하면 오류 발생하므로
        # pandas nullable 정수 타입 'Int64'(대문자)로 변환하여 null 보존
        # 실제 pa.time64('ns') 배열 생성은 build_arrow_table()에서 수행
        df['Time'] = (df['Time'] * 1_000_000_000).round().astype('Int64')

    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.strip()

    if 'LR' in df.columns:
        # fillna(0) 제거: LR은 -1(매도) 또는 1(매수)만 유효한 값이므로
        # 0으로 채우면 의미 없는 값이 데이터에 섞임 → null로 보존
        # pandas nullable 정수 타입 'Int8'(대문자)로 변환하여 null 보존
        df['LR'] = df['LR'].astype('Int8')

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype('float64')

    return df


def build_arrow_table(df: pd.DataFrame) -> pa.Table:
    """
    변환된 DataFrame을 최종 PyArrow Table로 변환합니다.

    Date, Time 컬럼은 pandas → PyArrow 자동 변환이 불안정하므로
    해당 컬럼만 pa.array()로 타입을 명시하여 생성합니다.

    - Date : pa.date32()      → Polars pl.Date (null 보존)
    - Time : pa.time64('ns')  → Polars pl.Time (나노초 정밀도 및 null 보존)
              pd.NA → None 변환 후 pa.array() 생성 (PyArrow는 pd.NA 미인식)
    나머지 컬럼은 pandas → PyArrow 자동 변환에 위임합니다.

    Parameters
    ----------
    df : pd.DataFrame
        process_chunk_for_polars() 처리가 완료된 청크

    Returns
    -------
    pa.Table
        최종 스키마가 확정된 PyArrow Table
    """
    arrays = []
    fields = []

    for col in df.columns:
        if col == 'Date':
            # Python date 객체 리스트 → pa.date32()로 명시 생성
            arr = pa.array(df['Date'].tolist(), type=pa.date32())
        elif col == 'Time':
            # 나노초 Int64(nullable) 리스트 → pa.time64('ns')로 명시 생성
            # cast() 방식은 PyArrow 버전에 따라 int64 → time64 직접 변환 미지원 오류 발생 가능
            # pd.NA는 PyArrow가 인식하지 못하므로 None으로 변환하여 null 보존
            time_list = [None if v is pd.NA else int(v) for v in df['Time']]
            arr = pa.array(time_list, type=pa.time64('ns'))
        else:
            arr = pa.array(df[col].tolist())

        arrays.append(arr)
        fields.append(pa.field(col, arr.type))

    schema = pa.schema(fields)
    return pa.table(dict(zip(df.columns, arrays)), schema=schema)


def convert_single_file(input_sas_file: Path, output_parquet_file: Path) -> None:
    """
    단일 SAS7BDAT 파일을 Parquet으로 변환합니다.

    청크 단위로 읽어 메모리 사용량을 제한하며, 각 청크를 동일한
    ParquetWriter에 순차적으로 append합니다.
    오류 발생 시 불완전한 출력 파일을 자동으로 삭제합니다.

    Parameters
    ----------
    input_sas_file : Path
        변환할 원본 .sas7bdat 파일 경로
    output_parquet_file : Path
        생성될 .parquet 파일 경로
    """
    start_time = time.time()
    print(f"\n▶ 변환 시작: {input_sas_file.name}")

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat,
        input_sas_file,
        chunksize=CHUNK_SIZE,
        disable_datetime_conversion=True  # SAS 날짜/시간을 원시 숫자로 읽어 직접 변환
    )

    writer = None
    total_rows = 0
    has_error = False

    try:
        for i, (df, meta) in enumerate(reader):
            chunk_start = time.time()

            df = process_chunk_for_polars(df)
            table = build_arrow_table(df)

            if writer is None:
                writer = pq.ParquetWriter(output_parquet_file, table.schema, compression='snappy')

            writer.write_table(table)

            rows = len(df)
            total_rows += rows
            print(f"  - Chunk {i+1} 완료: {rows:,} 행 ({time.time()-chunk_start:.2f}초)")

    except Exception as e:
        has_error = True
        print(f"\n[오류 발생 - 파일: {input_sas_file.name}] {e}")
        raise

    finally:
        # finally에서만 close() 호출하여 except + finally 이중 호출 방지
        if writer:
            writer.close()
        # 에러 발생 시 불완전한 파일 삭제
        if has_error and output_parquet_file.exists():
            output_parquet_file.unlink()
            print(f"  → 불완전한 출력 파일 삭제: {output_parquet_file.name}")

    print(f"▷ 변환 완료: 총 {total_rows:,} 행 ({time.time() - start_time:.2f}초)")


def run_batch_conversion() -> None:
    """
    설정된 연도 범위(START_YEAR ~ END_YEAR)의 모든 SAS 파일을 배치 변환합니다.

    디렉토리 구조:
      {BASE_INPUT_DIR}/KOR_{year}/*.sas7bdat  →  {BASE_OUTPUT_DIR}/KOR_{year}/*.parquet

    - 이미 변환된 파일(.parquet 존재)은 건너뜁니다 (중단 후 재실행 안전).
    - 원본 연도 폴더가 없으면 경고 후 다음 연도로 넘어갑니다.
    """
    total_start_time = time.time()
    total_files_processed = 0

    for year in range(START_YEAR, END_YEAR + 1):
        folder_name = f"KOR_{year}"
        input_folder = BASE_INPUT_DIR / folder_name
        output_folder = BASE_OUTPUT_DIR / folder_name

        if not input_folder.exists():
            print(f"\n[건너뜀] 폴더를 찾을 수 없습니다: {input_folder}")
            continue

        output_folder.mkdir(parents=True, exist_ok=True)

        sas_files = sorted(list(input_folder.glob("*.sas7bdat")))

        if not sas_files:
            print(f"\n[안내] {folder_name} 폴더에 변환할 sas7bdat 파일이 없습니다.")
            continue

        print(f"\n{'='*50}")
        print(f"[{year}년도 데이터 변환 시작] - 총 {len(sas_files)}개 파일")
        print(f"{'='*50}")

        for sas_file in sas_files:
            output_file = output_folder / f"{sas_file.stem}.parquet"

            if output_file.exists():
                print(f"\n[스킵] 이미 변환된 파일이 존재합니다: {output_file.name}")
                continue

            convert_single_file(sas_file, output_file)
            total_files_processed += 1

    total_end_time = time.time()
    print(f"\n{'*'*50}")
    print(f"[모든 작업 완료] 총 {total_files_processed}개의 파일 변환 성공!")
    print(f"총 소요 시간: {(total_end_time - total_start_time) / 60:.2f} 분")
    print(f"{'*'*50}")


if __name__ == "__main__":
    run_batch_conversion()