"""
SAS7BDAT → Parquet 배치 변환 스크립트
======================================

[개요]
한국 주식 시장 틱 데이터(SAS7BDAT 형식)를 Polars에 최적화된 Parquet 형식으로
청크(Chunk) 단위로 변환합니다. 메모리 효율을 위해 500만 행 단위로 읽고 씁니다.

[원본 데이터 컬럼 구조]
- Price    : float64  — 체결 가격
- Volume   : float64  — 거래량
- Symbol   : str      — 종목 코드 (공백 제거 후 저장)
- Date     : SAS 날짜 숫자 (1960-01-01 기준 경과 일수, float)
- Time     : SAS 시간 숫자 (자정 이후 경과 초, float, 소수점 이하 = 나노초 정밀도)
- MidPoint : float64  — 중간 호가
- QSpread  : float64  — 호가 스프레드
- LR       : int8     — 매수/매도 방향 (-1 or 1)

Price    : float64 / Volume   : float64 /Symbol   : str/  Date :date32 / Time : time64 /LR :int8

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
  입력: {BASE_INPUT_DIR}/{folder}/*.sas7bdat
  출력: {BASE_OUTPUT_DIR}/{folder}/*.parquet   ← 폴더명 그대로 유지

  예) E:\\vpin_project_sas7bdat\\KOR_2019\\KOR_201901.sas7bdat
   →  E:\\vpin_project_parquet\\KOR_2019\\KOR_201901.parquet
"""

import re
import pyreadstat
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import time
from pathlib import Path

# ==========================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# ==========================================

# SAS 원본 루트 폴더
BASE_INPUT_DIR  = Path(r"E:\vpin_project_sas7bdat")

# Parquet 출력 루트 폴더 (하위 파이프라인의 BASE_DIR과 동일)
BASE_OUTPUT_DIR = Path(r"E:\vpin_project_parquet")

CHUNK_SIZE = 5_000_000  # 500만 행 단위

# ==========================================
# (이하 수정 불필요)
# ==========================================

# 입력 폴더명 패턴: {나라코드}_{연도}  예) KOR_2019, US_2020
_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")


def process_chunk_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAS7BDAT에서 읽은 청크 DataFrame을 Polars 호환 Parquet 저장을 위해 변환합니다.

    변환 상세:
    - Date  : SAS 날짜(float, 1960-01-01 기준 경과 일수) → Python date 객체
              .dt.date를 사용해 date32 호환 타입으로 변환 (timestamp 경유 캐스팅 오류 방지)
    - Time  : SAS 시간(float, 자정 이후 경과 초) → 나노초 int64
              build_arrow_table()에서 pa.time64('ns')로 직접 배열 생성
    - Symbol: NaN → null 보존 후 문자열 공백 제거 ('nan' 문자열 방지)
    - LR    : NaN → null 보존 후 Int8 변환 (build_arrow_table에서 None 명시 변환)
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
        # NaN이 있으면 astype(str) 시 'nan' 문자열이 되므로
        # 먼저 원본 NaN을 None으로 보존한 뒤 문자열 변환·공백 제거
        df['Symbol'] = df['Symbol'].where(df['Symbol'].notna()).astype(str).str.strip()
        df['Symbol'] = df['Symbol'].where(df['Symbol'] != 'nan')

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

    - Date   : pa.date32()      → Polars pl.Date (null 보존)
    - Time   : pa.time64('ns')  → Polars pl.Time (나노초 정밀도 및 null 보존)
               pd.NA → None 변환 후 pa.array() 생성 (PyArrow는 pd.NA 미인식)
    - LR     : pa.int8()        → Polars pl.Int8  (pd.NA → None 명시 변환)
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
        elif col == 'LR':
            # Int8(nullable) → pd.NA는 PyArrow 미인식이므로 None으로 변환
            lr_list = [None if v is pd.NA else int(v) for v in df['LR']]
            arr = pa.array(lr_list, type=pa.int8())
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
    BASE_INPUT_DIR 아래의 모든 하위 폴더를 순회하여
    SAS 파일을 같은 이름의 출력 폴더에 Parquet으로 변환합니다.

      입력: BASE_INPUT_DIR/{folder}/*.sas7bdat
      출력: BASE_OUTPUT_DIR/{folder}/*.parquet

    - 이미 변환된 파일(.parquet 존재)은 건너뜁니다 (중단 후 재실행 안전).
    - SAS 파일이 없는 폴더는 건너뜁니다.
    """
    total_start_time = time.time()
    total_files_processed = 0

    print(f"\n{'='*60}")
    print(f"[SAS → Parquet 전체 변환]")
    print(f"  입력 루트: {BASE_INPUT_DIR}")
    print(f"  출력 루트: {BASE_OUTPUT_DIR}")
    print(f"{'='*60}")

    folders = sorted(p for p in BASE_INPUT_DIR.iterdir()
                     if p.is_dir() and _FOLDER_RE.match(p.name))
    if not folders:
        print("[오류] 하위 폴더를 찾을 수 없습니다. BASE_INPUT_DIR을 확인하세요.")
        return

    for folder in folders:
        sas_files = sorted(folder.glob("*.sas7bdat"))
        if not sas_files:
            continue

        output_dir = BASE_OUTPUT_DIR / folder.name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"[{folder.name}] {len(sas_files)}개 파일  →  {output_dir}")
        print(f"{'='*60}")

        folder_processed = 0
        for sas_file in sas_files:
            output_file = output_dir / f"{sas_file.stem}.parquet"

            if output_file.exists():
                print(f"  [스킵] {output_file.name}")
                continue

            convert_single_file(sas_file, output_file)
            folder_processed += 1
            total_files_processed += 1

        print(f"\n  [{folder.name} 완료] {folder_processed}개 변환")

    total_end_time = time.time()
    print(f"\n{'*'*60}")
    print(f"[모든 작업 완료] 총 {total_files_processed}개 파일 변환")
    print(f"총 소요 시간: {(total_end_time - total_start_time) / 60:.2f} 분")
    print(f"{'*'*60}")


if __name__ == "__main__":
    run_batch_conversion()