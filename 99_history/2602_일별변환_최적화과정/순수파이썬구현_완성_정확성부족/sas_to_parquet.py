"""
=============================================================================
SAS7BDAT → Parquet 배치 변환 모듈
=============================================================================

한국 주식 시장 틱 데이터(SAS7BDAT 형식)를 Polars에 최적화된 Parquet 형식으로
청크(Chunk) 단위로 변환합니다.

[원본 데이터 컬럼 구조]
- Price    : float64  — 체결 가격
- Volume   : float64  — 거래량
- Symbol   : str      — 종목 코드 (공백 제거 후 저장)
- Date     : SAS 날짜 숫자 (1960-01-01 기준 경과 일수, float)
- Time     : SAS 시간 숫자 (자정 이후 경과 초, float)
- MidPoint : float64  — 중간 호가
- QSpread  : float64  — 호가 스프레드
- LR       : int8     — 매수/매도 방향 (-1 or 1)

[핵심 타입 변환 로직]
  ① Date: SAS 날짜(float) → pd.to_datetime → .dt.date → pa.date32()
  ② Time: SAS 시간(float, 초) → 나노초 int64 → pa.time64('ns')
"""

import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

import config


def process_chunk_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """SAS7BDAT에서 읽은 청크를 Polars 호환 타입으로 변환한다."""
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], unit='D', origin='1960-01-01').dt.date

    if 'Time' in df.columns:
        df['Time'] = (df['Time'] * 1_000_000_000).round().astype('Int64')

    if 'Symbol' in df.columns:
        df['Symbol'] = df['Symbol'].astype(str).str.strip()

    if 'LR' in df.columns:
        df['LR'] = df['LR'].astype('Int8')

    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].astype('float64')

    return df


def build_arrow_table(df: pd.DataFrame) -> pa.Table:
    """변환된 DataFrame을 PyArrow Table로 변환한다."""
    arrays = []
    fields = []

    for col in df.columns:
        if col == 'Date':
            arr = pa.array(df['Date'].tolist(), type=pa.date32())
        elif col == 'Time':
            time_list = [None if v is pd.NA else int(v) for v in df['Time']]
            arr = pa.array(time_list, type=pa.time64('ns'))
        else:
            arr = pa.array(df[col].tolist())

        arrays.append(arr)
        fields.append(pa.field(col, arr.type))

    schema = pa.schema(fields)
    return pa.table(dict(zip(df.columns, arrays)), schema=schema)


def convert_single_file(input_sas_file: Path, output_parquet_file: Path) -> None:
    """단일 SAS7BDAT 파일을 Parquet으로 변환한다."""
    start_time = time.time()
    print(f"\n▶ 변환 시작: {input_sas_file.name}")

    reader = pyreadstat.read_file_in_chunks(
        pyreadstat.read_sas7bdat,
        input_sas_file,
        chunksize=config.SAS_CHUNK_SIZE,
        disable_datetime_conversion=True
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
                writer = pq.ParquetWriter(output_parquet_file, table.schema,
                                          compression='snappy')

            writer.write_table(table)

            rows = len(df)
            total_rows += rows
            print(f"  - Chunk {i+1} 완료: {rows:,} 행 ({time.time()-chunk_start:.2f}초)")

    except Exception as e:
        has_error = True
        print(f"\n[오류 발생 - 파일: {input_sas_file.name}] {e}")
        raise

    finally:
        if writer:
            writer.close()
        if has_error and output_parquet_file.exists():
            output_parquet_file.unlink()
            print(f"  → 불완전한 출력 파일 삭제: {output_parquet_file.name}")

    print(f"▷ 변환 완료: 총 {total_rows:,} 행 ({time.time() - start_time:.2f}초)")


def run(input_dir: str | None = None, output_dir: str | None = None,
        start_year: int | None = None, end_year: int | None = None) -> None:
    """
    설정된 연도 범위의 모든 SAS 파일을 배치 변환한다.

    Args:
        input_dir  : SAS 파일 루트 경로 (기본: config.SAS_INPUT_DIR)
        output_dir : Parquet 출력 루트 경로 (기본: config.SAS_OUTPUT_DIR)
        start_year : 시작 연도 (기본: config.SAS_START_YEAR)
        end_year   : 종료 연도 (기본: config.SAS_END_YEAR)
    """
    base_input  = Path(input_dir  or config.SAS_INPUT_DIR)
    base_output = Path(output_dir or config.SAS_OUTPUT_DIR)
    sy = start_year or config.SAS_START_YEAR
    ey = end_year   or config.SAS_END_YEAR

    total_start_time = time.time()
    total_files_processed = 0

    for year in range(sy, ey + 1):
        folder_name = f"KOR_{year}"
        input_folder = base_input / folder_name
        output_folder = base_output / folder_name

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
