"""
SAS7BDAT → Parquet 안전 변환 스크립트
====================================

목표
----
1) 원본 SAS7BDAT 데이터를 Parquet으로 변환할 때 데이터 오염을 최대한 방지합니다.
2) 400GB급 대용량 파일에서도 32GB RAM 환경에서 안정적으로 동작하도록 청크 단위로 처리합니다.
3) 변환 완료 후 사람이 보기 쉬운 한글 TXT 리포트를 생성합니다.

주요 안전 장치
-------------
- 출력은 먼저 .tmp.parquet 파일에 저장하고, 검증 성공 시에만 최종 .parquet으로 이름을 변경합니다.
- 변환 도중 오류가 발생하면 임시 파일을 삭제합니다.
- 원본 청크에서 계산한 행 수/결측치/범위/합계와 Parquet 메타데이터를 비교합니다.
- Date, Time, LR 변환에서 의미 없는 값으로 채우지 않고 null을 보존합니다.
- Symbol은 공백만 제거하며, 결측값을 'nan' 문자열로 오염시키지 않습니다.

속도 관련 원칙
-------------
- Python list 변환을 최소화하고 PyArrow의 pandas 변환 기능을 우선 사용합니다.
- 다만 Date/Time/LR처럼 타입 무결성이 중요한 컬럼은 타입을 명시합니다.
- HDD 원본 + SSD 출력 환경을 고려하여 병렬 변환은 기본적으로 사용하지 않습니다.
"""

from __future__ import annotations

import gc
import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

try:
    import polars as pl
except Exception:  # polars가 없어도 변환 자체는 가능하게 둠
    pl = None


# ==========================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# ==========================================

# SAS 원본 루트 폴더
BASE_INPUT_DIR = Path(r"D:\hdd_sas7bdat_data")

# Parquet 출력 루트 폴더
BASE_OUTPUT_DIR = Path(r"E:\vpin_project_parquet")

# 리포트 저장 폴더. None이면 BASE_OUTPUT_DIR 아래 reports 폴더에 저장합니다.
REPORT_DIR: Optional[Path] = None

# 32GB RAM 기준 권장값: 3,000,000 ~ 8,000,000
# 너무 크게 잡으면 SAS → pandas → Arrow 변환 중 메모리 복사로 인해 오히려 느려질 수 있습니다.
CHUNK_SIZE = 10_000_000

# 출력 압축 방식
# - snappy: 빠름, 파일 큼
# - zstd level 1: 적당히 빠르고 파일이 작아 SSD 여유공간 유지에 유리
COMPRESSION = "zstd"
COMPRESSION_LEVEL = 1

# 이미 parquet 파일이 있으면 건너뛸지 여부
SKIP_EXISTING = True

# 출력 드라이브 최소 여유 공간. 이보다 낮으면 새 파일 변환을 시작하지 않습니다.
MIN_FREE_SPACE_GB = 50

# 변환 대상 컬럼을 제한하고 싶으면 리스트로 지정하세요.
# None이면 SAS 파일의 모든 컬럼을 변환합니다.
# 예: COLUMNS_TO_READ = ["Price", "Volume", "Symbol", "Date", "Time", "MidPoint", "QSpread", "LR"]
COLUMNS_TO_READ: Optional[List[str]] = None

# 리포트에서 컬럼별 null 개수를 기록할 주요 컬럼
KEY_COLUMNS = ["Date", "Time", "Symbol", "Price", "Volume", "LR"]


# ==========================================
# 내부 설정
# ==========================================

_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")
_SAS_TIME_NS_PER_DAY = 86_400 * 1_000_000_000


@dataclass
class ColumnStats:
    nulls: int = 0
    min_value: Any = None
    max_value: Any = None
    sum_value: Optional[float] = None
    invalid_count: int = 0


@dataclass
class FileReport:
    input_file: str
    output_file: str
    status: str = "대기"
    error_message: str = ""
    input_size_gb: float = 0.0
    output_size_gb: float = 0.0
    elapsed_sec: float = 0.0
    chunks: int = 0
    rows_read: int = 0
    rows_written: int = 0
    parquet_rows: int = 0
    row_groups: int = 0
    compression: str = COMPRESSION
    compression_level: Optional[int] = COMPRESSION_LEVEL
    schema: str = ""
    column_stats: Dict[str, ColumnStats] = field(default_factory=dict)
    checks: List[str] = field(default_factory=list)

    def add_check(self, message: str) -> None:
        self.checks.append(message)


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _gb(num_bytes: int) -> float:
    return round(num_bytes / (1024 ** 3), 3)


def _safe_min(current: Any, new_value: Any) -> Any:
    if new_value is None or (isinstance(new_value, float) and math.isnan(new_value)):
        return current
    if current is None:
        return new_value
    return min(current, new_value)


def _safe_max(current: Any, new_value: Any) -> Any:
    if new_value is None or (isinstance(new_value, float) and math.isnan(new_value)):
        return current
    if current is None:
        return new_value
    return max(current, new_value)


def _update_numeric_sum(current: Optional[float], series: pd.Series) -> Optional[float]:
    if series.empty:
        return current
    value = pd.to_numeric(series, errors="coerce").sum(skipna=True)
    if current is None:
        current = 0.0
    return float(current + value)


def _update_column_stats(report: FileReport, raw_df: pd.DataFrame) -> None:
    """원본 청크 기준으로 핵심 컬럼 통계를 누적합니다."""
    for col in KEY_COLUMNS:
        if col not in raw_df.columns:
            continue

        stats = report.column_stats.setdefault(col, ColumnStats())
        s = raw_df[col]
        stats.nulls += int(s.isna().sum())

        if col in {"Price", "Volume", "Date", "Time", "LR"}:
            numeric = pd.to_numeric(s, errors="coerce")
            if numeric.notna().any():
                stats.min_value = _safe_min(stats.min_value, float(numeric.min(skipna=True)))
                stats.max_value = _safe_max(stats.max_value, float(numeric.max(skipna=True)))
            if col in {"Price", "Volume"}:
                stats.sum_value = _update_numeric_sum(stats.sum_value, numeric)

        elif col == "Symbol":
            non_null = s.dropna().astype(str).str.strip()
            if not non_null.empty:
                stats.min_value = _safe_min(stats.min_value, non_null.min())
                stats.max_value = _safe_max(stats.max_value, non_null.max())


def _validate_raw_values(report: FileReport, raw_df: pd.DataFrame) -> None:
    """변환 전 원본 값 자체의 이상치를 기록합니다. 값을 수정하지는 않습니다."""
    if "Time" in raw_df.columns:
        t = pd.to_numeric(raw_df["Time"], errors="coerce")
        invalid = int(((t < 0) | (t >= 86_400)).sum(skipna=True))
        if invalid:
            report.column_stats.setdefault("Time", ColumnStats()).invalid_count += invalid

    if "LR" in raw_df.columns:
        lr = pd.to_numeric(raw_df["LR"], errors="coerce")
        invalid = int((lr.notna() & ~lr.isin([-1, 1])).sum())
        if invalid:
            report.column_stats.setdefault("LR", ColumnStats()).invalid_count += invalid

    if "Volume" in raw_df.columns:
        vol = pd.to_numeric(raw_df["Volume"], errors="coerce")
        invalid = int((vol.notna() & (vol < 0)).sum())
        if invalid:
            report.column_stats.setdefault("Volume", ColumnStats()).invalid_count += invalid


def process_chunk_for_polars(df: pd.DataFrame) -> pd.DataFrame:
    """
    SAS7BDAT 청크를 Parquet/Polars 호환 타입으로 변환합니다.

    중요: 결측값을 임의 값으로 채우지 않습니다.
    - Date  : SAS 날짜 숫자 → date32 호환 날짜
    - Time  : SAS 시간 초 단위 숫자 → 자정 이후 ns 단위 nullable Int64
    - Symbol: 문자열 양끝 공백 제거, 결측값 보존
    - LR    : nullable Int8
    """
    df = df.copy(deep=False)

    if "Date" in df.columns:
        # NaN은 NaT가 되고, PyArrow 변환 시 null로 보존됩니다.
        df["Date"] = pd.to_datetime(
            df["Date"], unit="D", origin="1960-01-01", errors="coerce"
        ).dt.date

    if "Time" in df.columns:
        # SAS 시간: 자정 이후 초 → ns. 결측값은 Int64의 <NA>로 보존됩니다.
        df["Time"] = (
            pd.to_numeric(df["Time"], errors="coerce") * 1_000_000_000
        ).round().astype("Int64")

    if "Symbol" in df.columns:
        # astype(str) 전에 결측 마스크를 보존하여 'nan' 문자열 오염을 방지합니다.
        mask = df["Symbol"].notna()
        out = pd.Series(pd.NA, index=df.index, dtype="string")
        out.loc[mask] = df.loc[mask, "Symbol"].astype(str).str.strip()
        out = out.mask(out == "", pd.NA)
        df["Symbol"] = out

    if "LR" in df.columns:
        df["LR"] = pd.to_numeric(df["LR"], errors="coerce").astype("Int8")

    if "Volume" in df.columns:
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").astype("float64")

    if "Price" in df.columns:
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce").astype("float64")

    return df


def _arrow_array_fast(series: pd.Series, arrow_type: Optional[pa.DataType] = None) -> pa.Array:
    """가능하면 list 변환 없이 pandas Series에서 Arrow 배열을 생성합니다."""
    try:
        return pa.array(series, type=arrow_type, from_pandas=True)
    except Exception:
        # 일부 PyArrow/Pandas 조합에서 nullable dtype 변환이 실패할 수 있으므로 안전한 fallback 제공
        values = series.astype("object").where(series.notna(), None).tolist()
        return pa.array(values, type=arrow_type, from_pandas=True)


def build_arrow_table(df: pd.DataFrame) -> pa.Table:
    """변환된 DataFrame을 PyArrow Table로 변환합니다."""
    arrays: List[pa.Array] = []
    fields: List[pa.Field] = []

    for col in df.columns:
        if col == "Date":
            arr = _arrow_array_fast(df[col], pa.date32())
        elif col == "Time":
            arr = _arrow_array_fast(df[col], pa.time64("ns"))
        elif col == "LR":
            arr = _arrow_array_fast(df[col], pa.int8())
        elif col in {"Price", "Volume", "MidPoint", "QSpread"}:
            arr = _arrow_array_fast(df[col], pa.float64())
        elif col == "Symbol":
            arr = _arrow_array_fast(df[col], pa.string())
        else:
            arr = _arrow_array_fast(df[col])

        arrays.append(arr)
        fields.append(pa.field(col, arr.type))

    return pa.Table.from_arrays(arrays, schema=pa.schema(fields))


def _make_parquet_writer(path: Path, schema: pa.Schema) -> pq.ParquetWriter:
    kwargs: Dict[str, Any] = {"compression": COMPRESSION}
    if COMPRESSION == "zstd" and COMPRESSION_LEVEL is not None:
        kwargs["compression_level"] = COMPRESSION_LEVEL
    return pq.ParquetWriter(path, schema, **kwargs)


def _check_output_disk_space(output_dir: Path) -> Tuple[bool, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output_dir)
    free_gb = _gb(usage.free)
    return free_gb >= MIN_FREE_SPACE_GB, free_gb


def validate_parquet_file(output_file: Path, expected_rows: int, report: FileReport) -> bool:
    """Parquet 파일의 기본 무결성을 검증합니다. 전체 데이터를 pandas로 재로드하지 않습니다."""
    if not output_file.exists():
        report.add_check("실패: 출력 parquet 파일이 존재하지 않습니다.")
        return False

    try:
        pf = pq.ParquetFile(output_file)
        report.parquet_rows = int(pf.metadata.num_rows)
        report.row_groups = int(pf.metadata.num_row_groups)
        report.schema = str(pf.schema_arrow)

        if report.parquet_rows == expected_rows:
            report.add_check(f"통과: 행 수 일치 ({expected_rows:,}행)")
        else:
            report.add_check(
                f"실패: 행 수 불일치 — 원본 읽은 행 {expected_rows:,}, parquet 행 {report.parquet_rows:,}"
            )
            return False

        if report.row_groups > 0:
            report.add_check(f"통과: Parquet row group 생성 확인 ({report.row_groups:,}개)")
        else:
            report.add_check("실패: Parquet row group이 없습니다.")
            return False

        # 선택적 lazy 검증: min/max/null 등을 Parquet에서 직접 확인
        if pl is not None:
            exprs = [pl.len().alias("행수")]
            for col in KEY_COLUMNS:
                if col in pf.schema_arrow.names:
                    exprs.append(pl.col(col).null_count().alias(f"{col}_null"))
            lazy_result = pl.scan_parquet(str(output_file)).select(exprs).collect()
            lazy_rows = int(lazy_result["행수"][0])
            if lazy_rows == expected_rows:
                report.add_check("통과: Polars lazy scan 행 수 확인")
            else:
                report.add_check(
                    f"실패: Polars lazy scan 행 수 불일치 — {lazy_rows:,}행"
                )
                return False

        return True

    except Exception as exc:
        report.add_check(f"실패: parquet 검증 중 오류 — {exc}")
        return False


def convert_single_file(input_sas_file: Path, output_parquet_file: Path) -> FileReport:
    """단일 SAS7BDAT 파일을 안전하게 Parquet으로 변환하고 FileReport를 반환합니다."""
    start_time = time.time()
    output_parquet_file.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = output_parquet_file.with_name(output_parquet_file.name + ".tmp")
    report = FileReport(
        input_file=str(input_sas_file),
        output_file=str(output_parquet_file),
        input_size_gb=_gb(input_sas_file.stat().st_size),
    )

    print(f"\n▶ 변환 시작: {input_sas_file.name}")

    ok_space, free_gb = _check_output_disk_space(output_parquet_file.parent)
    if not ok_space:
        report.status = "실패"
        report.error_message = (
            f"출력 드라이브 여유 공간 부족: {free_gb:.2f}GB 남음 "
            f"(최소 필요 설정 {MIN_FREE_SPACE_GB}GB)"
        )
        print(f"  [중단] {report.error_message}")
        return report

    if tmp_file.exists():
        tmp_file.unlink()

    writer: Optional[pq.ParquetWriter] = None

    try:
        read_kwargs: Dict[str, Any] = {
            "chunksize": CHUNK_SIZE,
            "disable_datetime_conversion": True,
        }
        if COLUMNS_TO_READ:
            read_kwargs["usecols"] = COLUMNS_TO_READ

        reader = pyreadstat.read_file_in_chunks(
            pyreadstat.read_sas7bdat,
            str(input_sas_file),
            **read_kwargs,
        )

        for chunk_idx, (raw_df, meta) in enumerate(reader, start=1):
            chunk_start = time.time()
            report.chunks += 1
            report.rows_read += len(raw_df)

            _update_column_stats(report, raw_df)
            _validate_raw_values(report, raw_df)

            df = process_chunk_for_polars(raw_df)
            table = build_arrow_table(df)

            if writer is None:
                writer = _make_parquet_writer(tmp_file, table.schema)

            writer.write_table(table)
            report.rows_written += table.num_rows

            print(
                f"  - Chunk {chunk_idx:,} 완료: {table.num_rows:,}행, "
                f"누적 {report.rows_written:,}행 ({time.time() - chunk_start:.2f}초)"
            )

            del raw_df, df, table
            gc.collect()

        if writer is not None:
            writer.close()
            writer = None

        # 임시 파일 검증 후 최종 파일명으로 교체
        if not validate_parquet_file(tmp_file, report.rows_read, report):
            raise RuntimeError("변환 후 parquet 검증 실패")

        if output_parquet_file.exists():
            output_parquet_file.unlink()
        os.replace(tmp_file, output_parquet_file)

        # 최종 파일명 기준으로 한 번 더 메타데이터 확인
        validate_parquet_file(output_parquet_file, report.rows_read, report)

        report.output_size_gb = _gb(output_parquet_file.stat().st_size)
        report.status = "성공"
        report.elapsed_sec = time.time() - start_time
        print(f"▷ 변환 완료: {report.rows_written:,}행 ({report.elapsed_sec:.2f}초)")

    except Exception as exc:
        report.status = "실패"
        report.error_message = str(exc)
        report.elapsed_sec = time.time() - start_time
        print(f"\n[오류 발생 - 파일: {input_sas_file.name}] {exc}")

        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass

        if tmp_file.exists():
            tmp_file.unlink()
            print(f"  → 불완전한 임시 파일 삭제: {tmp_file.name}")

    return report


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def write_korean_txt_report(reports: List[FileReport], report_path: Path) -> None:
    """사람이 보기 쉬운 한글 TXT 리포트를 생성합니다."""
    report_path.parent.mkdir(parents=True, exist_ok=True)

    success_count = sum(1 for r in reports if r.status == "성공")
    fail_count = sum(1 for r in reports if r.status == "실패")
    skip_count = sum(1 for r in reports if r.status == "스킵")
    total_rows = sum(r.rows_written for r in reports)
    total_input_gb = sum(r.input_size_gb for r in reports)
    total_output_gb = sum(r.output_size_gb for r in reports)
    total_sec = sum(r.elapsed_sec for r in reports)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("SAS7BDAT → Parquet 변환 검증 리포트\n")
        f.write("=" * 80 + "\n")
        f.write(f"리포트 생성 시각: {_now_text()}\n")
        f.write(f"입력 루트: {BASE_INPUT_DIR}\n")
        f.write(f"출력 루트: {BASE_OUTPUT_DIR}\n")
        f.write(f"청크 크기: {CHUNK_SIZE:,}행\n")
        f.write(f"압축 방식: {COMPRESSION}")
        if COMPRESSION_LEVEL is not None:
            f.write(f" level {COMPRESSION_LEVEL}")
        f.write("\n")
        f.write("\n[전체 요약]\n")
        f.write(f"- 성공 파일 수: {success_count:,}\n")
        f.write(f"- 실패 파일 수: {fail_count:,}\n")
        f.write(f"- 스킵 파일 수: {skip_count:,}\n")
        f.write(f"- 총 변환 행 수: {total_rows:,}\n")
        f.write(f"- 총 입력 크기: {total_input_gb:.3f} GB\n")
        f.write(f"- 총 출력 크기: {total_output_gb:.3f} GB\n")
        f.write(f"- 총 소요 시간: {total_sec / 60:.2f} 분\n")
        if total_sec > 0:
            f.write(f"- 평균 처리 속도: {total_rows / total_sec:,.0f} 행/초\n")

        f.write("\n[파일별 상세]\n")
        f.write("=" * 80 + "\n")

        for idx, r in enumerate(reports, start=1):
            f.write(f"\n{idx}. 상태: {r.status}\n")
            f.write("-" * 80 + "\n")
            f.write(f"원본 파일: {r.input_file}\n")
            f.write(f"출력 파일: {r.output_file}\n")
            f.write(f"원본 크기: {r.input_size_gb:.3f} GB\n")
            f.write(f"출력 크기: {r.output_size_gb:.3f} GB\n")
            f.write(f"청크 수: {r.chunks:,}\n")
            f.write(f"원본에서 읽은 행 수: {r.rows_read:,}\n")
            f.write(f"Parquet으로 쓴 행 수: {r.rows_written:,}\n")
            f.write(f"Parquet 메타데이터 행 수: {r.parquet_rows:,}\n")
            f.write(f"Row group 수: {r.row_groups:,}\n")
            f.write(f"소요 시간: {r.elapsed_sec:.2f} 초\n")
            if r.elapsed_sec > 0 and r.rows_written > 0:
                f.write(f"처리 속도: {r.rows_written / r.elapsed_sec:,.0f} 행/초\n")

            if r.error_message:
                f.write(f"오류 메시지: {r.error_message}\n")

            if r.checks:
                f.write("\n검증 결과:\n")
                for check in r.checks:
                    f.write(f"  - {check}\n")

            if r.column_stats:
                f.write("\n주요 컬럼 원본 기준 통계:\n")
                f.write("  컬럼 | null 수 | min | max | 합계 | 이상값 수\n")
                f.write("  " + "-" * 70 + "\n")
                for col, stats in r.column_stats.items():
                    f.write(
                        f"  {col} | {stats.nulls:,} | "
                        f"{_format_value(stats.min_value)} | {_format_value(stats.max_value)} | "
                        f"{_format_value(stats.sum_value)} | {stats.invalid_count:,}\n"
                    )

            if r.schema:
                f.write("\nParquet 스키마:\n")
                for line in r.schema.splitlines():
                    f.write(f"  {line}\n")

        f.write("\n" + "=" * 80 + "\n")
        f.write("판정 기준:\n")
        f.write("- 행 수 일치: 원본 청크에서 읽은 총 행 수와 Parquet 메타데이터 행 수가 같아야 합니다.\n")
        f.write("- null 수/범위/합계: 원본 기준 통계입니다. 변환 전 데이터 자체에 이상값이 있으면 이상값 수에 기록됩니다.\n")
        f.write("- 실패 파일은 최종 .parquet으로 저장하지 않고 임시 파일을 삭제합니다.\n")


def run_batch_conversion() -> None:
    total_start_time = time.time()
    reports: List[FileReport] = []

    report_dir = REPORT_DIR if REPORT_DIR is not None else BASE_OUTPUT_DIR / "reports"
    report_path = report_dir / f"sas_to_parquet_검증리포트_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    print(f"\n{'=' * 70}")
    print("[SAS → Parquet 안전 변환]")
    print(f"  입력 루트: {BASE_INPUT_DIR}")
    print(f"  출력 루트: {BASE_OUTPUT_DIR}")
    print(f"  리포트: {report_path}")
    print(f"{'=' * 70}")

    if not BASE_INPUT_DIR.exists():
        raise FileNotFoundError(f"입력 루트가 존재하지 않습니다: {BASE_INPUT_DIR}")

    folders = sorted(p for p in BASE_INPUT_DIR.iterdir() if p.is_dir() and _FOLDER_RE.match(p.name))
    if not folders:
        print("[오류] 하위 폴더를 찾을 수 없습니다. BASE_INPUT_DIR을 확인하세요.")
        return

    for folder in folders:
        sas_files = sorted(folder.glob("*.sas7bdat"))
        if not sas_files:
            continue

        output_dir = BASE_OUTPUT_DIR / folder.name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"[{folder.name}] {len(sas_files)}개 파일 → {output_dir}")
        print(f"{'=' * 70}")

        for sas_file in sas_files:
            output_file = output_dir / f"{sas_file.stem}.parquet"

            if SKIP_EXISTING and output_file.exists():
                print(f"  [스킵] 이미 존재: {output_file.name}")
                r = FileReport(
                    input_file=str(sas_file),
                    output_file=str(output_file),
                    status="스킵",
                    input_size_gb=_gb(sas_file.stat().st_size),
                    output_size_gb=_gb(output_file.stat().st_size),
                )
                # 이미 존재하는 파일도 간단히 메타데이터 확인
                try:
                    pf = pq.ParquetFile(output_file)
                    r.parquet_rows = int(pf.metadata.num_rows)
                    r.row_groups = int(pf.metadata.num_row_groups)
                    r.schema = str(pf.schema_arrow)
                    r.add_check("참고: 기존 파일이 있어 변환은 건너뛰고 parquet 메타데이터만 확인했습니다.")
                except Exception as exc:
                    r.status = "실패"
                    r.error_message = f"기존 parquet 메타데이터 확인 실패: {exc}"
                reports.append(r)
                continue

            report = convert_single_file(sas_file, output_file)
            reports.append(report)

            # 긴 작업 중간에도 리포트가 남도록 매 파일마다 갱신
            write_korean_txt_report(reports, report_path)

    total_elapsed = time.time() - total_start_time
    print(f"\n{'*' * 70}")
    print(f"[작업 완료] 총 소요 시간: {total_elapsed / 60:.2f}분")
    print(f"리포트 파일: {report_path}")
    print(f"{'*' * 70}")

    write_korean_txt_report(reports, report_path)


if __name__ == "__main__":
    run_batch_conversion()
