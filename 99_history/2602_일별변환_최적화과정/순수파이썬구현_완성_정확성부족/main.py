"""
=============================================================================
메인 컨트롤러 — 4개 모듈(SAS변환, PIN, APIN, VPIN)을 통합 제어
=============================================================================

사용법:
  python main.py                   # 전체 파이프라인 실행 (SAS→Parquet, PIN, APIN, VPIN)
  python main.py --modules pin     # PIN만 실행
  python main.py --modules pin apin  # PIN + APIN만 실행
  python main.py --modules sas     # SAS→Parquet 변환만 실행
  python main.py --modules vpin    # VPIN만 실행

config.py에서 모든 설정을 관리합니다.
"""

import os
import sys
import argparse
import multiprocessing
from datetime import datetime

import polars as pl

import config
import sas_to_parquet
import pin
import apin
import vpin
import common


def resolve_run_id() -> str:
    """RUN_ID를 결정한다. RESUME_RUN_ID가 있으면 사용, 없으면 현재 시각."""
    if config.RESUME_RUN_ID:
        print(f"\n[재개 모드] RUN_ID = {config.RESUME_RUN_ID}")
        return config.RESUME_RUN_ID
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"\n[새 실행]   RUN_ID = {run_id}")
    return run_id


def resolve_year_filter() -> list[int] | None:
    """YEAR_FOLDERS 설정에서 연도 필터를 추출한다."""
    if config.YEAR_FOLDERS:
        return [int(yf.replace("KOR_", "")) for yf in config.YEAR_FOLDERS]
    return None


def print_pin_apin_summary(result: pl.DataFrame, model_name: str,
                           metric_col: str) -> None:
    """PIN/APIN 결과 요약을 출력한다."""
    if result.is_empty():
        print(f"\n[Warning] {model_name} 결과 DataFrame이 비어 있습니다.")
        return

    print(f"\n[미리보기]")
    print(result.head(20))

    print(f"\n[통계]")
    print(f"  전체 레코드             : {result.height:,}")
    ok = result.filter(pl.col(metric_col).is_not_null()).height
    print(f"  {metric_col} 추정 성공     : {ok:,}")
    print(f"  {metric_col} 추정 실패/없음: {result.height - ok:,}")

    sym_stats = (
        result
        .group_by("Symbol")
        .agg([
            pl.len().alias("total_days"),
            pl.col(metric_col).is_not_null().sum().alias("ok_days"),
        ])
        .with_columns(
            (pl.col("ok_days") / pl.col("total_days") * 100).alias("coverage_pct")
        )
    )
    print(f"\n[종목별 커버리지 샘플]")
    print(sym_stats.head(10))


def save_pin_apin_result(result: pl.DataFrame, model_name: str,
                         year_filter: list[int] | None, run_id: str,
                         output_dir: str) -> None:
    """PIN/APIN 결과를 parquet + 샘플 csv로 저장한다."""
    if result.is_empty():
        return

    year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
    output_filename = f"{model_name}_rolling_{year_tag}_{run_id}"

    model_dir = os.path.join(output_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)

    parquet_path = os.path.join(model_dir, f"{output_filename}.parquet")
    result.write_parquet(parquet_path, compression="zstd")
    print(f"\n[Saved] {parquet_path}")

    csv_path = os.path.join(model_dir, f"{output_filename}_sample.csv")
    result.head(1000).write_csv(csv_path)
    print(f"[Sample] {csv_path}")


def print_vpin_summary(results_dir: str) -> None:
    """VPIN 결과 통계를 출력한다."""
    if not results_dir or not os.path.exists(results_dir):
        print("\n[Warning] 결과 폴더가 없습니다.")
        return

    result_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("sym_") and f.endswith(".parquet")
    )
    print(f"\n[완료 통계]")
    print(f"  결과 폴더   : {results_dir}")
    print(f"  종목 파일   : {len(result_files)}개")

    if result_files:
        all_paths = [os.path.join(results_dir, f) for f in result_files]
        stats = (
            pl.scan_parquet(all_paths)
            .select([
                pl.len().alias("total_buckets"),
                pl.col("VPIN").is_not_null().sum().alias("vpin_ok"),
                pl.col("VPIN").min().alias("vpin_min"),
                pl.col("VPIN").max().alias("vpin_max"),
                pl.col("Datetime").min().alias("dt_min"),
                pl.col("Datetime").max().alias("dt_max"),
            ])
            .collect()
        )
        print(f"  전체 버킷   : {stats['total_buckets'][0]:,}")
        print(f"  VPIN 유효   : {stats['vpin_ok'][0]:,}")
        print(f"  VPIN NaN    : "
              f"{stats['total_buckets'][0] - stats['vpin_ok'][0]:,}")
        print(f"  VPIN 범위   : "
              f"{stats['vpin_min'][0]:.6f} ~ {stats['vpin_max'][0]:.6f}")
        print(f"  시간 범위   : {stats['dt_min'][0]} ~ {stats['dt_max'][0]}")

        sample_df = pl.read_parquet(all_paths[0])
        print(f"\n[샘플 미리보기: {result_files[0]}]")
        print(sample_df.head(10))


# =============================================================================
# 개별 모듈 실행 함수
# =============================================================================

def run_sas_conversion() -> None:
    """SAS → Parquet 변환을 실행한다."""
    print(f"\n{'#'*65}")
    print("# [모듈 1] SAS → Parquet 변환")
    print(f"{'#'*65}")
    sas_to_parquet.run()


def run_pin_pipeline(run_id: str, year_filter: list[int] | None) -> None:
    """PIN 파이프라인을 실행한다. (Step 1 전처리 + Step 2 추정)"""
    print(f"\n{'#'*65}")
    print("# [모듈 2] PIN 계산 (EKOP 1996)")
    print(f"{'#'*65}")

    # Step 1: 일별 B/S 집계
    daily_bs_path = common.run_bs_preprocessing(
        base_dir=config.BASE_DIR,
        year_folders=config.YEAR_FOLDERS,
        output_dir=config.OUTPUT_DIR,
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] 전처리 결과 파일이 없어 PIN 계산을 종료합니다.")
        return

    # Step 2: PIN 추정
    result = pin.run(
        daily_bs_path=daily_bs_path,
        output_dir=config.OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=config.CHECKPOINT_N,
    )

    save_pin_apin_result(result, "pin", year_filter, run_id, config.OUTPUT_DIR)
    print_pin_apin_summary(result, "PIN", "PIN")


def run_apin_pipeline(run_id: str, year_filter: list[int] | None) -> None:
    """APIN 파이프라인을 실행한다. (Step 1 전처리 + Step 2 추정)"""
    print(f"\n{'#'*65}")
    print("# [모듈 3] APIN 계산 (Duarte & Young 2009)")
    print(f"{'#'*65}")

    # Step 1: 일별 B/S 집계 (PIN과 공유)
    daily_bs_path = common.run_bs_preprocessing(
        base_dir=config.BASE_DIR,
        year_folders=config.YEAR_FOLDERS,
        output_dir=config.OUTPUT_DIR,
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] 전처리 결과 파일이 없어 APIN 계산을 종료합니다.")
        return

    # Step 2: APIN 추정
    result = apin.run(
        daily_bs_path=daily_bs_path,
        output_dir=config.OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=config.CHECKPOINT_N,
    )

    save_pin_apin_result(result, "apin", year_filter, run_id, config.OUTPUT_DIR)
    print_pin_apin_summary(result, "APIN", "APIN")


def run_vpin_pipeline(run_id: str, year_filter: list[int] | None) -> None:
    """VPIN 파이프라인을 실행한다. (Step 1 1분봉 + Step 2 VPIN)"""
    print(f"\n{'#'*65}")
    print("# [모듈 4] VPIN 계산")
    print(f"{'#'*65}")

    # Step 1: 1분봉 집계
    bars_dir = vpin.run_preprocessing(
        base_dir=config.BASE_DIR,
        year_folders=config.YEAR_FOLDERS,
        output_dir=config.OUTPUT_DIR,
    )

    if not bars_dir or not os.path.exists(bars_dir):
        print("[Error] 전처리 결과 폴더가 없어 VPIN 계산을 종료합니다.")
        return

    # Step 2: VPIN 계산
    results_dir = vpin.run(
        bars_dir=bars_dir,
        output_dir=config.OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=config.CHECKPOINT_N,
    )

    print_vpin_summary(results_dir)


# =============================================================================
# 메인 실행부
# =============================================================================

AVAILABLE_MODULES = {
    "sas":  "SAS → Parquet 변환",
    "pin":  "PIN 계산 (EKOP 1996)",
    "apin": "APIN 계산 (Duarte & Young 2009)",
    "vpin": "VPIN 계산",
}


def main():
    parser = argparse.ArgumentParser(
        description="PIN/APIN/VPIN 통합 파이프라인 컨트롤러"
    )
    parser.add_argument(
        "--modules", "-m",
        nargs="+",
        choices=list(AVAILABLE_MODULES.keys()),
        default=list(AVAILABLE_MODULES.keys()),
        help="실행할 모듈 (기본: 전체). 예) --modules pin apin",
    )
    args = parser.parse_args()

    multiprocessing.freeze_support()

    start = datetime.now()
    run_id = resolve_run_id()
    year_filter = resolve_year_filter()

    print(f"\n{'='*65}")
    print(f"실행 모듈: {', '.join(args.modules)}")
    print(f"{'='*65}")

    # 선택된 모듈을 순서대로 실행
    if "sas" in args.modules:
        run_sas_conversion()

    if "pin" in args.modules:
        run_pin_pipeline(run_id, year_filter)

    if "apin" in args.modules:
        run_apin_pipeline(run_id, year_filter)

    if "vpin" in args.modules:
        run_vpin_pipeline(run_id, year_filter)

    elapsed = datetime.now() - start
    print(f"\n{'='*65}")
    print(f"[전체 완료] 총 소요 시간: {elapsed}")
    print(f"{'='*65}")


if __name__ == "__main__":
    main()
