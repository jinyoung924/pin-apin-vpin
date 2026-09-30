# step1에서 1분봉 데이터를 concat할때 그걸 다 메모리에 올려서 메모리 터지는 문제 발생 -
"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 VPIN (Volume-Synchronized PIN) 계산
=============================================================================

■ 설계 원칙
  ┌─────────────────────────┬────────────────────────────────────────────┐
  │ VPIN Base 에서 유지      │ PIN 코드 에서 이식                         │
  ├─────────────────────────┼────────────────────────────────────────────┤
  │ VPIN 계산 로직           │ 다중 연도 폴더(KOR_YYYY) 탐색              │
  │ 파일 기반 워커            │ Step 1 결과 영구 저장 + 스킵 지원          │
  │ (결과를 디스크에 직접 저장)│ RUN_ID 세션 격리 + 체크포인트             │
  │ → RAM 누적 없음           │ 중단 재개 (RESUME_RUN_ID)                 │
  └─────────────────────────┴────────────────────────────────────────────┘

■ 전체 흐름

  [Step 1] 틱 데이터 → all_1m_bars.parquet
    ① 월별 parquet 를 순회하며 1분봉 집계 → 연도별 임시 parquet 로 저장
       (RAM = 한 연도치 데이터만 점유, 처리 후 즉시 해제)
    ② 연도별 파일 12개를 scan_parquet + sort + sink_parquet 으로 병합
       (group_by 없이 sort 만 사용 → Polars 스트리밍 호환)
    ③ all_1m_bars.parquet 로 저장, 이후 실행 시 스킵

  [Step 2] 1분봉 → 종목별 VPIN
    ① 배치 Lazy 스캔으로 종목별 sym_input/{sym}.parquet 분할
       (PARTITION_BATCH 개씩 scan → collect → 저장 → 해제)
    ② 워커: sym_input 파일 읽기 → VPIN 계산 → sym_result/{sym}.parquet 직접 저장
       (메인 프로세스로 DataFrame 반환 안 함 → RAM 누적 없음)
    ③ CHECKPOINT_N 종목마다 sym_result 파일들을 체크포인트로 통합 후 개별 삭제

■ Step 1 RAM 설계

  월별 처리 후 연도별로 묶어 저장하는 이유:
    sink_parquet 은 Polars 스트리밍 엔진을 사용하는데,
    group_by 는 스트리밍 모드에서 지원되지 않아 조용히 실패한다.
    대신 월별 → 연도별 concat(collect) → 연도 파일 저장 방식으로
    한 번에 RAM 에 올라오는 데이터를 "1년치"로 제한한다.
    마지막 병합은 sort 만 사용하므로 스트리밍 호환이다.

  피크 RAM 예상:
    2021 (최대 연도, ~14파일 × ~12M행 ≈ 168M행 × 30bytes ≈ 5GB)
    나머지 연도는 더 작음 → 연도 순차 처리로 중첩 없음

■ VPIN 계산 방식
  BVC: ProbBuy = CDF_t(ΔP / σ; df=0.25)
  버킷 크기: V = ADV(해당 연도) / buckets_per_day  ← 연도별 동적 산출
  VPIN = Σ|V_buy - V_sell| / (rolling_window × V)

  연도별 ADV 를 사용하는 이유:
    2010~2021 전체 평균 ADV 로 고정하면 2010(저거래) 에서는 버킷이
    거의 안 채워지고 2021(고거래) 에서는 과도하게 생성된다.
    연도별 ADV 를 쓰면 매년 "하루 평균 50개 버킷" 스케일이 유지된다.

  윈도우 갭 없음:
    sym_input 파일에 2010~2021 전체 데이터가 연속으로 들어 있어
    파일 경계 없이 버킷이 이어진다.
    초기 NaN(49개)은 종목 역사 전체에서 딱 한 번만 발생한다.

■ 설정값
  BASE_DIR               : 월별 .parquet 루트 경로
  YEAR_FOLDERS           : 처리 연도 폴더 (None → KOR_* 전체 자동 탐색)
  OUTPUT_DIR             : 출력 경로
  FORCE_REPROCESS_STEP1  : True → all_1m_bars.parquet 강제 재생성
  PARTITION_BATCH        : 종목 분할 배치 크기 (RAM 부족 시 줄임, 기본 200)
  CHECKPOINT_N           : 종목 N개마다 체크포인트 저장 (기본 100)
  IMAP_CHUNKSIZE         : 워커 IPC 청크 크기 (기본 10)
  ROLLING_WINDOW         : VPIN 롤링 버킷 수 (기본 50)
  BUCKETS_PER_DAY        : 버킷 크기 = ADV / 이 값 (기본 50)
  RESUME_RUN_ID          : 재개할 세션 RUN_ID (None → 새 세션)

=============================================================================
"""

import os
import glob
import tempfile
import shutil
import multiprocessing
import warnings
from datetime import datetime

import numpy as np
import polars as pl
from scipy.stats import t as t_dist
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 전역 설정
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = None   # ['KOR_2017']  # None → KOR_* 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

FORCE_REPROCESS_STEP1 = False  # True → all_1m_bars.parquet 강제 재생성

PARTITION_BATCH       = 200    # 종목 분할 배치 크기 (RAM 여유 시 500 으로 늘려도 됨)
CHECKPOINT_N          = 100    # 종목 N개마다 체크포인트 저장
IMAP_CHUNKSIZE        = 10     # 워커 IPC 청크 크기

ROLLING_WINDOW        = 50     # VPIN 롤링 버킷 수
BUCKETS_PER_DAY       = 50     # 버킷 크기 V = ADV / 이 값

# None → 현재 시각으로 새 RUN_ID 생성
# "20240115_0930" → 해당 세션 이어 쓰기 (완료 후 None 으로 복원)
RESUME_RUN_ID         = None


# =============================================================================
# 공통 유틸리티
# =============================================================================

def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다."""
    if year_folders is None:
        year_folders = sorted([
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("KOR_")
        ])

    parquet_files = []
    for yf in year_folders:
        folder_path = os.path.join(base_dir, yf)
        if not os.path.exists(folder_path):
            print(f"[Warning] 폴더 없음: {folder_path}")
            continue
        parquet_files.extend(sorted(glob.glob(os.path.join(folder_path, "*.parquet"))))

    print(f"\n[파일 탐색]")
    print(f"  기준 경로   : {base_dir}")
    print(f"  대상 연도   : {year_folders}")
    print(f"  parquet 파일: {len(parquet_files)}개")
    if parquet_files:
        print(f"  첫 번째     : {os.path.basename(parquet_files[0])}")
        print(f"  마지막      : {os.path.basename(parquet_files[-1])}")
    return parquet_files


def load_already_done_symbols(session_dir: str) -> set[str]:
    """세션 폴더의 체크포인트에서 완료된 종목 목록을 복원한다."""
    if not os.path.exists(session_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(session_dir) if f.startswith("vpin_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols: set[str] = set()
    for fname in checkpoint_files:
        try:
            df = pl.read_parquet(
                os.path.join(session_dir, fname), columns=["Symbol"]
            )
            done_symbols.update(df["Symbol"].unique().to_list())
        except Exception as e:
            print(f"[Warning] 체크포인트 읽기 실패: {fname} ({e})")

    print(f"[재개 모드] 체크포인트 {len(checkpoint_files)}개 → "
          f"완료 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


def flush_to_checkpoint(
    result_paths: list[str],
    checkpoint_idx: int,
    session_dir: str,
) -> None:
    """
    sym_result 파일 목록을 체크포인트 parquet 로 통합 저장하고
    개별 파일을 삭제한다.
    """
    if not result_paths:
        return

    dfs, ok_paths = [], []
    for path in result_paths:
        try:
            df = pl.read_parquet(path)
            if not df.is_empty():
                dfs.append(df)
                ok_paths.append(path)
        except Exception as e:
            print(f"\n[Warning] 결과 읽기 실패: {os.path.basename(path)} ({e})")

    if not dfs:
        return

    combined  = pl.concat(dfs, how="vertical")
    ckpt_path = os.path.join(
        session_dir, f"vpin_checkpoint_{checkpoint_idx:04d}.parquet"
    )
    combined.write_parquet(ckpt_path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] "
          f"{len(ok_paths)}종목 / {combined.height:,}버킷 "
          f"→ {os.path.basename(ckpt_path)}")

    for path in ok_paths:
        try:
            os.remove(path)
        except Exception:
            pass


# =============================================================================
# [Step 1] 틱 데이터 → 1분봉 집계
# =============================================================================

def _process_one_month(parquet_path: str) -> pl.DataFrame:
    """
    단일 월 parquet 를 읽어 1분봉으로 집계한다.

    입력: Symbol, Date, Time, Price, Volume
    출력: Symbol, Datetime, Price(종가), Volume(합계)
    """
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "Time", "Price", "Volume"])
        .filter(pl.col("Volume") > 0)
        .with_columns(
            pl.col("Date").dt.combine(pl.col("Time")).alias("Datetime")
        )
        .drop(["Date", "Time"])
        .sort(["Symbol", "Datetime"])
        .collect()
    )

    if df.is_empty():
        print("0 bars")
        return pl.DataFrame(schema={
            "Symbol": pl.String, "Datetime": pl.Datetime,
            "Price": pl.Float64, "Volume": pl.Float64,
        })

    bars = (
        df
        .group_by_dynamic("Datetime", every="1m", group_by="Symbol", closed="left")
        .agg([
            pl.col("Price").last().alias("Price"),
            pl.col("Volume").sum().alias("Volume"),
        ])
        .select(["Symbol", "Datetime", "Price", "Volume"])
        .sort(["Symbol", "Datetime"])
    )

    print(f"{bars.height:,} bars")
    return bars


def run_preprocessing(base_dir: str, year_folders: list | None, output_dir: str) -> str:
    """
    모든 월별 parquet 를 순회해 1분봉으로 집계하고
    all_1m_bars.parquet 를 저장한다.

    ■ RAM 전략 (2단계)

      1단계: 연도별 처리
        - 한 연도의 월별 파일들을 순서대로 읽어 1분봉 집계
        - 해당 연도 전체를 pl.concat + sort 후 연도 임시 파일로 저장
        - RAM = 한 연도 데이터만 점유 → 처리 후 즉시 해제
        - 피크: 가장 큰 연도(2021, ~14파일 × ~12M행 ≈ 168M행 ≈ 5GB)

      2단계: 연도 파일 병합 (sink_parquet 스트리밍)
        - 연도 파일 12개를 scan_parquet + sort + sink_parquet 으로 병합
        - sort 는 Polars 스트리밍 호환 (group_by 는 호환 안 됨 → 사용 안 함)
        - 월별 파일은 서로 다른 시간대를 커버하므로 중복 없음
          → group_by 없이 sort 만으로 충분

    반환: all_1m_bars.parquet 경로 (실패 시 빈 문자열)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_1m_bars.parquet")

    if not FORCE_REPROCESS_STEP1 and os.path.exists(output_path):
        print(f"\n{'='*65}")
        print(f"[Step 1 스킵] 기존 파일 존재: {output_path}")
        print(f"  재생성하려면 FORCE_REPROCESS_STEP1 = True 로 설정하세요")
        print(f"{'='*65}\n")
        return output_path

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("[Error] parquet 파일 없음")
        return ""

    print(f"\n{'='*65}")
    print("[Step 1] 전처리 시작: 틱 데이터 → 1분봉 집계")
    print(f"{'='*65}\n")

    # 파일을 연도 태그별로 그룹화
    def _year_tag(path: str) -> str:
        return next(
            (p for p in path.replace("\\", "/").split("/") if p.startswith("KOR_")),
            "KOR_UNKNOWN",
        )

    year_groups: dict[str, list[str]] = {}
    for path in parquet_files:
        year_groups.setdefault(_year_tag(path), []).append(path)

    # 임시 폴더: 연도별 중간 결과 저장용
    temp_dir        = tempfile.mkdtemp(prefix="vpin_step1_")
    year_temp_files = []   # 각 연도의 정렬된 1분봉 parquet 경로
    total_skipped   = 0

    try:
        # ── 1단계: 연도별 처리 ─────────────────────────────────────────────
        for year_tag in sorted(year_groups):
            paths = year_groups[year_tag]
            print(f"\n[{year_tag}] ({len(paths)}개 파일)")

            month_dfs = []
            for path in paths:
                df = _process_one_month(path)
                if df.is_empty():
                    print(f"    → [Warning] 데이터 없음, 건너뜀")
                    total_skipped += 1
                else:
                    month_dfs.append(df)

            if not month_dfs:
                print(f"  [{year_tag}] 유효 데이터 없음, 건너뜀")
                continue

            # 연도 내 월별 데이터를 concat + sort → 연도 임시 파일 저장
            print(f"  [{year_tag}] 정렬 및 저장 중...", end=" ", flush=True)
            year_df = (
                pl.concat(month_dfs, how="vertical")
                .sort(["Symbol", "Datetime"])
            )
            del month_dfs   # 월별 DataFrame 즉시 해제

            year_tmp = os.path.join(temp_dir, f"{year_tag}.parquet")
            year_df.write_parquet(year_tmp, compression="zstd")
            year_temp_files.append(year_tmp)

            print(f"{year_df.height:,}행 → {os.path.basename(year_tmp)}")
            del year_df   # 연도 DataFrame 즉시 해제

        if not year_temp_files:
            print("[Error] 유효한 데이터가 없습니다.")
            return ""

        # ── 2단계: 연도 파일 → 단일 parquet 병합 (PyArrow append) ──────────
        #
        # sink_parquet + sort 는 Polars 스트리밍 엔진 버그로 조용히 실패한다.
        # → PyArrow ParquetWriter 로 연도 파일을 한 장씩 append 해서 해결한다.
        #
        # 각 연도 파일은 이미 Symbol+Datetime 으로 정렬되어 있다.
        # 연도 자체가 시간 순서이므로 연도 파일을 순서대로 append 하면
        # Datetime 기준으로 전체가 정렬된 결과가 만들어진다.
        # (Symbol 기준 정렬이 필요하다면 Step 2 에서 sym_input 분할 시 sort 됨)
        import pyarrow.parquet as pq

        print(f"\n{'='*65}")
        print(f"[병합] {len(year_temp_files)}개 연도 파일 → {os.path.basename(output_path)}")
        print("  (PyArrow append 방식: RAM 에 한 연도씩만 올립니다)")

        writer     = None
        total_rows = 0
        try:
            for i, ytf in enumerate(year_temp_files, 1):
                year_name = os.path.basename(ytf)
                print(f"  append {i}/{len(year_temp_files)}: {year_name} ...",
                      end=" ", flush=True)

                # Polars 로 읽어 PyArrow Table 로 변환
                year_df    = pl.read_parquet(ytf)
                arrow_tbl  = year_df.to_arrow()
                total_rows += len(arrow_tbl)
                del year_df   # 즉시 해제

                if writer is None:
                    writer = pq.ParquetWriter(
                        output_path,
                        schema=arrow_tbl.schema,
                        compression="zstd",
                    )
                writer.write_table(arrow_tbl)
                del arrow_tbl
                print(f"{total_rows:,}행 누적")
        finally:
            if writer is not None:
                writer.close()

        if not os.path.exists(output_path):
            print("[Error] 병합 파일이 생성되지 않았습니다.")
            return ""

        # 통계 조회 (scan 으로만)
        stats = (
            pl.scan_parquet(output_path)
            .select([
                pl.len().alias("rows"),
                pl.col("Symbol").n_unique().alias("symbols"),
                pl.col("Datetime").min().alias("dt_min"),
                pl.col("Datetime").max().alias("dt_max"),
            ])
            .collect()
        )

        print(f"\n{'='*65}")
        print("[Step 1 완료]")
        print(f"{'='*65}")
        print(f"  저장 경로  : {output_path}")
        print(f"  전체 행 수 : {stats['rows'][0]:,}")
        print(f"  종목 수    : {stats['symbols'][0]:,}")
        print(f"  시간 범위  : {stats['dt_min'][0]} ~ {stats['dt_max'][0]}")
        print(f"  처리 파일  : {len(parquet_files) - total_skipped}개"
              f"  (건너뜀: {total_skipped}개)")
        print(f"{'='*65}\n")
        return output_path

    except Exception as e:
        import traceback
        print(f"\n[Error] Step 1 실패: {e}")
        print(traceback.format_exc())
        return ""

    finally:
        # 성공·실패 모두 임시 폴더 정리
        try:
            shutil.rmtree(temp_dir)
        except Exception as e:
            print(f"[Warning] 임시 폴더 삭제 실패: {temp_dir} ({e})")


# =============================================================================
# [Step 2] VPIN 계산 핵심 함수
# =============================================================================

def calculate_vpin_for_single_symbol(
    bar_df: pl.DataFrame,
    rolling_window: int = 50,
    buckets_per_day: int = 50,
) -> pl.DataFrame:
    """
    단일 종목의 1분봉 데이터를 받아 BVC 기반 VPIN 을 계산한다.

    ■ 연도별 동적 버킷 크기
      각 1분봉이 속한 연도의 ADV 를 기준으로 버킷 크기를 산출한다.
      → 매년 "하루 평균 buckets_per_day 개 버킷" 스케일 유지
      → 연도 간 VPIN 직접 비교 가능

    ■ 버킷 연속성
      연도가 바뀌어도 current_filled 를 초기화하지 않는다.
      → 연도 경계에서 윈도우 갭 없음
      → 초기 NaN 은 종목 역사 전체에서 딱 한 번
    """
    schema_def = {
        "Datetime": pl.Datetime, "Symbol": pl.String,
        "BucketNo": pl.Int64,    "VPIN":   pl.Float64,
    }
    empty_res = pl.DataFrame(schema=schema_def)

    if bar_df.height < 2:
        return empty_res

    times_dt   = bar_df["Datetime"].to_list()
    prices     = bar_df["Price"].to_numpy()
    volumes    = bar_df["Volume"].to_numpy()
    symbol_str = str(bar_df["Symbol"][0])

    # ── 연도별 ADV → 버킷 크기 매핑 ─────────────────────────────────────────
    year_stats = (
        bar_df
        .with_columns(pl.col("Datetime").dt.year().alias("Year"))
        .group_by("Year")
        .agg([
            pl.col("Volume").sum().alias("total_vol"),
            pl.col("Datetime").dt.date().n_unique().alias("n_days"),
        ])
    )
    year_bucket_map: dict[int, float] = {
        row["Year"]: (row["total_vol"] / row["n_days"] / buckets_per_day
                      if row["n_days"] > 0 else 1e-5)
        for row in year_stats.iter_rows(named=True)
    }
    if not year_bucket_map:
        return empty_res

    # ── BVC 매수/매도 거래량 추정 ─────────────────────────────────────────────
    delta_p  = np.diff(prices, prepend=prices[0])
    sigma_p  = np.std(delta_p) + 1e-9
    prob_buy = t_dist.cdf(delta_p / sigma_p, df=0.25)

    # ── Volume Bucketing (연도별 버킷 크기 동적 적용) ─────────────────────────
    bucket_records: list[tuple] = []
    current_buy    = 0.0
    current_sell   = 0.0
    current_filled = 0.0
    bucket_no      = 0
    last_year      = None
    bucket_size_v  = 0.0

    for i in range(len(volumes)):
        rem_vol   = float(volumes[i])
        p_b       = float(prob_buy[i])
        curr_time = times_dt[i]
        curr_year = curr_time.year

        if rem_vol <= 0:
            continue

        # 연도 변경 시 버킷 크기 갱신 (current_filled 유지 → 연속성 보장)
        if curr_year != last_year:
            bucket_size_v = year_bucket_map.get(curr_year, 1e-5)
            last_year     = curr_year

        if bucket_size_v <= 1e-5:
            continue

        while rem_vol > 0:
            needed    = bucket_size_v - current_filled
            alloc_vol = min(needed, rem_vol)

            current_buy    += alloc_vol * p_b
            current_sell   += alloc_vol * (1.0 - p_b)
            current_filled += alloc_vol
            rem_vol        -= alloc_vol

            if current_filled >= bucket_size_v - 1e-9:
                bucket_records.append((curr_time, bucket_no,
                                       abs(current_buy - current_sell)))
                bucket_no      += 1
                current_buy     = 0.0
                current_sell    = 0.0
                current_filled  = 0.0

    if not bucket_records:
        return empty_res

    # ── VPIN 롤링 계산 ────────────────────────────────────────────────────────
    res_times  = [r[0] for r in bucket_records]
    res_nos    = [r[1] for r in bucket_records]
    res_ois    = np.array([r[2] for r in bucket_records])
    bucket_bsv = np.array([year_bucket_map.get(t.year, 1e-5) for t in res_times])

    final_vpin = np.full(len(res_ois), np.nan)
    if len(res_ois) >= rolling_window:
        oi_sums   = np.convolve(res_ois, np.ones(rolling_window), mode="valid")
        last_bsv  = bucket_bsv[rolling_window - 1:]
        final_vpin[rolling_window - 1:] = oi_sums / (rolling_window * last_bsv)

    return pl.DataFrame({
        "Datetime": res_times,
        "Symbol":   [symbol_str] * len(res_times),
        "BucketNo": res_nos,
        "VPIN":     final_vpin,
    }).with_columns([
        pl.col("Datetime").cast(pl.Datetime),
        pl.col("Symbol").cast(pl.String),
        pl.col("BucketNo").cast(pl.Int64),
        pl.col("VPIN").cast(pl.Float64),
    ])


# =============================================================================
# [Step 2] 멀티프로세싱 워커
# =============================================================================

_ROLLING_WINDOW:  int = 50
_BUCKETS_PER_DAY: int = 50


def init_worker(rolling_window: int, buckets_per_day: int) -> None:
    """워커 프로세스 시작 시 VPIN 파라미터를 전역 변수에 저장한다."""
    global _ROLLING_WINDOW, _BUCKETS_PER_DAY
    _ROLLING_WINDOW  = rolling_window
    _BUCKETS_PER_DAY = buckets_per_day


def process_symbol_worker(args: tuple) -> str | None:
    """
    sym_input/{sym}.parquet 를 읽어 VPIN 을 계산하고
    sym_result/{sym}.parquet 에 직접 저장한 뒤 경로(문자열)만 반환한다.

    DataFrame 을 메인 프로세스로 반환하지 않으므로
    IPC 비용 = 경로 문자열(~30 bytes) 뿐 → RAM 누적 없음.
    """
    symbol, input_path, output_path = args
    try:
        bar_df = pl.read_parquet(input_path)
        if bar_df.is_empty():
            return None

        vpin_df = calculate_vpin_for_single_symbol(
            bar_df, _ROLLING_WINDOW, _BUCKETS_PER_DAY
        )
        if vpin_df.is_empty():
            return None

        vpin_df.write_parquet(output_path, compression="zstd")
        return output_path

    except Exception as e:
        import traceback
        print(f"\n[Error] {symbol}: {e}\n{traceback.format_exc()}")
        return None


# =============================================================================
# [Step 2] VPIN 계산 통합 실행 함수
# =============================================================================

def run_vpin_calculation(
    bars_path:    str,
    output_dir:   str,
    run_id:       str,
    year_filter:  list[int] | None,
    checkpoint_n: int,
) -> pl.DataFrame:
    """
    all_1m_bars.parquet 를 읽어 종목별 VPIN 을 병렬로 계산하고 결과를 반환한다.

    처리 흐름:
      1. scan_parquet 로 메타 확인 (전체 로드 없음)
      2. 세션 폴더 구성 + 체크포인트에서 완료 종목 로드
      3. PARTITION_BATCH 개씩 배치 Lazy 스캔으로 sym_input 파일 분할 저장
      4. multiprocessing.Pool 로 병렬 VPIN 계산
      5. CHECKPOINT_N 마다 sym_result 통합 → 체크포인트 저장
      6. 체크포인트 전체 병합 → 반환
    """
    intermediate_dir = os.path.join(output_dir, "intermediate")
    session_dir      = os.path.join(intermediate_dir, f"session_{run_id}")
    sym_input_dir    = os.path.join(session_dir, "sym_input")
    sym_result_dir   = os.path.join(session_dir, "sym_result")

    for d in [output_dir, session_dir, sym_input_dir, sym_result_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"\n{'='*65}")
    print("[Step 2] VPIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일   : {bars_path}")
    print(f"  RUN_ID      : {run_id}")
    print(f"  세션 폴더   : {session_dir}")

    # ── 1. 파일 메타 확인 (scan 만 사용, 전체 로드 없음) ─────────────────────
    print(f"\n[파일 메타 확인]")
    lazy_base = pl.scan_parquet(bars_path)
    if year_filter:
        lazy_base = lazy_base.filter(pl.col("Datetime").dt.year().is_in(year_filter))

    meta = lazy_base.select([
        pl.len().alias("rows"),
        pl.col("Symbol").n_unique().alias("symbols"),
        pl.col("Datetime").min().alias("dt_min"),
        pl.col("Datetime").max().alias("dt_max"),
    ]).collect()

    print(f"  전체 행 수  : {meta['rows'][0]:,}")
    print(f"  종목 수     : {meta['symbols'][0]:,}")
    print(f"  시간 범위   : {meta['dt_min'][0]} ~ {meta['dt_max'][0]}")
    if year_filter:
        print(f"  연도 필터   : {year_filter}")

    num_cores = multiprocessing.cpu_count()
    print(f"\n  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  롤링 버킷   : {ROLLING_WINDOW}개")
    print(f"  버킷/일     : {BUCKETS_PER_DAY}개")
    print(f"  분할 배치   : {PARTITION_BATCH}개/회")

    # ── 2. 완료 종목 확인 (체크포인트 기반) ──────────────────────────────────
    done_symbols   = load_already_done_symbols(session_dir)
    all_symbols    = (
        lazy_base.select("Symbol").unique().collect()["Symbol"].sort().to_list()
    )
    remaining_list = [s for s in all_symbols if s not in done_symbols]

    print(f"\n  전체 종목   : {len(all_symbols):,}개")
    print(f"  완료 종목   : {len(done_symbols):,}개")
    print(f"  처리 대상   : {len(remaining_list):,}개\n")

    if not remaining_list:
        print("  모든 종목이 완료됐습니다. 체크포인트 병합으로 진행합니다.")

    # ── 3. sym_input 파일 분할 저장 (배치 Lazy 스캔) ─────────────────────────
    # 이미 파일이 있는 종목은 재사용, 없는 종목만 배치 스캔으로 생성
    need_write     = [s for s in remaining_list
                      if not os.path.exists(
                          os.path.join(sym_input_dir, f"sym_{s}.parquet"))]
    already_split  = [s for s in remaining_list if s not in need_write]

    tasks: list[tuple] = []

    # 이미 분할된 종목 → task 에 바로 추가
    for sym in already_split:
        tasks.append((
            sym,
            os.path.join(sym_input_dir,  f"sym_{sym}.parquet"),
            os.path.join(sym_result_dir, f"sym_{sym}.parquet"),
        ))

    # 신규 분할 종목 → PARTITION_BATCH 개씩 배치 스캔
    if need_write:
        total_batches = (len(need_write) + PARTITION_BATCH - 1) // PARTITION_BATCH
        print(f"[종목 분할] 신규 {len(need_write)}개를 "
              f"{PARTITION_BATCH}개 배치 × {total_batches}회 스캔")

        written = 0
        for b_start in range(0, len(need_write), PARTITION_BATCH):
            batch_syms = need_write[b_start : b_start + PARTITION_BATCH]
            b_num      = b_start // PARTITION_BATCH + 1
            print(f"  배치 {b_num}/{total_batches} "
                  f"({len(batch_syms)}종목) ...", end=" ", flush=True)

            batch_df = (
                lazy_base
                .filter(pl.col("Symbol").is_in(batch_syms))
                .collect()
            )

            for part in batch_df.partition_by("Symbol", maintain_order=False):
                sym         = part["Symbol"][0]
                input_path  = os.path.join(sym_input_dir,  f"sym_{sym}.parquet")
                output_path = os.path.join(sym_result_dir, f"sym_{sym}.parquet")
                part.sort("Datetime").write_parquet(input_path, compression="zstd")
                written += 1
                tasks.append((sym, input_path, output_path))

            del batch_df
            print(f"완료 (누적 {written}개 저장)")

    del lazy_base
    print(f"\n  분할 완료: 신규 {len(need_write)}개 / 재사용 {len(already_split)}개\n")

    # ── 4. 병렬 처리 ─────────────────────────────────────────────────────────
    print(f"{'='*65}")
    print("VPIN 계산 시작...")
    print(f"{'='*65}\n")

    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("vpin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)
    result_paths: list[str] = []
    processed_count = 0

    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(ROLLING_WINDOW, BUCKETS_PER_DAY),
    ) as pool:
        for path in tqdm(
            pool.imap_unordered(
                process_symbol_worker, tasks, chunksize=IMAP_CHUNKSIZE
            ),
            total=len(tasks),
            desc="  Estimating",
        ):
            if path is not None:
                result_paths.append(path)
            processed_count += 1

            if processed_count % checkpoint_n == 0:
                flush_to_checkpoint(result_paths, next_checkpoint_idx, session_dir)
                next_checkpoint_idx += 1
                result_paths = []

    if result_paths:
        flush_to_checkpoint(result_paths, next_checkpoint_idx, session_dir)

    # ── 5. 체크포인트 병합 ───────────────────────────────────────────────────
    print(f"\n{'='*65}")
    all_ckpts = sorted(
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("vpin_checkpoint_")
    )
    print(f"체크포인트 {len(all_ckpts)}개 병합 중... (세션: {run_id})")

    if not all_ckpts:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    return (
        pl.concat([pl.read_parquet(p) for p in all_ckpts], how="vertical")
        .sort(["Symbol", "Datetime"])
    )


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()

    run_id = RESUME_RUN_ID if RESUME_RUN_ID else datetime.now().strftime("%Y%m%d_%H%M")
    start  = datetime.now()

    print(f"\n{'[재개 모드]' if RESUME_RUN_ID else '[새 실행]  '} RUN_ID = {run_id}")

    # ── Step 1 ───────────────────────────────────────────────────────────────
    bars_path = run_preprocessing(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR,
    )

    if not bars_path or not os.path.exists(bars_path):
        print("[Error] 전처리 결과 파일이 없어 종료합니다.")
        exit(1)

    year_filter = (
        [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
        if YEAR_FOLDERS else None
    )

    # ── Step 2 ───────────────────────────────────────────────────────────────
    result = run_vpin_calculation(
        bars_path=bars_path,
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    # ── 결과 저장 ─────────────────────────────────────────────────────────────
    if not result.is_empty():
        year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename = f"vpin_rolling_{year_tag}_{run_id}"

        parquet_path = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result.write_parquet(parquet_path, compression="zstd")
        print(f"\n[저장 완료] {parquet_path}")

        csv_path = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result.head(1000).write_csv(csv_path)
        print(f"[샘플 저장] {csv_path}")

        print("\n[미리보기]")
        print(result.head(20))

        print("\n[통계]")
        vpin_ok = result.filter(pl.col("VPIN").is_not_null()).height
        print(f"  전체 버킷     : {result.height:,}")
        print(f"  처리 종목     : {result['Symbol'].n_unique():,}")
        print(f"  VPIN 유효     : {vpin_ok:,}")
        print(f"  VPIN null     : {result.height - vpin_ok:,}"
              f"  (초기 윈도우 미충족)")
        print(f"  VPIN 범위     : "
              f"{result['VPIN'].min():.6f} ~ {result['VPIN'].max():.6f}")

        sym_stats = (
            result
            .group_by("Symbol")
            .agg([
                pl.len().alias("total_buckets"),
                pl.col("VPIN").is_not_null().sum().alias("vpin_buckets"),
                pl.col("VPIN").mean().alias("mean_vpin"),
            ])
            .with_columns(
                (pl.col("vpin_buckets") / pl.col("total_buckets") * 100)
                .alias("coverage_pct")
            )
            .sort("total_buckets", descending=True)
        )
        print("\n[종목별 통계 샘플]")
        print(sym_stats.head(10))
    else:
        print("\n[Warning] 결과 DataFrame 이 비어 있습니다.")

    print(f"\n총 소요 시간: {datetime.now() - start}")
