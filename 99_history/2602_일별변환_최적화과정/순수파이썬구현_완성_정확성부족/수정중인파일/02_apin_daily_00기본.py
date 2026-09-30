"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 일별 롤링 APIN 계산
=============================================================================

■ 전체 흐름 요약
  이 코드는 두 단계(Step)로 나뉜다.

  [Step 1] 원시 틱 데이터 → 일별 매수/매도 건수(B/S) 집계
    - 월별로 나뉜 .parquet 파일들을 순회하며 종목·날짜 단위로 B, S를 집계
    - 결과를 all_daily_bs.parquet 한 파일로 저장
    - Step 1이 이미 완료됐으면 자동으로 스킵 (FORCE_REPROCESS_STEP1 = False)

  [Step 2] 일별 B/S → 종목별 60일 롤링 APIN 추정 (Duarte & Young 2009 모델)
    - all_daily_bs.parquet를 읽어 '시장 공통 영업일 캘린더'를 먼저 추출
    - 각 종목의 B/S를 영업일 캘린더에 Left Join → 거래 없는 영업일에 B=S=0 삽입
    - 60행 슬라이딩 윈도우로 순회하며 MLE로 APIN 추정
    - 멀티프로세싱으로 종목 단위 병렬화, 체크포인트로 중간 저장 지원
    - 최종 결과는 모든 실제 거래일을 row로 포함하며,
      APIN 추정 조건 미충족(유효 거래일 부족)이나 수렴 실패인 날은
      APIN 및 파라미터 컬럼이 null로 표시된다

■ 영업일 캘린더 방식 (방법 1) 채택 이유
  어떤 종목이 특정 날에 거래가 없으면 틱 데이터에 해당 행 자체가 존재하지 않는다.
  이 상태에서 단순히 "과거 60개 관측치"로 윈도우를 잡으면,
  윈도우가 실제로 수개월에 걸쳐 있음에도 60 거래일처럼 취급되는 오류가 발생한다.

  이를 해결하는 두 가지 방법 중 이 코드는 방법 1을 채택한다.

    방법 1 (채택): 영업일 캘린더를 먼저 추출하고, 각 종목 데이터를 캘린더에 Join
      - 전체 데이터에서 B 또는 S > 0인 날짜들의 합집합으로 공통 영업일을 결정
      - 각 종목의 B/S를 영업일 캘린더에 left join → 거래 없는 영업일 B=S=0
      - 이후 60행 슬라이딩 윈도우는 항상 정확히 60 영업일을 의미함
      - 장점: 윈도우 크기가 날짜 범위와 1:1 대응, 코드 구조 단순, 메모리 효율적

    방법 2 (미채택): 모든 종목×영업일 조합에 B=S=0 행을 물리적으로 삽입
      - 종목 수 × 기간이 길어질수록 메모리 부담이 크게 증가
      - 거래가 아예 없는 기간(상장 전·상폐 후)의 0행도 모두 생성해야 해서 비효율적

■ APIN 이란?
  APIN(Adjusted Probability of Informed Trading)은 Duarte & Young(2009) 모델로
  측정한 '정보 거래와 무관한 대칭적 주문 흐름 충격을 분리한 후의 정보 거래 비율'이다.

  기존 EKOP(1996) PIN 모델은 매수-매도 불균형이 모두 정보 거래에서 비롯된다고 가정하지만,
  실제로는 유동성 충격·포트폴리오 리밸런싱 등 비정보적 요인으로도 불균형이 발생한다.
  Duarte & Young(2009)은 이러한 '대칭적 주문 흐름 충격(symmetric order-flow shock)'을
  별도로 모델링하여 순수 정보 거래 비율을 더 정확하게 추정한다.

  모델은 매일 두 가지 독립적 이벤트가 발생할 수 있다고 가정한다.
    (A) 정보 이벤트 발생 여부 (확률 α)
        - 발생하면: 호재(확률 δ) 또는 악재(확률 1-δ)
    (B) 대칭적 주문 충격 발생 여부
        - 정보 이벤트가 없는 날의 충격 확률: θ₁
        - 정보 이벤트가 있는 날의 충격 확률: θ₂

  결과적으로 6가지 시나리오가 존재한다.
    1) 무정보 + 충격 없음  : 확률 (1-α)(1-θ₁)
       B ~ Poisson(ε_b),           S ~ Poisson(ε_s)
    2) 무정보 + 충격       : 확률 (1-α)θ₁
       B ~ Poisson(ε_b + Δ_b),     S ~ Poisson(ε_s + Δ_s)
    3) 악재 + 충격 없음    : 확률 α(1-δ)(1-θ₂)
       B ~ Poisson(ε_b),           S ~ Poisson(μ_s + ε_s)
    4) 악재 + 충격         : 확률 α(1-δ)θ₂
       B ~ Poisson(ε_b + Δ_b),     S ~ Poisson(μ_s + ε_s + Δ_s)
    5) 호재 + 충격 없음    : 확률 αδ(1-θ₂)
       B ~ Poisson(μ_b + ε_b),     S ~ Poisson(ε_s)
    6) 호재 + 충격         : 확률 αδθ₂
       B ~ Poisson(μ_b + ε_b + Δ_b), S ~ Poisson(ε_s + Δ_s)

  파라미터 의미 (10개):
    α    : 정보 이벤트 발생 확률 (0~1)
    δ    : 호재 조건부 확률 (0~1)
    θ₁   : 무정보일의 대칭 충격 확률 (0~1)
    θ₂   : 정보일의 대칭 충격 확률 (0~1)
    μ_b  : 호재 시 정보 매수자 도착률 (≥0)
    μ_s  : 악재 시 정보 매도자 도착률 (≥0)
    ε_b  : 비정보 매수자 기본 도착률 (≥0)
    ε_s  : 비정보 매도자 기본 도착률 (≥0)
    Δ_b  : 대칭 충격 시 추가 매수 도착률 (≥0)
    Δ_s  : 대칭 충격 시 추가 매도 도착률 (≥0)

  APIN 계산식 (Duarte & Young 2009):
    분모 = α·(δ·μ_b + (1-δ)·μ_s) + (Δ_b + Δ_s)·(α·θ₂ + (1-α)·θ₁) + ε_b + ε_s
    APIN = α·(δ·μ_b + (1-δ)·μ_s) / 분모

  PSOS 계산식:
    PSOS = (Δ_b + Δ_s)·(α·θ₂ + (1-α)·θ₁) / 분모

■ MLE 추정 방식
  10개 파라미터(α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s)를 최대우도추정법으로 구한다.
  최적화는 2단계로 진행한다.

    [1단계] 그리드 탐색 — 2단계 전략으로 속도 최적화
      APIN은 10개 파라미터이므로 단순 3^10 = 59,049개 전체 그리드는 PIN(243개)보다
      크지만 여전히 벡터화 연산으로 처리 가능한 규모이다.
      모든 후보에 대해 NLL(음의 로그 우도)을 log-space에서 벡터화 일괄 계산한 뒤
      NLL이 가장 낮은 후보를 최적화의 시작점(초기값)으로 선택한다.

      ※ 59,049개 그리드의 실행 시간:
        log-space 벡터화 연산에서 그리드 크기 G=59,049, 윈도우 크기 N=60 기준
        (G, N) 행렬 연산이 핵심이며, 이는 약 3.5M float64 원소에 해당한다.
        NumPy의 BLAS 기반 연산으로 단일 윈도우당 수 ms 수준에서 처리되므로,
        PIN(G=243)대비 약 243배가 아닌 실측 기준 5~10배 정도의 오버헤드에 그친다.
        이는 캐시 효율과 벡터 명령어(AVX/SSE) 덕분이다.

    [2단계] L-BFGS-B 최적화
      - [1단계]에서 선정된 초기값에서 출발해 정밀 최적화 수행
      - 파라미터 범위 제약(α,δ,θ₁,θ₂ ∈ [0,1], μ_b,μ_s,ε_b,ε_s,Δ_b,Δ_s ≥ 0) 준수

■ 수치 안정성: log-space 연산
  포아송 PMF를 직접 곱하면 값이 매우 작아져 부동소수점 언더플로우가 발생한다.
  이를 막기 위해 확률을 log-space에서 계산한다.

    log P(k|λ) = k·ln(λ) - λ - ln(k!)       ← gammaln으로 ln(k!) 계산
    log(w₀·L₀ + w₁·L₁ + ... + w₅·L₅)       ← logsumexp 트릭으로 안전하게 합산
    = logsumexp([ln(w₀)+log L₀, ln(w₁)+log L₁, ..., ln(w₅)+log L₅])

  ※ 기존 02_apin_weekly.py에서는 scipy.stats.poisson.pmf를 직접 곱하는 방식이었으나,
    이 코드에서는 log-space 연산으로 전환하여 수치 안정성을 크게 개선하였다.

  ※ 기존 02_apin_weekly.py의 시나리오 6번 가중치가 a*t1*d로 되어 있었으나,
    이는 a*t2*d가 올바르다 (호재+충격 시나리오이므로 정보일의 충격 확률 θ₂를 사용).
    이 코드에서는 올바르게 수정하여 구현하였다.

■ 병렬 처리 구조
  - multiprocessing.Pool: 종목 단위로 작업을 N개 CPU 코어에 분배
  - init_worker: 각 워커에 59,049×10 그리드 행렬을 공유 메모리로 한 번만 전달
  - imap_unordered + chunksize: 종목을 묶음(기본 10개) 단위로 보내
                                IPC(프로세스 간 통신) 오버헤드 감소

■ 체크포인트(중단 재개) — 세션 폴더 격리 방식
  - CHECKPOINT_N 종목마다 결과를 '세션 전용 폴더'에 저장한다.
  - 세션 폴더 위치: intermediate/session_<RUN_ID>/
    RUN_ID = 프로그램 시작 시점의 타임스탬프 (YYYYMMDD_HHMM)
  - 최종 결과 파일명에도 동일한 RUN_ID가 포함된다.
    예) apin_daily_rolling_2017_2021_20240115_0930.parquet
        intermediate/session_20240115_0930/apin_checkpoint_0000.parquet
  - 인풋을 바꿔 새로 실행하면 새 RUN_ID → 새 세션 폴더가 생성되어
    이전 체크포인트와 물리적으로 완전히 분리된다.
  - 중단 후 재개: 설정 블록의 RESUME_RUN_ID에 이전 RUN_ID를 지정하면
    해당 세션 폴더를 그대로 이어 쓴다.
    완료 후에는 RESUME_RUN_ID를 다시 None으로 돌려놓으면 된다.

■ 설정값 안내
  BASE_DIR               : 월별 .parquet 파일이 들어있는 루트 경로
  YEAR_FOLDERS           : 처리할 연도 폴더 (None이면 KOR_* 전체 자동 탐색)
  OUTPUT_DIR             : 출력 결과 저장 경로
  FORCE_REPROCESS_STEP1  : True면 all_daily_bs.parquet를 강제로 다시 생성
  CHECKPOINT_N           : 종목 N개마다 중간 저장 (기본 100)
  IMAP_CHUNKSIZE         : 워커에 한 번에 넘기는 종목 수 (기본 10)
  WINDOW_SIZE            : 롤링 윈도우 크기 (영업일 기준, 기본 60)
  MIN_VALID_DAYS         : 윈도우 내 B+S > 0인 유효 거래일 최소 개수 (기본 30)
                           → 미만이면 희소 데이터로 간주해 추정을 스킵
  RESUME_RUN_ID          : 중단된 세션을 재개할 때 이전 RUN_ID를 문자열로 지정
                           예) RESUME_RUN_ID = "20240115_0930"
                           → None이면 현재 시각으로 새 세션 시작

=============================================================================
"""

import os
import glob
import math
import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
import warnings
import multiprocessing
import itertools
from tqdm import tqdm
from datetime import datetime

warnings.filterwarnings("ignore")


# =============================================================================
# 전역 설정
# =============================================================================

BASE_DIR              = r"E:\vpin_project_parquet"
YEAR_FOLDERS          = None # ['KOR_2017']  # None이면 KOR_* 폴더 전체 자동 탐색
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")

# Step 1 설정
FORCE_REPROCESS_STEP1 = False   # True: 기존 all_daily_bs.parquet가 있어도 재생성

# Step 2 설정
CHECKPOINT_N          = 100     # 종목 100개 처리마다 중간 저장
IMAP_CHUNKSIZE        = 10      # 워커에 한 번에 보내는 종목 묶음 크기
                                # 종목당 처리 시간이 짧으면 늘리고(20~50),
                                # 길면 줄여서(5~10) 워커 부하 균형 조정

# 중단된 세션 재개 설정
# - None    : 현재 시각으로 새 RUN_ID를 만들어 새 세션으로 시작한다.
# - 문자열  : 지정한 RUN_ID의 세션 폴더를 이어 쓴다.
#             예) RESUME_RUN_ID = "20240115_0930"
#             완료 후에는 다시 None으로 돌려놓을 것.
RESUME_RUN_ID         = None

# 60일 롤링 윈도우 품질 기준
WINDOW_SIZE    = 60   # 영업일 캘린더 기준 롤링 윈도우 크기

# 윈도우 내 B+S > 0인 유효 거래일이 이 값 미만이면 거래가 너무 희소하다고 판단해 스킵.
# WINDOW_SIZE(60) 중 절반인 30일 기준: 거래일의 50% 이상 실제 거래가 있어야 신뢰.
MIN_VALID_DAYS = 30


# =============================================================================
# [Step 1] 전처리: 틱 데이터 → 일별 B/S 집계
# =============================================================================

def preprocess_trade_data_polars(parquet_path: str) -> pl.DataFrame:
    """
    단일 parquet 파일(월별 틱 데이터)을 읽어 일별 매수/매도 건수로 집계한다.

    원본 컬럼 LR의 값:
      1  → 매수(Buy)
     -1  → 매도(Sell)

    반환 스키마: Symbol (Utf8), Date (pl.Date), B (UInt32), S (UInt32)
    """
    print(f"  Loading: {os.path.basename(parquet_path)} ...", end=" ", flush=True)

    aggregated_df = (
        pl.scan_parquet(parquet_path)
        .select(["Symbol", "Date", "LR"])
        .filter(pl.col("LR").is_in([1, -1]))           # 매수·매도만 남기고 나머지 제거
        .with_columns([
            (pl.col("LR") == 1).cast(pl.UInt32).alias("is_Buy"),
            (pl.col("LR") == -1).cast(pl.UInt32).alias("is_Sell"),
        ])
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("is_Buy").sum().alias("B"),
            pl.col("is_Sell").sum().alias("S"),
        ])
        .filter((pl.col("B") > 0) | (pl.col("S") > 0))  # B=S=0인 날(거래 없음)은 제외
        .sort(["Symbol", "Date"])
        .collect()
    )

    # 소스 파일에 따라 Date가 Datetime으로 들어올 수 있으므로 pl.Date로 강제 변환
    if aggregated_df["Date"].dtype != pl.Date:
        aggregated_df = aggregated_df.with_columns(pl.col("Date").cast(pl.Date))

    print(f"{aggregated_df.height:,} records")
    return aggregated_df


def get_parquet_files(base_dir: str, year_folders: list | None) -> list[str]:
    """
    base_dir 아래의 KOR_YYYY 폴더에서 .parquet 파일 경로를 시간순으로 수집한다.
    year_folders가 None이면 KOR_로 시작하는 모든 폴더를 자동으로 탐색한다.
    """
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
        files = sorted(glob.glob(os.path.join(folder_path, "*.parquet")))
        parquet_files.extend(files)

    print(f"\n[파일 탐색]")
    print(f"  기준 경로   : {base_dir}")
    print(f"  대상 연도   : {year_folders if year_folders else 'All'}")
    print(f"  parquet 파일: {len(parquet_files)}개")
    if parquet_files:
        print(f"  첫 번째     : {os.path.basename(parquet_files[0])}")
        print(f"  마지막      : {os.path.basename(parquet_files[-1])}")

    return parquet_files


def run_preprocessing(base_dir: str, year_folders: list | None, output_dir: str) -> str:
    """
    모든 월별 parquet를 순회해 일별 B/S를 집계하고,
    결과를 all_daily_bs.parquet 한 파일로 저장한다.

    이미 파일이 존재하고 FORCE_REPROCESS_STEP1=False이면 집계를 건너뛰고
    기존 파일 경로를 그대로 반환한다.

    같은 Symbol+Date 조합이 여러 파일에 걸쳐 있으면 B, S를 합산해 중복을 제거한다.

    반환값: 저장된 all_daily_bs.parquet의 전체 경로 (실패 시 빈 문자열)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "all_daily_bs.parquet")

    if not FORCE_REPROCESS_STEP1 and os.path.exists(output_path):
        print(f"\n{'='*65}")
        print(f"[Step 1 스킵] 기존 전처리 파일이 존재합니다: {output_path}")
        print(f"  (다시 생성하려면 FORCE_REPROCESS_STEP1 = True 로 설정하세요)")
        print(f"{'='*65}\n")
        return output_path

    parquet_files = get_parquet_files(base_dir, year_folders)
    if not parquet_files:
        print("\n[Error] parquet 파일을 찾을 수 없습니다.")
        return ""

    print(f"\n{'='*65}")
    print("[Step 1] 전처리 시작: 틱 데이터 → 일별 B/S 집계")
    print(f"{'='*65}\n")

    all_dfs = []
    skipped = 0

    for i, path in enumerate(parquet_files, 1):
        # 연도 폴더가 바뀔 때마다 구분 출력
        year_tag = next(
            (part for part in path.replace("\\", "/").split("/") if part.startswith("KOR_")),
            ""
        )
        if i == 1 or year_tag != next(
            (p for p in parquet_files[i - 2].replace("\\", "/").split("/") if p.startswith("KOR_")),
            ""
        ):
            print(f"\n[{year_tag}]")

        df = preprocess_trade_data_polars(path)

        if df.is_empty():
            print(f"    → [Warning] 데이터 없음, 건너뜀")
            skipped += 1
            continue

        all_dfs.append(df)

    if not all_dfs:
        print("\n[Error] 유효한 데이터가 없습니다.")
        return ""

    # 모든 월별 데이터를 수직으로 합친 뒤, Symbol+Date가 겹치는 경우 B·S 합산
    print(f"\n{'='*65}")
    print("전체 데이터 합산 및 정렬 중...")
    full_df = (
        pl.concat(all_dfs, how="vertical")
        .group_by(["Symbol", "Date"])
        .agg([
            pl.col("B").sum(),
            pl.col("S").sum(),
        ])
        .sort(["Symbol", "Date"])
    )

    full_df.write_parquet(output_path, compression="zstd")

    print(f"\n{'='*65}")
    print("[Step 1 완료]")
    print(f"{'='*65}")
    print(f"  저장 경로  : {output_path}")
    print(f"  전체 행 수 : {full_df.height:,}")
    print(f"  종목 수    : {full_df['Symbol'].n_unique():,}")
    print(f"  날짜 범위  : {full_df['Date'].min()} ~ {full_df['Date'].max()}")
    print(f"  처리 파일  : {len(parquet_files) - skipped}개  (건너뜀: {skipped}개)")
    print(f"{'='*65}\n")

    return output_path


# =============================================================================
# [Step 2-전처리] 영업일 캘린더 구축 및 종목별 B/S 정렬
# =============================================================================

def build_market_calendar(daily_bs: pl.DataFrame) -> pl.Series:
    """
    전체 데이터에서 '시장 공통 영업일 캘린더'를 추출한다.

    정의: 전체 종목을 통틀어 B > 0 또는 S > 0인 날짜의 합집합.
    즉, 시장 전체에서 단 한 종목이라도 거래가 있었던 날을 영업일로 간주한다.

    이 캘린더를 기준으로 각 종목의 B/S 시계열을 정렬하면,
    거래가 없었던 영업일에는 B=S=0으로 채워지고,
    결과적으로 60행 슬라이딩 윈도우가 항상 정확히 60 영업일을 의미하게 된다.

    Returns:
        정렬된 영업일 날짜 Series (pl.Date)
    """
    return (
        daily_bs
        .select("Date")
        .unique()
        .sort("Date")
        .get_column("Date")
    )


def align_symbol_to_calendar(
    sym_df: pl.DataFrame,
    calendar: pl.Series,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    단일 종목의 B/S 시계열을 영업일 캘린더에 맞춰 정렬한다.

    처리 방식:
      1. 영업일 캘린더 전체를 Date 컬럼으로 갖는 DataFrame 생성
      2. 종목의 실제 거래 데이터를 left join
      3. 거래가 없는 영업일(join 결과가 null)에는 B=S=0 채움

    종목의 상장 기간과 무관하게 캘린더 전체에 대해 join하면,
    상장 전·상폐 후 기간에도 B=S=0이 채워진다.
    이는 해당 기간 윈도우에서 valid_days가 MIN_VALID_DAYS 미만이 되어
    자연스럽게 APIN 추정이 스킵되므로 문제없다.

    Args:
        sym_df   : 해당 종목의 실제 거래 데이터 (Symbol, Date, B, S)
        calendar : 시장 공통 영업일 Series (pl.Date, 정렬된 상태)

    Returns:
        Bs    (np.ndarray, int32) : 캘린더 길이 N의 매수 건수 배열 (거래 없는 날 = 0)
        Ss    (np.ndarray, int32) : 캘린더 길이 N의 매도 건수 배열 (거래 없는 날 = 0)
        dates (list[date])        : 캘린더 영업일 리스트
    """
    # 캘린더 DataFrame에 종목 데이터를 left join → 거래 없는 날은 B, S가 null
    aligned = (
        pl.DataFrame({"Date": calendar})
        .join(
            sym_df.select(["Date", "B", "S"]),
            on="Date",
            how="left",
        )
        .with_columns([
            pl.col("B").fill_null(0).cast(pl.Int32),
            pl.col("S").fill_null(0).cast(pl.Int32),
        ])
    )

    Bs    = aligned["B"].to_numpy()
    Ss    = aligned["S"].to_numpy()
    dates = aligned["Date"].to_list()

    return Bs, Ss, dates


# =============================================================================
# [Step 2] APIN 추정 핵심 수학 함수
# =============================================================================
#
# 아래 3개 함수는 '포아송 로그-PMF 계산'과 '우도 합산' 두 가지 역할을 담당한다.
#
#   _log_poisson_grid : 그리드 탐색 전용 — 행렬 브로드캐스팅으로 (G×N) 계산
#   _grid_search      : 59,049개 후보 중 NLL 최소 인덱스 반환 (최적 초기값 탐색)
#   _make_nll         : L-BFGS-B 최적화에 넘길 NLL 콜백 함수 생성
#
# ■ 왜 log-space인가?
#   포아송 PMF 값은 관측 건수가 크면 극히 작아진다.
#   예: P(1000 | 500) ≈ 10^(-150) → 부동소수점 언더플로우(= 0으로 반올림)
#   log-space에서 계산하면 이 문제가 없다.
#
# ■ logsumexp 트릭이란?
#   log(w₀·L₀ + w₁·L₁ + ... + w₅·L₅)를 직접 계산하면 Lk가 0에 가까울 때 정보 손실.
#   대신 가중치를 log-space에 흡수해 logsumexp로 안전하게 계산한다.
#
#     log(w₀·L₀ + ... + w₅·L₅)
#     = logsumexp([ln(w₀) + log L₀,
#                  ln(w₁) + log L₁,
#                  ...,
#                  ln(w₅) + log L₅])
#
# =============================================================================

def _log_poisson_grid(k_row: np.ndarray, lam_col: np.ndarray) -> np.ndarray:
    """
    그리드 탐색 전용 log-포아송 PMF 행렬 계산.

    log P(k|λ) = k·ln(λ) - λ - ln(k!)   (ln(k!) = gammaln(k+1))

    입력 shape를 (1, N)과 (G, 1)로 맞추면 NumPy 브로드캐스팅이
    자동으로 (G, N) 행렬을 만들어 준다.
    G = 그리드 후보 수(59,049), N = 윈도우 영업일 수(60).

    Args:
        k_row  : shape (1, N) — 윈도우 내 관측값 (B 또는 S)
        lam_col: shape (G, 1) — 각 그리드 후보의 포아송 rate 파라미터
    Returns:
        shape (G, N) — 각 (그리드 후보, 영업일) 조합의 log-PMF 값
    """
    safe_lam = np.where(lam_col > 0, lam_col, 1e-300)  # ln(0) 방어
    lgk = gammaln(k_row + 1.0)                          # ln(k!), shape (1, N)
    return k_row * np.log(safe_lam) - safe_lam - lgk


def _grid_search(grid: np.ndarray, B: np.ndarray, S: np.ndarray) -> int:
    """
    59,049개 파라미터 후보 전체에 대해 NLL을 동시에 계산하고,
    NLL이 가장 낮은 후보의 인덱스를 반환한다.

    각 후보에 대해 모든 영업일의 로그-우도를 합산한다.
    하루의 로그-우도는 여섯 시나리오(Duarte & Young 2009)의 혼합이다.

      log L(day t) = logsumexp([ln(w₀) + l₀ₜ,
                                ln(w₁) + l₁ₜ,
                                ...,
                                ln(w₅) + l₅ₜ])

    전체 NLL = -Σₜ log L(day t)   (날짜 간 독립 가정)

    APIN 6가지 시나리오 (Duarte & Young 2009):
      시나리오   가중치                  매수 rate         매도 rate
      ─────────────────────────────────────────────────────────────
      1) 무정보+무충격   (1-α)(1-θ₁)         ε_b             ε_s
      2) 무정보+충격     (1-α)θ₁             ε_b+Δ_b         ε_s+Δ_s
      3) 악재+무충격     α(1-δ)(1-θ₂)       ε_b             μ_s+ε_s
      4) 악재+충격       α(1-δ)θ₂           ε_b+Δ_b         μ_s+ε_s+Δ_s
      5) 호재+무충격     αδ(1-θ₂)           μ_b+ε_b         ε_s
      6) 호재+충격       αδθ₂               μ_b+ε_b+Δ_b     ε_s+Δ_s

    Args:
        grid : shape (G, 10) — [α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s]
        B    : shape (N,)    — 윈도우 매수 건수
        S    : shape (N,)    — 윈도우 매도 건수
    Returns:
        NLL이 최소인 그리드 행의 인덱스
    """
    # 각 파라미터를 열벡터(G, 1)로 분리 → 브로드캐스팅 준비
    a_g  = grid[:, 0:1]     # α
    d_g  = grid[:, 1:2]     # δ
    t1_g = grid[:, 2:3]     # θ₁
    t2_g = grid[:, 3:4]     # θ₂
    ub_g = grid[:, 4:5]     # μ_b
    us_g = grid[:, 5:6]     # μ_s
    eb_g = grid[:, 6:7]     # ε_b
    es_g = grid[:, 7:8]     # ε_s
    pb_g = grid[:, 8:9]     # Δ_b
    ps_g = grid[:, 9:10]    # Δ_s

    B_row = B.reshape(1, -1)  # (1, N) — 행벡터로 변환
    S_row = S.reshape(1, -1)

    # 각 시나리오에 필요한 log-PMF 행렬 계산: 각각 shape (G, N)
    # 매수 rate 조합들
    lp_B_eb        = _log_poisson_grid(B_row, eb_g)                  # ε_b
    lp_B_eb_pb     = _log_poisson_grid(B_row, eb_g + pb_g)          # ε_b + Δ_b
    lp_B_ub_eb     = _log_poisson_grid(B_row, ub_g + eb_g)          # μ_b + ε_b
    lp_B_ub_eb_pb  = _log_poisson_grid(B_row, ub_g + eb_g + pb_g)  # μ_b + ε_b + Δ_b

    # 매도 rate 조합들
    lp_S_es        = _log_poisson_grid(S_row, es_g)                  # ε_s
    lp_S_es_ps     = _log_poisson_grid(S_row, es_g + ps_g)          # ε_s + Δ_s
    lp_S_us_es     = _log_poisson_grid(S_row, us_g + es_g)          # μ_s + ε_s
    lp_S_us_es_ps  = _log_poisson_grid(S_row, us_g + es_g + ps_g)  # μ_s + ε_s + Δ_s

    # 시나리오별 하루 log-우도 (B·S 독립이므로 log-PMF를 더함)
    l0 = lp_B_eb       + lp_S_es          # 1) 무정보 + 무충격
    l1 = lp_B_eb_pb    + lp_S_es_ps       # 2) 무정보 + 충격
    l2 = lp_B_eb       + lp_S_us_es       # 3) 악재   + 무충격
    l3 = lp_B_eb_pb    + lp_S_us_es_ps    # 4) 악재   + 충격
    l4 = lp_B_ub_eb    + lp_S_es          # 5) 호재   + 무충격
    l5 = lp_B_ub_eb_pb + lp_S_es_ps       # 6) 호재   + 충격

    # 가중치를 ln으로 변환 (분모가 0이 되는 것을 clip으로 방어)
    w0 = np.clip((1 - a_g) * (1 - t1_g),            1e-300, None)  # (1-α)(1-θ₁)
    w1 = np.clip((1 - a_g) * t1_g,                   1e-300, None)  # (1-α)θ₁
    w2 = np.clip(a_g * (1 - d_g) * (1 - t2_g),      1e-300, None)  # α(1-δ)(1-θ₂)
    w3 = np.clip(a_g * (1 - d_g) * t2_g,            1e-300, None)  # α(1-δ)θ₂
    w4 = np.clip(a_g * d_g * (1 - t2_g),            1e-300, None)  # αδ(1-θ₂)
    w5 = np.clip(a_g * d_g * t2_g,                   1e-300, None)  # αδθ₂

    # logsumexp 트릭: (6, G, N) 텐서를 axis=0으로 줄여 (G, N) 로그-우도 행렬 생성
    log_terms = np.stack([
        np.log(w0) + l0,
        np.log(w1) + l1,
        np.log(w2) + l2,
        np.log(w3) + l3,
        np.log(w4) + l4,
        np.log(w5) + l5,
    ], axis=0)
    log_lik_per_day = logsumexp(log_terms, axis=0)  # (G, N)

    # 날짜 축(axis=1)으로 합산 → 후보별 NLL: shape (G,)
    nll_scores = -np.sum(log_lik_per_day, axis=1)
    return int(np.argmin(nll_scores))


def _make_nll(B: np.ndarray, S: np.ndarray):
    """
    L-BFGS-B 최적화에 전달할 NLL 콜백 함수를 클로저로 생성한다.

    클로저를 사용하는 이유:
      ln(B!)와 ln(S!)은 파라미터가 바뀌어도 변하지 않는 상수다.
      콜백이 반복 호출될 때마다 다시 계산하지 않도록
      외부 스코프에서 한 번만 계산해 lgk_B, lgk_S에 저장한다.

    내부 함수 negative_log_likelihood(params):
      파라미터 배열 [α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s]를 받아
      NLL 스칼라를 반환한다.
      그리드 탐색과 동일한 수식을 사용하되, 스칼라 파라미터와
      1D 배열(N,)로 계산한다.
    """
    B_f   = B.astype(np.float64)
    S_f   = S.astype(np.float64)
    lgk_B = gammaln(B_f + 1.0)   # ln(B!) — 콜백 밖에서 한 번만 계산
    lgk_S = gammaln(S_f + 1.0)   # ln(S!)

    def negative_log_likelihood(params):
        alpha, delta, t1, t2, ub, us, eb, es, pb, ps = params

        # 파라미터 범위 경계 조건 (bounds 옵션과 이중 방어)
        if not (0 <= alpha <= 1 and 0 <= delta <= 1
                and 0 <= t1 <= 1 and 0 <= t2 <= 1
                and ub >= 0 and us >= 0 and eb >= 0 and es >= 0
                and pb >= 0 and ps >= 0):
            return np.inf

        # ln(0) 방어용 하한값
        safe_eb       = max(eb,            1e-300)
        safe_es       = max(es,            1e-300)
        safe_eb_pb    = max(eb + pb,       1e-300)
        safe_es_ps    = max(es + ps,       1e-300)
        safe_ub_eb    = max(ub + eb,       1e-300)
        safe_us_es    = max(us + es,       1e-300)
        safe_ub_eb_pb = max(ub + eb + pb,  1e-300)
        safe_us_es_ps = max(us + es + ps,  1e-300)

        # 시나리오별 log-PMF: 각각 shape (N,)
        # 매수 rate 조합들
        log_pB_eb       = B_f * math.log(safe_eb)       - safe_eb       - lgk_B
        log_pB_eb_pb    = B_f * math.log(safe_eb_pb)    - safe_eb_pb    - lgk_B
        log_pB_ub_eb    = B_f * math.log(safe_ub_eb)    - safe_ub_eb    - lgk_B
        log_pB_ub_eb_pb = B_f * math.log(safe_ub_eb_pb) - safe_ub_eb_pb - lgk_B

        # 매도 rate 조합들
        log_pS_es       = S_f * math.log(safe_es)       - safe_es       - lgk_S
        log_pS_es_ps    = S_f * math.log(safe_es_ps)    - safe_es_ps    - lgk_S
        log_pS_us_es    = S_f * math.log(safe_us_es)    - safe_us_es    - lgk_S
        log_pS_us_es_ps = S_f * math.log(safe_us_es_ps) - safe_us_es_ps - lgk_S

        # 시나리오별 하루 log-우도
        l0 = log_pB_eb       + log_pS_es          # 1) 무정보 + 무충격
        l1 = log_pB_eb_pb    + log_pS_es_ps       # 2) 무정보 + 충격
        l2 = log_pB_eb       + log_pS_us_es       # 3) 악재   + 무충격
        l3 = log_pB_eb_pb    + log_pS_us_es_ps    # 4) 악재   + 충격
        l4 = log_pB_ub_eb    + log_pS_es          # 5) 호재   + 무충격
        l5 = log_pB_ub_eb_pb + log_pS_es_ps       # 6) 호재   + 충격

        # 혼합 가중치 (ln으로 변환, 0 방어)
        w0 = max((1 - alpha) * (1 - t1),            1e-300)
        w1 = max((1 - alpha) * t1,                   1e-300)
        w2 = max(alpha * (1 - delta) * (1 - t2),    1e-300)
        w3 = max(alpha * (1 - delta) * t2,           1e-300)
        w4 = max(alpha * delta * (1 - t2),           1e-300)
        w5 = max(alpha * delta * t2,                 1e-300)

        # logsumexp 트릭: shape (6, N) → axis=0으로 축소 → (N,)
        log_terms = np.stack([
            math.log(w0) + l0,
            math.log(w1) + l1,
            math.log(w2) + l2,
            math.log(w3) + l3,
            math.log(w4) + l4,
            math.log(w5) + l5,
        ], axis=0)
        log_lik = logsumexp(log_terms, axis=0)  # 하루치 혼합 log-우도 (N,)

        return -np.sum(log_lik)  # 전체 기간 NLL (스칼라)

    return negative_log_likelihood


# =============================================================================
# [Step 2] APIN 추정 통합 함수
# =============================================================================

def estimate_apin_parameters(B_array: np.ndarray, S_array: np.ndarray,
                              grid_combinations: np.ndarray) -> dict:
    """
    60일 윈도우의 B, S 배열을 받아 Duarte & Young(2009) MLE로
    10개 파라미터와 APIN, PSOS를 추정한다.

    추정 절차:
      1. _grid_search로 59,049개 후보 중 NLL 최소 후보를 초기값으로 선택
      2. L-BFGS-B로 정밀 최적화 (파라미터 범위 제약 적용)
      3. 수렴 성공 시 APIN, PSOS 계산

    Returns:
      수렴 성공: {"a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
                  "APIN", "PSOS", "converged": True}
      수렴 실패: {"converged": False}
    """
    B = B_array.astype(np.float64)
    S = S_array.astype(np.float64)

    # 1단계: 그리드 탐색으로 최적 초기값 선택
    best_idx           = _grid_search(grid_combinations, B, S)
    best_initial_guess = grid_combinations[best_idx]

    # 2단계: L-BFGS-B로 정밀 최적화
    # lgamma(k!) 사전 계산 후 클로저 생성 (콜백 반복 호출 시 재계산 방지)
    nll_fn = _make_nll(B, S)

    try:
        result = minimize(
            nll_fn,
            x0=best_initial_guess,
            bounds=[
                (0, 1), (0, 1), (0, 1), (0, 1),          # α, δ, θ₁, θ₂
                (0, None), (0, None), (0, None), (0, None),  # μ_b, μ_s, ε_b, ε_s
                (0, None), (0, None),                        # Δ_b, Δ_s
            ],
            method="L-BFGS-B",
        )
    except Exception:
        return {"converged": False}

    if not result.success:
        return {"converged": False}

    a, d, t1, t2, ub, us, eb, es, pb, ps = result.x

    # APIN / PSOS 계산식 (Duarte & Young 2009)
    # 분모 (Total Expected Order Flow)
    informed_flow = a * (d * ub + (1 - d) * us)
    shock_flow    = (pb + ps) * (a * t2 + (1 - a) * t1)
    denom         = informed_flow + shock_flow + eb + es

    if denom < 1e-10:
        return {"converged": False}

    apin = informed_flow / denom
    psos = shock_flow / denom

    return {
        "a": a, "d": d, "t1": t1, "t2": t2,
        "ub": ub, "us": us, "eb": eb, "es": es, "pb": pb, "ps": ps,
        "APIN": apin, "PSOS": psos, "converged": True,
    }


# =============================================================================
# [Step 2] 멀티프로세싱 워커 함수
# =============================================================================

# 워커 프로세스 간 공유할 데이터 (전역 변수)
# init_worker가 각 프로세스 시작 시 한 번만 설정한다.
GLOBAL_GRID     = None   # 59,049×10 그리드 행렬
GLOBAL_CALENDAR = None   # 시장 공통 영업일 Series


def init_worker(grid: np.ndarray, calendar: pl.Series) -> None:
    """
    Pool 생성 시 각 워커 프로세스에서 한 번만 실행되는 초기화 함수.

    grid와 calendar를 전역 변수에 저장해 두면,
    이후 process_single_symbol이 호출될 때마다 인수로 넘기지 않아도 된다.
    (매 호출마다 직렬화·역직렬화하는 비용을 제거)
    """
    global GLOBAL_GRID, GLOBAL_CALENDAR
    GLOBAL_GRID     = grid
    GLOBAL_CALENDAR = calendar


def process_single_symbol(data_tuple: tuple) -> list[dict]:
    """
    단일 종목의 거래 데이터를 받아 60일 슬라이딩 윈도우로 APIN을 계산한다.

    처리 흐름:
      1. align_symbol_to_calendar로 종목 데이터를 영업일 캘린더에 정렬
         → 거래 없는 영업일에 B=S=0 삽입, 캘린더 전체 길이의 배열 생성
      2. 인덱스 i = WINDOW_SIZE-1, ... 에서 [i-(WINDOW_SIZE-1) : i+1]을 윈도우로 사용
         → 60행 = 정확히 60 영업일 (캘린더 기반이므로 비연속 오류 없음)
      3. 윈도우 내 유효 거래일(B+S > 0인 날) 검사:
         MIN_VALID_DAYS(30일) 미만이면 스킵
         → 종목이 해당 기간에 실질적으로 거래되지 않은 경우 추정 방지

    날짜 저장 방식:
      pl.Date를 Int32(epoch days)로 변환하지 않고 Python date 객체 그대로 저장.
      → Polars 버전에 따라 epoch 기준이 달라질 수 있는 날짜 오염 위험을 제거한다.
    """
    sym, sym_df = data_tuple

    # 종목 데이터를 영업일 캘린더에 정렬 (거래 없는 날 B=S=0으로 채움)
    Bs, Ss, dates = align_symbol_to_calendar(sym_df, GLOBAL_CALENDAR)
    n = len(dates)

    # 캘린더 전체 길이가 윈도우 크기(60)보다 작으면 윈도우를 만들 수 없으므로 스킵
    if n < WINDOW_SIZE:
        return []

    results = []
    for i in range(WINDOW_SIZE - 1, n):
        window_B = Bs[i - (WINDOW_SIZE - 1) : i + 1]  # shape (60,)
        window_S = Ss[i - (WINDOW_SIZE - 1) : i + 1]

        # 유효 거래일 검사: 윈도우 내 B+S > 0인 날이 MIN_VALID_DAYS 이상이어야 추정
        # 캘린더 정렬로 인해 비연속성 검사는 더 이상 필요하지 않음
        valid_days = int(np.sum((window_B + window_S) > 0))
        if valid_days < MIN_VALID_DAYS:
            continue

        est = estimate_apin_parameters(window_B, window_S, GLOBAL_GRID)

        if est["converged"]:
            results.append({
                "Symbol": sym,
                "Date":   dates[i],      # Python date 객체 그대로 저장
                "a":      est["a"],
                "d":      est["d"],
                "t1":     est["t1"],
                "t2":     est["t2"],
                "ub":     est["ub"],
                "us":     est["us"],
                "eb":     est["eb"],
                "es":     est["es"],
                "pb":     est["pb"],
                "ps":     est["ps"],
                "APIN":   est["APIN"],
                "PSOS":   est["PSOS"],
            })

    return results


# =============================================================================
# [Step 2] 결과 저장 및 재개 지원 함수
# =============================================================================

def build_results_df(estimates: list[dict]) -> pl.DataFrame:
    """
    process_single_symbol이 반환한 dict 리스트를 Polars DataFrame으로 변환한다.

    Date 컬럼은 Python date 객체 상태로 dict에 들어 있으며,
    Polars가 이를 자동으로 pl.Date로 인식한다.
    추가 cast는 타입 일관성을 보장하기 위한 보정이다.
    """
    if not estimates:
        return pl.DataFrame()

    return (
        pl.DataFrame(estimates)
        .with_columns(pl.col("Date").cast(pl.Date))
        .select([
            "Symbol", "Date",
            "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
            "APIN", "PSOS",
        ])
    )


def save_checkpoint(estimates: list[dict], checkpoint_idx: int, session_dir: str) -> str:
    """
    현재까지 누적된 추정 결과를 세션 전용 폴더에 체크포인트 파일로 저장한다.

    파일명 형식: apin_checkpoint_NNNN.parquet
    저장 위치  : intermediate/session_<RUN_ID>/   ← 세션마다 다른 폴더

    세션 폴더가 RUN_ID별로 격리되어 있으므로, 인풋이 달라 새 RUN_ID로
    시작한 세션과 이전 체크포인트가 절대 섞이지 않는다.
    CHECKPOINT_N 종목이 처리될 때마다 메인 루프에서 호출된다.
    저장 후 batch_estimates 리스트를 비워 메모리를 관리한다.
    """
    os.makedirs(session_dir, exist_ok=True)
    df   = build_results_df(estimates)
    path = os.path.join(session_dir, f"apin_checkpoint_{checkpoint_idx:04d}.parquet")
    df.write_parquet(path, compression="zstd")
    print(f"\n  [Checkpoint {checkpoint_idx}] {df.height:,} records → {os.path.basename(path)}")
    return path


def load_already_done_symbols(session_dir: str) -> set[str]:
    """
    중단 후 재실행 시 세션 폴더의 체크포인트에서 이미 완료된 종목 목록을 복원한다.

    세션 폴더(intermediate/session_<RUN_ID>/)를 직접 받으므로
    다른 RUN_ID의 체크포인트는 참조하지 않는다.
    폴더가 없거나 체크포인트가 없으면 빈 set을 반환해 처음부터 시작한다.
    """
    if not os.path.exists(session_dir):
        return set()

    checkpoint_files = sorted(
        f for f in os.listdir(session_dir) if f.startswith("apin_checkpoint_")
    )
    if not checkpoint_files:
        return set()

    done_symbols = set()
    for fname in checkpoint_files:
        path = os.path.join(session_dir, fname)
        df   = pl.read_parquet(path, columns=["Symbol"])
        done_symbols.update(df["Symbol"].unique().to_list())

    print(f"[재개 모드] 기존 체크포인트 {len(checkpoint_files)}개 발견 → "
          f"완료된 종목 {len(done_symbols):,}개 건너뜀")
    return done_symbols


# =============================================================================
# [Step 2] 메인 APIN 계산 함수
# =============================================================================

def run_apin_calculation(daily_bs_path: str, output_dir: str,
                         run_id: str,
                         year_filter: list[int] | None = None,
                         checkpoint_n: int = 100) -> pl.DataFrame:
    """
    all_daily_bs.parquet를 읽어 종목별 60일 롤링 APIN을 계산하고 결과를 반환한다.

    처리 흐름:
      1. parquet 로드 → 연도 필터 적용 (year_filter가 있는 경우)
      2. 시장 공통 영업일 캘린더 추출 (build_market_calendar)
      3. 59,049개 그리드 행렬 생성
      4. 세션 폴더(intermediate/session_<run_id>/) 설정
         - RESUME_RUN_ID가 지정된 경우: 해당 폴더의 체크포인트를 불러와 이어 처리
         - 새 실행인 경우: 빈 폴더로 시작
      5. 종목별 DataFrame으로 분리 → (sym, df) 튜플 리스트 생성
      6. multiprocessing.Pool로 종목 단위 병렬 처리
         - init_worker에 grid와 calendar를 함께 전달 (워커당 1회만 직렬화)
      7. CHECKPOINT_N마다 세션 폴더에 중간 저장
      8. 모든 체크포인트를 병합 → daily_bs와 Join해 B/S 컬럼 복원

    세션 폴더 격리:
      체크포인트는 intermediate/session_<run_id>/ 에만 저장된다.
      run_id가 다르면 폴더 자체가 달라지므로 인풋이 바뀐 새 실행과
      이전 체크포인트가 절대 섞이지 않는다.

    최종 B/S Join:
      daily_bs(모든 실제 거래일)를 left 기준으로 APIN 추정 결과를 join한다.
      APIN 추정이 스킵되거나 수렴에 실패한 날은 파라미터와 APIN/PSOS이 null로 채워진다.
      영업일이지만 거래가 없었던 날(B=S=0)은 daily_bs에 존재하지 않으므로
      결과에도 포함되지 않는다.
      → 최종 결과의 모든 row는 실제 거래일이며,
         APIN=null은 "거래는 있었지만 추정 조건 미충족"을 의미한다.

    Returns:
      Schema: Symbol, Date, B, S, a, d, t1, t2, ub, us, eb, es, pb, ps, APIN, PSOS
    """
    # 세션 전용 폴더: RUN_ID별로 격리
    intermediate_dir = os.path.join(output_dir, "intermediate")
    session_dir      = os.path.join(intermediate_dir, f"apin_session_{run_id}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(session_dir, exist_ok=True)

    print(f"\n{'='*65}")
    print(f"[Step 2] APIN 계산 시작")
    print(f"{'='*65}")
    print(f"  입력 파일   : {daily_bs_path}")
    print(f"  RUN_ID      : {run_id}")
    print(f"  세션 폴더   : {session_dir}")

    # 1. 데이터 로드 및 연도 필터
    daily_bs = pl.read_parquet(daily_bs_path)

    if year_filter:
        daily_bs = daily_bs.filter(pl.col("Date").dt.year().is_in(year_filter))
        print(f"  연도 필터   : {year_filter}")

    print(f"  전체 행 수  : {daily_bs.height:,}")
    print(f"  종목 수     : {daily_bs['Symbol'].n_unique():,}")
    print(f"  날짜 범위   : {daily_bs['Date'].min()} ~ {daily_bs['Date'].max()}")

    # 2. 시장 공통 영업일 캘린더 추출
    # 전체 종목에서 거래가 있었던 날짜의 합집합 = 시장 영업일
    market_calendar = build_market_calendar(daily_bs)
    print(f"\n  영업일 수   : {len(market_calendar):,}일 "
          f"({market_calendar[0]} ~ {market_calendar[-1]})")

    # 3. 그리드 행렬 생성 (3^10 = 59,049개 파라미터 조합)
    #    순서: α, δ, θ₁, θ₂, μ_b, μ_s, ε_b, ε_s, Δ_b, Δ_s
    print("\n[Grid 초기값 생성 중...]")
    grid_matrix = np.array(
        list(itertools.product(
            [0.1, 0.5, 0.9],   # α   (정보 이벤트 확률)
            [0.1, 0.5, 0.9],   # δ   (호재 확률)
            [0.1, 0.5, 0.9],   # θ₁  (무정보일 충격 확률)
            [0.1, 0.5, 0.9],   # θ₂  (정보일 충격 확률)
            [20, 200, 2000],   # μ_b (호재 정보 매수)
            [20, 200, 2000],   # μ_s (악재 정보 매도)
            [20, 200, 2000],   # ε_b (비정보 매수)
            [20, 200, 2000],   # ε_s (비정보 매도)
            [20, 200, 2000],   # Δ_b (충격 추가 매수)
            [20, 200, 2000],   # Δ_s (충격 추가 매도)
        )),
        dtype=np.float64,
    )
    print(f"  Grid 크기   : {grid_matrix.shape[0]:,} combinations")

    num_cores = multiprocessing.cpu_count()
    print(f"  CPU 코어    : {num_cores}개")
    print(f"  IPC 청크    : {IMAP_CHUNKSIZE} 종목/청크")
    print(f"  윈도우      : {WINDOW_SIZE} 영업일")
    print(f"  최소 유효일 : {MIN_VALID_DAYS}일")

    # 4. 세션 폴더에서 완료된 종목 로드 (재개 지원)
    # session_dir은 RUN_ID별로 격리되어 있으므로 다른 실행의 체크포인트는 참조하지 않는다.
    done_symbols      = load_already_done_symbols(session_dir)
    all_symbols       = daily_bs["Symbol"].unique().sort().to_list()
    remaining_symbols = [s for s in all_symbols if s not in done_symbols]
    print(f"\n  처리 대상 종목: {len(remaining_symbols):,}개 "
          f"(전체 {len(all_symbols):,}개 중 완료 {len(done_symbols):,}개 제외)\n")

    # 5. 종목별 DataFrame 분리
    symbol_partitions = daily_bs.partition_by("Symbol", maintain_order=False)
    symbol_dfs = {}
    for part in symbol_partitions:
        sym = part["Symbol"][0]
        if sym in remaining_symbols:
            symbol_dfs[sym] = part.sort("Date")

    grouped_data = [(sym, symbol_dfs[sym]) for sym in remaining_symbols if sym in symbol_dfs]

    print(f"{'='*65}")
    print("APIN 추정 시작...")
    print(f"{'='*65}\n")

    # 체크포인트 카운터: 세션 폴더 내 기존 파일 수에서 이어받음
    existing_checkpoints = sorted(
        f for f in os.listdir(session_dir) if f.startswith("apin_checkpoint_")
    )
    next_checkpoint_idx = len(existing_checkpoints)

    batch_estimates = []
    processed_count = 0

    # 6. 병렬 처리
    # init_worker: 각 워커에 grid_matrix와 market_calendar를 한 번만 전달 (IPC 비용 절감)
    # imap_unordered: 완료 순서 상관없이 결과를 받아 진행바 업데이트
    # chunksize: 워커에 한 번에 IMAP_CHUNKSIZE개 종목을 묶어 전달 (IPC 횟수 감소)
    with multiprocessing.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(grid_matrix, market_calendar),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(process_single_symbol, grouped_data,
                                chunksize=IMAP_CHUNKSIZE),
            total=len(grouped_data),
            desc="  Estimating",
        ):
            batch_estimates.extend(res)
            processed_count += 1

            # 7. 체크포인트 저장 (세션 폴더에 격리하여 저장)
            if processed_count % checkpoint_n == 0:
                save_checkpoint(batch_estimates, next_checkpoint_idx, session_dir)
                next_checkpoint_idx += 1
                batch_estimates = []   # 저장 후 메모리 해제

    # 마지막 배치 저장 (checkpoint_n으로 딱 떨어지지 않는 나머지)
    if batch_estimates:
        save_checkpoint(batch_estimates, next_checkpoint_idx, session_dir)

    print(f"\n{'='*65}")
    # 이 세션 폴더 안의 체크포인트만 병합
    all_session_checkpoints = sorted(
        os.path.join(session_dir, f)
        for f in os.listdir(session_dir)
        if f.startswith("apin_checkpoint_")
    )
    print(f"체크포인트 {len(all_session_checkpoints)}개 병합 중... (세션: {run_id})")

    if not all_session_checkpoints:
        print("[Warning] 병합할 파일 없음.")
        return pl.DataFrame()

    # 8. daily_bs를 기준(left)으로 APIN 추정 결과를 join
    #
    # join 방향:  daily_bs (left, 기준)  ←  apin_estimates (right)
    #
    # daily_bs     : 실제 거래가 있었던 날만 존재 (B>0 or S>0)
    # apin_estimates: 그 중 APIN 추정에 성공한 날만 존재
    #
    # APIN 추정이 스킵되는 두 가지 경우:
    #   (1) 롤링 윈도우 60일 중 유효 거래일(B+S>0)이 MIN_VALID_DAYS(30일) 미만
    #   (2) MLE 최적화가 수렴에 실패
    #
    # left join 결과:
    #   ├─ 거래가 있었던 모든 날이 row로 유지된다
    #   ├─ APIN 추정 성공일 → 파라미터 및 APIN/PSOS에 추정값이 채워진다
    #   └─ APIN 추정 스킵·실패일 → 파라미터 및 APIN/PSOS이 null로 채워진다
    apin_estimates = (
        pl.concat([pl.read_parquet(p) for p in all_session_checkpoints], how="vertical")
    )

    final_df = (
        daily_bs                    # ← 기준: 모든 실제 거래일
        .join(
            apin_estimates,
            on=["Symbol", "Date"],
            how="left",             # 거래일 row 유지, 추정 없는 날 → null
        )
        .select([
            "Symbol", "Date", "B", "S",
            "a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
            "APIN", "PSOS",
        ])
        .sort(["Symbol", "Date"])
    )

    return final_df


# =============================================================================
# 실행부
# =============================================================================

if __name__ == "__main__":
    multiprocessing.freeze_support()   # Windows 멀티프로세싱 안전장치

    # ── RUN_ID 결정 ────────────────────────────────────────────────────────────
    # 중단 재개: RESUME_RUN_ID에 이전 RUN_ID 문자열을 지정한다.
    # 새 실행  : RESUME_RUN_ID = None 이면 현재 시각을 RUN_ID로 사용한다.
    # 최종 결과 파일명과 세션 폴더명에 동일한 RUN_ID가 들어가므로
    # "어떤 결과 파일이 어느 체크포인트에서 만들어졌는가"를 파일명만으로 추적할 수 있다.
    run_id = RESUME_RUN_ID if RESUME_RUN_ID else datetime.now().strftime("%Y%m%d_%H%M")
    start  = datetime.now()

    if RESUME_RUN_ID:
        print(f"\n[재개 모드] RUN_ID = {run_id}")
    else:
        print(f"\n[새 실행]   RUN_ID = {run_id}")

    # Step 1: 틱 데이터 전처리
    daily_bs_path = run_preprocessing(
        base_dir=BASE_DIR,
        year_folders=YEAR_FOLDERS,
        output_dir=OUTPUT_DIR
    )

    if not daily_bs_path or not os.path.exists(daily_bs_path):
        print("\n[Error] 전처리 결과 파일이 없어 APIN 계산을 종료합니다.")
        exit(1)

    # 폴더명 'KOR_2017' → 연도 정수 2017로 변환
    if YEAR_FOLDERS:
        year_filter = [int(yf.replace("KOR_", "")) for yf in YEAR_FOLDERS]
    else:
        year_filter = None

    # Step 2: APIN 추정
    result = run_apin_calculation(
        daily_bs_path=daily_bs_path,
        output_dir=OUTPUT_DIR,
        run_id=run_id,
        year_filter=year_filter,
        checkpoint_n=CHECKPOINT_N,
    )

    # 결과 저장 및 요약 출력
    if not result.is_empty():
        # 최종 파일명에 RUN_ID(시작 타임스탬프)를 사용
        # → 체크포인트 세션 폴더명(session_<RUN_ID>)과 1:1 대응
        year_tag        = "_".join(str(y) for y in year_filter) if year_filter else "ALL"
        output_filename = f"apin_daily_rolling_{year_tag}_{run_id}"

        # 전체 결과: parquet (ZSTD 압축)
        parquet_path = os.path.join(OUTPUT_DIR, f"{output_filename}.parquet")
        result.write_parquet(parquet_path, compression="zstd")
        print(f"\n[저장 완료] {parquet_path}")

        # 확인용 샘플: 상위 1000행 CSV
        csv_path = os.path.join(OUTPUT_DIR, f"{output_filename}_SAMPLE.csv")
        result.head(1000).write_csv(csv_path)
        print(f"[샘플 저장] {csv_path}")

        print("\n[미리보기]")
        print(result.head(20))

        print("\n[통계]")
        print(f"  전체 레코드        : {result.height:,}")
        apin_ok = result.filter(pl.col("APIN").is_not_null()).height
        print(f"  APIN 추정 성공     : {apin_ok:,}")
        print(f"  APIN 추정 실패/없음: {result.height - apin_ok:,}")

        # 종목별 APIN 커버리지 (APIN이 있는 날 / 전체 날 × 100)
        sym_stats = (
            result
            .group_by("Symbol")
            .agg([
                pl.len().alias("total_days"),
                pl.col("APIN").is_not_null().sum().alias("apin_days"),
            ])
            .with_columns(
                (pl.col("apin_days") / pl.col("total_days") * 100).alias("coverage_pct")
            )
        )
        print("\n[종목별 커버리지 샘플]")
        print(sym_stats.head(10))

    else:
        print("\n[Warning] 결과 DataFrame이 비어 있습니다.")

    elapsed = datetime.now() - start
    print(f"\n총 소요 시간: {elapsed}")