"""
=============================================================================
[통합 파이프라인] 틱 데이터 전처리 및 일별 롤링 APIN 계산
                  — rpy2 + PINstimation::adjpin() 연동 버전 —
=============================================================================

■ 변경 이력  (기존 순수 Python MLE 대비)
  ─────────────────────────────────────────────────────────────────────────
  삭제  │ _log_poisson_grid, _grid_search, _make_nll  (파이썬 기반 MLE)
        │ GLOBAL_GRID  (59,049개 초기값 그리드 행렬)
        │ estimate_apin_parameters  (그리드→L-BFGS-B 2단계 최적화)
        │ scipy.optimize, scipy.special, itertools, math  임포트
  ─────────────────────────────────────────────────────────────────────────
  추가  │ rpy2 라이브러리를 통한 R 환경 호출
        │ PINstimation::adjpin() 실행 로직
        │   - EM 알고리즘 + 군집(Clustering) 기반 초기값 자동 적용
        │   - R 측에서 수치 최적화·수렴 판정까지 완결
  ─────────────────────────────────────────────────────────────────────────
  유지  │ [Step 1] Polars 기반 틱 데이터 전처리 (all_daily_bs.parquet)
        │ 데이터 정렬: build_market_calendar, align_symbol_to_calendar
        │ 멀티프로세싱 및 체크포인트 구조 (tqdm, Session 관리 등)
  ─────────────────────────────────────────────────────────────────────────

■ 전체 흐름 요약
  이 코드는 두 단계(Step)로 나뉜다.

  [Step 1] 원시 틱 데이터 → 일별 매수/매도 건수(B/S) 집계
    - 월별로 나뉜 .parquet 파일들을 순회하며 종목·날짜 단위로 B, S를 집계
    - 결과를 all_daily_bs.parquet 한 파일로 저장
    - Step 1이 이미 완료됐으면 자동으로 스킵 (FORCE_REPROCESS_STEP1 = False)

  [Step 2] 일별 B/S → 종목별 60일 롤링 APIN 추정 (Duarte & Young 2009 모델)
    - all_daily_bs.parquet를 읽어 '시장 공통 영업일 캘린더'를 먼저 추출
    - 각 종목의 B/S를 영업일 캘린더에 Left Join → 거래 없는 영업일에 B=S=0 삽입
    - 60행 슬라이딩 윈도우로 순회하며 R의 PINstimation::adjpin()으로 APIN 추정
    - 멀티프로세싱으로 종목 단위 병렬화, 체크포인트로 중간 저장 지원
    - 최종 결과는 모든 실제 거래일을 row로 포함하며,
      APIN 추정 조건 미충족(유효 거래일 부족)이나 수렴 실패인 날은
      APIN 및 파라미터 컬럼이 null로 표시된다

■ PINstimation::adjpin() 사용 방식
  기존 순수 Python 구현에서는 59,049개 그리드 탐색 후 L-BFGS-B로 정밀 최적화했으나,
  이 버전에서는 R의 PINstimation 패키지가 제공하는 adjpin() 함수를 사용한다.

  adjpin()의 내부 동작:
    1. 초기값 생성: 군집(Clustering) 기반 알고리즘으로 데이터 분포에 적합한
       초기 파라미터 세트를 자동 생성 (initialsets = "CL")
    2. EM 알고리즘: 각 초기값에서 출발하여 Expectation-Maximization으로 MLE 수렴
    3. 모델 선택: 여러 초기값에서 수렴한 결과 중 최대 우도(log-likelihood)를 달성한
       파라미터 세트를 최종 추정치로 선택

  장점:
    - EM 알고리즘이 그리드+L-BFGS-B 대비 국소 최적에 빠질 확률이 낮음
    - Clustering 기반 초기값이 데이터 적응적(adaptive)이므로 고정 그리드 대비
      더 적은 초기값으로 더 좋은 수렴 품질 달성
    - PINstimation 패키지가 수치 안정성·경계 조건을 내부적으로 처리

■ rpy2 + 멀티프로세싱 구조
  rpy2의 내장 R(embedded R)은 프로세스 단위로 격리되어야 한다.
  Python의 multiprocessing은 Windows에서 기본적으로 spawn 방식을 사용하므로
  각 워커 프로세스가 독립적인 R 런타임을 초기화한다.

  init_worker 에서:
    1. rpy2의 numpy 자동변환을 활성화
    2. PINstimation 패키지를 로드
    3. adjpin() 래퍼 R 함수를 정의하여 전역 변수에 저장
       → 이후 process_single_symbol 호출 시 매번 패키지를 다시 로드하지 않음

  ※ Linux/macOS에서 fork 방식이 기본인 경우를 대비해
    multiprocessing.set_start_method("spawn") 을 명시적으로 설정한다.

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

■ 병렬 처리 구조
  - multiprocessing.Pool (spawn 방식): 종목 단위로 작업을 N개 CPU 코어에 분배
  - init_worker: 각 워커에서 rpy2 초기화 + PINstimation 패키지 로드 (1회)
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

■ 사전 요구사항 (rpy2 + PINstimation)
  1. R 설치 (4.0 이상 권장): https://cran.r-project.org/
  2. PINstimation 패키지 설치 (R 콘솔에서):
       install.packages("PINstimation")
  3. rpy2 파이썬 패키지 설치:
       pip install rpy2
  4. R_HOME 환경변수 설정 (rpy2가 R을 찾지 못하는 경우):
       Windows 예시: set R_HOME=C:\Program Files\R\R-4.3.2
       또는 Python 코드에서: os.environ["R_HOME"] = r"C:\Program Files\R\R-4.3.2"

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
  ADJPIN_INITIALSETS     : PINstimation adjpin() 초기값 생성 알고리즘
                           "CL" = Clustering(기본), "GE" = Grid-based(Ersan 2016),
                           "random" = 랜덤 생성
  ADJPIN_NUM_INIT        : 초기값 세트 개수 (None이면 PINstimation 기본값 사용)

=============================================================================
"""

import os
import glob
import numpy as np
import polars as pl
import warnings
import multiprocessing
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

# PINstimation::adjpin() 설정
ADJPIN_INITIALSETS = "CL"   # "CL"=Clustering, "GE"=Grid(Ersan), "random"=랜덤
ADJPIN_NUM_INIT    = None   # None → PINstimation 기본값 사용; 정수 지정 시 초기값 개수 고정


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
# [Step 2] APIN 추정 — rpy2 + PINstimation::adjpin() 연동
# =============================================================================
#
# 기존 3개 함수(_log_poisson_grid, _grid_search, _make_nll)와
# estimate_apin_parameters를 모두 제거하고,
# R의 PINstimation::adjpin() 단일 호출로 대체한다.
#
# ■ adjpin()의 내부 동작
#   1. initialsets 인수에 따라 초기 파라미터 세트를 자동 생성
#      - "CL" (Clustering): 매수/매도 분포를 군집 분석하여 데이터 적응적 초기값 생성
#      - "GE" (Grid-Ersan): Ersan(2016) 방식의 그리드 탐색
#   2. 각 초기값에서 ECM(Expectation Conditional Maximization) 알고리즘 실행
#   3. 수렴한 결과 중 최대 우도를 달성한 파라미터를 최종 추정치로 선택
#
# ■ 파라미터 이름 매핑 (PINstimation → 이 파이프라인)
#   PINstimation 출력        이 코드의 컬럼명    Duarte & Young(2009) 표기
#   ──────────────────────────────────────────────────────────────────
#   alpha                    a                    α
#   delta                    d                    δ
#   theta                    t1                   θ  (= θ₁, 무정보일 충격 확률)
#   thetap                   t2                   θ' (= θ₂, 정보일 충격 확률)
#   epsilon.b (또는 epsilonb) eb                  ε_b
#   epsilon.s (또는 epsilons) es                  ε_s
#   mu.b (또는 mub)          ub                   μ_b
#   mu.s (또는 mus)          us                   μ_s
#   d.b (또는 db)            pb                   Δ_b
#   d.s (또는 ds)            ps                   Δ_s
#
# ※ PINstimation 버전에 따라 파라미터 이름에 마침표(.) 유무가 다를 수 있다.
#   아래 코드는 양쪽 모두 대응하도록 유연하게 파싱한다.
#
# =============================================================================

# R 래퍼 함수 코드 (문자열)
# ─────────────────────────────────────────────────────────────────────────────
# init_worker에서 ro.r()로 한 번만 정의하고, 이후 반복 호출한다.
# adjpin() 결과인 S4 객체에서 파라미터·ADJPIN·PSOS를 추출하여
# 길이 12의 numeric vector로 반환한다.
# 실패 시 NA 12개를 반환하여 Python 측에서 수렴 실패로 처리한다.
#
# 반환 벡터 순서 (인덱스 0~11):
#   [0]  alpha    [1]  delta     [2]  theta(θ₁)  [3]  thetap(θ₂)
#   [4]  mub      [5]  mus       [6]  eb          [7]  es
#   [8]  db       [9]  ds        [10] ADJPIN      [11] PSOS
# ─────────────────────────────────────────────────────────────────────────────

_R_WRAPPER_CODE = """
.run_adjpin <- function(buys_vec, sells_vec, init_method, num_init) {
    data <- data.frame(buys = as.numeric(buys_vec),
                       sells = as.numeric(sells_vec))

    result <- tryCatch({
        if (is.null(num_init)) {
            PINstimation::adjpin(data,
                                initialsets = init_method,
                                verbose     = FALSE)
        } else {
            PINstimation::adjpin(data,
                                initialsets = init_method,
                                num_init    = as.integer(num_init),
                                verbose     = FALSE)
        }
    }, error = function(e) {
        NULL
    })

    if (is.null(result)) return(rep(NA_real_, 12))

    # S4 객체에서 슬롯 추출
    adjpin_val <- tryCatch(slot(result, "adjpin"), error = function(e) NA_real_)
    psos_val   <- tryCatch(slot(result, "psos"),   error = function(e) NA_real_)

    # @parameters: data.frame 또는 named vector
    p <- tryCatch(slot(result, "parameters"), error = function(e) NULL)
    if (is.null(p)) return(rep(NA_real_, 12))

    # data.frame이면 첫 행을 named vector로 변환
    if (is.data.frame(p)) {
        p <- unlist(p[1, , drop = TRUE])
    }
    pnames <- tolower(gsub("[._]", "", names(p)))

    # 이름 매핑 함수: 여러 가지 표기법에 대응
    get_param <- function(candidates) {
        for (cand in candidates) {
            idx <- which(pnames == cand)
            if (length(idx) > 0) return(as.numeric(p[idx[1]]))
        }
        return(NA_real_)
    }

    alpha  <- get_param(c("alpha"))
    delta  <- get_param(c("delta"))
    theta  <- get_param(c("theta"))
    thetap <- get_param(c("thetap", "theta'", "thetaprime"))
    mub    <- get_param(c("mub", "mub"))
    mus    <- get_param(c("mus", "mus"))
    eb     <- get_param(c("epsilonb", "eb", "epsb"))
    es     <- get_param(c("epsilons", "es", "epss"))
    db     <- get_param(c("db", "deltab", "db"))
    ds     <- get_param(c("ds", "deltas", "ds"))

    return(c(alpha, delta, theta, thetap, mub, mus, eb, es, db, ds,
             adjpin_val, psos_val))
}
"""


def estimate_apin_via_r(B_array: np.ndarray, S_array: np.ndarray) -> dict:
    """
    60일 윈도우의 B, S 배열을 받아 R의 PINstimation::adjpin()으로
    10개 파라미터와 APIN, PSOS를 추정한다.

    워커 프로세스의 전역 변수 _R_ADJPIN_FUNC (R 함수 객체)을 호출하며,
    이 함수는 init_worker에서 한 번만 정의된다.

    adjpin() 내부에서 EM 알고리즘 + Clustering 초기값으로 MLE를 수행하고,
    최적 파라미터·ADJPIN·PSOS를 반환한다.

    Args:
        B_array : shape (N,) — 윈도우 매수 건수 (int 또는 float)
        S_array : shape (N,) — 윈도우 매도 건수 (int 또는 float)

    Returns:
      수렴 성공: {"a", "d", "t1", "t2", "ub", "us", "eb", "es", "pb", "ps",
                  "APIN", "PSOS", "converged": True}
      수렴 실패: {"converged": False}
    """
    import rpy2.robjects as ro

    try:
        # Python 배열 → R IntegerVector 변환
        buys_r  = ro.IntVector(B_array.astype(int).tolist())
        sells_r = ro.IntVector(S_array.astype(int).tolist())

        # PINstimation 설정 전달
        init_method_r = ro.StrVector([ADJPIN_INITIALSETS])
        if ADJPIN_NUM_INIT is not None:
            num_init_r = ro.IntVector([ADJPIN_NUM_INIT])
        else:
            num_init_r = ro.NULL

        # R 래퍼 함수 호출 → 길이 12의 numeric vector 반환
        result_vec = _R_ADJPIN_FUNC(buys_r, sells_r, init_method_r, num_init_r)
        values = list(result_vec)

    except Exception:
        return {"converged": False}

    # NA 체크: 첫 번째 원소(alpha)가 NA이면 수렴 실패로 판정
    if len(values) != 12:
        return {"converged": False}

    import math
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in values):
        return {"converged": False}

    a, d, t1, t2, ub, us, eb, es, pb, ps, apin, psos = values

    return {
        "a": float(a), "d": float(d),
        "t1": float(t1), "t2": float(t2),
        "ub": float(ub), "us": float(us),
        "eb": float(eb), "es": float(es),
        "pb": float(pb), "ps": float(ps),
        "APIN": float(apin), "PSOS": float(psos),
        "converged": True,
    }


# =============================================================================
# [Step 2] 멀티프로세싱 워커 함수
# =============================================================================

# 워커 프로세스 간 공유할 데이터 (전역 변수)
# init_worker가 각 프로세스 시작 시 한 번만 설정한다.
GLOBAL_CALENDAR  = None   # 시장 공통 영업일 Series
_R_ADJPIN_FUNC   = None   # R 래퍼 함수 객체 (.run_adjpin)


def init_worker(calendar: pl.Series) -> None:
    """
    Pool 생성 시 각 워커 프로세스에서 한 번만 실행되는 초기화 함수.

    역할:
      1. market_calendar를 전역 변수에 저장 (IPC 비용 절감)
      2. rpy2 내장 R 런타임 초기화
      3. PINstimation 패키지 로드
      4. .run_adjpin() R 래퍼 함수 정의 및 전역 변수 등록

    spawn 방식 멀티프로세싱에서 각 워커는 독립된 프로세스이므로
    R 런타임도 프로세스마다 별도로 초기화된다. 이는 의도된 동작이다.
    """
    global GLOBAL_CALENDAR, _R_ADJPIN_FUNC
    GLOBAL_CALENDAR = calendar

    # ── rpy2 초기화 ──────────────────────────────────────────────────────────
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri

    # numpy ↔ R 자동변환 활성화 (IntVector/FloatVector 없이도 동작하지만,
    # 명시적 변환을 사용하므로 여기서는 선택 사항)
    try:
        numpy2ri.activate()
    except RuntimeError:
        pass  # 이미 활성화되어 있으면 무시

    # PINstimation 패키지 로드
    ro.r('suppressPackageStartupMessages(library(PINstimation))')

    # R 래퍼 함수 정의 → R 글로벌 환경에 .run_adjpin으로 등록
    ro.r(_R_WRAPPER_CODE)

    # Python에서 호출 가능한 R 함수 객체를 전역 변수에 저장
    _R_ADJPIN_FUNC = ro.globalenv['.run_adjpin']


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
      4. estimate_apin_via_r()로 R의 adjpin() 호출하여 파라미터 추정

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

        est = estimate_apin_via_r(window_B, window_S)

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
# [Step 2] rpy2 환경 사전 검증
# =============================================================================

def verify_r_environment() -> bool:
    """
    R 실행 환경과 PINstimation 패키지가 올바르게 설치되어 있는지 검증한다.

    검증 항목:
      1. rpy2 임포트 가능 여부
      2. R 런타임 접근 가능 여부
      3. PINstimation 패키지 설치 여부
      4. adjpin() 함수 존재 여부

    메인 프로세스에서 한 번 실행하여 워커 시작 전에 문제를 조기 발견한다.
    (워커에서 실패하면 디버깅이 어렵기 때문)

    Returns:
        True: 환경 정상 / False: 문제 발견
    """
    print("\n[R 환경 검증 중...]")

    # 1. rpy2 임포트
    try:
        import rpy2.robjects as ro
        print(f"  rpy2 버전    : {__import__('rpy2').__version__}")
    except ImportError:
        print("  [Error] rpy2가 설치되지 않았습니다.")
        print("          설치: pip install rpy2")
        return False

    # 2. R 런타임 접근
    try:
        r_version = ro.r('R.version.string')[0]
        print(f"  R 버전       : {r_version}")
    except Exception as e:
        print(f"  [Error] R 런타임 접근 불가: {e}")
        print("          R이 설치되어 있고 R_HOME이 올바르게 설정되었는지 확인하세요.")
        return False

    # 3. PINstimation 패키지
    try:
        ro.r('suppressPackageStartupMessages(library(PINstimation))')
        pkg_ver = ro.r('packageVersion("PINstimation")')[0]
        print(f"  PINstimation : v{pkg_ver}")
    except Exception as e:
        print(f"  [Error] PINstimation 패키지 로드 실패: {e}")
        print('          R 콘솔에서 install.packages("PINstimation") 을 실행하세요.')
        return False

    # 4. adjpin 함수 존재 확인
    try:
        exists = ro.r('is.function(PINstimation::adjpin)')[0]
        if not exists:
            raise RuntimeError("adjpin is not a function")
        print(f"  adjpin()     : 확인 완료")
    except Exception as e:
        print(f"  [Error] adjpin() 함수를 찾을 수 없습니다: {e}")
        return False

    print("  → R 환경 검증 통과\n")
    return True


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
      3. R 환경 검증 (rpy2 + PINstimation 사전 확인)
      4. 세션 폴더(intermediate/session_<run_id>/) 설정
         - RESUME_RUN_ID가 지정된 경우: 해당 폴더의 체크포인트를 불러와 이어 처리
         - 새 실행인 경우: 빈 폴더로 시작
      5. 종목별 DataFrame으로 분리 → (sym, df) 튜플 리스트 생성
      6. multiprocessing.Pool (spawn)로 종목 단위 병렬 처리
         - init_worker에 calendar를 전달 (워커당 1회만 직렬화)
         - 각 워커에서 rpy2 + PINstimation 초기화
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
    print(f"[Step 2] APIN 계산 시작  (rpy2 + PINstimation)")
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

    # 3. R 환경 사전 검증
    if not verify_r_environment():
        print("\n[Error] R 환경이 올바르지 않아 APIN 계산을 종료합니다.")
        print("        위의 에러 메시지를 확인하고 환경을 설정한 뒤 다시 실행하세요.")
        return pl.DataFrame()

    # PINstimation 설정 출력
    print(f"\n[PINstimation 설정]")
    print(f"  초기값 방법  : {ADJPIN_INITIALSETS} "
          f"({'Clustering' if ADJPIN_INITIALSETS == 'CL' else 'Grid-Ersan' if ADJPIN_INITIALSETS == 'GE' else ADJPIN_INITIALSETS})")
    print(f"  초기값 개수  : {'패키지 기본값' if ADJPIN_NUM_INIT is None else ADJPIN_NUM_INIT}")

    num_cores = multiprocessing.cpu_count()
    print(f"\n  CPU 코어    : {num_cores}개")
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
    # ─────────────────────────────────────────────────────────────────────────
    # spawn 방식을 명시적으로 사용:
    #   - rpy2의 내장 R(embedded R)은 fork 시 상태 복제 문제가 발생할 수 있음
    #   - spawn은 각 워커를 완전히 새로운 프로세스로 시작하므로 안전
    #   - Windows에서는 이미 기본값이 spawn이지만, Linux/macOS 호환을 위해 명시
    #
    # init_worker: 각 워커에서 rpy2 + PINstimation을 한 번만 초기화
    #   - 기존: init_worker(grid_matrix, market_calendar) → 그리드 전달
    #   - 변경: init_worker(market_calendar)              → R 환경 초기화
    # ─────────────────────────────────────────────────────────────────────────
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(
        processes=num_cores,
        initializer=init_worker,
        initargs=(market_calendar,),
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
    #   (2) PINstimation adjpin()이 수렴에 실패 (NA 반환)
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
