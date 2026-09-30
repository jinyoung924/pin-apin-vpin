# Is-VPIN-Informative

한국 주식시장 틱 데이터에서 정보거래확률(PIN / APIN / VPIN)을 일별·롤링으로 추정하는 연구용 파이프라인.

---

## 모델 개요

### PIN — Probability of Informed Trading (EKOP 1996)

시장 마이크로스트럭처 모델로, 정보 거래자가 전체 거래에서 차지하는 비율을 추정한다. 매일 세 시나리오(무정보 / 악재 / 호재) 중 하나가 발생하며 매수·매도 건수가 포아송 분포를 따른다고 가정한다.

| 파라미터 | 의미 |
|---------|------|
| α | 정보 이벤트 발생 확률 |
| δ | 악재 조건부 확률 |
| μ | 정보 거래자 도착률 |
| ε_b | 비정보 매수자 기본 도착률 |
| ε_s | 비정보 매도자 기본 도착률 |

```
PIN = α·μ / (α·μ + ε_b + ε_s)
```

### APIN — Adjusted PIN (Duarte & Young 2009)

PIN 모델의 확장으로, 유동성 충격·포트폴리오 리밸런싱 등 비정보적 요인에 의한 대칭적 주문 흐름 충격(PSOS)을 별도로 모델링해 순수 정보 거래 비율을 더 정확히 추정한다. 정보 이벤트와 대칭 충격이 독립적으로 발생한다고 가정하여 6가지 시나리오로 확장된다.

| 파라미터 | 의미 |
|---------|------|
| α, δ | 정보 이벤트 확률, 호재 조건부 확률 |
| θ₁, θ₂ | 무정보일 / 정보일의 대칭 충격 확률 |
| μ_b, μ_s | 호재 시 정보 매수 / 악재 시 정보 매도 도착률 |
| ε_b, ε_s | 비정보 매수 / 매도 기본 도착률 |
| Δ_b, Δ_s | 충격 발생 시 추가 매수 / 매도 도착률 |

```
APIN = α·(δ·μ_b + (1-δ)·μ_s) / 분모
PSOS = (Δ_b + Δ_s)·(α·θ₂ + (1-α)·θ₁) / 분모
분모 = 위 두 분자의 합 + ε_b + ε_s
```

### VPIN — Volume-Synchronized PIN (Easley et al. 2012)

거래량 단위로 동기화된 버킷을 구성하고, BVC(Bulk Volume Classification)로 매수·매도 비율을 추정해 계산하는 고빈도 정보 거래 지표.

```
ProbBuy = CDF_t(ΔP / σ_ΔP ;  df=0.25)     # Student-t 분포
V       = ADV / BUCKETS_PER_DAY            # 연도별 동적 버킷 크기
VPIN    = Σ|V_buy - V_sell| / (n × V)     # n = ROLLING_WINDOW 버킷
```

---

## 폴더 구조

### 입력 데이터

```
E:\vpin_project_sas7bdat\          ← SAS 원본 (최초 변환 전)
├── KOR_2010\*.sas7bdat
├── KOR_2011\*.sas7bdat
└── ...

E:\vpin_project_parquet\           ← Parquet 변환 후 (파이프라인 입력)
├── KOR_2010\
│   ├── KOR_201001.parquet
│   ├── KOR_201002.parquet
│   └── ...                        # 월별 틱 데이터 파일
├── KOR_2011\
└── ...
```

**틱 데이터 컬럼**: `Symbol`, `Date` (pl.Date), `Time` (pl.Time), `Price`, `Volume`, `LR` (1=매수 / -1=매도), `MidPoint`, `QSpread`

### 출력 데이터

```
E:\vpin_project_parquet\output_data\
│
├── all_daily_bs.parquet                          ← PIN/APIN Step1: 종목·날짜별 B/S 집계
│
├── pin_daily_rolling_{year}_{RUN_ID}.parquet     ← PIN 최종 결과
├── pin_daily_rolling_{year}_{RUN_ID}_SAMPLE.csv  ← PIN 샘플 (상위 1000행)
│
├── apin_daily_rolling_{year}_{RUN_ID}.parquet    ← APIN 최종 결과
├── apin_daily_rolling_{year}_{RUN_ID}_SAMPLE.csv ← APIN 샘플 (상위 1000행)
│
├── 1m_bars\                                      ← VPIN Step1: 연도별 1분봉
│   ├── 1m_bars_KOR_2010.parquet
│   ├── 1m_bars_KOR_2011.parquet
│   └── ...
│
├── vpin_results\                                 ← VPIN Step2: 종목별 결과
│   ├── sym_005930.parquet
│   ├── sym_000660.parquet
│   └── ...
│
└── intermediate\
    └── session_{RUN_ID}\                         ← 체크포인트 (실행별 격리)
        ├── pin_checkpoint_0000.parquet
        ├── pin_checkpoint_0001.parquet
        ├── apin_checkpoint_0000.parquet
        └── sym_input\                            ← VPIN 종목별 분할 입력
            └── sym_{symbol}.parquet
```

### 출력 스키마

**PIN**: `Symbol, Date, B, S, a, d, u, eb, es, PIN`

**APIN**: `Symbol, Date, B, S, a, d, t1, t2, ub, us, eb, es, pb, ps, APIN, PSOS`

**VPIN**: `Datetime, Symbol, BucketNo, VPIN` (종목별 파일)

추정값이 null인 경우: 해당 날짜에 거래는 있었지만 윈도우 내 유효 거래일 부족 또는 MLE 수렴 실패.

---

## 파이프라인 실행 순서

### Step 0 — SAS → Parquet 변환 (최초 1회만)

```
python 00_sas_to_parquet_개선.py
```

SAS 날짜·시간 숫자를 `pl.Date` / `pl.Time`으로 변환하고 Parquet(Snappy 압축)으로 저장한다. 100만 행 단위 청크 처리로 메모리를 관리한다.

### Step 1+2 — PIN 추정

```
python 01_PIN.py
```

Step 1(일별 B/S 집계) + Step 2(60일 롤링 PIN 추정)가 한 파일에 통합되어 있다. Step 1 결과(`all_daily_bs.parquet`)가 이미 존재하면 자동 스킵된다.

### Step 1+2 — APIN 추정

```
python 02_APIN.py   
```

PIN과 동일한 구조(Step 1 + Step 2)이며, 10개 파라미터 추정과 PSOS 계산이 추가된다.

### Step 1+2 — VPIN 계산

```
python 03_VPIN.py
```

Step 1에서 1분봉을 연도별로 집계하고, Step 2에서 종목별 VPIN을 계산해 파일로 직접 저장한다(메인 프로세스로 DataFrame 반환 없음 → RAM 절약).

---

## 핵심 설정값

스크립트 상단 전역변수에서 모든 옵션을 제어한다.

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `BASE_DIR` | `E:\vpin_project_parquet` | 월별 parquet 루트 경로 |
| `YEAR_FOLDERS` | `None` | 처리 연도 (`None` → KOR_* 전체 자동 탐색, 예: `['KOR_2017']`) |
| `OUTPUT_DIR` | `BASE_DIR\output_data` | 결과 저장 경로 |
| `FORCE_REPROCESS_STEP1` | `False` | `True`이면 Step 1 결과를 강제 재생성 |
| `CHECKPOINT_N` | `100` | N 종목 처리마다 중간 저장 |
| `IMAP_CHUNKSIZE` | `10` | 워커에 한 번에 넘기는 종목 묶음 크기 |
| `WINDOW_SIZE` | `60` | 롤링 윈도우 크기 (영업일) |
| `MIN_VALID_DAYS` | `30` | 윈도우 내 실거래일(B+S>0) 최소 개수 |
| `RESUME_RUN_ID` | `None` | 중단 재개 시 이전 RUN_ID 문자열 지정 |
| `ROLLING_WINDOW` *(VPIN)* | `50` | VPIN 롤링 버킷 수 |
| `BUCKETS_PER_DAY` *(VPIN)* | `50` | 버킷 크기 = ADV / 이 값 |

---

## MLE 추정 방식 (PIN / APIN 공통)

1. **그리드 탐색**: 각 파라미터를 3수준으로 조합한 후보 전체에 대해 NLL을 log-space 행렬 연산으로 일괄 계산
   - PIN: 3^5 = **243**개 후보
   - APIN: 2^10 = **59,049**개 후보
2. **L-BFGS-B 정밀 최적화**: 그리드 탐색에서 선정된 최적 초기값에서 출발해 수렴

**영업일 캘린더 방식**: 전체 종목에서 거래가 있었던 날짜의 합집합을 공통 영업일로 정의하고, 각 종목 시계열을 캘린더에 left join하여 거래 없는 날을 B=S=0으로 채운다. 이를 통해 60행 슬라이딩 윈도우가 항상 정확히 60 영업일을 의미하도록 보장한다.

**수치 안정성**: 포아송 PMF를 log-space에서 계산하고(`k·ln(λ) - λ - gammaln(k+1)`), 혼합 우도 합산에 `scipy.special.logsumexp` 트릭을 적용해 부동소수점 언더플로우를 방지한다.

**병렬 처리**: `multiprocessing.Pool`로 종목 단위 병렬화. `init_worker`로 그리드 행렬과 캘린더를 각 워커에 1회만 전달해 IPC 오버헤드를 최소화한다.

---

## 중단 후 재개

장시간 실행 중 중단된 경우 다음 절차로 이어 실행한다.

```python
# 스크립트 상단에서
RESUME_RUN_ID = "20240115_0930"   # 이전 실행 로그에서 확인한 RUN_ID
```

세션 폴더(`intermediate/session_{RUN_ID}/`)에 저장된 체크포인트를 자동으로 로드하고, 완료된 종목을 건너뛰어 재개한다. 완료 후에는 `RESUME_RUN_ID = None`으로 복원한다.

---

## APIN 버전 이력

| 파일 | 특징 |
|------|------|
| `02_apin_daily_00기본.py` | 기본 구현 (3^10 전체 그리드) |
| `02_apin_daily_01그리드분할.py` | 그리드를 분할 탐색하는 실험 버전 |
| `02_apin_daily_02축소그리드.py` | 고정 축소 그리드로 속도 개선 시도 |
| `02_apin_daily_03동적그리드.py` | 데이터 분포 기반 동적 그리드 생성 |
| `02_apin_daily_04EM엔진.py` | EM 알고리즘과 MLE 방식 비교 |
| `02_apin_daily_05rpy라이브러리.py` | rpy2를 통한 R PINstimation 패키지 연동 |
| `02_apin_daily_06R직접사용.py` | R 스크립트 직접 호출 방식 |

번호가 높을수록 최신 시도. 현재 확정 버전 없음(개발 중).

---

## 의존 패키지

```
polars        # 데이터 처리 (lazy scan, join, group_by)
numpy         # 배열 연산, 브로드캐스팅
scipy         # optimize.minimize (L-BFGS-B), special.logsumexp, gammaln, stats.t
pyarrow       # parquet I/O, SAS 타입 변환
pyreadstat    # SAS7BDAT 파일 읽기
pandas        # SAS 날짜 변환 (00_sas_to_parquet 전용)
tqdm          # 진행 표시
multiprocessing  # 종목 단위 병렬 처리 (표준 라이브러리)
```

---

## 실행 환경

- OS: Windows (경로 구분자 `\`, `multiprocessing.freeze_support()` 적용)
- Python 가상환경: `.venv`

```
.\.venv\Scripts\activate
python 01_PIN.py
```
