# PINstimation 파이프라인 — 아키텍처 & 실행 가이드

---

## 1. 프로젝트 개요

R 패키지 `PINstimation`을 사용하여 여러 나라의 주식 종목별로 **PIN**, **Adjusted PIN (AdjPIN)**, **VPIN**을 계산하는 병렬 배치 파이프라인이다.

| 지표 | 모델 | 추정 방식 |
|------|------|-----------|
| **PIN** | Easley et al. (1996) | `pin_ea()` — 60 캘린더일 롤링 윈도우 |
| **AdjPIN** | Duarte & Young (2009) | `adjpin()` — 60 캘린더일 롤링 윈도우 |
| **VPIN** | Easley, de Prado & O'Hara (2012) | LR 직접 활용 수동 계산 — 전체 기간 단일 계산 |

---

## 2. 디렉토리 구조

```
project_root/
├── data/
│   └── raw/
│       ├── KR/                  ← 나라별 원본 Parquet 틱 데이터
│       ├── US/
│       └── JP/
├── output/
│   ├── KR/
│   │   ├── pin/
│   │   │   ├── checkpoints/     ← 종목별 중간저장 (sym_{Symbol}.csv)
│   │   │   ├── pin_KR_{timestamp}.csv      ← 최종 통합 결과
│   │   │   └── pin_KR_{timestamp}.parquet
│   │   ├── adjpin/
│   │   │   ├── checkpoints/
│   │   │   ├── adjpin_KR_{timestamp}.csv
│   │   │   └── adjpin_KR_{timestamp}.parquet
│   │   └── vpin/
│   │       ├── checkpoints/
│   │       ├── vpin_KR_{timestamp}.csv
│   │       └── vpin_KR_{timestamp}.parquet
│   ├── US/
│   └── JP/
├── R/
│   ├── 00_setup.R               ← 패키지 로드, 경로/파라미터, 체크포인트 유틸리티
│   ├── 01_data_prep.R           ← 데이터 전처리 함수
│   ├── 02_pin_worker.R          ← PIN 추정 워커
│   ├── 03_adjpin_worker.R       ← AdjPIN 추정 워커
│   ├── 04_vpin_worker.R         ← VPIN 계산 워커
│   └── main.R                   ← 메인 오케스트레이터
└── logs/
    └── {country}_run_{timestamp}.log
```

---

## 3. 실행 흐름

```
main.R
  │
  ├─ source("00_setup.R")          패키지, 경로, 파라미터 설정
  ├─ source("01_data_prep.R")      전처리 함수 정의
  ├─ source("02_pin_worker.R")     PIN 워커 정의
  ├─ source("03_adjpin_worker.R")  AdjPIN 워커 정의
  └─ source("04_vpin_worker.R")    VPIN 워커 정의
       │
       └─ for (country in COUNTRIES) {
            │
            │  [데이터 로드]
            │  load_raw_data(country)
            │    └─ arrow::open_dataset()로 parquet 파일 일괄 로드
            │
            │  [PIN 추정]
            │  prep_daily_bs() → run_pin_all()
            │    ├─ 체크포인트 확인: 완료 종목 자동 스킵
            │    ├─ 배치별 병렬 실행 (100종목/배치)
            │    ├─ 종목 완료 즉시 checkpoints/sym_{Symbol}.csv 저장
            │    ├─ 배치마다 진행상황 + ETA 출력
            │    └─ 전체 완료 후 최종 병합 (CSV + Parquet)
            │
            │  [AdjPIN 추정]
            │  (daily_bs 재사용) → run_adjpin_all()
            │    └─ 동일한 체크포인트/진행상황/병합 구조
            │
            │  [VPIN 계산]
            │  prep_vpin_ticks() → run_vpin_all()
            │    └─ 동일한 체크포인트/진행상황/병합 구조
            │
            └─ gc()  메모리 해제
          }
```

---

## 4. 원본 데이터 스펙

### 입력 컬럼

| 컬럼명 | 타입 | 설명 |
|--------|------|------|
| Price | float64 | 거래 가격 |
| Volume | float64 | 거래량 |
| Symbol | str | 종목 코드 |
| Date | date32 | 거래 날짜 |
| Time | time64 | 거래 시각 |
| LR | int8 | LR 알고리즘 분류: 1=매수, -1=매도, 0=미분류 |

### 저장 위치

```
data/raw/{country}/
  ├── {country}_2019/
  │   ├── data_01.parquet
  │   └── data_02.parquet
  └── {country}_2020/
      └── ...
```

---

## 5. PIN 추정 로직 (`02_pin_worker.R`)

### 5-1. 전처리

```
원본 틱 → filter(LR != 0)
  → group_by(Symbol, Date)
  → summarise(buys = count(LR==1), sells = count(LR==-1))
```

- `aggregate_trades()`는 사용하지 않음 (LR 분류 완료)
- `buys`/`sells` 집계 기준: **건수(count)**

### 5-2. 롤링 윈도우

| 항목 | 값 |
|------|-----|
| 윈도우 크기 | 60 캘린더일 (슬라이딩) |
| 스텝 | 1 영업일씩 이동 |
| 건너뛰기 조건 | 윈도우 내 유효 영업일 ≤ 30일 |

- **유효 영업일**: `buys + sells > 0` 인 날

### 5-3. 추정 함수

```r
PINstimation::pin_ea(data = window_data, verbose = FALSE)
```

- `pin_ea()`는 Ersan & Alici (2016) 초기 파라미터 알고리즘
- `pin()` 함수의 `initialsets`는 numeric이므로 직접 사용하지 않음
- EA 알고리즘을 쓰려면 반드시 `pin_ea()` 전용 함수를 호출해야 함

### 5-4. 출력 컬럼

| 컬럼명 | 설명 |
|--------|------|
| symbol | 종목 코드 |
| window_end | 롤링윈도우 마지막 날짜 |
| pin | 추정된 PIN 값 |
| alpha | 정보거래일 발생 확률 |
| delta | 나쁜 뉴스 비율 |
| mu | 정보거래자 주문 강도 |
| eb | 정상 매수 강도 |
| es | 정상 매도 강도 |
| loglik | 로그 우도 |
| valid_days | 유효 영업일 수 |
| created_at | 파일 생성 타임스탬프 |

### 5-5. 출력 파일

```
output/{country}/pin/{Symbol}_pin_{YYYYMMDD_HHMMSS}.csv
```

---

## 6. AdjPIN 추정 로직 (`03_adjpin_worker.R`)

### 6-1. 전처리

PIN과 **동일** — 같은 `daily_bs` 데이터프레임을 재사용

### 6-2. 롤링 윈도우

PIN과 **동일** (60 캘린더일, 유효 영업일 ≤ 30일이면 건너뜀)

### 6-3. 추정 함수

```r
PINstimation::adjpin(
  data        = window_data,
  method      = "ML",
  initialsets = "GE",
  verbose     = FALSE
)
```

- `method = "ML"`: Maximum Likelihood
- `initialsets = "GE"`: Gan-Wei-Johnstone 초기점 생성

### 6-4. 출력 컬럼

| 컬럼명 | 설명 |
|--------|------|
| symbol | 종목 코드 |
| window_end | 롤링윈도우 마지막 날짜 |
| adjpin | Adjusted PIN 값 |
| psos | 유동성 쇼크 확률 |
| alpha | 정보거래일 발생 확률 |
| delta | 나쁜 뉴스 비율 |
| loglik | 로그 우도 |
| aic | AIC |
| bic | BIC |
| valid_days | 유효 영업일 수 |
| created_at | 파일 생성 타임스탬프 |

---

## 7. VPIN 계산 로직 (`04_vpin_worker.R`)

### 7-1. 전처리

```
원본 틱 → filter(LR != 0)
  → Date + Time → timestamp(POSIXct)
  → Volume > 0 필터
```

### 7-2. 수동 계산 로직 (LR 직접 활용)

PINstimation::vpin() 내부의 BVC 재분류를 거치지 않고 LR 컬럼을 직접 사용한다.

```
1) ADV = 일평균 총거래량

2) VBS = ADV / samplength(50)
   → Volume Bucket Size: 하루 거래량을 50등분

3) 틱을 VBS 단위 버킷에 할당
   → cumsum(Volume) / VBS → ceiling → bucket_id

4) 버킷별 집계:
   buy_vol  = sum(Volume[LR == 1])
   sell_vol = sum(Volume[LR == -1])

5) OI = |buy_vol - sell_vol|
   → Order Imbalance: 매수-매도 거래량 불균형

6) VPIN = rolling_mean(OI, window=50) / VBS
   → zoo::rollapply()로 50버킷 롤링 평균 계산
```

### 7-3. 출력 컬럼

| 컬럼명 | 설명 |
|--------|------|
| symbol | 종목 코드 |
| bucket | 버킷 인덱스 |
| bucket_date | 버킷 마지막 거래 날짜 |
| buy_vol | 버킷 내 매수 거래량 |
| sell_vol | 버킷 내 매도 거래량 |
| oi | Order Imbalance |
| vpin | VPIN 값 |
| vbs | Volume Bucket Size |
| created_at | 파일 생성 타임스탬프 |

---

## 8. 병렬 처리 설계

### 동시성 전략

```
future::plan(multisession, workers = N_CORES)
  + furrr::future_map(symbols, worker_fn)
```

- **코어 수**: `parallel::detectCores(logical = TRUE) - 1`
- 각 워커 = 종목 1개를 독립적으로 처리
- 워커 간 공유 상태 없음

### 레이스 컨디션 방지

1. 각 워커는 **자신이 맡은 종목 1개에 대한 파일만** 생성
2. 공유 파일·공유 객체에 동시 접근 없음
3. 출력 파일명에 Symbol 포함 → 파일 단위 격리

```
{Symbol}_pin_{YYYYMMDD_HHMMSS}.csv      ← 종목별 독립 파일
{Symbol}_adjpin_{YYYYMMDD_HHMMSS}.csv
{Symbol}_vpin_{YYYYMMDD_HHMMSS}.csv
```

### 오류 처리

- 모든 워커 함수는 `tryCatch`로 래핑
- 개별 종목 실패가 전체 파이프라인을 중단시키지 않음
- 실패 정보는 로그 파일에 기록

---

## 9. 체크포인트 & 재개 시스템

### 동작 원리

```
종목 1개 계산 완료
  → checkpoints/sym_{Symbol}.csv 즉시 저장 (빈 결과도 저장)

재실행 시:
  → checkpoints/ 스캔 → 완료 종목 목록 추출
  → remaining = 전체 종목 - 완료 종목
  → remaining만 처리 (이어서 진행)

전체 완료 후:
  → checkpoints/*.csv 전부 병합 → 최종 결과 파일 생성
```

### 체크포인트 파일 구조

```
output/{country}/pin/checkpoints/
  ├── sym_005930.csv     ← 삼성전자 PIN 결과 (완료)
  ├── sym_000660.csv     ← SK하이닉스 PIN 결과 (완료)
  └── sym_035720.csv     ← 카카오 PIN 결과 (완료)
```

### 재실행이 안전한 이유

1. 종목별 고유 파일명 → 덮어쓰기 충돌 없음
2. 빈 결과도 체크포인트 생성 → 실패 종목 재시도 방지
3. 병합은 항상 checkpoints/ 전체를 다시 읽음 → 일관성 보장

### 진행상황 확인 방법

**방법 1: 콘솔 출력 (자동)**

```
[2026-03-15 14:30:22] [PIN] 배치 3/10 | 누적 300/1000종목 | 성공 98/100 | 5.2분 경과 | ETA 12분
```

**방법 2: 체크포인트 파일 수 확인 (별도 터미널)**

```bash
# macOS/Linux
ls output/KR/pin/checkpoints/sym_*.csv | wc -l

# PowerShell
(Get-ChildItem output\KR\pin\checkpoints\sym_*.csv).Count
```

**방법 3: 로그 파일 tail**

```bash
tail -f logs/KR_run_*.log
```

### 최종 결과 파일

모든 종목 완료 후 자동 병합:

```
output/{country}/pin/
  ├── checkpoints/              ← 종목별 중간 결과
  ├── pin_KR_20260315_143022.csv       ← 전체 통합 CSV
  └── pin_KR_20260315_143022.parquet   ← 전체 통합 Parquet
```

---

## 10. 파라미터 요약

| 파라미터 | 기본값 | 위치 | 설명 |
|----------|--------|------|------|
| `COUNTRIES` | `c("KR")` | 00_setup.R | 처리할 나라 목록 |
| `WINDOW_CALENDAR_DAYS` | 60 | 00_setup.R | 롤링 윈도우 (캘린더일) |
| `MIN_VALID_DAYS` | 30 | 00_setup.R | 최소 유효 영업일 |
| `VPIN_SAMPLENGTH` | 50 | 00_setup.R | VPIN 롤링 버킷 수 |
| `N_CORES` | 전체-1 | 00_setup.R | 병렬 워커 수 |

---

## 11. 사용 패키지

| 패키지 | 용도 |
|--------|------|
| PINstimation | PIN / AdjPIN 추정 핵심 패키지 |
| dplyr | 데이터 전처리 및 집계 |
| lubridate | 날짜/시간 파싱 |
| future | 병렬 처리 백엔드 |
| furrr | future 기반 map 함수 |
| zoo | rollapply (VPIN 롤링 평균) |
| readr | CSV 입출력 |
| arrow | Parquet 입력 |

설치:
```r
install.packages(c("PINstimation", "dplyr", "lubridate",
                   "future", "furrr", "zoo", "readr", "arrow", "here"))
```

---

## 12. 실행 방법

### 사전 준비

1. `data/raw/{country}/` 에 Parquet 파일 배치
2. `R/00_setup.R`에서 `COUNTRIES`, 경로 등 설정 수정

### 실행

```bash
Rscript R/main.R
```

### 출력 확인

```
output/{country}/pin/    → 종목별 PIN CSV
output/{country}/adjpin/ → 종목별 AdjPIN CSV
output/{country}/vpin/   → 종목별 VPIN CSV
logs/                    → 실행 로그
```

---

## 13. 핵심 제약 및 주의사항

1. `pin_ea()`와 `adjpin()`는 **단일 종목, 단일 윈도우** 데이터만 입력으로 받는다.
2. `buys`/`sells` 집계 기준은 **건수(count)**.
3. `LR == 0` (미분류)은 PIN/AdjPIN/VPIN 모두에서 제외.
4. 타임스탬프는 **파일 저장 직전**에 캡처.
5. 출력 디렉토리가 없으면 자동 생성.
6. 병렬 처리 중 한 종목 실패가 다른 종목에 영향을 주지 않음.
7. `pin()` 함수에 `initialsets = "EA"` 같은 문자열을 넘기면 안 됨 → 반드시 `pin_ea()` 전용 함수 사용.

---

*문서 생성일: 2026-03-15*
