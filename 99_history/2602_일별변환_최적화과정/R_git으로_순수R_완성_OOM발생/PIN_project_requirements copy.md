# PINstimation 계산 파이프라인 — 요구사항 명세서

---

## 1. 프로젝트 개요

R 패키지 `PINstimation` (https://github.com/monty-se/PINstimation) 을 사용하여
여러 나라의 주식 종목별로 **PIN**, **Adjusted PIN (AdjPIN)**, **VPIN** 을 계산하는
병렬 배치 파이프라인을 구축한다.

틱 데이터에 이미 LR(Lee-Ready) 알고리즘으로 분류된 매수/매도 컬럼이 존재하므로, trade classification 단계를 건너뛰고 바로 집계 후 추정한다.
---

## 2. 원본 데이터 스펙

### 2-1. 컬럼 정보

| 컬럼명   | 타입      | 설명                                     |
|----------|-----------|------------------------------------------|
| Price    | float64   | 거래 가격                                |
| Volume   | float64   | 거래량                                   |
| Symbol   | str       | 종목 코드                                |
| Date     | date32    | 거래 날짜                                |
| Time     | time64    | 거래 시각                                |
| LR       | int8      | LR 알고리즘 분류 결과 (1=매수, -1=매도, 0=미분류) |

### 2-2. 입출력 구조

<입력>
  루트: E:\vpin_project_parquet\input_data
  하위: 국가코드로 시작하는 폴더들이 다수 존재
  
  예시:
    E:\vpin_project_parquet\input_data\
    ├── KOR_2019\
    │   ├── kor_2019_01.parquet
    │   ├── kor_2019_02.parquet
    │   └── ...
    ├── KOR_2020\
    │   ├── kor_2020_01.parquet
    │   └── ...
    └── USA_2021\
        └── ...

  - 폴더명: {국가코드}_{연도} 또는 유사 형태 (국가코드로 시작)
  - 파일명: {국가코드}_{연도}_{청크번호}.parquet
    (한 연도의 데이터를 n개로 분할한 파일)
  - 하나의 국가를 처리할 때, 해당 국가코드로 시작하는 모든 폴더 내의
    .parquet 파일을 전부 읽어야 한다.

<출력>
루트: E:\vpin_project_parquet\output
├── output/
│   ├── KR/              ← 나라별 결과 출력
│   │   ├── pin/         ← 종목별 PIN 결과 파일
│   │   ├── adjpin/      ← 종목별 AdjPIN 결과 파일
│   │   └── vpin/        ← 종목별 VPIN 결과 파일
│   ├── US/
│   └── JP/


## 3. 파일 및 디렉토리 구조

```
project_root/

├── R/
│   ├── 00_setup.R
│   ├── 01_data_prep.R
│   ├── 02_pin_worker.R
│   ├── 03_adjpin_worker.R
│   ├── 04_vpin_worker.R
│   └── main.R
└── logs/
    └── {country}_run_{timestamp}.log
```

---

## 4. 공통 처리 규칙

### 4-1. 종목 큐 및 병렬 처리

- 처리 대상 종목 전체를 **정렬된 큐(ordered queue)** 로 만든다 (Symbol 알파벳순 정렬)
- 사용 코어 수: `parallel::detectCores(logical = TRUE) - 1`
- 각 워커 코어는 큐에서 종목을 **하나씩** 꺼내 처리한다
- 구현 방식: R의 `parallel::mclapply` 또는 `future` + `furrr` 패키지 사용
  - 권장: `future::plan(multisession, workers = n_cores)` + `furrr::future_map`

### 4-2. 레이스 컨디션 방지

- 각 워커는 **자신이 맡은 종목 하나에 대한 파일만** 독립적으로 생성한다
- 공유 파일·공유 객체에 동시 접근하지 않는다
- 출력 파일명에 Symbol을 포함시켜 파일 단위로 격리한다
- 예시 출력 파일명: `{Symbol}_pin_{created_at}.csv`

### 4-3. 출력 파일 버전 관리

- 모든 출력 파일에 **생성 타임스탬프**를 포함한다
- 타임스탬프 형식: `YYYYMMDD_HHMMSS` (예: `20250315_143022`)
- 파일명 예시:
  - `005930_pin_20250315_143022.csv`
  - `005930_adjpin_20250315_143022.csv`
  - `005930_vpin_20250315_143022.csv`
- 각 파일 내부에도 `created_at` 컬럼(또는 메타데이터 헤더 주석)으로 타임스탬프를 기록한다

### 4-4. 오류 처리

- 종목별 계산은 `tryCatch`로 감싸 개별 실패가 전체 파이프라인을 중단시키지 않도록 한다
- 실패한 종목은 로그 파일에 Symbol, 오류 메시지, 타임스탬프를 기록하고 건너뜬다

---

## 5. PIN 계산 (`02_pin_worker.R`)

### 5-1. 전처리

```
원본 틱 데이터
→ filter(LR != 0)             # 미분류 제거
→ group_by(Symbol, Date)
→ summarise(
    buys  = sum(LR ==  1),
    sells = sum(LR == -1)
  )
→ rename(date = Date)
→ PINstimation::pin_ea() 입력 형식 완성
```

- `aggregate_trades()` 는 사용하지 않는다 (LR 분류가 이미 완료되어 있으므로)

### 5-2. 롤링 윈도우 설정

| 항목               | 값                                          |
|--------------------|---------------------------------------------|
| 윈도우 크기        | 60 캘린더일 (슬라이딩)                       |
| 스텝               | 1 영업일씩 이동                             |
| 건너뛰기 조건      | 윈도우 내 **유효 영업일 ≤ 30일** 인 경우    |

- **유효 영업일 정의**: `buys + sells > 0` 인 날 (즉, buys와 sells가 모두 0이 아닌 날)
- 반대로 `buys == 0 AND sells == 0` 인 날 수가 **31일 이상**이면 해당 윈도우는 건너뛴다
- 건너뛴 윈도우는 결과 파일에 `NA` 로 기록하거나 행 자체를 생략한다 (일관성 있게 선택)

### 5-3. 추정 함수

```r
PINstimation::pin_ea(data = window_data, verbose = FALSE)
```

- `window_data`: `date | buys | sells` 형식의 data.frame

### 5-4. 출력 컬럼

| 컬럼명       | 설명                            |
|--------------|---------------------------------|
| symbol       | 종목 코드                       |
| window_end   | 해당 롤링윈도우의 마지막 날짜   |
| pin          | 추정된 PIN 값                   |
| alpha        | 정보거래일 발생 확률            |
| delta        | 나쁜 뉴스 비율                  |
| mu           | 정보거래자 주문 강도             |
| eb           | 정상 매수 강도                  |
| es           | 정상 매도 강도                  |
| loglik       | 로그 우도                       |
| valid_days   | 유효 영업일 수                  |
| created_at   | 파일 생성 타임스탬프            |

---

## 6. AdjPIN 계산 (`03_adjpin_worker.R`)

### 6-1. 전처리

PIN과 동일한 전처리 과정 적용 (LR → 일별 buys/sells 집계)

### 6-2. 롤링 윈도우 설정

PIN과 **동일한 조건** 적용 (60 캘린더일, 유효 영업일 ≤ 30일이면 건너뜀)

### 6-3. 추정 함수

```r
PINstimation::adjpin(
  data        = window_data,
  method      = "ML",
  initialsets = "GE",
  verbose     = FALSE
)
```

### 6-4. 출력 컬럼

| 컬럼명       | 설명                                        |
|--------------|---------------------------------------------|
| symbol       | 종목 코드                                   |
| window_end   | 해당 롤링윈도우의 마지막 날짜               |
| adjpin       | 추정된 AdjPIN 값                            |
| psos         | 유동성 쇼크 확률 (AdjPIN 고유 파라미터)     |
| alpha        | 정보거래일 발생 확률                        |
| delta        | 나쁜 뉴스 비율                              |
| loglik       | 로그 우도                                   |
| aic          | AIC                                         |
| bic          | BIC                                         |
| valid_days   | 유효 영업일 수                              |
| created_at   | 파일 생성 타임스탬프                        |

---

## 7. VPIN 계산 (`04_vpin_worker.R`)

### 7-1. 전처리

- 원본 틱 데이터에서 `Price`, `Volume`, `Date`, `Time` 을 사용
- `Date + Time` 을 합쳐 `timestamp (POSIXct)` 컬럼 생성
- **롤링 윈도우 없음** — 종목의 전체 기간에 대해 단일 계산

### 7-2. 추정 방법 — LR 컬럼 직접 활용 (수동 계산)

LR 분류가 원본 데이터에 이미 포함되어 있으므로 `PINstimation::vpin()` 내부의
재분류 과정을 거치지 않고 직접 계산한다.

- 타임바 단위로 `buy_vol = sum(Volume[LR==1])`, `sell_vol = sum(Volume[LR==-1])` 집계
- `OI = |buy_vol - sell_vol|`
- `VBS = 전체기간_일평균거래량 / samplength(=50)`
- `VPIN_t = rolling_mean(OI, window=50) / VBS`
- 구현에 `zoo::rollapply` 사용

### 7-3. 출력 컬럼

| 컬럼명       | 설명                               |
|--------------|------------------------------------|
| symbol       | 종목 코드                          |
| bucket       | 버킷 인덱스                        |
| bucket_date  | 버킷의 마지막 거래 날짜            |
| buy_vol      | 버킷 내 매수 거래량                |
| sell_vol     | 버킷 내 매도 거래량                |
| oi           | Order Imbalance (`|buy - sell|`)   |
| vpin         | VPIN 값                            |
| vbs          | Volume Bucket Size                 |
| created_at   | 파일 생성 타임스탬프               |

---

## 8. 메인 실행 스크립트 (`main.R`)

### 8-1. 실행 흐름

```
1. 00_setup.R       — 패키지 로드, 경로 설정, 코어 수 확인
2. 처리 대상 나라 목록 순회
   └─ 해당 나라 데이터 로드
   └─ 종목 목록 추출 → 알파벳순 정렬 → 큐 구성
   └─ 병렬 워커 실행
       ├─ 02_pin_worker.R    (종목별 독립 파일 생성)
       ├─ 03_adjpin_worker.R (종목별 독립 파일 생성)
       └─ 04_vpin_worker.R   (종목별 독립 파일 생성)
3. 실행 완료 로그 기록
```

### 8-2. 나라 설정 예시

```r
countries <- c("KR", "US", "JP")   # 처리할 나라 목록
# 입력:  data/raw/{country}/
# 출력:  output/{country}/pin/
#        output/{country}/adjpin/
#        output/{country}/vpin/
```

---

## 9. 사용 패키지 목록

| 패키지         | 용도                                      |
|----------------|-------------------------------------------|
| PINstimation   | PIN / AdjPIN / VPIN 추정 핵심 패키지      |
| dplyr          | 데이터 전처리 및 집계                     |
| lubridate      | 날짜/시간 파싱                            |
| future         | 병렬 처리 백엔드                          |
| furrr          | future 기반 map 함수                      |
| zoo            | rollapply (VPIN 직접 계산 시)             |
| here           | 프로젝트 상대경로 관리                    |
| readr          | CSV 입출력                                |

---

## 10. 제약 조건 및 주의사항

1. `pin_ea()` 와 `adjpin()` 은 **단일 종목, 단일 윈도우** 데이터만 입력으로 받는다.
   반드시 종목별·윈도우별로 분리하여 호출할 것.

2. `buys` / `sells` 집계 기준은 **건수(count)** 로 한다.
   볼륨 기준으로 변경 시 명시적으로 선택하고 주석으로 표기할 것.

3. `LR == 0` (미분류)은 PIN/AdjPIN/VPIN 계산 모두에서 제외한다.

4. 타임스탬프 생성 시각은 **파일 저장 직전** 에 캡처한다.
   (`created_at <- format(Sys.time(), "%Y%m%d_%H%M%S")`)

5. 출력 디렉토리가 없으면 자동 생성한다. (`dir.create(..., recursive = TRUE)`)

6. 병렬 처리 중 한 종목 실패가 다른 종목에 영향을 주지 않도록
   모든 워커 함수는 `tryCatch` 로 감싼다.

---

*문서 작성일: 2026-03-15*
