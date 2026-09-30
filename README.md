# PIN / APIN / VPIN 계산 파이프라인

주식 틱(체결) 데이터로부터 정보비대칭 지표 세 가지를 계산하는 Python 코드입니다.

| 지표 | 모형 | 계산 단위 |
|---|---|---|
| **PIN** (Probability of Informed Trading) | Easley, Kiefer, O'Hara & Paperman (1996) | 종목 × 주(week) |
| **APIN / PSOS** (Adjusted PIN / Probability of Symmetric Order-flow Shock) | Duarte & Young (2009) | 종목 × 주(week) |
| **VPIN** (Volume-synchronized PIN) | Easley, López de Prado & O'Hara (2012) | 종목 × 거래량 버킷 |

수백 GB 규모의 다년치 틱 데이터를 32GB RAM 장비 한 대에서 처리하는 것을 전제로,
Polars lazy/streaming 처리와 체크포인트 기반 재개 구조로 작성되어 있습니다.
데이터 파일은 저장소에 포함되어 있지 않습니다.

## 저장소 구성

```
├── 00_preprocess/        전처리: SAS7BDAT → Parquet 변환
│   ├── sas_to_parquet.py        ← 최종
│   └── 파켓변환.py              소규모용 간단 변환 스크립트
├── 01_pin_apin/          PIN + APIN 통합 파이프라인
│   └── pin_apin_weekly.py       ← 최종
├── 03_vpin/              VPIN 파이프라인
│   └── vpin_stateful.py         ← 최종
├── 04_utils/             대용량 parquet 분할·검증 보조 도구
└── 99_history/           과거 버전 보관 (참고용, 실행 비권장)
```

실제로 실행하는 파일은 `← 최종` 표시가 있는 세 개입니다.

```
sas_to_parquet.py  →  (필요시 04_utils의 분할/검증 도구)  →  pin_apin_weekly.py / vpin_stateful.py
```

## 실행 환경

- Python 3.9 이상
- 패키지: `polars`, `numpy`, `scipy`, `tqdm`, `pandas`, `pyarrow`, `pyreadstat`

```
pip install polars numpy scipy tqdm pandas pyarrow pyreadstat
```

코드가 `collect(streaming=True)`를 사용하므로 이 인자를 지원하는 Polars 버전이 필요합니다.

## 입력 데이터 형식

세 파이프라인 모두 같은 폴더 구조를 읽습니다.

```
{BASE_DIR}/
├── CN_2019/*.parquet
├── CN_2020/*.parquet
└── CN_2021/*.parquet        ← {COUNTRY}_{YYYY} 패턴의 폴더
```

| 파이프라인 | 필요 컬럼 |
|---|---|
| PIN / APIN | `Symbol`, `Date`, `LR` (매수 체결 = 1, 매도 체결 = -1) |
| VPIN | `Date`, `Time`, `Symbol`, `Price`, `Volume` |

지원 국가 코드: `KOR`, `US`, `JP`, `CA`, `FR`, `GR`, `HK`, `IT`, `UK`, `CN`

## 사용 방법

각 파일 상단의 `사용자 설정` 구역에서 경로와 국가 코드를 수정한 뒤 그대로 실행합니다.
명령행 인자는 없습니다.

```
python 00_preprocess/sas_to_parquet.py     # BASE_INPUT_DIR, BASE_OUTPUT_DIR 설정
python 01_pin_apin/pin_apin_weekly.py      # BASE_DIR, COUNTRY, MODELS_TO_RUN 설정
python 03_vpin/vpin_stateful.py            # BASE_DIR, COUNTRY 설정
```

결과는 `{BASE_DIR}/0416_output/{COUNTRY}/` 아래에 저장됩니다.
모든 단계가 체크포인트를 남기므로 **중단 후 다시 실행하면 완료된 부분은 건너뜁니다.**

## 코드 설명

### 1. `00_preprocess/sas_to_parquet.py` — SAS → Parquet 변환

원본 SAS7BDAT 파일(파일당 최대 수백 GB)을 데이터 손상 없이 Parquet으로 옮깁니다.

- `pyreadstat`으로 `CHUNK_SIZE`(기본 800만 행) 단위로 읽어 PyArrow로 이어 씁니다.
- `.tmp.parquet`에 먼저 쓰고, 검증을 통과했을 때만 최종 파일명으로 바꿉니다. 도중에 오류가 나면 임시 파일을 지웁니다.
- 원본 청크에서 집계한 행 수·결측 수·값 범위·합계를 출력 Parquet과 비교합니다.
- `Date`, `Time`, `LR`은 타입을 명시해 변환하고, 변환할 수 없는 값은 임의 값으로 채우지 않고 null로 둡니다. `Symbol`은 공백만 제거합니다.
- 출력 드라이브 여유 공간이 `MIN_FREE_SPACE_GB`보다 적으면 새 파일 변환을 시작하지 않습니다.
- 긴 읽기 구간에서도 진행 여부를 알 수 있도록 heartbeat 로그를 찍고, 오래 멈추면 스택을 덤프합니다.
- 변환이 끝나면 한글 TXT 리포트를 생성합니다.

### 2. `01_pin_apin/pin_apin_weekly.py` — weekly PIN / APIN

| 단계 | 내용 | 산출물 |
|---|---|---|
| Step 1 | 틱 parquet을 파일 단위로 읽어 종목·일자별 매수 건수 `B`, 매도 건수 `S` 집계 | `pin_apin/daily_counts/*.parquet` |
| Step 2 | 종목별로 주간 모수를 MLE 추정 (멀티프로세싱, 종목 단위 체크포인트) | `pin_apin/{pin,apin}/checkpoints/sym_{Symbol}.parquet` |
| Step 3 | 체크포인트를 병합해 최종 결과 저장 | `{model}_{COUNTRY}_weekly_checkpoint_{실행시각}.parquet`, 샘플 CSV |
| Step 4 | 분포·범위 이탈·오류 종목을 정리한 검증 리포트 | `*_report.txt` |

**추정 방식**

- 추정 window: 대상 주 + 직전 3주(총 4주). window 안의 거래일이 10일을 초과할 때만 추정합니다.
- 일별 `(B, S)`가 포아송 혼합분포를 따른다고 보고 로그우도를 최대화합니다.
- 초기값은 grid search로 고릅니다. 확률 모수는 `{0.1, 0.5, 0.9}`, 도착률 모수는 `{20, 200, 2000}`의 모든 조합(PIN 243개, APIN 59,049개)에서 우도가 가장 높은 점을 택한 뒤 L-BFGS-B로 수렴시킵니다.
- 수렴에 실패한 주는 결과에서 제외됩니다.

**산식**

```
PIN  = a·u / (a·u + eb + es)

denom = a·(d·ub + (1-d)·us) + (pb+ps)·(a·t2 + (1-a)·t1) + eb + es
APIN  = a·(d·ub + (1-d)·us) / denom
PSOS  = (pb+ps)·(a·t2 + (1-a)·t1) / denom
```

- `a`: 정보 사건 발생 확률, `d`: 정보 사건의 방향 확률
- `u`(`ub`, `us`): 정보거래자 도착률, `eb`/`es`: 비정보 매수/매도 도착률
- `t1`/`t2`: 대칭적 주문흐름 충격 발생 확률(정보 없음/있음), `pb`/`ps`: 충격 시 추가 매수/매도 도착률

**출력 컬럼**

- PIN: `Symbol, Date, B, S, Week, a, d, u, eb, es, PIN`
- APIN: `Symbol, Date, B, S, Week, APIN, PSOS, a, d, t1, t2, ub, us, eb, es, pb, ps`

결과는 일별 행으로 저장되며, 같은 주에 속한 날짜는 같은 주간 추정치를 갖습니다.

**주요 설정**: `MODELS_TO_RUN`(PIN/APIN 선택), `INPUT_FILES`(특정 파일만 처리),
`NUM_WORKERS`, `FORCE_REPROCESS_DAILY_CACHE`, `FORCE_RECOMPUTE_MODEL`

### 3. `03_vpin/vpin_stateful.py` — stateful VPIN

| 단계 | 내용 | 산출물 |
|---|---|---|
| Step 1 | 틱을 종목별 1분봉(마지막 체결가, 거래량 합)으로 변환 | `vpin/monthly_1m_bars/*.parquet` |
| Step 2 | 종목별로 1분봉을 기간 단위로 읽어 VPIN 계산 | `vpin/checkpoints/sym_{Symbol}.parquet`, `vpin/states/sym_{Symbol}.pkl` |
| Step 3 | 체크포인트를 병합해 최종 결과 저장 | `vpin_{COUNTRY}_stateful_{실행시각}.parquet`, 샘플 CSV |
| Step 4 | 분포·범위 이탈·버킷 번호 연속성을 점검한 검증 리포트 | `*_report.txt` |

**계산 방식**

1. 종목 전체 1분봉에서 가격 변화의 표준편차 `sigma_p`와 버킷 크기 `bucket_size_v = 일평균 거래량 / BUCKETS_PER_DAY`를 먼저 구합니다.
2. BVC(Bulk Volume Classification): 각 1분봉의 매수 비중을 `prob_buy = t.cdf(Δp / sigma_p, df=T_DIST_DF)`로 추정합니다.
3. 거래량을 `bucket_size_v` 크기의 버킷에 순서대로 채웁니다. 한 봉의 거래량이 버킷 경계를 넘으면 나누어 담습니다.
4. 버킷이 찰 때마다 주문 불균형 `oi = |buy − sell|`을 기록합니다.
5. `VPIN = 최근 ROLLING_WINDOW개 버킷의 oi 합 / (ROLLING_WINDOW × bucket_size_v)`

기본값은 `ROLLING_WINDOW = 50`, `BUCKETS_PER_DAY = 50`, `T_DIST_DF = 0.25`입니다.
처음 49개 버킷은 window가 채워지지 않아 VPIN이 NaN입니다.

**stateful 처리**

메모리를 아끼기 위해 종목별 1분봉을 `VPIN_CHUNK_YEARS`(기본 2년) 단위로 끊어 읽습니다.
채우던 버킷의 매수/매도량, 최근 `oi` 목록, 직전 가격, 버킷 번호를 `VpinState`에 담아 다음 구간으로
넘기므로, 전체 기간을 한 번에 계산한 것과 같은 결과가 나옵니다.

**출력 컬럼**: `Datetime`(버킷이 완성된 1분봉 시각), `Symbol`, `BucketNo`, `VPIN`

### 4. `04_utils/` — 보조 도구

| 파일 | 역할 |
|---|---|
| `0510_파일자르기01.py` | 대용량 parquet 2분할 (`_part1` / `_part2` 출력) |
| `0510_파일자르기02.py` | 폴더 단위 일괄 분할 (`_[1-2]` 형식으로 출력) |
| `0512_분할파일명_대괄호제거.py` | `_[1-2]` → `_1of2` 일괄 rename |
| `0511_빠진년도확인.py` | 월 커버리지 검사 (누락/중복 월 탐지) |
| `0512_tickcheck.py` | 분할 parquet 무결성 검증 (VPIN 입력 안전성 점검) |
| `파켓직접확인.py` | parquet 내용 간단 확인 |

### 5. `99_history/` — 과거 버전

현재 코드에 이르기까지의 시도를 보관한 폴더입니다. 알려진 버그나 성능 문제로 대체된 버전이므로 실행은 권장하지 않습니다.

| 폴더 | 내용 |
|---|---|
| `2512_주별월별_sas변환/` | 원본 SAS 코드(`sas_code/`)와 이를 Python으로 옮긴 코드(`python_code/`). 현재 PIN/APIN 계산 로직의 원본 |
| `2602_일별변환_최적화과정/` | 일별 계산 전환과 성능 최적화 반복(numba, logpmf, 그리드 분할, EM, R 연동). 정확성 부족으로 폐기한 순수 Python 모듈판과 OOM으로 폐기한 순수 R 구현 포함 |
| `2603_PIN_cal_R기반/` | Python 전처리 + R 계산 방식. `vpin_계산오류검정시도-실패/`에 VPIN 버그 분석 기록(`VPIN_BUG_ANALYSIS.md`) |
| `2604_suji_중간버전/` | 최종 직전의 VPIN·전처리 중간 버전 |

버전 계보:

- **PIN**: `2603 pin_r_skip.R` → `2512 01_pin_weekly` → `2602 01_pin_daily_00~09` → `pin_apin_weekly.py`
- **APIN**: `2603 apin_r 계열` → `2512 02_apin_weekly` → `2602 02_apin_daily_00~06` → `pin_apin_weekly.py`
- **VPIN**: `2603 vpin_r_light.R` → `2512 03_vpin_Base` → `2602 03_vpin_base_01~03` → `2604 vpin_01~04` → `vpin_stateful.py`
- **전처리**: `2603 00_sas_to_parquet` → `2602 00_sas_to_parquet_개선` → `2604 sas_to_parquet_00, 02` → `sas_to_parquet.py`

## 주의사항

- `0510_파일자르기02.py`는 출력 파일명에 대괄호(`_[1-2]`)를 씁니다. Polars `scan_parquet`은 대괄호를 glob 패턴으로
  해석해 읽기에 실패하므로, 분할 후 `0512_분할파일명_대괄호제거.py`로 이름을 바꾸고 `0512_tickcheck.py`로 검증해야 합니다.
- APIN은 grid가 59,049개라 워커당 메모리 사용량이 큽니다. 32GB RAM에서는 `NUM_WORKERS`를 낮게 잡으세요.
  VPIN은 `NUM_WORKERS = 1~2`를 권장합니다.
- `FORCE_*` 옵션을 `True`로 두면 해당 캐시·체크포인트 폴더를 삭제하고 처음부터 다시 계산합니다.

## 참고문헌

- Easley, D., Kiefer, N. M., O'Hara, M., & Paperman, J. B. (1996). Liquidity, information, and infrequently traded stocks. *Journal of Finance*, 51(4), 1405–1436.
- Duarte, J., & Young, L. (2009). Why is PIN priced? *Journal of Financial Economics*, 91(2), 119–138.
- Easley, D., López de Prado, M. M., & O'Hara, M. (2012). Flow toxicity and liquidity in a high-frequency world. *Review of Financial Studies*, 25(5), 1457–1493.
