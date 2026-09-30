# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 프로젝트 개요

한국 주식시장 틱 데이터에서 정보거래확률(PIN / APIN)을 일별·롤링으로 계산하는 연구용 파이프라인.

- **PIN**: EKOP(1996) 모델 — 5개 파라미터(α, δ, μ, ε_b, ε_s)
- **APIN**: Duarte & Young(2009) 모델 — 10개 파라미터(대칭 주문충격 분리)

## 실행 환경

가상환경 활성화 (Windows):
```
.\.venv\Scripts\activate
```

스크립트는 모두 `python <파일명>` 으로 직접 실행한다. 별도 빌드·테스트 명령 없음.

## 파이프라인 구조

### 단계 0 — SAS → Parquet 변환 (최초 1회)
```
00_sas_to_parquet_개선.py
```
`E:\vpin_project_sas7bdat\KOR_{year}\*.sas7bdat` → `E:\vpin_project_parquet\KOR_{year}\*.parquet`

SAS 날짜(1960-01-01 기준 경과 일수)·시간(자정 이후 경과 초)을 PyArrow를 통해 `date32` / `time64('ns')`로 변환.

### 단계 1 — 틱 데이터 → 일별 B/S 집계 (Step 1)
월별 parquet를 Polars `scan_parquet`(lazy)로 순회 → 종목·날짜 단위 매수(B)/매도(S) 건수 집계 → `all_daily_bs.parquet` 한 파일로 저장.

`FORCE_REPROCESS_STEP1 = False` 이면 기존 파일 재사용(스킵).

### 단계 2 — 60일 롤링 PIN/APIN 추정 (Step 2)
1. **영업일 캘린더 구축**: 전체 종목에서 B>0 or S>0인 날짜의 합집합
2. **종목 정렬**: 각 종목의 B/S를 캘린더에 left join → 거래 없는 날 B=S=0
3. **슬라이딩 윈도우**: 60행(= 60 영업일) 단위, 유효 거래일(B+S>0) ≥ MIN_VALID_DAYS(30)인 경우만 추정
4. **MLE 2단계**:
   - 그리드 탐색: 3^5 = 243개 후보 → NLL 최소 초기값 선택
   - L-BFGS-B 정밀 최적화
5. **병렬화**: `multiprocessing.Pool` + `init_worker`(grid·calendar 워커당 1회 직렬화) + `imap_unordered`
6. **체크포인트**: `intermediate/session_<RUN_ID>/pin_checkpoint_NNNN.parquet`에 N종목마다 저장

## 주요 파일

| 파일 | 역할 |
|------|------|
| `00_sas_to_parquet_개선.py` | SAS7BDAT → Parquet 변환 (최초 1회) |
| `01_PIN.py` | **현재 확정 버전** 통합 PIN 파이프라인 (Step1 + Step2) |
| `02_apin_daily_00원본.py` | APIN 파이프라인 원본 |
| `02_apin_daily_01그리드분할.py` | APIN 그리드 분할 시도 버전 |
| `pin_daily_세부설명/` | PIN 파이프라인을 단계별로 분리한 모듈화 버전 |
| `01_pin_daily_09거래량0을null.py` | `01_PIN.py`의 직전 개발 단계 버전 |

번호가 높을수록 최신 버전. 접두어 `01_pin_daily_*`는 PIN 개발 이력, `02_apin_daily_*`는 APIN 개발 이력.

## 핵심 설정값 (스크립트 상단 전역변수)

```python
BASE_DIR              = r"E:\vpin_project_parquet"   # 월별 parquet 루트
YEAR_FOLDERS          = None   # None → KOR_* 전체, 예) ['KOR_2017']
OUTPUT_DIR            = os.path.join(BASE_DIR, "output_data")
FORCE_REPROCESS_STEP1 = False  # True: all_daily_bs.parquet 강제 재생성
CHECKPOINT_N          = 100    # N종목마다 체크포인트 저장
WINDOW_SIZE           = 60     # 롤링 윈도우 크기 (영업일)
MIN_VALID_DAYS        = 30     # 윈도우 내 유효 거래일 최소값
RESUME_RUN_ID         = None   # 중단 재개: "20240115_0930" 형식 문자열
```

## 중단 재개 방법

1. `RESUME_RUN_ID = "YYYYMMDD_HHMM"` (이전 실행의 RUN_ID)으로 설정
2. 스크립트 재실행
3. 완료 후 `RESUME_RUN_ID = None`으로 복원

세션 폴더(`intermediate/session_<RUN_ID>/`)가 RUN_ID별로 격리되므로 인풋이 달라진 새 실행과 섞이지 않는다.

## 출력 파일

```
output_data/
├── all_daily_bs.parquet                        # 일별 B/S 집계 (Step1 결과)
├── pin_daily_rolling_{year}_{RUN_ID}.parquet   # 최종 PIN 결과
├── pin_daily_rolling_{year}_{RUN_ID}_SAMPLE.csv
└── intermediate/session_{RUN_ID}/
    ├── pin_checkpoint_0000.parquet
    └── pin_checkpoint_0001.parquet ...
```

최종 결과 스키마: `Symbol, Date, B, S, a, d, u, eb, es, PIN`
PIN=null은 "거래는 있었지만 추정 조건 미충족 또는 수렴 실패"를 의미.

## 수치 안정성 패턴

- 포아송 PMF → log-space에서 계산 (`k·ln(λ) - λ - gammaln(k+1)`)
- 혼합 우도 합산 → `scipy.special.logsumexp` 트릭
- 그리드 탐색에서 브로드캐스팅: `(1, N)` × `(G, 1)` → `(G, N)` 행렬 연산
- `_make_nll` 클로저: `gammaln(k+1)` 상수를 콜백 밖에서 사전 계산

## 주요 의존 패키지

`polars`, `numpy`, `scipy`, `pyarrow`, `pyreadstat`, `pandas`, `tqdm`, `multiprocessing`
