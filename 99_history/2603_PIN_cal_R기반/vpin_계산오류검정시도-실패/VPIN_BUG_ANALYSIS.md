# VPIN 주말 발생 문제 분석 보고서

작성일: 2026-04-12  
대상: `vpin_r_heavy.R` (CA 국가 기준 분석)

---

## 1. 문제 요약

VPIN 결과 parquet(`vpin_CA_*.parquet`)에서 **토요일 날짜**가 포함된 버킷이 발견됨.  
1분봉 입력 데이터(`all_1m_bars.parquet`)에는 주말 데이터가 없음.  
→ R 계산 단계에서 날짜가 잘못 배정되고 있음.

---

## 2. 근본 원인 — `tz="Asia/Seoul"` 하드코딩 버그

### 2.1 문제 코드 위치 (vpin_r_heavy.R)

```r
# ── POSIXct tzone 설정 ──  (worker_fn 내부)
ts_raw <- sym_df$Datetime
if (inherits(ts_raw, "POSIXct")) {
    attr(ts_raw, "tzone") <- "Asia/Seoul"   ← ★ 버그 1 (모든 나라에 Seoul 적용)
} else if (is.numeric(ts_raw)) {
    ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01",
                          tz = "Asia/Seoul")   ← ★ 버그 2
} else {
    ts_raw <- as.POSIXct(as.character(ts_raw),
                          tz = "Asia/Seoul")   ← ★ 버그 3
}

...

# ★ 날짜 추출: endtime 기준
date_vals <- as.Date(et_vals, tz = "Asia/Seoul")   ← ★ 버그 4 (KST 기준 날짜 추출)
```

### 2.2 버그 발생 메커니즘 (CA 기준)

| 단계 | 동작 | 값 예시 |
|------|------|---------|
| 1. Python 저장 | ET 현지 시각 → naive datetime | `2019-01-04 15:30:00` (금요일 오후 3:30 ET) |
| 2. Arrow → R 로드 | naive → UTC POSIXct 해석 | `2019-01-04 15:30:00 UTC` |
| 3. R tzone 변경 | 값 변경 없이 라벨만 변경 | 내부 epoch 불변, 표시만 `KST` |
| 4. `as.Date(tz="Asia/Seoul")` | UTC 15:30 = KST 00:30 **다음날** | **`2019-01-05` (토요일) ← 오류!** |

**결론:** 금요일 **ET 15:00 ~ 16:00** 구간의 모든 버킷이 **토요일 날짜**로 기록됨.

### 2.3 시뮬레이션 결과 (Python으로 R 동작 재현)

```
ET 현지 시각 (저장값)    | R이 만드는 날짜(KST) | 올바른 날짜(ET) | 버그?
─────────────────────────+──────────────────────+────────────────+──────
2019-01-04 09:30:00      | 2019-01-04           | 2019-01-04     | ✓ OK
2019-01-04 12:00:00      | 2019-01-04           | 2019-01-04     | ✓ OK
2019-01-04 14:59:00      | 2019-01-04           | 2019-01-04     | ✓ OK
2019-01-04 15:00:00      | 2019-01-05           | 2019-01-04     | ✗ BUG! ← 경계
2019-01-04 15:30:00      | 2019-01-05           | 2019-01-04     | ✗ BUG!
2019-01-04 16:00:00      | 2019-01-05           | 2019-01-04     | ✗ BUG!
```

**CA 장 마감 1시간** (15:00~16:00)의 버킷이 전부 다음날 날짜를 받음.  
금요일이면 → **토요일**로 기록.

---

## 3. 기존 검증 코드 유효성 평가

### 3.1 `0409_VpinValidate.py` — 유효함 ✓

- 주말/공휴일 VPIN 존재 여부 검출  
- `exchange_calendars`로 영업일 기반 검증  
- **평가:** 문제를 정확히 감지하는 1차 스크리닝으로 유효함

### 3.2 `0409_틱데이터주말체크.py` — 유효함 ✓

- `all_1m_bars.parquet`에 주말 봉이 있는지 확인  
- 결과: 주말 봉 없음 → 버그가 Python 전처리가 아닌 R 계산 단계에서 발생함을 확인  
- **평가:** 파이프라인 범위를 좁히는 데 필수적인 검증으로 유효함

### 3.3 `0412_01.py` — 유효함 ✓ (단, 부분적 오해 가능)

- Python 입력 vs R 출력 타임스탬프 1:1 대조  
- 금요일 마지막 틱이 있는데 R 결과가 토요일로 나올 때 "원본 없는 가짜 시간" 출력  
- **평가:** 올바른 방향이나, 실제로는 타임스탬프 자체(Starttime/Endtime epoch)는 금요일 오후 값이고, **Date 컬럼**만 잘못된 것임을 명확히 해야 함

### 3.4 `0412_02선형보간.py` — 부분적으로 오해의 소지 있음 ⚠️

- 주말 버킷이 "금요일 마지막 틱 ~ 월요일 첫 틱" 사이에 있는지 ASOF Join으로 확인  
- 결과: 98%+ 금-월 패턴으로 "보간 가설 확정"이라고 출력  
- **평가:** 이 결과는 **타임존 버그로도 완전히 설명됨**  
  - 금요일 15:00~16:00 ET 버킷 → 토요일 날짜로 오기록  
  - 그 버킷들은 금요일 마지막 틱(15:xx ET) ~ 월요일 첫 틱 사이에 당연히 존재  
  - 보간(interpolation)이 아니라 날짜 라벨 오류임!  
  - **확인 방법:** 주말 버킷의 `Starttime` 값의 시간(hour)이 15~16이면 타임존 버그, 0~14 또는 전체에 분포하면 보간 버그

### 3.5 `0412_vpin_rawcheck.py` — 유효함 ✓

- 원시 틱 데이터와 1분봉 집계 데이터의 요일 분포 비교  
- 각 파이프라인 단계에서 주말 데이터 유입 여부 확인  
- **평가:** 단계별 범위 좁히기에 유효함

### 3.6 `0412_vpin_val.py` — 매우 유효함 ✓✓

- `check_anomaly_hours()` 함수로 주말 버킷의 시간대 분포 분석  
- 해석 가이드: "새벽이면 타임존 버그, 장중에 퍼져있으면 원시 데이터 노이즈"  
- **평가:** 이 함수의 결과가 **핵심 증거**  
  - 주말 버킷 Starttime이 Hour 15~16에 집중 → 타임존 버그 확정  
  - 버그가 있다면 결과는 Hour 15, 16에 거의 100% 집중될 것

---

## 4. 타임존 버그 vs 보간 버그 구별법

| 구분 | 타임존 버그 (실제 원인) | 보간 버그 (잘못된 가설) |
|------|------------------------|------------------------|
| 주말 버킷 Starttime | 15:00~16:00에 집중 | 00:00~23:59 전 범위 분포 |
| 주말 버킷 Date | 항상 원본 날짜 +1 | 실제로 주말 날짜 |
| 보간 패턴 (0412_02) | 99%+ 금-월 패턴 (오해 소지) | 99%+ 금-월 패턴 |
| 1분봉 데이터에 주말 | 없음 | 없음 |
| Starttime epoch 확인 | 금요일 ET 값과 동일 | 토/일요일 ET 값 |

**즉, `0412_vpin_val.py`의 Hour 분포가 15~16에 집중되어 있으면 타임존 버그가 유일한 원인.**

---

## 5. 수정 방법

### 5.1 `vpin_r_heavy.R` 수정

**1) 설정 구역에 타임존 맵 추가:**

```r
# ── 국가별 타임존 맵 (★ 추가) ────────────────────────────────────────────────
tz_map <- list(
  KOR = "Asia/Seoul",
  US  = "America/New_York",
  CA  = "America/Toronto",
  JP  = "Asia/Tokyo",
  UK  = "Europe/London",
  FR  = "Europe/Paris",
  GR  = "Europe/Berlin",
  HK  = "Asia/Hong_Kong",
  IT  = "Europe/Rome"
)
LOCAL_TZ <- tz_map[[COUNTRY]]
```

**2) `symbol_args` 생성 시 `local_tz` 전달:**

```r
list(
  symbol          = sym,
  sym_file        = fpath,
  checkpoint_dir  = CHECKPOINT_DIR,
  timebarsize_sec = TIMEBARSIZE_SEC,
  buckets_per_day = BUCKETS_PER_DAY,
  samplength      = SAMPLENGTH,
  tradinghours    = TRADINGHOURS,
  local_tz        = LOCAL_TZ         # ← 추가
)
```

**3) `worker_fn` 내부 수정:**

```r
worker_fn <- function(args) {
  ...
  local_tz <- args$local_tz          # ← 추가

  # 변경 전:
  attr(ts_raw, "tzone") <- "Asia/Seoul"
  
  # 변경 후:
  attr(ts_raw, "tzone") <- local_tz

  # 변경 전:
  ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01", tz = "Asia/Seoul")
  
  # 변경 후:
  ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01", tz = local_tz)

  # 변경 전:
  date_vals <- as.Date(et_vals, tz = "Asia/Seoul")
  
  # 변경 후:
  date_vals <- as.Date(et_vals, tz = local_tz)
}
```

### 5.2 수정 후 처리 절차

1. 기존 체크포인트 삭제: `checkpoints/sym_*.rds` 전부 삭제
2. `Rscript vpin_r.R` 재실행
3. `python vpin_diagnose.py` 로 주말 VPIN 없는지 확인

---

## 6. 국가별 영향 범위

| 국가 | 현지 TZ | UTC 오프셋 | 날짜 밀림 기준 | 영향 |
|------|---------|-----------|---------------|------|
| KOR | Asia/Seoul | +9 | 해당 없음 | **없음** (KOR은 올바름) |
| CA | America/Toronto | -5 (EST) | 15:00 이후 | 장 마감 1시간 (15~16시) |
| US | America/New_York | -5 (EST) | 15:00 이후 | 장 마감 1시간 (15~16시) |
| UK | Europe/London | 0 (GMT) | 09:00 이후 | 장 거의 전체 (09:00~16:30) |
| JP | Asia/Tokyo | +9 | 해당 없음 | **없음** (JP = KST와 동일 UTC 오프셋) |
| FR | Europe/Paris | +1 (CET) | 08:00 이후 | 장 거의 전체 |
| GR | Europe/Berlin | +1 (CET) | 08:00 이후 | 장 거의 전체 |

> UK, FR, GR은 KST와 차이가 커서 훨씬 많은 버킷이 날짜 오류를 가짐

---

## 7. 검증 스크립트 안내

| 파일 | 목적 | 실행 환경 |
|------|------|----------|
| `vpin_diagnose.py` | 실제 parquet으로 타임존/보간 가설 통합 검증 | Windows (E: 드라이브 접근 필요) |
| `vpin_bug_simulation.py` | parquet 없이 타임존 버그 수치 재현 | 어디서나 실행 가능 |
| `0412_vpin_val.py` | Hour 분포로 타임존 vs 보간 판별 | Windows (E: 드라이브 접근 필요) |

**빠른 확인 방법:**  
`0412_vpin_val.py`를 실행하여 주말 버킷의 **발생 시간대 분포**를 확인.  
Hour 15~16에 집중 → 타임존 버그 확정 → `vpin_r_heavy.R` 수정 필요.
