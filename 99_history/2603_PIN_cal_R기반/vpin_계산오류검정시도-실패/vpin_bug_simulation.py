"""
=============================================================================
[VPIN 버그 시뮬레이션] R 코드의 타임존 버그를 Python에서 재현
=============================================================================

목적:
  vpin_r_heavy.R의 tz="Asia/Seoul" 하드코딩이 CA(캐나다) 데이터에서
  어떻게 주말 날짜를 만드는지 정확히 시뮬레이션한다.

실제 parquet 파일 없이 버그를 수치로 증명.

실행: python vpin_bug_simulation.py
=============================================================================
"""

from datetime import datetime, timezone, date
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
ET  = ZoneInfo("America/Toronto")   # CA 현지 타임존
UTC = timezone.utc

print("=" * 70)
print(" VPIN 타임존 버그 시뮬레이션 - R 코드의 as.Date(tz='Asia/Seoul') 재현")
print("=" * 70)

print("""
[버그 메커니즘]
  1. Python 전처리: ET 현지 시각을 naive datetime으로 parquet에 저장
  2. R/Arrow 로드: naive datetime → UTC POSIXct로 해석 (Arrow 기본값)
  3. R 코드: attr(ts_raw, "tzone") <- "Asia/Seoul"  ← 라벨만 변경, 값 불변
  4. R 코드: as.Date(et_vals, tz="Asia/Seoul")  ← KST 기준으로 날짜 추출
  5. KST = UTC+9, ET(EST) = UTC-5 → 차이 14시간
  6. naive값 15:00이상 (실제로 ET 15:00) → KST 자정(00:00) 넘어 → 날짜 +1!
""")

# CA 거래 시간대 (ET): 09:30 ~ 16:00
# naive 저장값 = ET 현지 시각 (그대로 저장)
# R이 이를 UTC로 오해 → KST 날짜 추출

print(f"{'─'*70}")
print(f"{'ET 현지 시각 (저장값)':>28} | {'R이 만드는 날짜(KST)':>22} | {'올바른 날짜(ET)':>16} | {'버그?':>5}")
print(f"{'─'*28}+{'─'*22}+{'─'*16}+{'─'*5}")

# 2019년 1월 4일 (금요일) 거래 시간대를 샘플링
test_cases = [
    # (날짜, 시:분, 설명)
    (date(2019, 1, 4), (9, 30),  "장 시작"),
    (date(2019, 1, 4), (12, 0),  "장 중간"),
    (date(2019, 1, 4), (14, 59), "14:59 ET"),
    (date(2019, 1, 4), (15, 0),  "15:00 ET ← 경계"),
    (date(2019, 1, 4), (15, 1),  "15:01 ET ← 버그 시작"),
    (date(2019, 1, 4), (15, 30), "15:30 ET"),
    (date(2019, 1, 4), (15, 59), "15:59 ET"),
    (date(2019, 1, 4), (16, 0),  "장 마감"),
]

bugs = []
for d, (h, m), desc in test_cases:
    # 1. Python이 저장한 naive 값 (ET 현지 시각)
    naive_dt = datetime(d.year, d.month, d.day, h, m, 0)

    # 2. R/Arrow: naive → UTC POSIXct로 해석 (epoch 기준)
    as_utc = naive_dt.replace(tzinfo=UTC)

    # 3. R: as.Date(tz="Asia/Seoul") — UTC epoch를 KST 날짜로 변환
    r_date = as_utc.astimezone(KST).date()

    # 4. 올바른 날짜: naive값이 실제로는 ET 시각이므로
    correct_date = naive_dt.date()   # ET 날짜 = naive 날짜

    bug = r_date != correct_date
    if bug:
        bugs.append((naive_dt, r_date, correct_date, desc))

    mark = " ✗ BUG!" if bug else " ✓ OK"
    print(f"{str(naive_dt):>28} | {str(r_date):>22} | {str(correct_date):>16} | {mark}")

print(f"{'─'*70}")

print(f"\n[결론]")
if bugs:
    print(f"  버그 발생 케이스 {len(bugs)}개:")
    for naive_dt, r_date, correct_date, desc in bugs:
        print(f"    {naive_dt} ({desc})")
        print(f"      R 날짜: {r_date} ({['월','화','수','목','금','토','일'][r_date.weekday()]}) ← 잘못됨")
        print(f"      올바른: {correct_date} ({['월','화','수','목','금','토','일'][correct_date.weekday()]})")
        print()


# 금요일 15:00 이후가 토요일로 바뀌는 것 명시적으로 보여주기
print("=" * 70)
print(" 금요일 15:00 이후 버킷이 토요일 날짜로 기록되는 현상")
print("=" * 70)

fri_cases = [
    datetime(2019, 1, 4, 14, 59, 59),  # 금요일 14:59:59
    datetime(2019, 1, 4, 15, 0, 0),    # 금요일 15:00:00 ← 경계
    datetime(2019, 1, 4, 15, 0, 1),    # 금요일 15:00:01 ← 토요일로 바뀜
    datetime(2019, 1, 4, 15, 30, 0),   # 금요일 15:30
    datetime(2019, 1, 4, 16, 0, 0),    # 금요일 16:00 (장 마감)
]

for dt in fri_cases:
    as_utc = dt.replace(tzinfo=UTC)
    r_date = as_utc.astimezone(KST).date()
    correct = dt.date()
    weekday_r = ["월","화","수","목","금","토","일"][r_date.weekday()]
    weekday_c = ["월","화","수","목","금","토","일"][correct.weekday()]
    bug_str = " ← ✗ 토요일!" if r_date.weekday() >= 5 else ""
    print(f"  {dt} → R 날짜: {r_date}({weekday_r}){bug_str}")

print(f"""
[CA 장 시간에서 영향받는 구간]
  장 시간: 09:30 ~ 16:00 ET
  정상 구간: 09:30 ~ 14:59 ET  (날짜 올바름)
  버그 구간: 15:00 ~ 16:00 ET  (날짜 1일 밀림 = 주말이면 토요일)

  → 금요일 15:00~16:00 사이에 마감되는 버킷은 전부 '토요일' 날짜를 받음
  → 이것이 VPIN 결과에 토요일이 나타나는 근본 원인
""")


# 실제 R 코드에서 어디를 고쳐야 하는지
print("=" * 70)
print(" R 코드 수정 위치 (vpin_r_heavy.R)")
print("=" * 70)
print("""
  수정 전 (현재):
  ─────────────────────────────────────────────
  attr(ts_raw, "tzone") <- "Asia/Seoul"       # 행 314
  ts_raw <- as.POSIXct(ts_raw,
             origin = "1970-01-01",
             tz = "Asia/Seoul")               # 행 315-316
  ts_raw <- as.POSIXct(as.character(ts_raw),
             tz = "Asia/Seoul")               # 행 317-318
  ...
  date_vals <- as.Date(et_vals, tz = "Asia/Seoul")  # 행 410
  ─────────────────────────────────────────────

  수정 후 (올바름):
  ─────────────────────────────────────────────
  # 0) COUNTRY별 타임존 맵 (설정 구역에 추가)
  tz_map <- list(
    KOR = "Asia/Seoul",     US = "America/New_York",
    CA  = "America/Toronto", JP = "Asia/Tokyo",
    UK  = "Europe/London",  FR = "Europe/Paris",
    GR  = "Europe/Berlin",  HK = "Asia/Hong_Kong",
    IT  = "Europe/Rome"
  )
  LOCAL_TZ <- tz_map[[COUNTRY]]

  # 1) worker_fn args에 local_tz 추가 (symbol_args 생성 시)
  list(
    ...
    local_tz = LOCAL_TZ   # ← 추가
  )

  # 2) worker_fn 내부 - args에서 꺼내기
  local_tz <- args$local_tz

  # 3) 타임존 적용
  attr(ts_raw, "tzone") <- local_tz           # "Asia/Seoul" → local_tz
  as.POSIXct(..., tz = local_tz)              # "Asia/Seoul" → local_tz

  # 4) 날짜 추출
  date_vals <- as.Date(et_vals, tz = local_tz)  # "Asia/Seoul" → local_tz
  ─────────────────────────────────────────────

  ★ KOR 데이터는 영향 없음 (KOR = Asia/Seoul 이므로 동일)
  ★ CA, US, UK, JP 등 비한국 데이터는 모두 수정 필요
""")

print("=" * 70)
print(" 시뮬레이션 완료")
print("=" * 70)
