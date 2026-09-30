"""
[검증 08] 종목코드 'A' — 연속된 목금토일월화수(1주) VPIN 확인
- 목~수 7일이 모두 데이터에 존재하는 주만 선택 (연속 보장)
- 각 주에 대해:
    (1) 요일별 요약: 버킷수 / 버킷번호 범위 / VPIN 건수 / 0건수 / VPIN 범위 / 평균
    (2) 행별 상세: BucketNo, Starttime, Endtime, 거래량, VPIN
- 세트 3개 (토·일 VPIN 변화 주 + 0고정 주 포함), txt 리포트로 저장
"""

import os
import glob
import warnings
from datetime import timedelta

import polars as pl

warnings.filterwarnings("ignore")

pl.Config.set_tbl_cols(-1)
pl.Config.set_tbl_width_chars(320)
pl.Config.set_tbl_rows(400)

# =============================================================================
# 설정
# =============================================================================
BASE_DIR    = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY     = "CA"
SYMBOL      = "RY"

OUTPUT_DIR  = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
REPORT_TXT  = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_report.txt")

WEEKDAY_KR  = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}
REQUIRED_WD = [4, 5, 6, 7, 1, 2, 3]     # 목 → 수 (7일 모두 있어야 연속 주)


def find_latest_parquet(directory):
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [f for f in sorted(glob.glob(pattern))
             if "_sample" not in f and "_cleaned" not in f]
    return files[-1] if files else ""


def thu_wed_range(weekend_date):
    """주말 날짜 → 해당 주 목요일 ~ 다음주 수요일 (7일 윈도우: 목~수)"""
    d = weekend_date
    thu = d - timedelta(days=(d.isoweekday() - 4) % 7)
    wed = thu + timedelta(days=6)         # 목+6 = 수 (연속 7일)
    return thu, wed


def day_summary(day_chunk, vpin_col, bucket_col):
    s = {"bucket_cnt": day_chunk.height}
    if bucket_col and day_chunk.height > 0:
        bnums = day_chunk[bucket_col].drop_nulls()
        s["bucket_range"] = f"{bnums.min()} ~ {bnums.max()}" if bnums.len() > 0 else "N/A"
    else:
        s["bucket_range"] = "N/A"

    if vpin_col and day_chunk.height > 0:
        v = day_chunk[vpin_col].drop_nulls()
        s["vpin_cnt"]  = v.len()
        s["vpin_zero"] = int((v == 0).sum())
        s["vpin_min"]  = float(v.min())  if v.len() > 0 else None
        s["vpin_max"]  = float(v.max())  if v.len() > 0 else None
        s["vpin_mean"] = round(float(v.mean()), 6) if v.len() > 0 else None
    else:
        s["vpin_cnt"] = s["vpin_zero"] = 0
        s["vpin_min"] = s["vpin_max"] = s["vpin_mean"] = None
    return s


def main():
    lines = []

    def w(msg=""):
        s = str(msg)
        print(s)
        lines.append(s + "\n")

    # ------------------------------------------------------------------
    # 1. 로드
    # ------------------------------------------------------------------
    path = find_latest_parquet(OUTPUT_DIR)
    df   = pl.scan_parquet(path).collect()

    df_a = (
        df.filter(pl.col("Symbol") == SYMBOL)
          .with_columns(pl.col("Date").dt.weekday().alias("_wd"))
          .with_columns(
              pl.col("_wd").map_elements(
                  lambda x: WEEKDAY_KR.get(x, "?"), return_dtype=pl.Utf8
              ).alias("요일")
          )
          .sort("Date")
    )

    # 컬럼 탐지
    vpin_col   = next((c for c in ["VPIN", "adjVPIN", "vpin"] if c in df_a.columns), None)
    bucket_col = next((c for c in ["BucketNo", "bucket_no", "Bucket"] if c in df_a.columns), None)
    start_col  = next((c for c in ["Starttime", "StartTime", "start_time", "start"] if c in df_a.columns), None)
    end_col    = next((c for c in ["Endtime",   "EndTime",   "end_time",   "end"]   if c in df_a.columns), None)
    vol_cands  = ["VBS", "OI", "aVBS", "tbuy", "tsell", "Tbuy", "Tsell"]
    vol_cols   = [c for c in vol_cands if c in df_a.columns]

    # 행별 상세 출력 컬럼: Date, 요일, BucketNo, Starttime, Endtime, 거래량(2개), VPIN
    row_cols = ["Date", "요일"]
    if bucket_col: row_cols.append(bucket_col)
    if start_col : row_cols.append(start_col)
    if end_col   : row_cols.append(end_col)
    row_cols += vol_cols[:2]
    if vpin_col  : row_cols.append(vpin_col)

    # 정렬 키 (날짜 → 버킷번호)
    sort_keys = ["Date"] + ([bucket_col] if bucket_col else [])

    # ------------------------------------------------------------------
    # 2. 연속 주(목~수 7일 모두 존재) 탐지 + varying / flat 분류
    # ------------------------------------------------------------------
    weekends = df_a.filter(pl.col("_wd").is_in([6, 7]))["Date"].unique().sort().to_list()
    seen = set()
    varying_weeks, flat_weeks = [], []

    for wd_date in weekends:
        thu, wed = thu_wed_range(wd_date)
        if (thu, wed) in seen:
            continue
        seen.add((thu, wed))

        chunk = df_a.filter((pl.col("Date") >= thu) & (pl.col("Date") <= wed)).sort(sort_keys)
        if chunk.height == 0:
            continue

        # ★ 연속성 검증: 목~수 7개 요일이 모두 존재해야 함
        present_wd = set(chunk["_wd"].unique().to_list())
        if not set(REQUIRED_WD).issubset(present_wd):
            continue

        # 토·일 VPIN 패턴
        we = chunk.filter(pl.col("_wd").is_in([6, 7]))
        if not vpin_col or we.height == 0:
            continue
        vpin_we = we[vpin_col].drop_nulls()
        if vpin_we.len() == 0:
            continue

        if (vpin_we == 0).all():
            flat_weeks.append((thu, wed, chunk))
        else:
            varying_weeks.append((thu, wed, chunk))

    # 세트 선정: 변화 주 2개 + 0고정 주 1개 (없으면 변화 주로 채움)
    flat_n  = min(1, len(flat_weeks))
    vary_n  = min(3 - flat_n, len(varying_weeks))
    selected = []
    for i in range(vary_n):
        selected.append(("변화 (토·일 VPIN ≠ 0)", varying_weeks[i]))
    for i in range(flat_n):
        selected.append(("일정 (토·일 VPIN = 0)", flat_weeks[i]))
    while len(selected) < 3 and len(varying_weeks) > len(selected):
        selected.append(("변화 (토·일 VPIN ≠ 0)", varying_weeks[len(selected)]))

    # ------------------------------------------------------------------
    # 3. 출력
    # ------------------------------------------------------------------
    w("=" * 110)
    w(f"  종목 [{SYMBOL}]  연속 목~수 1주 VPIN 육안 확인  |  국가: {COUNTRY}")
    w(f"  VPIN: {vpin_col}  |  Bucket: {bucket_col}  |  Time: {start_col}~{end_col}")
    w(f"  거래량: {vol_cols[:2]}")
    w(f"  세트 수: {len(selected)}  (변화 주 {vary_n}개 / 0고정 주 {flat_n}개)")
    w(f"  ※ '연속 주' 판정: 목·금·토·일·월·화·수 7개 요일 데이터가 모두 존재하는 주")
    w("=" * 110)

    for set_idx, (label, (thu, wed, chunk)) in enumerate(selected, 1):

        chunk = chunk.sort(sort_keys)

        w("")
        w("=" * 110)
        w(f"  ■ 세트 {set_idx}  |  {thu} (목) ~ {wed} (수)  |  유형: {label}")
        w("=" * 110)

        # 날짜 오름차순 = 목→금→토→일→월→화→수 (윈도우가 연속 7일이므로 자연 정렬)
        dates = sorted(chunk["Date"].unique().to_list())

        # ── (1) 요일별 요약 ──────────────────────────────────────────────
        w("")
        w("  [1] 요일별 요약")
        w(f"  {'날짜':<12} {'요일':<4} {'버킷수':>6}  {'버킷범위':<18} "
          f"{'VPIN건수':>8} {'0건수':>6}  {'VPIN범위 (min ~ max)':<32}  {'평균VPIN':<12}")
        w("  " + "-" * 108)

        for d in dates:
            dc     = chunk.filter(pl.col("Date") == d)
            wd_lbl = dc["요일"][0]
            s      = day_summary(dc, vpin_col, bucket_col)

            vpin_range = (
                f"{s['vpin_min']:.6f} ~ {s['vpin_max']:.6f}"
                if s["vpin_min"] is not None else "N/A"
            )
            mean_str = f"{s['vpin_mean']:.6f}" if s["vpin_mean"] is not None else "N/A"
            mark = " ◀ 주말" if wd_lbl in ["토", "일"] else ""

            w(f"  {str(d):<12} {wd_lbl:<4} {s['bucket_cnt']:>6}  "
              f"{s['bucket_range']:<18} {s['vpin_cnt']:>8} {s['vpin_zero']:>6}  "
              f"{vpin_range:<32}  {mean_str:<12}{mark}")

        # ── (2) 행별 상세: BucketNo, Starttime, Endtime, VPIN ─────────────
        w("")
        w("  [2] 행별 상세 (BucketNo / Starttime / Endtime / VPIN)")
        w("  " + "-" * 108)

        # 헤더
        col_widths = {
            "Date": 12, "요일": 4,
            bucket_col: 9  if bucket_col else 0,
            start_col : 20 if start_col  else 0,
            end_col   : 20 if end_col    else 0,
        }
        for c in vol_cols[:2]:
            col_widths[c] = max(12, len(c) + 2)
        if vpin_col:
            col_widths[vpin_col] = 14

        header = "  " + "  ".join(f"{c:<{col_widths.get(c,12)}}" for c in row_cols)
        w(header)
        w("  " + "-" * 108)

        for d in dates:
            dc     = chunk.filter(pl.col("Date") == d)
            wd_lbl = dc["요일"][0]
            is_we  = wd_lbl in ["토", "일"]

            # 날짜 구분 머리
            head_mark = " ★주말★" if is_we else ""
            w(f"  -- {d} ({wd_lbl}){head_mark} -- [{dc.height}행]")

            for row in dc.select(row_cols).iter_rows(named=True):
                parts = []
                for c in row_cols:
                    wc = col_widths.get(c, 12)
                    val = row.get(c, "")
                    if isinstance(val, float):
                        val = f"{val:.6f}"
                    parts.append(f"{str(val):<{wc}}")
                mark = "  ◀ 주말" if is_we else ""
                w("  " + "  ".join(parts) + mark)

    w("")
    w("=" * 110)
    w(f"  리포트 저장: {REPORT_TXT}")
    w("=" * 110)

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  저장 완료: {REPORT_TXT}")


if __name__ == "__main__":
    main()