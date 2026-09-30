"""
[검증 08] 종목코드 'A' — 연속된 목금토일월화수(1주) VPIN 확인
- 목~수 7일이 모두 데이터에 존재하는 주만 선택 (연속 보장)
- 각 주에 대해:
    (1) 요일별 요약: 버킷수 / 버킷번호 범위 / VPIN 건수 / 0건수 / VPIN 범위 / 평균
    (2) 행별 상세: BucketNo, Starttime, Endtime, 거래량, VPIN
- 세트 3개 (토·일 VPIN 변화 주 + 0고정 주 포함), txt 리포트로 저장
- [추가] 각 세트의 버킷별로 [Starttime, Endtime] 구간에 포함된 1분봉을
        all_1m_bars.parquet 에서 추출하여 {SCRIPT_NAME}_1분봉추출.txt 로 저장
- [추가] VPIN 계산에 사용된 정보를 버킷별로 보기 좋게 정리하여
        {SCRIPT_NAME}_vpin계산정보.txt 로 저장
        (AggBuyVol / AggSellVol / BucketVol / VBS / AOI / VPIN 등)
        * BucketVol = AggBuyVol + AggSellVol   (버킷 크기 V)
        * VBS       = AggBuyVol - AggSellVol   (부호있는 buy-sell 불균형, 파생)
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
SYMBOL      = "A"

OUTPUT_DIR  = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
REPORT_TXT  = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_report.txt")
BARS_TXT    = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_1분봉추출.txt")
CALC_TXT    = os.path.join(OUTPUT_DIR, f"{SCRIPT_NAME}_vpin계산정보.txt")

# 1분봉 데이터 경로
ONE_MIN_PATH = r"E:\vpin_project_parquet\processing_data_2019_2021\R_output\CA\vpin\all_1m_bars.parquet"

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


def _strip_tz(df, col):
    """Datetime 컬럼에 timezone 이 붙어 있으면 제거 (비교 호환용)."""
    if col not in df.columns:
        return df
    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return df.with_columns(pl.col(col).dt.replace_time_zone(None).alias(col))
    return df


def _to_datetime_if_needed(df, col):
    """Date/Utf8 → Datetime 캐스팅 (1분봉 Datetime 과 비교 가능하도록)."""
    if col is None or col not in df.columns:
        return df
    dtype = df.schema[col]
    if isinstance(dtype, pl.Datetime):
        return df
    if dtype == pl.Date:
        return df.with_columns(pl.col(col).cast(pl.Datetime).alias(col))
    if dtype == pl.Utf8:
        return df.with_columns(
            pl.col(col).str.to_datetime(strict=False).alias(col)
        )
    return df


def load_1m_bars(symbol):
    """해당 종목의 1분봉만 로드하고 timezone 제거."""
    if not os.path.exists(ONE_MIN_PATH):
        return None
    df = (
        pl.scan_parquet(ONE_MIN_PATH)
          .filter(pl.col("Symbol") == symbol)
          .collect()
    )
    df = _strip_tz(df, "Datetime")
    df = df.sort("Datetime")
    return df


def write_1m_bars_report(selected, df_bars,
                         bucket_col, start_col, end_col, vpin_col):
    """선정된 세트들의 버킷별 1분봉 데이터를 추출해 txt 로 저장."""
    lines = []

    def w(msg=""):
        lines.append(str(msg) + "\n")

    w("=" * 110)
    w(f"  종목 [{SYMBOL}]  버킷별 1분봉 데이터 추출  |  국가: {COUNTRY}")
    w(f"  1분봉 소스: {ONE_MIN_PATH}")
    w(f"  Bucket: {bucket_col}  |  Time: {start_col} ~ {end_col}  |  VPIN: {vpin_col}")
    if df_bars is None:
        w("  [경고] 1분봉 parquet 파일을 찾을 수 없습니다.")
        with open(BARS_TXT, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  저장 완료(경고): {BARS_TXT}")
        return
    w(f"  로드된 1분봉 건수(전체, 종목 [{SYMBOL}]): {df_bars.height:,}")
    w("=" * 110)

    if not (bucket_col and start_col and end_col):
        w("  [경고] 버킷번호/시작/종료 컬럼이 모두 있어야 추출 가능합니다.")
        with open(BARS_TXT, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  저장 완료(경고): {BARS_TXT}")
        return

    bar_cols = ["Datetime", "Open", "Close", "Volume"]

    for set_idx, (label, (thu, wed, chunk)) in enumerate(selected, 1):

        # 세트의 Starttime/Endtime 타입을 비교 가능한 Datetime 으로 맞춤
        chunk = _to_datetime_if_needed(chunk, start_col)
        chunk = _to_datetime_if_needed(chunk, end_col)
        chunk = _strip_tz(chunk, start_col)
        chunk = _strip_tz(chunk, end_col)

        # 세트 날짜 범위 내 1분봉만 빠르게 선필터 (성능)
        thu_dt = pl.lit(thu).cast(pl.Datetime)
        wed_next = pl.lit(wed + timedelta(days=1)).cast(pl.Datetime)
        bars_set = df_bars.filter(
            (pl.col("Datetime") >= thu_dt) & (pl.col("Datetime") < wed_next)
        )

        w("")
        w("=" * 110)
        w(f"  ■ 세트 {set_idx}  |  {thu} (목) ~ {wed} (수)  |  유형: {label}")
        w(f"  세트 버킷 수: {chunk.height}  |  세트 기간 1분봉 수: {bars_set.height:,}")
        w("=" * 110)

        # 버킷번호 오름차순
        chunk_sorted = chunk.sort([bucket_col]) if bucket_col in chunk.columns else chunk

        # 반복: 각 버킷 행
        select_cols = [bucket_col, start_col, end_col]
        if vpin_col and vpin_col in chunk_sorted.columns:
            select_cols.append(vpin_col)
        if "Date" in chunk_sorted.columns:
            select_cols.append("Date")
        if "요일" in chunk_sorted.columns:
            select_cols.append("요일")

        total_matched = 0

        for row in chunk_sorted.select(select_cols).iter_rows(named=True):
            bno   = row.get(bucket_col)
            st    = row.get(start_col)
            et    = row.get(end_col)
            vp    = row.get(vpin_col) if vpin_col else None
            dd    = row.get("Date")
            wd_lbl = row.get("요일")

            # [Starttime, Endtime] 구간 1분봉 추출 (Endtime 포함)
            if st is None or et is None:
                matched = bars_set.head(0)
            else:
                matched = bars_set.filter(
                    (pl.col("Datetime") >= pl.lit(st)) &
                    (pl.col("Datetime") <= pl.lit(et))
                ).sort("Datetime")

            total_matched += matched.height

            vp_str = f"{vp:.6f}" if isinstance(vp, float) else ("N/A" if vp is None else str(vp))
            mark = "  ◀ 주말" if wd_lbl in ["토", "일"] else ""

            w("")
            w(f"  [버킷 #{bno}]  Date: {dd} ({wd_lbl})  "
              f"Starttime: {st}  Endtime: {et}  VPIN: {vp_str}{mark}")
            w(f"  -- 1분봉 {matched.height}개 --")

            if matched.height == 0:
                w("     (해당 구간에 1분봉 데이터 없음)")
                continue

            # 헤더
            have_cols = [c for c in bar_cols if c in matched.columns]
            widths = {"Datetime": 26, "Open": 12, "Close": 12, "Volume": 12}
            header = "     " + "  ".join(f"{c:<{widths.get(c,12)}}" for c in have_cols)
            w(header)

            for brow in matched.select(have_cols).iter_rows(named=True):
                parts = []
                for c in have_cols:
                    wc = widths.get(c, 12)
                    val = brow.get(c, "")
                    if isinstance(val, float):
                        val = f"{val:.6f}"
                    parts.append(f"{str(val):<{wc}}")
                w("     " + "  ".join(parts))

        w("")
        w(f"  ▶ 세트 {set_idx} 누적 1분봉 매칭 건수: {total_matched:,}")

    w("")
    w("=" * 110)
    w(f"  1분봉 추출 리포트 저장: {BARS_TXT}")
    w("=" * 110)

    with open(BARS_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  저장 완료: {BARS_TXT}")


def _fmt_val(val, width=12, prec=6):
    """숫자면 정밀도 반영, 아니면 str 변환해서 좌측정렬."""
    if val is None:
        s = "N/A"
    elif isinstance(val, float):
        s = f"{val:.{prec}f}"
    elif isinstance(val, int):
        s = f"{val:,}"
    else:
        s = str(val)
    return f"{s:<{width}}"


def write_vpin_calc_info_report(selected, df_a,
                                bucket_col, start_col, end_col,
                                vpin_col, vol_cols):
    """
    VPIN 계산에 사용된 정보 (VBS, OI, tbuy, tsell, aVBS, VPIN 등)를
    버킷별로 보기 좋게 정리하여 txt 로 저장.
    """
    lines = []

    def w(msg=""):
        lines.append(str(msg) + "\n")

    # VPIN 계산 관련 후보 컬럼 전부 탐지
    #  - AggBuyVol / AggSellVol : 실제 VPIN 계산에 사용된 buy/sell 거래량
    #  - BucketVol              : 버킷 크기 (= AggBuyVol + AggSellVol, 파생)
    #  - VBS                    : 부호있는 buy-sell 불균형 (= AggBuyVol - AggSellVol, 파생)
    #  - AOI                    : |V_b - V_s| 계열 (Absolute Order Imbalance)
    calc_cands = [
        "BucketNo", "bucket_no", "Bucket",
        "Starttime", "StartTime", "start_time", "start",
        "Endtime", "EndTime", "end_time", "end",
        "Duration", "duration",
        "AggBuyVol", "AggSellVol", "BucketVol",
        "Volume", "volume", "BucketVolume", "V",
        "tbuy", "tsell", "Tbuy", "Tsell",
        "VBS", "aVBS", "absVBS",
        "AOI", "OI", "orderImbalance",
        "sigma", "Sigma", "ret_std",
        "VPIN", "adjVPIN", "vpin",
        "n_trades", "ntrades", "tradecnt",
    ]
    calc_cols = [c for c in calc_cands if c in df_a.columns]

    # 열 너비 계산 (값 길이에 따라 가변)
    def width_for(c):
        if c in ("Date",):                                  return 12
        if c in ("요일",):                                    return 4
        if c in ("BucketNo", "bucket_no", "Bucket"):         return 9
        if c in ("Starttime", "StartTime", "start_time",
                 "start", "Endtime", "EndTime",
                 "end_time", "end"):                         return 20
        if c in ("AggBuyVol", "AggSellVol", "BucketVol",
                 "Volume", "volume", "BucketVolume", "V"):   return 14
        if c in ("VBS", "aVBS", "absVBS",
                 "AOI", "OI", "orderImbalance"):             return 14
        if c in ("VPIN", "adjVPIN", "vpin",
                 "sigma", "Sigma", "ret_std"):               return 14
        if c in ("Duration", "duration",
                 "n_trades", "ntrades", "tradecnt"):         return 10
        return 14

    display_cols = ["Date", "요일"] + calc_cols

    w("=" * 140)
    w(f"  종목 [{SYMBOL}]  VPIN 계산에 사용된 정보 (버킷별 상세)  |  국가: {COUNTRY}")
    w(f"  VPIN: {vpin_col}  |  Bucket: {bucket_col}  |  Time: {start_col} ~ {end_col}")
    w(f"  계산 컬럼: {calc_cols}")
    w(f"  거래량 계열(탐지): {vol_cols}")
    w("=" * 140)

    if not calc_cols:
        w("  [경고] VPIN 계산 관련 컬럼이 데이터에 없어 출력할 내용이 없습니다.")
        with open(CALC_TXT, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print(f"  저장 완료(경고): {CALC_TXT}")
        return

    for set_idx, (label, (thu, wed, chunk)) in enumerate(selected, 1):

        # 세트의 요약(총량/평균 등) 계산
        set_rows = chunk.height
        vpin_series = chunk[vpin_col].drop_nulls() if vpin_col and vpin_col in chunk.columns else None

        w("")
        w("=" * 140)
        w(f"  ■ 세트 {set_idx}  |  {thu} (목) ~ {wed} (수)  |  유형: {label}  |  버킷 수: {set_rows}")
        if vpin_series is not None and vpin_series.len() > 0:
            w(f"    └ VPIN: n={vpin_series.len()}, "
              f"min={float(vpin_series.min()):.6f}, "
              f"max={float(vpin_series.max()):.6f}, "
              f"mean={float(vpin_series.mean()):.6f}, "
              f"0건수={int((vpin_series == 0).sum())}")

        # 거래량 계열 집계 (존재하는 것만, AggBuyVol/AggSellVol/BucketVol 우선)
        vol_summary_cols = [
            "AggBuyVol", "AggSellVol", "BucketVol",
            "VBS", "aVBS", "absVBS",
            "AOI", "OI",
            "tbuy", "tsell", "Tbuy", "Tsell",
            "Volume",
        ]
        agg_exprs = []
        for c in vol_summary_cols:
            if c in chunk.columns:
                agg_exprs.append(pl.col(c).sum().alias(f"{c}_sum"))
                agg_exprs.append(pl.col(c).mean().alias(f"{c}_mean"))
        if agg_exprs:
            agg = chunk.select(agg_exprs).row(0, named=True)
            parts = []
            seen = set()
            for c in vol_summary_cols:
                if c in chunk.columns and c not in seen:
                    seen.add(c)
                    s = agg.get(f"{c}_sum")
                    m = agg.get(f"{c}_mean")
                    s_str = f"{s:,.4f}" if isinstance(s, float) else (f"{s:,}" if s is not None else "N/A")
                    m_str = f"{m:,.4f}" if isinstance(m, float) else (f"{m:,}" if m is not None else "N/A")
                    parts.append(f"{c}(합 {s_str} / 평균 {m_str})")
            if parts:
                w(f"    └ 거래량 요약: " + "  |  ".join(parts))

        # 버킷 크기(BucketVol) 분포 요약
        if "BucketVol" in chunk.columns:
            bv = chunk["BucketVol"].drop_nulls()
            if bv.len() > 0:
                w(f"    └ 버킷 크기(BucketVol = AggBuyVol+AggSellVol): "
                  f"n={bv.len()}, "
                  f"min={float(bv.min()):,.2f}, "
                  f"max={float(bv.max()):,.2f}, "
                  f"mean={float(bv.mean()):,.2f}, "
                  f"median={float(bv.median()):,.2f}")
        w("=" * 140)

        # 날짜 오름차순 표시
        dates = sorted(chunk["Date"].unique().to_list())

        # 헤더
        w("")
        header = "  " + "  ".join(f"{c:<{width_for(c)}}" for c in display_cols)
        w(header)
        w("  " + "-" * (len(header) - 2))

        # 날짜별로 모아서 출력
        sort_keys_local = (["Date"] + [bucket_col]) if bucket_col and bucket_col in chunk.columns else ["Date"]
        chunk_sorted = chunk.sort(sort_keys_local)

        for d in dates:
            dc = chunk_sorted.filter(pl.col("Date") == d)
            if dc.height == 0:
                continue
            wd_lbl = dc["요일"][0]
            is_we = wd_lbl in ["토", "일"]
            mark = " ★주말★" if is_we else ""
            w(f"  -- {d} ({wd_lbl}){mark} -- [{dc.height}행]")

            for row in dc.select(display_cols).iter_rows(named=True):
                parts = []
                for c in display_cols:
                    parts.append(_fmt_val(row.get(c), width=width_for(c)))
                tail = "  ◀ 주말" if is_we else ""
                w("  " + "  ".join(parts) + tail)

        # 세트 말미 소계
        w("")
        w(f"  ▶ 세트 {set_idx} 합계/평균 (계산 컬럼별)")
        num_cols = [c for c in calc_cols
                    if chunk.schema[c] in (pl.Float32, pl.Float64,
                                            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                                            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)]
        if num_cols:
            stat_exprs = []
            for c in num_cols:
                stat_exprs += [
                    pl.col(c).sum().alias(f"{c}__sum"),
                    pl.col(c).mean().alias(f"{c}__mean"),
                    pl.col(c).min().alias(f"{c}__min"),
                    pl.col(c).max().alias(f"{c}__max"),
                ]
            stats = chunk.select(stat_exprs).row(0, named=True)
            head2 = f"  {'컬럼':<16}  {'합계':>18}  {'평균':>18}  {'최소':>18}  {'최대':>18}"
            w(head2)
            w("  " + "-" * (len(head2) - 2))
            for c in num_cols:
                s = stats.get(f"{c}__sum")
                m = stats.get(f"{c}__mean")
                mn = stats.get(f"{c}__min")
                mx = stats.get(f"{c}__max")
                def _ns(v):
                    if v is None: return "N/A"
                    if isinstance(v, float): return f"{v:,.6f}"
                    return f"{v:,}"
                w(f"  {c:<16}  {_ns(s):>18}  {_ns(m):>18}  {_ns(mn):>18}  {_ns(mx):>18}")

    w("")
    w("=" * 140)
    w(f"  VPIN 계산정보 리포트 저장: {CALC_TXT}")
    w("=" * 140)

    with open(CALC_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"  저장 완료: {CALC_TXT}")


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

    # --- [추가] AggBuyVol / AggSellVol 이 있으면 BucketVol(버킷크기) 와 VBS 파생 ---
    #   BucketVol = AggBuyVol + AggSellVol   (VPIN 의 V, 버킷 크기)
    #   VBS       = AggBuyVol - AggSellVol   (부호있는 buy-sell 불균형)
    if "AggBuyVol" in df_a.columns and "AggSellVol" in df_a.columns:
        new_cols = []
        if "BucketVol" not in df_a.columns:
            new_cols.append((pl.col("AggBuyVol") + pl.col("AggSellVol")).alias("BucketVol"))
        if "VBS" not in df_a.columns:
            new_cols.append((pl.col("AggBuyVol") - pl.col("AggSellVol")).alias("VBS"))
        if new_cols:
            df_a = df_a.with_columns(new_cols)

    # 컬럼 탐지
    vpin_col   = next((c for c in ["VPIN", "adjVPIN", "vpin"] if c in df_a.columns), None)
    bucket_col = next((c for c in ["BucketNo", "bucket_no", "Bucket"] if c in df_a.columns), None)
    start_col  = next((c for c in ["Starttime", "StartTime", "start_time", "start"] if c in df_a.columns), None)
    end_col    = next((c for c in ["Endtime",   "EndTime",   "end_time",   "end"]   if c in df_a.columns), None)
    # 거래량 계열: AggBuyVol/AggSellVol/BucketVol/VBS/AOI 를 최우선
    vol_cands  = ["AggBuyVol", "AggSellVol", "BucketVol", "VBS", "AOI",
                  "OI", "aVBS", "tbuy", "tsell", "Tbuy", "Tsell"]
    vol_cols   = [c for c in vol_cands if c in df_a.columns]

    # 행별 상세 출력 컬럼: Date, 요일, BucketNo, Starttime, Endtime, VPIN
    row_cols = ["Date", "요일"]
    if bucket_col: row_cols.append(bucket_col)
    if start_col : row_cols.append(start_col)
    if end_col   : row_cols.append(end_col)
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
        for c in row_cols:
            if c in col_widths:
                continue
            if c in ("AggBuyVol", "AggSellVol", "BucketVol"):
                col_widths[c] = 14
            elif c == "VBS":
                col_widths[c] = 14
            elif c == "AOI":
                col_widths[c] = 14
            else:
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

    # ------------------------------------------------------------------
    # 4. [추가] 버킷별 1분봉 추출 리포트
    # ------------------------------------------------------------------
    print("\n[1분봉 추출] 시작 ...")
    df_bars = load_1m_bars(SYMBOL)
    write_1m_bars_report(
        selected, df_bars,
        bucket_col=bucket_col,
        start_col=start_col,
        end_col=end_col,
        vpin_col=vpin_col,
    )

    # ------------------------------------------------------------------
    # 5. [추가] VPIN 계산에 사용된 정보(VBS 등) 정리 리포트
    # ------------------------------------------------------------------
    print("\n[VPIN 계산정보] 시작 ...")
    write_vpin_calc_info_report(
        selected, df_a,
        bucket_col=bucket_col,
        start_col=start_col,
        end_col=end_col,
        vpin_col=vpin_col,
        vol_cols=vol_cols,
    )


if __name__ == "__main__":
    main()