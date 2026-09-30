"""
=============================================================================
[검증] tick parquet 폴더의 월(month) 커버리지 검사

배경
  · 파일명이 202003 다음에 202007로 점프해서, 4/5/6월 데이터가 누락인지
    아니면 다른 파일명에 들어있는지 확인이 필요.
  · 파일명만 보면 안 됨 (비표준 파일명에 들어있을 수도 있고,
    한 파일이 여러 월에 걸쳐 있을 수도 있음).
  · 따라서 파일 내부의 실제 Datetime을 보고 month coverage를 계산한다.

검사 흐름
  1) BASE_DIR 아래 {COUNTRY}_YYYY 폴더들을 모두 스캔
  2) 각 parquet 파일에서 Date 컬럼의 min/max month를 추출
     - parquet metadata의 row group statistics를 우선 사용 (메모리·시간 부담 0)
     - 없거나 신뢰 불가면 polars로 lazy min/max
  3) 데이터셋 전체 month 범위 안에서
     - 어느 파일에도 안 들어있는 month → 누락
     - 여러 파일에 걸쳐있는 month → 중복 (분할 케이스인지 확인 필요)
  4) 콘솔 요약 + 상세 리포트(.txt) + 매트릭스 CSV 출력

사용법
  · USER CONFIG의 BASE_DIR, COUNTRY 설정 후 실행
  · 결과 리포트는 BASE_DIR 아래에 저장됨
=============================================================================
"""

import os
import re
import sys
import csv
import glob
from datetime import datetime, date
from typing import List, Dict, Tuple, Optional, Set, Any

import pyarrow.parquet as pq
import polars as pl


# =============================================================================
# USER CONFIG
# =============================================================================

BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY = "CN"

# Date 컬럼명 (코드 보니 'Date'로 통일되어 있음)
DATE_COLUMN = "Date"

# 검사에서 제외할 폴더명들 (대소문자 무시 부분 일치)
EXCLUDE_FOLDER_KEYWORDS = ["backup", "_old", "trash", "tmp"]

# 결과 리포트 저장 폴더 (None이면 BASE_DIR)
REPORT_DIR: Optional[str] = None


# =============================================================================
# 1. 폴더/파일 검색
# =============================================================================

_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")


def is_excluded_path(path: str) -> bool:
    """경로 어딘가에 제외 키워드가 들어가면 True."""
    p_low = path.lower()
    return any(kw.lower() in p_low for kw in EXCLUDE_FOLDER_KEYWORDS)


def collect_country_files(base_dir: str, country: str) -> List[Tuple[str, str]]:
    """
    {COUNTRY}_YYYY 폴더들을 돌며 그 안의 *.parquet 파일들을 수집.
    백업 폴더 등은 제외.

    return: [(folder_basename, file_path), ...]
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"BASE_DIR이 존재하지 않습니다: {base_dir}")

    out: List[Tuple[str, str]] = []
    folder_count = 0
    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        m = _YEAR_FOLDER_RE.match(entry.name)
        if not m or m.group(1) != country:
            continue
        if is_excluded_path(entry.path):
            print(f"  [Skip 폴더] {entry.name} (excluded keyword)")
            continue
        folder_count += 1

        for fpath in sorted(glob.glob(os.path.join(entry.path, "*.parquet"))):
            if is_excluded_path(fpath):
                continue
            out.append((entry.name, fpath))

    print(f"  {country}_YYYY 폴더 수: {folder_count}")
    print(f"  parquet 파일 수    : {len(out)}")
    return out


# =============================================================================
# 2. 파일별 month 커버리지 추출
# =============================================================================

def filename_guess_month(filename: str) -> Optional[str]:
    """
    파일명에서 'YYYYMM'을 추정 (참고용).
    예: '202003.parquet' → '2020-03', '202003_part1.parquet' → '2020-03'
        못 찾으면 None.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"(\d{4})(\d{2})", base)
    if not m:
        return None
    yyyy = int(m.group(1))
    mm = int(m.group(2))
    if not (1 <= mm <= 12) or not (1900 <= yyyy <= 2100):
        return None
    return f"{yyyy:04d}-{mm:02d}"


def _to_date(v: Any) -> Optional[date]:
    """다양한 표현(date, datetime, str)을 date로 통일."""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        # ISO 형식 시도
        try:
            return datetime.fromisoformat(v).date()
        except Exception:
            return None
    return None


def months_in_range(d_min: date, d_max: date) -> List[str]:
    """[d_min, d_max] 사이의 모든 month를 'YYYY-MM' 문자열 리스트로."""
    if d_min > d_max:
        return []
    months = []
    y, m = d_min.year, d_min.month
    end_y, end_m = d_max.year, d_max.month
    while (y, m) <= (end_y, end_m):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def file_month_coverage_via_metadata(path: str) -> Optional[Tuple[date, date, str]]:
    """
    parquet의 row group statistics에서 Date 컬럼의 min/max를 직접 읽는다.
    파일 자체를 읽지 않고 metadata만 보므로 매우 빠름.

    return: (min_date, max_date, "metadata") | None (통계 없거나 실패)
    """
    try:
        pf = pq.ParquetFile(path)
        schema = pf.schema_arrow
        if DATE_COLUMN not in [f.name for f in schema]:
            return None

        col_idx = [f.name for f in schema].index(DATE_COLUMN)

        global_min: Optional[date] = None
        global_max: Optional[date] = None
        any_stats = False

        for rg_idx in range(pf.num_row_groups):
            rg = pf.metadata.row_group(rg_idx)
            col_meta = rg.column(col_idx)
            stats = col_meta.statistics
            if stats is None or not stats.has_min_max:
                continue
            any_stats = True
            mn = _to_date(stats.min)
            mx = _to_date(stats.max)
            if mn is None or mx is None:
                continue
            if global_min is None or mn < global_min:
                global_min = mn
            if global_max is None or mx > global_max:
                global_max = mx

        if not any_stats or global_min is None or global_max is None:
            return None
        return (global_min, global_max, "metadata")
    except Exception:
        return None


def file_month_coverage_via_polars(path: str) -> Optional[Tuple[date, date, str]]:
    """
    metadata로 안 될 때 fallback. Date 컬럼만 lazy scan으로 min/max를 구함.
    컬럼 1개만 읽으니 메모리 부담 작음.

    return: (min_date, max_date, "polars") | None
    """
    try:
        result = (
            pl.scan_parquet(path)
            .select([
                pl.col(DATE_COLUMN).min().alias("dmin"),
                pl.col(DATE_COLUMN).max().alias("dmax"),
            ])
            .collect(streaming=True)
            .row(0, named=True)
        )
        dmin = _to_date(result["dmin"])
        dmax = _to_date(result["dmax"])
        if dmin is None or dmax is None:
            return None
        return (dmin, dmax, "polars")
    except Exception:
        return None


def get_file_month_coverage(path: str) -> Dict[str, Any]:
    """
    return: {
        "path": str, "filename": str, "folder": str, "size_mb": float,
        "filename_guess_month": Optional[str],
        "min_date": Optional[date], "max_date": Optional[date],
        "months": List[str],          # 이 파일이 포함하는 모든 month
        "source": str,                # "metadata" | "polars" | "error"
        "error": Optional[str],
    }
    """
    rec = {
        "path": path,
        "filename": os.path.basename(path),
        "folder": os.path.basename(os.path.dirname(path)),
        "size_mb": round(os.path.getsize(path) / (1024 * 1024), 1),
        "filename_guess_month": filename_guess_month(path),
        "min_date": None,
        "max_date": None,
        "months": [],
        "source": "error",
        "error": None,
    }

    # 1) metadata 우선
    cov = file_month_coverage_via_metadata(path)
    # 2) fallback
    if cov is None:
        cov = file_month_coverage_via_polars(path)

    if cov is None:
        rec["error"] = "min/max를 읽지 못함 (metadata와 polars 모두 실패)"
        return rec

    dmin, dmax, source = cov
    rec["min_date"] = dmin
    rec["max_date"] = dmax
    rec["months"] = months_in_range(dmin, dmax)
    rec["source"] = source
    return rec


# =============================================================================
# 3. 누락/중복 분석
# =============================================================================

def analyze_coverage(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    return: {
        "all_months": [str],          # 데이터셋 전체 month 범위 (min~max)
        "covered_months": Set[str],   # 적어도 한 파일이 담당
        "missing_months": [str],      # 어느 파일에도 안 들어있음
        "month_to_files": Dict[str, List[str]],
        "duplicated_months": [str],   # 2개 이상 파일에 등장
    }
    """
    valid = [r for r in records if r["min_date"] is not None]
    if not valid:
        return {
            "all_months": [], "covered_months": set(), "missing_months": [],
            "month_to_files": {}, "duplicated_months": [],
        }

    overall_min = min(r["min_date"] for r in valid)
    overall_max = max(r["max_date"] for r in valid)
    all_months = months_in_range(overall_min, overall_max)

    month_to_files: Dict[str, List[str]] = {m: [] for m in all_months}
    for r in valid:
        for m in r["months"]:
            if m in month_to_files:
                month_to_files[m].append(r["filename"])
            else:
                # 범위 밖이면 추가
                month_to_files.setdefault(m, []).append(r["filename"])

    covered_months = {m for m, files in month_to_files.items() if files}
    missing_months = [m for m in all_months if m not in covered_months]
    duplicated_months = [m for m in all_months if len(month_to_files.get(m, [])) >= 2]

    return {
        "all_months": all_months,
        "covered_months": covered_months,
        "missing_months": missing_months,
        "month_to_files": month_to_files,
        "duplicated_months": duplicated_months,
    }


# =============================================================================
# 4. 출력 (콘솔 + txt + csv)
# =============================================================================

def print_console_summary(records: List[Dict[str, Any]], analysis: Dict[str, Any]) -> None:
    print("\n" + "=" * 70)
    print("[월 커버리지 분석 결과]")
    print("=" * 70)
    n_total = len(records)
    n_ok = sum(1 for r in records if r["error"] is None)
    n_err = n_total - n_ok
    print(f"  총 파일 수      : {n_total}")
    print(f"  분석 성공       : {n_ok}")
    print(f"  분석 실패       : {n_err}")

    if not analysis["all_months"]:
        print("\n  [분석 가능한 데이터 없음]")
        return

    print(f"\n  데이터셋 month 범위 : {analysis['all_months'][0]} ~ {analysis['all_months'][-1]}")
    print(f"  전체 month 수       : {len(analysis['all_months'])}")
    print(f"  커버된 month 수     : {len(analysis['covered_months'])}")
    print(f"  누락 month 수       : {len(analysis['missing_months'])}")
    print(f"  중복 month 수       : {len(analysis['duplicated_months'])}")

    if analysis["missing_months"]:
        print(f"\n  ⚠️ 누락된 month:")
        for m in analysis["missing_months"]:
            print(f"    - {m}")
    else:
        print(f"\n  ✅ 누락된 month 없음")

    if analysis["duplicated_months"]:
        print(f"\n  ℹ️ 여러 파일에 걸친 month (분할 또는 비정상 중복 가능성):")
        for m in analysis["duplicated_months"]:
            files = analysis["month_to_files"][m]
            print(f"    - {m}: {len(files)}개 파일 → {files}")

    if n_err > 0:
        print(f"\n  ⚠️ 분석 실패 파일:")
        for r in records:
            if r["error"]:
                print(f"    - {r['folder']}/{r['filename']}: {r['error']}")


def write_text_report(
    records: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    out_path: str,
) -> None:
    lines = []
    add = lines.append
    add("=" * 75)
    add(f"[월 커버리지 분석 리포트]  ({COUNTRY})")
    add(f"생성 시각 : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    add(f"검색 루트 : {BASE_DIR}")
    add("=" * 75)
    add("")

    add("[A] 요약")
    add(f"  총 파일 수 : {len(records)}")
    add(f"  분석 성공  : {sum(1 for r in records if r['error'] is None)}")
    add(f"  분석 실패  : {sum(1 for r in records if r['error'])}")
    if analysis["all_months"]:
        add(f"  month 범위 : {analysis['all_months'][0]} ~ {analysis['all_months'][-1]}")
        add(f"  전체 month : {len(analysis['all_months'])}")
        add(f"  커버 month : {len(analysis['covered_months'])}")
        add(f"  누락 month : {len(analysis['missing_months'])}")
        add(f"  중복 month : {len(analysis['duplicated_months'])}")
    add("")

    add("[B] 누락된 month")
    if analysis["missing_months"]:
        for m in analysis["missing_months"]:
            add(f"  - {m}")
    else:
        add("  (없음)")
    add("")

    add("[C] 여러 파일에 걸친 month")
    if analysis["duplicated_months"]:
        for m in analysis["duplicated_months"]:
            files = analysis["month_to_files"][m]
            add(f"  - {m}: {len(files)}개")
            for fn in files:
                add(f"      └ {fn}")
    else:
        add("  (없음)")
    add("")

    add("[D] 파일별 커버리지")
    add(f"  {'folder':<12} {'filename':<35} {'guess':<9} {'min':<11} {'max':<11} {'months':<8} {'src':<9} {'size_MB':>9}")
    add("  " + "-" * 100)
    for r in sorted(records, key=lambda x: (x["folder"], x["filename"])):
        if r["error"]:
            add(f"  {r['folder']:<12} {r['filename']:<35} [ERROR: {r['error']}]")
            continue
        n_months = len(r["months"])
        add(
            f"  {r['folder']:<12} {r['filename']:<35} "
            f"{(r['filename_guess_month'] or '-'):<9} "
            f"{str(r['min_date']):<11} {str(r['max_date']):<11} "
            f"{n_months:<8} {r['source']:<9} {r['size_mb']:>9,.1f}"
        )
    add("")

    add("[E] month별 담당 파일")
    if analysis["all_months"]:
        for m in analysis["all_months"]:
            files = analysis["month_to_files"].get(m, [])
            if not files:
                add(f"  {m}: ⚠️ 누락")
            elif len(files) == 1:
                add(f"  {m}: {files[0]}")
            else:
                add(f"  {m}: {len(files)}개")
                for fn in files:
                    add(f"      └ {fn}")
    add("")

    add("=" * 75)
    add("[리포트 끝]")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_matrix_csv(
    records: List[Dict[str, Any]],
    analysis: Dict[str, Any],
    out_path: str,
) -> None:
    """파일 × month 매트릭스 CSV. 각 셀은 1(포함) 또는 0(미포함)."""
    months = analysis["all_months"]
    sorted_records = sorted(
        [r for r in records if r["error"] is None],
        key=lambda x: (x["folder"], x["filename"]),
    )
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        header = ["folder", "filename", "min_date", "max_date", "size_MB"] + months
        w.writerow(header)
        for r in sorted_records:
            row = [
                r["folder"],
                r["filename"],
                str(r["min_date"]) if r["min_date"] else "",
                str(r["max_date"]) if r["max_date"] else "",
                f"{r['size_mb']:.1f}",
            ]
            covered = set(r["months"])
            row.extend(["1" if m in covered else "" for m in months])
            w.writerow(row)


# =============================================================================
# 5. 실행부
# =============================================================================

def main():
    print("=" * 70)
    print("[월 커버리지 검사]")
    print("=" * 70)
    print(f"  검색 루트 : {BASE_DIR}")
    print(f"  나라코드  : {COUNTRY}")
    print(f"  Date 컬럼 : {DATE_COLUMN}")
    print(f"  제외 키워드: {EXCLUDE_FOLDER_KEYWORDS}")
    print("=" * 70)

    print("\n[1] 파일 수집")
    files = collect_country_files(BASE_DIR, COUNTRY)
    if not files:
        print("  처리할 parquet 파일이 없습니다.")
        sys.exit(1)

    print("\n[2] 파일별 month 커버리지 추출")
    records: List[Dict[str, Any]] = []
    for i, (folder, fpath) in enumerate(files, 1):
        rec = get_file_month_coverage(fpath)
        if rec["error"]:
            tag = "ERROR"
        elif rec["source"] == "metadata":
            tag = "meta "
        else:
            tag = "poll "
        date_range = (
            f"{rec['min_date']} ~ {rec['max_date']}"
            if rec["min_date"] else f"[{rec['error']}]"
        )
        print(f"  [{i:>3}/{len(files)}] [{tag}] {folder}/{rec['filename']}  ({date_range})")
        records.append(rec)

    print("\n[3] 누락/중복 분석")
    analysis = analyze_coverage(records)

    print_console_summary(records, analysis)

    # 리포트 저장
    report_dir = REPORT_DIR or BASE_DIR
    os.makedirs(report_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_path = os.path.join(report_dir, f"coverage_report_{COUNTRY}_{ts}.txt")
    csv_path = os.path.join(report_dir, f"coverage_matrix_{COUNTRY}_{ts}.csv")
    write_text_report(records, analysis, txt_path)
    write_matrix_csv(records, analysis, csv_path)

    print(f"\n[리포트 저장]")
    print(f"  상세 리포트  : {txt_path}")
    print(f"  매트릭스 CSV : {csv_path}")

    # exit code
    n_err = sum(1 for r in records if r["error"])
    if analysis["missing_months"]:
        sys.exit(2)  # 누락 month 있음
    if n_err > 0:
        sys.exit(3)  # 분석 실패 파일 있음
    sys.exit(0)


if __name__ == "__main__":
    main()