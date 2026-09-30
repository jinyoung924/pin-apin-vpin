"""
=============================================================================
[검증 스크립트] 분할된 parquet들이 vpin 계산 코드를 돌리기에 안전한지 점검
=============================================================================

검증 항목 (각각 PASS/WARN/FAIL 로 보고)

A. 파일 단위 검증
  A1. 파일이 polars로 열림 (glob 패턴 오해 없이) — 사고난 _[1-2] 케이스 재발 방지
  A2. vpin 필수 컬럼 5개 존재: Date, Time, Symbol, Price, Volume
  A3. Date/Time dtype이 datetime/time 계열로 변환 가능
  A4. row 수가 0 보다 큼
  A5. Price/Volume의 NULL 비율이 비정상적으로 높지 않음
  A6. Symbol이 1개 이상 존재
  A7. 파일 안에서 같은 Symbol 안의 (Date,Time)이 시간 역행 없는지 확인 (정렬되어 있어야 hot path)

B. 분할 짝(pair) 검증 — *_1of2.parquet 과 *_2of2.parquet 짝맞춤
  B1. 두 짝이 모두 존재
  B2. 두 짝의 스키마 일치
  B3. 두 짝 사이에 Symbol 교집합 없음 (분할 무결성 핵심)
  B4. 시간 경계: part1의 max(Datetime) <= part2의 min(Datetime)  (or 종목 분리상 자연스러움)

C. 원본 대조 검증 (옵션, _original_backup 폴더가 있을 때)
  C1. 원본 row 수 == 자른 조각들 row 수 합
  C2. 원본 Symbol 집합 == 자른 조각들 Symbol 집합 합

D. 전체 입력 셋 검증
  D1. vpin이 보는 입력 파일 목록 (sorted glob)이 시간 순서대로 잘 정렬되는지 샘플 점검
  D2. 같은 종목이 너무 많은 파일에 흩어져 있지 않은지 (참고용 통계)
=============================================================================
"""

import os
import re
import sys
import glob
import time
import traceback
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Set

import polars as pl
import pyarrow.parquet as pq


# =============================================================================
# USER CONFIG
# =============================================================================

# 검증할 폴더들 (재귀 탐색. _original_backup은 비교 대상으로만 사용)
TARGET_FOLDERS: List[str] = [
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2015",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2016",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2019",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2020",
]

BACKUP_DIR_NAME = "_original_backup"

# vpin 코드가 요구하는 필수 컬럼
REQUIRED_COLUMNS = ["Date", "Time", "Symbol", "Price", "Volume"]

# 시간 역행 검사용 샘플 크기 (전수 검사하면 너무 느려서 종목별 head를 본다)
TIME_MONOTONIC_SAMPLE_HEAD = 100_000

# NULL 비율 경고 임계값
NULL_RATIO_WARN_THRESHOLD = 0.01  # 1% 초과면 WARN

# 분할 짝 매칭용 정규식
_PART_RE = re.compile(r"^(.+)_(1of2|2of2)\.parquet$", re.IGNORECASE)


# =============================================================================
# 결과 누적
# =============================================================================

class Report:
    def __init__(self):
        self.passes: int = 0
        self.warns: List[Tuple[str, str]] = []   # (scope, message)
        self.fails: List[Tuple[str, str]] = []   # (scope, message)
        self.infos: List[Tuple[str, str]] = []   # (scope, message)

    def ok(self, scope: str, msg: str = ""):
        self.passes += 1
        # PASS는 화면이 너무 길어져서 요약에서만 카운트

    def warn(self, scope: str, msg: str):
        self.warns.append((scope, msg))
        print(f"  ⚠️  WARN [{scope}] {msg}")

    def fail(self, scope: str, msg: str):
        self.fails.append((scope, msg))
        print(f"  ❌ FAIL [{scope}] {msg}")

    def info(self, scope: str, msg: str):
        self.infos.append((scope, msg))


# =============================================================================
# 파일 수집
# =============================================================================

def collect_parquet_files(folders: List[str], include_backup: bool) -> Tuple[List[str], Dict[str, str]]:
    """
    folders 안의 *.parquet 재귀 수집.
    Returns:
        files: 메인 검증 대상 파일 (백업 제외)
        backup_map: {원본파일명: 백업파일경로}  (백업 폴더 안 파일)
    """
    files: List[str] = []
    backup_map: Dict[str, str] = {}
    seen: Set[str] = set()

    for folder in folders:
        folder_abs = os.path.abspath(folder)
        if not os.path.isdir(folder_abs):
            print(f"  [Warn] 폴더 없음 (건너뜀): {folder_abs}")
            continue

        for root, dirs, fns in os.walk(folder_abs):
            in_backup = (os.path.basename(root) == BACKUP_DIR_NAME)
            if in_backup and not include_backup:
                dirs[:] = []
                continue

            for fn in fns:
                if not fn.lower().endswith(".parquet"):
                    continue
                full = os.path.abspath(os.path.join(root, fn))
                if full in seen:
                    continue
                seen.add(full)
                if in_backup:
                    backup_map[fn] = full
                else:
                    files.append(full)

    files.sort()
    return files, backup_map


# =============================================================================
# A. 파일 단위 검증
# =============================================================================

def verify_single_file(path: str, report: Report) -> Dict[str, Any]:
    """
    개별 parquet에 대한 검증.
    Returns: {ok: bool, n_rows, symbols, ...}
    """
    name = os.path.basename(path)
    info: Dict[str, Any] = {"name": name, "path": path, "ok": False}

    # A1. polars로 glob 오해 없이 열리는가
    try:
        # glob=False로 명시해서 [1-2] 같은 문자가 들어가도 안전하게 열리는지 검사하는 게 아니라,
        # 우리는 "이름변경이 끝났다"는 가정이므로 기본값으로 열어본다. 그게 vpin 코드의 호출 방식.
        lf = pl.scan_parquet(path)
        schema = lf.collect_schema()
    except Exception as e:
        msg = str(e).splitlines()[0]
        report.fail(name, f"polars로 열기 실패: {msg}")
        if "expanded paths were empty" in str(e) or "glob" in str(e).lower():
            report.fail(name, "→ 파일명에 glob 특수문자([, ], ?, * 등)가 남아있을 가능성. 이름변경 스크립트를 다시 돌려주세요.")
        return info

    columns = list(schema.names())
    info["columns"] = columns

    # A2. 필수 컬럼
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        report.fail(name, f"필수 컬럼 누락: {missing}")
        return info

    # A3. dtype 검증
    try:
        date_dtype = schema["Date"]
        time_dtype = schema["Time"]
        # polars의 datetime/date/time 계열 dtype 인지 확인
        date_ok = date_dtype in (pl.Date, pl.Datetime) or "date" in str(date_dtype).lower() or "datetime" in str(date_dtype).lower()
        time_ok = time_dtype == pl.Time or "time" in str(time_dtype).lower()
        if not date_ok:
            report.fail(name, f"Date 컬럼 dtype 의심: {date_dtype}")
            return info
        if not time_ok:
            report.fail(name, f"Time 컬럼 dtype 의심: {time_dtype}")
            return info
    except Exception as e:
        report.fail(name, f"dtype 확인 실패: {e}")
        return info

    # A4. row 수 (pyarrow가 metadata만 보고 빠름)
    try:
        n_rows = pq.ParquetFile(path).metadata.num_rows
    except Exception as e:
        report.fail(name, f"pyarrow로 metadata 읽기 실패: {e}")
        return info

    info["n_rows"] = n_rows
    if n_rows == 0:
        report.fail(name, "row 수가 0")
        return info

    # A5, A6, A7: 실데이터 확인 (스트리밍으로 가볍게)
    try:
        stats = (
            pl.scan_parquet(path)
            .select([
                pl.len().alias("n"),
                pl.col("Price").null_count().alias("price_null"),
                pl.col("Volume").null_count().alias("volume_null"),
                pl.col("Symbol").null_count().alias("symbol_null"),
                pl.col("Symbol").n_unique().alias("symbol_unique"),
            ])
            .collect(streaming=True)
            .row(0, named=True)
        )
    except Exception as e:
        report.fail(name, f"통계 계산 실패: {e}")
        return info

    info.update(stats)

    # A5. NULL 비율
    if n_rows > 0:
        price_ratio = stats["price_null"] / n_rows
        volume_ratio = stats["volume_null"] / n_rows
        symbol_ratio = stats["symbol_null"] / n_rows
        if price_ratio > NULL_RATIO_WARN_THRESHOLD:
            report.warn(name, f"Price NULL 비율 {price_ratio:.3%} (>{NULL_RATIO_WARN_THRESHOLD:.0%})")
        if volume_ratio > NULL_RATIO_WARN_THRESHOLD:
            report.warn(name, f"Volume NULL 비율 {volume_ratio:.3%} (>{NULL_RATIO_WARN_THRESHOLD:.0%})")
        if symbol_ratio > 0:
            report.fail(name, f"Symbol NULL이 {stats['symbol_null']:,}건 발견")
            return info

    # A6. Symbol unique
    if stats["symbol_unique"] == 0:
        report.fail(name, "Symbol unique = 0")
        return info

    # A7. 시간 역행 (같은 Symbol 안에서 TickDatetime이 단조증가하는지 head 샘플로 확인)
    try:
        # TickDatetime = Date + Time
        head = (
            pl.scan_parquet(path)
            .select(["Date", "Time", "Symbol"])
            .head(TIME_MONOTONIC_SAMPLE_HEAD)
            .with_columns(
                pl.col("Date").dt.combine(pl.col("Time")).alias("TickDt")
            )
            .with_columns(
                # 같은 Symbol 안에서 직전 TickDt
                pl.col("TickDt").shift(1).over("Symbol").alias("prev_dt")
            )
            .with_columns(
                # 직전이 더 큰 경우 = 시간 역행
                (pl.col("TickDt") < pl.col("prev_dt")).fill_null(False).alias("is_regress")
            )
            .select(pl.col("is_regress").sum().alias("n_regress"))
            .collect(streaming=True)
            .row(0, named=True)
        )
        n_regress = int(head["n_regress"])
        if n_regress > 0:
            # vpin Step 1이 어차피 재정렬하긴 하지만 정보로 알려준다
            report.warn(
                name,
                f"head {TIME_MONOTONIC_SAMPLE_HEAD:,} rows 중 시간 역행 {n_regress:,}건 "
                f"(Step 1이 재정렬하므로 vpin 결과에는 무해)"
            )
    except Exception as e:
        report.warn(name, f"시간 단조 검사 실패: {e}")

    info["ok"] = True
    return info


# =============================================================================
# B. 분할 짝(pair) 검증
# =============================================================================

def pair_files(files: List[str]) -> Tuple[Dict[str, Tuple[str, Optional[str]]], List[str]]:
    """
    *_1of2.parquet 과 *_2of2.parquet 을 짝지음.
    Returns:
        pairs: {base_stem: (part1_path, part2_path or None)}
        unpaired_singles: 짝이 아닌 단독 파일들 (자르지 않은 원본도 여기 들어감)
    """
    by_stem: Dict[str, Dict[str, str]] = defaultdict(dict)
    singles: List[str] = []

    for path in files:
        fn = os.path.basename(path)
        stem = os.path.splitext(fn)[0]
        m = _PART_RE.match(fn)
        if m:
            base, part = m.group(1), m.group(2).lower()
            by_stem[base][part] = path
        else:
            singles.append(path)

    pairs: Dict[str, Tuple[str, Optional[str]]] = {}
    for base, parts in by_stem.items():
        pairs[base] = (parts.get("1of2"), parts.get("2of2"))
    return pairs, singles


def verify_pair(base: str, p1: Optional[str], p2: Optional[str], report: Report) -> None:
    scope = f"PAIR:{base}"

    # B1. 두 짝 존재
    if p1 is None:
        report.fail(scope, f"_1of2 누락 (짝: {os.path.basename(p2) if p2 else '?'})")
        return
    if p2 is None:
        report.fail(scope, f"_2of2 누락 (짝: {os.path.basename(p1)})")
        return

    # B2. 스키마 일치 (이름과 dtype)
    try:
        s1 = pl.scan_parquet(p1).collect_schema()
        s2 = pl.scan_parquet(p2).collect_schema()
        d1 = {n: str(s1[n]) for n in s1.names()}
        d2 = {n: str(s2[n]) for n in s2.names()}
        if d1 != d2:
            diff_keys = set(d1.keys()) ^ set(d2.keys())
            if diff_keys:
                report.fail(scope, f"컬럼 집합 불일치: 차이={diff_keys}")
            else:
                mismatches = [n for n in d1 if d1[n] != d2[n]]
                report.fail(scope, f"dtype 불일치: {mismatches}")
            return
    except Exception as e:
        report.fail(scope, f"스키마 비교 실패: {e}")
        return

    # B3, B4: 종목 분리 및 시간 경계
    try:
        def gather(path: str) -> Dict[str, Any]:
            # Symbol 집합 (unique는 행으로 펼쳐지므로 별도 쿼리)
            sym_df = (
                pl.scan_parquet(path)
                .select(pl.col("Symbol").unique())
                .collect(streaming=True)
            )
            symbols = set(sym_df.to_series().to_list())

            # 시간 최소/최대 (1행짜리 결과)
            rng = (
                pl.scan_parquet(path)
                .with_columns(
                    pl.col("Date").dt.combine(pl.col("Time")).alias("TickDt")
                )
                .select([
                    pl.col("TickDt").min().alias("min_dt"),
                    pl.col("TickDt").max().alias("max_dt"),
                ])
                .collect(streaming=True)
                .row(0, named=True)
            )
            return {
                "symbols": symbols,
                "min_dt": rng["min_dt"],
                "max_dt": rng["max_dt"],
            }

        g1 = gather(p1)
        g2 = gather(p2)
    except Exception as e:
        report.fail(scope, f"분할 무결성 데이터 수집 실패: {e}")
        return

    # B3. 종목 교집합 없음
    overlap = g1["symbols"] & g2["symbols"]
    if overlap:
        sample = sorted(overlap)[:5]
        report.fail(
            scope,
            f"두 part에 같은 종목이 존재 (분할 무결성 위반). "
            f"교집합 {len(overlap)}개, 예: {sample}"
        )
    else:
        # 통계는 info로
        n1, n2 = len(g1["symbols"]), len(g2["symbols"])
        report.info(scope, f"종목 분리 OK: part1={n1}, part2={n2}, 합={n1+n2}")

    # B4. 시간 경계 (part1.max <= part2.min) — 종목이 시간순 정렬돼 있다는 가정 하의 sanity check.
    # 분할은 종목 경계 기준이라 같은 시간이 양쪽에 흩어질 수도 있고, 시간 역행이 발생하더라도
    # vpin 입장에서는 종목별로 다시 모이므로 무해. 따라서 WARN으로만 처리.
    if g1["max_dt"] is not None and g2["min_dt"] is not None:
        if g1["max_dt"] > g2["min_dt"]:
            report.warn(
                scope,
                f"시간 경계 비단조: part1.max={g1['max_dt']}, part2.min={g2['min_dt']} "
                f"(분할이 종목 경계 기준이라 정상일 수 있음. vpin 결과엔 무해)"
            )
        else:
            report.info(
                scope,
                f"시간 경계 OK: part1 [{g1['min_dt']} ~ {g1['max_dt']}] → "
                f"part2 [{g2['min_dt']} ~ {g2['max_dt']}]"
            )


# =============================================================================
# C. 원본 대조 검증
# =============================================================================

def verify_against_backup(
    pairs: Dict[str, Tuple[Optional[str], Optional[str]]],
    backup_map: Dict[str, str],
    report: Report,
) -> None:
    """
    pairs 각각에 대해 원본(_original_backup 안)이 있으면 row 수/Symbol 집합 비교.
    """
    for base, (p1, p2) in pairs.items():
        if p1 is None or p2 is None:
            continue
        original_name = f"{base}.parquet"
        backup_path = backup_map.get(original_name)
        if backup_path is None:
            # 백업이 없을 수 있음 (이미 한 번 더 자른 경우 등). info만.
            continue
        scope = f"BACKUP:{base}"

        try:
            orig_meta = pq.ParquetFile(backup_path).metadata
            orig_rows = orig_meta.num_rows
            p1_rows = pq.ParquetFile(p1).metadata.num_rows
            p2_rows = pq.ParquetFile(p2).metadata.num_rows
        except Exception as e:
            report.fail(scope, f"row 수 조회 실패: {e}")
            continue

        # C1. row 수 일치
        if orig_rows != p1_rows + p2_rows:
            report.fail(
                scope,
                f"row 수 불일치: 원본={orig_rows:,}, part1+part2={p1_rows+p2_rows:,} "
                f"(차이 {orig_rows - (p1_rows+p2_rows):+,})"
            )
        else:
            report.info(scope, f"row 수 일치: 원본={orig_rows:,} == {p1_rows:,}+{p2_rows:,}")

        # C2. Symbol 집합 일치
        try:
            orig_syms = set(
                pl.scan_parquet(backup_path)
                .select(pl.col("Symbol").unique())
                .collect(streaming=True)
                .to_series()
                .to_list()
            )
            split_syms = set(
                pl.scan_parquet([p1, p2])
                .select(pl.col("Symbol").unique())
                .collect(streaming=True)
                .to_series()
                .to_list()
            )
        except Exception as e:
            report.fail(scope, f"Symbol 집합 수집 실패: {e}")
            continue

        missing = orig_syms - split_syms
        extra = split_syms - orig_syms
        if missing or extra:
            details = []
            if missing:
                details.append(f"누락 {len(missing)}개 예: {sorted(missing)[:5]}")
            if extra:
                details.append(f"추가 {len(extra)}개 예: {sorted(extra)[:5]}")
            report.fail(scope, f"Symbol 집합 불일치: {' / '.join(details)}")
        else:
            report.info(scope, f"Symbol 집합 일치: {len(orig_syms)}개")


# =============================================================================
# D. 전체 입력 셋 검증 (vpin 코드 관점)
# =============================================================================

def verify_input_set(files: List[str], report: Report) -> None:
    """
    vpin 코드의 get_parquet_files()가 보는 것과 같은 'sorted glob' 결과를 시뮬레이션하고,
    그 정렬이 시간 순서를 깨지 않는지 샘플 확인.
    """
    # D1. 정렬된 입력에서 파일별 시간 범위를 뽑아 보고, 인접 파일의 범위가 거꾸로 가지 않는지 확인.
    print("\n  [D1] 입력 파일 정렬 순서 vs 시간 순서 점검...")
    ranges: List[Tuple[str, Any, Any]] = []
    for path in files:
        try:
            r = (
                pl.scan_parquet(path)
                .with_columns(pl.col("Date").dt.combine(pl.col("Time")).alias("TickDt"))
                .select([
                    pl.col("TickDt").min().alias("mn"),
                    pl.col("TickDt").max().alias("mx"),
                ])
                .collect(streaming=True)
                .row(0, named=True)
            )
            ranges.append((os.path.basename(path), r["mn"], r["mx"]))
        except Exception as e:
            report.warn(f"INPUT_SET", f"{os.path.basename(path)} 범위 추출 실패: {e}")

    inversions = 0
    for i in range(len(ranges) - 1):
        _, mn_a, mx_a = ranges[i]
        name_b, mn_b, _ = ranges[i + 1]
        if mx_a is None or mn_b is None:
            continue
        if mx_a > mn_b:
            inversions += 1
            if inversions <= 5:
                report.info(
                    "INPUT_SET",
                    f"인접 파일 시간 역행: {ranges[i][0]} (max={mx_a}) → {name_b} (min={mn_b})"
                )
    if inversions == 0:
        report.info("INPUT_SET", f"인접 파일 시간 순서 OK (총 {len(ranges)}개 파일)")
    else:
        # 시간 역행이 있어도 vpin은 종목별로 모으므로 무해. WARN으로.
        report.warn(
            "INPUT_SET",
            f"인접 파일 시간 역행 {inversions}건 발견. "
            f"vpin 결과에는 무해 (종목별 체크포인트에서 재정렬됨). "
            f"단, 분할 part 파일들이 동시에 존재하면 자연스럽게 생길 수 있음."
        )

    # D2. 같은 종목이 너무 많은 파일에 흩어져 있지 않은지 (참고 통계)
    print("\n  [D2] 종목별 파일 산재 통계 수집 중...")
    sym_to_files: Dict[str, int] = defaultdict(int)
    for path in files:
        try:
            syms = (
                pl.scan_parquet(path)
                .select(pl.col("Symbol").unique())
                .collect(streaming=True)
                .to_series()
                .to_list()
            )
            for s in syms:
                sym_to_files[s] += 1
        except Exception:
            continue

    if sym_to_files:
        counts = list(sym_to_files.values())
        avg = sum(counts) / len(counts)
        mx = max(counts)
        report.info(
            "INPUT_SET",
            f"종목 수={len(sym_to_files):,}, 종목별 평균 파일 수={avg:.1f}, 최대={mx}"
        )


# =============================================================================
# 메인
# =============================================================================

def main():
    print("=" * 72)
    print("[VPIN 입력 검증] 분할 후 안전성 점검")
    print("=" * 72)
    print(f"  검증 폴더 수      : {len(TARGET_FOLDERS)}")
    for i, fd in enumerate(TARGET_FOLDERS, 1):
        print(f"    {i:>2}. {fd}")
    print(f"  필수 컬럼         : {REQUIRED_COLUMNS}")
    print(f"  시간 단조 sample  : {TIME_MONOTONIC_SAMPLE_HEAD:,} rows / file")
    print(f"  NULL 경고 임계값  : {NULL_RATIO_WARN_THRESHOLD:.0%}")
    print("=" * 72)

    report = Report()
    t0 = time.time()

    # 0) 파일 수집 (메인 + 백업 분리)
    print("\n[0] 파일 수집...")
    files, backup_map = collect_parquet_files(TARGET_FOLDERS, include_backup=True)
    print(f"  → 검증 대상 (메인) : {len(files)}개")
    print(f"  → 백업 폴더 파일   : {len(backup_map)}개  (대조용으로만 사용, 검증 안 함)")
    if not files:
        print("\n[Info] 검증할 파일이 없습니다. 종료.")
        return

    # 1) A. 파일 단위
    print("\n" + "=" * 72)
    print("[A] 파일 단위 검증")
    print("=" * 72)
    file_infos: Dict[str, Dict[str, Any]] = {}
    for i, path in enumerate(files, 1):
        if i % 20 == 0 or i == 1 or i == len(files):
            print(f"  ... [{i}/{len(files)}] {os.path.basename(path)}")
        try:
            info = verify_single_file(path, report)
            file_infos[path] = info
        except Exception as e:
            report.fail(os.path.basename(path), f"검증 중 예외: {e}")
            for line in traceback.format_exc().splitlines()[:5]:
                print(f"      {line}")

    # 2) B. 분할 짝
    print("\n" + "=" * 72)
    print("[B] 분할 짝 검증 (*_1of2 vs *_2of2)")
    print("=" * 72)
    pairs, singles = pair_files(files)
    print(f"  → 짝(pair) 수      : {len(pairs)}")
    print(f"  → 단독 파일 수     : {len(singles)}  (자르지 않은 원본 등)")
    for base in sorted(pairs.keys()):
        p1, p2 = pairs[base]
        verify_pair(base, p1, p2, report)

    # 3) C. 원본 대조
    print("\n" + "=" * 72)
    print("[C] 원본 대조 검증 (백업 폴더가 있을 때만)")
    print("=" * 72)
    if backup_map:
        verify_against_backup(pairs, backup_map, report)
    else:
        print("  (백업 폴더 없음 — 스킵)")

    # 4) D. 전체 입력 셋
    print("\n" + "=" * 72)
    print("[D] 전체 입력 셋 검증 (vpin이 보는 관점)")
    print("=" * 72)
    verify_input_set(files, report)

    elapsed = time.time() - t0

    # 5) 요약
    print("\n" + "=" * 72)
    print("[검증 요약]")
    print("=" * 72)
    print(f"  소요 시간       : {elapsed:.1f}s")
    print(f"  파일 수         : {len(files)}")
    print(f"  분할 짝         : {len(pairs)}")
    print(f"  ✅ PASS         : {report.passes}")
    print(f"  ⚠️  WARN         : {len(report.warns)}")
    print(f"  ❌ FAIL         : {len(report.fails)}")

    if report.warns:
        print(f"\n  [WARN 상세]")
        for scope, msg in report.warns:
            print(f"    ⚠️  [{scope}] {msg}")

    if report.fails:
        print(f"\n  [FAIL 상세]")
        for scope, msg in report.fails:
            print(f"    ❌ [{scope}] {msg}")
        print(f"\n  → vpin 코드를 돌리기 전에 FAIL을 모두 해결하세요.")
        sys.exit(1)
    elif report.warns:
        print(f"\n  → 치명적 문제 없음. WARN은 정보성 (vpin 결과에는 무해할 가능성 높음).")
        print(f"     내용을 확인한 뒤 vpin 코드를 돌려도 됩니다.")
    else:
        print(f"\n  → ✅ 모든 검증 통과. vpin 코드를 안전하게 돌릴 수 있습니다.")


if __name__ == "__main__":
    main()