"""
=============================================================================
[사전처리] 큰 월별 tick parquet들을 종목 경계 기준으로 정확히 2개 part로 분할
            (폴더 목록 처리 + 점진 확장 검색 버전)

이번 버전의 변경점
  · TARGET_FILENAMES (파일 리스트) → TARGET_FOLDERS (폴더 리스트)
    지정한 폴더 안에 있는 *.parquet 파일을 모두 자동으로 찾아 처리한다.
  · 이미 한 번 쪼개진 *_part1.parquet, *_part2.parquet 도 다시 반으로 자른다.
    또한 이번 버전에서 만들어지는 *_[1/2].parquet, *_[2/2].parquet 도
    다음 회차에 다시 입력이 될 수 있다.
  · _original_backup 폴더 안에 들어있는 원본 백업 파일은 절대 자르지 않는다.
    (재귀 탐색 시 _original_backup 디렉터리는 스킵)
  · 분할 결과 파일명 규칙
      [원래파일명]_[1-2].parquet
      [원래파일명]_[2-2].parquet
    (파일명에 슬래시(/)는 OS가 폴더 구분자로 해석해 쓸 수 없어 하이픈으로 표기)
    예) cn_201603.parquet
          → cn_201603_[1-2].parquet
          → cn_201603_[2-2].parquet
        cn_201603_part1.parquet
          → cn_201603_part1_[1-2].parquet
          → cn_201603_part1_[2-2].parquet
        cn_201603_part1_[1-2].parquet
          → cn_201603_part1_[1-2]_[1-2].parquet
          → cn_201603_part1_[1-2]_[2-2].parquet

배경
  · 매우 큰 파일(예: 14억 row, 1,400+ row groups)에서는 한 종목이 다수의
    row group에 걸쳐 있을 수 있다.
  · 검색 범위를 점진적으로 확장해서 가장 가까운 종목 경계를 찾는다.
  · 절반 비율은 양보하더라도 종목 경계는 절대 깨지 않는다.

분할 비율 허용 범위
  · MIN_SPLIT_RATIO ~ MAX_SPLIT_RATIO 사이의 경계만 유효
  · 이 범위 밖이면 단일 종목 파일에 가까운 것으로 보고 분할 포기
=============================================================================
"""

import os
import re
import sys
import glob
import shutil
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple

import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# USER CONFIG
# =============================================================================

# 이 폴더들 안에 들어있는 *.parquet 파일을 모두 찾아서 반으로 자른다.
# (재귀 탐색. 단, _original_backup 폴더는 건너뜀)
TARGET_FOLDERS: List[str] = [
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2016",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2017",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2018",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2019",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2020",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2021"

    # 새 폴더가 생기면 여기에 추가
]

# 이 이름의 폴더 안에 있는 파일은 절대 자르지 않는다 (원본 백업 디렉터리)
EXCLUDE_DIR_NAMES = {"_original_backup"}

# 백업 디렉터리 (None → 각 파일의 부모 폴더 아래 _original_backup 자동 생성)
BACKUP_DIR: Optional[str] = None

SYMBOL_COLUMN = "Symbol"
DRY_RUN = False
SKIP_IF_EXISTS = True
STOP_ON_ERROR = False

# --- 분할 비율 허용 범위 ---
MIN_SPLIT_RATIO = 0.30
MAX_SPLIT_RATIO = 0.70

# --- 검색 윈도우 점진 확장 한도 ---
SEARCH_RADIUS_SCHEDULE = [1, 3, 7, 15, 31, 63, 127, 255]


# =============================================================================
# 1. 폴더에서 자를 대상 parquet 파일 수집
# =============================================================================

# 이번 회차에서 새로 생성되는 분할 결과물의 접미사 패턴.
# 한 회차 안에서 방금 만든 산출물을 또 입력으로 잡는 사고를 막기 위해 인식만 해둔다.
# (탐색 시점에 이미 존재하던 파일만 입력으로 받고, 진행 도중 새로 생기는 파일은 무시)
_NEW_SUFFIX_RE = re.compile(r"_\[[12]-2\]\.parquet$", re.IGNORECASE)


def _should_skip_dir(dirname: str) -> bool:
    return dirname in EXCLUDE_DIR_NAMES


def collect_parquet_files(folders: List[str]) -> List[str]:
    """
    folders 안의 *.parquet 파일을 재귀적으로 모두 모은다.
    단, 경로 어딘가에 EXCLUDE_DIR_NAMES(예: _original_backup)가 들어있는 파일은 제외.
    중복은 절대경로 기준으로 제거.
    """
    collected: List[str] = []
    seen: set = set()

    for folder in folders:
        folder_abs = os.path.abspath(folder)
        if not os.path.isdir(folder_abs):
            print(f"  [Warn] 폴더가 존재하지 않습니다 (건너뜀): {folder_abs}")
            continue

        for root, dirs, files in os.walk(folder_abs):
            # _original_backup 같은 폴더는 더 안 내려감
            dirs[:] = [d for d in dirs if not _should_skip_dir(d)]

            for fn in files:
                if not fn.lower().endswith(".parquet"):
                    continue
                full = os.path.join(root, fn)
                full_abs = os.path.abspath(full)
                if full_abs in seen:
                    continue
                seen.add(full_abs)
                collected.append(full_abs)

    collected.sort()
    return collected


# =============================================================================
# 2. sniff
# =============================================================================

def sniff_symbol_ordering(path: str, n_samples: int = 5, sample_size: int = 200) -> bool:
    pf = pq.ParquetFile(path)
    n_rows = pf.metadata.num_rows
    n_row_groups = pf.num_row_groups
    print(f"  [Sniff] 전체 row 수={n_rows:,}, row group 수={n_row_groups:,}")
    if n_rows < sample_size * n_samples or n_row_groups < n_samples:
        print(f"  [Sniff] 파일이 작아 sniff 생략 (안전 가정)")
        return True
    sample_rg_indices = sorted(set(
        int(i * (n_row_groups - 1) / (n_samples - 1)) for i in range(n_samples)
    ))
    all_safe = True
    for rg_idx in sample_rg_indices:
        tbl = pf.read_row_group(rg_idx, columns=[SYMBOL_COLUMN])
        symbols = tbl.column(SYMBOL_COLUMN).to_pylist()
        head = symbols[:sample_size]
        unique_count = len(set(head))
        ratio = unique_count / len(head) if head else 0.0
        verdict = "OK" if ratio < 0.1 else "WARN"
        if ratio >= 0.1:
            all_safe = False
        print(
            f"    rg #{rg_idx:>4}: head {len(head)} rows, "
            f"unique={unique_count}, ratio={ratio:.3f} [{verdict}]"
        )
    if all_safe:
        print(f"  [Sniff] OK (종목이 연속 블록으로 정렬됨)")
    else:
        print(f"  [Sniff] WARN (종목이 sample 안에서 많이 섞임)")
    return all_safe


# =============================================================================
# 3. row group 시작 오프셋 사전 계산 + 종목 경계 점진 확장 검색
# =============================================================================

def compute_rg_offsets(pf: pq.ParquetFile) -> List[int]:
    """row group i의 시작 row 인덱스를 미리 누적 계산."""
    offsets = [0]
    for i in range(pf.num_row_groups):
        offsets.append(offsets[-1] + pf.metadata.row_group(i).num_rows)
    return offsets  # length = num_row_groups + 1


def collect_boundaries_in_window(
    pf: pq.ParquetFile,
    rg_offsets: List[int],
    rg_lo: int,
    rg_hi: int,
    cache_symbols: Dict[int, List],
    cache_first_last: Dict[int, Tuple],
) -> List[int]:
    """
    [rg_lo, rg_hi] 범위의 row group들을 보면서, 'Symbol이 직전 row와 다른 row의
    global 인덱스'를 수집.
    """
    boundaries: List[int] = []

    def get_symbols_full(rg_idx: int) -> List:
        if rg_idx in cache_symbols:
            return cache_symbols[rg_idx]
        tbl = pf.read_row_group(rg_idx, columns=[SYMBOL_COLUMN])
        syms = tbl.column(SYMBOL_COLUMN).to_pylist()
        cache_symbols[rg_idx] = syms
        if syms:
            cache_first_last[rg_idx] = (syms[0], syms[-1])
        return syms

    def get_first_last(rg_idx: int) -> Tuple:
        if rg_idx in cache_first_last:
            return cache_first_last[rg_idx]
        syms = get_symbols_full(rg_idx)
        return cache_first_last[rg_idx] if syms else (None, None)

    # 1) 각 row group 내부의 경계
    for rg_idx in range(rg_lo, rg_hi + 1):
        syms = get_symbols_full(rg_idx)
        if not syms:
            continue
        base = rg_offsets[rg_idx]
        prev = syms[0]
        # 첫 row의 경계 처리
        if rg_idx > 0:
            _, prev_rg_last = get_first_last(rg_idx - 1)
            if prev_rg_last is not None and prev_rg_last != syms[0]:
                boundaries.append(base)
        elif rg_idx == 0:
            boundaries.append(0)  # 파일 맨 처음

        for i in range(1, len(syms)):
            if syms[i] != prev:
                boundaries.append(base + i)
                prev = syms[i]

    return boundaries


def find_midpoint_split_progressive(
    path: str,
    min_ratio: float = MIN_SPLIT_RATIO,
    max_ratio: float = MAX_SPLIT_RATIO,
    radius_schedule: Optional[List[int]] = None,
) -> Optional[int]:
    """
    target row (절반 지점) 주변에서 종목 경계를 점진 확장 검색.
    """
    if radius_schedule is None:
        radius_schedule = SEARCH_RADIUS_SCHEDULE

    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    n_rg = pf.num_row_groups
    if total_rows <= 1 or n_rg == 0:
        return None

    rg_offsets = compute_rg_offsets(pf)
    target = total_rows // 2
    min_idx = int(total_rows * min_ratio)
    max_idx = int(total_rows * max_ratio)

    center_rg = 0
    for i in range(n_rg):
        if rg_offsets[i] <= target < rg_offsets[i + 1]:
            center_rg = i
            break

    cache_symbols: Dict[int, List] = {}
    cache_first_last: Dict[int, Tuple] = {}
    seen_window: Optional[Tuple[int, int]] = None

    print(
        f"  [Split point] target row={target:,}, "
        f"허용 비율 범위=[{min_ratio:.2f}, {max_ratio:.2f}] "
        f"(row {min_idx:,} ~ {max_idx:,})"
    )

    for radius in radius_schedule:
        rg_lo = max(0, center_rg - radius)
        rg_hi = min(n_rg - 1, center_rg + radius)

        if seen_window == (rg_lo, rg_hi):
            # 윈도우가 더 안 커진 상황
            if rg_lo == 0 and rg_hi == n_rg - 1:
                break
            continue
        seen_window = (rg_lo, rg_hi)

        boundaries = collect_boundaries_in_window(
            pf, rg_offsets, rg_lo, rg_hi, cache_symbols, cache_first_last
        )

        valid = [b for b in boundaries if min_idx <= b <= max_idx and 0 < b < total_rows]

        n_loaded_rg = rg_hi - rg_lo + 1
        n_loaded_rows = rg_offsets[rg_hi + 1] - rg_offsets[rg_lo]
        print(
            f"    radius={radius:>3}: rg [{rg_lo:>4}..{rg_hi:>4}] "
            f"({n_loaded_rg} rgs, {n_loaded_rows:,} rows scanned), "
            f"boundaries found={len(boundaries)}, valid={len(valid)}"
        )

        if valid:
            split_idx = min(valid, key=lambda x: abs(x - target))
            ratio_l = split_idx / total_rows * 100
            ratio_r = 100 - ratio_l
            print(
                f"  [Split point] 선택됨: row {split_idx:,} / {total_rows:,} "
                f"({ratio_l:.1f}% : {ratio_r:.1f}%)"
            )
            return split_idx

        if rg_lo == 0 and rg_hi == n_rg - 1:
            break

    print(
        f"  [Split point] 어떤 검색 윈도우로도 유효한 종목 경계를 찾지 못했습니다. "
        f"파일이 사실상 단일 종목으로만 이루어졌거나, 한 종목이 너무 압도적입니다."
    )
    return None


# =============================================================================
# 4. 실제 분할
# =============================================================================

def _build_part_paths(input_path: str, output_dir: str) -> Tuple[str, str]:
    """
    출력 파일명 규칙:
      [원래파일명]_[1-2].parquet
      [원래파일명]_[2-2].parquet
    (파일명에는 슬래시를 쓸 수 없어 하이픈으로 표기)
    """
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    part1_final = os.path.join(output_dir, f"{base_name}_[1-2].parquet")
    part2_final = os.path.join(output_dir, f"{base_name}_[2-2].parquet")
    return part1_final, part2_final


def split_into_two_parts(
    input_path: str,
    output_dir: str,
    split_row_idx: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows
    part1_final, part2_final = _build_part_paths(input_path, output_dir)
    part1_tmp = part1_final + ".tmp"
    part2_tmp = part2_final + ".tmp"

    part1_rows = 0
    part2_rows = 0
    part1_symbols = set()
    part2_symbols = set()

    if dry_run:
        cumulative = 0
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx, columns=[SYMBOL_COLUMN])
            syms = tbl.column(SYMBOL_COLUMN).to_pylist()
            n = len(syms)
            for i, s in enumerate(syms):
                if cumulative + i < split_row_idx:
                    part1_rows += 1
                    part1_symbols.add(s)
                else:
                    part2_rows += 1
                    part2_symbols.add(s)
            cumulative += n
        overlap = part1_symbols & part2_symbols
        if overlap:
            raise RuntimeError(f"종목 교집합 발견 (DRY-RUN 검증 실패): {len(overlap)}개")
        return {
            "part_paths": [],
            "part1_rows": part1_rows,
            "part2_rows": part2_rows,
            "part1_symbols": len(part1_symbols),
            "part2_symbols": len(part2_symbols),
            "dry_run": True,
        }

    for tmp in (part1_tmp, part2_tmp):
        if os.path.exists(tmp):
            os.remove(tmp)

    writer1 = pq.ParquetWriter(part1_tmp, schema, compression="zstd")
    writer2 = pq.ParquetWriter(part2_tmp, schema, compression="zstd")
    try:
        cumulative = 0
        for rg_idx in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg_idx)
            n = tbl.num_rows
            rg_start = cumulative
            rg_end = cumulative + n

            if rg_end <= split_row_idx:
                writer1.write_table(tbl)
                part1_rows += n
                part1_symbols.update(tbl.column(SYMBOL_COLUMN).to_pylist())
            elif rg_start >= split_row_idx:
                writer2.write_table(tbl)
                part2_rows += n
                part2_symbols.update(tbl.column(SYMBOL_COLUMN).to_pylist())
            else:
                local_split = split_row_idx - rg_start
                syms = tbl.column(SYMBOL_COLUMN).to_pylist()
                if 0 < local_split < n and syms[local_split] == syms[local_split - 1]:
                    raise RuntimeError(
                        f"split 지점이 종목 경계가 아닙니다 (rg {rg_idx}, local {local_split})"
                    )
                left = tbl.slice(0, local_split)
                right = tbl.slice(local_split, n - local_split)
                if left.num_rows > 0:
                    writer1.write_table(left)
                    part1_rows += left.num_rows
                    part1_symbols.update(left.column(SYMBOL_COLUMN).to_pylist())
                if right.num_rows > 0:
                    writer2.write_table(right)
                    part2_rows += right.num_rows
                    part2_symbols.update(right.column(SYMBOL_COLUMN).to_pylist())
            cumulative = rg_end
    finally:
        writer1.close()
        writer2.close()

    if part1_rows + part2_rows != total_rows:
        for tmp in (part1_tmp, part2_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)
        raise RuntimeError(
            f"row 수 불일치! 원본={total_rows:,}, 분할 합={part1_rows + part2_rows:,}"
        )

    overlap = part1_symbols & part2_symbols
    if overlap:
        for tmp in (part1_tmp, part2_tmp):
            if os.path.exists(tmp):
                os.remove(tmp)
        sample = list(overlap)[:5]
        raise RuntimeError(
            f"종목이 두 part에 걸쳐 있습니다 (분할 무결성 위반). "
            f"교집합 {len(overlap)}개, 예: {sample}"
        )

    os.replace(part1_tmp, part1_final)
    os.replace(part2_tmp, part2_final)
    return {
        "part_paths": [part1_final, part2_final],
        "part1_rows": part1_rows,
        "part2_rows": part2_rows,
        "part1_symbols": len(part1_symbols),
        "part2_symbols": len(part2_symbols),
        "dry_run": False,
    }


# =============================================================================
# 5. 단일 파일 처리
# =============================================================================

def process_one_file(target_path: str) -> Dict[str, Any]:
    """
    target_path: 자를 parquet 파일의 절대경로
    """
    filename = os.path.basename(target_path)
    result: Dict[str, Any] = {"filename": filename, "path": target_path}

    if not os.path.isfile(target_path):
        result["status"] = "error"
        result["message"] = "파일이 존재하지 않습니다"
        raise FileNotFoundError(target_path)

    output_dir = os.path.dirname(target_path)
    src_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  대상: {target_path}  ({src_size_mb:,.1f} MB)")
    result["found_path"] = target_path
    result["src_size_mb"] = round(src_size_mb, 1)

    part1_final, part2_final = _build_part_paths(target_path, output_dir)
    existing = [p for p in (part1_final, part2_final) if os.path.exists(p)]
    if SKIP_IF_EXISTS and existing:
        print(f"  [Skip] 이미 분할 결과가 있어 건너뜁니다:")
        for p in existing:
            print(f"    - {os.path.basename(p)}")
        result["status"] = "skipped"
        result["message"] = "이미 part 파일 존재"
        return result

    safe = sniff_symbol_ordering(target_path)
    if not safe:
        result["status"] = "error"
        result["message"] = "종목 정렬 검증 실패 (sniff WARN)"
        raise RuntimeError(result["message"])

    split_idx = find_midpoint_split_progressive(target_path)
    if split_idx is None:
        result["status"] = "split_skipped"
        result["message"] = (
            "유효한 종목 경계를 찾지 못해 분할 불가 "
            f"(허용 비율 [{MIN_SPLIT_RATIO:.2f}, {MAX_SPLIT_RATIO:.2f}] 안에 경계 없음). "
            f"한 종목이 파일의 대부분을 차지할 가능성이 큽니다."
        )
        return result

    split_result = split_into_two_parts(target_path, output_dir, split_idx, dry_run=DRY_RUN)
    result.update({
        "split_row_idx": split_idx,
        "part1_rows": split_result["part1_rows"],
        "part2_rows": split_result["part2_rows"],
        "part1_symbols": split_result["part1_symbols"],
        "part2_symbols": split_result["part2_symbols"],
    })

    if DRY_RUN:
        print(
            f"  [DRY-RUN] part1={split_result['part1_rows']:,} rows, "
            f"{split_result['part1_symbols']:,} symbols / "
            f"part2={split_result['part2_rows']:,} rows, "
            f"{split_result['part2_symbols']:,} symbols"
        )
        result["status"] = "success_dryrun"
        result["message"] = "DRY-RUN 검증 완료"
        return result

    backup_dir = BACKUP_DIR or os.path.join(output_dir, "_original_backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(target_path))
    if os.path.exists(backup_path):
        print(f"  [Backup] 백업이 이미 존재. 원본 이동 생략: {backup_path}")
        result["backup_path"] = backup_path
        result["backup_action"] = "already_existed"
    else:
        shutil.move(target_path, backup_path)
        print(f"  [Backup] 원본 이동: → {backup_path}")
        result["backup_path"] = backup_path
        result["backup_action"] = "moved"

    print(f"  [완료] 생성된 part 파일:")
    for p in split_result["part_paths"]:
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"    - {os.path.basename(p)} ({sz:,.1f} MB)")

    result["status"] = "success"
    result["message"] = "정상 분할 완료"
    result["part_paths"] = split_result["part_paths"]
    return result


# =============================================================================
# 6. 실행부
# =============================================================================

def main():
    print("=" * 70)
    print("[Tick parquet 2-part 분할 (폴더 목록 + 점진 확장 검색)]")
    print("=" * 70)
    print(f"  대상 폴더 수    : {len(TARGET_FOLDERS)}개")
    for i, fd in enumerate(TARGET_FOLDERS, 1):
        print(f"    {i:>2}. {fd}")
    print(f"  제외 폴더명     : {sorted(EXCLUDE_DIR_NAMES)}")
    print(f"  허용 분할 비율  : {MIN_SPLIT_RATIO:.2f} ~ {MAX_SPLIT_RATIO:.2f}")
    print(f"  검색 반경 일정  : {SEARCH_RADIUS_SCHEDULE}")
    print(f"  DRY_RUN         : {DRY_RUN}")
    print(f"  STOP_ON_ERROR   : {STOP_ON_ERROR}")
    print(f"  SKIP_IF_EXISTS  : {SKIP_IF_EXISTS}")
    print("=" * 70)

    if not TARGET_FOLDERS:
        print("\n[Error] TARGET_FOLDERS가 비어 있습니다.")
        sys.exit(1)

    # 1) 대상 파일 수집
    print("\n[1/2] 대상 parquet 파일 수집 중...")
    target_files = collect_parquet_files(TARGET_FOLDERS)
    print(f"  → 총 {len(target_files)}개 파일 발견")
    for i, p in enumerate(target_files, 1):
        sz = os.path.getsize(p) / (1024 * 1024)
        print(f"    {i:>3}. {p}  ({sz:,.1f} MB)")

    if not target_files:
        print("\n[Info] 처리할 parquet 파일이 없습니다.")
        return

    # 2) 처리
    started = datetime.now()
    results: List[Dict[str, Any]] = []
    print(f"\n[2/2] 분할 처리 시작 ({len(target_files)}개 파일)...")
    for i, target_path in enumerate(target_files, 1):
        print(f"\n{'-' * 70}")
        print(f"[{i}/{len(target_files)}] {target_path}")
        print(f"{'-' * 70}")
        try:
            res = process_one_file(target_path)
            results.append(res)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  [Error] {err_msg}")
            for line in traceback.format_exc().splitlines():
                print(f"    {line}")
            results.append({
                "filename": os.path.basename(target_path),
                "path": target_path,
                "status": "error",
                "message": err_msg,
            })
            if STOP_ON_ERROR:
                print(f"\n[중단] STOP_ON_ERROR=True 이므로 다음 파일을 처리하지 않습니다.")
                break

    elapsed = datetime.now() - started

    # 요약
    print("\n" + "=" * 70)
    print("[처리 요약]")
    print("=" * 70)
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_dryrun = sum(1 for r in results if r["status"] == "success_dryrun")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_split_skipped = sum(1 for r in results if r["status"] == "split_skipped")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_unprocessed = len(target_files) - n_total

    print(f"  소요 시간       : {elapsed}")
    print(f"  발견 파일 수    : {len(target_files)}")
    print(f"  처리 완료       : {n_total}")
    print(f"    └ 성공        : {n_success}")
    if n_dryrun:
        print(f"    └ DRY-RUN     : {n_dryrun}")
    print(f"    └ 스킵        : {n_skipped}")
    print(f"    └ 분할 불가   : {n_split_skipped}  (단일/압도 종목 가능성)")
    print(f"    └ 에러        : {n_error}")
    if n_unprocessed > 0:
        print(f"  미처리          : {n_unprocessed}")

    print(f"\n  [상세]")
    for r in results:
        status_tag = {
            "success":         "[OK]      ",
            "success_dryrun":  "[DRY-RUN] ",
            "skipped":         "[SKIP]    ",
            "split_skipped":   "[NO-CUT]  ",
            "error":           "[ERROR]   ",
        }.get(r["status"], f"[{r['status']}]")
        line = f"    {status_tag} {r['filename']}"
        if r["status"] in ("success", "success_dryrun"):
            line += (
                f"  → part1: {r['part1_rows']:,} rows / "
                f"part2: {r['part2_rows']:,} rows"
            )
        elif r["status"] in ("error", "split_skipped"):
            line += f"  ({r['message']})"
        print(line)

    if n_split_skipped > 0:
        print(
            f"\n  ℹ️ 분할 불가 파일은 단일 종목 또는 한 종목이 압도적인 경우입니다.\n"
            f"     필요하면 MIN_SPLIT_RATIO/MAX_SPLIT_RATIO를 풀어주거나(예: 0.05/0.95),\n"
            f"     해당 파일은 OOM 위험을 감수하고 원본 그대로 처리해야 할 수 있습니다."
        )
    if n_error > 0:
        sys.exit(2 if not STOP_ON_ERROR else 1)


if __name__ == "__main__":
    main()