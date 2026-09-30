"""
=============================================================================
[사전처리] 큰 월별 tick parquet들을 종목 경계 기준으로 정확히 2개 part로 분할
            (목록 처리 버전)

사용법
  1) 아래 USER CONFIG에서 TARGET_FILENAMES 리스트에 분할할 파일 이름들을 지정
     예: TARGET_FILENAMES = ["201412.parquet", "201506.parquet", "201811.parquet"]
  2) 스크립트는 BASE_DIR 아래의 CN_YYYY 폴더들을 자동으로 돌면서 각 파일을 찾는다.
  3) 찾은 각 파일을 정확히 2개의 part로 분할한다 (part1, part2).
     · 분할 지점은 "전체 row의 절반에 가장 가까운 종목 경계"
     · 종목 경계는 절대 깨지 않는다.
  4) 한 파일 처리 중 오류가 나면 STOP_ON_ERROR 설정에 따라 중단/계속 결정.
  5) 마지막에 전체 처리 결과 요약을 출력한다.
=============================================================================
"""

import os
import re
import sys
import glob
import shutil
import traceback
from datetime import datetime
from typing import List, Optional, Dict, Any

import pyarrow as pa
import pyarrow.parquet as pq


# =============================================================================
# USER CONFIG
# =============================================================================

# 분할 대상 파일 이름들 (확장자 포함). 폴더 경로는 자동 검색.
TARGET_FILENAMES: List[str] = [


    "cn_201508.parquet",
    "cn_2015011.parquet",
    "cn_2015012.parquet"

    # "201811.parquet",
]

# 검색 시작 루트 (이 아래에 CN_YYYY 폴더들이 있다고 가정)
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"

# 나라 코드 (CN_2010 처럼 폴더 prefix)
COUNTRY = "CN"

# 원본 파일 백업 위치. None이면 발견된 파일이 있는 폴더 아래 _original_backup 폴더에 저장.
BACKUP_DIR: Optional[str] = None  # None → 자동

# Symbol 컬럼명
SYMBOL_COLUMN = "Symbol"

# True면 실제 파일 안 만들고 "어디서 자를지"만 출력
DRY_RUN = False

# 이미 part 파일이 존재하면 그 파일은 건너뛴다
SKIP_IF_EXISTS = True

# True: 한 파일 처리 중 에러 발생 시 즉시 전체 중단
# False: 에러 발생 파일만 건너뛰고 다음 파일 계속 처리 (권장)
STOP_ON_ERROR = False


# =============================================================================
# 1. 파일 검색
# =============================================================================

_YEAR_FOLDER_RE = re.compile(r"^([A-Z]+)_(\d{4})$")


def find_target_file(base_dir: str, country: str, filename: str) -> str:
    """
    base_dir 아래에서 {COUNTRY}_YYYY 패턴 폴더들을 돌며 filename을 찾는다.
    여러 곳에서 발견되면 에러. 아예 못 찾아도 에러.
    """
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"BASE_DIR이 존재하지 않습니다: {base_dir}")

    matches: List[str] = []
    searched_folders: List[str] = []

    for entry in sorted(os.scandir(base_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        m = _YEAR_FOLDER_RE.match(entry.name)
        if not m or m.group(1) != country:
            continue
        searched_folders.append(entry.path)
        candidate = os.path.join(entry.path, filename)
        if os.path.isfile(candidate):
            matches.append(candidate)

    if not matches:
        raise FileNotFoundError(
            f"파일을 찾지 못했습니다: filename={filename}, base_dir={base_dir}\n"
            f"  검색한 폴더: {[os.path.basename(f) for f in searched_folders]}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"같은 이름의 파일이 여러 폴더에서 발견되었습니다:\n  "
            + "\n  ".join(matches)
        )

    return matches[0]


# =============================================================================
# 2. 사전 검증: 종목이 연속 블록으로 정렬되어 있는지 확인
# =============================================================================

def sniff_symbol_ordering(path: str, n_samples: int = 5, sample_size: int = 200) -> bool:
    """파일의 여러 지점을 sniff해서 종목이 연속 블록으로 정렬되어 있는지 확인."""
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
    sniff_lines = []
    for rg_idx in sample_rg_indices:
        tbl = pf.read_row_group(rg_idx, columns=[SYMBOL_COLUMN])
        symbols = tbl.column(SYMBOL_COLUMN).to_pylist()
        head = symbols[:sample_size]
        unique_count = len(set(head))
        ratio = unique_count / len(head) if head else 0.0
        verdict = "OK" if ratio < 0.1 else "WARN"
        if ratio >= 0.1:
            all_safe = False
        sniff_lines.append(
            f"    rg #{rg_idx:>4}: head {len(head)} rows, unique={unique_count}, ratio={ratio:.3f} [{verdict}]"
        )

    for line in sniff_lines:
        print(line)
    if all_safe:
        print(f"  [Sniff] OK (종목이 연속 블록으로 정렬됨)")
    else:
        print(f"  [Sniff] WARN (종목이 sample 안에서 많이 섞임)")
    return all_safe


# =============================================================================
# 3. 분할 지점 사전 계산: "전체 row의 절반에 가장 가까운 종목 경계"
# =============================================================================

def find_midpoint_split(path: str) -> int:
    """파일을 streaming 스캔해 전체 row의 절반 지점에 가장 가까운 종목 경계 row 인덱스를 찾는다."""
    pf = pq.ParquetFile(path)
    total_rows = pf.metadata.num_rows
    target = total_rows // 2

    cumulative = 0
    target_rg = None
    for rg_idx in range(pf.num_row_groups):
        n_in_rg = pf.metadata.row_group(rg_idx).num_rows
        if cumulative + n_in_rg >= target:
            target_rg = rg_idx
            break
        cumulative += n_in_rg

    if target_rg is None:
        target_rg = pf.num_row_groups - 1

    candidate_rgs = [target_rg]
    if target_rg > 0:
        candidate_rgs.insert(0, target_rg - 1)
    if target_rg < pf.num_row_groups - 1:
        candidate_rgs.append(target_rg + 1)

    offsets = {}
    cum = 0
    for i in range(pf.num_row_groups):
        if i in candidate_rgs:
            offsets[i] = cum
        cum += pf.metadata.row_group(i).num_rows

    prev_last_symbol = None
    if candidate_rgs[0] > 0:
        prev_tbl = pf.read_row_group(candidate_rgs[0] - 1, columns=[SYMBOL_COLUMN])
        prev_syms = prev_tbl.column(SYMBOL_COLUMN).to_pylist()
        if prev_syms:
            prev_last_symbol = prev_syms[-1]
        del prev_tbl, prev_syms

    region_rows = []
    for rg_idx in candidate_rgs:
        tbl = pf.read_row_group(rg_idx, columns=[SYMBOL_COLUMN])
        syms = tbl.column(SYMBOL_COLUMN).to_pylist()
        base = offsets[rg_idx]
        region_rows.extend((base + i, s) for i, s in enumerate(syms))
        del tbl, syms

    boundary_indices = []
    last_seen = prev_last_symbol
    for global_idx, sym in region_rows:
        if last_seen is None or sym != last_seen:
            boundary_indices.append(global_idx)
        last_seen = sym

    if not boundary_indices:
        raise RuntimeError("후보 영역에서 종목 경계를 찾지 못했습니다.")

    valid_boundaries = [b for b in boundary_indices if 0 < b < total_rows]
    if not valid_boundaries:
        raise RuntimeError(
            "유효한 split 지점이 없습니다. 파일이 단일 종목으로만 이루어져 있을 수 있습니다."
        )
    split_idx = min(valid_boundaries, key=lambda x: abs(x - target))

    print(
        f"  [Split point] row {split_idx:,} / {total_rows:,} "
        f"({split_idx / total_rows * 100:.1f}% : {(total_rows - split_idx) / total_rows * 100:.1f}%)"
    )
    return split_idx


# =============================================================================
# 4. 실제 분할: row group 단위 streaming으로 두 writer에 분배
# =============================================================================

def split_into_two_parts(
    input_path: str,
    output_dir: str,
    split_row_idx: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    return: {"part_paths": [..], "part1_rows": int, "part2_rows": int,
             "part1_symbols": int, "part2_symbols": int}
    """
    pf = pq.ParquetFile(input_path)
    schema = pf.schema_arrow
    total_rows = pf.metadata.num_rows

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    part1_final = os.path.join(output_dir, f"{base_name}_part1.parquet")
    part2_final = os.path.join(output_dir, f"{base_name}_part2.parquet")
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

    # 검증
    if part1_rows + part2_rows != total_rows:
        # 실패 시 tmp 정리
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
# 5. 단일 파일 처리 (검색 → sniff → split point → 분할 → 백업)
# =============================================================================

def process_one_file(filename: str) -> Dict[str, Any]:
    """
    return: {"filename": str, "status": "success"|"skipped"|"error",
             "message": str, ...추가 통계}
    """
    result: Dict[str, Any] = {"filename": filename}

    # 1) 검색
    target_path = find_target_file(BASE_DIR, COUNTRY, filename)
    output_dir = os.path.dirname(target_path)
    src_size_mb = os.path.getsize(target_path) / (1024 * 1024)
    print(f"  발견: {target_path}  ({src_size_mb:,.1f} MB)")
    result["found_path"] = target_path
    result["src_size_mb"] = round(src_size_mb, 1)

    # 2) 이미 part가 있으면 skip
    base = os.path.splitext(os.path.basename(target_path))[0]
    existing = sorted(
        glob.glob(os.path.join(output_dir, f"{base}_part1.parquet"))
        + glob.glob(os.path.join(output_dir, f"{base}_part2.parquet"))
    )
    if SKIP_IF_EXISTS and existing:
        print(f"  [Skip] 이미 분할 결과가 있어 건너뜁니다:")
        for p in existing:
            print(f"    - {os.path.basename(p)}")
        result["status"] = "skipped"
        result["message"] = "이미 part 파일 존재"
        return result

    # 3) sniff
    safe = sniff_symbol_ordering(target_path)
    if not safe:
        result["status"] = "error"
        result["message"] = "종목 정렬 검증 실패 (sniff WARN)"
        raise RuntimeError(result["message"])

    # 4) split point
    split_idx = find_midpoint_split(target_path)

    # 5) 분할
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

    # 6) 원본 백업
    backup_dir = BACKUP_DIR or os.path.join(output_dir, "_original_backup")
    os.makedirs(backup_dir, exist_ok=True)
    backup_path = os.path.join(backup_dir, os.path.basename(target_path))
    if os.path.exists(backup_path):
        print(f"  [Backup] 백업이 이미 존재. 원본 이동 생략: {backup_path}")
        print(f"           ⚠️ 원본이 아직 남아있다면 직접 옮겨야 합니다 (Step 1 결과 중복 위험).")
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
# 6. 실행부 (목록 처리)
# =============================================================================

def main():
    print("=" * 70)
    print("[Tick parquet 2-part 분할 사전처리 (목록 처리)]")
    print("=" * 70)
    print(f"  대상 파일 수   : {len(TARGET_FILENAMES)}개")
    for i, fn in enumerate(TARGET_FILENAMES, 1):
        print(f"    {i:>2}. {fn}")
    print(f"  검색 루트      : {BASE_DIR}")
    print(f"  나라코드       : {COUNTRY}")
    print(f"  DRY_RUN        : {DRY_RUN}")
    print(f"  STOP_ON_ERROR  : {STOP_ON_ERROR}")
    print(f"  SKIP_IF_EXISTS : {SKIP_IF_EXISTS}")
    print("=" * 70)

    if not TARGET_FILENAMES:
        print("\n[Error] TARGET_FILENAMES가 비어 있습니다.")
        sys.exit(1)

    if not os.path.isdir(BASE_DIR):
        print(f"\n[Error] BASE_DIR이 존재하지 않습니다: {BASE_DIR}")
        sys.exit(1)

    started = datetime.now()
    results: List[Dict[str, Any]] = []

    for i, filename in enumerate(TARGET_FILENAMES, 1):
        print(f"\n{'-' * 70}")
        print(f"[{i}/{len(TARGET_FILENAMES)}] {filename}")
        print(f"{'-' * 70}")
        try:
            res = process_one_file(filename)
            results.append(res)
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"  [Error] {err_msg}")
            print("  상세:")
            for line in traceback.format_exc().splitlines():
                print(f"    {line}")
            results.append({
                "filename": filename,
                "status": "error",
                "message": err_msg,
            })
            if STOP_ON_ERROR:
                print(f"\n[중단] STOP_ON_ERROR=True 이므로 다음 파일을 처리하지 않습니다.")
                break

    elapsed = datetime.now() - started

    # =========================================================================
    # 요약 리포트
    # =========================================================================
    print("\n" + "=" * 70)
    print("[처리 요약]")
    print("=" * 70)
    n_total = len(results)
    n_success = sum(1 for r in results if r["status"] == "success")
    n_dryrun = sum(1 for r in results if r["status"] == "success_dryrun")
    n_skipped = sum(1 for r in results if r["status"] == "skipped")
    n_error = sum(1 for r in results if r["status"] == "error")
    n_unprocessed = len(TARGET_FILENAMES) - n_total

    print(f"  소요 시간     : {elapsed}")
    print(f"  요청 파일 수  : {len(TARGET_FILENAMES)}")
    print(f"  처리 완료     : {n_total}")
    print(f"    └ 성공      : {n_success}")
    if n_dryrun:
        print(f"    └ DRY-RUN   : {n_dryrun}")
    print(f"    └ 스킵      : {n_skipped}")
    print(f"    └ 에러      : {n_error}")
    if n_unprocessed > 0:
        print(f"  미처리        : {n_unprocessed}  (STOP_ON_ERROR로 중단됨)")

    print(f"\n  [상세]")
    for r in results:
        status_tag = {
            "success": "[OK]      ",
            "success_dryrun": "[DRY-RUN] ",
            "skipped": "[SKIP]    ",
            "error": "[ERROR]   ",
        }.get(r["status"], f"[{r['status']}]")
        line = f"    {status_tag} {r['filename']}"
        if r["status"] in ("success", "success_dryrun"):
            line += (
                f"  → part1: {r['part1_rows']:,} rows / "
                f"part2: {r['part2_rows']:,} rows"
            )
        elif r["status"] == "error":
            line += f"  ({r['message']})"
        print(line)

    if n_error > 0:
        print(f"\n  ⚠️ 에러 {n_error}건이 있습니다. 해당 파일들은 수동 점검 필요.")
        sys.exit(2 if not STOP_ON_ERROR else 1)


if __name__ == "__main__":
    main()