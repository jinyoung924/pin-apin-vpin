"""
=============================================================================
[일회성 스크립트] 분할된 parquet 파일명에서 대괄호 표기 제거

문제
  · 앞서 분할 스크립트가 [원래파일명]_[1-2].parquet / _[2-2].parquet
    형식으로 파일을 만들었음
  · Polars의 pl.scan_parquet은 경로를 기본적으로 glob 패턴으로 해석함
  · glob 문법에서 [1-2] 는 "문자 클래스"로 해석되어 실제 파일을 못 찾음
    → "expanded paths were empty" 에러

해결
  · 대괄호와 하이픈을 빼고 _1of2 / _2of2 표기로 일괄 변경
    예) cn_201601_[1-2].parquet        → cn_201601_1of2.parquet
        cn_201601_[2-2].parquet        → cn_201601_2of2.parquet
        cn_201603_part1_[1-2].parquet  → cn_201603_part1_1of2.parquet
        cn_201603_part1_[2-2].parquet  → cn_201603_part1_2of2.parquet

특징
  · 지정한 폴더들을 재귀 탐색
  · _original_backup 폴더 안 파일은 절대 건드리지 않음
  · DRY_RUN 모드로 미리 확인 가능 (기본값 True)
  · 충돌 검사: 새 이름의 파일이 이미 있으면 그 파일은 스킵하고 경고
  · 진행 로그/요약 출력
=============================================================================
"""

import os
import re
import sys
from typing import List, Tuple


# =============================================================================
# USER CONFIG
# =============================================================================

# 이름변경을 적용할 폴더들 (재귀 탐색. 단, _original_backup은 스킵)
TARGET_FOLDERS: List[str] = [
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2017",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2018",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2019",
    r"E:\vpin_project_parquet\processing_data_2019_2021\CN_2021",
]

# True면 실제로 이름을 안 바꾸고 미리보기만 출력
# 미리보기로 확인한 후 False로 바꿔서 다시 실행하세요.
DRY_RUN = False

# 이 이름의 폴더는 건드리지 않음
EXCLUDE_DIR_NAMES = {"_original_backup"}


# =============================================================================
# 핵심 로직
# =============================================================================

# 매칭 패턴:
#   ..._[1-2].parquet  → ..._1of2.parquet
#   ..._[2-2].parquet  → ..._2of2.parquet
# 파일 끝부분만 본다. 중간에 _[1-2]_가 들어간 케이스도 처리하려면 추가로 다룬다.
_END_PATTERN = re.compile(r"_\[([12])-2\]\.parquet$", re.IGNORECASE)
_MID_PATTERN = re.compile(r"_\[([12])-2\]_")


def convert_name(old_name: str) -> str:
    """
    파일명에 들어있는 _[1-2] / _[2-2] 토큰을 _1of2 / _2of2 로 모두 바꾼다.
    파일 끝부분과 중간 모두 처리.
    """
    new_name = old_name
    # 중간에 있는 _[N-2]_ 토큰 처리 (재분할로 생겼을 수 있는 케이스)
    new_name = _MID_PATTERN.sub(lambda m: f"_{m.group(1)}of2_", new_name)
    # 끝 _[N-2].parquet 처리
    new_name = _END_PATTERN.sub(lambda m: f"_{m.group(1)}of2.parquet", new_name)
    return new_name


def collect_candidates(folders: List[str]) -> List[Tuple[str, str]]:
    """
    (old_path, new_path) 튜플 리스트 반환.
    이름 변경이 필요한 파일만 골라 담는다.
    """
    pairs: List[Tuple[str, str]] = []
    seen: set = set()

    for folder in folders:
        folder_abs = os.path.abspath(folder)
        if not os.path.isdir(folder_abs):
            print(f"  [Warn] 폴더 없음 (건너뜀): {folder_abs}")
            continue

        for root, dirs, files in os.walk(folder_abs):
            # _original_backup 등은 더 안 들어감
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIR_NAMES]

            for fn in files:
                if not fn.lower().endswith(".parquet"):
                    continue
                new_name = convert_name(fn)
                if new_name == fn:
                    continue  # 변경 불필요
                old_path = os.path.abspath(os.path.join(root, fn))
                new_path = os.path.abspath(os.path.join(root, new_name))
                if old_path in seen:
                    continue
                seen.add(old_path)
                pairs.append((old_path, new_path))

    pairs.sort()
    return pairs


def main():
    print("=" * 70)
    print("[parquet 파일명 정리: _[1-2]/_[2-2] → _1of2/_2of2]")
    print("=" * 70)
    print(f"  대상 폴더 수    : {len(TARGET_FOLDERS)}개")
    for i, fd in enumerate(TARGET_FOLDERS, 1):
        print(f"    {i:>2}. {fd}")
    print(f"  제외 폴더명     : {sorted(EXCLUDE_DIR_NAMES)}")
    print(f"  DRY_RUN         : {DRY_RUN}  {'(미리보기만)' if DRY_RUN else '(실제 변경)'}")
    print("=" * 70)

    if not TARGET_FOLDERS:
        print("\n[Error] TARGET_FOLDERS가 비어 있습니다.")
        sys.exit(1)

    # 1) 대상 수집
    print("\n[1/3] 이름변경 대상 수집 중...")
    pairs = collect_candidates(TARGET_FOLDERS)
    print(f"  → 변경 대상 {len(pairs)}개 파일")

    if not pairs:
        print("\n[Info] 변경할 파일이 없습니다. 종료합니다.")
        return

    # 2) 충돌 검사
    print("\n[2/3] 충돌 검사...")
    conflicts: List[Tuple[str, str]] = []
    new_path_seen: dict = {}  # new_path -> old_path (이 실행 안에서의 중복도 검사)
    safe_pairs: List[Tuple[str, str]] = []
    for old_path, new_path in pairs:
        if os.path.exists(new_path) and os.path.abspath(new_path) != os.path.abspath(old_path):
            conflicts.append((old_path, new_path))
            continue
        if new_path in new_path_seen:
            # 이 실행 안에서 새 이름이 겹치는 경우 (이론적으로는 거의 없음)
            conflicts.append((old_path, new_path))
            continue
        new_path_seen[new_path] = old_path
        safe_pairs.append((old_path, new_path))

    if conflicts:
        print(f"  [Warn] 충돌 {len(conflicts)}건 발견 (해당 파일은 스킵):")
        for old_path, new_path in conflicts:
            print(f"    - {os.path.basename(old_path)}")
            print(f"      → {os.path.basename(new_path)}  (이미 존재)")
    else:
        print(f"  → 충돌 없음")

    # 3) 실행 또는 미리보기
    print(f"\n[3/3] {'미리보기' if DRY_RUN else '이름변경 실행'} ({len(safe_pairs)}개)...")
    for i, (old_path, new_path) in enumerate(safe_pairs, 1):
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        parent = os.path.dirname(old_path)
        parent_short = os.path.basename(parent)
        print(f"  [{i:>3}/{len(safe_pairs)}] {parent_short}/")
        print(f"        {old_name}")
        print(f"     →  {new_name}")
        if not DRY_RUN:
            try:
                os.rename(old_path, new_path)
            except OSError as e:
                print(f"     [Error] 이름변경 실패: {e}")

    # 요약
    print("\n" + "=" * 70)
    print("[요약]")
    print("=" * 70)
    print(f"  변경 대상         : {len(pairs)}개")
    print(f"  충돌로 스킵       : {len(conflicts)}개")
    print(f"  {'미리보기 완료' if DRY_RUN else '실제 변경 완료'} : {len(safe_pairs)}개")
    if DRY_RUN:
        print(f"\n  → 위 목록이 의도대로면, 스크립트 상단의 DRY_RUN = False 로 바꿔서 다시 실행하세요.")
    else:
        print(f"\n  → 완료. 이제 vpin 코드를 다시 실행하면 됩니다.")


if __name__ == "__main__":
    main()