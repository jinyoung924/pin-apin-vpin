import polars as pl
import pyarrow.parquet as pq

file_path = r"E:\vpin_project_parquet\계산끝난데이터\KOR_2019\kor_201910.parquet"

# ── 1. 파켓 원본 스키마 확인 ───────────────────────────────────────────
print("=== [1] 파켓 원본 스키마 ===")
schema = pq.read_schema(file_path)
print(schema)

# ── 2. Polars로 바로 읽기 (time64[ns] 네이티브 지원) ──────────────────
print("\n=== [2] 데이터 로드 (Polars) ===")
df = pl.read_parquet(file_path)
print(f"  행 수: {df.height:,}  |  열 수: {df.width}")

# ── 3. Polars 스키마 확인 ─────────────────────────────────────────────
print("\n=== [3] Polars 스키마 ===")
print(df.schema)

# ── 4. 상위 5행 미리보기 ──────────────────────────────────────────────
print("\n=== [4] 데이터 미리보기 (상위 5행) ===")
print(df.head())

# ── 5. 전체 컬럼 결측치 & 타입 요약 ──────────────────────────────────
print("\n=== [5] 전체 컬럼 요약 ===")
summary = pl.DataFrame({
    "컬럼"        : df.columns,
    "타입"        : [str(t) for t in df.dtypes],
    "결측치 수"    : [df[col].null_count() for col in df.columns],
    "결측치 비율%" : [round(df[col].null_count() / df.height * 100, 2) for col in df.columns],
    "유니크 값 수" : [df[col].n_unique() for col in df.columns],
})
print(summary)

# ── 6. Time 컬럼 나노초 정밀도 검사 ──────────────────────────────────
time_col = 'Time'

if time_col in df.columns:
    print(f"\n=== [6] '{time_col}' 컬럼 나노초 정밀도 검사 ===")
    print(f"  Polars dtype : {df[time_col].dtype}")

    # Polars의 Time 타입은 나노초 정수로 변환 가능
    # cast to Int64 → 하루 시작(00:00:00)으로부터의 나노초
    time_ns = df[time_col].cast(pl.Int64)

    # 나노초 잔여 계산
    us_remainder = time_ns % 1_000          # µs 초과 나노초
    ms_remainder = time_ns % 1_000_000      # ms 초과
    s_remainder  = time_ns % 1_000_000_000  # 초 초과

    total       = df.height
    nonzero_ns  = (us_remainder != 0).sum()
    nonzero_us  = (ms_remainder != 0).sum()
    nonzero_ms  = (s_remainder  != 0).sum()

    print(f"\n  전체 행 수                    : {total:>10,}")
    print(f"  🔬 나노초 정밀도 존재 (ns≠0)  : {nonzero_ns:>10,}  ({nonzero_ns/total*100:.2f}%)")
    print(f"  🔬 마이크로초 정밀도 (µs≠0)   : {nonzero_us:>10,}  ({nonzero_us/total*100:.2f}%)")
    print(f"  🔬 밀리초 정밀도 (ms≠0)       : {nonzero_ms:>10,}  ({nonzero_ms/total*100:.2f}%)")

    # 나노초 잔여값 분포
    print(f"\n  ── 나노초 잔여값 분포 (상위 10개) ──")
    ns_dist = (
        df.select(pl.Series("ns_rem", us_remainder))
        .group_by("ns_rem")
        .len()
        .sort("ns_rem")
        .head(10)
    )
    print(ns_dist)

    # 샘플 5건 출력
    print(f"\n  ── 샘플 5건 ──")
    print(df.select(time_col).head())

    # 시간 연속성 점검
    print(f"\n  ── 시간 연속성 점검 ──")
    is_sorted    = df[time_col].is_sorted()
    dup_count    = df.height - df[time_col].n_unique()
    print(f"  단조 증가 여부  : {is_sorted}  {'✅' if is_sorted else '⚠️ 역전 존재'}")
    print(f"  중복 타임스탬프 : {dup_count:,} 건  {'✅' if dup_count == 0 else '⚠️'}")

# ── 7. 수치형 컬럼 이상값 점검 ────────────────────────────────────────
print("\n=== [7] 수치형 컬럼 NaN / Inf 점검 ===")
num_cols = [col for col, dtype in zip(df.columns, df.dtypes)
            if dtype in (pl.Float32, pl.Float64)]
any_issue = False
for col in num_cols:
    nan_cnt = df[col].null_count()
    inf_cnt = df[col].is_infinite().sum()
    if nan_cnt > 0 or inf_cnt > 0:
        print(f"  ⚠️  {col:30s} | NaN={nan_cnt:,}  Inf={inf_cnt:,}")
        any_issue = True
if not any_issue:
    print("  ✅ 모든 수치형 컬럼 정상 (NaN/Inf 없음)")

# ── 8. 최종 판정 ──────────────────────────────────────────────────────
print("\n=== [8] 최종 판정 ===")
if time_col in df.columns:
    if nonzero_ns == total:
        print("  ✅ 모든 행에 나노초 정밀도가 완벽히 보존되어 있습니다.")
    elif nonzero_ns > 0:
        print(f"  🔬 일부 ({nonzero_ns:,}건 / {nonzero_ns/total*100:.1f}%) 만 나노초 존재")
    else:
        print("  ⚠️  나노초 정밀도 없음")
        if nonzero_us > 0:
            print(f"  → 마이크로초 정밀도는 존재 ({nonzero_us:,}건)")
        elif nonzero_ms > 0:
            print(f"  → 밀리초 정밀도는 존재 ({nonzero_ms:,}건)")
        else:
            print(f"  → 초 단위로만 저장된 것으로 보입니다.")