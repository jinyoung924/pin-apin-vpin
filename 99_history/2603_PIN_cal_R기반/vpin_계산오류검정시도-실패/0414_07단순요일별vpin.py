import os
import glob
import polars as pl

# ── 설정 (원본 코드와 동일) ──────────────────────────────────────
BASE_DIR = r"E:\vpin_project_parquet\processing_data_2019_2021"
COUNTRY  = "CA"
OUTPUT_DIR = os.path.join(BASE_DIR, "R_output", COUNTRY, "vpin")

def find_latest_parquet(directory: str) -> str:
    pattern = os.path.join(directory, f"vpin_{COUNTRY}_*.parquet")
    files = [
        f for f in sorted(glob.glob(pattern))
        if "_sample"  not in os.path.basename(f)
        and "_cleaned" not in os.path.basename(f)
    ]
    return files[-1] if files else ""

VPIN_PATH = find_latest_parquet(OUTPUT_DIR)

# ── 요일별 VPIN 행 수 출력 ────────────────────────────────────────
df = pl.scan_parquet(VPIN_PATH).collect()

DAY_NAMES = {1: "월", 2: "화", 3: "수", 4: "목", 5: "금", 6: "토", 7: "일"}

result = (
    df
    .with_columns(pl.col("Date").dt.weekday().alias("Weekday"))
    .group_by("Weekday")
    .agg(pl.len().alias("Count"))
    .sort("Weekday")
    .with_columns(
        pl.col("Weekday").map_elements(lambda x: DAY_NAMES[x], return_dtype=pl.String).alias("요일")
    )
    .select(["Weekday", "요일", "Count"])
)

print(result)