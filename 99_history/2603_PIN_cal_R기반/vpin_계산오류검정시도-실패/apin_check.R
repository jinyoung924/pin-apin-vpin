# =============================================================================
# [APIN 결과 분석] 파켓 파일 로드 및 에러(Timeout, Est_Error) 내역 확인
# =============================================================================

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
})

# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================
BASE_DIR <- "E:/vpin_project_parquet/processing_data"
COUNTRY  <- "US"

# ★ 분석할 실제 파켓 파일의 이름을 입력하세요 (예: "apin_KOR_20231026_1430.parquet")
FILE_NAME <- "pin_US_20260320_0547.parquet" 

INPUT_PATH <- file.path(BASE_DIR, "R_output", COUNTRY, "pin", FILE_NAME)
OUTPUT_CSV <- file.path(BASE_DIR, "R_output", COUNTRY, "pin", paste0("error_report_", FILE_NAME, ".csv"))

# =============================================================================
# [1] 데이터 로드
# =============================================================================
if (!file.exists(INPUT_PATH)) {
  stop(sprintf("[Error] 파일을 찾을 수 없습니다: %s\n파일명을 다시 확인해주세요.", INPUT_PATH))
}

cat(sprintf("\n[1] 데이터 로드 중: %s\n", INPUT_PATH))
df <- arrow::read_parquet(INPUT_PATH)

cat(sprintf("  총 레코드 수: %s 개\n", format(nrow(df), big.mark = ",")))
cat(sprintf("  총 종목 수  : %s 개\n", format(length(unique(df$Symbol)), big.mark = ",")))


# =============================================================================
# [2] skip_reason 분포 확인
# =============================================================================
cat(sprintf("\n%s\n[2] skip_reason 전체 분포\n%s\n", strrep("=", 60), strrep("=", 60)))

reason_summary <- df %>%
  group_by(skip_reason) %>%
  summarise(
    Count = n(),
    Ratio = round(n() / nrow(df) * 100, 2)
  ) %>%
  arrange(desc(Count))

print(as.data.frame(reason_summary), row.names = FALSE)


# =============================================================================
# [3] Est_Error 및 Timeout 발생 내역 확인
# =============================================================================
cat(sprintf("\n%s\n[3] Est_Error 및 Timeout 발생 상세 분석\n%s\n", strrep("=", 60), strrep("=", 60)))

# 에러가 발생한 행만 필터링
error_df <- df %>%
  filter(skip_reason %in% c("Est_Error", "Timeout")) %>%
  select(Symbol, Date, valid_days, skip_reason) %>%
  arrange(skip_reason, Symbol, Date)

if (nrow(error_df) > 0) {
  cat(sprintf("  발견된 에러(Est_Error/Timeout) 총 건수: %s 건\n\n", format(nrow(error_df), big.mark = ",")))
  
  # 3-1. 어떤 종목에서 에러가 많이 발생했는지 집계 (상위 15개 종목 출력)
  cat("  [종목별 에러 발생 건수 Top 15]\n")
  error_by_symbol <- error_df %>%
    group_by(Symbol, skip_reason) %>%
    summarise(Error_Count = n(), .groups = 'drop') %>%
    arrange(desc(Error_Count))
  
  print(head(as.data.frame(error_by_symbol), 15), row.names = FALSE)
  
  # 3-2. 상세 내역 샘플 출력
  cat("\n  [에러 발생 상세 내역 샘플 (최근 날짜 기준 10건)]\n")
  sample_errors <- error_df %>% 
    arrange(desc(Date)) %>% 
    head(10)
  print(as.data.frame(sample_errors), row.names = FALSE)
  
  # 3-3. 전체 에러 내역 CSV 저장
  write.csv(error_df, OUTPUT_CSV, row.names = FALSE)
  cat(sprintf("\n  => 에러 전체 내역이 CSV로 저장되었습니다.\n     경로: %s\n", OUTPUT_CSV))
  
} else {
  cat("  축하합니다! Est_Error나 Timeout이 발생한 데이터가 없습니다.\n")
}

cat(sprintf("%s\n", strrep("=", 60)))