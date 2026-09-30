# =============================================================================
# 00_setup.R — 패키지 로드, 경로 설정, 공통 유틸리티
# =============================================================================
#
# 모든 스크립트가 source()로 이 파일을 먼저 로드한다.
# 사용자는 이 파일의 ★ 설정 구역만 수정하면 된다.
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(dplyr)
  library(lubridate)
  library(future)
  library(furrr)
  library(zoo)
  library(readr)
  library(arrow)
})


# =============================================================================
# ★ 사용자 설정 구역
# =============================================================================

# 처리할 나라 목록
COUNTRIES <- c("KOR")

# 프로젝트 루트 (이 파일 기준 자동 감지 또는 수동 설정)
PROJECT_ROOT <- "C:/Users/hisjk/Documents/PIN_cal/R_git"


# 입력 데이터 경로 (나라별 하위 디렉토리 포함)
DATA_RAW_DIR <- file.path(PROJECT_ROOT, "data", "raw")

# 출력 경로
OUTPUT_DIR <- file.path(PROJECT_ROOT, "output")

# 로그 경로
LOG_DIR <- file.path(PROJECT_ROOT, "logs")


# =============================================================================
# ★ 롤링 윈도우 파라미터
# =============================================================================

WINDOW_CALENDAR_DAYS <- 60    # 롤링 윈도우 크기 (캘린더일)
MIN_VALID_DAYS       <- 30    # 윈도우 내 최소 유효 영업일 수
                              # 유효 영업일: buys + sells > 0 인 날

# =============================================================================
# ★ VPIN 파라미터
# =============================================================================

VPIN_SAMPLENGTH <- 50         # VPIN 롤링 윈도우 버킷 수


# =============================================================================
# ★ 병렬 설정
# =============================================================================

# logical=FALSE로 물리 코어만 세고, 절반만 사용하여 워커 오버헤드를 줄인다.
N_CORES <- max(1L, parallel::detectCores(logical = FALSE) %/% 2L)

# PINstimation .onLoad()가 future.globals.maxSize = +Inf로 설정하므로 되돌린다.
# 워커 1개에 전달되는 데이터가 1GB를 넘으면 경고가 나도록 설정.
options(future.globals.maxSize = 1 * 1024^3)  # 1GB


# =============================================================================
# 공통 유틸리티 함수
# =============================================================================

#' 타임스탬프 문자열 생성
get_timestamp <- function() {
  format(Sys.time(), "%Y%m%d_%H%M%S")
}

#' 나라별 입력 디렉토리 경로
get_input_dir <- function(country) {
  file.path(DATA_RAW_DIR, country)
}

#' 나라별 출력 디렉토리 경로 (하위 type: "pin", "adjpin", "vpin")
get_output_dir <- function(country, type) {
  d <- file.path(OUTPUT_DIR, country, type)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

#' 로그 파일 경로
get_log_path <- function(country) {
  dir.create(LOG_DIR, recursive = TRUE, showWarnings = FALSE)
  file.path(LOG_DIR, sprintf("%s_run_%s.log", country, get_timestamp()))
}

#' 로그 메시지 출력 + 파일 기록
log_msg <- function(msg, log_file = NULL) {
  ts_msg <- sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), msg)
  cat(ts_msg, "\n")
  if (!is.null(log_file)) {
    cat(ts_msg, "\n", file = log_file, append = TRUE)
  }
}

#' 안전한 tryCatch 래퍼 — 개별 종목 실패를 격리
safe_run <- function(expr, symbol, log_file = NULL) {
  tryCatch(
    expr,
    error = function(e) {
      msg <- sprintf("FAIL [%s]: %s", symbol, conditionMessage(e))
      log_msg(msg, log_file)
      NULL
    }
  )
}


# =============================================================================
# 체크포인트 유틸리티
# =============================================================================

#' 체크포인트 디렉토리 경로
get_checkpoint_dir <- function(country, type) {
  d <- file.path(OUTPUT_DIR, country, type, "checkpoints")
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

#' 완료된 종목 목록 조회 (체크포인트 파일 기반)
get_completed_symbols <- function(checkpoint_dir, ext = "csv") {
  pattern <- sprintf("^sym_.*\\.%s$", ext)
  ckpt_files <- list.files(checkpoint_dir, pattern = pattern)
  gsub(sprintf("^sym_|\\.%s$", ext), "", ckpt_files)
}

#' 체크포인트 파일 저장 (종목 1개 완료 시 즉시 호출)
save_checkpoint <- function(result_df, symbol, checkpoint_dir, ext = "csv") {
  out_path <- file.path(checkpoint_dir, sprintf("sym_%s.%s", symbol, ext))
  if (ext == "csv") {
    readr::write_csv(result_df, out_path)
  } else {
    arrow::write_parquet(result_df, out_path, compression = "zstd")
  }
  out_path
}

#' 체크포인트 전체 병합 → 최종 결과 파일 생성
merge_checkpoints <- function(checkpoint_dir, output_dir, country, type,
                              ext = "csv", log_file = NULL) {
  pattern <- sprintf("^sym_.*\\.%s$", ext)
  ckpt_files <- list.files(checkpoint_dir, pattern = pattern, full.names = TRUE)

  if (length(ckpt_files) == 0) {
    log_msg(sprintf("[병합] %s/%s: 체크포인트 없음", country, type), log_file)
    return(NULL)
  }

  log_msg(sprintf("[병합] %s/%s: %d개 체크포인트 파일 병합 중...",
                  country, type, length(ckpt_files)), log_file)

  all_dfs <- lapply(ckpt_files, function(f) {
    tryCatch({
      if (ext == "csv") readr::read_csv(f, show_col_types = FALSE)
      else arrow::read_parquet(f)
    }, error = function(e) NULL)
  })
  non_empty <- Filter(function(df) !is.null(df) && nrow(df) > 0, all_dfs)

  if (length(non_empty) == 0) {
    log_msg(sprintf("[병합] %s/%s: 유효 결과 없음", country, type), log_file)
    return(NULL)
  }

  final_df <- do.call(rbind, non_empty)
  rownames(final_df) <- NULL

  # 정렬
  if ("symbol" %in% names(final_df) && "window_end" %in% names(final_df)) {
    final_df <- final_df[order(final_df$symbol, final_df$window_end), ]
  } else if ("symbol" %in% names(final_df) && "bucket" %in% names(final_df)) {
    final_df <- final_df[order(final_df$symbol, final_df$bucket), ]
  }

  # 최종 파일 저장
  run_id <- format(Sys.time(), "%Y%m%d_%H%M%S")
  stem <- sprintf("%s_%s_%s", type, country, run_id)

  csv_path     <- file.path(output_dir, paste0(stem, ".csv"))
  parquet_path <- file.path(output_dir, paste0(stem, ".parquet"))

  readr::write_csv(final_df, csv_path)
  arrow::write_parquet(final_df, parquet_path, compression = "zstd")

  log_msg(sprintf("[병합] %s/%s 완료: %s행, %d종목",
                  country, type,
                  format(nrow(final_df), big.mark = ","),
                  length(unique(final_df$symbol))), log_file)
  log_msg(sprintf("  → CSV    : %s", csv_path), log_file)
  log_msg(sprintf("  → Parquet: %s", parquet_path), log_file)

  final_df
}


# =============================================================================
# 환경 정보 출력
# =============================================================================

cat(sprintf("\n%s\n", strrep("=", 65)))
cat(sprintf("[00_setup] PINstimation 파이프라인 설정 완료\n"))
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  프로젝트 루트  : %s\n", PROJECT_ROOT))
cat(sprintf("  입력 디렉토리  : %s\n", DATA_RAW_DIR))
cat(sprintf("  출력 디렉토리  : %s\n", OUTPUT_DIR))
cat(sprintf("  처리 나라      : %s\n", paste(COUNTRIES, collapse = ", ")))
cat(sprintf("  윈도우 크기    : %d 캘린더일\n", WINDOW_CALENDAR_DAYS))
cat(sprintf("  최소 유효일    : %d일\n", MIN_VALID_DAYS))
cat(sprintf("  VPIN 버킷 윈도우: %d\n", VPIN_SAMPLENGTH))
cat(sprintf("  병렬 코어      : %d개\n", N_CORES))
cat(sprintf("%s\n\n", strrep("=", 65)))
