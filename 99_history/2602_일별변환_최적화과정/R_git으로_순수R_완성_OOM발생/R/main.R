# =============================================================================
# main.R — PINstimation 파이프라인 메인 실행 스크립트
# =============================================================================
#
# 실행 흐름:
#   1. 모듈 로드 (00_setup ~ 04_vpin_worker)
#   2. 나라별 순회:
#      a) stream_daily_bs()       → Arrow lazy 일별 B/S 집계 (OOM 방지)
#      b) run_pin_all()           → PIN 추정 (체크포인트 + 병렬)
#      c) run_adjpin_all()        → AdjPIN 추정 (체크포인트 + 병렬)
#      d) rm(daily_bs); gc()      → PIN/AdjPIN 메모리 해제
#      e) get_vpin_symbols()      → Arrow lazy 종목 목록 조회
#      f) run_vpin_all_arrow()    → VPIN 병렬 계산 (predicate pushdown)
#   3. 완료 로그
#
# 실행:
#   Rscript R/main.R
#
# =============================================================================

cat(sprintf("\n%s\n", strrep("#", 65)))
cat(sprintf("# PINstimation 배치 파이프라인\n"))
cat(sprintf("# 시작: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("%s\n\n", strrep("#", 65)))


# =============================================================================
# [1] 모듈 로드
# =============================================================================

# 스크립트 디렉토리 자동 감지
script_dir <- NULL

# 방법 1: Rscript --file= 인자에서 추출
args <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
if (length(file_arg) > 0) {
  script_dir <- dirname(normalizePath(sub("^--file=", "", file_arg[1])))
}

# 방법 2: RStudio에서 source() 시
if (is.null(script_dir)) {
  src_dir <- tryCatch(getSrcDirectory(function(x) x), error = function(e) "")
  if (length(src_dir) > 0 && nchar(src_dir) > 0) {
    script_dir <- src_dir
  }
}

# 방법 3: 현재 작업 디렉토리 기준 탐색
if (is.null(script_dir)) {
  if (dir.exists("R") && file.exists("R/00_setup.R")) {
    script_dir <- "R"
  } else if (file.exists("00_setup.R")) {
    script_dir <- "."
  } else {
    script_dir <- "R"
  }
}

cat(sprintf("[모듈 로드] 스크립트 디렉토리: %s\n", script_dir))

source(file.path(script_dir, "00_setup.R"))
source(file.path(script_dir, "01_data_prep.R"))
source(file.path(script_dir, "02_pin_worker.R"))
source(file.path(script_dir, "03_adjpin_worker.R"))
source(file.path(script_dir, "04_vpin_worker.R"))


# =============================================================================
# [2] 나라별 처리 루프
# =============================================================================

pipeline_start <- Sys.time()

for (country in COUNTRIES) {

  cat(sprintf("\n%s\n", strrep("=", 65)))
  cat(sprintf("[%s] 파이프라인 시작\n", country))
  cat(sprintf("%s\n", strrep("=", 65)))

  log_file <- get_log_path(country)
  log_msg(sprintf("=== %s 파이프라인 시작 ===", country), log_file)


  # ── 2-a) PIN/AdjPIN용 Arrow lazy 집계 ──────────────────────────────────
  # arrow::open_dataset() → lazy 쿼리 → collect()
  # Arrow C++ 엔진이 파일별 스트리밍 처리, R에는 집계 결과만 로드
  log_msg(sprintf("[%s] PIN/AdjPIN 데이터 Arrow lazy 집계 중...", country), log_file)

  daily_bs <- tryCatch(
    stream_daily_bs(country, log_file),
    error = function(e) {
      log_msg(sprintf("[ERROR] %s Arrow lazy 집계 실패: %s", country, e$message), log_file)
      NULL
    }
  )


  # ── 2-b) PIN 추정 ──────────────────────────────────────────────────────
  if (!is.null(daily_bs) && nrow(daily_bs) > 0) {
    log_msg(sprintf("[%s] PIN 추정 시작...", country), log_file)
    run_pin_all(daily_bs, country, log_file)
  }


  # ── 2-c) AdjPIN 추정 ───────────────────────────────────────────────────
  if (!is.null(daily_bs) && nrow(daily_bs) > 0) {
    log_msg(sprintf("[%s] AdjPIN 추정 시작...", country), log_file)
    run_adjpin_all(daily_bs, country, log_file)
  }

  # PIN/AdjPIN 메모리 해제 — VPIN 전에 확보
  rm(daily_bs); gc()


  # ── 2-d) VPIN — Arrow predicate pushdown ────────────────────────────────
  # 종목 목록만 lazy 조회 → 각 워커가 filter(Symbol==sym)으로 자기 데이터만 로드
  log_msg(sprintf("[%s] VPIN 시작 (Arrow predicate pushdown)...", country), log_file)

  vpin_symbols <- tryCatch(
    get_vpin_symbols(country, log_file),
    error = function(e) {
      log_msg(sprintf("[ERROR] %s VPIN 종목 조회 실패: %s", country, e$message), log_file)
      character(0)
    }
  )

  if (length(vpin_symbols) > 0) {
    log_msg(sprintf("[%s] VPIN 계산 시작 (%d종목)...", country, length(vpin_symbols)), log_file)
    run_vpin_all_arrow(vpin_symbols, country, log_file)
  }

  gc()
  log_msg(sprintf("=== %s 파이프라인 완료 ===", country), log_file)
}


# =============================================================================
# [3] 전체 완료
# =============================================================================

total_elapsed <- as.numeric(difftime(Sys.time(), pipeline_start, units = "mins"))

cat(sprintf("\n%s\n", strrep("#", 65)))
cat(sprintf("# 전체 파이프라인 완료\n"))
cat(sprintf("# 처리 나라: %s\n", paste(COUNTRIES, collapse = ", ")))
cat(sprintf("# 총 소요 시간: %.1f분\n", total_elapsed))
cat(sprintf("# 종료: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("%s\n", strrep("#", 65)))
