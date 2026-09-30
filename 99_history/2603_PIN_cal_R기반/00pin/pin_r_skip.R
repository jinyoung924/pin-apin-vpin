# =============================================================================
# [PIN 전체계산] PINstimation::pin() 롤링 추정
# =============================================================================
#
# ── 개요 ─────────────────────────────────────────────────────────────────────
# 모델: EKOP(1996) — 5개 파라미터 (alpha, delta, mu, eps.b, eps.s)
# 방법: 60영업일 롤링 윈도우를 전 종목·전 기간에 걸쳐 병렬로 추정한다.
#
# ── 버그 수정 내역 ────────────────────────────────────────────────────────────
# [수정 1] pin() 에 존재하지 않는 num_init 파라미터 제거
#   - PINstimation::pin() 시그니처: pin(data, initialsets, factorization, verbose)
#   - num_init 을 넘기면 "unused argument" 에러 → converged=FALSE → 결과 0건
#   - NUM_INITIAL_SETS 설정값 및 num_init 인자 전달 코드를 모두 제거함
#
# [수정 2] PIN_INITIALSETS 값 수정
#   - 잘못된 값: "GE"  →  올바른 값: "EA"
#   - "GE" 는 adjpin() 전용 초기점 알고리즘 {"GE", "CL", "RANDOM"} 중 하나
#   - pin() 전용 함수: pin_ea() / pin_yz() / pin_gwj()
#
# [수정 3] @parameters 슬롯의 키 이름 복원
#   - pin() @parameters 의 실제 키: "eps.b", "eps.s"  (공식 예제 출력 확인)
#   - 1차 수정에서 "epsilon.b"/"epsilon.s" 로 잘못 변경했던 것을 원래대로 복원
#   - "epsilon.b"/"epsilon.s" 키는 존재하지 않으므로 항상 NA 반환됨
#
# [수정 4] pin() 호출 방식 전면 수정 (공식 문서 기반)
#   - pin()의 실제 시그니처: pin(data, initialsets, factorization, verbose)
#   - pin()의 initialsets는 **numeric vector/dataframe** 이며 문자열("EA")이 아님
#   - pin()에는 method 파라미터가 존재하지 않음 (method는 adjpin() 전용)
#   - "EA"를 initialsets로 넘기면 타입 에러 → converged=FALSE → 결과 0건
#   - EA 알고리즘을 사용하려면 전용 함수 pin_ea(data, verbose) 를 사용해야 함
#   - pin() → pin_ea() / pin_yz() / pin_gwj() 로 전환
#   - PIN_METHOD 설정 제거 (불필요)
#
# ── 입출력 ───────────────────────────────────────────────────────────────────
# 입력 : R_output/{COUNTRY}/full_daily_bs.parquet
#          └─ 01_preprocess.py Step2 출력.
#             전 종목·전 영업일에 대해 일별 매수(B)·매도(S) 수량이 정렬된 상태.
#             거래 없는 날은 B=S=0 행이 삽입되어 있어 60행 = 정확히 60 영업일.
# 출력 : R_output/{COUNTRY}/pin/
#          ├─ pin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet       ← 전체 결과 (주 출력)
#          ├─ pin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv ← 눈으로 확인용
#          └─ checkpoints/sym_{Symbol}.parquet             ← 종목별 중간 저장
#
# ── 실행 흐름 ────────────────────────────────────────────────────────────────
# [1] full_daily_bs.parquet 로드
#       전 종목의 일별 B/S 데이터를 메모리에 올린다.
#
# [2] 체크포인트 확인 (중단 재개)
#       checkpoints/sym_*.parquet 파일 목록을 스캔해 완료 종목을 파악한다.
#       remaining_syms = 전체 종목 - 완료 종목 → 재실행 시 이어서 진행.
#
# [3] 종목별 데이터 분할
#       daily_bs를 Symbol 기준으로 분리해 워커에 넘길 args 리스트를 구성한다.
#
# [4] 병렬 추정 — makeCluster(PSOCK) + parLapply
#       워커 1개 = 종목 1개의 전체 기간 처리.
#       각 워커 내부 루프 (i = WINDOW_SIZE ~ n_days, 1행씩 슬라이딩):
#         window_B/S  = [i-WINDOW_SIZE+1 ~ i] 구간의 60영업일 슬라이스
#         valid_days  = 윈도우 내 B+S>0인 날 수
#         valid_days < MIN_VALID_DAYS → 스킵하지 않고 NA행 기록 (skip_reason="Low_Valid_Days")
#         pin() 수렴 실패(error) → NA행 기록 (skip_reason="Est_Error")
#         수렴 성공 → (Symbol, Date, valid_days, a,d,u,eb,es, PIN, skip_reason="Success") 행 추가
#       워커 완료 즉시 checkpoints/sym_{Symbol}.parquet 저장.
#       각 워커가 독립 파일에 쓰므로 레이스컨디션 없음.
#       CHECKPOINT_N 종목마다 진행률·ETA 로그 출력.
#
# [5] 체크포인트 병합 → 최종 저장
#       checkpoints/sym_*.parquet 전부 rbind → Symbol·Date 정렬.
#       parquet (전체 결과) + CSV (상위 1000행 샘플) 저장.
#       PIN 결과는 최대 ~152만 행 수준으로 rbind 피크 메모리 ~400 MB 이내.
#
# ── 체크포인트 설계 ──────────────────────────────────────────────────────────
# 1종목이 완료될 때마다 즉시 개별 파일로 저장하는 이유:
#   - 수천 종목 계산 중 언제든 중단되어도 완료분은 보존됨
#   - 재실행 시 remaining_syms만 계산하므로 중복 연산 없음
#   - 워커별 파일 분리로 동시 쓰기 충돌(레이스컨디션) 원천 방지
#
# ── 캘린더 정렬이 필요한 이유 ────────────────────────────────────────────────
# 원본 틱 데이터에는 거래가 없는 날의 행이 존재하지 않는다.
# 정렬 없이 60행 롤링을 잡으면 "60행 = 실제로는 여러 달"이 될 수 있다.
# 01_preprocess.py Step2에서 거래 없는 날에 B=S=0 행을 삽입했으므로
# 이 스크립트의 60행은 항상 정확히 60 영업일을 의미한다.
#
# ── 실행 ─────────────────────────────────────────────────────────────────────
#   Rscript 00pin/02_r_pin.R
#
# ── 의존 패키지 (최초 1회) ───────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "parallel"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(parallel)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

# 틱 parquet 루트 폴더 (Python과 동일한 경로)
BASE_DIR <- "E:/vpin_project_parquet/expend_data_2010_2019"

# ssu :  "C:/vpin_project/vpin_project_parquet/processing_data"
# suji : "E:/vpin_project_parquet/processing_data"

# 처리할 나라코드 (01_preprocess.py 의 COUNTRY 와 일치해야 함)
COUNTRY <- "KOR"

# ── 롤링 윈도우 파라미터 ──────────────────────────────────────────────────────
WINDOW_SIZE    <- 60   # 롤링 윈도우 크기 (영업일)
MIN_VALID_DAYS <- 30   # 윈도우 내 실제 거래일(B+S>0) 최솟값

# ── PINstimation PIN 추정 함수 선택 ──────────────────────────────────────────
# [수정 4] pin()의 실제 시그니처: pin(data, initialsets, factorization, verbose)
#   - pin()의 initialsets는 **numeric vector/dataframe** 이며 문자열("EA")이 아님
#   - pin()에는 method 파라미터가 존재하지 않음 (method는 adjpin() 전용)
#   - EA/YZ/GWJ 알고리즘은 전용 함수 pin_ea() / pin_yz() / pin_gwj() 로 호출
# 따라서 pin_ea(data, verbose) 형태로 직접 호출해야 함
PIN_INITIALSETS <- "EA"   # "EA" (Ersan-Alici) | "YZ" (Yan-Zhang) | "GWJ" (Gan-Wei-Johnstone)

# ── 병렬 설정 ─────────────────────────────────────────────────────────────────
NUM_WORKERS  <- parallel::detectCores(logical = TRUE)
CHECKPOINT_N <- 100   # N 종목마다 진행 로그 출력

# =============================================================================
# (이하 수정 불필요) — 경로 자동 생성
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", COUNTRY, "full_daily_bs.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "pin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n[PIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  나라코드    : %s\n", COUNTRY))
cat(sprintf("  INPUT_PATH  : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR  : %s\n", OUTPUT_DIR))
cat(sprintf("  윈도우 크기 : %d영업일  |  최소 유효일: %d일\n", WINDOW_SIZE, MIN_VALID_DAYS))
cat(sprintf("  InitSets    : %s (pin_%s 함수 사용)\n", PIN_INITIALSETS, tolower(PIN_INITIALSETS)))
cat(sprintf("  CPU 코어    : %d개  |  로그 간격: %d종목\n", NUM_WORKERS, CHECKPOINT_N))


# =============================================================================
# [1] 데이터 로드
# =============================================================================

if (!file.exists(INPUT_PATH))
  stop(sprintf("[Error] 파일 없음: %s\n먼저 01_preprocess.py를 실행하세요.", INPUT_PATH))

cat(sprintf("\n[1] 데이터 로드: %s\n", INPUT_PATH))
daily_bs      <- arrow::read_parquet(INPUT_PATH)
daily_bs$Date <- as.Date(daily_bs$Date)
daily_bs$B    <- as.integer(daily_bs$B)
daily_bs$S    <- as.integer(daily_bs$S)

all_symbols <- sort(unique(daily_bs$Symbol))
n_total     <- length(all_symbols)

cat(sprintf("  전체 종목 수: %d개\n", n_total))
cat(sprintf("  전체 행     : %s\n", format(nrow(daily_bs), big.mark = ",")))
cat(sprintf("  날짜 범위   : %s ~ %s\n", min(daily_bs$Date), max(daily_bs$Date)))


# =============================================================================
# [2] 재개 확인
# =============================================================================

ckpt_files     <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$")
completed_syms <- gsub("^sym_|\\.parquet$", "", ckpt_files)
remaining_syms <- setdiff(all_symbols, completed_syms)

cat(sprintf("\n[2] 체크포인트 확인\n"))
cat(sprintf("  완료 종목  : %d개\n", length(completed_syms)))
cat(sprintf("  처리 예정  : %d개\n", length(remaining_syms)))

if (length(remaining_syms) == 0)
  cat("\n  모든 종목 처리 완료. 최종 병합으로 건너뜁니다.\n")


# =============================================================================
# [3] 종목 데이터 분할 + 워커 인자 구성
# =============================================================================

worker_process_symbol <- function(args) {
  suppressPackageStartupMessages({
    library(PINstimation)
    library(arrow)
  })

  sym_df         <- args$sym_df
  symbol         <- args$symbol
  checkpoint_dir <- args$checkpoint_dir
  window_size    <- args$window_size
  min_valid_days <- args$min_valid_days
  initialsets    <- args$initialsets   # "EA", "YZ", or "GWJ"

  n_days <- nrow(sym_df)

  # [수정 4] pin() 대신 전용 함수 pin_ea() / pin_yz() / pin_gwj() 사용
  #   - pin()의 initialsets는 numeric vector/dataframe 이며 문자열이 아님
  #   - pin()에 method 파라미터가 존재하지 않음
  #   - EA/YZ/GWJ 알고리즘은 전용 함수로 호출해야 함
  # [수정 3] @parameters 키를 "eps.b" / "eps.s" 로 복원 (공식 예제 출력과 일치)
  # 반환값: list(converged, reason, ...) 형태
  #   converged = TRUE  → 추정 성공, reason = "Success"
  #   converged = FALSE → 실패, reason = "Est_Error"
  estimate_window <- function(window_df) {
    tryCatch({
      res <- switch(initialsets,
                    "EA"  = pin_ea(data  = window_df, verbose = FALSE),
                    "YZ"  = pin_yz(data  = window_df, verbose = FALSE),
                    "GWJ" = pin_gwj(data = window_df, verbose = FALSE),
                    stop(paste("Unknown initialsets:", initialsets)))

      pin_val <- as.numeric(res@pin)
      # NA/NaN 수렴 실패 체크
      if (is.na(pin_val) || is.nan(pin_val)) {
        return(list(converged = FALSE, reason = "Est_Error"))
      }

      params <- res@parameters
      list(a  = as.numeric(params["alpha"]),
           d  = as.numeric(params["delta"]),
           u  = as.numeric(params["mu"]),
           eb = as.numeric(params["eps.b"]),
           es = as.numeric(params["eps.s"]),
           PIN = pin_val,
           converged = TRUE,
           reason    = "Success")
    }, error = function(e) list(converged = FALSE, reason = "Est_Error"))
  }

  # ── NA 행 생성 헬퍼: 파라미터를 NA로 채우고 skip_reason을 기록 ────────────
  make_na_row <- function(symbol, date, valid_days, reason) {
    data.frame(
      Symbol      = symbol,
      Date        = as.Date(date),
      valid_days  = valid_days,
      a           = NA_real_,
      d           = NA_real_,
      u           = NA_real_,
      eb          = NA_real_,
      es          = NA_real_,
      PIN         = NA_real_,
      skip_reason = reason,
      stringsAsFactors = FALSE
    )
  }

  results <- list()
  if (n_days >= window_size) {
    for (i in window_size:n_days) {
      s          <- i - window_size + 1
      window_B   <- sym_df$B[s:i]
      window_S   <- sym_df$S[s:i]
      valid_days <- sum((window_B + window_S) > 0)

      # 유효 거래일 부족 → NA 행 + skip_reason = "Low_Valid_Days"
      if (valid_days < min_valid_days) {
        results[[length(results) + 1]] <- make_na_row(
          symbol, sym_df$Date[i], valid_days, "Low_Valid_Days"
        )
        next
      }

      est <- estimate_window(data.frame(B = window_B, S = window_S))

      if (!isTRUE(est$converged)) {
        # 수렴 실패 → NA 행 + skip_reason = "Est_Error"
        results[[length(results) + 1]] <- make_na_row(
          symbol, sym_df$Date[i], valid_days, est$reason
        )
        next
      }

      results[[length(results) + 1]] <- data.frame(
        Symbol = symbol, Date = as.Date(sym_df$Date[i]), valid_days = valid_days,
        a = est$a, d = est$d, u = est$u, eb = est$eb, es = est$es, PIN = est$PIN,
        skip_reason = "Success",
        stringsAsFactors = FALSE
      )
    }
  }

  result_df <- if (length(results) > 0) do.call(rbind, results) else
    data.frame(Symbol = character(0), Date = as.Date(character(0)),
               valid_days = integer(0), a = numeric(0), d = numeric(0),
               u = numeric(0), eb = numeric(0), es = numeric(0), PIN = numeric(0),
               skip_reason = character(0))

  out_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".parquet"))
  arrow::write_parquet(result_df, out_path, compression = "zstd")
  return(list(symbol = symbol, n_rows = nrow(result_df)))
}


if (length(remaining_syms) > 0) {

  cat(sprintf("\n[3] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  split_bs <- split(daily_bs[, c("Date", "B", "S")], daily_bs$Symbol)

  symbol_args <- lapply(remaining_syms, function(sym) {
    rows <- split_bs[[sym]]
    rows <- rows[order(rows$Date), ]
    list(sym_df = rows, symbol = sym, checkpoint_dir = CHECKPOINT_DIR,
         window_size = WINDOW_SIZE, min_valid_days = MIN_VALID_DAYS,
         initialsets = PIN_INITIALSETS)
  })
  cat("  분할 완료\n")

  # =============================================================================
  # [4] 병렬 처리
  # =============================================================================

  n_remaining <- length(remaining_syms)
  n_workers   <- min(NUM_WORKERS, n_remaining)

  cat(sprintf("\n[4] pin() 병렬 추정 — 워커 %d개, 윈도우 %d일\n%s\n",
              n_workers, WINDOW_SIZE, strrep("-", 65)))

  start_time <- Sys.time()
  cl <- makeCluster(n_workers, type = "PSOCK")
  on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

  batch_starts <- seq(1, n_remaining, by = CHECKPOINT_N)

  for (bi in seq_along(batch_starts)) {
    b_start <- batch_starts[bi]
    b_end   <- min(b_start + CHECKPOINT_N - 1, n_remaining)
    tryCatch(
      parLapply(cl, symbol_args[b_start:b_end], worker_process_symbol),
      error = function(e) cat(sprintf("  [Warning] 배치 %d 오류: %s\n", bi, e$message))
    )
    n_done  <- length(completed_syms) + b_end
    elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    eta_min <- if (b_end > 0) elapsed / b_end * (n_remaining - b_end) else NA
    cat(sprintf("  [%s] %d / %d 종목 완료  (%.1f분 경과%s)\n",
                format(Sys.time(), "%H:%M:%S"), n_done, n_total, elapsed,
                if (!is.na(eta_min)) sprintf(" | 예상 잔여 %.0f분", eta_min) else ""))
  }

  cat(sprintf("\n  총 소요 시간: %.1f분\n",
              as.numeric(difftime(Sys.time(), start_time, units = "mins"))))
}


# =============================================================================
# [5] 체크포인트 병합 → 최종 결과 저장
# =============================================================================

cat(sprintf("\n[5] 체크포인트 병합 중...\n"))

all_ckpt_files <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$", full.names = TRUE)
cat(sprintf("  체크포인트 파일 수: %d개\n", length(all_ckpt_files)))

all_dfs   <- lapply(all_ckpt_files, function(f) tryCatch(arrow::read_parquet(f), error = function(e) NULL))
non_empty <- Filter(function(df) !is.null(df) && nrow(df) > 0, all_dfs)

if (length(non_empty) == 0) {
  cat("[Warning] 추정 결과가 없습니다.\n")
} else {
  final_df           <- do.call(rbind, non_empty)
  rownames(final_df) <- NULL
  final_df           <- final_df[order(final_df$Symbol, final_df$Date), ]

  run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
  result_stem  <- sprintf("pin_%s_%s", COUNTRY, run_id)
  parquet_path <- file.path(OUTPUT_DIR, paste0(result_stem, ".parquet"))
  csv_path     <- file.path(OUTPUT_DIR, paste0(result_stem, "_sample1000.csv"))

  arrow::write_parquet(final_df, parquet_path, compression = "zstd")
  write.csv(head(final_df, 1000), csv_path, row.names = FALSE)

  cat(sprintf("\n%s\n[결과 요약]\n%s\n", strrep("=", 65), strrep("=", 65)))
  cat(sprintf("  나라코드    : %s\n", COUNTRY))
  cat(sprintf("  날짜 범위   : %s ~ %s\n", min(final_df$Date), max(final_df$Date)))
  cat(sprintf("  전체 레코드 : %s\n", format(nrow(final_df), big.mark = ",")))
  cat(sprintf("  완료 종목   : %d개 / 전체 %d개\n",
              length(unique(final_df$Symbol)), n_total))
  cat(sprintf("  PIN 범위    : %.4f ~ %.4f  (평균 %.4f)\n",
              min(final_df$PIN, na.rm = TRUE),
              max(final_df$PIN, na.rm = TRUE),
              mean(final_df$PIN, na.rm = TRUE)))

  # ── skip_reason별 집계 출력 ──────────────────────────────────────────────
  cat(sprintf("\n  [skip_reason 분포]\n"))
  reason_table <- table(final_df$skip_reason, useNA = "ifany")
  for (r in names(reason_table)) {
    cat(sprintf("    %-20s: %s\n", r, format(reason_table[r], big.mark = ",")))
  }

  cat(sprintf("\n  parquet (전체)     : %s\n", parquet_path))
  cat(sprintf("  CSV (샘플 1000행)  : %s\n", csv_path))
  cat(sprintf("  체크포인트         : %s\n", CHECKPOINT_DIR))
  cat(sprintf("%s\n", strrep("=", 65)))
}