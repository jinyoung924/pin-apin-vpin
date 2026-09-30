# =============================================================================
# [APIN 전체계산] PINstimation::adjpin() 롤링 추정
# =============================================================================
#
# 모델: Duarte & Young(2009) — Adjusted PIN (APIN)
#   표준 PIN의 비정보성 주문흐름 충격(SPOS)을 분리해 10개 파라미터로 확장.
#   핵심 지표: APIN (Adjusted PIN), PSOS (Prob. of Symmetric Order-flow Shock)
#
# ── adjpin() 실제 시그니처 (PINstimation 공식 문서 기준) ─────────────────────
#   adjpin(data, method = "ECM", initialsets = "GE", num_init = 20,
#          restricted = list(), ..., verbose = TRUE)
#
#   data        : {B, S} 열을 포함하는 data.frame (일별 매수·매도 건수)
#   method      : "ECM" (기본) 또는 "ML"
#   initialsets : "GE" | "CL" | "RANDOM" 또는 사용자 정의 data.frame
#   num_init    : GE·CL·RANDOM 방식일 때 생성할 초기점 개수 (기본값 20)
#   restricted  : list() = unrestricted (10개 파라미터 전체 추정)
#   verbose     : 진행 메시지 출력 여부
#
# ── 추정 전략 ────────────────────────────────────────────────────────────────
#   ECM + Restricted (θ=θ', 9개 자유 파라미터) 단일 방법으로 통일.
#   · Duarte & Young(2009) Table 4 Test 5: θ=θ' 제약의 기각률 9%로
#     우연 수준과 유사 → 제약이 무해(innocuous)하다고 결론.
#     논문의 모든 실증 결과(AdjPIN, PSOS, Fama-MacBeth 등)는
#     이 preferred model(Model 4, θ=θ')로 수행됨.
#   · PINstimation: restricted = list(theta = TRUE) 옵션으로 구현.
#   · 폴백(ML, 다른 Restricted) 없이 동일한 방법론으로 일관된 결과 생성
#   · 타임아웃(기본 120초)으로 교착 및 비유동 종목 방지
#   · 타임아웃 또는 수렴 실패 시 해당 윈도우를 스킵하지 않고
#     파라미터를 NA로 채우고 skip_reason에 사유를 기록
#
# ── 적응형 초기화 전략 (Adaptive Initialization) ────────────────────────────
#   60영업일 롤링 윈도우에서 연속 두 윈도우는 59일(98.3%)이 겹친다.
#   → 파라미터 공간이 거의 동일하므로 이전 추정값을 warm-start 초기점으로 활용.
#   → 전략:
#     · 첫 번째 윈도우 (i = WINDOW_SIZE):
#         initialsets = "GE",   num_init = NUM_INIT_FIRST (기본 20)
#     · 이후 윈도우 (직전 추정 성공):
#         initialsets = 이전 파라미터 1행 data.frame (warm-start)
#         + num_init = NUM_INIT_ROLL (기본 5) 개의 GE 랜덤 초기점 추가
#     · 이후 윈도우 (직전 추정 실패):
#         initialsets = "GE",   num_init = NUM_INIT_FIRST
#
# ── 병렬 전략: parLapplyLB (Load Balanced) ──────────────────────────────────
#   기존 배치 방식(parLapply + 100종목 배치)은 배치 내 가장 느린 종목이
#   끝날 때까지 전체 워커가 대기하는 동기 블로킹 문제가 있었음.
#   → parLapplyLB로 전환: 워커가 종목 1개를 끝내면 즉시 다음 종목을 받아감.
#   → 느린 종목이 있어도 나머지 워커는 계속 작업을 진행함.
#   → 체크포인트는 종목 단위로 즉시 저장되어 중단 후 재개 가능.
#
# ── 진행 상황 모니터링 ──────────────────────────────────────────────────────
#   R 콘솔에는 전체 완료 후에만 통계가 출력됨.
#   실행 중 진행 상황은 별도 터미널에서 체크포인트 파일 수로 확인:
#     PowerShell:  (Get-ChildItem checkpoints\sym_*.parquet).Count
#   종목 1개가 끝날 때마다 파일이 1개씩 늘어남.
#
# ─── 입출력 ───────────────────────────────────────────────────────────────────
# 입력  : R_output/{COUNTRY}/full_daily_bs.parquet  (01_preprocess.py 출력)
# 출력  : R_output/{COUNTRY}/apin/
#           apin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet   ← 전체 추정 결과
#           apin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv  ← 눈으로 확인용 1000행
#           checkpoints/sym_{Symbol}.parquet          ← 종목별 체크포인트
#
# ─── 실행 ─────────────────────────────────────────────────────────────────────
#   Rscript 01apin/02_r_apin.R
#
# ─── 의존 패키지 (최초 1회) ───────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "parallel", "R.utils"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(parallel)
  library(R.utils)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR <- "C:/vpin_project/vpin_project_parquet/processing_data"

# ssu :  "C:/vpin_project/vpin_project_parquet/processing_data"
# suji : "E:/vpin_project_parquet/processing_data"

COUNTRY  <- "KOR"

# ── 롤링 윈도우 파라미터 ──────────────────────────────────────────────────────
WINDOW_SIZE    <- 60   # 롤링 윈도우 크기 (영업일)
MIN_VALID_DAYS <- 30   # 윈도우 내 실제 거래일(B+S>0) 최솟값

# ── adjpin() 파라미터 ─────────────────────────────────────────────────────────
ADJPIN_METHOD   <- "ECM"   # ECM 단일 방법
ADJPIN_INITSETS <- "GE"    # 초기점 생성 방법

# Duarte & Young(2009) preferred model: θ=θ' 제약 (Model 4)
# → restricted = list(theta = TRUE) 로 구현
ADJPIN_RESTRICTED <- list(theta = TRUE)

# 적응형 num_init 설정
NUM_INIT_FIRST <- 20   # 첫 번째 윈도우 또는 직전 실패 시
NUM_INIT_ROLL  <-  5   # 직전 성공 후 warm-start 사용 시

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS <- parallel::detectCores(logical = TRUE)

# ── 타임아웃 설정 ────────────────────────────────────────────────────────────
# adjpin() 단일 호출의 시간 제한 (초)
# 정상 윈도우는 수 초 이내에 완료됨. 120초면 충분히 넉넉함.
TIMEOUT_SEC <- 120


# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", COUNTRY, "full_daily_bs.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "apin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n[APIN 전체계산] 시작: %s\n%s\n",
            strrep("=", 65), format(Sys.time(), "%Y-%m-%d %H:%M:%S"), strrep("=", 65)))
cat(sprintf("  나라코드    : %s\n", COUNTRY))
cat(sprintf("  INPUT_PATH  : %s\n", INPUT_PATH))
cat(sprintf("  OUTPUT_DIR  : %s\n", OUTPUT_DIR))
cat(sprintf("  윈도우 크기 : %d영업일  |  최소 유효일: %d일\n", WINDOW_SIZE, MIN_VALID_DAYS))
cat(sprintf("  Method      : %s Restricted (θ=θ')  |  InitSets: %s\n", ADJPIN_METHOD, ADJPIN_INITSETS))
cat(sprintf("  num_init    : 첫 윈도우 = %d  |  롤링 = %d (warm-start)\n",
            NUM_INIT_FIRST, NUM_INIT_ROLL))
cat(sprintf("  CPU 코어    : %d개\n", NUM_WORKERS))
cat(sprintf("  타임아웃    : %d초 (%.0f분)\n", TIMEOUT_SEC, TIMEOUT_SEC / 60))
cat(sprintf("  병렬 방식   : parLapplyLB (동적 할당, 배치 블로킹 없음)\n"))


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
# [2] 재개 확인 — 체크포인트 파일이 있는 종목은 자동 스킵
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
# [3] 워커 함수 정의
# =============================================================================

worker_process_symbol <- function(args) {
  suppressPackageStartupMessages({
    library(PINstimation)
    library(arrow)
    library(R.utils)
  })

  sym_df         <- args$sym_df
  symbol         <- args$symbol
  checkpoint_dir <- args$checkpoint_dir
  window_size    <- args$window_size
  min_valid_days <- args$min_valid_days
  method         <- args$method
  initsets       <- args$initsets
  restricted     <- args$restricted       # Duarte & Young(2009) θ=θ' 제약
  num_init_first <- args$num_init_first
  num_init_roll  <- args$num_init_roll
  timeout_sec    <- args$timeout_sec

  n_days <- nrow(sym_df)

  # ── warm-start용 data.frame 생성 ──────────────────────────────────────────
  # Duarte & Young(2009) preferred model: θ=θ' 제약 적용
  # → warm-start 초기점에서도 thetap = theta로 설정하여 제약 일관성 유지
  make_warm_start_df <- function(prev_params) {
    data.frame(
      alpha  = prev_params["alpha"],
      delta  = prev_params["delta"],
      theta  = prev_params["theta"],
      thetap = prev_params["theta"],   # θ=θ' 제약: thetap을 theta와 동일하게 설정
      eps.b  = prev_params["eps.b"],
      eps.s  = prev_params["eps.s"],
      mu.b   = prev_params["mu.b"],
      mu.s   = prev_params["mu.s"],
      d.b    = prev_params["d.b"],
      d.s    = prev_params["d.s"],
      stringsAsFactors = FALSE
    )
  }

  # ── adjpin() 단일 호출 + 타임아웃 래퍼 ────────────────────────────────────
  # 반환값: list(converged, reason, ...) 형태
  #   converged = TRUE  → 추정 성공, reason = "Success"
  #   converged = FALSE → 실패, reason = "Timeout" | "Est_Error"
  safe_adjpin <- function(window_df, cur_initialsets, cur_num_init) {
    tryCatch({
      withTimeout({
        res <- adjpin(
          data        = window_df,
          method      = method,
          initialsets = cur_initialsets,
          num_init    = cur_num_init,
          restricted  = restricted,   # Duarte & Young(2009) preferred model: θ=θ'
          verbose     = FALSE
        )

        apin_val <- as.numeric(res@adjpin)
        if (is.na(apin_val) || is.nan(apin_val)) {
          return(list(converged = FALSE, reason = "Est_Error"))
        }

        params <- res@parameters
        list(
          converged  = TRUE,
          reason     = "Success",
          a          = as.numeric(params["alpha"]),
          d          = as.numeric(params["delta"]),
          t1         = as.numeric(params["theta"]),
          t2         = as.numeric(params["thetap"]),
          ub         = as.numeric(params["mu.b"]),
          us         = as.numeric(params["mu.s"]),
          eb         = as.numeric(params["eps.b"]),
          es         = as.numeric(params["eps.s"]),
          pb         = as.numeric(params["d.b"]),
          ps         = as.numeric(params["d.s"]),
          APIN       = apin_val,
          PSOS       = as.numeric(res@psos),
          params_raw = params
        )
      }, timeout = timeout_sec, onTimeout = "error")
    }, error = function(e) {
      # withTimeout의 타임아웃 에러 메시지 패턴으로 Timeout과 기타 에러를 구분
      if (grepl("elapsed time limit|timeout", e$message, ignore.case = TRUE)) {
        list(converged = FALSE, reason = "Timeout")
      } else {
        list(converged = FALSE, reason = "Est_Error")
      }
    })
  }

  # ── NA 행 생성 헬퍼: 파라미터를 NA로 채우고 skip_reason을 기록 ────────────
  make_na_row <- function(symbol, date, valid_days, reason) {
    data.frame(
      Symbol      = symbol,
      Date        = as.Date(date),
      valid_days  = valid_days,
      a           = NA_real_,
      d           = NA_real_,
      t1          = NA_real_,
      t2          = NA_real_,
      ub          = NA_real_,
      us          = NA_real_,
      eb          = NA_real_,
      es          = NA_real_,
      pb          = NA_real_,
      ps          = NA_real_,
      APIN        = NA_real_,
      PSOS        = NA_real_,
      skip_reason = reason,
      stringsAsFactors = FALSE
    )
  }

  # ── 상태 변수 ─────────────────────────────────────────────────────────────
  prev_params   <- NULL
  results       <- list()
  count_success <- 0L
  count_skip    <- 0L

  if (n_days >= window_size) {
    for (i in window_size:n_days) {
      s          <- i - window_size + 1
      window_B   <- sym_df$B[s:i]
      window_S   <- sym_df$S[s:i]
      valid_days <- sum((window_B + window_S) > 0)

      # 유효 거래일 부족 → NA 행 + skip_reason = "Low_Valid_Days"
      if (valid_days < min_valid_days) {
        count_skip <- count_skip + 1L
        results[[length(results) + 1]] <- make_na_row(
          symbol, sym_df$Date[i], valid_days, "Low_Valid_Days"
        )
        next
      }

      window_df <- data.frame(B = window_B, S = window_S)

      # 적응형 초기화: warm-start 또는 fresh GE
      if (!is.null(prev_params)) {
        cur_initialsets <- make_warm_start_df(prev_params)
        cur_num_init    <- num_init_roll
      } else {
        cur_initialsets <- initsets
        cur_num_init    <- num_init_first
      }

      # ── ECM Restricted (θ=θ') 추정 (타임아웃 내 수렴 시도) ────────────
      est <- safe_adjpin(window_df, cur_initialsets, cur_num_init)

      if (!isTRUE(est$converged)) {
        # 수렴 실패 / 타임아웃 → NA 행 + skip_reason (Timeout 또는 Est_Error)
        count_skip  <- count_skip + 1L
        prev_params <- NULL
        results[[length(results) + 1]] <- make_na_row(
          symbol, sym_df$Date[i], valid_days, est$reason
        )
        next
      }

      # 수렴 성공 → 다음 윈도우를 위해 파라미터 저장
      count_success <- count_success + 1L
      prev_params   <- est$params_raw

      results[[length(results) + 1]] <- data.frame(
        Symbol      = symbol,
        Date        = as.Date(sym_df$Date[i]),
        valid_days  = valid_days,
        a           = est$a,
        d           = est$d,
        t1          = est$t1,
        t2          = est$t2,
        ub          = est$ub,
        us          = est$us,
        eb          = est$eb,
        es          = est$es,
        pb          = est$pb,
        ps          = est$ps,
        APIN        = est$APIN,
        PSOS        = est$PSOS,
        skip_reason = "Success",
        stringsAsFactors = FALSE
      )
    }
  }

  # ── 결과 조립 + 체크포인트 저장 ───────────────────────────────────────────
  if (length(results) > 0) {
    result_df <- do.call(rbind, results)
  } else {
    result_df <- data.frame(
      Symbol = character(0), Date = as.Date(character(0)),
      valid_days = integer(0),
      a = numeric(0), d = numeric(0), t1 = numeric(0), t2 = numeric(0),
      ub = numeric(0), us = numeric(0), eb = numeric(0), es = numeric(0),
      pb = numeric(0), ps = numeric(0), APIN = numeric(0), PSOS = numeric(0),
      skip_reason = character(0),
      stringsAsFactors = FALSE
    )
  }

  out_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".parquet"))
  arrow::write_parquet(result_df, out_path, compression = "zstd")

  return(list(
    symbol  = symbol,
    n_rows  = nrow(result_df),
    success = count_success,
    skipped = count_skip
  ))
}


# =============================================================================
# [3-b] 종목 데이터 분할 + 병렬 처리 (parLapplyLB — 동적 할당)
# =============================================================================

if (length(remaining_syms) > 0) {

  cat(sprintf("\n[3] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  split_bs <- split(daily_bs[, c("Date", "B", "S")], daily_bs$Symbol)

  symbol_args <- lapply(remaining_syms, function(sym) {
    rows <- split_bs[[sym]]
    rows <- rows[order(rows$Date), ]
    list(
      sym_df         = rows,
      symbol         = sym,
      checkpoint_dir = CHECKPOINT_DIR,
      window_size    = WINDOW_SIZE,
      min_valid_days = MIN_VALID_DAYS,
      method         = ADJPIN_METHOD,
      initsets       = ADJPIN_INITSETS,
      restricted     = ADJPIN_RESTRICTED,   # Duarte & Young(2009) θ=θ' 제약 전달
      num_init_first = NUM_INIT_FIRST,
      num_init_roll  = NUM_INIT_ROLL,
      timeout_sec    = TIMEOUT_SEC
    )
  })
  cat("  분할 완료\n")

  # ==========================================================================
  # [4] 병렬 처리 — parLapplyLB (Load Balanced, 동적 할당)
  # ==========================================================================
  #
  # parLapplyLB는 워커가 작업을 끝내는 즉시 큐에서 다음 종목을 가져감.
  # → 느린 종목(비유동주, 수렴 어려움)이 있어도 나머지 워커는 멈추지 않음.
  # → 기존 배치 방식에서 1개 느린 종목이 99개 완료된 워커를 블로킹하던 문제 해소.
  #
  # 진행 상황 확인 (별도 터미널에서):
  #   PowerShell: (Get-ChildItem checkpoints\sym_*.parquet).Count
  # ==========================================================================

  n_remaining <- length(remaining_syms)
  n_workers   <- min(NUM_WORKERS, n_remaining)

  cat(sprintf(paste0(
    "\n[4] adjpin() 병렬 추정 — ECM Restricted θ=θ' (parLapplyLB)\n",
    "    워커 %d개 | 윈도우 %d일 | 타임아웃 %d초(%.0f분)\n",
    "    수렴 실패 또는 타임아웃 → NA행 기록 (skip_reason)\n%s\n"),
    n_workers, WINDOW_SIZE, TIMEOUT_SEC, TIMEOUT_SEC / 60,
    strrep("-", 65)))

  cat("    진행 상황 확인 (별도 터미널에서 체크포인트 파일 수 확인):\n")
  cat(sprintf("      (Get-ChildItem '%s\\sym_*.parquet').Count\n\n",
              gsub("/", "\\\\", CHECKPOINT_DIR)))

  start_time <- Sys.time()
  cl <- makeCluster(n_workers, type = "PSOCK")
  on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

  cat(sprintf("    %s  추정 시작 (%d종목) ...\n",
              format(Sys.time(), "%H:%M:%S"), n_remaining))

  all_results <- tryCatch(
    parLapplyLB(cl, symbol_args, worker_process_symbol),
    error = function(e) {
      cat(sprintf("\n  [Error] 병렬 처리 오류: %s\n", e$message))
      NULL
    }
  )

  elapsed_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
  cat(sprintf("    %s  추정 완료 (%.1f분 소요)\n\n",
              format(Sys.time(), "%H:%M:%S"), elapsed_min))

  # ── 전체 통계 요약 ──────────────────────────────────────────────────────
  if (!is.null(all_results)) {
    total_success <- sum(vapply(all_results, function(x) x$success, integer(1)))
    total_skip    <- sum(vapply(all_results, function(x) x$skipped, integer(1)))
    total_windows <- total_success + total_skip

    cat(sprintf("  [추정 통계 — 전체 %s 윈도우]\n",
                format(total_windows, big.mark = ",")))
    cat(sprintf("    성공 (ECM)      : %s (%.1f%%)\n",
                format(total_success, big.mark = ","),
                if (total_windows > 0) total_success / total_windows * 100 else 0))
    cat(sprintf("    스킵            : %s (%.1f%%)\n",
                format(total_skip, big.mark = ","),
                if (total_windows > 0) total_skip / total_windows * 100 else 0))
    cat(sprintf("    총 소요 시간    : %.1f분\n", elapsed_min))
    cat(sprintf("    종목당 평균     : %.1f초\n", elapsed_min * 60 / n_remaining))
  }
}


# =============================================================================
# [5] 체크포인트 병합 → 최종 결과 저장
# =============================================================================

cat(sprintf("\n[5] 체크포인트 병합 중...\n"))

all_ckpt_files <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.parquet$", full.names = TRUE)
cat(sprintf("  체크포인트 파일 수: %d개\n", length(all_ckpt_files)))

all_dfs   <- lapply(all_ckpt_files, function(f) {
  tryCatch(arrow::read_parquet(f), error = function(e) NULL)
})
non_empty <- Filter(function(df) !is.null(df) && nrow(df) > 0, all_dfs)

if (length(non_empty) == 0) {
  cat("[Warning] 추정 결과가 없습니다.\n")
} else {
  final_df           <- do.call(rbind, non_empty)
  rownames(final_df) <- NULL
  final_df           <- final_df[order(final_df$Symbol, final_df$Date), ]

  run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
  result_stem  <- sprintf("apin_%s_%s", COUNTRY, run_id)
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
  cat(sprintf("  APIN 범위   : %.4f ~ %.4f  (평균 %.4f)\n",
              min(final_df$APIN, na.rm = TRUE),
              max(final_df$APIN, na.rm = TRUE),
              mean(final_df$APIN, na.rm = TRUE)))

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