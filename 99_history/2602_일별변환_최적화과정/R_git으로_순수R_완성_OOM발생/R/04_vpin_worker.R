# =============================================================================
# 04_vpin_worker.R — VPIN 수동 계산 워커
# =============================================================================
#
# 모델  : Easley, de Prado & O'Hara (2012) — Volume-Synchronized PIN
# 방법  : LR 컬럼 직접 활용 (PINstimation::vpin() 내부 재분류 건너뜀)
#
# 계산 로직:
#   1) ADV = 일평균 거래량
#   2) VBS = ADV / samplength(50)
#   3) 누적 거래량 기준 버킷 할당
#   4) 버킷별 buy_vol/sell_vol 집계
#   5) OI = |buy_vol - sell_vol|
#   6) VPIN = rolling_mean(OI, window=50) / VBS
#
# 데이터 로딩: Arrow predicate pushdown — 종목별로 필요한 행만 읽음
#
# 체크포인트:
#   output/{country}/vpin/checkpoints/sym_{Symbol}.csv
# =============================================================================


#' 단일 종목의 VPIN 수동 계산
run_vpin_worker <- function(sym_data, symbol, checkpoint_dir,
                            samplength = VPIN_SAMPLENGTH, log_file = NULL) {

  sym_data <- sym_data[order(sym_data$timestamp), ]

  empty_df <- data.frame(
    symbol = character(0), bucket = integer(0),
    bucket_date = as.Date(character(0)),
    buy_vol = numeric(0), sell_vol = numeric(0),
    oi = numeric(0), vpin = numeric(0), vbs = numeric(0),
    created_at = character(0), stringsAsFactors = FALSE
  )

  if (nrow(sym_data) == 0) {
    save_checkpoint(empty_df, symbol, checkpoint_dir)
    return(list(symbol = symbol, n_rows = 0L))
  }

  # ── 1) ADV ──
  sym_data$trade_date <- as.Date(sym_data$timestamp)
  daily_vol <- tapply(sym_data$Volume, sym_data$trade_date, sum, na.rm = TRUE)
  adv <- mean(daily_vol, na.rm = TRUE)

  if (is.na(adv) || adv <= 0) {
    save_checkpoint(empty_df, symbol, checkpoint_dir)
    return(list(symbol = symbol, n_rows = 0L))
  }

  # ── 2) VBS ──
  vbs <- adv / samplength
  if (vbs <= 0) {
    save_checkpoint(empty_df, symbol, checkpoint_dir)
    return(list(symbol = symbol, n_rows = 0L))
  }

  # ── 3) 버킷 할당 ──
  cum_vol <- cumsum(sym_data$Volume)
  bucket_id <- ceiling(cum_vol / vbs)
  bucket_id[bucket_id == 0] <- 1L
  sym_data$bucket <- as.integer(bucket_id)

  n_buckets <- max(sym_data$bucket, na.rm = TRUE)
  if (n_buckets < samplength) {
    save_checkpoint(empty_df, symbol, checkpoint_dir)
    return(list(symbol = symbol, n_rows = 0L))
  }

  # ── 4) 버킷별 집계 ──
  bucket_agg <- sym_data |>
    dplyr::group_by(bucket) |>
    dplyr::summarise(
      bucket_date = max(trade_date, na.rm = TRUE),
      buy_vol     = sum(Volume[LR == 1L], na.rm = TRUE),
      sell_vol    = sum(Volume[LR == -1L], na.rm = TRUE),
      .groups     = "drop"
    ) |>
    dplyr::arrange(bucket)

  # ── 5) OI ──
  bucket_agg$oi <- abs(bucket_agg$buy_vol - bucket_agg$sell_vol)

  # ── 6) VPIN ──
  bucket_agg$vpin <- zoo::rollapply(
    bucket_agg$oi, width = samplength,
    FUN = mean, align = "right", fill = NA_real_
  ) / vbs

  # 결과 구성
  result_df <- data.frame(
    symbol      = symbol,
    bucket      = bucket_agg$bucket,
    bucket_date = bucket_agg$bucket_date,
    buy_vol     = bucket_agg$buy_vol,
    sell_vol    = bucket_agg$sell_vol,
    oi          = bucket_agg$oi,
    vpin        = bucket_agg$vpin,
    vbs         = vbs,
    stringsAsFactors = FALSE
  )

  result_df$created_at <- format(Sys.time(), "%Y%m%d_%H%M%S")
  save_checkpoint(result_df, symbol, checkpoint_dir)

  list(symbol = symbol, n_rows = nrow(result_df))
}


#' 나라 전체 종목 VPIN — Arrow predicate pushdown + 병렬 계산
#'
#' arrow::open_dataset()로 lazy Dataset을 열고,
#' 각 워커가 filter(Symbol == sym)으로 자기 종목 데이터만 로드.
#' Arrow 엔진이 parquet row group 메타데이터를 활용해 필요한 행만 읽음.
#'
#' @param all_symbols character vector: 처리할 종목 코드 목록
#' @param country 나라 코드
#' @param log_file 로그 파일 경로
run_vpin_all_arrow <- function(all_symbols, country, log_file = NULL) {
  output_dir     <- get_output_dir(country, "vpin")
  checkpoint_dir <- get_checkpoint_dir(country, "vpin")
  input_dir      <- get_input_dir(country)

  # ── 체크포인트 확인 ──
  completed <- get_completed_symbols(checkpoint_dir, ext = "csv")
  remaining <- setdiff(all_symbols, completed)

  log_msg(sprintf("[VPIN] %s: 전체 %d종목 | 완료 %d | 잔여 %d",
                  country, length(all_symbols), length(completed), length(remaining)),
          log_file)

  if (length(remaining) == 0) {
    log_msg("[VPIN] 모든 종목 처리 완료. 병합 단계로 이동.", log_file)
  } else {
    future::plan(future::multisession, workers = N_CORES)
    on.exit(future::plan(future::sequential), add = TRUE)

    BATCH_SIZE <- 100L
    batch_starts <- seq(1L, length(remaining), by = BATCH_SIZE)
    start_time <- Sys.time()

    for (bi in seq_along(batch_starts)) {
      b_start <- batch_starts[bi]
      b_end   <- min(b_start + BATCH_SIZE - 1L, length(remaining))
      batch_syms <- remaining[b_start:b_end]

      batch_results <- furrr::future_map(batch_syms, function(sym) {
        suppressPackageStartupMessages({
          library(dplyr); library(zoo); library(readr); library(arrow)
        })

        # 각 워커가 독립적으로 Dataset을 열고 predicate pushdown으로 로드
        ds <- arrow::open_dataset(input_dir, format = "parquet")
        sym_data <- load_vpin_sym(sym, ds)

        if (is.null(sym_data) || nrow(sym_data) == 0) {
          empty <- data.frame(
            symbol = character(0), bucket = integer(0),
            bucket_date = as.Date(character(0)),
            buy_vol = numeric(0), sell_vol = numeric(0),
            oi = numeric(0), vpin = numeric(0), vbs = numeric(0),
            created_at = character(0), stringsAsFactors = FALSE
          )
          save_checkpoint(empty, sym, checkpoint_dir)
          return(list(symbol = sym, n_rows = 0L))
        }

        result <- tryCatch(
          run_vpin_worker(sym_data, sym, checkpoint_dir),
          error = function(e) {
            empty <- data.frame(
              symbol = character(0), bucket = integer(0),
              bucket_date = as.Date(character(0)),
              buy_vol = numeric(0), sell_vol = numeric(0),
              oi = numeric(0), vpin = numeric(0), vbs = numeric(0),
              created_at = character(0), stringsAsFactors = FALSE
            )
            save_checkpoint(empty, sym, checkpoint_dir)
            list(symbol = sym, n_rows = 0L, error = e$message)
          }
        )

        rm(sym_data); gc()
        result
      }, .options = furrr::furrr_options(seed = TRUE))

      # 진행상황
      n_done  <- length(completed) + b_end
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      speed   <- b_end / max(elapsed, 0.01)
      eta_min <- (length(remaining) - b_end) / max(speed, 0.01)
      n_ok    <- sum(vapply(batch_results, function(r) r$n_rows > 0, logical(1)))

      log_msg(sprintf(
        "  [VPIN] 배치 %d/%d | 누적 %d/%d종목 | 성공 %d/%d | %.1f분 경과 | ETA %.0f분",
        bi, length(batch_starts), n_done, length(all_symbols),
        n_ok, length(batch_syms), elapsed, eta_min
      ), log_file)
      gc()  # 배치별 중간 메모리 회수
    }

    total_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    log_msg(sprintf("[VPIN] %s 계산 완료: %.1f분 소요", country, total_min), log_file)
  }

  # ── 최종 병합 ──
  merge_checkpoints(checkpoint_dir, output_dir, country, "vpin",
                    ext = "csv", log_file = log_file)
}
