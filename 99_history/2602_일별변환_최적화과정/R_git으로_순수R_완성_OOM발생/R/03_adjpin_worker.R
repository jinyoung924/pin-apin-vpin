# =============================================================================
# 03_adjpin_worker.R — AdjPIN 롤링 추정 워커
# =============================================================================
#
# 모델  : Duarte & Young (2009) — Adjusted PIN
# 함수  : PINstimation::adjpin(data, method, initialsets, verbose)
# 윈도우: 60 캘린더일 슬라이딩, 1영업일씩 이동 (PIN과 동일)
# 건너뛰기: 윈도우 내 유효 영업일(buys+sells>0) ≤ 30일
#
# 체크포인트:
#   output/{country}/adjpin/checkpoints/sym_{Symbol}.csv
#   종목 1개 완료 즉시 저장 → 재실행 시 자동 스킵
#
# adjpin() 시그니처:
#   adjpin(data, method="ECM", initialsets="GE", num_init=20,
#          restricted=list(), verbose=TRUE)
#   반환 S4: @adjpin, @psos, @parameters, @loglikelihood, @aic, @bic
# =============================================================================


#' 단일 종목의 전체 기간 AdjPIN 롤링 추정
run_adjpin_worker <- function(sym_data, symbol, checkpoint_dir, log_file = NULL) {

  sym_data <- sym_data[order(sym_data$date), ]
  all_dates <- unique(sym_data$date)

  empty_df <- data.frame(
    symbol = character(0), window_end = as.Date(character(0)),
    adjpin = numeric(0), psos = numeric(0),
    alpha = numeric(0), delta = numeric(0),
    loglik = numeric(0), aic = numeric(0), bic = numeric(0),
    valid_days = integer(0), created_at = character(0),
    stringsAsFactors = FALSE
  )

  if (length(all_dates) == 0) {
    save_checkpoint(empty_df, symbol, checkpoint_dir)
    return(list(symbol = symbol, n_rows = 0L))
  }

  results <- list()

  for (i in seq_along(all_dates)) {
    window_end   <- all_dates[i]
    window_start <- window_end - WINDOW_CALENDAR_DAYS + 1

    window_data <- sym_data[sym_data$date >= window_start &
                            sym_data$date <= window_end, ]
    if (nrow(window_data) == 0) next

    valid_days <- sum((window_data$buys + window_data$sells) > 0)
    if (valid_days <= MIN_VALID_DAYS) next

    est <- tryCatch({
      adjpin_input <- data.frame(buys = window_data$buys, sells = window_data$sells)
      res <- PINstimation::adjpin(
        data = adjpin_input, method = "ML",
        initialsets = "GE", verbose = FALSE
      )
      adjpin_val <- as.numeric(res@adjpin)
      if (is.na(adjpin_val) || is.nan(adjpin_val)) stop("AdjPIN is NA/NaN")

      params <- res@parameters
      list(
        adjpin = adjpin_val,
        psos   = as.numeric(res@psos),
        alpha  = as.numeric(params["alpha"]),
        delta  = as.numeric(params["delta"]),
        loglik = tryCatch(as.numeric(res@loglikelihood), error = function(e) NA_real_),
        aic    = tryCatch(as.numeric(res@aic), error = function(e) NA_real_),
        bic    = tryCatch(as.numeric(res@bic), error = function(e) NA_real_),
        ok     = TRUE
      )
    }, error = function(e) list(ok = FALSE))

    if (!isTRUE(est$ok)) next

    results[[length(results) + 1]] <- data.frame(
      symbol = symbol, window_end = window_end,
      adjpin = est$adjpin, psos = est$psos,
      alpha = est$alpha, delta = est$delta,
      loglik = est$loglik, aic = est$aic, bic = est$bic,
      valid_days = valid_days, stringsAsFactors = FALSE
    )
  }

  if (length(results) > 0) {
    result_df <- do.call(rbind, results)
  } else {
    result_df <- empty_df
  }

  result_df$created_at <- format(Sys.time(), "%Y%m%d_%H%M%S")
  save_checkpoint(result_df, symbol, checkpoint_dir)

  list(symbol = symbol, n_rows = nrow(result_df))
}


#' 나라 전체 종목 AdjPIN 병렬 추정
run_adjpin_all <- function(daily_bs, country, log_file = NULL) {
  output_dir     <- get_output_dir(country, "adjpin")
  checkpoint_dir <- get_checkpoint_dir(country, "adjpin")
  all_symbols    <- make_symbol_queue(daily_bs)

  # ── 체크포인트 확인 ──
  completed <- get_completed_symbols(checkpoint_dir, ext = "csv")
  remaining <- setdiff(all_symbols, completed)

  log_msg(sprintf("[AdjPIN] %s: 전체 %d종목 | 완료 %d | 잔여 %d",
                  country, length(all_symbols), length(completed), length(remaining)),
          log_file)

  if (length(remaining) == 0) {
    log_msg("[AdjPIN] 모든 종목 처리 완료. 병합 단계로 이동.", log_file)
  } else {
    future::plan(future::multisession, workers = N_CORES)
    on.exit(future::plan(future::sequential), add = TRUE)

    BATCH_SIZE <- 50L  # AdjPIN은 느리므로 배치 크기를 줄임
    batch_starts <- seq(1L, length(remaining), by = BATCH_SIZE)
    start_time <- Sys.time()

    for (bi in seq_along(batch_starts)) {
      b_start <- batch_starts[bi]
      b_end   <- min(b_start + BATCH_SIZE - 1L, length(remaining))
      batch_syms <- remaining[b_start:b_end]

      # 배치 종목별로 미리 split — 워커에는 자기 종목 데이터만 전달됨
      split_data <- lapply(
        setNames(batch_syms, batch_syms),
        function(s) daily_bs[daily_bs$Symbol == s, ]
      )

      batch_results <- furrr::future_map(batch_syms, function(sym) {
        suppressPackageStartupMessages({
          library(PINstimation); library(dplyr); library(readr); library(arrow)
        })
        sym_data <- split_data[[sym]]
        tryCatch(
          run_adjpin_worker(sym_data, sym, checkpoint_dir, log_file),
          error = function(e) {
            empty <- data.frame(
              symbol = character(0), window_end = as.Date(character(0)),
              adjpin = numeric(0), psos = numeric(0),
              alpha = numeric(0), delta = numeric(0),
              loglik = numeric(0), aic = numeric(0), bic = numeric(0),
              valid_days = integer(0), created_at = character(0),
              stringsAsFactors = FALSE
            )
            save_checkpoint(empty, sym, checkpoint_dir)
            list(symbol = sym, n_rows = 0L, error = e$message)
          }
        )
      }, .options = furrr::furrr_options(seed = TRUE))

      rm(split_data)  # 배치 끝나면 즉시 해제

      # 진행상황
      n_done  <- length(completed) + b_end
      elapsed <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      speed   <- b_end / max(elapsed, 0.01)
      eta_min <- (length(remaining) - b_end) / max(speed, 0.01)
      n_ok    <- sum(vapply(batch_results, function(r) r$n_rows > 0, logical(1)))

      log_msg(sprintf(
        "  [AdjPIN] 배치 %d/%d | 누적 %d/%d종목 | 성공 %d/%d | %.1f분 경과 | ETA %.0f분",
        bi, length(batch_starts), n_done, length(all_symbols),
        n_ok, length(batch_syms), elapsed, eta_min
      ), log_file)
      gc()  # 배치별 중간 메모리 회수
    }

    total_min <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
    log_msg(sprintf("[AdjPIN] %s 추정 완료: %.1f분 소요", country, total_min), log_file)
  }

  # ── 최종 병합 ──
  merge_checkpoints(checkpoint_dir, output_dir, country, "adjpin",
                    ext = "csv", log_file = log_file)
}
