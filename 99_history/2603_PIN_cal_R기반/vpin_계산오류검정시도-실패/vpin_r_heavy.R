# =============================================================================
# [VPIN 파이프라인 - Step 2] PINstimation::vpin() 버킷별 VPIN 계산 (병렬)
# =============================================================================
#
# ─── PINstimation::vpin() 내부 동작 (model_vpin.R 참조) ────────────────────
#
#   vpin(data, timebarsize, buckets, samplength, tradinghours, verbose)
#
#   data 는 반드시 틱 레벨: {timestamp, price, volume}  (열 이름 무관, 순서만)
#
#   내부 처리 흐름:
#     Step 0: data[,1:3] → timestamp, price, volume 으로 rename
#     Step 1: timestamp를 timebarsize초 간격(interval)에 매핑
#             → interval 별로:
#                 dp  = last(price) - first(price)    ← 바 내 가격변동
#                 tbv = sum(volume)                   ← 바 내 총거래량
#     Step 2: VBS = (총거래량 / 거래일수) / buckets,  sdp = sd(dp)
#     Step 3~5: 큰 바 분할, 볼륨 버킷 할당, 경계 리밸런싱
#     Step 6: z = dp/sdp → zb = Φ(z) → bvol = tbv*zb, svol = tbv*(1-zb)
#             → 버킷별 agg.bvol, agg.svol, aoi 집계
#     Step 7: VPIN = rolling_sum(aoi, samplength) / (samplength * VBS)
#
# ─── dp=0 문제와 해결 ──────────────────────────────────────────────────────
#
#   Python 전처리가 1분봉으로 집계하면 bar당 행이 1개 → dp = 0
#
#   해결: Python이 Open(=first tick price) + Close(=last tick price) 저장
#         R에서 bar 1개를 pseudo-tick 2개로 확장:
#
#         tick1: (timestamp = Datetime,      price = Open,   volume = 0)
#         tick2: (timestamp = Datetime + 1s, price = Close,  volume = Volume)
#
#         vpin()이 60초 바로 재집계하면:
#           dp  = last(Close) - first(Open) = Close - Open   ← 올바른 가격변동
#           tbv = 0 + Volume = Volume                        ← 올바른 거래량
#
# ─── 날짜(Date) 추출 ──────────────────────────────────────────────────────
#
#   @bucketdata 의 endtime 은 POSIXct (날짜+시간 포함)
#     → as.Date(endtime) 로 버킷이 속한 날짜를 직접 추출하여 Date 열로 저장
#
# ─── 메모리 절약 전략 (종목별 파일 입력) ──────────────────────────────────
#
#   이전 버전은 all_1m_bars.parquet 단일 파일을 bars_dt로 통째로 읽어
#   split(bars_dt, Symbol) 하는 구조 → 4000종목 시 RAM 부족.
#
#   현재 버전:
#     1) Python 전처리(vpin_pre.py)가 bars_by_symbol/{종목코드}.parquet
#        형태로 종목별 개별 파일을 생성함
#     2) R 메인 프로세스는 파일 목록과 행 수만 파악 (데이터 미로딩)
#     3) 각 PSOCK 워커가 자신의 종목 파일 경로를 받아 직접 읽고 처리
#        → 메인 프로세스 메모리에는 종목 데이터가 올라가지 않음
#     4) 워커도 종목 1개만 메모리에 보유 → 처리 후 GC
#
#   메인 프로세스 메모리 사용: O(종목 수) — 파일경로 목록 + 행 수 벡터만
#   워커 프로세스 메모리 사용: O(종목 1개) — 해당 종목 데이터 + vpin 결과만
#
# ─── 입출력 ──────────────────────────────────────────────────────────────────
#   입력: R_output/{COUNTRY}/vpin/bars_by_symbol/
#         종목별 parquet 파일들 (예: 005930.parquet, 000660.parquet ...)
#         스키마: Symbol, Datetime, Open, Close, Volume
#
#   출력: R_output/{COUNTRY}/vpin/
#         vpin_{COUNTRY}_{ID}.parquet             ← 버킷별 결과 (Date 열 포함)
#         vpin_{COUNTRY}_{ID}_sample1000.csv      ← 확인용
#         checkpoints/sym_{Symbol}.rds            ← 종목별 체크포인트
#
#   ★ Date 열은 버킷의 endtime(POSIXct)에서 as.Date()로 직접 추출.
#     일별 VPIN(@dailyvpin) 계산 및 조인 불필요.
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   Rscript vpin_r.R
#
# ─── 의존 패키지 ─────────────────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "data.table", "dplyr", "parallel"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(data.table)
  library(dplyr)
  library(parallel)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR <- "E:/vpin_project_parquet/processing_data"
COUNTRY  <- "CA"

# ── vpin() 파라미터 ──────────────────────────────────────────────────────────
TIMEBARSIZE_SEC <- 60L    # 타임바 크기(초). 1분봉 = 60
BUCKETS_PER_DAY <- 50L    # 하루당 버킷 수
SAMPLENGTH      <- 50L    # 롤링 윈도우 버킷 수
TRADINGHOURS    <- 7L     # 하루 거래 시간(시간). KRX 6.5h → 7로 올림

# ── 필터링 파라미터 ──────────────────────────────────────────────────────────
MIN_ROWS <- 500           # 종목당 최소 행 수 (미만 스킵)

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS <- max(1L, parallel::detectCores(logical = TRUE) - 2L)
BATCH_N     <- 100L


# =============================================================================
# (이하 수정 불필요)
# =============================================================================

SYMBOL_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin",
                            "bars_by_symbol")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
LOG_PATH       <- file.path(OUTPUT_DIR, "vpin_errors.log")

dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n", strrep("=", 65)))
cat(sprintf("[VPIN 계산] 시작: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  나라코드       : %s\n", COUNTRY))
cat(sprintf("  입력 폴더      : %s\n", SYMBOL_DIR))
cat(sprintf("  출력 폴더      : %s\n", OUTPUT_DIR))
cat(sprintf("  timebarsize    : %d초\n", TIMEBARSIZE_SEC))
cat(sprintf("  buckets/day    : %d\n", BUCKETS_PER_DAY))
cat(sprintf("  samplength     : %d\n", SAMPLENGTH))
cat(sprintf("  tradinghours   : %dh\n", TRADINGHOURS))
cat(sprintf("  최소 행 수     : %d\n", MIN_ROWS))
cat(sprintf("  병렬 워커      : %d개\n", NUM_WORKERS))
cat(sprintf("  배치 크기      : %d종목\n", BATCH_N))


# =============================================================================
# [1] 종목별 파일 목록 스캔 + 스키마 검증
# =============================================================================
#
# bars_by_symbol/ 폴더의 .parquet 파일들을 열거한다.
# 데이터 자체는 읽지 않고 파일 목록만 파악 → 메모리 사용 거의 없음.
# 스키마 검증은 첫 번째 파일만으로 수행한다.
# =============================================================================

if (!dir.exists(SYMBOL_DIR))
  stop(sprintf("[Error] 입력 폴더 없음: %s\n  먼저 vpin_pre.py를 실행하세요.", SYMBOL_DIR))

cat(sprintf("\n[1] 종목별 파일 스캔 중...\n"))

sym_files <- list.files(SYMBOL_DIR, pattern = "\\.parquet$", full.names = TRUE)
if (length(sym_files) == 0)
  stop(sprintf("[Error] parquet 파일 없음: %s\n  먼저 vpin_pre.py를 실행하세요.", SYMBOL_DIR))

cat(sprintf("  파일 수: %d\n", length(sym_files)))

# ── 스키마 검증: 첫 번째 파일에서 Open/Close 열 확인 ──
sample_schema <- names(arrow::read_parquet(sym_files[1], as_data_frame = TRUE))
if (!all(c("Open", "Close") %in% sample_schema)) {
  if ("Price" %in% sample_schema) {
    stop(paste0(
      "[Error] 입력 파일에 'Open', 'Close' 열이 없고 'Price'만 있습니다.\n",
      "  dp=0 문제를 해결하려면 수정된 vpin_pre.py로 재실행하세요.\n",
      "  (FORCE_REPROCESS = True 설정 후 python vpin_pre.py)"
    ))
  }
  stop("[Error] 입력 파일에 필수 열 'Open', 'Close'가 없습니다.")
}
cat(sprintf("  스키마 검증 OK: %s\n", paste(sample_schema, collapse = ", ")))


# =============================================================================
# [2] 행 수 기반 필터링 (데이터 미로딩)
# =============================================================================
#
# 각 종목 파일의 행 수만 빠르게 확인하여 MIN_ROWS 미만인 종목을 제외한다.
# arrow::read_parquet(as_data_frame=FALSE)는 Arrow Table을 반환하며,
# $num_rows는 parquet 파일 메타데이터에서 바로 읽히므로
# 실제 데이터를 R 메모리에 올리지 않는다.
# =============================================================================

cat(sprintf("\n[2] 종목별 행 수 확인 + 필터링\n"))

# 종목코드를 파일명에서 추출 (확장자 제거)
sym_names <- tools::file_path_sans_ext(basename(sym_files))

# 행 수 확인: Arrow Table의 num_rows는 parquet 메타데이터에서 직접 읽힘
# (as_data_frame=FALSE → 데이터를 R 메모리에 올리지 않고 Arrow Table 반환)
sym_nrows <- vapply(sym_files, function(f) {
  tbl <- arrow::read_parquet(f, as_data_frame = FALSE)
  tbl$num_rows
}, integer(1), USE.NAMES = FALSE)

names(sym_nrows) <- sym_names

# 필터링
valid_mask <- sym_nrows >= MIN_ROWS
skip_count <- sum(!valid_mask)

if (skip_count > 0)
  cat(sprintf("  최소 행 수(%d) 미달 → %d종목 제외\n", MIN_ROWS, skip_count))

valid_files  <- sym_files[valid_mask]
valid_names  <- sym_names[valid_mask]
valid_nrows  <- sym_nrows[valid_mask]

# 전체 통계 (합산만 — 데이터 안 읽음)
total_rows_all <- sum(as.numeric(valid_nrows))

cat(sprintf("  유효 종목: %d개  |  총 행 수(추정): %s\n",
            length(valid_names), format(total_rows_all, big.mark = ",")))

all_symbols <- sort(valid_names)
n_total     <- length(all_symbols)

# 파일 경로 lookup 테이블 (종목명 → 파일 경로)
sym_file_map <- setNames(valid_files, valid_names)


# =============================================================================
# [3] 체크포인트 확인 (재개 지원)
# =============================================================================

cat(sprintf("\n[3] 체크포인트 확인\n"))

ckpt_files     <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.rds$")
completed_syms <- gsub("^sym_|\\.rds$", "", ckpt_files)
remaining_syms <- setdiff(all_symbols, completed_syms)

cat(sprintf("  완료: %d종목  |  잔여: %d종목\n",
            length(completed_syms), length(remaining_syms)))


# =============================================================================
# [4] 워커 함수 정의 + 병렬 처리
# =============================================================================
#
# 워커 내부에서 1분봉 (Open, Close, Volume) → pseudo-tick 2개로 확장 후
# PINstimation::vpin() 호출.
#
# ── pseudo-tick 확장 원리 (PINstimation model_vpin.R 기준) ──
#
#   vpin() 내부 Step 1 (line 614-627):
#     dataset를 interval로 group_by한 뒤
#       dp  = dplyr::last(price) - dplyr::first(price)
#       tbv = sum(volume, na.rm = TRUE)
#
#   → 같은 60초 interval 안에 2행이 있어야 dp ≠ 0
#
#   1분봉 1개 (Datetime=t, Open=O, Close=C, Volume=V) 를:
#     row1: (timestamp = t,      price = O, volume = 0)
#     row2: (timestamp = t + 1s, price = C, volume = V)
#   로 확장하면:
#     - 둘 다 같은 60초 interval [t, t+60) 에 속함
#     - arrange(interval, timestamp) 후:
#         first(price) = O  (t가 더 이르므로)
#         last(price)  = C  (t+1이 더 늦으므로)
#     - dp  = C - O  ✓
#     - tbv = 0 + V  ✓
#
# ── 메모리 절약: 워커가 파일을 직접 읽는 구조 ──
#
#   이전 버전: 메인 프로세스가 bars_dt 전체를 메모리에 올린 뒤 split →
#              각 워커에 data.frame 전송. 메인 + 워커 모두 데이터 보유.
#   현재 버전: 워커에 파일 경로(sym_file)만 전달.
#              워커가 arrow::read_parquet()로 자기 종목만 읽음.
#              메인 프로세스 메모리에는 종목 데이터가 전혀 올라가지 않음.
#              PSOCK 직렬화 비용도 경로 문자열 1개뿐이라 거의 없음.
#
# ── 출력: 버킷별 결과 + Date 열 ──
#
#   @bucketdata 에서 endtime (POSIXct) 추출 → as.Date() 로 날짜 부여
# =============================================================================

worker_fn <- function(args) {
  suppressPackageStartupMessages({
    library(PINstimation)
    library(arrow)
  })

  symbol         <- args$symbol
  sym_file       <- args$sym_file       # ★ 파일 경로 (데이터 대신)
  checkpoint_dir <- args$checkpoint_dir
  tbs            <- args$timebarsize_sec
  bpd            <- args$buckets_per_day
  sl             <- args$samplength
  th             <- args$tradinghours

  # 빈 결과 템플릿
  empty_bucket <- data.frame(
    Symbol = character(0), BucketNo = integer(0),
    Date = as.Date(character(0)),
    Starttime = as.POSIXct(character(0)),
    Endtime = as.POSIXct(character(0)),
    AggBuyVol = numeric(0), AggSellVol = numeric(0),
    AOI = numeric(0), VPIN = numeric(0),
    stringsAsFactors = FALSE
  )
  ckpt_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".rds"))

  # ── ★ 워커가 직접 파일에서 읽기 ──
  sym_df <- tryCatch(
    arrow::read_parquet(sym_file, as_data_frame = TRUE),
    error = function(e) NULL
  )
  if (is.null(sym_df) || nrow(sym_df) == 0) {
    saveRDS(empty_bucket, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("파일 읽기 실패 또는 빈 파일: ", sym_file)))
  }

  # ── POSIXct tzone 설정 ──
  ts_raw <- sym_df$Datetime
  if (inherits(ts_raw, "POSIXct")) {
    attr(ts_raw, "tzone") <- "Asia/Seoul"
  } else if (is.numeric(ts_raw)) {
    ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01", tz = "Asia/Seoul")
  } else {
    ts_raw <- as.POSIXct(as.character(ts_raw), tz = "Asia/Seoul")
  }

  n_bars <- length(ts_raw)
  open_prices  <- as.numeric(sym_df$Open)
  close_prices <- as.numeric(sym_df$Close)
  volumes      <- as.numeric(sym_df$Volume)

  # 읽은 데이터 즉시 해제
  rm(sym_df); gc()

  # ══════════════════════════════════════════════════════════════════════
  # pseudo-tick 확장: bar 1개 → tick 2개
  #   tick1: (t,     Open,  0)       ← vpin() 내부에서 first(price)가 됨
  #   tick2: (t+1s,  Close, Volume)  ← vpin() 내부에서 last(price)가 됨
  # ══════════════════════════════════════════════════════════════════════
  vpin_input <- data.frame(
    timestamp = c(ts_raw, ts_raw + 1),
    price     = c(open_prices, close_prices),
    volume    = c(rep(0, n_bars), volumes),
    stringsAsFactors = FALSE
  )
  vpin_input <- vpin_input[order(vpin_input$timestamp), ]

  # NA 제거 + 최소 행 수 검증
  ok <- complete.cases(vpin_input)
  if (sum(ok) < 200L) {
    saveRDS(empty_bucket, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("유효 행 부족: ", sum(ok))))
  }
  vpin_input <- vpin_input[ok, ]

  # ── vpin() 호출 ──
  result <- tryCatch(
    vpin(data         = vpin_input,
         timebarsize  = tbs,
         buckets      = bpd,
         samplength   = sl,
         tradinghours = th,
         verbose      = FALSE),
    error = function(e) e
  )

  if (inherits(result, "error")) {
    saveRDS(empty_bucket, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("vpin() error: ", conditionMessage(result))))
  }

  is_ok <- tryCatch(result@success, error = function(e) NA)
  if (identical(is_ok, FALSE)) {
    err_msg <- tryCatch(result@errorMessage, error = function(e) "unknown")
    saveRDS(empty_bucket, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("@success=FALSE: ", err_msg)))
  }

  # ══════════════════════════════════════════════════════════════════════
  # @bucketdata 추출 → Date 열 추가
  # ══════════════════════════════════════════════════════════════════════
  out_bucket <- tryCatch({
    bd <- result@bucketdata
    if (is.null(bd) || !is.data.frame(bd) || nrow(bd) == 0)
      stop("@bucketdata is empty")

    names(bd) <- tolower(names(bd))

    # VPIN 값
    if ("vpin" %in% names(bd)) {
      vpin_vals <- as.numeric(bd$vpin)
    } else {
      vpin_vec <- result@vpin
      if (length(vpin_vec) != nrow(bd)) stop("@vpin length mismatch")
      vpin_vals <- as.numeric(vpin_vec)
    }

    # starttime / endtime (POSIXct)
    get_ts <- function(col) {
      if (col %in% names(bd)) {
        v <- bd[[col]]
        if (!inherits(v, "POSIXct"))
          v <- as.POSIXct(v, tz = "Asia/Seoul")
        else
          attr(v, "tzone") <- "Asia/Seoul"
        return(v)
      }
      rep(as.POSIXct(NA), nrow(bd))
    }
    st_vals <- get_ts("starttime")
    et_vals <- get_ts("endtime")

    # ★ 날짜 추출: endtime 기준
    date_vals <- as.Date(et_vals, tz = "Asia/Seoul")

    # 매수/매도 볼륨, AOI
    abv <- if ("agg.bvol" %in% names(bd)) as.numeric(bd[["agg.bvol"]]) else rep(NA_real_, nrow(bd))
    asv <- if ("agg.svol" %in% names(bd)) as.numeric(bd[["agg.svol"]]) else rep(NA_real_, nrow(bd))
    aoi <- if ("aoi" %in% names(bd)) as.numeric(bd[["aoi"]]) else abs(abv - asv)

    data.frame(
      Symbol     = symbol,
      BucketNo   = seq_len(nrow(bd)),
      Date       = date_vals,
      Starttime  = st_vals,
      Endtime    = et_vals,
      AggBuyVol  = abv,
      AggSellVol = asv,
      AOI        = aoi,
      VPIN       = vpin_vals,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    # fallback: @vpin 벡터만 사용
    vpin_vec <- tryCatch(result@vpin, error = function(e2) numeric(0))
    if (length(vpin_vec) == 0) return(NULL)
    data.frame(
      Symbol     = symbol,
      BucketNo   = seq_along(vpin_vec),
      Date       = rep(as.Date(NA), length(vpin_vec)),
      Starttime  = rep(as.POSIXct(NA), length(vpin_vec)),
      Endtime    = rep(as.POSIXct(NA), length(vpin_vec)),
      AggBuyVol  = rep(NA_real_, length(vpin_vec)),
      AggSellVol = rep(NA_real_, length(vpin_vec)),
      AOI        = rep(NA_real_, length(vpin_vec)),
      VPIN       = as.numeric(vpin_vec),
      stringsAsFactors = FALSE
    )
  })

  if (is.null(out_bucket) || nrow(out_bucket) == 0) {
    saveRDS(empty_bucket, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = "no VPIN results extracted"))
  }

  # ── 결과 저장 ──
  saveRDS(out_bucket, ckpt_path)
  return(list(symbol = symbol, n_rows = nrow(out_bucket), error = NULL))
}


# ─── 메인 루프: 파일 경로 전달 → 배치별 병렬 처리 ──────────────────────────
#
# 이전 버전과 달리 bars_dt를 split하지 않는다.
# 종목별 파일 경로만 담은 경량 args 리스트를 워커에 전달한다.
# 워커가 arrow::read_parquet()로 자신의 종목 파일을 직접 읽으므로
# 메인 프로세스 메모리에는 종목 데이터가 올라가지 않는다.
# ────────────────────────────────────────────────────────────────────────

if (length(remaining_syms) == 0) {
  cat("\n  모든 종목 처리 완료. 병합 단계로 이동합니다.\n")
} else {

  cat(sprintf("\n[4] 종목 인자 생성 (%d종목) — 파일 경로 전달 방식\n",
              length(remaining_syms)))

  # ★ 데이터를 메모리에 올리지 않고, 파일 경로만 전달
  symbol_args <- lapply(remaining_syms, function(sym) {
    fpath <- sym_file_map[[sym]]
    if (is.null(fpath) || !file.exists(fpath)) return(NULL)
    list(
      symbol          = sym,
      sym_file        = fpath,         # ★ 데이터 대신 파일 경로
      checkpoint_dir  = CHECKPOINT_DIR,
      timebarsize_sec = TIMEBARSIZE_SEC,
      buckets_per_day = BUCKETS_PER_DAY,
      samplength      = SAMPLENGTH,
      tradinghours    = TRADINGHOURS
    )
  })
  symbol_args <- Filter(Negate(is.null), symbol_args)

  n_remaining <- length(symbol_args)
  cat(sprintf("  준비 완료: %d종목 (메인 프로세스 메모리 사용: 파일 경로만)\n",
              n_remaining))

  if (n_remaining == 0) {
    cat("  처리할 종목이 없습니다.\n")
  } else {

    n_workers <- min(NUM_WORKERS, n_remaining)

    cat(sprintf("\n[4] vpin() 병렬 추정 — 워커 %d개  |  배치 %d종목\n",
                n_workers, BATCH_N))
    cat(sprintf("%s\n", strrep("-", 65)))

    cl <- makeCluster(n_workers, type = "PSOCK")
    on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

    all_errors  <- list()
    n_success   <- 0L
    n_fail      <- 0L
    start_time  <- Sys.time()
    batch_starts <- seq(1L, n_remaining, by = BATCH_N)

    for (bi in seq_along(batch_starts)) {
      b_start <- batch_starts[bi]
      b_end   <- min(b_start + BATCH_N - 1L, n_remaining)
      batch_args <- symbol_args[b_start:b_end]
      batch_size <- length(batch_args)

      batch_results <- tryCatch(
        parLapply(cl, batch_args, worker_fn),
        error = function(e) {
          cat(sprintf("\n  [!] 배치 %d 클러스터 에러: %s\n", bi, e$message))
          cat("      클러스터 재생성 + 순차 재시도...\n")
          tryCatch(stopCluster(cl), error = function(e2) NULL)
          cl <<- makeCluster(n_workers, type = "PSOCK")
          lapply(batch_args, function(a) {
            tryCatch(worker_fn(a),
                     error = function(e3) {
                       list(symbol = a$symbol, n_rows = 0L,
                            error = paste0("순차 재시도 실패: ", e3$message))
                     })
          })
        }
      )

      batch_ok <- 0L
      if (!is.null(batch_results)) {
        for (res in batch_results) {
          if (!is.null(res$error)) {
            all_errors[[res$symbol]] <- res$error
            n_fail <- n_fail + 1L
          } else if (res$n_rows > 0L) {
            batch_ok  <- batch_ok + 1L
            n_success <- n_success + 1L
          } else {
            n_fail <- n_fail + 1L
          }
        }
      }

      done_so_far <- length(completed_syms) + b_end
      elapsed     <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      speed       <- b_end / max(elapsed, 0.01)
      eta_min     <- (n_remaining - b_end) / max(speed, 0.01)

      cat(sprintf("  [%s] 배치 %d/%d (%d종목) | 누적 %d/%d | 성공 %d 실패 %d | %.1f분%s\n",
                  format(Sys.time(), "%H:%M:%S"),
                  bi, length(batch_starts), batch_size,
                  done_so_far, n_total,
                  n_success, n_fail, elapsed,
                  if (eta_min > 0) sprintf(" | ETA %.0f분", eta_min) else ""))
    }

    tryCatch(stopCluster(cl), error = function(e) NULL)

    cat(sprintf("\n  추정 완료: 성공 %d / 실패 %d / 합계 %d  (%.1f분)\n",
                n_success, n_fail, n_remaining,
                as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

    if (length(all_errors) > 0) {
      err_lines <- paste0(names(all_errors), " : ", unlist(all_errors))
      writeLines(c(
        sprintf("=== VPIN 에러 로그 %s ===", format(Sys.time(), "%Y-%m-%d %H:%M:%S")),
        sprintf("에러: %d / %d 종목", length(all_errors), n_remaining),
        "", err_lines
      ), LOG_PATH)
      cat(sprintf("  에러 로그: %s\n", LOG_PATH))
    }
  }
}


# =============================================================================
# [5] 체크포인트 병합 → 최종 결과 (버킷별 + 일별)
# =============================================================================

if (exists("symbol_args")) { rm(symbol_args); gc() }

cat(sprintf("\n[5] 체크포인트 병합 (배치 처리)\n"))

all_rds <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.rds$", full.names = TRUE)
cat(sprintf("  RDS 파일: %d개\n", length(all_rds)))

if (length(all_rds) == 0) stop("[Error] 체크포인트 파일이 없습니다.")

# ── 중간 청크 디렉토리 (버킷별) ──
CHUNK_DIR_B <- file.path(OUTPUT_DIR, "_merge_chunks_bucket")
dir.create(CHUNK_DIR_B, recursive = TRUE, showWarnings = FALSE)

# 기존 청크 정리
old <- list.files(CHUNK_DIR_B, full.names = TRUE)
if (length(old) > 0) file.remove(old)

MERGE_BATCH  <- 200L
merge_starts <- seq(1L, length(all_rds), by = MERGE_BATCH)
chunk_count_b <- 0L
total_rows    <- 0L
total_syms    <- 0L

cat(sprintf("  배치 크기: %d  |  배치 수: %d\n", MERGE_BATCH, length(merge_starts)))

for (bi in seq_along(merge_starts)) {
  b_start <- merge_starts[bi]
  b_end   <- min(b_start + MERGE_BATCH - 1L, length(all_rds))
  batch_files <- all_rds[b_start:b_end]

  bucket_list <- list()

  for (f in batch_files) {
    obj <- tryCatch(readRDS(f), error = function(e) NULL)
    if (is.null(obj)) next

    # ── 새 형식: data.frame 직접 ──
    if (is.data.frame(obj) && nrow(obj) > 0) {
      bucket_list[[length(bucket_list) + 1L]] <- obj
    }
    # ── 구 형식 호환: list(buckets=..., daily=...) (기존 체크포인트) ──
    else if (is.list(obj) && !is.data.frame(obj)) {
      bk <- obj$buckets
      if (!is.null(bk) && is.data.frame(bk) && nrow(bk) > 0)
        bucket_list[[length(bucket_list) + 1L]] <- bk
    }
  }

  # 버킷별 청크
  if (length(bucket_list) > 0L) {
    batch_dt <- data.table::rbindlist(bucket_list, use.names = TRUE, fill = TRUE)
    batch_dt <- batch_dt[!is.na(VPIN)]
    rm(bucket_list)

    if (nrow(batch_dt) > 0L) {
      chunk_count_b <- chunk_count_b + 1L
      chunk_path <- file.path(CHUNK_DIR_B, sprintf("chunk_%04d.parquet", chunk_count_b))
      arrow::write_parquet(batch_dt, chunk_path, compression = "zstd")
      total_rows <- total_rows + nrow(batch_dt)
      total_syms <- total_syms + data.table::uniqueN(batch_dt$Symbol)

      cat(sprintf("  배치 %d/%d: 버킷 %s행, %d종목\n",
                  bi, length(merge_starts),
                  format(nrow(batch_dt), big.mark = ","),
                  data.table::uniqueN(batch_dt$Symbol)))
    }
    rm(batch_dt); gc()
  }
}

if (chunk_count_b == 0L) {
  cat("[Warning] 유효한 VPIN 결과가 없습니다.\n")
  cat("  가능한 원인:\n")
  cat("  1) 종목별 데이터가 너무 적어 samplength개 버킷을 채울 수 없음\n")
  cat("  2) 입력 데이터의 Open/Close/Volume 값 문제\n")
  if (file.exists(LOG_PATH))
    cat(sprintf("  -> 에러 로그 확인: %s\n", LOG_PATH))
  quit(status = 0)
}

cat(sprintf("\n  버킷 청크 %d개 생성 (총 %s행, %d종목)\n",
            chunk_count_b, format(total_rows, big.mark = ","), total_syms))


# =============================================================================
# [6] Arrow 레이지 병합 → 최종 결과 저장 (버킷별 단일 파일)
# =============================================================================
#
#   Date 열은 worker_fn 내부에서 버킷의 endtime 기준으로
#   as.Date(et_vals, tz = "Asia/Seoul") 로 이미 계산되어 있음.
#   일별 VPIN(@dailyvpin) 계산 및 조인은 불필요.
#
# =============================================================================

cat(sprintf("\n[6] 최종 결과 저장\n"))

run_id <- format(Sys.time(), "%Y%m%d_%H%M")
stem   <- sprintf("vpin_%s_%s", COUNTRY, run_id)

parquet_path <- file.path(OUTPUT_DIR, paste0(stem, ".parquet"))
csv_path     <- file.path(OUTPUT_DIR, paste0(stem, "_sample1000.csv"))

ds_b <- arrow::open_dataset(CHUNK_DIR_B, format = "parquet")

result_tbl <- ds_b |>
  dplyr::arrange(Symbol, BucketNo) |>
  dplyr::compute()

arrow::write_parquet(result_tbl, parquet_path, compression = "zstd")

cat(sprintf("  버킷별 parquet : %s\n", parquet_path))

# 샘플 CSV
sample_df <- ds_b |>
  dplyr::arrange(Symbol, BucketNo) |>
  head(1000L) |>
  dplyr::collect()
write.csv(sample_df, csv_path, row.names = FALSE)
rm(sample_df)

# ── 통계 ──
stats <- ds_b |>
  dplyr::summarise(
    n_rows    = dplyr::n(),
    n_syms    = dplyr::n_distinct(Symbol),
    vpin_min  = min(VPIN, na.rm = TRUE),
    vpin_max  = max(VPIN, na.rm = TRUE),
    vpin_mean = mean(VPIN, na.rm = TRUE)
  ) |>
  dplyr::collect()

rm(result_tbl); gc()

cat(sprintf("\n%s\n", strrep("=", 65)))
cat(sprintf("[결과 요약]\n"))
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  나라코드      : %s\n", COUNTRY))
cat(sprintf("  총 버킷 수    : %s\n", format(stats$n_rows, big.mark = ",")))
cat(sprintf("  완료 종목     : %d / %d\n", stats$n_syms, n_total))
cat(sprintf("  VPIN 범위     : %.6f ~ %.6f\n", stats$vpin_min, stats$vpin_max))
cat(sprintf("  VPIN 평균     : %.6f\n", stats$vpin_mean))
cat(sprintf("\n  출력 컬럼: Symbol, BucketNo, Date, Starttime, Endtime,\n"))
cat(sprintf("             AggBuyVol, AggSellVol, AOI, VPIN\n"))
cat(sprintf("             (Date = 버킷 endtime 기준 날짜)\n"))
cat(sprintf("\n  버킷별 parquet : %s\n", parquet_path))
cat(sprintf("  CSV (샘플)     : %s\n", csv_path))
cat(sprintf("  체크포인트     : %s\n", CHECKPOINT_DIR))
if (file.exists(LOG_PATH))
  cat(sprintf("  에러 로그      : %s\n", LOG_PATH))
cat(sprintf("%s\n", strrep("=", 65)))

# 중간 청크 정리
unlink(CHUNK_DIR_B, recursive = TRUE)
cat(sprintf("  중간 청크 정리 완료\n"))

cat(sprintf("[완료] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))