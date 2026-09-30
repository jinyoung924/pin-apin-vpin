# =============================================================================
# [VPIN 파이프라인 - Step 2] PINstimation::vpin() 버킷별 VPIN 계산 (병렬)
# =============================================================================
#
# 모델: Easley et al.(2012) — Volume-Synchronized PIN (VPIN)
#
# ─── vpin() 시그니처 (PINstimation 공식 문서) ────────────────────────────────
#   vpin(data, timebarsize = 60, buckets = 50, samplength = 50,
#        tradinghours = 24, verbose = TRUE)
#
#   data         : data.frame — 열 순서 {timestamp, price, volume} (이름 무관)
#   timebarsize  : 타임바 크기(초). 1분봉 = 60
#   buckets      : 하루당 버킷 수 (ADV를 이 수로 나눠 VBS 산출)
#   samplength   : VPIN 롤링 윈도우 버킷 수
#   tradinghours : 하루 거래 시간(시간)
#   verbose      : 진행 출력
#
# ─── 반환 S4 객체: estimate.vpin ──────────────────────────────────────────────
#   @success      : logical
#   @errorMessage : character
#   @parameters   : numeric vector
#   @bucketdata   : data.frame (bucket, agg.bVol, agg.sVol, aoi,
#                               starttime, endtime, duration, vpin)
#   @vpin         : numeric vector
#   @dailyvpin    : data.frame (day, dvpin, dwvpin)
#
# ─── ★ dp=0 문제 해결 ──────────────────────────────────────────────────────
#   기존: Python이 Price(=last만) 저장 → vpin()이 dp=last-first=0
#   수정: Python이 Open+Close 저장 → R에서 2개 pseudo-tick으로 확장
#         tick1: (timestamp,   Open,  0)       ← first price = Open
#         tick2: (timestamp+1, Close, Volume)  ← last price  = Close
#         → dp = Close - Open (올바른 가격변동)
#
# ─── 입출력 ──────────────────────────────────────────────────────────────────
#   입력: R_output/{COUNTRY}/vpin/all_1m_bars.parquet  (01_preprocess.py 출력)
#         스키마: Symbol(Utf8), Datetime(Datetime),
#                Open(Float64), Close(Float64), Volume(Float64)
#
#   출력: R_output/{COUNTRY}/vpin/
#         vpin_{COUNTRY}_{YYYYMMDD_HHMM}.parquet        ← 전체 버킷별 결과
#         vpin_{COUNTRY}_{YYYYMMDD_HHMM}_sample1000.csv ← 확인용
#         checkpoints/sym_{Symbol}.rds                   ← 종목별 체크포인트
#
# ─── 병렬 처리 전략 ─────────────────────────────────────────────────────────
#   1) 메인: bars_dt를 종목별 리스트로 분할 (data.table split — 빠름)
#   2) 메인: PSOCK 클러스터 생성, 배치 단위로 parLapply 호출
#   3) 워커: 종목 data.frame 수신 → pseudo-tick 확장 → vpin() 계산
#            → RDS 직접 저장 → 상태 문자열만 메인으로 반환 (IPC 최소화)
#   4) 메인: 배치마다 진행 상황 출력 + 클러스터 건강 확인
#   5) 레이스 컨디션 없음: 워커별 고유 파일(sym_{종목}.rds) 쓰기
#
# ─── 실행 ────────────────────────────────────────────────────────────────────
#   Rscript 02_r_vpin.R
#
# ─── 의존 패키지 ─────────────────────────────────────────────────────────────
#   install.packages(c("PINstimation", "arrow", "data.table", "dplyr", "parallel"))
# =============================================================================

suppressPackageStartupMessages({
  library(PINstimation)
  library(arrow)
  library(data.table)
  library(dplyr)       # arrow::open_dataset() 파이프라인에 필요
  library(parallel)
})


# =============================================================================
# ★ 사용자 설정 구역 — 여기만 수정하면 됩니다
# =============================================================================

BASE_DIR <- "E:/vpin_project_parquet/processing_data"
COUNTRY  <- "KOR"

# ── vpin() 파라미터 ──────────────────────────────────────────────────────────
# PINstimation::vpin()은 모든 인자에 정수(integer)를 요구한다. 반드시 L 접미사 사용.
TIMEBARSIZE_SEC <- 60L    # 타임바 크기(초). 1분봉 = 60
BUCKETS_PER_DAY <- 50L    # 하루당 버킷 수
SAMPLENGTH      <- 50L    # 롤링 윈도우 버킷 수
TRADINGHOURS    <- 7L     # 하루 거래 시간(시간, 정수만 허용). KRX 6.5h → 7로 올림
                          #   US: 7L (6.5h)  |  JP: 6L (5h)  |  HK: 6L (5.5h)

# ── 필터링 파라미터 ──────────────────────────────────────────────────────────
MIN_ROWS <- 500           # 종목당 최소 행 수 (미만 스킵)

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS    <- max(1L, parallel::detectCores(logical = TRUE) - 2L)
                          # 논리 코어 전체 - 2 (OS + 메인 프로세스 여유분)
                          # 16코어 서버 → 14워커, 8코어 → 6워커
                          # 수동 지정 시: NUM_WORKERS <- 14L 등으로 직접 설정 가능
BATCH_N        <- 100L    # 배치당 종목 수 (배치마다 클러스터 건강 확인)


# =============================================================================
# (이하 수정 불필요)
# =============================================================================

INPUT_PATH     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin",
                            "all_1m_bars.parquet")
OUTPUT_DIR     <- file.path(BASE_DIR, "R_output", COUNTRY, "vpin")
CHECKPOINT_DIR <- file.path(OUTPUT_DIR, "checkpoints")
LOG_PATH       <- file.path(OUTPUT_DIR, "vpin_errors.log")

dir.create(CHECKPOINT_DIR, recursive = TRUE, showWarnings = FALSE)

cat(sprintf("\n%s\n", strrep("=", 65)))
cat(sprintf("[VPIN 계산] 시작: %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  나라코드       : %s\n", COUNTRY))
cat(sprintf("  입력 파일      : %s\n", INPUT_PATH))
cat(sprintf("  출력 폴더      : %s\n", OUTPUT_DIR))
cat(sprintf("  timebarsize    : %d초\n", TIMEBARSIZE_SEC))
cat(sprintf("  buckets/day    : %d\n", BUCKETS_PER_DAY))
cat(sprintf("  samplength     : %d\n", SAMPLENGTH))
cat(sprintf("  tradinghours   : %dh\n", TRADINGHOURS))
cat(sprintf("  최소 행 수     : %d\n", MIN_ROWS))
cat(sprintf("  CPU 코어 감지  : 논리 %d개 / 물리 %d개\n",
            parallel::detectCores(logical = TRUE),
            parallel::detectCores(logical = FALSE)))
cat(sprintf("  병렬 워커      : %d개\n", NUM_WORKERS))
cat(sprintf("  배치 크기      : %d종목\n", BATCH_N))


# =============================================================================
# [1] 데이터 로드
# =============================================================================

if (!file.exists(INPUT_PATH))
  stop(sprintf("[Error] 파일 없음: %s\n  먼저 01_preprocess.py를 실행하세요.",
               INPUT_PATH))

cat(sprintf("\n[1] 데이터 로드 중...\n"))

bars_dt <- as.data.table(arrow::read_parquet(INPUT_PATH))

# ── 스키마 검증: Open/Close 열 존재 여부 확인 ──
if (!all(c("Open", "Close") %in% names(bars_dt))) {
  # 구버전 parquet (Price만 있는 경우) → 에러 메시지
  if ("Price" %in% names(bars_dt)) {
    stop(paste0(
      "[Error] 입력 파일에 'Open', 'Close' 열이 없고 'Price'만 있습니다.\n",
      "  dp=0 문제를 해결하려면 vpin_pre.py를 수정 버전으로 재실행하세요.\n",
      "  (FORCE_REPROCESS = True 설정 후 python vpin_pre.py)"
    ))
  } else {
    stop("[Error] 입력 파일에 'Open', 'Close' 열이 없습니다.")
  }
}

# Datetime → POSIXct 변환 (안전하게)
if (!inherits(bars_dt$Datetime, "POSIXct")) {
  bars_dt[, Datetime := as.POSIXct(Datetime, tz = "Asia/Seoul")]
} else {
  attr(bars_dt$Datetime, "tzone") <- "Asia/Seoul"
}

cat(sprintf("  전체 행       : %s\n", format(nrow(bars_dt), big.mark = ",")))
cat(sprintf("  전체 종목 수  : %d\n", uniqueN(bars_dt$Symbol)))
cat(sprintf("  시간 범위     : %s ~ %s\n",
            format(min(bars_dt$Datetime)), format(max(bars_dt$Datetime))))

# ── dp 검증 출력 ──
n_nonzero <- sum(bars_dt$Open != bars_dt$Close, na.rm = TRUE)
pct_nonzero <- n_nonzero / nrow(bars_dt) * 100
cat(sprintf("  dp≠0 봉 비율  : %s / %s (%.1f%%)\n",
            format(n_nonzero, big.mark = ","),
            format(nrow(bars_dt), big.mark = ","),
            pct_nonzero))


# =============================================================================
# [2] 데이터 정제
# =============================================================================

cat(sprintf("\n[2] 데이터 정제\n"))

n_before <- nrow(bars_dt)

# NA, 0 이하 제거
bars_dt <- bars_dt[!is.na(Datetime) & !is.na(Open) & !is.na(Close) & !is.na(Volume)]
bars_dt <- bars_dt[Open > 0 & Close > 0 & Volume > 0]

n_removed <- n_before - nrow(bars_dt)
if (n_removed > 0)
  cat(sprintf("  제거된 행: %s개\n", format(n_removed, big.mark = ",")))

# 종목별 행 수 필터
sym_counts <- bars_dt[, .N, by = Symbol]
valid_syms <- sym_counts[N >= MIN_ROWS, Symbol]
skip_syms  <- sym_counts[N < MIN_ROWS, Symbol]

if (length(skip_syms) > 0)
  cat(sprintf("  최소 행 수(%d) 미달 → %d종목 제외\n", MIN_ROWS, length(skip_syms)))

bars_dt <- bars_dt[Symbol %in% valid_syms]
setorder(bars_dt, Symbol, Datetime)

all_symbols <- sort(valid_syms)
n_total     <- length(all_symbols)

cat(sprintf("  유효 종목: %d개  |  유효 행: %s\n",
            n_total, format(nrow(bars_dt), big.mark = ",")))


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
# [4] 종목별 데이터 분할 → PSOCK 병렬 처리
# =============================================================================
#
# ★ 핵심 수정: 워커 내부에서 1분봉 (Open, Close, Volume) →
#   2개 pseudo-tick으로 확장한 뒤 vpin()에 전달
#
#   tick1: (Datetime,     Open,  0)       ← first price = Open
#   tick2: (Datetime + 1, Close, Volume)  ← last price  = Close
#
#   이렇게 하면 vpin()이 60초 타임바로 재집계할 때:
#     dp  = last(price) - first(price) = Close - Open  ← 올바른 가격변동
#     tbv = 0 + Volume = Volume                        ← 올바른 거래량
#
# 워커 설계:
#   - 워커는 (symbol, sym_df, params, checkpoint_dir) 를 받는다
#   - 워커 내부에서 POSIXct tzone을 명시적으로 재설정 (PSOCK 직렬화 유실 방지)
#   - ★ Open/Close → pseudo-tick 확장 후 vpin() 호출
#   - vpin() 계산 후 RDS를 직접 저장 → 상태 리스트만 반환 (IPC 최소화)
#   - 각 워커가 고유 파일(sym_{종목}.rds)에 쓰므로 레이스 컨디션 없음
#
# 배치 전략:
#   - BATCH_N 종목씩 parLapply 호출
#   - 배치마다 진행 상황 + ETA 출력
#   - 배치 단위 tryCatch: 워커 크래시 시 클러스터 재생성 + 순차 재시도
# =============================================================================

# ─── 워커 함수 (자기 완결적 — 클로저 의존 없음) ──────────────────────────────

worker_fn <- function(args) {
  # 패키지 로드 (각 워커 프로세스에서 독립적으로)
  suppressPackageStartupMessages(library(PINstimation))

  symbol         <- args$symbol
  sym_df         <- args$sym_df
  checkpoint_dir <- args$checkpoint_dir
  tbs            <- args$timebarsize_sec
  bpd            <- args$buckets_per_day
  sl             <- args$samplength
  th             <- args$tradinghours

  # 빈 결과 템플릿
  empty_df <- data.frame(
    Symbol = character(0), BucketNo = integer(0),
    Endtime = as.POSIXct(character(0)),
    AggBuyVol = numeric(0), AggSellVol = numeric(0),
    VPIN = numeric(0), stringsAsFactors = FALSE
  )
  ckpt_path <- file.path(checkpoint_dir, paste0("sym_", symbol, ".rds"))

  # ── POSIXct tzone 복원 (PSOCK 직렬화 시 유실됨) ──
  ts_raw <- sym_df$Datetime
  if (inherits(ts_raw, "POSIXct")) {
    attr(ts_raw, "tzone") <- "Asia/Seoul"
  } else if (is.numeric(ts_raw)) {
    ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01", tz = "Asia/Seoul")
  } else {
    ts_raw <- as.POSIXct(as.character(ts_raw), tz = "Asia/Seoul")
  }

  # ══════════════════════════════════════════════════════════════════════════
  # ★ 핵심: 1분봉 (Open, Close, Volume) → 2개 pseudo-tick으로 확장
  # ══════════════════════════════════════════════════════════════════════════
  #
  #   tick1: (timestamp = Datetime,     price = Open,  volume = 0)
  #   tick2: (timestamp = Datetime + 1, price = Close, volume = Volume)
  #
  #   → vpin()이 timebarsize=60 로 집계하면:
  #     dp  = last(Close) - first(Open) = Close - Open  ← 올바른 dp
  #     tbv = 0 + Volume                                ← 올바른 volume
  # ══════════════════════════════════════════════════════════════════════════

  n_bars <- length(ts_raw)

  open_prices  <- as.numeric(sym_df$Open)
  close_prices <- as.numeric(sym_df$Close)
  volumes      <- as.numeric(sym_df$Volume)

  # tick1: bar 시작 시점, Open 가격, volume = 0
  ts_tick1  <- ts_raw
  p_tick1   <- open_prices
  v_tick1   <- rep(0, n_bars)

  # tick2: bar 시작 + 1초, Close 가격, volume = 전체 거래량
  ts_tick2  <- ts_raw + 1  # 1초 뒤
  p_tick2   <- close_prices
  v_tick2   <- volumes

  # 두 틱을 인터리브하여 시간순 정렬
  vpin_input <- data.frame(
    timestamp = c(ts_tick1, ts_tick2),
    price     = c(p_tick1, p_tick2),
    volume    = c(v_tick1, v_tick2),
    stringsAsFactors = FALSE
  )
  vpin_input <- vpin_input[order(vpin_input$timestamp), ]

  # NA 제거
  ok <- complete.cases(vpin_input)
  if (sum(ok) < 200L) {  # 최소 100봉 * 2틱 = 200행
    saveRDS(empty_df, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("유효 행 부족: ", sum(ok), "/", length(ok))))
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

  # 에러 객체 반환 시
  if (inherits(result, "error")) {
    saveRDS(empty_df, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("vpin() error: ", conditionMessage(result))))
  }

  # @success 확인
  is_ok <- tryCatch(result@success, error = function(e) NA)
  if (identical(is_ok, FALSE)) {
    err_msg <- tryCatch(result@errorMessage, error = function(e) "unknown")
    saveRDS(empty_df, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = paste0("@success=FALSE: ", err_msg)))
  }

  # ── @bucketdata 추출 ──
  out_df <- tryCatch({
    bd <- result@bucketdata
    if (is.null(bd) || !is.data.frame(bd) || nrow(bd) == 0)
      stop("@bucketdata is empty")

    names(bd) <- tolower(names(bd))

    # VPIN 값
    if ("vpin" %in% names(bd)) {
      vpin_vals <- as.numeric(bd$vpin)
    } else {
      vpin_vec <- result@vpin
      if (length(vpin_vec) != nrow(bd))
        stop("@vpin length mismatch")
      vpin_vals <- as.numeric(vpin_vec)
    }

    # 타임스탬프 (endtime 우선)
    ts_col <- intersect(c("endtime", "starttime"), names(bd))[1]
    if (!is.na(ts_col)) {
      ts_vals <- bd[[ts_col]]
      if (!inherits(ts_vals, "POSIXct"))
        ts_vals <- as.POSIXct(ts_vals, tz = "Asia/Seoul")
      else
        attr(ts_vals, "tzone") <- "Asia/Seoul"
    } else {
      ts_vals <- rep(as.POSIXct(NA), nrow(bd))
    }

    # 매수/매도 볼륨
    abv <- if ("agg.bvol" %in% names(bd)) as.numeric(bd[["agg.bvol"]]) else rep(NA_real_, nrow(bd))
    asv <- if ("agg.svol" %in% names(bd)) as.numeric(bd[["agg.svol"]]) else rep(NA_real_, nrow(bd))

    data.frame(
      Symbol     = symbol,
      BucketNo   = seq_len(nrow(bd)),
      Endtime    = ts_vals,
      AggBuyVol  = abv,
      AggSellVol = asv,
      VPIN       = vpin_vals,
      stringsAsFactors = FALSE
    )
  }, error = function(e) {
    # @bucketdata 실패 → @vpin 벡터 직접 사용
    vpin_vec <- tryCatch(result@vpin, error = function(e2) numeric(0))
    if (length(vpin_vec) == 0) return(NULL)
    data.frame(
      Symbol     = symbol,
      BucketNo   = seq_along(vpin_vec),
      Endtime    = rep(as.POSIXct(NA), length(vpin_vec)),
      AggBuyVol  = rep(NA_real_, length(vpin_vec)),
      AggSellVol = rep(NA_real_, length(vpin_vec)),
      VPIN       = as.numeric(vpin_vec),
      stringsAsFactors = FALSE
    )
  })

  if (is.null(out_df) || nrow(out_df) == 0) {
    saveRDS(empty_df, ckpt_path)
    return(list(symbol = symbol, n_rows = 0L,
                error = "no VPIN results extracted"))
  }

  # ── 결과 저장 (워커가 직접 저장 → IPC에 data.frame 태우지 않음) ──
  saveRDS(out_df, ckpt_path)
  return(list(symbol = symbol, n_rows = nrow(out_df), error = NULL))
}


# ─── 메인 루프: 데이터 분할 → 배치별 병렬 처리 ──────────────────────────────

if (length(remaining_syms) == 0) {
  cat("\n  모든 종목 처리 완료. 병합 단계로 이동합니다.\n")
} else {

  cat(sprintf("\n[4] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  # ★ 수정: Open, Close, Volume 모두 전달 (기존: Datetime, Price, Volume)
  bars_remaining <- bars_dt[Symbol %in% remaining_syms]
  split_list     <- split(
    bars_remaining[, .(Datetime, Open, Close, Volume)],
    bars_remaining$Symbol
  )
  rm(bars_remaining); gc()

  # 워커 인자 리스트 구성
  symbol_args <- lapply(remaining_syms, function(sym) {
    rows <- split_list[[sym]]
    if (is.null(rows) || nrow(rows) == 0) return(NULL)
    rows <- rows[order(rows$Datetime), ]
    list(
      symbol          = sym,
      sym_df          = rows,
      checkpoint_dir  = CHECKPOINT_DIR,
      timebarsize_sec = TIMEBARSIZE_SEC,
      buckets_per_day = BUCKETS_PER_DAY,
      samplength      = SAMPLENGTH,
      tradinghours    = TRADINGHOURS
    )
  })
  symbol_args <- Filter(Negate(is.null), symbol_args)
  rm(split_list); gc()

  n_remaining <- length(symbol_args)
  cat(sprintf("  분할 완료: %d종목\n", n_remaining))

  if (n_remaining == 0) {
    cat("  처리할 종목이 없습니다.\n")
  } else {

    # ── PSOCK 클러스터 생성 ──
    n_workers <- min(NUM_WORKERS, n_remaining)

    cat(sprintf("\n[4] vpin() 병렬 추정 — 워커 %d개  |  배치 %d종목\n",
                n_workers, BATCH_N))
    cat(sprintf("%s\n", strrep("-", 65)))

    cl <- makeCluster(n_workers, type = "PSOCK")
    on.exit(tryCatch(stopCluster(cl), error = function(e) NULL), add = TRUE)

    # 에러 집계
    all_errors  <- list()
    n_success   <- 0L
    n_fail      <- 0L
    start_time  <- Sys.time()

    # 배치 분할
    batch_starts <- seq(1L, n_remaining, by = BATCH_N)

    for (bi in seq_along(batch_starts)) {
      b_start <- batch_starts[bi]
      b_end   <- min(b_start + BATCH_N - 1L, n_remaining)
      batch_args <- symbol_args[b_start:b_end]
      batch_size <- length(batch_args)

      # ── 배치 실행 (tryCatch로 클러스터 크래시 방어) ──
      batch_results <- tryCatch(
        parLapply(cl, batch_args, worker_fn),
        error = function(e) {
          cat(sprintf("\n  [!] 배치 %d 클러스터 에러: %s\n", bi, e$message))
          cat("      클러스터 재생성 시도...\n")

          # 클러스터 재생성
          tryCatch(stopCluster(cl), error = function(e2) NULL)
          cl <<- makeCluster(n_workers, type = "PSOCK")

          # 재시도: 이 배치를 순차로 처리 (안전 모드)
          cat("      이 배치를 순차 처리합니다.\n")
          lapply(batch_args, function(a) {
            tryCatch(worker_fn(a),
                     error = function(e3) {
                       list(symbol = a$symbol, n_rows = 0L,
                            error = paste0("순차 재시도 실패: ", e3$message))
                     })
          })
        }
      )

      # ── 배치 결과 집계 ──
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

      # ── 진행 상황 출력 ──
      done_so_far <- length(completed_syms) + b_end
      elapsed     <- as.numeric(difftime(Sys.time(), start_time, units = "mins"))
      speed       <- b_end / max(elapsed, 0.01)
      eta_min     <- (n_remaining - b_end) / max(speed, 0.01)

      cat(sprintf("  [%s] 배치 %d/%d (%d종목) | 누적 %d/%d | 성공 %d 실패 %d | %.1f분%s\n",
                  format(Sys.time(), "%H:%M:%S"),
                  bi, length(batch_starts), batch_size,
                  done_so_far, n_total,
                  n_success, n_fail,
                  elapsed,
                  if (eta_min > 0) sprintf(" | ETA %.0f분", eta_min) else ""))
    }

    # 클러스터 종료
    tryCatch(stopCluster(cl), error = function(e) NULL)

    cat(sprintf("\n  추정 완료: 성공 %d / 실패 %d / 합계 %d  (%.1f분)\n",
                n_success, n_fail, n_remaining,
                as.numeric(difftime(Sys.time(), start_time, units = "mins"))))

    # 에러 로그 저장
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
# [5] 체크포인트 병합 → 최종 결과 (메모리 절약 배치 처리)
# =============================================================================
#
# 전략: bars_dt 해제 → RDS를 200개씩 배치로 읽어 중간 parquet 청크로 저장
#       → arrow::open_dataset()으로 레이지 병합 → 최종 parquet 1개 출력
# =============================================================================

# 메모리 확보
if (exists("bars_dt"))     { rm(bars_dt);     gc() }
if (exists("symbol_args")) { rm(symbol_args); gc() }

cat(sprintf("\n[5] 체크포인트 병합 (배치 처리)\n"))

all_rds <- list.files(CHECKPOINT_DIR, pattern = "^sym_.*\\.rds$", full.names = TRUE)
cat(sprintf("  RDS 파일: %d개\n", length(all_rds)))

if (length(all_rds) == 0) {
  stop("[Error] 체크포인트 파일이 없습니다.")
}

# ── 중간 parquet 청크 저장 디렉토리 ──
CHUNK_DIR <- file.path(OUTPUT_DIR, "_merge_chunks")
dir.create(CHUNK_DIR, recursive = TRUE, showWarnings = FALSE)

# 기존 청크 정리 (재실행 대비)
old_chunks <- list.files(CHUNK_DIR, full.names = TRUE)
if (length(old_chunks) > 0) file.remove(old_chunks)

MERGE_BATCH  <- 200L
merge_starts <- seq(1L, length(all_rds), by = MERGE_BATCH)
chunk_count  <- 0L
total_rows   <- 0L
total_syms   <- 0L

cat(sprintf("  배치 크기: %d  |  배치 수: %d\n", MERGE_BATCH, length(merge_starts)))

for (bi in seq_along(merge_starts)) {
  b_start <- merge_starts[bi]
  b_end   <- min(b_start + MERGE_BATCH - 1L, length(all_rds))
  batch_files <- all_rds[b_start:b_end]

  batch_list <- list()
  for (f in batch_files) {
    df_i <- tryCatch(readRDS(f), error = function(e) NULL)
    if (!is.null(df_i) && is.data.frame(df_i) && nrow(df_i) > 0) {
      batch_list[[length(batch_list) + 1L]] <- df_i
    }
  }

  if (length(batch_list) == 0L) next

  batch_dt <- data.table::rbindlist(batch_list, use.names = TRUE, fill = TRUE)
  batch_dt <- batch_dt[!is.na(VPIN)]
  rm(batch_list); gc()

  if (nrow(batch_dt) == 0L) { rm(batch_dt); next }

  chunk_count <- chunk_count + 1L
  chunk_path  <- file.path(CHUNK_DIR, sprintf("chunk_%04d.parquet", chunk_count))
  arrow::write_parquet(batch_dt, chunk_path, compression = "zstd")

  total_rows <- total_rows + nrow(batch_dt)
  total_syms <- total_syms + data.table::uniqueN(batch_dt$Symbol)

  cat(sprintf("  배치 %d/%d: %s행, %d종목 → %s\n",
              bi, length(merge_starts),
              format(nrow(batch_dt), big.mark = ","),
              data.table::uniqueN(batch_dt$Symbol),
              basename(chunk_path)))

  rm(batch_dt); gc()
}

if (chunk_count == 0L) {
  cat("[Warning] 유효한 VPIN 결과가 없습니다.\n")
  cat("  가능한 원인:\n")
  cat("  1) 종목별 데이터가 너무 적어 samplength개 버킷을 채울 수 없음\n")
  cat("  2) 입력 데이터의 timestamp/price/volume 형식 문제\n")
  if (file.exists(LOG_PATH))
    cat(sprintf("  -> 에러 로그 확인: %s\n", LOG_PATH))
  quit(status = 0)
}

cat(sprintf("\n  청크 %d개 생성 완료 (총 %s행, %d종목)\n",
            chunk_count, format(total_rows, big.mark = ","), total_syms))


# =============================================================================
# [6] Arrow 레이지 병합 → 최종 결과 저장
# =============================================================================

cat(sprintf("\n[6] 최종 결과 저장\n"))

run_id       <- format(Sys.time(), "%Y%m%d_%H%M")
stem         <- sprintf("vpin_%s_%s", COUNTRY, run_id)
parquet_path <- file.path(OUTPUT_DIR, paste0(stem, ".parquet"))
csv_path     <- file.path(OUTPUT_DIR, paste0(stem, "_sample1000.csv"))

ds <- arrow::open_dataset(CHUNK_DIR, format = "parquet")

result_tbl <- ds |>
  dplyr::arrange(Symbol, BucketNo) |>
  dplyr::compute()

arrow::write_parquet(result_tbl, parquet_path, compression = "zstd")
rm(result_tbl); gc()

cat(sprintf("  parquet 저장 완료: %s\n", parquet_path))

# 샘플 CSV
sample_df <- ds |>
  dplyr::arrange(Symbol, BucketNo) |>
  head(1000L) |>
  dplyr::collect()
write.csv(sample_df, csv_path, row.names = FALSE)
rm(sample_df)

# 통계
stats <- ds |>
  dplyr::summarise(
    n_rows    = dplyr::n(),
    n_syms    = dplyr::n_distinct(Symbol),
    vpin_min  = min(VPIN, na.rm = TRUE),
    vpin_max  = max(VPIN, na.rm = TRUE),
    vpin_mean = mean(VPIN, na.rm = TRUE)
  ) |>
  dplyr::collect()

cat(sprintf("\n%s\n", strrep("=", 65)))
cat(sprintf("[결과 요약]\n"))
cat(sprintf("%s\n", strrep("=", 65)))
cat(sprintf("  나라코드      : %s\n", COUNTRY))
cat(sprintf("  총 버킷 수    : %s\n", format(stats$n_rows, big.mark = ",")))
cat(sprintf("  완료 종목     : %d / %d\n", stats$n_syms, n_total))
cat(sprintf("  VPIN 범위     : %.6f ~ %.6f\n", stats$vpin_min, stats$vpin_max))
cat(sprintf("  VPIN 평균     : %.6f\n", stats$vpin_mean))
cat(sprintf("\n  parquet       : %s\n", parquet_path))
cat(sprintf("  CSV (샘플)    : %s\n", csv_path))
cat(sprintf("  체크포인트    : %s\n", CHECKPOINT_DIR))
if (file.exists(LOG_PATH))
  cat(sprintf("  에러 로그     : %s\n", LOG_PATH))
cat(sprintf("%s\n", strrep("=", 65)))

# 중간 청크 정리
unlink(CHUNK_DIR, recursive = TRUE)
cat(sprintf("  중간 청크 정리 완료\n"))

cat(sprintf("[완료] %s\n", format(Sys.time(), "%Y-%m-%d %H:%M:%S")))