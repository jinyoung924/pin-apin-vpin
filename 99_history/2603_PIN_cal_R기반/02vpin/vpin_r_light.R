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
# ─── 타임존 수정 (v2) ─────────────────────────────────────────────────────
#
#   기존: 모든 POSIXct 변환에 "Asia/Seoul" 하드코딩
#     → UK 등 타 나라 데이터의 경우 시간이 +9h 밀려
#       영업일이 아닌 날짜(주말)에 VPIN이 산출되는 버그 발생
#   수정: TZ_MAP을 통해 COUNTRY에 해당하는 현지 타임존(TIMEZONE)을 사용.
#     Python 전처리(vpin_pre_light.py)에서 parquet에 타임존을 부여하므로
#     arrow::read_parquet() 시 올바른 타임존의 POSIXct로 자동 복원된다.
#
# ─── TRADINGHOURS 수정 (v3) ───────────────────────────────────────────────
#
#   기존: TRADINGHOURS <- 7L 하드코딩 (KRX 6.5h→7 올림)
#     → UK(8.5h), JP(5h) 등 나라마다 거래시간이 다르므로
#       VBS = (일평균거래량) / buckets 계산에 사용되는 거래일수가 왜곡됨
#   수정: TRADING_HOURS_MAP을 통해 나라별 실제 거래시간을 올림(ceiling)하여
#     정수로 자동 적용. PINstimation::vpin()의 tradinghours는 정수만 허용.
#     올림 이유: 내림 시 비거래시간 과다 차감 → 음수 듀레이션 위험.
#
# ─── 입출력 ──────────────────────────────────────────────────────────────────
#   입력: R_output/{COUNTRY}/vpin/all_1m_bars.parquet
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

BASE_DIR <- "E:/vpin_project_parquet/processing_data_2019_2021"
COUNTRY  <- "CA"

# ── vpin() 파라미터 ──────────────────────────────────────────────────────────
TIMEBARSIZE_SEC <- 60L    # 타임바 크기(초). 1분봉 = 60
BUCKETS_PER_DAY <- 50L    # 하루당 버킷 수
SAMPLENGTH      <- 50L    # 롤링 윈도우 버킷 수

# ── 필터링 파라미터 ──────────────────────────────────────────────────────────
MIN_ROWS <- 500           # 종목당 최소 행 수 (미만 스킵)

# ── 병렬 설정 ────────────────────────────────────────────────────────────────
NUM_WORKERS <- max(1L, parallel::detectCores(logical = TRUE) - 2L)
BATCH_N     <- 100L


# =============================================================================
# (이하 수정 불필요)
# =============================================================================

# ── 나라별 현지 타임존 매핑 ──
TZ_MAP <- list(
  KOR = "Asia/Seoul",
  US  = "America/New_York",
  JP  = "Asia/Tokyo",
  CA  = "America/Toronto",
  FR  = "Europe/Paris",
  GR  = "Europe/Berlin",
  HK  = "Asia/Hong_Kong",
  IT  = "Europe/Rome",
  UK  = "Europe/London"
)

# ── 나라별 일일 거래시간 매핑 (정수, 올림) ──
# PINstimation::vpin()의 tradinghours는 정수만 허용.
# 실제 거래시간이 정수가 아닌 경우 올림(ceiling) 적용.
#   올림 이유: 내림 시 비거래시간을 과다 차감 → 음수 듀레이션 위험.
#              올림은 보수적 방향(듀레이션 약간 과대)이므로 안전.
# 점심시간이 있는 시장(JP, HK)은 실거래시간만 합산 후 올림.
#   KOR: 09:00-15:30 = 6.5h → 7
#   US : 09:30-16:00 = 6.5h → 7
#   JP : 09:00-11:30 + 12:30-15:00 = 5.0h → 5
#   CA : 09:30-16:00 = 6.5h → 7
#   FR : 09:00-17:30 = 8.5h → 9
#   GR : 09:00-17:30 = 8.5h → 9 (Xetra)
#   HK : 09:30-12:00 + 13:00-16:00 = 5.5h → 6 (점심 휴장)
#   IT : 09:00-17:30 = 8.5h → 9
#   UK : 08:00-16:30 = 8.5h → 9
TRADING_HOURS_MAP <- list(
  KOR = 7L,
  US  = 7L,
  JP  = 5L,
  CA  = 7L,
  FR  = 9L,
  GR  = 9L,
  HK  = 6L,
  IT  = 9L,
  UK  = 9L
)

TIMEZONE <- TZ_MAP[[COUNTRY]]
if (is.null(TIMEZONE))
  stop(sprintf("[Error] TZ_MAP에 '%s' 없음. 지원: %s",
               COUNTRY, paste(names(TZ_MAP), collapse = ", ")))

TRADINGHOURS <- TRADING_HOURS_MAP[[COUNTRY]]
if (is.null(TRADINGHOURS))
  stop(sprintf("[Error] TRADING_HOURS_MAP에 '%s' 없음.", COUNTRY))

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
cat(sprintf("  타임존         : %s\n", TIMEZONE))
cat(sprintf("  입력 파일      : %s\n", INPUT_PATH))
cat(sprintf("  출력 폴더      : %s\n", OUTPUT_DIR))
cat(sprintf("  timebarsize    : %d초\n", TIMEBARSIZE_SEC))
cat(sprintf("  buckets/day    : %d\n", BUCKETS_PER_DAY))
cat(sprintf("  samplength     : %d\n", SAMPLENGTH))
cat(sprintf("  tradinghours   : %dh\n", TRADINGHOURS))
cat(sprintf("  최소 행 수     : %d\n", MIN_ROWS))
cat(sprintf("  병렬 워커      : %d개\n", NUM_WORKERS))
cat(sprintf("  배치 크기      : %d종목\n", BATCH_N))


# =============================================================================
# [1] 데이터 로드 + 스키마 검증
# =============================================================================

if (!file.exists(INPUT_PATH))
  stop(sprintf("[Error] 파일 없음: %s\n  먼저 vpin_pre.py를 실행하세요.", INPUT_PATH))

cat(sprintf("\n[1] 데이터 로드 중...\n"))

bars_dt <- as.data.table(arrow::read_parquet(INPUT_PATH))

# ── 스키마 검증: Open/Close 열 필수 ──
if (!all(c("Open", "Close") %in% names(bars_dt))) {
  if ("Price" %in% names(bars_dt)) {
    stop(paste0(
      "[Error] 입력 파일에 'Open', 'Close' 열이 없고 'Price'만 있습니다.\n",
      "  dp=0 문제를 해결하려면 수정된 vpin_pre.py로 재실행하세요.\n",
      "  (FORCE_REPROCESS = True 설정 후 python vpin_pre.py)"
    ))
  }
  stop("[Error] 입력 파일에 필수 열 'Open', 'Close'가 없습니다.")
}

# ── Datetime → POSIXct 보정 ──
# Python에서 tz-aware parquet을 쓰므로 arrow가 올바른 POSIXct를 반환한다.
# 방어 코드: parquet에 tz가 누락된 레거시 파일 대응
if (!inherits(bars_dt$Datetime, "POSIXct")) {
  bars_dt[, Datetime := as.POSIXct(Datetime, tz = TIMEZONE)]
}

cat(sprintf("  전체 행       : %s\n", format(nrow(bars_dt), big.mark = ",")))
cat(sprintf("  전체 종목 수  : %d\n", uniqueN(bars_dt$Symbol)))
cat(sprintf("  시간 범위     : %s ~ %s\n",
            format(min(bars_dt$Datetime)), format(max(bars_dt$Datetime))))

# dp 검증
n_nonzero <- sum(bars_dt$Open != bars_dt$Close, na.rm = TRUE)
pct_nz <- n_nonzero / nrow(bars_dt) * 100
cat(sprintf("  dp≠0 봉 비율  : %s / %s (%.1f%%)\n",
            format(n_nonzero, big.mark = ","),
            format(nrow(bars_dt), big.mark = ","), pct_nz))


# =============================================================================
# [2] 데이터 정제
# =============================================================================

cat(sprintf("\n[2] 데이터 정제\n"))

n_before <- nrow(bars_dt)
bars_dt <- bars_dt[!is.na(Datetime) & !is.na(Open) & !is.na(Close) & !is.na(Volume)]
bars_dt <- bars_dt[Open > 0 & Close > 0 & Volume > 0]

n_removed <- n_before - nrow(bars_dt)
if (n_removed > 0)
  cat(sprintf("  제거된 행: %s개\n", format(n_removed, big.mark = ",")))

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
# ── 출력: 버킷별 결과 + Date 열 ──
#
#   @bucketdata 에서 endtime (POSIXct) 추출 → as.Date() 로 날짜 부여
# =============================================================================

worker_fn <- function(args) {
  suppressPackageStartupMessages(library(PINstimation))

  symbol         <- args$symbol
  sym_df         <- args$sym_df
  checkpoint_dir <- args$checkpoint_dir
  tbs            <- args$timebarsize_sec
  bpd            <- args$buckets_per_day
  sl             <- args$samplength
  th             <- args$tradinghours
  tz             <- args$timezone

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

  # ── POSIXct tzone 복원 (PSOCK 직렬화 시 유실됨) ──
  ts_raw <- sym_df$Datetime
  if (inherits(ts_raw, "POSIXct")) {
    attr(ts_raw, "tzone") <- tz
  } else if (is.numeric(ts_raw)) {
    ts_raw <- as.POSIXct(ts_raw, origin = "1970-01-01", tz = tz)
  } else {
    ts_raw <- as.POSIXct(as.character(ts_raw), tz = tz)
  }

  n_bars <- length(ts_raw)
  open_prices  <- as.numeric(sym_df$Open)
  close_prices <- as.numeric(sym_df$Close)
  volumes      <- as.numeric(sym_df$Volume)

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
          v <- as.POSIXct(v, tz = tz)
        else
          attr(v, "tzone") <- tz
        return(v)
      }
      rep(as.POSIXct(NA), nrow(bd))
    }
    st_vals <- get_ts("starttime")
    et_vals <- get_ts("endtime")

    # ★ 날짜 추출: endtime 기준
    date_vals <- as.Date(et_vals, tz = tz)

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


# ─── 메인 루프: 데이터 분할 → 배치별 병렬 처리 ──────────────────────────────

if (length(remaining_syms) == 0) {
  cat("\n  모든 종목 처리 완료. 병합 단계로 이동합니다.\n")
} else {

  cat(sprintf("\n[4] 종목 데이터 분할 (%d종목)...\n", length(remaining_syms)))

  bars_remaining <- bars_dt[Symbol %in% remaining_syms]
  split_list     <- split(
    bars_remaining[, .(Datetime, Open, Close, Volume)],
    bars_remaining$Symbol
  )
  rm(bars_remaining); gc()

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
      tradinghours    = TRADINGHOURS,
      timezone        = TIMEZONE
    )
  })
  symbol_args <- Filter(Negate(is.null), symbol_args)
  rm(split_list); gc()

  n_remaining <- length(symbol_args)
  cat(sprintf("  분할 완료: %d종목\n", n_remaining))

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

if (exists("bars_dt"))     { rm(bars_dt);     gc() }
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
#   as.Date(et_vals, tz = TIMEZONE) 로 이미 계산되어 있음.
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
cat(sprintf("  타임존        : %s\n", TIMEZONE))
cat(sprintf("  거래시간      : %dh\n", TRADINGHOURS))
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