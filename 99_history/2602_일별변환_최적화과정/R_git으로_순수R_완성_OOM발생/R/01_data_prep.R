# =============================================================================
# 01_data_prep.R — Arrow Lazy 데이터 전처리
# =============================================================================
#
# OOM 방지: arrow::open_dataset()로 lazy 쿼리 → Arrow C++ 엔진이 스트리밍 처리
# → collect() 시점에 최종 집계 결과만 R 메모리에 로드
#
# 제공 함수:
#   1) stream_daily_bs()       : PIN/AdjPIN용 — Arrow lazy로 일별 B/S 집계
#   2) get_vpin_dataset()      : VPIN용 — Arrow Dataset 객체 반환
#   3) load_vpin_sym()         : VPIN용 — 종목 1개 predicate pushdown 로드
#   4) get_vpin_symbols()      : VPIN용 — 전체 종목 목록 조회
#   5) make_symbol_queue()     : 종목 큐 생성 (알파벳순 정렬)
#
# =============================================================================


#' PIN/AdjPIN용 일별 매수/매도 건수 — Arrow Lazy 집계
#'
#' arrow::open_dataset()로 23개 parquet 파일을 lazy 스캔.
#' Arrow C++ 엔진이 파일별로 스트리밍 처리하므로 R 메모리에는
#' 최종 집계 결과(작은 df)만 로드된다.
#'
#' @param country 나라 코드
#' @return data.frame: Symbol, date, buys, sells
stream_daily_bs <- function(country, log_file = NULL) {
  input_dir <- get_input_dir(country)

  if (!dir.exists(input_dir))
    stop(sprintf("입력 디렉토리 없음: %s", input_dir))

  parquet_files <- list.files(input_dir, pattern = "\\.parquet$",
                              full.names = TRUE, recursive = TRUE)

  if (length(parquet_files) == 0)
    stop(sprintf("parquet 파일 없음: %s", input_dir))

  log_msg(sprintf("[Arrow Lazy 집계] %s: %d개 parquet 파일 → lazy 스캔",
                  country, length(parquet_files)), log_file)

  # Arrow Dataset — 모든 파일을 lazy로 연결 (메모리 0)
  ds <- arrow::open_dataset(input_dir, format = "parquet")

  log_msg("[Arrow Lazy 집계] lazy 쿼리 실행 중 (Arrow C++ 엔진)...", log_file)

  # lazy 쿼리: Arrow 엔진이 파일별 스트리밍 → 집계 결과만 R로 전달
  result <- ds |>
    dplyr::select(Date, Symbol, LR) |>
    dplyr::filter(LR != 0L) |>
    dplyr::group_by(Symbol, Date) |>
    dplyr::summarise(
      buys  = sum(LR == 1L,  na.rm = TRUE),
      sells = sum(LR == -1L, na.rm = TRUE),
      .groups = "drop"
    ) |>
    dplyr::collect()

  # 타입 정리
  result$Date   <- as.Date(result$Date)
  result$Symbol <- as.character(result$Symbol)
  result <- dplyr::rename(result, date = Date)
  result <- dplyr::arrange(result, Symbol, date)

  log_msg(sprintf("[Arrow Lazy 집계 완료] %s행, 종목 %d개, 날짜 %s ~ %s",
                  format(nrow(result), big.mark = ","),
                  length(unique(result$Symbol)),
                  min(result$date), max(result$date)), log_file)
  result
}


#' VPIN용 Arrow Dataset 객체 반환
#'
#' open_dataset()만 호출하고 collect()하지 않음.
#' 이후 load_vpin_sym()에서 종목별 predicate pushdown으로 로드.
#'
#' @param country 나라 코드
#' @return arrow::Dataset 객체
get_vpin_dataset <- function(country) {
  input_dir <- get_input_dir(country)
  arrow::open_dataset(input_dir, format = "parquet")
}


#' VPIN용 전체 종목 목록 조회 — Arrow lazy
#'
#' @param country 나라 코드
#' @return character vector: 정렬된 종목 코드
get_vpin_symbols <- function(country, log_file = NULL) {
  ds <- get_vpin_dataset(country)

  log_msg("[VPIN] Arrow lazy로 종목 목록 조회 중...", log_file)

  symbols <- ds |>
    dplyr::filter(LR != 0L, Volume > 0) |>
    dplyr::distinct(Symbol) |>
    dplyr::collect()

  result <- sort(as.character(symbols$Symbol))
  log_msg(sprintf("[VPIN] 종목 %d개 발견", length(result)), log_file)
  result
}


#' VPIN 종목 1개 틱 데이터 로드 — predicate pushdown
#'
#' Arrow 엔진이 parquet 파일의 row group 메타데이터를 활용해
#' 해당 종목의 행만 읽음. 전체 파일을 메모리에 올리지 않음.
#'
#' @param sym 종목 코드
#' @param ds arrow::Dataset 객체 (get_vpin_dataset()에서 생성)
#' @return data.frame (timestamp 기준 정렬) 또는 NULL
load_vpin_sym <- function(sym, ds) {
  df <- ds |>
    dplyr::filter(Symbol == sym, LR != 0L, Volume > 0) |>
    dplyr::select(Symbol, Date, Time, Price, Volume, LR) |>
    dplyr::collect()

  if (nrow(df) == 0) return(NULL)

  # 타입 정리
  df$Symbol <- as.character(df$Symbol)
  df$LR     <- as.integer(df$LR)
  df$Volume <- as.numeric(df$Volume)
  df$Price  <- as.numeric(df$Price)

  # timestamp 생성
  df$timestamp <- as.POSIXct(
    paste(as.character(df$Date), format(df$Time, "%H:%M:%S")),
    format = "%Y-%m-%d %H:%M:%S"
  )
  df <- df[!is.na(df$timestamp), ]
  df <- df[, c("Symbol", "timestamp", "Price", "Volume", "LR")]
  df[order(df$timestamp), ]
}


#' 종목 큐 생성 (알파벳순 정렬)
#'
#' @param df data.frame (Symbol 컬럼 포함)
#' @return character vector
make_symbol_queue <- function(df) {
  sort(unique(df$Symbol))
}
