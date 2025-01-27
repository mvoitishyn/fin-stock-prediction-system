library(readr)
library(dplyr)
library(lubridate)
library(ggplot2)
library(TTR)

raw_aapl <- readr::read_csv("data/processed/stockdata/AAPL.csv")

# open price only starts from 2000

min_date <- "2000-01-03"
max_date <-  "2017-12-08"

# easy preprocess
aapl_df <- raw_aapl %>%
    dplyr::rename(Date = "Unnamed: 0") %>%
    dplyr::select(Date, Open, High, Low, Close, Volume) %>%
    dplyr::mutate(Date = lubridate::date(Date)) %>%
    dplyr::filter(Date >= min_date & Date <= max_date)


# easy plot
ggplot2::ggplot(aapl_df, aes(x = Date, y = Close)) +
    geom_line(color = "blue") +
    labs(
        title = "AAPL Close Price Over Time",
        x = "Date",
        y = "Close Price"
    ) +
    theme_minimal()


#  daily return
aapl_df <- aapl_df %>%
    dplyr::mutate(
        Daily_Return = (Close / lag(Close) - 1) * 100
    )
aapl_df <- aapl_df %>%
    dplyr::filter(!is.na(Daily_Return))

# box plot for daily returns
ggplot(aapl_df, aes(y = Daily_Return)) +
    geom_boxplot(fill = "blue", alpha = 0.7, outlier.color = "red", outlier.size = 2) +
    labs(
        title = "Distribution of AAPL Daily Returns",
        x = "",
        y = "Daily Return (%)"
    ) +
    theme_minimal()

# find lower and upper threshold
mean(aapl_df$Daily_Return)
max(aapl_df$Daily_Return)
min(aapl_df$Daily_Return)
summary(aapl_df$Daily_Return)

iqr <- IQR(aapl_df$Daily_Return, na.rm = TRUE)
iqr
q1 <- quantile(aapl_df$Daily_Return, 0.25, na.rm = TRUE)
q1
q3 <- quantile(aapl_df$Daily_Return, 0.75, na.rm = TRUE)
q3

lower_threshold <- q1 - 1.5 * iqr
lower_threshold
upper_threshold <- q3 + 1.5 * iqr
upper_threshold

# find extreme market movements
extreme_moments <- aapl_df %>%
    dplyr::filter(Daily_Return < lower_threshold | Daily_Return > upper_threshold)

extreme_summary <- extreme_moments %>%
    dplyr::mutate(
        Type = ifelse(Daily_Return > 0, "Positive", "Negative")
    ) %>%
    dplyr::arrange(Date)

# moving averages
# - for trend 
# - for noise reduction
# - in theory support and threshold

aapl_df <- aapl_df %>%
    dplyr::mutate(VolumeSMA10 = TTR::SMA(Volume, n = 10)) %>%
    dplyr::filter(!is.na(VolumeSMA10))