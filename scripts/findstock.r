library(readr)

all_files <- list.files(path = "data/raw/stockdata", recursive = TRUE, full.names = TRUE)

matching_files <- grep("VOO", all_files, value = TRUE)
print(matching_files)

df <- readr::read_csv("data/raw/stockdata/NYSEAMERICAN_IVOO_30Y.csv")
