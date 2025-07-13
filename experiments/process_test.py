import os
from system.process import process_dataset 

tickers = ["AMD","INTC", "NVDA", "QCOM", "TSM"]

min_date = "2000-01-01"
max_date = "2018-01-01"
output_dir = "../data/processed/stocksdata"


for ticker in tickers:
    print(f"Processing ticker: {ticker}")
    
    processed_df = process_dataset(ticker, min_date, max_date)
    
    output_path = os.path.join(output_dir, f"{ticker}_processed.csv")
    print(output_path)
    
    processed_df.to_csv(output_path, index=False)
    
    print(f"Saved processed data for {ticker} to {output_path}")


