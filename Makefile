datafolders:
	mkdir -p data/raw/macrodata
	kdir -p data/raw/stockdata
	mkdir -p data/processed/stockdata

parsestockdata:
	python scripts/parsedata.py