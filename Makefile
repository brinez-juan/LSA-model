.PHONY: help preprocess preprocess-sample label split test lint fmt

# Put src/ on the Python path so `python -m lyremo.*` and pytest imports resolve.
export PYTHONPATH := src

help:
	@echo ""
	@echo "LyRemo — available commands:"
	@echo ""
	@echo "  Data pipeline:"
	@echo "    make preprocess          Run full preprocessing on raw dataset"
	@echo "    make preprocess-sample   Run preprocessing on 10k sample (for testing)"
	@echo "    make label               Pseudo-label verses with GoEmotions teacher"
	@echo "    make split               Create train/val/test splits"
	@echo ""
	@echo "  Development:"
	@echo "    make test                Run test suite"
	@echo "    make lint                Run ruff linter"
	@echo "    make fmt                 Format code with black"
	@echo ""

preprocess:
	python -m lyremo.data.preprocess \
		--input data/raw/genius_raw.csv \
		--output data/processed/genius_verses.parquet

preprocess-sample:
	python -m lyremo.data.preprocess \
		--input data/raw/genius_raw.csv \
		--output data/processed/genius_verses_sample.parquet \
		--sample 10000

label:
	python -m lyremo.data.label \
		--input data/processed/genius_verses.parquet \
		--output data/processed/genius_labeled.parquet

split:
	python -m lyremo.data.split \
		--input data/processed/genius_labeled.parquet

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

fmt:
	black src/ tests/
