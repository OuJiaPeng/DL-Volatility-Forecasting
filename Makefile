.PHONY: install panel baselines eval all test lint

install:
	pip install -e ".[dev]"

panel:
	python -m volforecast.cli panel

baselines eval:
	python -m volforecast.cli eval

all:
	python -m volforecast.cli all

# real SPX data (Databento cache must be pulled first: scripts/pull_spx_history.py)
spx:
	python -m volforecast.cli all --config configs/spx.yaml --model

test:
	pytest -q

lint:
	ruff check volforecast tests

status:
	python scripts/status.py
