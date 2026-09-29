# Reproduce everything:  make setup && make data && make all
# Quick debug pass on the 5% subsample:  make all MODE=dev
MODE ?= full
RUN = uv run causalml
STAGES = eda stats predict causal cate targeting robustness scaling

.PHONY: setup data $(STAGES) report all notebooks test lint requirements clean-dev

setup:            ## create the environment from uv.lock
	uv sync

data:             ## download from Kaggle, validate, split, write parquet
	$(RUN) data

$(STAGES):
	$(RUN) $@ --mode $(MODE)

report:           ## regenerate README result tables from results/metrics/*.json
	$(RUN) report --mode $(MODE)

all: $(STAGES) report

notebooks:        ## execute the notebooks in place (they read results/ and the dev sample)
	uv run jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:
	uv run pytest

lint:
	uv run ruff check src tests

requirements:     ## refresh requirements.txt from uv.lock (pip users)
	uv export --no-hashes --no-dev --no-emit-project --format requirements-txt > requirements.txt

clean-dev:
	rm -rf results/dev
