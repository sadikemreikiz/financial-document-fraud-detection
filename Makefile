.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-data check data audit reports check-reports bench-ocr bench-synthesis trufor learned encoder-check clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-16s %s\n", $$1, $$2}'

install:  ## Create the environment and install git hooks
	uv sync
	uv run pre-commit install

lint:  ## Check code style and common bugs
	uv run ruff check .
	uv run ruff format --check .

format:  ## Auto-format and auto-fix
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Run mypy in strict mode
	uv run mypy

test:  ## Run tests that need no dataset, with coverage (as in CI)
	uv run pytest -m "not data" --cov

test-data:  ## Run checks on the real dataset (needs `make data`)
	uv run pytest -m data

check: lint typecheck test  ## Everything CI runs

data:  ## Download, verify and prepare the Find-it-again dataset
	uv run docfraud data download
	uv run docfraud data prepare

audit:  ## Regenerate the data audit report
	uv run docfraud data audit

# Reports that need the dataset and the OCR cache. Not included: the gold evaluations of
# reports/reading, which depend on the parser version that produced them (see their headers),
# the OCR benchmark (make bench-ocr), the timing of the synthetic forgery engine (make
# bench-synthesis) and the development rounds of system 6 (make trufor) and of systems 2 to 4
# (make learned), whose models run on the Apple GPU.
reports:  ## Regenerate every report that the current code produces
	uv run docfraud data audit
	uv run python experiments/parser_development.py
	uv run python experiments/rules_development.py
	uv run python experiments/forensics_preregistration.py
	uv run python experiments/invisible_traces.py
	uv run python experiments/jpeg_trace_comparison.py
	uv run python experiments/system1_development.py
	uv run python experiments/global_trace_probe.py
	uv run docfraud synthesis download-fonts
	uv run python experiments/synthetic_engine.py
	uv run docfraud synthesis review-report --round 1
	uv run docfraud observation report
	uv run docfraud marks report
	uv run docfraud reading score-review annotations/findit2/gold-review.json
	uv run docfraud reading score-review annotations/findit2/gold-review-b.json \
		--manifest annotations/findit2/gold-fields-b/manifest.json --authentic 4 --forged 1 \
		--record annotations/findit2/gold-review-b.json \
		--resolutions annotations/findit2/gold-review-b-resolutions.json \
		--output reports/reading/gold-verification-b.md
	uv run docfraud baseline report
	uv run docfraud rules report

check-reports: reports  ## Fail if a committed report differs from its regeneration
	git diff --exit-code -- reports

bench-ocr:  ## Compare OCR engines on training receipts (downloads docTR and PyTorch)
	uv run --with "python-doctr[torch]>=1.1.0" --with "torch>=2.14.1" --with "torchvision>=0.29.1" \
		python experiments/ocr_benchmark.py

bench-synthesis:  ## Time the synthetic forgery engine (reports/synthesis/engine-timing.md)
	uv run docfraud synthesis download-fonts
	uv run python experiments/synthetic_engine.py --timing

trufor:  ## System 6 development round on the design fold (downloads TruFor and its libraries)
	uv run docfraud forensics download-trufor
	uv run --with torch==2.14.1 --with torchvision==0.29.1 --with timm==1.0.30 --with yacs==0.1.8 \
		python -I experiments/system6_development.py

# Keeps a Mac from sleeping while the round runs; the lid must stay open and the power connected.
KEEP_AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -is)

learned:  ## Development round of systems 2 to 4 on the design fold, about 11 hours (resumes)
	uv run docfraud forensics download-resnet
	uv run docfraud synthesis download-fonts
	$(KEEP_AWAKE) uv run docfraud forensics repeat-check
	$(KEEP_AWAKE) uv run docfraud forensics train-round
	$(KEEP_AWAKE) uv run docfraud forensics measure-round

encoder-check:  ## One-time check that the learned models' encoder is torchvision's ResNet-18
	uv run docfraud forensics download-resnet
	uv run --with torchvision==0.29.1 python -I experiments/encoder_equivalence.py

clean:  ## Remove caches and build outputs (never touches data/ or artifacts/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
