.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test test-data check data audit reports check-reports bench-ocr bench-synthesis trufor learned clean-measurement whole-training figures encoder-check fusion-preregistration fusion-diagnostics llm-preregistration clean

help:  ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

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
# bench-synthesis), the development rounds of system 6 (make trufor) and of systems 2 to 4
# (make learned), whose models run on the Apple GPU, and the data behind ADR-0029
# (make fusion-preregistration), which also needs the stored LLM answers and Phase 4 rounds.
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
	uv run python experiments/copy_move_development.py
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
	uv run docfraud copy-move report
	uv run docfraud fusion training-report
	uv run docfraud fusion input-shift-report
	uv run docfraud fusion validation-report
	uv run docfraud detector validation-report

check-reports: reports  ## Fail if a committed report differs from its regeneration
	git diff --exit-code -- reports

fusion-preregistration:  ## Data behind ADR-0029 (needs artifacts/llm, artifacts/learned/rounds, artifacts/fusion)
	uv run python experiments/fusion_preregistration.py
	uv run python experiments/copy_move_feasibility.py
	uv run python experiments/input_shift_method.py

llm-preregistration:  ## Data behind ADR-0031 (needs artifacts/llm and artifacts/fusion)
	uv run python experiments/llm_6b_preregistration.py

fusion-diagnostics:  ## Diagnostics of Phase 5's records (needs artifacts/llm and artifacts/fusion)
	uv run python experiments/input_shift_rules.py

bench-ocr:  ## Compare OCR engines on training receipts (downloads docTR and PyTorch)
	uv run --with "python-doctr[torch]>=1.1.0" --with "torch>=2.14.1" --with "torchvision>=0.29.1" \
		python experiments/ocr_benchmark.py

bench-synthesis:  ## Time the synthetic forgery engine (reports/synthesis/engine-timing.md)
	uv run docfraud synthesis download-fonts
	uv run python experiments/synthetic_engine.py --timing

# TruFor's own libraries, which are not project dependencies (ADR-0019).
WITH_TRUFOR := uv run --with torch==2.14.1 --with torchvision==0.29.1 --with timm==1.0.30 \
	--with yacs==0.1.8

trufor:  ## System 6 development round on the design fold (downloads TruFor and its libraries)
	uv run docfraud forensics download-trufor
	$(WITH_TRUFOR) python -I experiments/system6_development.py

# Keeps a Mac from sleeping while the round runs; the lid must stay open and the power connected.
KEEP_AWAKE := $(shell command -v caffeinate >/dev/null 2>&1 && echo caffeinate -is)

learned:  ## Development round of systems 2 to 4 on the design fold, about 11 hours (resumes)
	uv run docfraud forensics download-resnet
	uv run docfraud synthesis download-fonts
	$(KEEP_AWAKE) uv run docfraud forensics repeat-check
	$(KEEP_AWAKE) uv run docfraud forensics train-round
	$(KEEP_AWAKE) uv run docfraud forensics measure-round
	uv run python -I experiments/learned_development.py

# The single measurement of every system on the clean folds (ADR-0020, docs/clean-measurement.md):
# the 36 runs of systems 2 to 4 first, then every system's predictions, then the report. Every
# step resumes or skips what it has done; rerun the target from the same commit after a stop.
whole-training:  ## System 2 on the whole training split, seeds 0 to 2, about 3 hours (resumes)
	uv run docfraud forensics download-resnet
	uv run docfraud synthesis download-fonts
	$(KEEP_AWAKE) uv run docfraud forensics train-whole

clean-measurement:  ## Clean measurement of every Phase 4 system, about 42 hours (resumes)
	uv run docfraud forensics download-resnet
	uv run docfraud synthesis download-fonts
	uv run docfraud forensics download-trufor
	$(KEEP_AWAKE) uv run docfraud forensics train-clean
	$(KEEP_AWAKE) uv run docfraud forensics measure-clean
	$(KEEP_AWAKE) uv run docfraud forensics measure-lattice
	uv run docfraud forensics measure-probe
	$(KEEP_AWAKE) $(WITH_TRUFOR) docfraud forensics measure-trufor
	uv run python -I experiments/clean_measurement.py

figures:  ## Figures of the clean measurement and the validation check (needs their runs)
	uv run --with matplotlib==3.10.7 python -I experiments/figures.py

encoder-check:  ## One-time check that the learned models' encoder is torchvision's ResNet-18
	uv run docfraud forensics download-resnet
	uv run --with torchvision==0.29.1 python -I experiments/encoder_equivalence.py

clean:  ## Remove caches and build outputs (never touches data/ or artifacts/)
	rm -rf .mypy_cache .ruff_cache .pytest_cache .coverage htmlcov dist build
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
