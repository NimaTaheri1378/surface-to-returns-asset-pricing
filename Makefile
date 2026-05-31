.PHONY: install test safety figures release-audit

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	python -m pytest -q

safety:
	python scripts/public_safety_scan.py --allow-any-root

figures:
	python scripts/build_visual_abstract.py
	python scripts/build_paper_figure_package.py
	python scripts/build_visual_evidence_pack.py

release-audit:
	python scripts/build_visual_release_audit.py
	python scripts/public_safety_scan.py --allow-any-root
