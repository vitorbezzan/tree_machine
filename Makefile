.DEFAULT_GOAL:= default

.PHONY: install
install:
	python -m pip install pip-tools build twine
	python -m pip install ".[dev, experimental]"

.PHONY: format
format:
	pre-commit run --all-files

.PHONY: mypy
mypy:
	mypy src/ --ignore-missing-imports

.PHONY: flint
flint: format mypy

.PHONY: build
build: install
	python -m build .

.PHONY: test
test:
	pytest -rP tests/

.PHONY: docs
docs:
	playwright install
	cp ./README.md ./docs/index.md
	mkdocs build
	touch ./site/.nojekyll

.PHONY: default
default: build
	twine check dist/*

.PHONY: vulnerabilities
vulnerabilities:
	bandit -r ./src

.PHONY: dependencies
dependencies:
	pip-audit
