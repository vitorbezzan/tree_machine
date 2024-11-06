.DEFAULT_GOAL:= default

.PHONY: install
install:
	python -m pip install pip-tools build twine
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	python -m pip install -r requirements.txt

.PHONY: format
format:
	isort src/
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

.PHONY: safety
safety:
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	safety check -r requirements.txt

.PHONY: docs
docs: install
	mkdocs build
	touch ./site/.nojekyll

.PHONY: default
default: build
	twine check dist/*
