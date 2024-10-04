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
docs:
	cp -r docs_build/ docs_temp/
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-apidoc -o ./docs_temp ./src
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-build -b html docs_temp/ docs/
	rm -rf docs_temp/

.PHONY: default
default: build
	twine check dist/*
