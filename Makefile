.DEFAULT_GOAL:= default

.PHONY: install
install:
	python -m pip install pip-tools
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	python -m pip install -r requirements.txt

.PHONY: format
format:
	isort src/
	pre-commit run --all-files

.PHONY: pylint
pylint:
	python -m pylint  --init-hook='import sys; sys.setrecursionlimit(5000)' src/ --disable=useless-suppression
	python -m pylint tests/ --disable=missing-docstring,missing-function-docstring,invalid-name,redefined-outer-name,redefined-outer-name,too-many-public-methods,duplicate-code,too-many-lines

.PHONY: mypy
mypy:
	mypy src/bezzanlabs

.PHONY: flint
flint: format mypy

.PHONY: wheel
wheel:
	python -m pip wheel . --no-deps

.PHONY: build
build:
	python -m pip install "setuptools_cythonize==1.0.7"
	python setup_.py bdist_wheel --cythonize

.PHONY: test
test:
	pytest tests/

.PHONY: safety
safety:
	safety check -r requirements.txt

.PHONY: docs
docs:
	cp -r docs/ docs_temp/
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-apidoc -o ./docs_temp ./src/bezzanlabs
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-build -b html docs_temp/ docs/build/
	rm -rf docs_temp/

.PHONY: default
default:
	python -m pip install pip-tools
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	python -m pip install -r requirements.txt
	python -m pip install "setuptools_cythonize==1.0.7"
	python setup_.py bdist_wheel --cythonize
