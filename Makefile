# Installs all package dependencies for development.
.PHONY: install
install:
	python -m pip install pip-tools
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	python -m pip install -r requirements.txt
	rm -rf requirements.txt

# Format code
.PHONY: format
format:
	isort src/
	pre-commit run --all-files

# Lint code using pylint. To be scrapped soon.
.PHONY: pylint
lint:
	python -m pylint  --init-hook='import sys; sys.setrecursionlimit(5000)' src/ --disable=useless-suppression
	python -m pylint tests/ --disable=missing-docstring,missing-function-docstring,invalid-name,redefined-outer-name,redefined-outer-name,too-many-public-methods,duplicate-code,too-many-lines

# mypy checks
.PHONY: mypy
mypy:
	mypy src/bezzanlabs

# Convenient helper - format and lint code
.PHONY: flint
flint: format mypy

# Create wheel file
.PHONY: wheel
wheel:
	python -m pip wheel . --no-deps

# Create binary wheel file. Run after `make install`.
.PHONY: build
build:
	python -m pip install pip-tools
	python -m piptools compile --extra dev -o requirements.txt pyproject.toml
	python -m pip install -r requirements.txt
	python -m pip install "setuptools_cythonize==1.0.7"
	python setup_.py bdist_wheel --cythonize
	rm -rf requirements.txt

# Tests all packages
.PHONY: test
test:
	pytest tests/

# Safety checks all packages. Run after `make install`.
.PHONY: safety
safety:
	safety check -r requirements.txt

# Builds documentation for package. Run after `make install`.
.PHONY: docs
docs:
	cp -r docs/ docs_temp/
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-apidoc -o ./docs_temp ./src/bezzanlabs
	export PYTHONPATH=$$PYTHONPATH:"." && sphinx-build -b html docs_temp/ docs/build/
	rm -rf docs_temp/
