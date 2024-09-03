PYTHON_VERSION ?= 3.9
PY_DIRECTORIES=src/ tests/

.PHONY: install-requirements
install-requirements:
	PYENV_VERSION=${PYTHON_VERSION} python -m venv .venv
	.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org --upgrade pip
	.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -e .

.PHONY: install-dev-requirements
install-dev-requirements: install-requirements
	.venv/bin/pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements-dev.txt

.PHONY: tests
tests:
	.venv/bin/pytest -v -s

.PHONY: format
format:
	.venv/bin/black ${PY_DIRECTORIES}
	.venv/bin/isort ${PY_DIRECTORIES}

.PHONY: clean
clean:
	rm -r .venv/
