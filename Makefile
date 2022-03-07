.PHONY: clean lint

clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

lint:
	isort dmcol/ main.py
	autoflake --in-place dmlcol/*.py main.py
	black dmlcol/ main.py
	# flake8 dmlcol/ main.py
	pylint dmlcol/ main.py
