PY=python

.PHONY: venv install test lint run

venv:
	$(PY) -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt && pip install -r requirements-dev.txt

test:
	. .venv/bin/activate && pytest -q

lint:
	. .venv/bin/activate && ruff check .

run:
	. .venv/bin/activate && $(PY) manage.py runserver 0.0.0.0:8000
