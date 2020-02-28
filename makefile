setup:
	python3 -m venv ~/.env
	export FLASK_ENV=development

install:
	pip install --upgrade pip && pip install -r requirements.txt
