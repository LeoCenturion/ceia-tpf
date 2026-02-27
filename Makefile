run-mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
.PHONY: run-mlflow

run-optuna-dashboard:
	poetry run optuna-dashboard sqlite:///optuna-study.db
.PHONY: run-optuna-dashboard

pair-notebooks:
	./scripts/manage_notebooks.sh pair
.PHONY: pair-notebooks

sync-notebooks:
	./scripts/manage_notebooks.sh sync
.PHONY: sync-notebooks

jupyter:
	poetry run jupyter notebook
.PHONY: jupyter
