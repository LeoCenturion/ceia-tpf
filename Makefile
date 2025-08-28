run-mlflow:
	poetry run mlflow ui --backend-store-uri sqlite:///mlflow.db
.PHONY: run-mlflow

run-optuna-dashboard:
	poetry run optuna-dashboard sqlite:///optuna-study.db
.PHONY: run-optuna-dashboard
