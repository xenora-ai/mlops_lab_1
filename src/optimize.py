import os
import random
import json
import matplotlib.pyplot as plt
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold


def set_global_seed(seed: int) -> None:
    """Фіксує seed для відтворюваності експериментів."""
    random.seed(seed)
    np.random.seed(seed)


def load_processed_data(prepared_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантажує 4 попередньо розділені файли (X_train, y_train, X_test, y_test) з вказаної директорії.
    """
    abs_dir = to_absolute_path(prepared_dir)

    try:
        X_train = pd.read_csv(os.path.join(abs_dir, "X_train.csv")).values
        y_train = pd.read_csv(os.path.join(abs_dir, "y_train.csv")).values.flatten()
        X_test = pd.read_csv(os.path.join(abs_dir, "X_test.csv")).values
        y_test = pd.read_csv(os.path.join(abs_dir, "y_test.csv")).values.flatten()

        return X_train, X_test, y_train, y_test
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Не знайдено файли даних у директорії {abs_dir}. Перевірте config.yaml. Деталі: {e}")


def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    """Ініціалізує модель на основі конфігурації."""
    if model_type == "random_forest":
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **params)

    raise ValueError(f"Unknown model.type='{model_type}'. Expecting 'random_forest'.")


def evaluate(model: Any, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
             metric: str) -> float:
    """Навчає модель та обчислює задану метрику для регресії."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if metric == "r2":
        return float(r2_score(y_test, y_pred))
    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_test, y_pred)))

    raise ValueError("Unsupported metrics. Use 'r2' or 'rmse'.")


def evaluate_cv(model: Any, X: np.ndarray, y: np.ndarray, metric: str, seed: int, n_splits: int = 5) -> float:
    """Проводить крос-валідацію. Змінено StratifiedKFold на KFold для неперервних величин."""
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        m = clone(model)
        scores.append(evaluate(m, X_tr, y_tr, X_te, y_te, metric))

    return float(np.mean(scores))


def make_sampler(sampler_name: str, seed: int, grid_space: Dict[str, Any] = None) -> optuna.samplers.BaseSampler:
    """Створює об'єкт семплера для Optuna."""
    sampler_name = sampler_name.lower()
    if sampler_name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "grid":
        if not grid_space:
            raise ValueError("For sampler='grid' need to set grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)

    raise ValueError("sampler should be: tpe, random, grid")


def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    """Генерує параметри для поточної спроби Optuna на основі Hydra Groups."""
    if model_type == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", cfg.model.n_estimators.low, cfg.model.n_estimators.high),
            "max_depth": trial.suggest_int("max_depth", cfg.model.max_depth.low, cfg.model.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split", cfg.model.min_samples_split.low,
                                                   cfg.model.min_samples_split.high),
        }

    raise ValueError(f"Unknown model.type='{model_type}'.")


def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    """Фабрика для створення функції objective з урахуванням конфігурації."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        # Створення вкладеного запуску в MLflow
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("seed", cfg.seed)
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(model, X, y, metric=cfg.hpo.metric, seed=cfg.seed, n_splits=cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)

            mlflow.log_metric(cfg.hpo.metric, score)
            return score

    return objective


def register_model_if_enabled(model_uri: str, model_name: str, stage: str) -> None:
    """Реєструє найкращу модель у Model Registry."""
    client = mlflow.tracking.MlflowClient()
    mv = mlflow.register_model(model_uri, model_name)
    client.transition_model_version_stage(name=model_name, version=mv.version, stage=stage)
    client.set_model_version_tag(model_name, mv.version, "registered_by", "lab3")
    client.set_model_version_tag(model_name, mv.version, "stage", stage)


def main(cfg: DictConfig) -> None:
    """Головна функція оркестрації пайплайну."""
    set_global_seed(cfg.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Завантаження даних з директорії, вказаної у конфізі
    X_train, X_test, y_train, y_test = load_processed_data(cfg.data.prepared_dir)

    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        # Перевіряємо, чи є в нашому hpo-конфігу (grid.yaml) опис сітки
        if "grid_space" in cfg.hpo:
            grid_space = OmegaConf.to_container(cfg.hpo.grid_space, resolve=True)
        else:
            raise ValueError("Для sampler='grid' необхідно визначити hpo.grid_space у файлі hpo/grid.yaml")

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)

    # Батьківський запуск, що об'єднує всі trials
    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler", cfg.hpo.sampler)
        mlflow.set_tag("seed", cfg.seed)

        # Збереження конфігурації як артефакту
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")

        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)

        # Запуск процесу оптимізації
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial
        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")

        # Навчання фінальної найкращої моделі
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        best_score = evaluate(best_model, X_train, y_train, X_test, y_test, metric=cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", best_score)

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        metrics_dict = {
            "r2": float(best_score),
            "best_r2_in_hpo": float(best_trial.value)
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics_dict, f, indent=4)

        y_pred = best_model.predict(X_test)
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted Salary")
        plt.savefig("confusion_matrix.png")

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            register_model_if_enabled(model_uri, cfg.mlflow.model_name, stage=cfg.mlflow.stage)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
