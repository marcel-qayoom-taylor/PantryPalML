#!/usr/bin/env python3
"""
LightGBM Hyperparameter Tuning for Ranking (Lambdarank) with Optuna.

This script loads the prepared train/val datasets and optimizes key LightGBM
hyperparameters to maximize validation NDCG@K.

Outputs:
- Saves best params to `output/hybrid_models/best_params.json`
- Prints a summary to stdout

CLI:
    python -m recipe_recommender.tuning.tune_ranker --trials 50 --timeout 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import optuna
import pandas as pd
import lightgbm as lgb

from recipe_recommender.config import get_ml_config, get_feature_columns_to_exclude
from recipe_recommender.utils import setup_logging


logger = setup_logging(__name__)


def load_datasets(config):
    train_file = config.output_dir / "hybrid_train_data.csv"
    val_file = config.output_dir / "hybrid_val_data.csv"

    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)

    # Load feature columns file to ensure consistency with training
    feature_file = config.output_dir / "hybrid_feature_columns.txt"
    if feature_file.exists():
        feature_columns = [
            line.strip() for line in feature_file.read_text().splitlines()
        ]
    else:
        # Fallback: derive from training data (excluding known cols)
        excluded = set(["user_id", "recipe_id", "label", "rating", "datetime"]) | set(
            get_feature_columns_to_exclude()
        )
        feature_columns = [c for c in train_df.columns if c not in excluded]

    # Keep only numeric/boolean columns (LightGBM requirement)
    numeric_features = []
    for col in feature_columns:
        if col in train_df.columns:
            dtype = train_df[col].dtype
            if dtype in ["int64", "float64", "bool", "int32", "float32"]:
                numeric_features.append(col)

    X_train = train_df[numeric_features]
    y_train = train_df["label"]
    X_val = val_df[numeric_features]
    y_val = val_df["label"]

    # Group arrays (per user) for ranking
    train_groups = train_df.groupby("user_id").size().values
    val_groups = val_df.groupby("user_id").size().values

    return X_train, y_train, train_groups, X_val, y_val, val_groups, numeric_features


def objective_factory(config, X_train, y_train, train_groups, X_val, y_val, val_groups):
    def objective(trial: optuna.Trial) -> float:
        params: Dict[str, Any] = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": list(config.ndcg_eval_at),
            "boosting_type": "gbdt",
            "verbosity": -1,
            "random_state": config.random_state,
            # Tunables - constrained to reasonable ranges
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int(
                "num_leaves", 16, 64
            ),  # Much more reasonable range
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
            # Optional regularization
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        }

        train_dataset = lgb.Dataset(X_train, label=y_train, group=train_groups)
        val_dataset = lgb.Dataset(
            X_val, label=y_val, group=val_groups, reference=train_dataset
        )

        booster = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset],
            valid_names=["validation"],
            num_boost_round=config.max_boost_rounds,
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds, verbose=False),
            ],
        )

        # best_score is a dict like {"validation": {"ndcg@5": value, ...}}
        val_scores = booster.best_score.get("validation", {})

        # Aggregate NDCG across requested cutoffs
        ndcgs = []
        for k in config.ndcg_eval_at:
            key = f"ndcg@{k}"
            if key in val_scores:
                ndcgs.append(val_scores[key])
        # Fallback to single metric if aggregated not available
        score = float(np.mean(ndcgs)) if ndcgs else float(list(val_scores.values())[0])
        return score

    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Tune LightGBM ranking hyperparameters with Optuna"
    )
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=None, help="Time budget in seconds"
    )
    parser.add_argument(
        "--study-name", type=str, default="gbm_ranking_tuning", help="Optuna study name"
    )
    parser.add_argument(
        "--early-stop", type=int, default=20, help="Stop if no improvement for N trials"
    )
    args = parser.parse_args()

    config = get_ml_config()
    logger.info("Loading datasets for tuning")
    (
        X_train,
        y_train,
        train_groups,
        X_val,
        y_val,
        val_groups,
        feature_columns,
    ) = load_datasets(config)

    logger.info(
        f"Tuning with {X_train.shape[0]:,} train rows, {X_val.shape[0]:,} val rows, {len(feature_columns)} features"
    )

    objective = objective_factory(
        config, X_train, y_train, train_groups, X_val, y_val, val_groups
    )

    # Use TPE sampler for better optimization
    sampler = optuna.samplers.TPESampler(seed=config.random_state, n_startup_trials=10)
    study = optuna.create_study(
        direction="maximize", study_name=args.study_name, sampler=sampler
    )

    # Early stopping callback - stop if no improvement for N consecutive trials
    class EarlyStoppingCallback:
        def __init__(self, early_stopping_rounds: int, direction: str = "maximize"):
            self.early_stopping_rounds = early_stopping_rounds
            self.direction = direction
            self.best_value = None
            self.no_improvement_count = 0

        def __call__(self, study, trial):
            if self.best_value is None:
                self.best_value = trial.value
                return

            if self.direction == "maximize":
                if trial.value > self.best_value:
                    self.best_value = trial.value
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1
            else:
                if trial.value < self.best_value:
                    self.best_value = trial.value
                    self.no_improvement_count = 0
                else:
                    self.no_improvement_count += 1

            if self.no_improvement_count >= self.early_stopping_rounds:
                study.stop()
                logger.info(
                    f"Early stopping triggered after {self.no_improvement_count} trials without improvement"
                )

    early_stop_callback = EarlyStoppingCallback(args.early_stop)

    logger.info(
        f"Starting optimization with early stopping after {args.early_stop} trials without improvement"
    )
    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        callbacks=[early_stop_callback],
    )

    best_params = study.best_params
    best_value = study.best_value

    logger.info("Best NDCG: %.6f", best_value)
    logger.info("Best params: %s", json.dumps(best_params, indent=2))

    # Save best params JSON into model_dir
    model_dir: Path = config.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / "best_params.json"
    with open(out_path, "w") as f:
        json.dump({"best_value": best_value, "params": best_params}, f, indent=2)
    logger.info("Saved best params to %s", out_path)


if __name__ == "__main__":
    main()
