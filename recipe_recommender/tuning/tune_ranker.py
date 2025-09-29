#!/usr/bin/env python3
"""
LightGBM Ranker Hyperparameter Tuning with Optuna.

Wrapper around existing tune_gbm to keep CLI consistent with new naming.
"""

from recipe_recommender.tuning.tune_gbm import main


if __name__ == "__main__":
    main()
