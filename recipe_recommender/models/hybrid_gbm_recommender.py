#!/usr/bin/env python3
"""
Hybrid GBM Recipe Recommender

A production-ready hybrid recommendation model using gradient boosting with:
- Clear separation of concerns
- Step-by-step workflow
- Better documentation
- Centralized configuration
"""

import json
import warnings
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    ndcg_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from recipe_recommender.config import get_feature_columns_to_exclude, get_ml_config
from recipe_recommender.utils import setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)


class HybridGBMRecommender:
    """
    A production-ready hybrid recommendation model using gradient boosting.

    This class breaks down the ML pipeline into clear, understandable steps:
    1. Load data
    2. Prepare features
    3. Train model
    4. Evaluate performance
    5. Save results
    """

    def __init__(self, config=None):
        """
        Initialize the recommender with configuration.

        The config object contains all the settings
        like file paths, model parameters, etc. This makes it easy
        to change settings without hunting through code.

        Args:
            config: ML configuration object (uses defaults if None)
        """
        self.config = config or get_ml_config()

        # Model components
        self.model: lgb.Booster | None = None
        self.feature_columns: list[str] = []
        self.training_metadata: dict = {}

        # Data storage
        self.train_data: pd.DataFrame | None = None
        self.val_data: pd.DataFrame | None = None
        self.test_data: pd.DataFrame | None = None
        self.recipe_features: pd.DataFrame | None = None

        logger.info(f"Initialized GBM Recommender with {self.config.model_type}")

    def load_training_data(self) -> bool:
        """
        Load the prepared training datasets.

        This method loads the CSV files that were created
        by the data builder. It expects train/validation/test splits.

        Returns:
            bool: True if successful, False otherwise
        """
        logger.info("Loading training data")

        try:
            # Load the three data splits
            train_file = self.config.output_dir / "hybrid_train_data.csv"
            val_file = self.config.output_dir / "hybrid_val_data.csv"
            test_file = self.config.output_dir / "hybrid_test_data.csv"

            self.train_data = pd.read_csv(train_file)
            self.val_data = pd.read_csv(val_file)
            self.test_data = pd.read_csv(test_file)

            logger.info("Successfully loaded training data:")
            logger.info(f"   Train: {len(self.train_data):,} samples")
            logger.info(f"   Validation: {len(self.val_data):,} samples")
            logger.info(f"   Test: {len(self.test_data):,} samples")

            # Load feature columns list
            self._load_feature_columns()

            # Load metadata about the training data
            self._load_training_metadata()

            return True

        except Exception:
            logger.exception("Failed to load training data")
            return False

    def _load_feature_columns(self) -> None:
        """Load the list of features to use in training."""
        feature_file = self.config.output_dir / "hybrid_feature_columns.txt"
        if feature_file.exists():
            self.feature_columns = [
                line.strip() for line in feature_file.read_text().splitlines()
            ]
            logger.info(f"Loaded {len(self.feature_columns)} feature columns")
        else:
            logger.warning("Feature columns file not found")

    def _load_training_metadata(self) -> None:
        """Load metadata about how the training data was created."""
        metadata_file = self.config.output_dir / "hybrid_training_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                self.training_metadata = json.load(f)
            logger.info("Loaded training metadata")
        else:
            logger.warning("Training metadata not found")

    def load_recipe_features(self) -> bool:
        """Load full recipe features for scoring all recipes."""
        try:
            # Prefer encoded recipe features if present
            encoded_path = (
                self.config.output_dir / self.config.encoded_recipe_features_filename
            )
            raw_path = self.config.output_dir / self.config.raw_recipe_features_filename

            if encoded_path.exists():
                self.recipe_features = pd.read_csv(encoded_path)
                logger.info(f"Loaded encoded recipe features from {encoded_path.name}")
            else:
                self.recipe_features = pd.read_csv(raw_path)
                logger.info(f"Loaded raw recipe features from {raw_path.name}")
            self.recipe_features["recipe_id"] = self.recipe_features[
                "recipe_id"
            ].astype(str)
            logger.info(f"Loaded {len(self.recipe_features)} recipes for scoring")
            return True
        except Exception:
            logger.exception("Error loading recipe features")
            return False

    def prepare_features(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Prepare the features for machine learning.

        This step filters out text/string columns that
        aren't useful for ML and keeps only numeric features that the
        model can learn from.

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        logger.info("Preparing features for training")

        if self.train_data is None:
            raise ValueError(
                "Training data not loaded. Call load_training_data() first."
            )

        # Get columns to exclude from ML training
        excluded_columns = get_feature_columns_to_exclude()

        # Filter to only numeric/boolean features
        ml_features = []
        for col in self.feature_columns:
            if col not in excluded_columns and col in self.train_data.columns:
                dtype = self.train_data[col].dtype
                if dtype in ["int64", "float64", "bool", "int32", "float32"]:
                    ml_features.append(col)
                else:
                    logger.debug(f"   Excluding non-numeric column: {col}")

        # Update feature columns to filtered list
        self.feature_columns = ml_features

        logger.info(f"Using {len(ml_features)} numeric features")

        # Prepare training data
        X_train = self.train_data[self.feature_columns]
        y_train = self.train_data["label"]
        X_val = self.val_data[self.feature_columns]
        y_val = self.val_data["label"]

        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(
            f"   Positive samples: {y_train.sum():,} ({y_train.mean() * 100:.1f}%)"
        )

        return X_train, y_train, X_val, y_val

    def train_model(self) -> bool:
        """
        Train the gradient boosting model.

        This is where the actual machine learning happens.
        The model learns patterns from the training data to predict which
        recipes a user might like.

        Returns:
            bool: True if training successful
        """
        logger.info(f"Training {self.config.model_type.upper()} model")

        # Prepare the features
        X_train, y_train, X_val, y_val = self.prepare_features()

        # Set up model parameters for ranking (Lambdarank)
        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "ndcg_eval_at": list(self.config.ndcg_eval_at),
            "boosting_type": "gbdt",
            "num_leaves": self.config.num_leaves,
            "learning_rate": self.config.learning_rate,
            "feature_fraction": self.config.feature_fraction,
            "bagging_fraction": self.config.bagging_fraction,
            "bagging_freq": self.config.bagging_freq,
            "min_child_samples": self.config.min_child_samples,
            "lambda_l1": getattr(self.config, "lambda_l1", 0.0),
            "lambda_l2": getattr(self.config, "lambda_l2", 0.0),
            "random_state": self.config.random_state,
            "verbosity": -1,
        }

        # Build group arrays per user for train/val
        train_groups = self.train_data.groupby("user_id").size().values
        val_groups = self.val_data.groupby("user_id").size().values

        # Create LightGBM datasets with groups
        train_dataset = lgb.Dataset(X_train, label=y_train, group=train_groups)
        val_dataset = lgb.Dataset(
            X_val, label=y_val, group=val_groups, reference=train_dataset
        )

        # Train the model
        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[train_dataset, val_dataset],
            valid_names=["train", "validation"],
            num_boost_round=self.config.max_boost_rounds,
            callbacks=[
                lgb.early_stopping(self.config.early_stopping_rounds),
                lgb.log_evaluation(100),
            ],
        )

        logger.info("Model training completed")
        return True

    def evaluate_model(self) -> dict[str, float]:
        """
        Evaluate how well the trained model performs.

        This tests the model on data it hasn't seen before
        to see how accurately it can predict user preferences.

        Returns:
            Dict of performance metrics
        """
        logger.info("Evaluating model performance")

        if self.model is None:
            logger.error("No trained model available")
            return {}

        # Prepare test data
        X_test = self.test_data[self.feature_columns]
        y_test = self.test_data["label"]

        # Get model predictions (scores for ranking)
        y_pred_proba = self.model.predict(
            X_test, num_iteration=self.model.best_iteration
        )
        # For binary metrics, we can derive labels via 0.5 threshold (for reference only)
        y_pred = (y_pred_proba > 0.5).astype(int)

        # Calculate performance metrics
        # These metrics tell us how good the model is
        metrics = {
            "auc": roc_auc_score(
                y_test, y_pred_proba
            ),  # Overall discrimination ability
            "precision": precision_score(
                y_test, y_pred
            ),  # Accuracy of positive predictions
            "recall": recall_score(y_test, y_pred),  # How many positives we caught
            "f1": f1_score(y_test, y_pred),  # Balance of precision and recall
            "average_precision": average_precision_score(y_test, y_pred_proba),
        }

        # Calculate ranking metrics per user (if possible)
        ranking_metrics = self._evaluate_ranking_metrics(y_test, y_pred_proba)
        metrics.update(ranking_metrics)

        # Log the results (ranking-focused)
        logger.info("Model performance:")
        if "ndcg_5" in metrics:
            logger.info(f"   NDCG@5: {metrics['ndcg_5']:.4f}")
            logger.info(f"   NDCG@10: {metrics['ndcg_10']:.4f}")
        logger.info(f"   AUC: {metrics['auc']:.4f} (reference)")
        logger.info(f"   Precision: {metrics['precision']:.4f} (reference)")
        logger.info(f"   Recall: {metrics['recall']:.4f} (reference)")
        logger.info(f"   F1-Score: {metrics['f1']:.4f} (reference)")

        return metrics

    def _evaluate_ranking_metrics(self, y_true, y_pred_proba) -> dict:
        """Evaluate ranking metrics per user."""
        try:
            # Add user_id for grouping
            test_with_preds = self.test_data[["user_id"]].copy()
            test_with_preds["y_true"] = y_true.to_numpy()
            test_with_preds["y_pred_proba"] = y_pred_proba

            user_ndcg_5 = []
            user_ndcg_10 = []
            user_recall_5 = []
            user_recall_10 = []

            for user_id in test_with_preds["user_id"].unique():
                user_data = test_with_preds[test_with_preds["user_id"] == user_id]

                if len(user_data) < 2:  # Need at least 2 items for ranking
                    continue

                y_true_user = user_data["y_true"].to_numpy().reshape(1, -1)
                y_score_user = user_data["y_pred_proba"].to_numpy().reshape(1, -1)

                try:
                    ndcg_5 = ndcg_score(
                        y_true_user, y_score_user, k=min(5, len(user_data))
                    )
                    ndcg_10 = ndcg_score(
                        y_true_user, y_score_user, k=min(10, len(user_data))
                    )

                    user_ndcg_5.append(ndcg_5)
                    user_ndcg_10.append(ndcg_10)
                except Exception:
                    continue

                # Recall@k (only for users with at least one positive)
                y_true_flat = y_true_user.ravel()
                pos_total = float(y_true_flat.sum())
                if pos_total > 0:
                    order = np.argsort(-y_score_user.ravel())
                    top5 = order[: min(5, len(order))]
                    top10 = order[: min(10, len(order))]
                    rec5 = y_true_flat[top5].sum() / pos_total
                    rec10 = y_true_flat[top10].sum() / pos_total
                    user_recall_5.append(rec5)
                    user_recall_10.append(rec10)

            if user_ndcg_5:
                result = {
                    "ndcg_5": np.mean(user_ndcg_5),
                    "ndcg_10": np.mean(user_ndcg_10),
                    "users_evaluated": len(user_ndcg_5),
                }
                if user_recall_5:
                    result["recall_5"] = float(np.mean(user_recall_5))
                    result["recall_10"] = float(np.mean(user_recall_10))
                return result
            return {"users_evaluated": 0}

        except Exception as e:
            logger.warning(f"Could not calculate ranking metrics: {e}")
            return {}

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get which features the model found most important.

        This shows which user/recipe characteristics
        matter most for making good recommendations.

        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            logger.warning("No trained model available")
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")

        importance_df = pd.DataFrame(
            {"feature": self.feature_columns, "importance": importance}
        ).sort_values("importance", ascending=False)

        return importance_df

    def save_model(self) -> bool:
        """
        Save the trained model for later use.

        This saves your trained model so you can use it
        later without having to retrain from scratch.

        Returns:
            bool: True if save successful
        """
        logger.info("Saving trained model")

        if self.model is None:
            logger.error("No model to save")
            return False

        # Save the model file
        model_file = (
            self.config.model_dir / f"hybrid_{self.config.model_type}_model.txt"
        )
        self.model.save_model(str(model_file))

        # Save metadata about the model
        metadata = {
            "model_type": self.config.model_type,
            "feature_columns": self.feature_columns,
            "training_metadata": self.training_metadata,
            "model_file": str(model_file),
            "trained_at": datetime.now().isoformat(),
            "config": {
                "num_leaves": self.config.num_leaves,
                "learning_rate": self.config.learning_rate,
                "random_state": self.config.random_state,
            },
        }

        metadata_file = (
            self.config.model_dir / f"hybrid_{self.config.model_type}_metadata.json"
        )
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Model saved to: {model_file.name}")
        logger.info(f"Metadata saved to: {metadata_file.name}")

        return True


def main():
    """
    Main function to train and evaluate the model.

    This is the complete workflow from start to finish.
    """
    print("Hybrid GBM Recipe Recommender")
    print("-" * 60)

    # Initialize with configuration
    recommender = HybridGBMRecommender()

    # Step 1: Load data
    if not recommender.load_training_data():
        print("Failed to load training data")
        return

    if not recommender.load_recipe_features():
        print("Failed to load recipe features")
        return

    # Step 2: Train model
    if not recommender.train_model():
        print("Failed to train model")
        return

        # Step 3: Evaluate performance
    recommender.evaluate_model()

    # Step 4: Show feature importance
    importance = recommender.get_feature_importance()
    if not importance.empty:
        print("\nTop 10 most important features:")
        for _, row in importance.head(10).iterrows():
            print(f"   {row['feature']}: {row['importance']:.0f}")

    # Step 5: Save the model
    recommender.save_model()

    print("\nTraining completed successfully")
    print("   Model can now score and rank recipes for any user")
    print("   Next: Use the trained model for real-time recommendations")


if __name__ == "__main__":
    main()
