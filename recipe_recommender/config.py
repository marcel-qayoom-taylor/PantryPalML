#!/usr/bin/env python3
"""
Configuration Management for ML Pipeline

Centralized configuration to make the system more maintainable
and easier to understand for beginners.
"""

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class TextEncodingConfig:
    """Configuration for optional text feature encodings.

    Turn these on to let text like recipe name or description
    contribute numeric signals to the model. They are off by default.
    """

    enable_text_features: bool = False

    # Encoders
    author_id_encoding: str = "freq"  # one of: none|freq|target
    tags_encoding: str = "topk_multi_hot"  # one of: none|topk_multi_hot|hashing
    name_encoding: str = "hashing"  # one of: none|hashing
    desc_encoding: str = "none"  # one of: none|tfidf|hashing
    instr_encoding: str = "none"  # one of: none|tfidf|hashing

    # Hyperparameters
    tags_top_k: int = 50
    name_hash_dim: int = 128
    desc_max_features: int = 500
    instr_hash_dim: int = 512
    hashing_alternate_sign: bool = False
    ngram_range: tuple[int, int] = (1, 2)


@dataclass
class MLConfig:
    """Configuration for the ML pipeline."""

    # Paths
    project_root: Path
    output_dir: Path
    input_dir: Path
    model_dir: Path

    # Files
    raw_recipe_features_filename: str = "enhanced_recipe_features_from_db.csv"
    encoded_recipe_features_filename: str = "enhanced_recipe_features_encoded.csv"

    # Model settings
    random_state: int = 42
    model_type: str = "lightgbm"

    # Training parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    negative_sampling_ratio: int = 4

    # Model hyperparameters
    num_leaves: int = 31
    learning_rate: float = 0.1
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 5
    min_child_samples: int = 20
    lambda_l1: float = 0.0
    lambda_l2: float = 0.0
    max_boost_rounds: int = 1000
    early_stopping_rounds: int = 50
    # Ranking evaluation cutoffs (used by Lambdarank)
    ndcg_eval_at: tuple[int, int, int] = (5, 10, 20)

    # Feature engineering
    interaction_weights: dict[str, float] = None
    text_encoding: TextEncodingConfig | None = None

    def __post_init__(self):
        """Initialize default values and create directories."""
        if self.interaction_weights is None:
            self.interaction_weights = {
                "Recipe Viewed": 1.0,
                "Recipe Link Clicked": 2.0,
                "Recipe Favourited": 4.0,
                "Recipe Cooked": 5.0,
                "Recipe Cook Started": 3.0,
                "Recipe Added To Collections": 3.5,
                "Recipe Removed From Collections": -2.0,
                "Recipe Search Queried": 0.5,
            }

        # Create directories if they don't exist
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.model_dir.mkdir(exist_ok=True, parents=True)


def get_ml_config() -> MLConfig:
    """
    Get the ML configuration with automatic path detection.

    This function finds the project root automatically
    and sets up all the paths you need for the ML pipeline.

    Returns:
        MLConfig: Complete configuration object
    """
    # Find project root (where this config.py file is located)
    config_file = Path(__file__)
    project_root = (
        config_file.parent.parent
    )  # Go up two levels from recipe_recommender/config.py

    # Set up all paths relative to project root
    recipe_recommender_dir = project_root / "recipe_recommender"

    config = MLConfig(
        project_root=project_root,
        output_dir=recipe_recommender_dir / "output",
        input_dir=recipe_recommender_dir / "input",
        model_dir=recipe_recommender_dir / "output" / "hybrid_models",
        text_encoding=TextEncodingConfig(),
    )

    # Apply tuned best params if available
    try:
        best_params_path = config.model_dir / "best_params.json"
        if best_params_path.exists():
            data = json.loads(best_params_path.read_text())
            params = data.get("params", {})
            for key in [
                "learning_rate",
                "num_leaves",
                "min_child_samples",
                "feature_fraction",
                "bagging_fraction",
                "bagging_freq",
                "lambda_l1",
                "lambda_l2",
            ]:
                if key in params and hasattr(config, key):
                    setattr(config, key, params[key])
    except Exception:
        # Best params loading is optional; ignore errors
        pass

    return config


def get_feature_columns_to_exclude() -> list:
    """
    Get columns that should be excluded from ML training.

    These are text/string columns that contain information
    but aren't useful as numerical features for the ML model.
    """
    return [
        "recipe_name",
        "recipe_img",
        "author_name",
        "author_id",
        "tags",
        "recipe_url",
        "description",
        "instruction",
        "note",
        "created_at",
        "updated_at",
        "first_interaction",
        "last_interaction",
        "primary_device",
        "primary_platform",
    ]


if __name__ == "__main__":
    """Test the configuration setup."""
    config = get_ml_config()
    print("📋 ML Configuration:")
    print(f"   Project Root: {config.project_root}")
    print(f"   Output Dir: {config.output_dir}")
    print(f"   Input Dir: {config.input_dir}")
    print(f"   Model Dir: {config.model_dir}")
    print(f"   Random State: {config.random_state}")
    print(f"   Interaction Weights: {len(config.interaction_weights)} event types")
