#!/usr/bin/env python3
"""
Configuration Management for ML Pipeline

Centralized configuration to make the system more maintainable
and easier to understand for beginners.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MLConfig:
    """Configuration for the ML pipeline."""

    # Paths
    project_root: Path
    output_dir: Path
    input_dir: Path
    model_dir: Path

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
    max_boost_rounds: int = 1000
    early_stopping_rounds: int = 50

    # Feature engineering
    interaction_weights: dict[str, float] = None

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

    For beginners: This function finds the project root automatically
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

    return MLConfig(
        project_root=project_root,
        output_dir=recipe_recommender_dir / "output",
        input_dir=recipe_recommender_dir / "input",
        model_dir=recipe_recommender_dir / "output" / "hybrid_models",
    )


def get_feature_columns_to_exclude() -> list:
    """
    Get columns that should be excluded from ML training.

    For beginners: These are text/string columns that contain information
    but aren't useful as numerical features for the ML model.
    """
    return [
        "recipe_name",
        "recipe_img",
        "author_name",
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
