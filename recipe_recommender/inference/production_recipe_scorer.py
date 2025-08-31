#!/usr/bin/env python3
"""
Production Recipe Scorer API

This API takes a user's interaction history and scores/ranks all available recipes
using the trained hybrid GBM model. It provides real-time recipe recommendations
based on learned user preferences and rich recipe content features.
"""

import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from recipe_recommender.config import get_ml_config
from recipe_recommender.utils import load_json_file, safe_load_csv, setup_logging

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)


class ProductionRecipeScorer:
    """
    Production API for scoring and ranking recipes for users.

    For beginners: This class loads a trained ML model and uses it to
    score recipes for users based on their interaction history.
    """

    def __init__(self, config=None, model_path: str | None = None):
        """
        Initialize the production recipe scorer.

        Args:
            config: ML configuration object (uses defaults if None)
            model_path: Path to trained model directory (overrides config if provided)
        """
        self.config = config or get_ml_config()
        self.model_dir = Path(model_path) if model_path else self.config.model_dir

        # Model components
        self.model: lgb.Booster | None = None
        self.feature_columns: list[str] = []
        self.metadata: dict = {}

        # Data
        self.recipe_features: pd.DataFrame = pd.DataFrame()
        self.user_profiles: dict = {}  # Cache user profiles

        logger.info("🎯 Initializing Production Recipe Scorer...")

        # Load model and data
        self._load_model_components()
        self._load_recipe_data()

    def _load_model_components(self) -> None:
        """
        Load the trained model and metadata.

        For beginners: This loads the previously trained LightGBM model
        and its configuration so we can use it for predictions.
        """
        logger.info("🔧 Loading trained model components...")

        try:
            # Load LightGBM model
            model_file = self.model_dir / f"hybrid_{self.config.model_type}_model.txt"
            if model_file.exists():
                self.model = lgb.Booster(model_file=str(model_file))
                logger.info(f"✅ Loaded {self.config.model_type.upper()} model")
            else:
                raise FileNotFoundError(f"Model file not found: {model_file}")

            # Load metadata using utility
            metadata_file = (
                self.model_dir / f"hybrid_{self.config.model_type}_metadata.json"
            )
            self.metadata = load_json_file(metadata_file)
            if self.metadata:
                self.feature_columns = self.metadata["feature_columns"]
                logger.info(
                    f"✅ Loaded model metadata with {len(self.feature_columns)} features"
                )
            else:
                logger.warning(
                    "⚠️  Metadata file not found, using default feature columns"
                )

        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            raise

    def _load_recipe_data(self) -> None:
        """
        Load recipe features for scoring.

        For beginners: This loads all the recipe data so we can score
        any recipe for any user.
        """
        try:
            recipe_file = (
                self.config.output_dir / "enhanced_recipe_features_from_db.csv"
            )
            self.recipe_features = safe_load_csv(recipe_file)

            if self.recipe_features is None:
                raise FileNotFoundError("Recipe features file not found")

            self.recipe_features["recipe_id"] = self.recipe_features[
                "recipe_id"
            ].astype(str)
            logger.info(f"✅ Loaded {len(self.recipe_features)} recipes for scoring")

        except Exception as e:
            logger.error(f"❌ Error loading recipe data: {e}")
            raise

    def create_user_profile_from_interactions(self, interactions: list[dict]) -> dict:
        """
        Create user profile from a list of interactions.

        For beginners: This analyzes a user's recipe interactions to understand
        their preferences and behavior patterns.

        Args:
            interactions: List of interaction dictionaries with keys:
                - recipe_id: str
                - event_type: str ('Recipe Viewed', 'Recipe Cooked', etc.)
                - timestamp: int (unix timestamp)

        Returns:
            Dictionary of user profile features
        """
        if not interactions:
            return self._get_default_user_profile()

        # Convert to DataFrame
        interactions_df = pd.DataFrame(interactions)

        # Add rating weights from config
        interaction_weights = self.config.interaction_weights

        interactions_df["rating"] = (
            interactions_df["event_type"].map(interaction_weights).fillna(1.0)
        )
        interactions_df["datetime"] = pd.to_datetime(
            interactions_df["timestamp"], unit="s", errors="coerce"
        )

        # Calculate user profile features
        profile = {
            "total_interactions": len(interactions_df),
            "avg_rating": interactions_df["rating"].mean(),
            "total_rating": interactions_df["rating"].sum(),
            "rating_std": interactions_df["rating"].std() or 0.0,
            "unique_recipes": interactions_df["recipe_id"].nunique(),
        }

        # Temporal features
        if (
            "datetime" in interactions_df.columns
            and not interactions_df["datetime"].isna().all()
        ):
            first_interaction = interactions_df["datetime"].min()
            last_interaction = interactions_df["datetime"].max()
            activity_days = max((last_interaction - first_interaction).days, 1)

            profile.update(
                {
                    "activity_days": activity_days,
                    "interactions_per_day": len(interactions_df) / activity_days,
                    "engagement_score": profile["total_rating"] / activity_days,
                }
            )
        else:
            profile.update(
                {
                    "activity_days": 1,
                    "interactions_per_day": len(interactions_df),
                    "engagement_score": profile["total_rating"],
                }
            )

        # Device preferences (simplified for API)
        profile.update(
            {
                "is_mobile_user": 1,  # Default assumption
                "is_ios_user": 1,  # Default assumption
            }
        )

        return profile

    def _get_default_user_profile(self) -> dict:
        """Get default profile for new users."""
        return {
            "total_interactions": 0,
            "avg_rating": 2.5,  # Neutral rating
            "total_rating": 0,
            "rating_std": 0.0,
            "unique_recipes": 0,
            "activity_days": 1,
            "interactions_per_day": 0,
            "engagement_score": 0,
            "is_mobile_user": 1,
            "is_ios_user": 1,
        }

    def score_all_recipes_for_user(
        self,
        user_interactions: list[dict],
        n_recommendations: int = 20,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Score all available recipes for a user based on their interaction history.

        For beginners: This is the main method that takes a user's history
        and scores all recipes to find the best recommendations.

        Args:
            user_interactions: List of user's recipe interactions
            n_recommendations: Number of top recommendations to return
            min_score: Minimum score threshold for recommendations

        Returns:
            List of ranked recipe recommendations with scores and metadata
        """
        logger.info(f"🎯 Scoring {len(self.recipe_features)} recipes for user...")

        # Create user profile from interactions
        user_profile = self.create_user_profile_from_interactions(user_interactions)

        # Create user-recipe feature combinations
        user_recipe_features = []

        for _, recipe in self.recipe_features.iterrows():
            # Combine user profile with recipe features
            combination = user_profile.copy()

            # Add recipe features
            for col, value in recipe.items():
                combination[col] = value

            # Add interaction features
            combination = self._calculate_interaction_features(combination)

            user_recipe_features.append(combination)

        # Convert to DataFrame and select model features
        features_df = pd.DataFrame(user_recipe_features)

        # Select only the features the model was trained on
        available_features = [
            col for col in self.feature_columns if col in features_df.columns
        ]
        X_score = features_df[available_features]

        # Fill any missing features with defaults
        for col in self.feature_columns:
            if col not in X_score.columns:
                X_score[col] = 0  # Default value for missing features

        # Ensure correct column order
        X_score = X_score[self.feature_columns]

        # Get model predictions
        scores = self.model.predict(X_score, num_iteration=self.model.best_iteration)

        # Create recommendations with scores and metadata
        recommendations = []
        for idx, recipe in self.recipe_features.iterrows():
            score = float(scores[idx])

            if score >= min_score:
                rec = {
                    "recipe_id": str(recipe["recipe_id"]),
                    "recipe_name": recipe.get("recipe_name", "Unknown"),
                    "score": score,
                    "author_name": recipe.get("author_name", "Unknown"),
                    "total_time": recipe.get("total_time", None),
                    "servings": recipe.get("servings", None),
                    "tags": recipe.get("tags", []),
                    "ingredient_count": recipe.get("ingredient_count", 0),
                    "complexity_score": recipe.get("complexity_score", 0.0),
                    "recipe_url": recipe.get("recipe_url", ""),
                    "description": (
                        recipe.get("description", "")[:200] + "..."
                        if len(str(recipe.get("description", ""))) > 200
                        else recipe.get("description", "")
                    ),
                }
                recommendations.append(rec)

        # Sort by score and return top N
        recommendations.sort(key=lambda x: x["score"], reverse=True)

        logger.info(f"✅ Generated scores for {len(self.recipe_features)} recipes")
        logger.info(f"   Score range: {np.min(scores):.4f} - {np.max(scores):.4f}")
        logger.info(f"   Above threshold ({min_score}): {len(recommendations)} recipes")
        logger.info(
            f"   Returning top {min(n_recommendations, len(recommendations))} recommendations"
        )

        return recommendations[:n_recommendations]

    def _calculate_interaction_features(self, combination: dict) -> dict:
        """Calculate interaction features for a user-recipe combination."""

        # User's average rating vs recipe complexity
        if "avg_rating" in combination and "complexity_score" in combination:
            avg_rating = combination.get("avg_rating", 0) or 0
            complexity = combination.get("complexity_score", 0) or 0
            combination["user_complexity_match"] = float(avg_rating) * float(complexity)
        else:
            combination["user_complexity_match"] = 0.0

        # User engagement vs recipe characteristics
        if "engagement_score" in combination and "ingredient_count" in combination:
            engagement = combination.get("engagement_score", 0) or 0
            ingredients = combination.get("ingredient_count", 0) or 0
            combination["user_recipe_engagement_match"] = float(engagement) / (
                float(ingredients) + 1
            )
        else:
            combination["user_recipe_engagement_match"] = 0.0

        # User activity level vs recipe time requirements
        if "interactions_per_day" in combination and "total_time" in combination:
            activity = combination.get("interactions_per_day", 0) or 0
            total_time = combination.get("total_time", 30) or 30
            combination["user_time_compatibility"] = float(activity) / (
                float(total_time) + 1
            )
        else:
            combination["user_time_compatibility"] = 0.0

        return combination

    def get_user_recipe_recommendations(
        self, user_id: str, interaction_history: list[dict], n_recommendations: int = 10
    ) -> dict:
        """
        Get recipe recommendations for a user.

        Args:
            user_id: User identifier
            interaction_history: List of user's recipe interactions
            n_recommendations: Number of recommendations to return

        Returns:
            Dictionary with recommendations and metadata
        """
        start_time = datetime.now()

        recommendations = self.score_all_recipes_for_user(
            interaction_history, n_recommendations=n_recommendations, min_score=0.0
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "metadata": {
                "model_type": "hybrid_lightgbm",
                "total_recipes_scored": len(self.recipe_features),
                "processing_time_seconds": processing_time,
                "generated_at": datetime.now().isoformat(),
                "user_interaction_count": len(interaction_history),
                "recommendation_count": len(recommendations),
            },
        }


def demo_usage():
    """
    Demonstrate how to use the production recipe scorer.

    For beginners: This shows a complete example of how to get
    recipe recommendations for a user.
    """
    logger.info("🧪 DEMO: Production Recipe Scorer Usage")
    logger.info("=" * 60)

    # Initialize the scorer
    scorer = ProductionRecipeScorer()

    # Example user interaction history
    sample_interactions = [
        {"recipe_id": "100", "event_type": "Recipe Viewed", "timestamp": 1755665991},
        {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000},
        {
            "recipe_id": "1291",
            "event_type": "Recipe Favourited",
            "timestamp": 1755666100,
        },
    ]

    # Get recommendations
    logger.info(
        f"🎯 Getting recommendations for user with {len(sample_interactions)} interactions..."
    )

    result = scorer.get_user_recipe_recommendations(
        user_id="demo_user",
        interaction_history=sample_interactions,
        n_recommendations=5,
    )

    # Display results
    logger.info("\n📊 RECOMMENDATION RESULTS:")
    logger.info(
        f"   Processing time: {result['metadata']['processing_time_seconds']:.3f} seconds"
    )
    logger.info(
        f"   Total recipes scored: {result['metadata']['total_recipes_scored']}"
    )

    logger.info("\n🍳 TOP 5 RECOMMENDATIONS:")
    for i, rec in enumerate(result["recommendations"], 1):
        logger.info(f"   {i}. {rec['recipe_name']} (Score: {rec['score']:.4f})")
        logger.info(f"      Author: {rec['author_name']}")
        logger.info(f"      Time: {rec['total_time']}min, Servings: {rec['servings']}")
        logger.info(f"      Tags: {rec['tags']}")
        logger.info(
            f"      Ingredients: {rec['ingredient_count']}, Complexity: {rec['complexity_score']:.2f}"
        )
        logger.info("")

    return result


def api_example():
    """
    Example of how this would be used in a production API.

    For beginners: This shows how to integrate the scorer into a web API.
    """
    logger.info("\n🌐 PRODUCTION API EXAMPLE")
    logger.info("=" * 40)

    logger.info(
        """
# Example FastAPI endpoint:

from fastapi import FastAPI
from production_recipe_scorer import ProductionRecipeScorer

app = FastAPI()
scorer = ProductionRecipeScorer()

@app.post("/recommendations")
async def get_recommendations(user_id: str, interactions: List[Dict]):
    '''Get recipe recommendations for a user.'''

    recommendations = scorer.get_user_recipe_recommendations(
        user_id=user_id,
        interaction_history=interactions,
        n_recommendations=20
    )

    return recommendations

# Usage:
# POST /recommendations
# {
#   "user_id": "user123",
#   "interactions": [
#     {"recipe_id": "100", "event_type": "Recipe Viewed", "timestamp": 1755665991},
#     {"recipe_id": "100", "event_type": "Recipe Cooked", "timestamp": 1755666000}
#   ]
# }
"""
    )


def main():
    """
    Main function to test the production recipe scorer.

    For beginners: This runs a complete test of the recommendation system.
    """
    logger.info("🚀 PRODUCTION RECIPE SCORER TEST")
    logger.info("=" * 70)

    try:
        # Run demo
        demo_usage()

        # Show API example
        api_example()

        logger.info("🎉 Production Recipe Scorer is ready!")
        logger.info("   ✅ Can score all recipes for any user in real-time")
        logger.info("   ✅ Handles user interaction history")
        logger.info("   ✅ Returns ranked recommendations with metadata")
        logger.info("   ✅ Ready for integration into production API")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        logger.error("   Make sure the hybrid model has been trained first")


if __name__ == "__main__":
    main()
