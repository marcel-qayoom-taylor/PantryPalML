#!/usr/bin/env python3
"""
Hybrid Recommendation Data Builder

This script combines user interaction history from events with the rich recipe
database to create training data for a hybrid recommendation model that can
score and rank recipes based on user preferences.
"""

import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from recipe_recommender.config import get_ml_config
from recipe_recommender.utils import (
    safe_load_csv,
    safe_save_csv,
    save_json_file,
    setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)


class HybridRecommendationDataBuilder:
    """
    Build training data combining user interactions with rich recipe features.

    For beginners: This class takes raw user interaction data and recipe information
    to create the training dataset that the ML model learns from.
    """

    def __init__(self, config=None):
        """
        Initialize the data builder with configuration.

        Args:
            config: ML configuration object (uses defaults if None)
        """
        self.config = config or get_ml_config()

        # Data storage
        self.user_interactions: pd.DataFrame = pd.DataFrame()
        self.recipe_features: pd.DataFrame = pd.DataFrame()
        self.recipe_ingredients: pd.DataFrame = pd.DataFrame()
        self.ingredients: pd.DataFrame = pd.DataFrame()

        logger.info("🏗️ Initialized Hybrid Recommendation Data Builder")

    def load_real_recipe_data(self) -> bool:
        """
        Load the extracted real recipe database.

        For beginners: This loads the recipe data that was fetched from Supabase,
        including recipe details, ingredients, and relationships between them.

        Returns:
            bool: True if successful
        """
        logger.info("📊 Loading real recipe database...")

        try:
            # Load enhanced recipe features
            recipe_file = (
                self.config.output_dir / "enhanced_recipe_features_from_db.csv"
            )
            self.recipe_features = safe_load_csv(recipe_file)
            if self.recipe_features is None:
                logger.error(
                    "Enhanced recipe features not found. Run fetch_real_recipes.py first."
                )
                return False

            logger.info(
                f"✅ Loaded {len(self.recipe_features)} recipes with enhanced features"
            )

            # Load recipe-ingredient relationships
            ingredients_file = (
                self.config.output_dir / "real_recipe_ingredients_from_db.csv"
            )
            self.recipe_ingredients = safe_load_csv(ingredients_file)
            if self.recipe_ingredients is not None:
                logger.info(
                    f"✅ Loaded {len(self.recipe_ingredients)} recipe-ingredient relationships"
                )

            # Load ingredient data
            ingredient_master_file = (
                self.config.output_dir / "real_ingredients_from_db.csv"
            )
            self.ingredients = safe_load_csv(ingredient_master_file)
            if self.ingredients is not None:
                logger.info(f"✅ Loaded {len(self.ingredients)} ingredients")

            return True

        except Exception as e:
            logger.error(f"❌ Error loading real recipe data: {e}")
            return False

    def extract_user_interactions_from_events(self) -> bool:
        """
        Extract user interactions from event files.

        For beginners: This reads the JSON event files containing user interactions
        (recipe views, cooks, favorites) and converts them into structured data.

        Returns:
            bool: True if successful
        """
        logger.info("📱 Extracting user interactions from events...")

        interactions = []

        # Process both v1 and v2 events
        for version, filename in [
            ("v1", "v1_events_20250827.json"),
            ("v2", "v2_events_20250827.json"),
        ]:
            file_path = self.config.input_dir / filename
            logger.info(f"   Processing {filename}...")

            with open(file_path) as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        event = json.loads(line.strip())

                        # Check if this is a recipe-related event
                        event_name = event.get("event", "")
                        if (
                            "Recipe" in event_name
                            and event_name in self.config.interaction_weights
                        ):
                            props = event.get("properties", {})

                            recipe_id = props.get("recipe_id")
                            user_id = event.get("distinct_id", props.get("distinct_id"))
                            timestamp = props.get("time", event.get("timestamp"))

                            if recipe_id and user_id and timestamp:
                                # Create interaction record
                                interaction = {
                                    "user_id": user_id,
                                    "recipe_id": str(
                                        recipe_id
                                    ),  # Ensure string consistency
                                    "event_type": event_name,
                                    "rating": self.config.interaction_weights[
                                        event_name
                                    ],
                                    "timestamp": timestamp,
                                    "version": version,
                                    "device_type": props.get("device_type", "unknown"),
                                    "platform": props.get("$os", "unknown"),
                                }

                                # Add contextual features
                                if "total_ingredients" in props:
                                    interaction["recipe_total_ingredients"] = props[
                                        "total_ingredients"
                                    ]
                                if "owned_ingredients" in props:
                                    interaction["user_owned_ingredients"] = props[
                                        "owned_ingredients"
                                    ]

                                interactions.append(interaction)

                    except (json.JSONDecodeError, Exception) as e:
                        if line_num <= 5:  # Only show first few errors
                            logger.warning(f"Error processing line {line_num}: {e}")
                        continue

        if interactions:
            self.user_interactions = pd.DataFrame(interactions)

            # Convert timestamp to datetime
            self.user_interactions["datetime"] = pd.to_datetime(
                self.user_interactions["timestamp"], unit="s", errors="coerce"
            )

            logger.info(f"✅ Extracted {len(self.user_interactions)} interactions")
            logger.info(f"   Users: {self.user_interactions['user_id'].nunique()}")
            logger.info(f"   Recipes: {self.user_interactions['recipe_id'].nunique()}")
            logger.info(
                f"   Event types: {list(self.user_interactions['event_type'].unique())}"
            )

            return True

        logger.error("❌ No interactions found")
        return False

    def create_user_profiles(self) -> pd.DataFrame:
        """
        Create user profile features from interaction history.

        For beginners: This analyzes each user's behavior patterns to create
        features like average rating, cooking frequency, recipe preferences, etc.

        Returns:
            DataFrame with user profile features
        """
        logger.info("👤 Creating user profiles...")

        if self.user_interactions.empty:
            logger.warning("No user interactions available")
            return pd.DataFrame()

        # Calculate user-level features
        user_profiles = (
            self.user_interactions.groupby("user_id")
            .agg(
                {
                    "rating": ["count", "mean", "sum", "std"],
                    "recipe_id": "nunique",
                    "datetime": ["min", "max"],
                    "device_type": lambda x: (
                        x.mode().iloc[0] if not x.empty else "unknown"
                    ),
                    "platform": lambda x: (
                        x.mode().iloc[0] if not x.empty else "unknown"
                    ),
                }
            )
            .round(3)
        )

        # Flatten column names
        user_profiles.columns = [
            "total_interactions",
            "avg_rating",
            "total_rating",
            "rating_std",
            "unique_recipes",
            "first_interaction",
            "last_interaction",
            "primary_device",
            "primary_platform",
        ]

        # Calculate activity period
        user_profiles["activity_days"] = (
            user_profiles["last_interaction"] - user_profiles["first_interaction"]
        ).dt.days + 1

        user_profiles["interactions_per_day"] = (
            user_profiles["total_interactions"] / user_profiles["activity_days"]
        ).round(3)

        # User engagement level
        user_profiles["engagement_score"] = (
            user_profiles["total_rating"] / user_profiles["activity_days"]
        ).round(3)

        # Fill missing values
        user_profiles["rating_std"] = user_profiles["rating_std"].fillna(0)
        user_profiles["interactions_per_day"] = user_profiles[
            "interactions_per_day"
        ].fillna(0)

        logger.info(f"✅ Created profiles for {len(user_profiles)} users")
        return user_profiles

    def create_user_recipe_pairs(
        self, negative_sampling_ratio: int | None = None
    ) -> pd.DataFrame:
        """
        Create user-recipe pairs with positive and negative samples.

        For beginners: This creates training examples by pairing users with recipes
        they interacted with (positive) and recipes they haven't seen (negative).

        Args:
            negative_sampling_ratio: How many negative samples per positive (uses config default)

        Returns:
            DataFrame with user-recipe pairs and labels
        """
        # Use config default if not specified
        if negative_sampling_ratio is None:
            negative_sampling_ratio = self.config.negative_sampling_ratio

        logger.info("🔗 Creating user-recipe training pairs...")

        # Get all positive interactions (actual user-recipe pairs)
        positive_pairs = (
            self.user_interactions.groupby(["user_id", "recipe_id"])
            .agg(
                {
                    "rating": "max",  # Take highest rating if multiple interactions
                    "event_type": lambda x: x.iloc[-1],  # Most recent event type
                    "datetime": "max",  # Most recent interaction
                    "version": "first",
                }
            )
            .reset_index()
        )

        positive_pairs["label"] = 1
        logger.info(f"✅ Created {len(positive_pairs)} positive pairs")

        # Create negative samples
        logger.info("🔄 Generating negative samples...")
        all_users = positive_pairs["user_id"].unique()
        all_recipes = self.recipe_features["recipe_id"].astype(str).unique()

        # For efficiency, sample negatives per user
        negative_pairs = []

        for user_id in all_users:
            # Get recipes this user has interacted with
            user_recipes = set(
                positive_pairs[positive_pairs["user_id"] == user_id]["recipe_id"]
            )

            # Get recipes they haven't interacted with
            available_recipes = list(set(all_recipes) - user_recipes)

            # Sample negative examples (recipes they haven't interacted with)
            n_positives = len(user_recipes)
            n_negatives = min(
                n_positives * negative_sampling_ratio, len(available_recipes)
            )

            if n_negatives > 0:
                rng = np.random.default_rng()
                negative_recipe_sample = rng.choice(
                    available_recipes, size=n_negatives, replace=False
                )

                for recipe_id in negative_recipe_sample:
                    negative_pairs.append(
                        {
                            "user_id": user_id,
                            "recipe_id": recipe_id,
                            "rating": 0.0,
                            "label": 0,
                        }
                    )

        negative_df = pd.DataFrame(negative_pairs)
        logger.info(f"✅ Created {len(negative_df)} negative pairs")

        # Combine positive and negative samples
        all_pairs = pd.concat(
            [
                positive_pairs[["user_id", "recipe_id", "rating", "label"]],
                negative_df[["user_id", "recipe_id", "rating", "label"]],
            ],
            ignore_index=True,
        )

        logger.info(f"📊 Total training pairs: {len(all_pairs)}")
        logger.info(
            f"   Positive: {len(positive_pairs)} ({len(positive_pairs) / len(all_pairs) * 100:.1f}%)"
        )
        logger.info(
            f"   Negative: {len(negative_df)} ({len(negative_df) / len(all_pairs) * 100:.1f}%)"
        )

        return all_pairs

    def create_training_features(
        self, user_recipe_pairs: pd.DataFrame, user_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive training features combining user, recipe, and interaction data.

        For beginners: This combines user profile data with recipe characteristics
        to create the features that the ML model will learn from.

        Args:
            user_recipe_pairs: DataFrame with user-recipe combinations
            user_profiles: DataFrame with user behavior features

        Returns:
            DataFrame ready for ML training
        """
        logger.info("🔧 Creating comprehensive training features...")

        # Start with user-recipe pairs
        training_data = user_recipe_pairs.copy()

        # Add user profile features
        training_data = training_data.merge(
            user_profiles.reset_index(), on="user_id", how="left"
        )

        # Add recipe features (ensure recipe_id is string for consistency)
        recipe_features = self.recipe_features.copy()
        recipe_features["recipe_id"] = recipe_features["recipe_id"].astype(str)

        training_data = training_data.merge(recipe_features, on="recipe_id", how="left")

        # Add recipe ingredient features
        if not self.recipe_ingredients.empty:
            # Get ingredient diversity per recipe
            ingredient_diversity = (
                self.recipe_ingredients.groupby("recipe_id")
                .agg({"ingredient_id": "count"})
                .rename(columns={"ingredient_id": "ingredient_count_db"})
            )

            ingredient_diversity.index = ingredient_diversity.index.astype(str)
            training_data = training_data.merge(
                ingredient_diversity, left_on="recipe_id", right_index=True, how="left"
            )

        # Create user-recipe interaction features
        training_data = self._add_interaction_features(training_data)

        # Handle missing values
        training_data = self._handle_missing_values(training_data)

        logger.info(
            f"✅ Created training dataset with {len(training_data)} samples and {len(training_data.columns)} features"
        )

        return training_data

    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features that capture user-recipe compatibility."""

        # User's average rating vs recipe complexity
        if "avg_rating" in df.columns and "complexity_score" in df.columns:
            df["user_complexity_match"] = df["avg_rating"] * df["complexity_score"]

        # User engagement vs recipe popularity
        if "engagement_score" in df.columns and "ingredient_count" in df.columns:
            df["user_recipe_engagement_match"] = df["engagement_score"] / (
                df["ingredient_count"] + 1
            )

        # User activity level vs recipe time requirements
        if "interactions_per_day" in df.columns and "total_time" in df.columns:
            df["user_time_compatibility"] = df["interactions_per_day"] / (
                df["total_time"].fillna(30) + 1
            )

        # Device/platform compatibility
        if "primary_device" in df.columns:
            df["is_mobile_user"] = (df["primary_device"] == "phone").astype(int)
        if "primary_platform" in df.columns:
            df["is_ios_user"] = (df["primary_platform"] == "ios").astype(int)

        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the training dataset."""

        # Numeric columns - fill with median or 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ["label", "rating", "recipe_id"]:
                df[col] = df[col].fillna(
                    df[col].median() if not df[col].isna().all() else 0
                )

        # Categorical columns - fill with 'unknown' or mode
        categorical_cols = df.select_dtypes(include=["object"]).columns
        for col in categorical_cols:
            if col not in ["user_id", "recipe_id"]:
                mode_value = (
                    df[col].mode().iloc[0] if not df[col].mode().empty else "unknown"
                )
                df[col] = df[col].fillna(mode_value)

        return df

    def prepare_training_data(
        self,
    ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
        """
        Prepare complete training dataset with train/val/test splits.

        For beginners: This is the main method that orchestrates the entire
        data preparation pipeline from raw events to ML-ready training data.

        Returns:
            Tuple of (train_data, val_data, test_data) or (None, None, None) if failed
        """
        logger.info("🎯 PREPARING TRAINING DATA FOR HYBRID RECOMMENDATION MODEL")
        logger.info("=" * 70)

        # Load all data
        if not self.load_real_recipe_data():
            return None, None, None

        if not self.extract_user_interactions_from_events():
            return None, None, None

        # Create user profiles
        user_profiles = self.create_user_profiles()
        if user_profiles.empty:
            logger.error("❌ Failed to create user profiles")
            return None, None, None

        # Create user-recipe pairs
        user_recipe_pairs = self.create_user_recipe_pairs()
        if user_recipe_pairs.empty:
            logger.error("❌ Failed to create user-recipe pairs")
            return None, None, None

        # Create training features
        training_data = self.create_training_features(user_recipe_pairs, user_profiles)
        if training_data.empty:
            logger.error("❌ Failed to create training features")
            return None, None, None

        # Split data temporally (most recent interactions for testing)
        logger.info("📊 Creating train/validation/test splits...")

        # Sort by timestamp for temporal split
        if "datetime" in training_data.columns:
            training_data = training_data.sort_values("datetime").reset_index(drop=True)

            # Use config settings for split sizes
            n_total = len(training_data)
            n_test = int(self.config.test_size * n_total)
            n_val = int(self.config.validation_size * n_total)

            test_data = training_data.iloc[-n_test:].copy()
            val_data = training_data.iloc[-(n_test + n_val) : -n_test].copy()
            train_data = training_data.iloc[: -(n_test + n_val)].copy()
        else:
            # Random split if no timestamp
            total_test_val = self.config.test_size + self.config.validation_size
            train_data, temp_data = train_test_split(
                training_data,
                test_size=total_test_val,
                random_state=self.config.random_state,
                stratify=training_data["label"],
            )
            # Split temp_data into validation and test
            val_ratio = self.config.validation_size / total_test_val
            val_data, test_data = train_test_split(
                temp_data,
                test_size=(1 - val_ratio),
                random_state=self.config.random_state,
                stratify=temp_data["label"],
            )

        logger.info("✅ Data splits created:")
        logger.info(
            f"   Train: {len(train_data)} samples ({len(train_data) / len(training_data) * 100:.1f}%)"
        )
        logger.info(
            f"   Validation: {len(val_data)} samples ({len(val_data) / len(training_data) * 100:.1f}%)"
        )
        logger.info(
            f"   Test: {len(test_data)} samples ({len(test_data) / len(training_data) * 100:.1f}%)"
        )

        # Save datasets using utilities
        safe_save_csv(train_data, self.config.output_dir / "hybrid_train_data.csv")
        safe_save_csv(val_data, self.config.output_dir / "hybrid_val_data.csv")
        safe_save_csv(test_data, self.config.output_dir / "hybrid_test_data.csv")

        # Save feature columns
        feature_columns = [
            col
            for col in training_data.columns
            if col not in ["user_id", "recipe_id", "label", "rating", "datetime"]
        ]

        feature_file = self.config.output_dir / "hybrid_feature_columns.txt"
        with open(feature_file, "w") as f:
            f.writelines(f"{col}\n" for col in feature_columns)

        # Save metadata using utility
        metadata = {
            "total_samples": len(training_data),
            "n_users": training_data["user_id"].nunique(),
            "n_recipes": training_data["recipe_id"].nunique(),
            "n_features": len(feature_columns),
            "positive_ratio": (training_data["label"] == 1).mean(),
            "feature_columns": feature_columns,
            "created_at": datetime.now().isoformat(),
        }

        save_json_file(
            metadata, self.config.output_dir / "hybrid_training_metadata.json"
        )

        logger.info("\n💾 Saved training data:")
        logger.info(f"   - hybrid_train_data.csv ({len(train_data)} samples)")
        logger.info(f"   - hybrid_val_data.csv ({len(val_data)} samples)")
        logger.info(f"   - hybrid_test_data.csv ({len(test_data)} samples)")
        logger.info(
            f"   - hybrid_feature_columns.txt ({len(feature_columns)} features)"
        )
        logger.info("   - hybrid_training_metadata.json")

        return train_data, val_data, test_data


def main():
    """
    Main function to build hybrid recommendation training data.

    For beginners: This runs the complete data preparation pipeline
    from raw events to ML-ready training datasets.
    """
    logger.info("🚀 HYBRID RECOMMENDATION DATA BUILDER")
    logger.info("=" * 80)

    builder = HybridRecommendationDataBuilder()

    train_data, val_data, test_data = builder.prepare_training_data()

    if train_data is not None:
        logger.info("\n🎉 SUCCESS! Hybrid recommendation training data ready:")
        logger.info("   - Combined user interactions with rich recipe features")
        logger.info(
            f"   - {len(train_data) + len(val_data) + len(test_data)} total training samples"
        )
        logger.info("   - Ready for GBM training to score and rank recipes")

        # Show feature summary
        feature_columns = [
            col
            for col in train_data.columns
            if col not in ["user_id", "recipe_id", "label", "rating", "datetime"]
        ]

        logger.info(f"\n📊 FEATURE SUMMARY ({len(feature_columns)} total):")
        user_features = [
            col
            for col in feature_columns
            if col.startswith(
                (
                    "total_",
                    "avg_",
                    "unique_",
                    "primary_",
                    "engagement_",
                    "activity_",
                    "interactions_",
                )
            )
        ]
        recipe_features = [
            col
            for col in feature_columns
            if col.startswith(
                (
                    "recipe_",
                    "ingredient_",
                    "complexity_",
                    "cook_",
                    "prep_",
                    "total_time",
                    "servings",
                )
            )
        ]
        interaction_features = [
            col for col in feature_columns if "match" in col or "compatibility" in col
        ]

        logger.info(f"   👤 User features: {len(user_features)}")
        logger.info(f"   🍳 Recipe features: {len(recipe_features)}")
        logger.info(f"   🔗 Interaction features: {len(interaction_features)}")
        logger.info(
            f"   📱 Other features: {len(feature_columns) - len(user_features) - len(recipe_features) - len(interaction_features)}"
        )

    else:
        logger.error("❌ Failed to build training data")


if __name__ == "__main__":
    main()
