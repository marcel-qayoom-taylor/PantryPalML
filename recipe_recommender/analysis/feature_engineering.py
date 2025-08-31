"""
Feature Engineering for Recipe Recommendation System

This script creates comprehensive features for the GBM model including:
- User features: cooking patterns, preferences, engagement history
- Recipe features: popularity, complexity, seasonal trends
- User-Recipe features: ingredient overlap, compatibility scores
- Context features: temporal patterns, seasonality
"""

import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RecipeFeatureEngineer:
    """Comprehensive feature engineering for recipe recommendations."""

    def __init__(self):
        self.user_features = {}
        self.recipe_features = {}
        self.tfidf_vectorizer = None
        self.ingredient_similarity_matrix = None

    def load_data(self, data_path):
        """Load interaction data and metadata."""
        print("Loading interaction data and metadata...")

        # Load pickled data
        with open(data_path / "interaction_data.pickle", "rb") as f:
            data = pickle.load(f)

        self.interaction_matrix = data["interaction_matrix"]
        self.user_recipe_interactions = data["user_recipe_interactions"]
        self.recipe_metadata = data["recipe_metadata"]

        # Load additional data
        self.train_interactions = pd.read_csv(data_path / "train_interactions.csv")
        self.test_interactions = pd.read_csv(data_path / "test_interactions.csv")

        print(
            f"Loaded data for {len(self.interaction_matrix)} users and {len(self.recipe_metadata)} recipes"
        )

    def create_user_features(self):
        """Create comprehensive user features."""
        print("Creating user features...")

        user_features = {}

        for user_id in self.interaction_matrix.index:
            # Get user's interaction history
            user_interactions = self.train_interactions[
                self.train_interactions["user_id"] == user_id
            ]

            if len(user_interactions) == 0:
                # Cold start user - use defaults
                user_features[user_id] = self._get_default_user_features()
                continue

            features = {}

            # Basic engagement metrics
            features["total_interactions"] = len(user_interactions)
            features["unique_recipes"] = user_interactions["recipe_id"].nunique()
            features["avg_rating"] = user_interactions["rating"].mean()
            features["max_rating"] = user_interactions["rating"].max()

            # Interaction type preferences
            event_counts = user_interactions["event"].apply(
                lambda x: eval(x) if isinstance(x, str) else x
            )
            all_events = [event for events in event_counts for event in events]
            event_counter = Counter(all_events)

            features["views_count"] = event_counter.get("Recipe Viewed", 0)
            features["cooked_count"] = event_counter.get("Recipe Cooked", 0)
            features["favourited_count"] = event_counter.get("Recipe Favourited", 0)
            features["clicked_count"] = event_counter.get("Recipe Link Clicked", 0)

            # Cooking frequency (recipes cooked / total recipes)
            features["cooking_rate"] = (
                features["cooked_count"] / features["unique_recipes"]
                if features["unique_recipes"] > 0
                else 0
            )
            features["favourite_rate"] = (
                features["favourited_count"] / features["unique_recipes"]
                if features["unique_recipes"] > 0
                else 0
            )

            # Temporal patterns
            timestamps = pd.to_datetime(user_interactions["timestamp"], unit="s")
            features["days_active"] = (timestamps.max() - timestamps.min()).days + 1
            features["interactions_per_day"] = (
                features["total_interactions"] / features["days_active"]
            )

            # Time of day preferences (from original events data would be needed)
            # For now, use simple heuristics
            features["preferred_time"] = self._get_time_preference(user_id)

            # Recipe complexity preferences
            user_recipes = user_interactions["recipe_id"].tolist()
            complexity_scores = []
            ingredient_counts = []

            for recipe_id in user_recipes:
                recipe_meta = self.recipe_metadata.get(recipe_id, {})
                ingredients = recipe_meta.get("recipe_ingredients", [])
                ingredient_counts.append(len(ingredients))

                # Complexity score based on cooking time and ingredient count
                cooking_time = self._parse_cooking_time(
                    recipe_meta.get("recipe_cooking_time")
                )
                complexity = (
                    len(ingredients) * 0.3 + (cooking_time / 60) * 0.7
                    if cooking_time
                    else len(ingredients) * 0.5
                )
                complexity_scores.append(complexity)

            features["avg_recipe_complexity"] = (
                np.mean(complexity_scores) if complexity_scores else 5.0
            )
            features["avg_ingredient_count"] = (
                np.mean(ingredient_counts) if ingredient_counts else 8.0
            )
            features["prefers_simple_recipes"] = (
                1 if features["avg_recipe_complexity"] < 5 else 0
            )

            # Cuisine/tag preferences
            all_tags = []
            all_authors = []

            for recipe_id in user_recipes:
                recipe_meta = self.recipe_metadata.get(recipe_id, {})
                tags = recipe_meta.get("recipe_tags", [])
                all_tags.extend(tags)
                all_authors.append(recipe_meta.get("recipe_author", ""))

            tag_counter = Counter(all_tags)
            author_counter = Counter(all_authors)

            # Top preferences
            features["top_cuisine"] = (
                tag_counter.most_common(1)[0][0] if tag_counter else "main"
            )
            features["cuisine_diversity"] = len(tag_counter)
            features["top_author"] = (
                author_counter.most_common(1)[0][0] if author_counter else "Unknown"
            )
            features["author_loyalty"] = (
                author_counter.most_common(1)[0][1] / len(user_recipes)
                if author_counter
                else 0
            )

            # Ingredient preferences (top ingredients)
            all_ingredients = []
            for recipe_id in user_recipes:
                recipe_meta = self.recipe_metadata.get(recipe_id, {})
                ingredients = recipe_meta.get("recipe_ingredients", [])
                all_ingredients.extend(ingredients)

            ingredient_counter = Counter(all_ingredients)
            features["ingredient_diversity"] = len(ingredient_counter)
            features["top_ingredient"] = (
                ingredient_counter.most_common(1)[0][0]
                if ingredient_counter
                else "unknown"
            )

            # Store top 5 ingredients for similarity calculations
            features["top_ingredients"] = [
                ing for ing, count in ingredient_counter.most_common(5)
            ]

            user_features[user_id] = features

        self.user_features = user_features
        print(f"Created features for {len(user_features)} users")

    def create_recipe_features(self):
        """Create comprehensive recipe features."""
        print("Creating recipe features...")

        recipe_features = {}

        # Calculate popularity metrics
        recipe_popularity = (
            self.train_interactions.groupby("recipe_id")
            .agg(
                {"rating": ["count", "sum", "mean", "std"], "timestamp": ["min", "max"]}
            )
            .round(3)
        )

        recipe_popularity.columns = [
            "interaction_count",
            "total_rating",
            "avg_rating",
            "rating_std",
            "first_interaction",
            "last_interaction",
        ]
        recipe_popularity["rating_std"] = recipe_popularity["rating_std"].fillna(0)

        for recipe_id, recipe_meta in self.recipe_metadata.items():
            features = {}

            # Basic metadata features
            features["recipe_name"] = recipe_meta.get("recipe_name", "Unknown")
            features["recipe_author"] = recipe_meta.get("recipe_author", "Unknown")
            features["version"] = 1 if recipe_meta.get("version") == "v1" else 2

            # Ingredient features
            ingredients = recipe_meta.get("recipe_ingredients", [])
            features["ingredient_count"] = len(ingredients)
            features["has_ingredients"] = 1 if ingredients else 0

            # Complexity features
            cooking_time = self._parse_cooking_time(
                recipe_meta.get("recipe_cooking_time")
            )
            features["cooking_time_minutes"] = (
                cooking_time if cooking_time else 30
            )  # Default 30 min
            features["is_quick"] = 1 if features["cooking_time_minutes"] <= 30 else 0
            features["is_slow_cook"] = (
                1 if features["cooking_time_minutes"] >= 120 else 0
            )

            servings = self._parse_servings(recipe_meta.get("recipe_servings"))
            features["servings"] = servings if servings else 4  # Default 4 servings
            features["is_family_size"] = 1 if features["servings"] >= 6 else 0

            # Recipe complexity score
            features["complexity_score"] = (
                features["ingredient_count"] * 0.3
                + (features["cooking_time_minutes"] / 60) * 0.5
                + (features["servings"] / 4) * 0.2
            )

            # Tag features
            tags = recipe_meta.get("recipe_tags", [])
            features["tag_count"] = len(tags)
            features["is_main"] = 1 if "main" in tags else 0
            features["is_quick_tag"] = 1 if "quick" in tags else 0
            features["is_healthy"] = (
                1 if any(tag in ["healthy", "light", "low-fat"] for tag in tags) else 0
            )
            features["primary_tag"] = tags[0] if tags else "main"

            # Popularity features (from interactions)
            if recipe_id in recipe_popularity.index:
                pop_data = recipe_popularity.loc[recipe_id]
                features["interaction_count"] = pop_data["interaction_count"]
                features["total_rating"] = pop_data["total_rating"]
                features["avg_rating"] = pop_data["avg_rating"]
                features["rating_std"] = pop_data["rating_std"]
                features["days_since_first"] = (
                    datetime.now().timestamp() - pop_data["first_interaction"]
                ) / 86400
                features["days_since_last"] = (
                    datetime.now().timestamp() - pop_data["last_interaction"]
                ) / 86400

                # Popularity scores
                features["popularity_score"] = (
                    np.log1p(features["interaction_count"]) * features["avg_rating"]
                )
                features["is_popular"] = (
                    1
                    if features["interaction_count"]
                    >= recipe_popularity["interaction_count"].median()
                    else 0
                )
                features["is_trending"] = 1 if features["days_since_last"] <= 30 else 0
            else:
                # Cold start recipe
                features.update(
                    {
                        "interaction_count": 0,
                        "total_rating": 0,
                        "avg_rating": 2.5,
                        "rating_std": 0,
                        "days_since_first": 999,
                        "days_since_last": 999,
                        "popularity_score": 0,
                        "is_popular": 0,
                        "is_trending": 0,
                    }
                )

            # Author popularity
            author_recipes = [
                r
                for r, meta in self.recipe_metadata.items()
                if meta.get("recipe_author") == recipe_meta.get("recipe_author")
            ]
            features["author_recipe_count"] = len(author_recipes)
            features["is_popular_author"] = (
                1 if features["author_recipe_count"] >= 5 else 0
            )

            recipe_features[recipe_id] = features

        self.recipe_features = recipe_features
        print(f"Created features for {len(recipe_features)} recipes")

    def create_ingredient_similarity_matrix(self):
        """Create ingredient-based similarity matrix for recipes."""
        print("Creating ingredient similarity matrix...")

        # Prepare ingredient text for each recipe
        recipe_texts = []
        recipe_ids = []

        for recipe_id, meta in self.recipe_metadata.items():
            ingredients = meta.get("recipe_ingredients", [])
            if ingredients:
                # Join ingredients as text
                ingredient_text = " ".join(ingredients)
                recipe_texts.append(ingredient_text)
                recipe_ids.append(recipe_id)

        if not recipe_texts:
            print("No ingredient data found!")
            return

        # Create TF-IDF vectors for ingredients
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,  # Don't remove food-related words
            ngram_range=(1, 2),  # Include bigrams for compound ingredients
            min_df=2,
        )

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(recipe_texts)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)

        # Create DataFrame for easier access
        self.ingredient_similarity_matrix = pd.DataFrame(
            similarity_matrix, index=recipe_ids, columns=recipe_ids
        )

        print(f"Created similarity matrix for {len(recipe_ids)} recipes")

    def create_user_recipe_features(self, user_id, recipe_id):
        """Create user-recipe interaction features."""
        features = {}

        user_feat = self.user_features.get(user_id, self._get_default_user_features())
        recipe_feat = self.recipe_features.get(recipe_id, {})

        # Basic compatibility features
        features["user_recipe_complexity_match"] = abs(
            user_feat.get("avg_recipe_complexity", 5)
            - recipe_feat.get("complexity_score", 5)
        )
        features["user_ingredient_count_match"] = abs(
            user_feat.get("avg_ingredient_count", 8)
            - recipe_feat.get("ingredient_count", 8)
        )

        # Preference matching
        features["matches_time_preference"] = (
            1
            if recipe_feat.get("is_quick", 0)
            and user_feat.get("prefers_simple_recipes", 0)
            else 0
        )
        features["matches_cuisine_preference"] = (
            1 if user_feat.get("top_cuisine") == recipe_feat.get("primary_tag") else 0
        )
        features["matches_author_preference"] = (
            1 if user_feat.get("top_author") == recipe_feat.get("recipe_author") else 0
        )

        # Popularity vs user preferences
        features["popularity_vs_user_avg"] = recipe_feat.get(
            "avg_rating", 2.5
        ) - user_feat.get("avg_rating", 2.5)
        features["complexity_vs_user_pref"] = recipe_feat.get(
            "complexity_score", 5
        ) - user_feat.get("avg_recipe_complexity", 5)

        # Ingredient similarity to user's preferences
        user_ingredients = set(user_feat.get("top_ingredients", []))
        recipe_ingredients = set(
            self.recipe_metadata.get(recipe_id, {}).get("recipe_ingredients", [])
        )

        if user_ingredients and recipe_ingredients:
            intersection = len(user_ingredients.intersection(recipe_ingredients))
            union = len(user_ingredients.union(recipe_ingredients))
            features["ingredient_jaccard"] = intersection / union if union > 0 else 0
            features["ingredient_overlap_count"] = intersection
            features["has_familiar_ingredients"] = 1 if intersection > 0 else 0
        else:
            features["ingredient_jaccard"] = 0
            features["ingredient_overlap_count"] = 0
            features["has_familiar_ingredients"] = 0

        # Recipe similarity to user's history (using ingredient similarity)
        if (
            hasattr(self, "ingredient_similarity_matrix")
            and recipe_id in self.ingredient_similarity_matrix.index
        ):
            user_recipes = self.train_interactions[
                self.train_interactions["user_id"] == user_id
            ]["recipe_id"].tolist()

            similarities = []
            for user_recipe in user_recipes[-10:]:  # Last 10 recipes
                if user_recipe in self.ingredient_similarity_matrix.index:
                    sim = self.ingredient_similarity_matrix.loc[recipe_id, user_recipe]
                    similarities.append(sim)

            features["avg_similarity_to_history"] = (
                np.mean(similarities) if similarities else 0
            )
            features["max_similarity_to_history"] = (
                np.max(similarities) if similarities else 0
            )
        else:
            features["avg_similarity_to_history"] = 0
            features["max_similarity_to_history"] = 0

        # Temporal features
        features["recipe_age"] = (
            recipe_feat.get("days_since_first", 999) / 365
        )  # In years
        features["recipe_freshness"] = 1 / (
            1 + recipe_feat.get("days_since_last", 999) / 30
        )  # Fresher = higher score

        return features

    def create_context_features(self, timestamp=None):
        """Create contextual features for recommendation time."""
        features = {}

        dt = datetime.fromtimestamp(timestamp) if timestamp else datetime.now()

        # Time-based features
        features["hour"] = dt.hour
        features["day_of_week"] = dt.weekday()
        features["month"] = dt.month
        features["is_weekend"] = 1 if dt.weekday() >= 5 else 0
        features["is_breakfast_time"] = 1 if 6 <= dt.hour <= 10 else 0
        features["is_lunch_time"] = 1 if 11 <= dt.hour <= 14 else 0
        features["is_dinner_time"] = 1 if 17 <= dt.hour <= 21 else 0

        # Seasonal features
        if dt.month in [12, 1, 2]:
            features["season"] = "winter"
        elif dt.month in [3, 4, 5]:
            features["season"] = "spring"
        elif dt.month in [6, 7, 8]:
            features["season"] = "summer"
        else:
            features["season"] = "autumn"

        # One-hot encode season
        features["is_winter"] = 1 if features["season"] == "winter" else 0
        features["is_spring"] = 1 if features["season"] == "spring" else 0
        features["is_summer"] = 1 if features["season"] == "summer" else 0
        features["is_autumn"] = 1 if features["season"] == "autumn" else 0

        return features

    def _get_default_user_features(self):
        """Default features for cold start users."""
        return {
            "total_interactions": 0,
            "unique_recipes": 0,
            "avg_rating": 2.5,
            "max_rating": 2.5,
            "views_count": 0,
            "cooked_count": 0,
            "favourited_count": 0,
            "clicked_count": 0,
            "cooking_rate": 0,
            "favourite_rate": 0,
            "days_active": 1,
            "interactions_per_day": 0,
            "preferred_time": "evening",
            "avg_recipe_complexity": 5.0,
            "avg_ingredient_count": 8.0,
            "prefers_simple_recipes": 0,
            "top_cuisine": "main",
            "cuisine_diversity": 1,
            "top_author": "Unknown",
            "author_loyalty": 0,
            "ingredient_diversity": 0,
            "top_ingredient": "unknown",
            "top_ingredients": [],
        }

    def _get_time_preference(self, user_id):
        """Simple heuristic for time preference."""
        # In a real system, this would analyze user's interaction times
        # For now, use a simple hash-based assignment
        return ["morning", "afternoon", "evening"][hash(str(user_id)) % 3]

    def _parse_cooking_time(self, time_str):
        """Parse cooking time string to minutes."""
        if not time_str:
            return None

        try:
            # Handle different formats: "30", "30m", "1h 30m", etc.
            time_str = str(time_str).lower().strip()

            if "h" in time_str and "m" in time_str:
                # Format: "1h 30m"
                parts = time_str.split("h")
                hours = int(parts[0].strip())
                minutes = (
                    int(parts[1].replace("m", "").strip()) if parts[1].strip() else 0
                )
                return hours * 60 + minutes
            if "h" in time_str:
                # Format: "1h"
                hours = float(time_str.replace("h", "").strip())
                return int(hours * 60)
            if "m" in time_str:
                # Format: "30m"
                return int(time_str.replace("m", "").strip())
            # Assume it's already in minutes
            return int(float(time_str))
        except (ValueError, TypeError, IndexError):
            return None

    def _parse_servings(self, servings_str):
        """Parse servings string to number."""
        if not servings_str:
            return None

        try:
            return int(float(str(servings_str).strip()))
        except (ValueError, TypeError):
            return None

    def save_features(self, output_dir):
        """Save all engineered features."""
        output_dir = Path(output_dir)

        # Save user features
        user_features_df = pd.DataFrame.from_dict(self.user_features, orient="index")
        user_features_df.to_csv(output_dir / "user_features.csv")
        print(f"💾 Saved user features to {output_dir / 'user_features.csv'}")

        # Save recipe features
        recipe_features_df = pd.DataFrame.from_dict(
            self.recipe_features, orient="index"
        )
        recipe_features_df.to_csv(output_dir / "recipe_features.csv")
        print(f"💾 Saved recipe features to {output_dir / 'recipe_features.csv'}")

        # Save similarity matrix
        if self.ingredient_similarity_matrix is not None:
            self.ingredient_similarity_matrix.to_csv(
                output_dir / "ingredient_similarity_matrix.csv"
            )
            print(
                f"💾 Saved ingredient similarity matrix to {output_dir / 'ingredient_similarity_matrix.csv'}"
            )

        # Save feature engineering pipeline
        with open(output_dir / "feature_engineer.pickle", "wb") as f:
            pickle.dump(self, f)
        print(
            f"💾 Saved feature engineering pipeline to {output_dir / 'feature_engineer.pickle'}"
        )


def main():
    """Main feature engineering pipeline."""
    print("🔧 Starting Feature Engineering Pipeline...")

    # Paths
    script_dir = Path(__file__).parent
    data_dir = script_dir.parent / "output"

    # Initialize feature engineer
    engineer = RecipeFeatureEngineer()

    # Load data
    engineer.load_data(data_dir)

    # Create features
    engineer.create_user_features()
    engineer.create_recipe_features()
    engineer.create_ingredient_similarity_matrix()

    # Save features
    engineer.save_features(data_dir)

    # Display feature summary
    print("\n" + "=" * 60)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print("=" * 60)
    print(f"👥 User features: {len(engineer.user_features)} users")
    print(f"🍳 Recipe features: {len(engineer.recipe_features)} recipes")
    print(
        f"🔗 Ingredient similarity matrix: {engineer.ingredient_similarity_matrix.shape if engineer.ingredient_similarity_matrix is not None else 'None'}"
    )
    print("\nGenerated files:")
    print("- user_features.csv: User behavioral and preference features")
    print("- recipe_features.csv: Recipe metadata and popularity features")
    print("- ingredient_similarity_matrix.csv: Recipe-recipe similarity matrix")
    print("- feature_engineer.pickle: Complete feature engineering pipeline")


if __name__ == "__main__":
    main()
