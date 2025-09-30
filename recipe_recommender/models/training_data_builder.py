#!/usr/bin/env python3
"""
Training Data Builder

This script combines user interaction history from events with the rich recipe
database to create training data for a recommendation model that can
score and rank recipes based on user preferences.
"""

import json
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
import joblib

from recipe_recommender.config import get_feature_columns_to_exclude, get_ml_config
from recipe_recommender.utils import (
    safe_load_csv,
    safe_save_csv,
    save_json_file,
    setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging(__name__)


class TrainingDataBuilder:
    """
    Build training data combining user interactions with rich recipe features.

    This class takes raw user interaction data and recipe information
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

        logger.info("Initialized Training Data Builder")

    # ------------------------------
    # Text encoding helpers
    # ------------------------------
    @staticmethod
    def _parse_tags(value) -> list[str]:
        """Parse tags that may be stored as JSON, comma-string, or list."""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        text = str(value).strip()
        if not text:
            return []
        # Try JSON list
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
        except Exception:
            pass
        # Fallback: split by comma or pipe or space
        parts = re.split(r"[,|]", text)
        if len(parts) == 1:
            # maybe space separated
            parts = text.split()
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _sanitize_tag(name: str) -> str:
        s = str(name).lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_")
        return s[:40]

    def _fit_text_encoders(self, train_data: pd.DataFrame) -> dict:
        """Fit text encoders on training split only and return encoder objects.

        Returns a dict with encoder objects, mappings, and generated column schemas.
        """
        cfg = self.config.text_encoding
        if not cfg or not cfg.enable_text_features:
            return {}

        logger.info("Fitting text encoders on training split")

        # Use only recipes present in training split to fit vectorizers
        train_recipe_ids = set(train_data["recipe_id"].astype(str).unique())
        recipes_train = self.recipe_features.copy()
        recipes_train["recipe_id"] = recipes_train["recipe_id"].astype(str)
        recipes_train = recipes_train[recipes_train["recipe_id"].isin(train_recipe_ids)]

        encoders: dict = {
            "config": {
                "author_id_encoding": cfg.author_id_encoding,
                "tags_encoding": cfg.tags_encoding,
                "name_encoding": cfg.name_encoding,
                "desc_encoding": cfg.desc_encoding,
                "instr_encoding": cfg.instr_encoding,
                "tags_top_k": cfg.tags_top_k,
                "name_hash_dim": cfg.name_hash_dim,
                "desc_max_features": cfg.desc_max_features,
                "instr_hash_dim": cfg.instr_hash_dim,
                "hashing_alternate_sign": cfg.hashing_alternate_sign,
                "ngram_range": cfg.ngram_range,
            }
        }

        # Author encoders
        if "author_id" in recipes_train.columns:
            author_series = recipes_train["author_id"].fillna("unknown").astype(str)
            if cfg.author_id_encoding == "freq":
                counts = author_series.value_counts()
                freq = (counts / counts.sum()).astype(float)
                encoders["author_freq_mapping"] = freq.to_dict()
            elif cfg.author_id_encoding == "target":
                # Average positive rate per author from training pairs (Laplace smoothing)
                tmp = train_data.copy()
                if "author_id" not in tmp.columns:
                    # Merge from recipe features
                    tmp = tmp.merge(
                        recipes_train[["recipe_id", "author_id"]],
                        on="recipe_id",
                        how="left",
                    )
                tmp["author_id"] = tmp["author_id"].fillna("unknown").astype(str)
                grp = (
                    tmp.groupby("author_id")["label"]
                    .agg(["sum", "count"])
                    .reset_index()
                )
                # Laplace smoothing: (sum + 1) / (count + 2)
                grp["te"] = (grp["sum"] + 1.0) / (grp["count"] + 2.0)
                encoders["author_target_mapping"] = dict(
                    zip(grp["author_id"], grp["te"])
                )

        # Tags encoder
        if "tags" in recipes_train.columns:
            if cfg.tags_encoding == "topk_multi_hot":
                tag_counts: dict[str, int] = {}
                for v in recipes_train["tags"].tolist():
                    for t in self._parse_tags(v):
                        tag_counts[t] = tag_counts.get(t, 0) + 1
                topk = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[
                    : cfg.tags_top_k
                ]
                vocab = [self._sanitize_tag(k) for k, _ in topk]
                encoders["tags_topk_vocab"] = vocab
            elif cfg.tags_encoding == "hashing":
                encoders["tags_hashing"] = HashingVectorizer(
                    n_features=cfg.name_hash_dim,  # reuse small dim for tags
                    alternate_sign=cfg.hashing_alternate_sign,
                    analyzer="word",
                    ngram_range=cfg.ngram_range,
                    norm=None,
                    binary=True,
                )

        # Recipe name encoder
        if cfg.name_encoding == "hashing" and "recipe_name" in recipes_train.columns:
            encoders["name_hashing"] = HashingVectorizer(
                n_features=cfg.name_hash_dim,
                alternate_sign=cfg.hashing_alternate_sign,
                analyzer="word",
                ngram_range=cfg.ngram_range,
                norm=None,
            )

        # Description encoder
        if "description" in recipes_train.columns:
            if cfg.desc_encoding == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=cfg.desc_max_features,
                    ngram_range=cfg.ngram_range,
                )
                corpus = recipes_train["description"].fillna("").astype(str).tolist()
                vectorizer.fit(corpus)
                encoders["desc_tfidf"] = vectorizer
            elif cfg.desc_encoding == "hashing":
                encoders["desc_hashing"] = HashingVectorizer(
                    n_features=cfg.desc_max_features,
                    alternate_sign=cfg.hashing_alternate_sign,
                    analyzer="word",
                    ngram_range=cfg.ngram_range,
                    norm=None,
                )

        # Instruction encoder
        if "instruction" in recipes_train.columns:
            if cfg.instr_encoding == "tfidf":
                vectorizer = TfidfVectorizer(
                    max_features=cfg.desc_max_features,
                    ngram_range=cfg.ngram_range,
                )
                corpus = recipes_train["instruction"].fillna("").astype(str).tolist()
                vectorizer.fit(corpus)
                encoders["instr_tfidf"] = vectorizer
            elif cfg.instr_encoding == "hashing":
                encoders["instr_hashing"] = HashingVectorizer(
                    n_features=cfg.instr_hash_dim,
                    alternate_sign=cfg.hashing_alternate_sign,
                    analyzer="word",
                    ngram_range=cfg.ngram_range,
                    norm=None,
                )

        return encoders

    def _transform_recipe_text_features(
        self, df: pd.DataFrame, encoders: dict
    ) -> pd.DataFrame:
        """Transform recipe-level text fields into numeric feature columns.

        Returns a DataFrame with `recipe_id` and new numeric columns (float32/int32).
        """
        if not encoders:
            return df[["recipe_id"]].copy()

        cfg = self.config.text_encoding
        df2 = df.copy()
        df2["recipe_id"] = df2["recipe_id"].astype(str)

        new_cols: dict[str, np.ndarray] = {}

        # Author encodings
        if "author_id" in df2.columns:
            author_vals = df2["author_id"].fillna("unknown").astype(str)
            if "author_freq_mapping" in encoders:
                mapping = encoders["author_freq_mapping"]
                new_cols["author_id_freq"] = (
                    author_vals.map(mapping).fillna(0.0).astype(np.float32).to_numpy()
                )
            if "author_target_mapping" in encoders:
                mapping = encoders["author_target_mapping"]
                new_cols["author_id_target"] = (
                    author_vals.map(mapping).fillna(0.5).astype(np.float32).to_numpy()
                )

        # Tags
        if (
            "tags" in df2.columns
            and cfg.tags_encoding == "topk_multi_hot"
            and "tags_topk_vocab" in encoders
        ):
            vocab = encoders["tags_topk_vocab"]
            # Create a multi-hot matrix
            mh = np.zeros((len(df2), len(vocab)), dtype=np.float32)
            vocab_index = {t: i for i, t in enumerate(vocab)}
            for row_idx, v in enumerate(df2["tags"].tolist()):
                for t in [self._sanitize_tag(x) for x in self._parse_tags(v)]:
                    if t in vocab_index:
                        mh[row_idx, vocab_index[t]] = 1.0
            for i, t in enumerate(vocab):
                new_cols[f"tag__{t}"] = mh[:, i]
        elif (
            "tags" in df2.columns
            and cfg.tags_encoding == "hashing"
            and "tags_hashing" in encoders
        ):
            vec = encoders["tags_hashing"]
            corpus = [" ".join(self._parse_tags(v)) for v in df2["tags"].tolist()]
            X = vec.transform(corpus)
            # Ensure dense array
            X = X.toarray().astype(np.float32)
            for i in range(X.shape[1]):
                new_cols[f"tags_hash_{i}"] = X[:, i]

        # Recipe name
        if (
            cfg.name_encoding == "hashing"
            and "name_hashing" in encoders
            and "recipe_name" in df2.columns
        ):
            vec = encoders["name_hashing"]
            corpus = df2["recipe_name"].fillna("").astype(str).tolist()
            X = vec.transform(corpus).toarray().astype(np.float32)
            for i in range(X.shape[1]):
                new_cols[f"name_hash_{i}"] = X[:, i]

        # Description
        if (
            cfg.desc_encoding == "tfidf"
            and "desc_tfidf" in encoders
            and "description" in df2.columns
        ):
            vec = encoders["desc_tfidf"]
            X = (
                vec.transform(df2["description"].fillna("").astype(str).tolist())
                .toarray()
                .astype(np.float32)
            )
            # Use feature names for readability
            try:
                names = vec.get_feature_names_out()
            except Exception:
                names = [f"desc_tfidf_{i}" for i in range(X.shape[1])]
            for i in range(X.shape[1]):
                col = f"desc_tfidf_{names[i]}" if i < len(names) else f"desc_tfidf_{i}"
                # Sanitize column name
                col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)[:60]
                new_cols[col] = X[:, i]
        elif (
            cfg.desc_encoding == "hashing"
            and "desc_hashing" in encoders
            and "description" in df2.columns
        ):
            vec = encoders["desc_hashing"]
            X = (
                vec.transform(df2["description"].fillna("").astype(str).tolist())
                .toarray()
                .astype(np.float32)
            )
            for i in range(X.shape[1]):
                new_cols[f"desc_hash_{i}"] = X[:, i]

        # Instruction
        if (
            cfg.instr_encoding == "tfidf"
            and "instr_tfidf" in encoders
            and "instruction" in df2.columns
        ):
            vec = encoders["instr_tfidf"]
            X = (
                vec.transform(df2["instruction"].fillna("").astype(str).tolist())
                .toarray()
                .astype(np.float32)
            )
            try:
                names = vec.get_feature_names_out()
            except Exception:
                names = [f"instr_tfidf_{i}" for i in range(X.shape[1])]
            for i in range(X.shape[1]):
                col = (
                    f"instr_tfidf_{names[i]}" if i < len(names) else f"instr_tfidf_{i}"
                )
                col = re.sub(r"[^a-zA-Z0-9_]+", "_", col)[:60]
                new_cols[col] = X[:, i]
        elif (
            cfg.instr_encoding == "hashing"
            and "instr_hashing" in encoders
            and "instruction" in df2.columns
        ):
            vec = encoders["instr_hashing"]
            X = (
                vec.transform(df2["instruction"].fillna("").astype(str).tolist())
                .toarray()
                .astype(np.float32)
            )
            for i in range(X.shape[1]):
                new_cols[f"instr_hash_{i}"] = X[:, i]

        # Build result
        result = pd.DataFrame({"recipe_id": df2["recipe_id"].astype(str)})
        for col, arr in new_cols.items():
            result[col] = arr
        return result

    def load_real_recipe_data(self) -> bool:
        """
        Load the extracted real recipe database.

        This loads the recipe data that was fetched from Supabase,
        including recipe details, ingredients, and relationships between them.

        Returns:
            bool: True if successful
        """
        logger.info("Loading real recipe database")

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
                f"Loaded {len(self.recipe_features)} recipes with enhanced features"
            )

            # Load recipe-ingredient relationships
            ingredients_file = (
                self.config.output_dir / "real_recipe_ingredients_from_db.csv"
            )
            self.recipe_ingredients = safe_load_csv(ingredients_file)
            if self.recipe_ingredients is not None:
                logger.info(
                    f"Loaded {len(self.recipe_ingredients)} recipe-ingredient relationships"
                )

            # Load ingredient data
            ingredient_master_file = (
                self.config.output_dir / "real_ingredients_from_db.csv"
            )
            self.ingredients = safe_load_csv(ingredient_master_file)
            if self.ingredients is not None:
                logger.info(f"Loaded {len(self.ingredients)} ingredients")

            return True

        except Exception as e:
            logger.exception("Error loading real recipe data")
            return False

    def extract_user_interactions_from_events(self) -> bool:
        """
        Extract user interactions from event files.

        This reads the JSON event files containing user interactions
        (recipe views, cooks, favorites) and converts them into structured data.

        Returns:
            bool: True if successful
        """
        logger.info("Extracting user interactions from events")

        interactions = []

        # Process both v1 and v2 events
        for version, filename in [
            ("v1", "v1_events_20250827.json"),
            ("v2", "v2_events_20250920.json"),
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

            logger.info(f"Extracted {len(self.user_interactions)} interactions")
            logger.info(f"   Users: {self.user_interactions['user_id'].nunique()}")
            logger.info(f"   Recipes: {self.user_interactions['recipe_id'].nunique()}")
            logger.info(
                f"   Event types: {list(self.user_interactions['event_type'].unique())}"
            )

            return True

        logger.error("No interactions found")
        return False

    def map_recipe_ids_to_latest(self) -> bool:
        """
        Map v1 recipe IDs to their corresponding v2 recipe IDs using the mapping file.

        This ensures consistent recipe identifiers across v1 and v2 datasets,
        using v2 recipe IDs as the canonical identifiers.

        Returns:
            bool: True if successful, False if mapping file not found
        """
        if self.user_interactions.empty:
            logger.warning("No user interactions to map")
            return True

        # Load the recipe ID mapping
        mapping_file = self.config.output_dir / "recipe_id_mapping_v1_v2.json"
        if not mapping_file.exists():
            logger.warning(
                f"Recipe ID mapping file not found at {mapping_file}. Skipping recipe ID mapping."
            )
            logger.warning(
                "Run analyze_recipe_duplicates.py first to generate the mapping."
            )
            return True

        logger.info("Loading recipe ID mapping to standardize v1/v2 recipe IDs")

        try:
            with open(mapping_file) as f:
                mapping_data = json.load(f)

            # Create a dictionary mapping v1 recipe IDs to v2 recipe IDs
            v1_to_v2_mapping = {
                item["v1_recipe_id"]: item["v2_recipe_id"] for item in mapping_data
            }

            logger.info(f"Loaded mapping for {len(v1_to_v2_mapping)} recipe ID pairs")

            # Count interactions before mapping
            v1_interactions_before = len(
                self.user_interactions[self.user_interactions["version"] == "v1"]
            )

            # Apply mapping to v1 interactions
            v1_mask = self.user_interactions["version"] == "v1"
            v1_interactions = self.user_interactions[v1_mask].copy()

            # Map v1 recipe IDs to v2 recipe IDs where mapping exists
            mapped_recipe_ids = v1_interactions["recipe_id"].map(v1_to_v2_mapping)
            mapped_count = mapped_recipe_ids.notna().sum()

            # Update recipe IDs for mapped recipes
            self.user_interactions.loc[v1_mask, "recipe_id"] = mapped_recipe_ids.fillna(
                v1_interactions["recipe_id"]
            )

            # Update the version for successfully mapped interactions
            mapped_mask = v1_mask & mapped_recipe_ids.notna()
            self.user_interactions.loc[mapped_mask, "version"] = "v1_mapped_to_v2"

            logger.info(f"Recipe ID mapping results:")
            logger.info(f"   v1 interactions processed: {v1_interactions_before}")
            logger.info(f"   Successfully mapped to v2 IDs: {mapped_count}")
            logger.info(
                f"   Unmapped v1 interactions: {v1_interactions_before - mapped_count}"
            )

            # Log final recipe distribution
            unique_recipes_after = self.user_interactions["recipe_id"].nunique()
            logger.info(
                f"   Total unique recipe IDs after mapping: {unique_recipes_after}"
            )

            version_counts = self.user_interactions["version"].value_counts()
            for version, count in version_counts.items():
                logger.info(f"   {version} interactions: {count}")

            return True

        except Exception as e:
            logger.error(f"Error loading or applying recipe ID mapping: {e}")
            logger.warning("Continuing without recipe ID mapping")
            return False

    def create_user_profiles(self) -> pd.DataFrame:
        """
        Create user profile features from interaction history.

        This analyzes each user's behavior patterns to create
        features like average rating, cooking frequency, recipe preferences, etc.

        Returns:
            DataFrame with user profile features
        """
        logger.info("Creating user profiles")

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

        logger.info(f"Created profiles for {len(user_profiles)} users")
        return user_profiles

    def create_user_recipe_pairs(
        self, negative_sampling_ratio: int | None = None
    ) -> pd.DataFrame:
        """
        Create user-recipe pairs with engagement-weighted relevance scores.

        This creates training examples by pairing users with recipes they interacted with,
        using different relevance scores based on engagement type (view=1, cook=5, etc.)
        and negative samples for recipes they haven't seen.

        Args:
            negative_sampling_ratio: How many negative samples per positive (uses config default)

        Returns:
            DataFrame with user-recipe pairs and engagement-weighted labels
        """
        # Use config default if not specified
        if negative_sampling_ratio is None:
            negative_sampling_ratio = self.config.negative_sampling_ratio

        logger.info("Creating user-recipe training pairs with engagement weighting")

        # Use engagement weights from config for consistency
        engagement_weights = self.config.interaction_weights

        # Get all positive interactions (actual user-recipe pairs)
        # For users with multiple interactions on same recipe, take the highest engagement
        positive_pairs = (
            self.user_interactions.groupby(["user_id", "recipe_id"])
            .agg(
                {
                    "rating": "max",  # Take highest rating if multiple interactions
                    "event_type": lambda x: self._get_highest_engagement_event(
                        x.tolist(), engagement_weights
                    ),
                    "datetime": "max",  # Most recent interaction
                    "version": "first",
                }
            )
            .reset_index()
        )

        # Apply engagement-based labeling
        positive_pairs["label"] = positive_pairs["event_type"].map(
            lambda event: engagement_weights.get(event, 1.0)
        )

        # Handle negative engagement signals (e.g., "Recipe Removed From Collections")
        # Convert negative weights to very low positive weights to maintain ranking framework
        positive_pairs.loc[positive_pairs["label"] < 0, "label"] = 0.1

        logger.info(f"Created {len(positive_pairs)} engagement-weighted pairs")
        logger.info(
            f"   Engagement distribution: {positive_pairs['event_type'].value_counts().to_dict()}"
        )
        logger.info(
            f"   Label distribution: {positive_pairs['label'].value_counts().to_dict()}"
        )

        # Create negative samples
        logger.info("Generating negative samples")
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
        logger.info(f"Created {len(negative_df)} negative pairs")

        # Combine positive and negative samples
        all_pairs = pd.concat(
            [
                positive_pairs[["user_id", "recipe_id", "rating", "label"]],
                negative_df[["user_id", "recipe_id", "rating", "label"]],
            ],
            ignore_index=True,
        )

        logger.info(f"Total training pairs: {len(all_pairs)}")
        logger.info(
            f"   Positive: {len(positive_pairs)} ({len(positive_pairs) / len(all_pairs) * 100:.1f}%)"
        )
        logger.info(
            f"   Negative: {len(negative_df)} ({len(negative_df) / len(all_pairs) * 100:.1f}%)"
        )

        return all_pairs

    def _get_highest_engagement_event(
        self, event_list: list[str], engagement_weights: dict[str, float]
    ) -> str:
        """
        From a list of events for the same user-recipe pair, return the highest engagement event.

        Args:
            event_list: List of event types for this user-recipe pair
            engagement_weights: Mapping of event types to engagement scores

        Returns:
            Event type with highest engagement weight
        """
        if not event_list:
            return "Recipe Viewed"  # Default fallback

        # Find event with highest weight (most positive engagement)
        highest_event = event_list[0]
        highest_weight = engagement_weights.get(highest_event, 1.0)

        for event in event_list:
            weight = engagement_weights.get(event, 1.0)
            if weight > highest_weight:
                highest_weight = weight
                highest_event = event

        return highest_event

    def create_training_features(
        self, user_recipe_pairs: pd.DataFrame, user_profiles: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create comprehensive training features combining user, recipe, and interaction data.

        This combines user profile data with recipe characteristics
        to create the features that the ML model will learn from.

        Args:
            user_recipe_pairs: DataFrame with user-recipe combinations
            user_profiles: DataFrame with user behavior features

        Returns:
            DataFrame ready for ML training
        """
        logger.info("Creating comprehensive training features")

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
            f"Created training dataset with {len(training_data)} samples and {len(training_data.columns)} features"
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

        This is the main method that orchestrates the entire
        data preparation pipeline from raw events to ML-ready training data.

        Returns:
            Tuple of (train_data, val_data, test_data) or (None, None, None) if failed
        """
        logger.info("Preparing training data for recommendation model")

        # Load all data
        if not self.load_real_recipe_data():
            return None, None, None

        if not self.extract_user_interactions_from_events():
            return None, None, None

        # Map v1 recipe IDs to v2 recipe IDs for consistency
        if not self.map_recipe_ids_to_latest():
            logger.warning("Recipe ID mapping failed, continuing with original IDs")

        # Create user profiles
        user_profiles = self.create_user_profiles()
        if user_profiles.empty:
            logger.error("Failed to create user profiles")
            return None, None, None

        # Create user-recipe pairs
        user_recipe_pairs = self.create_user_recipe_pairs()
        if user_recipe_pairs.empty:
            logger.error("Failed to create user-recipe pairs")
            return None, None, None

        # Drop users with no interactions (safety) or no positives
        # Keep only users present in interactions and with at least 1 positive label
        logger.info(
            "Filtering users with insufficient interactions/positives for ranking"
        )
        users_with_interactions = set(self.user_interactions["user_id"].unique())
        user_pos_counts = (
            user_recipe_pairs[user_recipe_pairs["label"] == 1].groupby("user_id").size()
        )
        users_with_positive = set(user_pos_counts.index.tolist())

        eligible_users = users_with_interactions.intersection(users_with_positive)
        before_count = len(user_recipe_pairs)
        user_recipe_pairs = user_recipe_pairs[
            user_recipe_pairs["user_id"].isin(eligible_users)
        ].reset_index(drop=True)
        after_count = len(user_recipe_pairs)
        logger.info(
            f"   Kept {len(eligible_users)} users eligible for ranking; pairs: {before_count} -> {after_count}"
        )

        # Create training features
        training_data = self.create_training_features(user_recipe_pairs, user_profiles)
        if training_data.empty:
            logger.error("Failed to create training features")
            return None, None, None

        # Split data temporally (most recent interactions for testing)
        logger.info("Creating train/validation/test splits")

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

        logger.info("Data splits created:")
        logger.info(
            f"   Train: {len(train_data)} samples ({len(train_data) / len(training_data) * 100:.1f}%)"
        )
        logger.info(
            f"   Validation: {len(val_data)} samples ({len(val_data) / len(training_data) * 100:.1f}%)"
        )
        logger.info(
            f"   Test: {len(test_data)} samples ({len(test_data) / len(training_data) * 100:.1f}%)"
        )

        # Optionally fit and apply text encoders (train-only fit)
        text_feature_columns: list[str] = []
        text_encoder_artifacts: dict = {}
        if self.config.text_encoding and self.config.text_encoding.enable_text_features:
            encoders = self._fit_text_encoders(train_data)

            # Transform full recipe catalog to encoded features
            encoded_recipe_features = self._transform_recipe_text_features(
                self.recipe_features, encoders
            )

            # Merge encoded features into each split
            for split_name, df_ref in (
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
                ("all", training_data),
            ):
                df_ref.merge(
                    encoded_recipe_features, on="recipe_id", how="left", copy=False
                )
            # The above merge with copy=False modifies in place only in pandas >= 2.0; ensure assignment
            train_data = train_data.merge(
                encoded_recipe_features, on="recipe_id", how="left"
            )
            val_data = val_data.merge(
                encoded_recipe_features, on="recipe_id", how="left"
            )
            test_data = test_data.merge(
                encoded_recipe_features, on="recipe_id", how="left"
            )
            training_data = training_data.merge(
                encoded_recipe_features, on="recipe_id", how="left"
            )

            # Save encoded recipe features CSV for inference
            encoded_catalog = self.recipe_features.copy()
            encoded_catalog = encoded_catalog.merge(
                encoded_recipe_features, on="recipe_id", how="left"
            )
            safe_save_csv(
                encoded_catalog,
                self.config.output_dir / self.config.encoded_recipe_features_filename,
            )

            # Persist vectorizers (non-hashing) and mappings
            vec_dir = self.config.model_dir / "text_vectorizers"
            vec_dir.mkdir(parents=True, exist_ok=True)
            text_encoder_artifacts["vectorizers_dir"] = str(vec_dir)
            artifact_paths: dict[str, str] = {}

            if "desc_tfidf" in encoders:
                path = vec_dir / "desc_tfidf.joblib"
                joblib.dump(encoders["desc_tfidf"], path)
                artifact_paths["desc_tfidf"] = str(path)
            if "instr_tfidf" in encoders:
                path = vec_dir / "instr_tfidf.joblib"
                joblib.dump(encoders["instr_tfidf"], path)
                artifact_paths["instr_tfidf"] = str(path)
            if "author_target_mapping" in encoders:
                path = vec_dir / "author_target_mapping.json"
                save_json_file(encoders["author_target_mapping"], path)
                artifact_paths["author_target_mapping"] = str(path)
            if "author_freq_mapping" in encoders:
                path = vec_dir / "author_freq_mapping.json"
                save_json_file(encoders["author_freq_mapping"], path)
                artifact_paths["author_freq_mapping"] = str(path)

            text_encoder_artifacts["artifacts"] = artifact_paths
            text_encoder_artifacts["config"] = encoders.get("config", {})

            # Collect newly added column names
            text_feature_columns = [
                c
                for c in training_data.columns
                if c.startswith("author_id_")
                or c.startswith("tag__")
                or c.startswith("name_hash_")
                or c.startswith("desc_tfidf_")
                or c.startswith("desc_hash_")
                or c.startswith("instr_tfidf_")
                or c.startswith("instr_hash_")
            ]

        # Save datasets using utilities
        safe_save_csv(train_data, self.config.output_dir / "hybrid_train_data.csv")
        safe_save_csv(val_data, self.config.output_dir / "hybrid_val_data.csv")
        safe_save_csv(test_data, self.config.output_dir / "hybrid_test_data.csv")

        # Save feature columns (exclude IDs, labels, timestamps, and raw excluded text columns)
        excluded_cols = set(
            ["user_id", "recipe_id", "label", "rating", "datetime"]
        ) | set(get_feature_columns_to_exclude())
        feature_columns = [
            col for col in training_data.columns if col not in excluded_cols
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

        # Extend metadata with text encoding details
        if self.config.text_encoding and self.config.text_encoding.enable_text_features:
            metadata.update(
                {
                    "text_features_enabled": True,
                    "text_feature_columns": text_feature_columns,
                    "text_encoders": text_encoder_artifacts,
                    "encoded_recipe_features_file": str(
                        self.config.output_dir
                        / self.config.encoded_recipe_features_filename
                    ),
                }
            )

        save_json_file(
            metadata, self.config.output_dir / "hybrid_training_metadata.json"
        )

        logger.info("Saved training data:")
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
    Main function to build recommendation training data.

    This runs the complete data preparation pipeline
    from raw events to ML-ready training datasets.
    """
    logger.info("Training data builder")

    builder = TrainingDataBuilder()

    train_data, val_data, test_data = builder.prepare_training_data()

    if train_data is not None:
        logger.info("Training data ready:")
        logger.info("   - Combined user interactions with recipe features")
        logger.info(
            f"   - {len(train_data) + len(val_data) + len(test_data)} total training samples"
        )
        logger.info("   - Ready for model training to score and rank recipes")

        # Show feature summary
        feature_columns = [
            col
            for col in train_data.columns
            if col not in ["user_id", "recipe_id", "label", "rating", "datetime"]
        ]

        logger.info(f"Feature summary ({len(feature_columns)} total):")
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

        logger.info(f"   User features: {len(user_features)}")
        logger.info(f"   Recipe features: {len(recipe_features)}")
        logger.info(f"   Interaction features: {len(interaction_features)}")
        logger.info(
            f"   Other features: {len(feature_columns) - len(user_features) - len(recipe_features) - len(interaction_features)}"
        )

    else:
        logger.error("Failed to build training data")


if __name__ == "__main__":
    main()
