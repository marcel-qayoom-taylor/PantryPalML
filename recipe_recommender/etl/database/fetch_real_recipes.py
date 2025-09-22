#!/usr/bin/env python3
"""
Fetch Real Recipe Data from Supabase Production Database

This script connects to the actual PantryPal Supabase database and fetches
comprehensive recipe data including ingredients, collections, and metadata.
"""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from supabase_config import SupabaseConfig


class RealRecipeDataFetcher:
    """Fetch comprehensive recipe data from the real PantryPal database."""

    def __init__(self, use_service_role: bool = False):
        """Initialize the recipe data fetcher."""
        self.config = SupabaseConfig()
        self.client = self.config.get_client(use_service_role=use_service_role)
        self.output_dir = Path(
            "/Users/marcelqayoomtaylor/Documents/GitHub/PantryPalUtils/ml_etl/output"
        )

        # Store fetched data
        self.recipes_df = pd.DataFrame()
        self.ingredients_df = pd.DataFrame()
        self.recipe_ingredients_df = pd.DataFrame()

    def fetch_recipes(self, limit: int | None = None) -> pd.DataFrame:
        """Fetch all recipe data."""
        print("Fetching recipes from database...")

        try:
            query = self.client.table("recipe").select("*")
            if limit:
                query = query.limit(limit)

            response = query.execute()

            if response.data:
                df = pd.DataFrame(response.data)
                print(f"Fetched {len(df)} recipes")

                # Convert datetime columns with flexible parsing
                datetime_cols = ["created_at", "updated_at"]
                for col in datetime_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(
                            df[col], format="ISO8601", errors="coerce"
                        )

                self.recipes_df = df
                return df
            print("No recipe data found")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching recipes: {e}")
            return pd.DataFrame()

    def fetch_ingredients(self) -> pd.DataFrame:
        """Fetch all ingredient data."""
        print("Fetching ingredients from database...")

        try:
            response = self.client.table("ingredient").select("*").execute()

            if response.data:
                df = pd.DataFrame(response.data)
                print(f"Fetched {len(df)} ingredients")
                self.ingredients_df = df
                return df
            print("No ingredient data found")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching ingredients: {e}")
            return pd.DataFrame()

    def fetch_recipe_ingredients(self) -> pd.DataFrame:
        """Fetch recipe-ingredient relationships."""
        print("Fetching recipe-ingredient relationships...")

        try:
            response = self.client.table("ingredients_of_recipe").select("*").execute()

            if response.data:
                df = pd.DataFrame(response.data)
                print(f"Fetched {len(df)} recipe-ingredient relationships")
                self.recipe_ingredients_df = df
                return df
            print("No recipe-ingredient data found")
            return pd.DataFrame()

        except Exception as e:
            print(f"Error fetching recipe-ingredient relationships: {e}")
            return pd.DataFrame()

    def fetch_collections(self) -> pd.DataFrame:
        """Fetch recipe collections."""
        print("Fetching recipe collections...")

        try:
            # Fetch collections
            collections_response = self.client.table("collection").select("*").execute()
            collection_recipes_response = (
                self.client.table("collection_of_recipes").select("*").execute()
            )

            collections_data = []
            if collections_response.data:
                collections_data.extend(collections_response.data)

            collection_recipes_data = []
            if collection_recipes_response.data:
                collection_recipes_data.extend(collection_recipes_response.data)

            print(
                f"Fetched {len(collections_data)} collections and {len(collection_recipes_data)} collection-recipe relationships"
            )

            return {
                "collections": pd.DataFrame(collections_data),
                "collection_recipes": pd.DataFrame(collection_recipes_data),
            }

        except Exception as e:
            print(f"Error fetching collections: {e}")
            return {"collections": pd.DataFrame(), "collection_recipes": pd.DataFrame()}

    def analyze_data_structure(self):
        """Analyze the structure of the fetched data."""
        print("\nDatabase structure analysis")
        print("-" * 60)

        if not self.recipes_df.empty:
            print(f"\nRECIPES TABLE ({len(self.recipes_df)} records):")
            print(f"   Columns: {list(self.recipes_df.columns)}")

            # Show sample data
            if len(self.recipes_df) > 0:
                sample = self.recipes_df.iloc[0]
                print("   Sample recipe:")
                for col, val in sample.items():
                    display_val = (
                        str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                    )
                    print(f"     {col}: {display_val}")

        if not self.ingredients_df.empty:
            print(f"\nINGREDIENTS TABLE ({len(self.ingredients_df)} records):")
            print(f"   Columns: {list(self.ingredients_df.columns)}")

            if len(self.ingredients_df) > 0:
                sample = self.ingredients_df.iloc[0]
                print("   Sample ingredient:")
                for col, val in sample.items():
                    display_val = (
                        str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                    )
                    print(f"     {col}: {display_val}")

        if not self.recipe_ingredients_df.empty:
            print(
                f"\nRECIPE-INGREDIENTS TABLE ({len(self.recipe_ingredients_df)} records):"
            )
            print(f"   Columns: {list(self.recipe_ingredients_df.columns)}")

            # Show relationship statistics
            unique_recipes = (
                self.recipe_ingredients_df["recipe_id"].nunique()
                if "recipe_id" in self.recipe_ingredients_df.columns
                else 0
            )
            unique_ingredients = (
                self.recipe_ingredients_df["ingredient_id"].nunique()
                if "ingredient_id" in self.recipe_ingredients_df.columns
                else 0
            )
            print(
                f"   Covers {unique_recipes} recipes and {unique_ingredients} ingredients"
            )

    def save_comprehensive_dataset(self):
        """Save all fetched data to CSV files."""
        print("\nSaving comprehensive dataset")
        print("-" * 40)

        files_saved = []

        if not self.recipes_df.empty:
            filename = "real_recipes_from_db.csv"
            self.recipes_df.to_csv(self.output_dir / filename, index=False)
            files_saved.append(f"{filename} ({len(self.recipes_df)} records)")

        if not self.ingredients_df.empty:
            filename = "real_ingredients_from_db.csv"
            self.ingredients_df.to_csv(self.output_dir / filename, index=False)
            files_saved.append(f"{filename} ({len(self.ingredients_df)} records)")

        if not self.recipe_ingredients_df.empty:
            filename = "real_recipe_ingredients_from_db.csv"
            self.recipe_ingredients_df.to_csv(self.output_dir / filename, index=False)
            files_saved.append(
                f"{filename} ({len(self.recipe_ingredients_df)} records)"
            )

        if files_saved:
            print("Saved files:")
            for file_info in files_saved:
                print(f"   - {file_info}")
        else:
            print("No data to save")

        # Generate summary report
        summary_file = self.output_dir / "real_database_summary.txt"
        with open(summary_file, "w") as f:
            f.write("PANTRYPAL REAL DATABASE SUMMARY\n")
            f.write("=" * 50 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write(f"RECIPES: {len(self.recipes_df)} records\n")
            if not self.recipes_df.empty:
                f.write(f"   Columns: {list(self.recipes_df.columns)}\n")

            f.write(f"\nINGREDIENTS: {len(self.ingredients_df)} records\n")
            if not self.ingredients_df.empty:
                f.write(f"   Columns: {list(self.ingredients_df.columns)}\n")

            f.write(
                f"\nRECIPE-INGREDIENT RELATIONSHIPS: {len(self.recipe_ingredients_df)} records\n"
            )
            if not self.recipe_ingredients_df.empty:
                f.write(f"   Columns: {list(self.recipe_ingredients_df.columns)}\n")
                unique_recipes = (
                    self.recipe_ingredients_df["recipe_id"].nunique()
                    if "recipe_id" in self.recipe_ingredients_df.columns
                    else 0
                )
                unique_ingredients = (
                    self.recipe_ingredients_df["ingredient_id"].nunique()
                    if "ingredient_id" in self.recipe_ingredients_df.columns
                    else 0
                )
                f.write(
                    f"   Coverage: {unique_recipes} recipes, {unique_ingredients} ingredients\n"
                )

        print(f"Summary saved to: {summary_file.name}")

    def build_enhanced_features(self) -> pd.DataFrame:
        """Build enhanced recipe features from the real database."""
        print("\nBuilding enhanced recipe features")
        print("-" * 50)

        if self.recipes_df.empty:
            print("No recipe data available")
            return pd.DataFrame()

        # Start with recipe base data
        enhanced_recipes = self.recipes_df.copy()

        # Add ingredient counts and details
        if not self.recipe_ingredients_df.empty:
            ingredient_stats = (
                self.recipe_ingredients_df.groupby("recipe_id")
                .agg({"ingredient_id": ["count", "nunique"]})
                .round(2)
            )

            ingredient_stats.columns = ["ingredient_count", "unique_ingredients"]
            enhanced_recipes = enhanced_recipes.merge(
                ingredient_stats, left_on="recipe_id", right_index=True, how="left"
            )

            # Fill missing values
            enhanced_recipes["ingredient_count"] = enhanced_recipes[
                "ingredient_count"
            ].fillna(0)
            enhanced_recipes["unique_ingredients"] = enhanced_recipes[
                "unique_ingredients"
            ].fillna(0)

        # Skip category diversity for now (categories are stored as lists)
        # Could be added later by flattening the category lists

        # Create complexity score
        complexity_factors = []

        if "ingredient_count" in enhanced_recipes.columns:
            complexity_factors.append(enhanced_recipes["ingredient_count"] / 20)

        if "cook_time" in enhanced_recipes.columns:
            enhanced_recipes["cook_time"] = pd.to_numeric(
                enhanced_recipes["cook_time"], errors="coerce"
            )
            complexity_factors.append(enhanced_recipes["cook_time"].fillna(30) / 60)

        if "prep_time" in enhanced_recipes.columns:
            enhanced_recipes["prep_time"] = pd.to_numeric(
                enhanced_recipes["prep_time"], errors="coerce"
            )
            complexity_factors.append(enhanced_recipes["prep_time"].fillna(15) / 60)

        if complexity_factors:
            enhanced_recipes["complexity_score"] = np.mean(complexity_factors, axis=0)

        print(
            f"Enhanced {len(enhanced_recipes)} recipes with {len(enhanced_recipes.columns)} features"
        )

        # Save enhanced features
        enhanced_recipes.to_csv(
            self.output_dir / "enhanced_recipe_features_from_db.csv", index=False
        )
        print("Saved enhanced features to: enhanced_recipe_features_from_db.csv")

        return enhanced_recipes


def main():
    """Main function to fetch and analyze real recipe data."""
    print("Fetching real PantryPal recipe data")
    print("-" * 70)

    try:
        # Initialize fetcher with service role for admin access
        fetcher = RealRecipeDataFetcher(use_service_role=True)

        # Fetch all data
        print("Connecting to PantryPal staging database...")

        recipes_df = fetcher.fetch_recipes()
        ingredients_df = fetcher.fetch_ingredients()
        recipe_ingredients_df = fetcher.fetch_recipe_ingredients()
        fetcher.fetch_collections()

        # Analyze data structure
        fetcher.analyze_data_structure()

        # Save comprehensive dataset
        fetcher.save_comprehensive_dataset()

        # Build enhanced features
        fetcher.build_enhanced_features()

        print("\nComprehensive recipe data extracted:")
        print(f"   - {len(recipes_df)} recipes with full metadata")
        print(f"   - {len(ingredients_df)} ingredients")
        print(f"   - {len(recipe_ingredients_df)} recipe-ingredient relationships")
        print("   - Enhanced feature set ready for ML model")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you're connected to the staging database")


if __name__ == "__main__":
    main()
