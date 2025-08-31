#!/usr/bin/env python3
"""
Complete Inference Example

This script demonstrates how to:
1. Fetch a user's interaction history from cleaned Mixpanel data
2. Run inference to get recipe recommendations
3. Display the results

Usage:
    python run_inference_example.py 123a02b5-cac8-4d1b-973e-3a9fe0f2303d
"""

import argparse
import json

from recipe_recommender.utils.fetch_user_interactions import UserInteractionFetcher
from recipe_recommender.inference.production_recipe_scorer import ProductionRecipeScorer
from recipe_recommender.utils import setup_logging

logger = setup_logging(__name__)


def run_inference_for_user(user_id: str, n_recommendations: int = 10) -> dict:
    """
    Complete pipeline: fetch interactions and get recommendations.

    Args:
        user_id: User profile ID
        n_recommendations: Number of recommendations to return

    Returns:
        Dictionary with recommendations and metadata
    """
    logger.info(f"🎯 Running complete inference pipeline for user: {user_id}")
    logger.info("=" * 60)

    # Step 1: Fetch user interactions
    logger.info("📖 Step 1: Fetching user interaction history...")
    fetcher = UserInteractionFetcher()
    interactions = fetcher.fetch_user_interactions(user_id)

    if not interactions:
        logger.warning("⚠️  No recipe interactions found for this user")
        logger.info("   Will use default user profile for recommendations")

    # Step 2: Initialize the production scorer
    logger.info("🤖 Step 2: Loading trained model...")
    scorer = ProductionRecipeScorer()

    # Step 3: Get recommendations
    logger.info(f"🎯 Step 3: Generating {n_recommendations} recommendations...")
    result = scorer.get_user_recipe_recommendations(
        user_id=user_id,
        interaction_history=interactions,
        n_recommendations=n_recommendations,
    )

    return result


def display_recommendations(result: dict):
    """Display recommendations in a user-friendly format."""

    print("\n" + "=" * 80)
    print(f"🍳 RECIPE RECOMMENDATIONS FOR USER: {result['user_id']}")
    print("=" * 80)

    metadata = result["metadata"]
    print(f"📊 Model: {metadata['model_type']}")
    print(f"⚡ Processing time: {metadata['processing_time_seconds']:.3f} seconds")
    print(f"📈 Total recipes scored: {metadata['total_recipes_scored']:,}")
    print(f"🎯 User interactions: {metadata['user_interaction_count']}")
    print(f"🏆 Recommendations returned: {metadata['recommendation_count']}")
    print(f"🕐 Generated at: {metadata['generated_at']}")

    print(f"\n🏆 TOP {len(result['recommendations'])} RECOMMENDATIONS:")
    print("-" * 80)

    for i, rec in enumerate(result["recommendations"], 1):
        print(f"\n{i}. {rec['recipe_name']}")
        print(f"   📊 Score: {rec['score']:.4f}")
        print(f"   👨‍🍳 Author: {rec['author_name']}")

        # Time and servings
        time_info = []
        if rec.get("total_time"):
            time_info.append(f"{rec['total_time']}min")
        if rec.get("servings"):
            time_info.append(f"{rec['servings']} servings")
        if time_info:
            print(f"   ⏱️  {' • '.join(time_info)}")

        # Complexity and ingredients
        complexity_info = []
        if rec.get("ingredient_count"):
            complexity_info.append(f"{rec['ingredient_count']} ingredients")
        if rec.get("complexity_score"):
            complexity_info.append(f"complexity: {rec['complexity_score']:.2f}")
        if complexity_info:
            print(f"   🔧 {' • '.join(complexity_info)}")

        # Tags
        if rec.get("tags") and rec["tags"]:
            try:
                # Handle tags that might be a string or list
                tags = rec["tags"]
                if isinstance(tags, str):
                    tags = (
                        tags.replace("[", "")
                        .replace("]", "")
                        .replace("'", "")
                        .split(", ")
                    )
                if isinstance(tags, list) and len(tags) > 0 and tags[0]:
                    print(f"   🏷️  {', '.join(tags[:5])}")  # Show first 5 tags
            except Exception:
                # Skip tags if there's any parsing error
                pass

        # Description
        if rec.get("description") and str(rec["description"]).strip():
            desc = str(rec["description"]).strip()
            if len(desc) > 100:
                desc = desc[:100] + "..."
            print(f"   📝 {desc}")

        # URL
        if rec.get("recipe_url"):
            print(f"   🔗 {rec['recipe_url']}")


def main():
    """Command-line interface for running inference."""
    parser = argparse.ArgumentParser(
        description="Run recipe recommendations for a user"
    )
    parser.add_argument("user_id", help="User profile ID (distinct_id)")
    parser.add_argument(
        "--recommendations",
        "-n",
        type=int,
        default=10,
        help="Number of recommendations (default: 10)",
    )
    parser.add_argument("--output", "-o", help="Save results to JSON file")
    parser.add_argument(
        "--interactions-only",
        action="store_true",
        help="Only fetch and display interactions, don't run inference",
    )

    args = parser.parse_args()

    logger.info("🚀 RECIPE RECOMMENDATION INFERENCE")
    logger.info("=" * 50)

    try:
        if args.interactions_only:
            # Just fetch and display interactions
            fetcher = UserInteractionFetcher()
            interactions = fetcher.fetch_user_interactions(args.user_id)

            print(f"\n🎯 INTERACTIONS FOR USER {args.user_id}:")
            print(f"Found {len(interactions)} recipe interactions\n")

            for i, interaction in enumerate(interactions, 1):
                from datetime import datetime

                dt = datetime.fromtimestamp(interaction["timestamp"])
                print(f"   {i}. {interaction['event_type']}")
                print(f"      Recipe ID: {interaction['recipe_id']}")
                print(f"      Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                if "route" in interaction:
                    print(f"      Route: {interaction['route']}")
                print()

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(interactions, f, indent=2)
                print(f"💾 Saved interactions to {args.output}")

        else:
            # Run complete inference pipeline
            result = run_inference_for_user(args.user_id, args.recommendations)

            # Display results
            try:
                display_recommendations(result)
            except Exception as e:
                logger.error(f"❌ Error displaying recommendations: {e}")
                logger.info("📊 Raw result structure:")
                logger.info(
                    f"   Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}"
                )
                if isinstance(result, dict) and "recommendations" in result:
                    logger.info(
                        f"   Recommendations count: {len(result['recommendations'])}"
                    )
                    if result["recommendations"]:
                        logger.info(
                            f"   First recommendation keys: {list(result['recommendations'][0].keys())}"
                        )
                raise

            # Save to file if requested
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(result, f, indent=2)
                logger.info(f"💾 Saved recommendations to {args.output}")

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
