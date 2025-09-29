#!/usr/bin/env python3
"""
Script to analyze recipe duplicates between v1 and v2 event data files.
Creates a mapping between v1 and v2 recipe IDs based on recipe_url.
"""

import json
import csv
from pathlib import Path
from collections import defaultdict, Counter
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_events(file_path):
    """Load events from JSON file."""
    logger.info(f"Loading events from {file_path}")
    events = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                event = json.loads(line.strip())
                events.append(event)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse line {line_num}: {e}")
                continue
    logger.info(f"Loaded {len(events)} events from {file_path}")
    return events


def extract_recipe_events(events, version):
    """Extract recipe-related events and their metadata."""
    recipe_events = []
    recipe_event_types = [
        "Recipe Viewed",
        "Recipe Cooked",
        "Recipe Link Clicked",
        "Recipe Favourited",
        "Recipe Unfavourited",
    ]

    for event in events:
        if event.get("event") in recipe_event_types:
            properties = event.get("properties", {})
            recipe_id = properties.get("recipe_id")
            recipe_url = properties.get("recipe_url")
            recipe_name = properties.get("recipe_name")

            if recipe_id and recipe_url:
                recipe_events.append(
                    {
                        "version": version,
                        "event_type": event.get("event"),
                        "recipe_id": recipe_id,
                        "recipe_url": recipe_url,
                        "recipe_name": recipe_name,
                        "recipe_author": properties.get("recipe_author"),
                        "recipe_cooking_time": properties.get("recipe_cooking_time"),
                        "recipe_servings": properties.get("recipe_servings"),
                        "recipe_tags": properties.get("recipe_tags"),
                        "timestamp": properties.get("time"),
                    }
                )

    logger.info(f"Extracted {len(recipe_events)} recipe events from {version}")
    return recipe_events


def analyze_duplicates(v1_recipes, v2_recipes):
    """Analyze duplicates between v1 and v2 recipes based on recipe_url."""
    logger.info("Analyzing duplicates...")

    # Create dictionaries grouped by recipe_url
    v1_by_url = defaultdict(list)
    v2_by_url = defaultdict(list)

    for recipe in v1_recipes:
        v1_by_url[recipe["recipe_url"]].append(recipe)

    for recipe in v2_recipes:
        v2_by_url[recipe["recipe_url"]].append(recipe)

    # Get unique recipes by URL from each version
    v1_unique_recipes = {}
    v2_unique_recipes = {}

    for url, recipes in v1_by_url.items():
        # Take the first recipe for each URL (they should be the same except for ID)
        v1_unique_recipes[url] = recipes[0]

    for url, recipes in v2_by_url.items():
        v2_unique_recipes[url] = recipes[0]

    # Find common URLs (duplicates)
    common_urls = set(v1_unique_recipes.keys()) & set(v2_unique_recipes.keys())
    v1_only_urls = set(v1_unique_recipes.keys()) - set(v2_unique_recipes.keys())
    v2_only_urls = set(v2_unique_recipes.keys()) - set(v1_unique_recipes.keys())

    logger.info(f"Found {len(common_urls)} recipes in both v1 and v2")
    logger.info(f"Found {len(v1_only_urls)} recipes only in v1")
    logger.info(f"Found {len(v2_only_urls)} recipes only in v2")

    return {
        "common_urls": common_urls,
        "v1_only_urls": v1_only_urls,
        "v2_only_urls": v2_only_urls,
        "v1_unique_recipes": v1_unique_recipes,
        "v2_unique_recipes": v2_unique_recipes,
    }


def create_mapping(analysis_result):
    """Create mapping between v1 and v2 recipe IDs."""
    logger.info("Creating recipe ID mapping...")

    mapping = []
    common_urls = analysis_result["common_urls"]
    v1_recipes = analysis_result["v1_unique_recipes"]
    v2_recipes = analysis_result["v2_unique_recipes"]

    for url in common_urls:
        v1_recipe = v1_recipes[url]
        v2_recipe = v2_recipes[url]

        mapping.append(
            {
                "recipe_url": url,
                "v1_recipe_id": v1_recipe["recipe_id"],
                "v2_recipe_id": v2_recipe["recipe_id"],
                "recipe_name": v1_recipe["recipe_name"] or v2_recipe["recipe_name"],
                "recipe_author": v1_recipe["recipe_author"]
                or v2_recipe["recipe_author"],
                "are_identical": v1_recipe["recipe_name"] == v2_recipe["recipe_name"],
            }
        )

    logger.info(f"Created mapping for {len(mapping)} duplicate recipes")
    return mapping


def save_results(mapping, analysis_result, output_dir):
    """Save results to files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save the main mapping as CSV
    mapping_file = output_path / "recipe_id_mapping_v1_v2.csv"
    with open(mapping_file, "w", newline="", encoding="utf-8") as f:
        if mapping:
            writer = csv.DictWriter(f, fieldnames=mapping[0].keys())
            writer.writeheader()
            writer.writerows(mapping)
    logger.info(f"Saved recipe mapping to {mapping_file}")

    # Save detailed analysis
    analysis_file = output_path / "recipe_duplicate_analysis.json"
    with open(analysis_file, "w") as f:
        analysis_summary = {
            "total_duplicates": len(mapping),
            "v1_only_count": len(analysis_result["v1_only_urls"]),
            "v2_only_count": len(analysis_result["v2_only_urls"]),
            "common_count": len(analysis_result["common_urls"]),
            "v1_only_urls": list(analysis_result["v1_only_urls"]),
            "v2_only_urls": list(analysis_result["v2_only_urls"]),
        }
        json.dump(analysis_summary, f, indent=2)
    logger.info(f"Saved analysis summary to {analysis_file}")

    # Save the full mapping as JSON as well for easier programmatic access
    mapping_json_file = output_path / "recipe_id_mapping_v1_v2.json"
    with open(mapping_json_file, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info(f"Saved recipe mapping as JSON to {mapping_json_file}")

    return mapping_file, analysis_file, mapping_json_file


def print_summary(mapping, analysis_result):
    """Print summary of the analysis."""
    print("\n" + "=" * 50)
    print("RECIPE DUPLICATE ANALYSIS SUMMARY")
    print("=" * 50)

    print(f"📊 Total unique recipes in v1: {len(analysis_result['v1_unique_recipes'])}")
    print(f"📊 Total unique recipes in v2: {len(analysis_result['v2_unique_recipes'])}")
    print(f"🔗 Recipes appearing in both versions: {len(mapping)}")
    print(f"📝 Recipes only in v1: {len(analysis_result['v1_only_urls'])}")
    print(f"📝 Recipes only in v2: {len(analysis_result['v2_only_urls'])}")

    if mapping:
        print(f"\n🔍 Sample mappings:")
        for i, item in enumerate(mapping[:5]):
            print(f"  {i+1}. {item['recipe_name']}")
            print(f"     v1: {item['v1_recipe_id']}")
            print(f"     v2: {item['v2_recipe_id']}")
            print(f"     URL: {item['recipe_url'][:60]}...")
            print()

    # Calculate the duplicate issue
    total_before_dedup = len(analysis_result["v1_unique_recipes"]) + len(
        analysis_result["v2_unique_recipes"]
    )
    total_after_dedup = len(analysis_result["v1_unique_recipes"]) + len(
        analysis_result["v2_only_urls"]
    )
    duplicates_removed = total_before_dedup - total_after_dedup

    print(f"🎯 DUPLICATE IMPACT:")
    print(f"   Before deduplication: {total_before_dedup} recipes")
    print(f"   After deduplication: {total_after_dedup} recipes")
    print(f"   Duplicates to remove: {duplicates_removed} recipes")
    print("=" * 50)


def main():
    """Main function."""
    # File paths
    v1_file = Path("recipe_recommender/input/v1_events_20250827.json")
    v2_file = Path("recipe_recommender/input/v2_events_20250920.json")
    output_dir = Path("recipe_recommender/output")

    # Load events
    v1_events = load_events(v1_file)
    v2_events = load_events(v2_file)

    # Extract recipe events
    v1_recipes = extract_recipe_events(v1_events, "v1")
    v2_recipes = extract_recipe_events(v2_events, "v2")

    # Analyze duplicates
    analysis_result = analyze_duplicates(v1_recipes, v2_recipes)

    # Create mapping
    mapping = create_mapping(analysis_result)

    # Save results
    saved_files = save_results(mapping, analysis_result, output_dir)

    # Print summary
    print_summary(mapping, analysis_result)

    print(f"\n✅ Files saved:")
    for file_path in saved_files:
        print(f"   - {file_path}")

    return mapping, analysis_result


if __name__ == "__main__":
    mapping, analysis_result = main()
