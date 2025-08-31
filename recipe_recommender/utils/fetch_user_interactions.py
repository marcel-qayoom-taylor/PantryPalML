#!/usr/bin/env python3
"""
Fetch User Interactions from Cleaned Mixpanel Data

This script fetches a user's interaction history from the cleaned combined_events.csv
file and formats it for use with the production recipe scorer.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import argparse
import json

from recipe_recommender.config import get_ml_config
from recipe_recommender.utils import setup_logging

logger = setup_logging(__name__)


class UserInteractionFetcher:
    """
    Fetches user interaction history from cleaned Mixpanel data.

    For beginners: This class reads the combined events CSV file and extracts
    all recipe-related interactions for a specific user.
    """

    def __init__(self, config=None):
        """Initialize the fetcher with configuration."""
        self.config = config or get_ml_config()
        self.events_file = self.config.output_dir / "combined_events.csv"

        # Recipe-related events that are useful for recommendations
        self.recipe_events = {
            "Recipe Viewed",
            "Recipe Cooked",
            "Recipe Favourited",
            "Recipe Link Clicked",
            "Recipe Cook Started",
            "Recipe Added To Collections",
            "Recipe Removed From Collections",
            "Recipe Search Queried",
        }

        logger.info(f"🔍 Initialized UserInteractionFetcher")
        logger.info(f"   Events file: {self.events_file}")
        logger.info(f"   Tracking {len(self.recipe_events)} event types")

    def fetch_user_interactions(
        self, user_id: str, include_recipe_ids: bool = True
    ) -> List[Dict]:
        """
        Fetch all recipe interactions for a specific user.

        Args:
            user_id: The user profile ID (distinct_id from Mixpanel)
            include_recipe_ids: Whether to try to extract recipe IDs from events

        Returns:
            List of interaction dictionaries formatted for the recommendation model
        """
        logger.info(f"🎯 Fetching interactions for user: {user_id}")

        if not self.events_file.exists():
            raise FileNotFoundError(f"Events file not found: {self.events_file}")

        # Read the events file in chunks to handle large files efficiently
        interactions = []
        chunk_size = 10000
        total_rows = 0
        matching_rows = 0

        logger.info("📖 Reading events file in chunks...")

        try:
            for chunk in pd.read_csv(self.events_file, chunksize=chunk_size):
                total_rows += len(chunk)

                # Filter for this user and recipe events
                user_events = chunk[
                    (chunk["distinct_id"] == user_id)
                    & (chunk["event"].isin(self.recipe_events))
                ]

                matching_rows += len(user_events)

                # Convert to interaction format
                for _, row in user_events.iterrows():
                    interaction = {
                        "event_type": row["event"],
                        "timestamp": int(row["timestamp"]),
                        "recipe_id": (
                            str(row["recipe_id"])
                            if pd.notna(row.get("recipe_id")) and include_recipe_ids
                            else "unknown"
                        ),
                    }

                    # Add optional metadata
                    if "current_route" in row and pd.notna(row["current_route"]):
                        interaction["route"] = row["current_route"]

                    interactions.append(interaction)

                # Progress logging
                if total_rows % 50000 == 0:
                    logger.info(
                        f"   Processed {total_rows:,} rows, found {matching_rows} matches..."
                    )

        except Exception as e:
            logger.error(f"❌ Error reading events file: {e}")
            raise

        # Sort by timestamp
        interactions.sort(key=lambda x: x["timestamp"])

        logger.info(
            f"✅ Found {len(interactions)} recipe interactions for user {user_id}"
        )
        logger.info(f"   Processed {total_rows:,} total events")
        logger.info(f"   Time range: {self._format_timestamp_range(interactions)}")
        logger.info(f"   Event breakdown: {self._get_event_breakdown(interactions)}")

        return interactions

    def _extract_recipe_id(self, row) -> str:
        """
        Extract recipe ID from the event data.

        For beginners: Recipe IDs might be embedded in routes like /recipes/[id]
        or we might need to use other methods to identify which recipe was interacted with.
        """
        # Try to extract from current_route
        if "current_route" in row and pd.notna(row["current_route"]):
            route = str(row["current_route"])

            # Look for patterns like /recipes/[id] or /recipes/123
            if "/recipes/" in route:
                parts = route.split("/recipes/")
                if len(parts) > 1:
                    recipe_part = (
                        parts[1].split("/")[0].split("?")[0]
                    )  # Remove query params
                    if recipe_part and recipe_part != "[id]":
                        return recipe_part

        # For now, return a placeholder - in production you might have recipe IDs
        # stored differently or need to join with other data
        return "unknown"

    def _format_timestamp_range(self, interactions: List[Dict]) -> str:
        """Format the timestamp range for logging."""
        if not interactions:
            return "No interactions"

        from datetime import datetime

        first = datetime.fromtimestamp(interactions[0]["timestamp"])
        last = datetime.fromtimestamp(interactions[-1]["timestamp"])

        return f"{first.strftime('%Y-%m-%d')} to {last.strftime('%Y-%m-%d')}"

    def _get_event_breakdown(self, interactions: List[Dict]) -> Dict[str, int]:
        """Get a breakdown of event types."""
        breakdown = {}
        for interaction in interactions:
            event_type = interaction["event_type"]
            breakdown[event_type] = breakdown.get(event_type, 0) + 1
        return breakdown

    def get_user_summary(self, user_id: str) -> Dict:
        """
        Get a summary of user activity without fetching all interactions.

        Args:
            user_id: The user profile ID

        Returns:
            Dictionary with user activity summary
        """
        logger.info(f"📊 Getting activity summary for user: {user_id}")

        if not self.events_file.exists():
            raise FileNotFoundError(f"Events file not found: {self.events_file}")

        summary = {
            "user_id": user_id,
            "total_events": 0,
            "recipe_events": 0,
            "event_types": {},
            "first_seen": None,
            "last_seen": None,
            "days_active": 0,
        }

        chunk_size = 10000
        timestamps = []

        try:
            for chunk in pd.read_csv(self.events_file, chunksize=chunk_size):
                user_events = chunk[chunk["distinct_id"] == user_id]

                if len(user_events) > 0:
                    summary["total_events"] += len(user_events)

                    # Count recipe events
                    recipe_events = user_events[
                        user_events["event"].isin(self.recipe_events)
                    ]
                    summary["recipe_events"] += len(recipe_events)

                    # Track event types
                    for event_type in user_events["event"]:
                        summary["event_types"][event_type] = (
                            summary["event_types"].get(event_type, 0) + 1
                        )

                    # Collect timestamps
                    timestamps.extend(user_events["timestamp"].tolist())

        except Exception as e:
            logger.error(f"❌ Error reading events file: {e}")
            raise

        # Calculate time-based metrics
        if timestamps:
            timestamps.sort()
            summary["first_seen"] = timestamps[0]
            summary["last_seen"] = timestamps[-1]
            summary["days_active"] = max(
                1, (timestamps[-1] - timestamps[0]) // (24 * 3600)
            )

        logger.info(
            f"✅ User summary: {summary['total_events']} total events, {summary['recipe_events']} recipe events"
        )

        return summary


def main():
    """
    Command-line interface for fetching user interactions.
    """
    parser = argparse.ArgumentParser(
        description="Fetch user interactions from cleaned Mixpanel data"
    )
    parser.add_argument("user_id", help="User profile ID (distinct_id)")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument(
        "--summary", "-s", action="store_true", help="Show summary only"
    )
    parser.add_argument(
        "--no-recipe-ids", action="store_true", help="Don't try to extract recipe IDs"
    )

    args = parser.parse_args()

    logger.info("🚀 USER INTERACTION FETCHER")
    logger.info("=" * 50)

    try:
        fetcher = UserInteractionFetcher()

        if args.summary:
            # Get summary only
            summary = fetcher.get_user_summary(args.user_id)

            print("\n📊 USER ACTIVITY SUMMARY:")
            print(f"   User ID: {summary['user_id']}")
            print(f"   Total Events: {summary['total_events']:,}")
            print(f"   Recipe Events: {summary['recipe_events']:,}")
            print(f"   Days Active: {summary['days_active']}")

            if summary["first_seen"] and summary["last_seen"]:
                from datetime import datetime

                first = datetime.fromtimestamp(summary["first_seen"])
                last = datetime.fromtimestamp(summary["last_seen"])
                print(f"   First Seen: {first.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"   Last Seen: {last.strftime('%Y-%m-%d %H:%M:%S')}")

            print("\n📈 EVENT BREAKDOWN:")
            for event_type, count in sorted(
                summary["event_types"].items(), key=lambda x: x[1], reverse=True
            ):
                print(f"   {event_type}: {count:,}")

        else:
            # Get full interactions
            interactions = fetcher.fetch_user_interactions(
                args.user_id, include_recipe_ids=not args.no_recipe_ids
            )

            if args.output:
                # Save to file
                output_path = Path(args.output)
                with open(output_path, "w") as f:
                    json.dump(interactions, f, indent=2)
                logger.info(
                    f"💾 Saved {len(interactions)} interactions to {output_path}"
                )
            else:
                # Print to console
                print(f"\n🎯 INTERACTIONS FOR USER {args.user_id}:")
                print(f"Found {len(interactions)} recipe interactions\n")

                for i, interaction in enumerate(interactions[:10], 1):  # Show first 10
                    from datetime import datetime

                    dt = datetime.fromtimestamp(interaction["timestamp"])
                    print(f"   {i}. {interaction['event_type']}")
                    print(f"      Recipe ID: {interaction['recipe_id']}")
                    print(f"      Time: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                    if "route" in interaction:
                        print(f"      Route: {interaction['route']}")
                    print()

                if len(interactions) > 10:
                    print(f"   ... and {len(interactions) - 10} more interactions")

                print("\n💡 To save to file, use: --output interactions.json")
                print(
                    "💡 To use with inference, pass this list to ProductionRecipeScorer"
                )

    except Exception as e:
        logger.error(f"❌ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
