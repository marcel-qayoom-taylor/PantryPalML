import json
from pathlib import Path

import pandas as pd

from .helpers import (
    clean_event_name,
    get_day_of_week,
    get_device_type,
    get_is_weekend,
    get_platform_from_props,
    get_time_of_day,
)


def load_events_to_dataframe(file_path, event_version):
    """Load events from JSON file and convert to DataFrame."""
    records = []

    with open(file_path) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                event_data = json.loads(line)
                props = event_data.get("properties", {})

                # Skip development events for v2
                if event_version == "v2" and props.get("is_development", True):
                    continue

                # Extract common fields
                record = {
                    "event": event_data.get("event"),
                    "timestamp": props.get("time"),
                    "distinct_id": props.get("distinct_id"),
                    "device_id": props.get("$device_id") or props.get("device_id"),
                    "country_code": props.get("mp_country_code"),
                    "city": props.get("$city") or props.get("city"),
                    "region": props.get("$region") or props.get("region"),
                    "screen_height": props.get("$screen_height")
                    or props.get("screen_height"),
                    "screen_width": props.get("$screen_width")
                    or props.get("screen_width"),
                    "recipe_id": props.get("recipe_id"),  # Extract recipe ID
                }

                # Version-specific fields
                if event_version == "v1":
                    record.update(
                        {
                            "current_route": "",
                            "platform": "ios",
                            "app_version": props.get("$app_version_string", "1.0.0"),
                        }
                    )
                else:  # v2
                    record.update(
                        {
                            "current_route": props.get("current_route", ""),
                            "platform": get_platform_from_props(props),
                            "app_version": props.get("$app_version_string", "2.0.0"),
                        }
                    )

                records.append(record)

            except json.JSONDecodeError:
                continue

    return pd.DataFrame(records)


def add_derived_columns(df):
    """Add derived columns to the DataFrame."""
    # Clean event names first
    df["event"] = df["event"].apply(clean_event_name)

    # Apply derived field functions
    df["device_type"] = df.apply(
        lambda row: get_device_type(row["screen_height"], row["screen_width"]), axis=1
    )
    df["time_of_day"] = df["timestamp"].apply(get_time_of_day)
    df["day_of_week"] = df["timestamp"].apply(get_day_of_week)
    df["is_weekend"] = df["timestamp"].apply(get_is_weekend)

    return df


def main():
    # Get paths relative to this script's location
    script_dir = Path(__file__).parent
    ml_etl_dir = (
        script_dir.parent.parent
    )  # Go up two levels: events -> etl -> recipe_recommender

    v1_file = ml_etl_dir / "input" / "v1_events_20250827.json"
    v2_file = ml_etl_dir / "input" / "v2_events_20250827.json"
    output_file = ml_etl_dir / "output" / "combined_events.csv"

    print("Loading v1 events...")
    v1_df = load_events_to_dataframe(str(v1_file), "v1")
    print(f"Loaded {len(v1_df)} v1 events")

    print("Loading v2 events...")
    v2_df = load_events_to_dataframe(str(v2_file), "v2")
    print(f"Loaded {len(v2_df)} v2 events (filtered for is_development=false)")

    # Combine DataFrames
    combined_df = pd.concat([v1_df, v2_df], ignore_index=True)
    print(f"Combined total events: {len(combined_df)}")

    # Add derived columns
    print("Adding derived columns...")
    combined_df = add_derived_columns(combined_df)

    # Define column order
    column_order = [
        "event",
        "timestamp",
        "distinct_id",
        "device_id",
        "country_code",
        "city",
        "region",
        "screen_height",
        "screen_width",
        "current_route",
        "platform",
        "app_version",
        "recipe_id",  # Add recipe_id column
        "device_type",
        "time_of_day",
        "day_of_week",
        "is_weekend",
    ]

    # Reorder columns
    combined_df = combined_df[column_order]

    # Save to CSV
    combined_df.to_csv(str(output_file), index=False)
    print(f"Combined CSV written to {output_file}")

    # Print summary stats
    print("\n📊 Summary:")
    print(f"Total events: {len(combined_df):,}")
    print(f"V1 events: {len(v1_df):,}")
    print(f"V2 events: {len(v2_df):,}")
    print("\nPlatform distribution:")
    print(combined_df["platform"].value_counts())
    print("\nDevice type distribution:")
    print(combined_df["device_type"].value_counts())


if __name__ == "__main__":
    main()
