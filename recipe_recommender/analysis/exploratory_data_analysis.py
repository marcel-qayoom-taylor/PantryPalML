"""
Exploratory Data Analysis for PantryPal Recipe Recommendation System

This script analyzes user interaction patterns, recipe engagement, and temporal trends
to inform feature engineering and model development for the recommendation system.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set up plotting
plt.style.use("default")
sns.set_palette("husl")


def load_data(file_path):
    """Load and basic preprocessing of the combined events CSV."""
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)

    # Convert timestamp to datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["year"] = df["datetime"].dt.year

    print(f"Loaded {len(df):,} events from {df['distinct_id'].nunique():,} users")
    return df


def analyze_basic_stats(df):
    """Generate basic statistics about the dataset."""
    print("\n" + "=" * 60)
    print("BASIC DATASET STATISTICS")
    print("=" * 60)

    print(f"📊 Total Events: {len(df):,}")
    print(f"👥 Unique Users: {df['distinct_id'].nunique():,}")
    print(f"📱 Unique Devices: {df['device_id'].nunique():,}")
    print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"🕒 Time Span: {(df['date'].max() - df['date'].min()).days} days")

    # Events per user
    events_per_user = df.groupby("distinct_id").size()
    print("\n📈 Events per User:")
    print(f"   Mean: {events_per_user.mean():.1f}")
    print(f"   Median: {events_per_user.median():.1f}")
    print(f"   Min: {events_per_user.min()}")
    print(f"   Max: {events_per_user.max()}")

    return events_per_user


def analyze_recipe_interactions(df):
    """Analyze recipe-specific interaction patterns."""
    print("\n" + "=" * 60)
    print("RECIPE INTERACTION ANALYSIS")
    print("=" * 60)

    # Define recipe-related events
    recipe_events = [
        "Recipe Viewed",
        "Recipe Cooked",
        "Recipe Favourited",
        "Recipe Link Clicked",
        "Recipe Added To Collections",
        "Recipe Removed From Collections",
        "Recipe Cook Started",
        "Recipe Search Queried",
        "Recipe Search Filter Applied",
        "Recipe Filter Applied",
    ]

    recipe_df = df[df["event"].isin(recipe_events)].copy()
    print(f"🍳 Total Recipe Interactions: {len(recipe_df):,}")
    print(f"👥 Users with Recipe Interactions: {recipe_df['distinct_id'].nunique():,}")

    # Recipe interaction distribution
    recipe_event_counts = recipe_df["event"].value_counts()
    print("\n📊 Recipe Event Distribution:")
    for event, count in recipe_event_counts.items():
        percentage = (count / len(recipe_df)) * 100
        print(f"   {event}: {count:,} ({percentage:.1f}%)")

    # Users by recipe engagement level
    user_recipe_counts = recipe_df.groupby("distinct_id").size()
    print("\n👥 Recipe Interactions per User:")
    print(f"   Mean: {user_recipe_counts.mean():.1f}")
    print(f"   Median: {user_recipe_counts.median():.1f}")
    print(f"   Users with 1 interaction: {(user_recipe_counts == 1).sum():,}")
    print(
        f"   Users with 2-5 interactions: {((user_recipe_counts >= 2) & (user_recipe_counts <= 5)).sum():,}"
    )
    print(
        f"   Users with 6-10 interactions: {((user_recipe_counts >= 6) & (user_recipe_counts <= 10)).sum():,}"
    )
    print(f"   Users with 10+ interactions: {(user_recipe_counts > 10).sum():,}")

    return recipe_df, user_recipe_counts


def analyze_temporal_patterns(df):
    """Analyze temporal usage patterns."""
    print("\n" + "=" * 60)
    print("TEMPORAL PATTERN ANALYSIS")
    print("=" * 60)

    # Daily activity patterns
    daily_events = df.groupby("date").size()
    print("📅 Daily Activity:")
    print(f"   Mean events per day: {daily_events.mean():.1f}")
    print(f"   Peak day: {daily_events.max():,} events on {daily_events.idxmax()}")
    print(f"   Quietest day: {daily_events.min():,} events on {daily_events.idxmin()}")

    # Hourly patterns
    hourly_events = df.groupby("hour").size()
    peak_hours = hourly_events.nlargest(3)
    print("\n🕐 Hourly Patterns:")
    print(
        f"   Peak hours: {', '.join([f'{h}:00 ({count:,} events)' for h, count in peak_hours.items()])}"
    )

    # Day of week patterns
    dow_events = df.groupby("day_of_week").size()
    print("\n📆 Day of Week Patterns:")
    for day, count in dow_events.items():
        percentage = (count / len(df)) * 100
        print(f"   {day}: {count:,} events ({percentage:.1f}%)")

    # Weekend vs Weekday
    weekend_events = df[df["is_weekend"]]
    weekday_events = df[not df["is_weekend"]]
    print("\n🗓️  Weekend vs Weekday:")
    print(
        f"   Weekend: {len(weekend_events):,} events ({len(weekend_events) / len(df) * 100:.1f}%)"
    )
    print(
        f"   Weekday: {len(weekday_events):,} events ({len(weekday_events) / len(df) * 100:.1f}%)"
    )

    return daily_events, hourly_events


def analyze_user_segments(df, events_per_user):
    """Segment users by activity level."""
    print("\n" + "=" * 60)
    print("USER SEGMENTATION ANALYSIS")
    print("=" * 60)

    # Define user segments based on activity
    def categorize_user(event_count):
        if event_count == 1:
            return "One-time User"
        if event_count <= 5:
            return "Light User"
        if event_count <= 20:
            return "Moderate User"
        if event_count <= 50:
            return "Active User"
        return "Power User"

    user_segments = events_per_user.apply(categorize_user)
    segment_counts = user_segments.value_counts()

    print("👥 User Segments:")
    for segment, count in segment_counts.items():
        percentage = (count / len(events_per_user)) * 100
        avg_events = events_per_user[user_segments == segment].mean()
        print(
            f"   {segment}: {count:,} users ({percentage:.1f}%) - Avg: {avg_events:.1f} events"
        )

    return user_segments


def analyze_ingredient_interactions(df):
    """Analyze pantry and grocery interactions for ingredient preferences."""
    print("\n" + "=" * 60)
    print("INGREDIENT INTERACTION ANALYSIS")
    print("=" * 60)

    ingredient_events = [
        "Pantry Item Added",
        "Pantry Item Removed",
        "Pantry Items Added",
        "Grocery Item Added",
        "Grocery Item Removed",
        "Grocery Items Added",
        "Custom Ingredient Added",
        "Custom Ingredients Added",
    ]

    ingredient_df = df[df["event"].isin(ingredient_events)].copy()
    print(f"🥬 Total Ingredient Interactions: {len(ingredient_df):,}")
    print(
        f"👥 Users with Ingredient Interactions: {ingredient_df['distinct_id'].nunique():,}"
    )

    # Ingredient interaction distribution
    ingredient_event_counts = ingredient_df["event"].value_counts()
    print("\n📊 Ingredient Event Distribution:")
    for event, count in ingredient_event_counts.items():
        percentage = (count / len(ingredient_df)) * 100
        print(f"   {event}: {count:,} ({percentage:.1f}%)")

    return ingredient_df


def create_user_activity_summary(df):
    """Create a comprehensive user activity summary for modeling."""
    print("\n" + "=" * 60)
    print("USER ACTIVITY SUMMARY FOR MODELING")
    print("=" * 60)

    user_summary = (
        df.groupby("distinct_id")
        .agg(
            {
                "event": "count",
                "datetime": ["min", "max"],
                "day_of_week": lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                ),
                "time_of_day": lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                ),
                "device_type": lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                ),
                "platform": lambda x: (
                    x.mode().iloc[0] if len(x.mode()) > 0 else "Unknown"
                ),
            }
        )
        .round(2)
    )

    user_summary.columns = [
        "total_events",
        "first_seen",
        "last_seen",
        "preferred_day",
        "preferred_time",
        "device_type",
        "platform",
    ]

    # Calculate user lifespan
    user_summary["days_active"] = (
        user_summary["last_seen"] - user_summary["first_seen"]
    ).dt.days + 1
    user_summary["events_per_day"] = (
        user_summary["total_events"] / user_summary["days_active"]
    )

    print(f"📊 Created user summary with {len(user_summary)} users")
    print(
        f"📈 Average events per user per day: {user_summary['events_per_day'].mean():.2f}"
    )

    return user_summary


def generate_insights_report(df, recipe_df, user_summary):
    """Generate key insights for model development."""
    print("\n" + "=" * 60)
    print("KEY INSIGHTS FOR RECOMMENDATION SYSTEM")
    print("=" * 60)

    # Recipe engagement insights
    recipe_users = recipe_df["distinct_id"].nunique()
    total_users = df["distinct_id"].nunique()
    recipe_engagement_rate = recipe_users / total_users * 100

    print("🎯 Recipe Engagement:")
    print(f"   {recipe_engagement_rate:.1f}% of users have recipe interactions")
    print(
        f"   {total_users - recipe_users:,} users have no recipe interactions (cold start problem)"
    )

    # Conversion funnel
    recipe_events = ["Recipe Viewed", "Recipe Favourited", "Recipe Cooked"]
    funnel = {}
    for event in recipe_events:
        users_with_event = recipe_df[recipe_df["event"] == event][
            "distinct_id"
        ].nunique()
        funnel[event] = users_with_event

    print("\n🔄 Recipe Conversion Funnel:")
    for event, users in funnel.items():
        percentage = users / recipe_users * 100
        print(f"   {event}: {users:,} users ({percentage:.1f}% of recipe users)")

    # Data sparsity analysis
    total_possible_interactions = recipe_users * recipe_df["event"].nunique()
    actual_interactions = len(recipe_df)
    sparsity = (1 - actual_interactions / total_possible_interactions) * 100

    print("\n📊 Data Characteristics:")
    print(f"   Recipe interaction sparsity: {sparsity:.2f}%")
    print(
        f"   Average recipe interactions per engaged user: {len(recipe_df) / recipe_users:.1f}"
    )

    # Recommendation strategy insights
    active_recipe_users = user_summary[user_summary["total_events"] >= 5].index
    active_with_recipes = recipe_df[recipe_df["distinct_id"].isin(active_recipe_users)][
        "distinct_id"
    ].nunique()

    print("\n🚀 Recommendation Strategy Insights:")
    print(f"   Active users (5+ events): {len(active_recipe_users):,}")
    print(f"   Active users with recipe interactions: {active_with_recipes:,}")
    print(
        f"   Users suitable for personalized recommendations: {active_with_recipes:,}"
    )
    print(
        f"   Users needing popularity-based recommendations: {total_users - active_with_recipes:,}"
    )


def main():
    """Run complete exploratory data analysis."""
    # Load data
    data_path = Path(__file__).parent.parent / "output" / "combined_events.csv"
    df = load_data(data_path)

    # Run analyses
    events_per_user = analyze_basic_stats(df)
    recipe_df, user_recipe_counts = analyze_recipe_interactions(df)
    daily_events, hourly_events = analyze_temporal_patterns(df)
    analyze_user_segments(df, events_per_user)
    analyze_ingredient_interactions(df)
    user_summary = create_user_activity_summary(df)
    generate_insights_report(df, recipe_df, user_summary)

    # Save user summary for later use
    output_path = Path(__file__).parent.parent / "output" / "user_activity_summary.csv"
    user_summary.to_csv(output_path)
    print(f"\n💾 Saved user activity summary to {output_path}")

    print("\n" + "=" * 60)
    print("✅ EXPLORATORY DATA ANALYSIS COMPLETE")
    print("=" * 60)
    print("\nNext Steps:")
    print("1. Use insights to design user-recipe interaction matrix")
    print("2. Engineer features based on temporal and behavioral patterns")
    print("3. Design appropriate train/test splits accounting for temporal nature")
    print("4. Consider hybrid approach: collaborative filtering + content-based")


if __name__ == "__main__":
    main()
