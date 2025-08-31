"""
Helper functions for event data processing and feature engineering.
"""

from datetime import datetime

import pandas as pd


def get_device_type(screen_height, screen_width):
    """
    Derive device type based on screen dimensions.

    Args:
        screen_height: Screen height in pixels
        screen_width: Screen width in pixels

    Returns:
        str: 'mobile', 'tablet', or 'unknown'
    """
    if pd.isna(screen_height) or pd.isna(screen_width):
        return "unknown"

    try:
        height = int(screen_height)
        width = int(screen_width)
        # Use the larger dimension as the determining factor
        larger_dimension = max(height, width)

        # Common breakpoints for device classification
        if larger_dimension >= 1024:  # iPad and larger tablets typically 1024px+
            return "tablet"
        if larger_dimension >= 568:  # iPhone and most mobile devices
            return "mobile"
        return "mobile"  # Default smaller screens to mobile
    except (ValueError, TypeError):
        return "unknown"


def get_time_of_day(timestamp):
    """
    Categorize timestamp into time of day periods.

    Args:
        timestamp: Unix timestamp

    Returns:
        str: 'morning', 'afternoon', 'evening', 'night', or 'unknown'
    """
    if pd.isna(timestamp):
        return "unknown"

    try:
        dt = datetime.fromtimestamp(int(timestamp))
        hour = dt.hour

        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 17:
            return "afternoon"
        if 17 <= hour < 21:
            return "evening"
        return "night"
    except (ValueError, TypeError, OSError):
        return "unknown"


def get_day_of_week(timestamp):
    """
    Get day of week from timestamp.

    Args:
        timestamp: Unix timestamp

    Returns:
        str: Day name (e.g., 'Monday') or 'unknown'
    """
    if pd.isna(timestamp):
        return "unknown"

    try:
        dt = datetime.fromtimestamp(int(timestamp))
        return dt.strftime("%A")
    except (ValueError, TypeError, OSError):
        return "unknown"


def get_is_weekend(timestamp):
    """
    Check if timestamp falls on a weekend.

    Args:
        timestamp: Unix timestamp

    Returns:
        bool: True if weekend, False if weekday, None if unknown
    """
    if pd.isna(timestamp):
        return None

    try:
        dt = datetime.fromtimestamp(int(timestamp))
        # weekday() returns 0-6 where Monday is 0, Sunday is 6
        return dt.weekday() >= 5  # Saturday (5) or Sunday (6)
    except (ValueError, TypeError, OSError):
        return None


def get_platform_from_props(props):
    """
    Extract platform from v2 event properties.

    Args:
        props: Event properties dictionary

    Returns:
        str: Platform identifier ('ios', 'android', 'web', etc.)
    """
    platform = props.get("platform") or props.get("mp_lib", "unknown")

    if platform == "react-native":
        # For react-native, check OS to determine platform
        os_info = props.get("$os", "").lower()
        if "ios" in os_info:
            return "ios"
        if "android" in os_info:
            return "android"
        return "mobile"

    return platform


def clean_event_name(event_name):
    """
    Clean up technical event names to more readable versions.

    Args:
        event_name: Raw event name from tracking

    Returns:
        str: Cleaned, human-readable event name
    """
    if pd.isna(event_name) or not event_name:
        return "Unknown Event"

    # Event name mapping from technical to readable names
    event_mapping = {
        "$ae_first_open": "First Open",
        "$ae_session": "Session",
        "$identify": "User Identified",
    }

    # Return mapped name or original if no mapping exists
    return event_mapping.get(event_name, event_name)
