#!/usr/bin/env python3
"""
Common utilities for the ML pipeline.

This module contains shared functionality to reduce code duplication
and make the codebase easier to maintain for beginners.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up consistent logging across all modules.

    This replaces print statements with proper logging
    that can be controlled, filtered, and saved to files.

    Args:
        name: Name of the logger (usually __name__)
        level: Logging level (INFO, DEBUG, WARNING, ERROR)

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:  # Only set up once
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


def configure_logging(level: int = logging.INFO) -> None:
    """
    Configure the root logger once with a consistent formatter and level.

    Use this in CLI entrypoints to control verbosity; library code should
    only call setup_logging(__name__).

    Args:
        level: Logging level for root (e.g., logging.INFO or logging.DEBUG)
    """
    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.setLevel(level)
        for handler in root_logger.handlers:
            handler.setLevel(level)
        return

    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
    root_logger.setLevel(level)


def load_json_file(file_path: Path) -> dict[str, Any] | None:
    """
    Safely load a JSON file with error handling.

    Args:
        file_path: Path to the JSON file

    Returns:
        Dict with file contents, or None if failed
    """
    try:
        with open(file_path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.getLogger(__name__).error(f"Failed to load JSON from {file_path}: {e}")
        return None


def save_json_file(data: dict[str, Any], file_path: Path) -> bool:
    """
    Safely save data to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Where to save the file

    Returns:
        True if successful
    """
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2, default=str)  # default=str handles datetime
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save JSON to {file_path}: {e}")
        return False


def safe_load_csv(file_path: Path, **kwargs) -> pd.DataFrame | None:
    """
    Safely load a CSV file with error handling.

    This function loads CSV files and handles common
    errors like missing files or bad data gracefully.

    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments for pd.read_csv

    Returns:
        DataFrame or None if failed
    """
    try:
        if not file_path.exists():
            logging.getLogger(__name__).error(f"CSV file not found: {file_path}")
            return None

        df = pd.read_csv(file_path, **kwargs)
        logging.getLogger(__name__).info(
            f"Loaded CSV with {len(df):,} rows from {file_path.name}"
        )
        return df

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to load CSV from {file_path}: {e}")
        return None


def safe_save_csv(df: pd.DataFrame, file_path: Path, **kwargs) -> bool:
    """
    Safely save a DataFrame to CSV.

    Args:
        df: DataFrame to save
        file_path: Where to save the file
        **kwargs: Additional arguments for df.to_csv

    Returns:
        True if successful
    """
    try:
        # Create parent directories if they don't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(file_path, index=False, **kwargs)
        logging.getLogger(__name__).info(
            f"Saved CSV with {len(df):,} rows to {file_path.name}"
        )
        return True

    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save CSV to {file_path}: {e}")
        return False


def validate_dataframe(
    df: pd.DataFrame, required_columns: list, name: str = "DataFrame"
) -> bool:
    """
    Validate that a DataFrame has the required structure.

    This checks that your data has all the columns
    you expect before trying to use it in ML.

    Args:
        df: DataFrame to validate
        required_columns: List of column names that must exist
        name: Name for logging (e.g., "training data")

    Returns:
        True if valid
    """
    logger = logging.getLogger(__name__)

    if df is None or df.empty:
        logger.error(f"{name} is empty or None")
        return False

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logger.error(f"{name} missing required columns: {missing_columns}")
        return False

    logger.info(
        f"{name} validation passed: {len(df):,} rows, {len(df.columns)} columns"
    )
    return True


def create_timestamp_suffix() -> str:
    """
    Create a timestamp suffix for file names.

    Returns:
        Timestamp string like "20240301_143022"
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_number(num: float, decimals: int = 2) -> str:
    """
    Format a number for display with thousand separators.

    Args:
        num: Number to format
        decimals: Number of decimal places

    Returns:
        Formatted string like "1,234.56"
    """
    return f"{num:,.{decimals}f}"


def calculate_positive_ratio(df: pd.DataFrame, label_column: str = "label") -> float:
    """
    Calculate the ratio of positive samples in a dataset.

    This tells you how balanced your dataset is.
    If the ratio is very low (like 0.01), most samples are negative.

    Args:
        df: DataFrame with labels
        label_column: Name of the column with 0/1 labels

    Returns:
        Ratio of positive samples (0.0 to 1.0)
    """
    if label_column not in df.columns:
        logging.getLogger(__name__).warning(f"Label column '{label_column}' not found")
        return 0.0

    return df[label_column].mean()


def print_dataset_summary(df: pd.DataFrame, name: str = "Dataset") -> None:
    """
    Print a summary of a dataset for debugging.

    This gives you a quick overview of your data
    to spot potential issues.

    Args:
        df: DataFrame to summarize
        name: Name for the summary
    """
    logger = logging.getLogger(__name__)

    logger.info(f"{name} summary:")
    logger.info(f"   Shape: {df.shape}")
    logger.info(f"   Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

    # Show data types
    type_counts = df.dtypes.value_counts()
    logger.info(f"   Data types: {dict(type_counts)}")

    # Show missing values
    missing = df.isna().sum()
    if missing.sum() > 0:
        logger.info(f"   Missing values: {missing[missing > 0].to_dict()}")
    else:
        logger.info("   No missing values")

    # If there's a label column, show class distribution
    if "label" in df.columns:
        pos_ratio = calculate_positive_ratio(df)
        logger.info(f"   Positive ratio: {pos_ratio:.3f} ({pos_ratio * 100:.1f}%)")
