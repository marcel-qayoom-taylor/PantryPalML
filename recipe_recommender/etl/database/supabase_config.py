#!/usr/bin/env python3
"""
Supabase Configuration and Client Setup

This module handles Supabase connection and authentication. See README.md
"Supabase Setup" section for environment variables and instructions.
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client
from recipe_recommender.utils import setup_logging

logger = setup_logging(__name__)

# Load environment variables from .env file
load_dotenv()


class SupabaseConfig:
    """Configuration manager for Supabase connection."""

    def __init__(self):
        self.url = os.getenv("SUPABASE_URL")
        self.anon_key = os.getenv("SUPABASE_ANON_KEY")
        self.service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

        # Validate required configuration
        if not self.url:
            msg = "SUPABASE_URL environment variable is required"
            raise ValueError(msg)
        if not self.anon_key:
            msg = "SUPABASE_ANON_KEY environment variable is required"
            raise ValueError(msg)

    def get_client(self, use_service_role: bool = False) -> Client:
        """
        Create and return a Supabase client.

        Args:
            use_service_role: If True, uses service role key (admin access)
                            If False, uses anon key (row-level security applies)
        """
        key = self.service_role_key if use_service_role else self.anon_key

        if use_service_role and not self.service_role_key:
            msg = "SUPABASE_SERVICE_ROLE_KEY is required for admin access"
            raise ValueError(msg)

        return create_client(self.url, key)

    def test_connection(self) -> bool:
        """Test the Supabase connection."""
        try:
            client = self.get_client()
            # Try to access a basic endpoint
            (client.table("recipe").select("count", count="exact").limit(1).execute())
            logger.info("Supabase connection successful")
            return True
        except Exception:
            logger.exception("Supabase connection failed")
            return False

    @staticmethod
    def setup_instructions():
        """Deprecated: See README.md for setup steps."""
        logger.info(
            "See README.md 'Supabase Setup' for environment variable instructions"
        )


def main():
    """Test the Supabase configuration."""
    try:
        config = SupabaseConfig()
        logger.info("Testing Supabase configuration")
        config.test_connection()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        SupabaseConfig.setup_instructions()
    except Exception:
        logger.exception("Unexpected error during Supabase configuration test")


if __name__ == "__main__":
    main()
