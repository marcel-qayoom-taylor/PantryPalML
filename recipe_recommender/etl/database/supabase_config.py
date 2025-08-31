#!/usr/bin/env python3
"""
Supabase Configuration and Client Setup

This module handles Supabase connection and authentication.
Set up your environment variables or create a .env file with:

SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here
"""

import os

from dotenv import load_dotenv
from supabase import Client, create_client

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
            print("✅ Supabase connection successful!")
            return True
        except Exception as e:
            print(f"❌ Supabase connection failed: {e}")
            return False

    @staticmethod
    def setup_instructions():
        """Print setup instructions for the user."""
        print("🔧 SUPABASE SETUP INSTRUCTIONS")
        print("=" * 50)
        print("1. Go to your Supabase project dashboard")
        print("2. Navigate to Settings > API")
        print("3. Copy the following values:")
        print("   - Project URL")
        print("   - anon/public key")
        print("   - service_role key (optional, for admin access)")
        print()
        print("4. Set environment variables or create a .env file:")
        print("   SUPABASE_URL=https://your-project-id.supabase.co")
        print("   SUPABASE_ANON_KEY=your-anon-key-here")
        print("   SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here")
        print()
        print("5. Test the connection by running:")
        print("   python ml_etl/database/supabase_config.py")


def main():
    """Test the Supabase configuration."""
    try:
        config = SupabaseConfig()
        print("🔧 Testing Supabase configuration...")
        config.test_connection()
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        print()
        SupabaseConfig.setup_instructions()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    main()
