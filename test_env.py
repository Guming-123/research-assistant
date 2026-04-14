#!/usr/bin/env python3
"""
Test script to verify .env file is loaded correctly
"""

import os
from pathlib import Path

# Load .env
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)
    print(f"✓ Loaded .env from: {env_path}")
except ImportError:
    print("✗ python-dotenv not installed")
    exit(1)

# Check environment variables
print("\nEnvironment Variables:")
print("-" * 50)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")
model = os.getenv("OPENAI_MODEL")

if api_key:
    print(f"✓ OPENAI_API_KEY: {api_key[:20]}...{api_key[-10:]}")
else:
    print("✗ OPENAI_API_KEY: NOT SET")

if base_url:
    print(f"✓ OPENAI_BASE_URL: {base_url}")
else:
    print("✗ OPENAI_BASE_URL: NOT SET")

if model:
    print(f"✓ OPENAI_MODEL: {model}")
else:
    print("✗ OPENAI_MODEL: NOT SET")

print("\n" + "-" * 50)
if api_key and base_url and model:
    print("✓ All required environment variables are set!")
else:
    print("✗ Some environment variables are missing")