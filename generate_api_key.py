#!/usr/bin/env python3
"""
🔑 API Key Generator
Generate secure random API keys for your API
"""

import secrets
import string
import sys

def generate_api_key(length=32, format_type="urlsafe"):
    """
    Generate a secure random API key
    
    Args:
        length: Length of the key
        format_type: 'urlsafe', 'hex', or 'alphanumeric'
    
    Returns:
        str: Generated API key
    """
    if format_type == "urlsafe":
        # URL-safe base64 encoded (recommended)
        return secrets.token_urlsafe(length)
    
    elif format_type == "hex":
        # Hexadecimal format
        return secrets.token_hex(length)
    
    elif format_type == "alphanumeric":
        # Alphanumeric only
        alphabet = string.ascii_letters + string.digits
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    else:
        raise ValueError(f"Unknown format: {format_type}")

def main():
    print("=" * 60)
    print("🔑 API Key Generator")
    print("=" * 60)
    print()
    
    # Generate different types
    print("1️⃣  URL-Safe (Recommended for API keys):")
    urlsafe_key = generate_api_key(32, "urlsafe")
    print(f"   {urlsafe_key}")
    print()
    
    print("2️⃣  Hexadecimal:")
    hex_key = generate_api_key(32, "hex")
    print(f"   {hex_key}")
    print()
    
    print("3️⃣  Alphanumeric:")
    alphanum_key = generate_api_key(48, "alphanumeric")
    print(f"   {alphanum_key}")
    print()
    
    print("=" * 60)
    print("💡 How to use:")
    print("=" * 60)
    print()
    print("1. Copy one of the keys above")
    print("2. Open your .env file")
    print("3. Set: API_KEY=<your-key-here>")
    print("4. Set: ENABLE_AUTH=true")
    print()
    print("Example .env entry:")
    print(f"   API_KEY={urlsafe_key}")
    print(f"   ENABLE_AUTH=true")
    print()
    print("⚠️  Keep your API key secret! Do NOT share or commit to git.")
    print()

if __name__ == "__main__":
    main()