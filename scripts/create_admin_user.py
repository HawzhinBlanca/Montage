#!/usr/bin/env python3
"""
Create initial admin user and API key for Montage system
Run this once after setting up authentication
"""

import sys
import os
from pathlib import Path

# Add src to path

from api.auth import api_key_auth, UserRole
from core.db import Database

def create_admin_user():
    """Create initial admin user with API key"""
    
    admin_user_id = "admin"
    admin_name = "System Administrator"
    
    print("🔐 Creating initial admin user...")
    
    try:
        # Generate admin API key
        api_key = api_key_auth.generate_api_key(
            user_id=admin_user_id,
            name=admin_name,
            role=UserRole.ADMIN
        )
        
        print("✅ Admin user created successfully!")
        print(f"👤 User ID: {admin_user_id}")
        print(f"🔑 API Key: {api_key}")
        print("")
        print("🚨 SECURITY WARNING:")
        print("1. Store this API key securely - it won't be shown again")
        print("2. Use this key to create additional users via the API")
        print("3. Add it to your .env file as ADMIN_API_KEY for testing")
        print("")
        print("📝 Usage examples:")
        print("# Create API key for new user:")
        print(f'curl -X POST "http://localhost:8000/auth/api-key" \\')
        print(f'  -H "Authorization: Bearer {api_key}" \\')
        print('  -H "Content-Type: application/json" \\')
        print('  -d \'{"name": "User API Key", "role": "user"}\'')
        print("")
        print("# Upload video:")
        print("curl -X POST \"http://localhost:8000/process\" \\")
        print(f'  -H "Authorization: Bearer {api_key}" \\')
        print('  -F "file=@video.mp4"')
        
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_admin_user()