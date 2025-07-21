#!/usr/bin/env python3
"""Test Gemini API directly"""

import os
import google.generativeai as genai

# Get API key
api_key = os.getenv("GEMINI_API_KEY", "***REMOVED***")
print(f"Testing Gemini API with key: {api_key[:8]}...")

try:
    # Configure Gemini
    genai.configure(api_key=api_key)

    # Create model
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Test with simple text
    test_text = "Welcome to the program. We are very, very excited to have Professor David Sinclair, a well-renowned geneticist."

    prompt = f"""Analyze this transcript segment for highlight potential:

"{test_text}"

Rate the highlight potential (1-10) and provide a brief title (max 40 chars). 
Respond in JSON format:
{{"score": 8, "title": "Professor David Sinclair Introduction"}}"""

    print("Sending request to Gemini...")
    response = model.generate_content(prompt)

    print("Response received!")
    print(f"Response: {response.text}")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
