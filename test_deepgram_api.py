#!/usr/bin/env python3
"""Test Deepgram API directly"""

import os
import sys
from deepgram import DeepgramClient, PrerecordedOptions

# Get API key
api_key = os.getenv("DEEPGRAM_API_KEY")
if not api_key:
    print("DEEPGRAM_API_KEY not set")
    sys.exit(1)

print(f"Testing Deepgram API with key: {api_key[:8]}...")

try:
    # Initialize client
    client = DeepgramClient(api_key)

    # Test with the 1-minute video
    with open("test_video_1min.mp4", "rb") as audio:
        options = PrerecordedOptions(
            model="nova-2",
            smart_format=True,
            punctuate=True,
            paragraphs=True,
            utterances=True,
            diarize=True,
        )

        print("Sending request to Deepgram...")
        response = client.listen.rest.v("1").transcribe_file({"buffer": audio}, options)

        print("Response received!")
        if hasattr(response, "results"):
            print(f"Transcript preview: {str(response.results)[:200]}...")

except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
