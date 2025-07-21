#!/usr/bin/env python3
"""
Test client for Montage API
"""

import sys
import time
import requests
from pathlib import Path

API_BASE = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200


def test_process_video(video_path: str):
    """Test video processing"""
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return None

    print(f"\nUploading video: {video_path}")

    with open(video_path, "rb") as f:
        files = {"file": (Path(video_path).name, f, "video/mp4")}
        data = {"mode": "smart", "vertical": "false"}

        response = requests.post(f"{API_BASE}/process", files=files, data=data)

    if response.status_code != 200:
        print(f"Error: {response.status_code} - {response.text}")
        return None

    job_data = response.json()
    print(f"Job created: {job_data}")
    return job_data["job_id"]


def poll_job_status(job_id: str, max_wait: int = 300):
    """Poll job status until completion"""
    print(f"\nPolling job status for {job_id}...")

    start_time = time.time()
    while time.time() - start_time < max_wait:
        response = requests.get(f"{API_BASE}/status/{job_id}")

        if response.status_code != 200:
            print(f"Error getting status: {response.text}")
            return None

        status_data = response.json()
        status = status_data["status"]

        print(f"Status: {status}")

        if status == "completed":
            print(f"Job completed! Download URL: {status_data.get('download_url')}")
            return status_data
        elif status == "failed":
            print(f"Job failed: {status_data.get('error')}")
            return None

        time.sleep(5)

    print("Timeout waiting for job completion")
    return None


def download_result(job_id: str, output_path: str = None):
    """Download processed video"""
    if not output_path:
        output_path = f"downloaded_{job_id}.mp4"

    print(f"\nDownloading result to {output_path}...")

    response = requests.get(f"{API_BASE}/download/{job_id}", stream=True)

    if response.status_code != 200:
        print(f"Error downloading: {response.status_code} - {response.text}")
        return False

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded successfully: {output_path}")
    return True


def main():
    """Main test flow"""
    if len(sys.argv) < 2:
        print("Usage: python test_api_client.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Test health
    if not test_health():
        print("API health check failed!")
        sys.exit(1)

    # Process video
    job_id = test_process_video(video_path)
    if not job_id:
        print("Failed to create job!")
        sys.exit(1)

    # Poll status
    result = poll_job_status(job_id)
    if not result:
        print("Job failed or timed out!")
        sys.exit(1)

    # Download result
    if not download_result(job_id):
        print("Failed to download result!")
        sys.exit(1)

    # Get metrics
    print("\nFetching system metrics...")
    response = requests.get(f"{API_BASE}/metrics")
    if response.status_code == 200:
        print(f"Metrics: {response.json()}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
