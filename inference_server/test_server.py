#!/usr/bin/env python3
"""
Test script for Grounding DINO Inference Server

Tests all endpoints including Prometheus metrics.
"""

import requests
import sys
from pathlib import Path


def test_endpoint(name, method, url, **kwargs):
    """Test an endpoint and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    try:
        if method == "GET":
            response = requests.get(url, **kwargs)
        elif method == "POST":
            response = requests.post(url, **kwargs)
        else:
            print(f"Unknown method: {method}")
            return False

        print(f"Status Code: {response.status_code}")

        if response.headers.get('content-type', '').startswith('application/json'):
            print(f"Response JSON:")
            import json
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"Response (first 500 chars):")
            print(response.text[:500])

        return response.status_code < 400

    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    base_url = "http://localhost:6061"

    if len(sys.argv) > 1:
        base_url = sys.argv[1]

    print(f"Testing server at: {base_url}")

    # Test 1: Root endpoint
    test_endpoint(
        "Root Endpoint",
        "GET",
        f"{base_url}/"
    )

    # Test 2: Health check
    test_endpoint(
        "Health Check",
        "GET",
        f"{base_url}/health"
    )

    # Test 3: Configuration
    test_endpoint(
        "Configuration",
        "GET",
        f"{base_url}/config"
    )

    # Test 4: Statistics
    test_endpoint(
        "Statistics",
        "GET",
        f"{base_url}/stats"
    )

    # Test 5: Prometheus Metrics
    test_endpoint(
        "Prometheus Metrics",
        "GET",
        f"{base_url}/metrics"
    )

    # Test 6: Detection (if image provided)
    if len(sys.argv) > 2:
        image_path = Path(sys.argv[2])
        if image_path.exists():
            print(f"\nTesting detection with image: {image_path}")
            with open(image_path, 'rb') as f:
                test_endpoint(
                    "Bird Detection",
                    "POST",
                    f"{base_url}/detect",
                    files={'image': f},
                    data={'device_id': 'test-device'}
                )
        else:
            print(f"\nImage not found: {image_path}")
    else:
        print("\nSkipping detection test (no image provided)")
        print("Usage: python test_server.py [base_url] [image_path]")

    print(f"\n{'='*60}")
    print("Testing complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
