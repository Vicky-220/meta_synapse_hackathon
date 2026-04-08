#!/usr/bin/env python3
import argparse
import os
import sys
from threading import Thread
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Install it with 'pip install requests'.")
    sys.exit(1)

DEFAULT_HF_URL = "https://vicky0406-synapse-openenv.hf.space"
BASE_URL = os.environ.get("OPENENV_BASE_URL", DEFAULT_HF_URL)
SESSION = requests.Session()


def request(method, path, **kwargs):
    url = urljoin(BASE_URL, path)
    try:
        return SESSION.request(method, url, timeout=10, **kwargs), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


def check_step(name, method, path, json_data=None, expected_status=200):
    response, error = request(method, path, json=json_data)
    if error:
        print(f"[FAIL] {name}: {error}")
        return False
    if response.status_code != expected_status:
        print(f"[FAIL] {name}: expected {expected_status}, got {response.status_code}")
        return False
    print(f"[PASS] {name}")
    return True


def check_concurrent_health(workers=4):
    results = [False] * workers

    def worker(index):
        response, error = request("get", "/health")
        results[index] = (response is not None and response.status_code == 200)

    threads = [Thread(target=worker, args=(i,)) for i in range(workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    if all(results):
        print(f"[PASS] Concurrent health check ({workers} workers)")
        return True

    print(f"[FAIL] Concurrent health check ({workers} workers)")
    return False


def run_tests(verbose=False):
    print(f"CHECKING OPENENV DEPLOYMENT AT: {BASE_URL}")

    if not check_step("Health", "get", "/health"):
        return
    if not check_concurrent_health(4):
        return
    if not check_step("OpenAPI schema", "get", "/openapi.json"):
        return
    if not check_step("Reset", "post", "/reset", json_data={}):
        return
    if not check_step("State", "get", "/state"):
        return

    action = {
        "action": {
            "action_type": "ask_question",
            "question": "Do you have any pain when you urinate?"
        }
    }
    if not check_step("Step", "post", "/step", json_data=action):
        return
    if not check_step("State after step", "get", "/state"):
        return

    print("[PASS] All checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenEnv deployment checker")
    parser.add_argument("--url", help="Base URL for the OpenEnv deployment")
    args = parser.parse_args()
    if args.url:
        BASE_URL = args.url
    run_tests()
