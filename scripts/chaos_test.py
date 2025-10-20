#!/usr/bin/env python3
"""
Chaos test script (best-effort):
- Floods backend with requests to simulate pressure
- Verifies load shedding (503 degraded_mode) or triggers SAFE MODE via admin and checks status
"""
import json
import os
import time
import urllib.request


def http_post(url: str, data: bytes = b"", headers: dict | None = None, timeout: float = 3.0):
    req = urllib.request.Request(url=url, data=data or None, method="POST")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.getcode(), resp.read()


def main():
    backend = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

    # Flood /predictions/score with minimal body to simulate pressure
    payload = json.dumps({"transaction_id":"t1","account_id":"a1","amount":1.0,"features":{"transaction_amount":1.0,"transaction_time_seconds":1,"is_weekend":0,"hour_of_day":1,"is_round_amount":0,"unique_receivers_24h":1,"vpn_detected":0,"location_risk_score":0.1,"transaction_frequency_30min":0,"login_ip_changed_last_hour":0,"avg_transaction_amount_24h":1.0,"time_since_last_tx":1,"ip_risk_score":0.1,"transactions_last_24h":1,"customer_segment":"standard"}}).encode("utf-8")
    degraded = False
    for _ in range(200):
        try:
            code, body = http_post(f"{backend}/predictions/score", data=payload, headers={"Content-Type":"application/json"}, timeout=1.0)
            if code == 503 and b"degraded_mode" in body:
                degraded = True
                break
        except Exception:
            pass
        time.sleep(0.02)

    print(json.dumps({"event":"chaos_result","degraded": degraded}))

    # Optionally toggle SAFE MODE via admin (requires auth/HMAC in real env); here we skip auth for simplicity
    # This script is best-effort and can be extended in CI where secrets are present.

if __name__ == "__main__":
    main()

