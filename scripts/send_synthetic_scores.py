import os
import json
import time
import uuid
import numpy as np
import requests

from backend.security.jwt_utils import generate_jwt
from backend.security.hmac_utils import sign_request_v2


# Secrets (base64 of 'test-secret' and 'hmac-secret')
os.environ.setdefault("JWT_SECRET_KEY", "dGVzdC1zZWNyZXQ=")
os.environ.setdefault("API_HMAC_KEY", "aG1hYy1zZWNyZXQ=")

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PATH = "/predictions/score"


def make_features_row() -> dict:
    r = {}
    r["transaction_amount"] = float(np.round(np.random.uniform(10, 2000), 2))
    r["transaction_time_seconds"] = int(np.random.randint(0, 30 * 24 * 3600))
    r["is_weekend"] = int(np.random.randint(0, 2))
    r["hour_of_day"] = int(np.random.randint(0, 24))
    r["location_risk_score"] = float(np.round(np.random.uniform(0.1, 0.9), 3))
    r["transaction_frequency_30min"] = int(np.random.randint(0, 10))
    r["login_ip_changed_last_hour"] = int(np.random.randint(0, 2))
    r["avg_transaction_amount_24h"] = float(np.round(np.random.uniform(50, 1500), 2))
    r["time_since_last_tx"] = int(np.random.randint(0, 86400))
    r["ip_risk_score"] = float(np.round(np.random.uniform(0, 1), 3))
    r["vpn_detected"] = int(np.random.randint(0, 2))
    r["transactions_last_24h"] = int(np.random.randint(0, 20))
    r["is_round_amount"] = 0 if r["transaction_amount"] % 1 else 1
    r["unique_receivers_24h"] = int(np.random.randint(0, 20))
    r["customer_segment"] = np.random.choice(["standard", "premium", "business"], p=[0.7, 0.2, 0.1])
    return r


def make_payload(i: int) -> dict:
    features = make_features_row()
    tx_id = f"txn_{i:04d}"
    acct = f"acc_{np.random.randint(1000, 9999)}"
    device_id = str(np.random.randint(100000, 999999))
    payload = {
        "transaction_id": tx_id,
        "account_id": acct,
        "amount": features["transaction_amount"],
        "currency": "EUR",
        "timestamp": "2025-01-01T00:00:00Z",
        "merchant_category": "grocery",
        "channel": "web",
        "device_id": device_id,
        "geolocation": {
            "lat": float(np.round(np.random.uniform(-90, 90), 4)),
            "lon": float(np.round(np.random.uniform(-180, 180), 4)),
        },
        "features": features,
    }
    return payload


def sign_headers(body_bytes: bytes) -> dict:
    ts = str(int(time.time()))
    token = generate_jwt({"sub": "tester", "roles": ["admin", "service", "read"]}, expires_in=300)
    sig = sign_request_v2("POST", PATH, ts, body_bytes)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-Timestamp": ts,
        "X-Request-Signature": sig,
        "Idempotency-Key": uuid.uuid4().hex,
        "X-Device-ID": "DEVICE123",
        "Content-Type": "application/json; charset=utf-8",
    }
    return headers


def send_one(i: int) -> None:
    payload = make_payload(i)
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = sign_headers(body_bytes)
    url = f"{API_URL}{PATH}"
    r = requests.post(url, data=body_bytes, headers=headers, timeout=20)
    print(f"[{i}] Status: {r.status_code} | {r.text[:300]}")


if __name__ == "__main__":
    np.random.seed(42)
    for i in range(3):
        try:
            send_one(i)
        except Exception as e:
            print(f"[{i}] Error: {e}")

