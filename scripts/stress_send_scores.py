import argparse
import asyncio
import json
import os
import time
import uuid
from typing import Any, Dict

import numpy as np
import httpx

from backend.security.jwt_utils import generate_jwt
from backend.security.hmac_utils import sign_request_v2


# Defaults for local dev (base64 of 'test-secret' and 'hmac-secret')
os.environ.setdefault("JWT_SECRET_KEY", "dGVzdC1zZWNyZXQ=")
os.environ.setdefault("API_HMAC_KEY", "aG1hYy1zZWNyZXQ=")

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
PATH = "/predictions/score"


def make_features_row() -> dict:
    r: Dict[str, Any] = {}
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


def make_payload(i: int, *, vary_device: bool) -> dict:
    features = make_features_row()
    tx_id = f"txn_{i:08d}_{uuid.uuid4().hex[:6]}"
    acct = f"acc_{np.random.randint(1000, 9999)}"
    device_id = uuid.uuid4().hex[:12] if vary_device else "DEVICE123"
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


def sign_headers(body_bytes: bytes, *, device_id: str) -> dict:
    ts = str(int(time.time()))
    token = generate_jwt({"sub": uuid.uuid4().hex, "roles": ["admin", "service", "read"]}, expires_in=300)
    sig = sign_request_v2("POST", PATH, ts, body_bytes)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-Timestamp": ts,
        "X-Request-Signature": sig,
        "Idempotency-Key": uuid.uuid4().hex,
        "X-Device-ID": device_id,
        "Content-Type": "application/json; charset=utf-8",
    }
    return headers


async def send_one(i: int, client: httpx.AsyncClient, *, vary_device: bool, timeout: float) -> dict:
    payload = make_payload(i, vary_device=vary_device)
    body_bytes = json.dumps(payload).encode("utf-8")
    headers = sign_headers(body_bytes, device_id=str(payload.get("device_id", "DEVICE123")))
    url = f"{API_URL}{PATH}"
    try:
        t0 = time.perf_counter()
        r = await client.post(url, content=body_bytes, headers=headers, timeout=timeout)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        data = {"index": i, "status": r.status_code, "text": r.text, "duration_ms": round(dt_ms, 2)}
        try:
            j = r.json()
            data.update({
                "transaction_id": j.get("transaction_id"),
                "decision": j.get("decision"),
                "fraud_probability": j.get("fraud_probability"),
            })
        except Exception:
            pass
        return data
    except Exception as e:
        return {"index": i, "status": 0, "error": str(e)}


async def runner(total: int, concurrency: int, vary_device: bool, timeout: float, machine: bool) -> None:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict] = []
    start = time.perf_counter()
    async with httpx.AsyncClient() as client:
        async def _task(i: int):
            async with sem:
                res = await send_one(i, client, vary_device=vary_device, timeout=timeout)
                results.append(res)
                if machine:
                    print(json.dumps(res, ensure_ascii=False))
                return res

        await asyncio.gather(*(_task(i) for i in range(total)))

    elapsed = time.perf_counter() - start
    ok = sum(1 for r in results if r.get("status") == 200)
    fraud = sum(1 for r in results if r.get("decision") == "FRAUD")
    errors = sum(1 for r in results if r.get("status") != 200)
    rate = total / elapsed if elapsed > 0 else 0.0
    durs = [r.get("duration_ms") for r in results if isinstance(r.get("duration_ms"), (int, float))]
    durs_sorted = sorted(durs)
    def _pct(p):
        if not durs_sorted:
            return None
        k = max(0, min(len(durs_sorted)-1, int(len(durs_sorted)*p)))
        return round(durs_sorted[k], 2)
    summary = {
        "total": total,
        "ok": ok,
        "fraud": fraud,
        "errors": errors,
        "elapsed_sec": round(elapsed, 3),
        "req_per_sec": round(rate, 2),
        "latency_avg_ms": round(sum(durs)/len(durs), 2) if durs else None,
        "latency_p50_ms": _pct(0.50),
        "latency_p95_ms": _pct(0.95),
    }
    if machine:
        print(json.dumps({"summary": summary}, ensure_ascii=False))
    else:
        line = (
            f"Total: {total} | OK: {ok} | FRAUD: {fraud} | ERRORS: {errors} | "
            f"Elapsed: {summary['elapsed_sec']}s | RPS: {summary['req_per_sec']} | "
            f"p50: {summary['latency_p50_ms']} ms | p95: {summary['latency_p95_ms']} ms | avg: {summary['latency_avg_ms']} ms"
        )
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concurrent stress sender for fraud scoring")
    parser.add_argument("--total", type=int, default=500, help="Total number of requests")
    parser.add_argument("--concurrency", type=int, default=50, help="Max in-flight requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout (seconds)")
    parser.add_argument("--vary-devices", action="store_true", help="Vary X-Device-ID to avoid device rate limit")
    parser.add_argument("--machine", action="store_true", help="Print one JSON line per result and a summary JSON")
    args = parser.parse_args()

    np.random.seed(42)
    asyncio.run(runner(args.total, args.concurrency, args.vary_devices, args.timeout, args.machine))
