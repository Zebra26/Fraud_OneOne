#!/usr/bin/env python3
"""
Observability validation script.

Performs:
1) Metrics reachability for backend and inference and checks for expected series
2) Triggers Safe Mode and verifies Prometheus alert firing/clearing
3) Queries Loki for logs with a known correlation_id across backend and inference
4) Emits a JSON pass/fail summary and exits non-zero on failure (CI-friendly)

Environment variables (with defaults):
- BACKEND_URL (http://localhost:8000)
- INFERENCE_URL (http://localhost:8080)
- PROMETHEUS_URL (http://localhost:9090)
- LOKI_URL (http://localhost:3100)
- JWT_TOKEN (optional pre-generated). If not set, JWT is generated from:
  - JWT_SECRET_KEY (required to generate)
  - JWT_ALGORITHM (HS256)
  - JWT_ISSUER (fraud_k)
  - JWT_AUDIENCE (fraud_api)
- API_HMAC_KEY (required) — base64 or raw

Usage:
  python scripts/validate_observability.py
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def generate_jwt_from_env() -> str:
    token = os.getenv("JWT_TOKEN")
    if token:
        return token
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError("JWT_SECRET_KEY or JWT_TOKEN is required")
    try:
        key = base64.b64decode(secret)
    except Exception:
        key = secret.encode("utf-8")
    alg = os.getenv("JWT_ALGORITHM", "HS256")
    iss = os.getenv("JWT_ISSUER", "fraud_k")
    aud = os.getenv("JWT_AUDIENCE", "fraud_api")
    now = int(time.time())
    header = {"alg": alg, "typ": "JWT"}
    payload = {
        "iss": iss,
        "aud": aud,
        "iat": now,
        "exp": now + 3600,
        "sub": "obs-check",
        "roles": ["admin", "service", "read"],
    }
    header_b64 = _b64url(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64url(json.dumps(payload, separators=(",", ":")).encode())
    to_sign = f"{header_b64}.{payload_b64}".encode()
    sig = hmac.new(key, to_sign, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url(sig)}"


def hmac_signature(ts: str, body: bytes, method: str, path: str) -> str:
    raw = os.getenv("API_HMAC_KEY")
    if not raw:
        raise RuntimeError("API_HMAC_KEY is required")
    try:
        key = base64.b64decode(raw)
    except Exception:
        key = raw.encode("utf-8")
    data = f'{method}|{path}|{ts}|' .encode() + (body or b'')
    mac = hmac.new(key, data, hashlib.sha256).digest()
    return base64.b64encode(mac).decode()


def http_request(url: str, method: str = "GET", headers: Dict[str, str] | None = None, body: bytes | None = None, timeout: int = 10) -> Tuple[int, bytes]:
    req = urllib.request.Request(url=url, method=method)
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    data = body if body is not None else None
    with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
        return resp.getcode(), resp.read()


def fetch_metrics(base_url: str, token: str) -> str:
    path = '/metrics'
    url = base_url.rstrip("/") + path
    ts = str(int(time.time()))
    body = b""
    sig = hmac_signature(ts, body, "GET", path)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-Timestamp": ts,
        "X-Request-Signature": sig,
    }
    code, content = http_request(url, headers=headers)
    if code != 200:
        raise RuntimeError(f"metrics fetch failed: {code}")
    return content.decode("utf-8", errors="ignore")


def assert_metrics_series(metrics_text: str, expected: List[str]) -> List[str]:
    missing = []
    for name in expected:
        if name not in metrics_text:
            missing.append(name)
    return missing


def prometheus_alerts(prom_url: str) -> List[Dict]:
    url = prom_url.rstrip("/") + "/api/v1/alerts"
    code, content = http_request(url)
    if code != 200:
        raise RuntimeError(f"prom alerts failed: {code}")
    payload = json.loads(content)
    return payload.get("data", {}).get("alerts", [])


def alert_present(alerts: List[Dict], name: str) -> bool:
    for a in alerts:
        if a.get("labels", {}).get("alertname") == name:
            if a.get("state") in {"firing", "active"}:
                return True
    return False


def toggle_safe_mode(backend_url: str, token: str, enable: bool, reason: str = "obs-check") -> None:
    path = "/admin/safe-mode/enable" if enable else "/admin/safe-mode/disable"
    url = backend_url.rstrip("/") + path
    if enable:
        body_dict = {"reason": reason}
        body = json.dumps(body_dict).encode()
    else:
        body = b""
    ts = str(int(time.time()))
    sig = hmac_signature(ts, body, "POST", path)
    headers = {
        "Authorization": f"Bearer {token}",
        "X-Request-Timestamp": ts,
        "X-Request-Signature": sig,
        "Content-Type": "application/json",
    }
    code, content = http_request(url, method="POST", headers=headers, body=(body if enable else None))
    if code != 200:
        raise RuntimeError(f"safe mode toggle failed: {code} {content[:200]}")


def loki_query_range(loki_url: str, query: str, start_ns: int, end_ns: int, limit: int = 50) -> Dict:
    params = {
        "query": query,
        "start": str(start_ns),
        "end": str(end_ns),
        "limit": str(limit),
        "direction": "backward",
    }
    url = loki_url.rstrip("/") + "/loki/api/v1/query_range?" + urllib.parse.urlencode(params)
    code, content = http_request(url)
    if code != 200:
        raise RuntimeError(f"loki query failed: {code}")
    return json.loads(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate observability stack")
    parser.add_argument("--output", choices=["events", "json"], default="events", help="Print events or only final JSON summary")
    parser.add_argument("--output-file", default=None, help="Write final JSON summary to file")
    args = parser.parse_args()
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    inference_url = os.getenv("INFERENCE_URL", "http://localhost:8080")
    prometheus_url = os.getenv("PROMETHEUS_URL", "http://localhost:9090")
    loki_url = os.getenv("LOKI_URL", "http://localhost:3100")

    results: Dict[str, object] = {"steps": []}
    passed = True

    try:
        token = generate_jwt_from_env()
    except Exception as e:
        if args.output == "events":
            print(json.dumps({"event": "jwt_error", "error": str(e)}))
        sys.exit(1)

    # 1) Metrics reachability
    try:
        be_metrics = fetch_metrics(backend_url, token)
        inf_metrics = fetch_metrics(inference_url, token)
        be_expected = [
            "fraud_backend_request_latency_seconds_bucket",
            "auth_failures_total",
            "model_decisions_total",
        ]
        inf_expected = [
            "infer_request_latency_ms_bucket",
            "model_decisions_total",
            "safe_mode_state",
        ]
        be_missing = assert_metrics_series(be_metrics, be_expected)
        inf_missing = assert_metrics_series(inf_metrics, inf_expected)
        step = {
            "step": "metrics_reachability",
            "backend_missing": be_missing,
            "inference_missing": inf_missing,
        }
        results["steps"].append(step)
        ok = not be_missing and not inf_missing
        passed = passed and ok
        if args.output == "events":
            print(json.dumps({"event": "metrics_checked", **step}))
    except Exception as e:
        results["steps"].append({"step": "metrics_reachability", "error": str(e)})
        if args.output == "events":
            print(json.dumps({"event": "metrics_error", "error": str(e)}))
        passed = False

    # 2) Alert firing/clearing (Safe Mode)
    try:
        toggle_safe_mode(backend_url, token, True, reason="obs-validation")
        # wait up to ~150s (scrape + for: 1m)
        fired = False
        for _ in range(30):
            alerts = prometheus_alerts(prometheus_url)
            if alert_present(alerts, "SafeModeActive"):
                fired = True
                break
            time.sleep(5)
        toggle_safe_mode(backend_url, token, False)
        cleared = False
        for _ in range(24):
            alerts = prometheus_alerts(prometheus_url)
            if not alert_present(alerts, "SafeModeActive"):
                cleared = True
                break
            time.sleep(5)
        step = {"step": "safe_mode_alert", "fired": fired, "cleared": cleared}
        results["steps"].append(step)
        if args.output == "events":
            print(json.dumps({"event": "safe_mode_alert_checked", **step}))
        passed = passed and fired and cleared
    except Exception as e:
        results["steps"].append({"step": "safe_mode_alert", "error": str(e)})
        if args.output == "events":
            print(json.dumps({"event": "safe_mode_alert_error", "error": str(e)}))
        passed = False

    # 3) Auth-failure alert simulation
    try:
        unauth_url = backend_url.rstrip("/") + "/admin/health"
        try:
            http_request(unauth_url)
        except Exception:
            pass
        fired = False
        for _ in range(24):
            alerts = prometheus_alerts(prometheus_url)
            if alert_present(alerts, "AuthFailures"):
                fired = True
                break
            time.sleep(5)
        step = {"step": "auth_fail_alert", "fired": fired}
        results["steps"].append(step)
        if args.output == "events":
            print(json.dumps({"event": "auth_fail_alert_checked", **step}))
        passed = passed and fired
    except Exception as e:
        results["steps"].append({"step": "auth_fail_alert", "error": str(e)})
        if args.output == "events":
            print(json.dumps({"event": "auth_fail_alert_error", "error": str(e)}))
        passed = False

    # 4) High latency simulation (best-effort)
    try:
        end_time = time.time() + 60
        path = "/admin/health"
        while time.time() < end_time:
            try:
                ts = str(int(time.time()))
                body = b""
                sig = hmac_signature(ts, body, "GET", path)
                headers = {"Authorization": f"Bearer {token}", "X-Request-Timestamp": ts, "X-Request-Signature": sig}
                http_request(backend_url.rstrip("/") + path, headers=headers)
            except Exception:
                pass
            time.sleep(0.05)
        p95q = "histogram_quantile(0.95, sum(rate(request_latency_ms_bucket[5m])) by (le))"
        res = prometheus_query(prometheus_url, p95q)
        val = None
        try:
            r = res.get("data", {}).get("result", [])
            if r:
                val = float(r[0]["value"][1])
        except Exception:
            val = None
        alerts = prometheus_alerts(prometheus_url)
        highlat = alert_present(alerts, "BackendHighLatencyP95")
        step = {"step": "latency_alert", "p95_ms": val, "alert_fired": highlat}
        results["steps"].append(step)
        if args.output == "events":
            print(json.dumps({"event": "latency_alert_checked", **step}))
        passed = passed and (val is not None)
    except Exception as e:
        results["steps"].append({"step": "latency_alert", "error": str(e)})
        if args.output == "events":
            print(json.dumps({"event": "latency_alert_error", "error": str(e)}))
        passed = False

    # 5) Loki logs with correlation_id across backend and inference
    try:
        corr = f"obs-{int(time.time()*1000)}"
        # generate a request that should pass auth and produce logs
        ts = str(int(time.time()))
        body = b""
        path = "/admin/health"
        sig = hmac_signature(ts, body, "GET", path)
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Request-Timestamp": ts,
            "X-Request-Signature": sig,
            "X-Correlation-ID": corr,
        }
        # hit backend admin/health
        http_request(backend_url.rstrip("/") + path, headers=headers)
        # allow promtail ingestion time
        time.sleep(5)
        now_ns = int(time.time() * 1e9)
        start_ns = now_ns - int(5 * 60 * 1e9)
        q_backend = '{job="fraud_k",service="backend",correlation_id="%s"}' % corr
        q_infer = '{job="fraud_k",service="ml-inference",correlation_id="%s"}' % corr
        resp_be = loki_query_range(loki_url, q_backend, start_ns, now_ns, limit=20)
        resp_inf = loki_query_range(loki_url, q_infer, start_ns, now_ns, limit=20)
        def _has_stream(data: Dict) -> bool:
            return bool(data.get("data", {}).get("result"))
        has_be = _has_stream(resp_be)
        has_inf = _has_stream(resp_inf)
        step = {"step": "loki_logs", "backend_found": has_be, "inference_found": has_inf, "correlation_id": corr}
        results["steps"].append(step)
        if args.output == "events":
            print(json.dumps({"event": "loki_checked", **step}))
        passed = passed and has_be and has_inf
    except Exception as e:
        results["steps"].append({"step": "loki_logs", "error": str(e)})
        if args.output == "events":
            print(json.dumps({"event": "loki_error", "error": str(e)}))
        passed = False

    # Summary
    results["passed"] = passed
    summary = {"event": "summary", "passed": passed, "steps": results["steps"]}
    print(json.dumps(summary)) if args.output == "json" else print(json.dumps(summary))
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(summary, f)
        except Exception:
            pass
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()




