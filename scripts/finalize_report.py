#!/usr/bin/env python3
"""
Finalize validation results for Fraud_One realtime fraud detection platform.

Aggregates observability, performance, and test results into a single report:
- observability_report.json  → Metrics validation (Prometheus/Grafana)
- k6_summary.json (optional) → Load testing summary (latency, throughput)

Outputs:
- fraud_one_validation_summary.json
Prints a green ✅ message if passed, red ❌ if failed.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# --- File paths ---
OBSERVABILITY_PATH = Path("observability_report.json")
K6_PATH = Path("k6_summary.json")
OUTPUT_PATH = Path("fraud_one_validation_summary.json")

# --- Helper to load JSON safely ---
def load_json(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {"error": f"Invalid JSON in {path.name}"}
    else:
        return {"missing": True}

def main():
    report = {
        "project": "Fraud_One",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "components": {},
        "passed": False,
    }

    # --- Load observability report ---
    observability = load_json(OBSERVABILITY_PATH)
    report["components"]["observability"] = observability

    obs_passed = (
        isinstance(observability, dict)
        and observability.get("passed") is True
    )

    # --- Load k6 performance summary (optional) ---
    if K6_PATH.exists():
        k6_summary = load_json(K6_PATH)
        report["components"]["performance"] = k6_summary
        report["performance_present"] = True
    else:
        report["performance_present"] = False

    # --- Determine global result ---
    report["passed"] = obs_passed

    # --- Write consolidated report ---
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # --- Print result message ---
    if obs_passed:
        print("✅ Fraud_One platform successfully verified and operational")
        sys.exit(0)
    else:
        print("❌ Validation did not pass. See fraud_one_validation_summary.json for details")
        sys.exit(1)

if __name__ == "__main__":
    main()
