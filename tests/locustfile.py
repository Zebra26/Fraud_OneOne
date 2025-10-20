from datetime import datetime, timezone
from itertools import count
from random import random

from locust import HttpUser, between, task


_txn_counter = count(1)


def build_payload() -> dict:
    txn_id = f"txn_load_{next(_txn_counter)}"
    return {
        "transaction_id": txn_id,
        "account_id": "acc_locust",
        "amount": 2500.0 + random() * 100,
        "currency": "EUR",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "merchant_category": "electronics",
        "channel": "mobile",
        "device_id": "device-loadtest",
        "features": {
            "feat_high_amount": 1.0,
            "feat_risky_cat": 0.0,
            "feat_velocity": 0.35,
            "feat_device_change": 0.0,
        },
    }


class FraudAPIUser(HttpUser):
    wait_time = between(0.01, 0.1)

    @task
    def score_transaction(self) -> None:
        payload = build_payload()
        with self.client.post("/predictions/score", json=payload, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status {response.status_code}: {response.text}")
