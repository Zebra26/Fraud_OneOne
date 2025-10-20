# Redis Cache

- Stores recent feature vectors to guarantee sub-100 ms inference.
- Key pattern: `features:{transaction_id}` and `risk:{transaction_id}` (TTL 1h / 10 min).
- Acts as hot Feature Store complementing Feast/Hive for historical access.

