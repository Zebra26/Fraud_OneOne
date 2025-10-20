#!/bin/bash
set -euo pipefail

BROKER="${KAFKA_BROKER:-kafka:9092}"

create_topic() {
  local topic_name=$1
  kafka-topics --create --if-not-exists \
    --bootstrap-server "$BROKER" \
    --replication-factor 1 \
    --partitions 3 \
    --topic "$topic_name"
}

create_topic "${KAFKA_TOPIC_TRANSACTIONS:-transactions}"
create_topic "${KAFKA_TOPIC_PREDICTIONS:-predictions}"

echo "Kafka topics initialisés pour la chaîne XAI (SHAP)."

