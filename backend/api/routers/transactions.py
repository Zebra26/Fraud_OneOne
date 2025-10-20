import logging
from datetime import datetime

from fastapi import APIRouter, Depends

from ..config import get_settings
from ..dependencies import get_kafka_producer, get_mongo_client, get_redis_cache
from ..schemas import TransactionIn, TransactionIngestResponse


router = APIRouter(prefix="/transactions", tags=["transactions"])

logger = logging.getLogger(__name__)


@router.post("/ingest", response_model=TransactionIngestResponse)
async def ingest_transaction(
    payload: TransactionIn,
    settings=Depends(get_settings),
    producer=Depends(get_kafka_producer),
    mongo=Depends(get_mongo_client),
    redis_cache=Depends(get_redis_cache),
):
    transaction_dict = payload.dict()
    transaction_dict["timestamp"] = transaction_dict["timestamp"].isoformat()
    transaction_dict["received_at"] = datetime.utcnow().isoformat()

    message_id = await producer.send_transaction_async(settings.kafka_topic_transactions, transaction_dict)
    mongo_id = await mongo.insert_transaction({"kafka_message_id": message_id, **transaction_dict})

    await redis_cache.set_feature_vector(f"features:{payload.transaction_id}", payload.json())

    logger.info(
        "Transaction ingérée",
        extra={"transaction_id": payload.transaction_id, "kafka_message_id": message_id, "mongo_id": mongo_id},
    )

    return TransactionIngestResponse(status="queued", kafka_topic=settings.kafka_topic_transactions, message_id=message_id)
