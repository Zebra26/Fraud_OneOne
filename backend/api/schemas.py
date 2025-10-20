from datetime import datetime
from typing import Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


class GraphContext(BaseModel):
    source_id: str
    target_id: str

    class Config:
        extra = "forbid"


class TransactionFeatures(BaseModel):
    transaction_amount: float = Field(..., ge=0)
    transaction_time_seconds: int = Field(..., ge=0)
    is_weekend: int = Field(..., ge=0, le=1)
    hour_of_day: int = Field(..., ge=0, le=23)
    is_round_amount: int = Field(..., ge=0, le=1)
    unique_receivers_24h: int = Field(..., ge=0)
    vpn_detected: int = Field(..., ge=0, le=1)
    location_risk_score: float = Field(..., ge=0, le=1)
    transaction_frequency_30min: int = Field(..., ge=0)
    login_ip_changed_last_hour: int = Field(..., ge=0, le=1)
    avg_transaction_amount_24h: float = Field(..., ge=0)
    time_since_last_tx: int = Field(..., ge=0)
    ip_risk_score: float = Field(..., ge=0, le=1)
    transactions_last_24h: int = Field(..., ge=0)
    customer_segment: Literal["standard", "premium", "business"]

    class Config:
        extra = "forbid"


class TransactionIn(BaseModel):
    transaction_id: str = Field(..., example="txn_123")
    account_id: str = Field(..., example="acc_456")
    amount: float = Field(..., gt=0)
    currency: str = Field("EUR")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    merchant_category: Optional[str]
    channel: Optional[str]
    device_id: Optional[str]
    geolocation: Optional[Dict[str, float]]
    graph_context: Optional[GraphContext] = None
    sequence: Optional[List[List[float]]] = None
    features: TransactionFeatures

    class Config:
        extra = "forbid"


class TransactionIngestResponse(BaseModel):
    status: str
    kafka_topic: str
    message_id: str


class ShapFeatureContribution(BaseModel):
    feature: str
    contribution: float


class PredictionExplanation(BaseModel):
    base_value: float
    shap_values: List[ShapFeatureContribution]
    compliance_reference: str = Field(
        "Explicabilite (XAI) : SHAP placeholder - advanced model does not provide per-feature contributions."
    )
    model_version: str = Field(..., description="Version du modele ayant genere la decision")


class RiskBreakdownSchema(BaseModel):
    supervised_probability: float
    anomaly_risk: float
    deep_reconstruction_risk: float
    aggregated_risk: float


class PredictionResponse(BaseModel):
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    decision: Literal["FRAUD", "NORMAL"] = Field(..., description="FRAUD | NORMAL")
    risk_breakdown: RiskBreakdownSchema
    explanation: PredictionExplanation


class HealthResponse(BaseModel):
    status: str
    dependencies: Dict[str, str]
