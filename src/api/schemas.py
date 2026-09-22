from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from preprocessing.features import LABELS


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000, description="Raw transaction description")
    amount: Optional[float] = Field(None, description="Transaction amount in pounds")
    currency: str = Field("GBP", max_length=8)
    direction: Optional[str] = Field(None, description="debit or credit")
    transaction_type: Optional[str] = Field(None, max_length=64)
    mcc: Optional[str] = Field(None, max_length=8)
    subject_id: Optional[str] = Field(None, max_length=128, description="Optional subject identifier for deletion support")
    transaction_id: Optional[str] = Field(None, max_length=128)

    @field_validator("direction")
    @classmethod
    def _validate_direction(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        normalised = value.strip().lower()
        if normalised in {"debit", "credit", "in", "out", "incoming", "outgoing", "deposit", "payment"}:
            return value
        raise ValueError("direction must be debit or credit")


class CategoryPrediction(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    predicted_category: str
    confidence: float
    top_3_predictions: List[CategoryPrediction]
    redacted_description: str
    cleaned_description: str
    pii_entities_redacted: List[str]
    requires_review: bool
    model_version: str


class BatchPredictRequest(BaseModel):
    transactions: List[PredictRequest] = Field(..., min_length=1, max_length=100)


class BatchPredictResponse(BaseModel):
    results: List[PredictResponse]
    total: int


class FeedbackRequest(BaseModel):
    redacted_description: str = Field(..., min_length=1, max_length=2000)
    original_prediction: str = Field(..., max_length=64)
    corrected_category: str = Field(..., max_length=64)
    note: Optional[str] = Field(None, max_length=2000)
    prediction_id: Optional[int] = Field(None)
    subject_id: Optional[str] = Field(None, max_length=128)

    @field_validator("corrected_category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        if value not in LABELS:
            raise ValueError(f"corrected_category must be one of {LABELS}")
        return value


class FeedbackResponse(BaseModel):
    id: int
    stored: str = "redacted"


class HealthResponse(BaseModel):
    status: str
    model_version: str
    fixture_mode: bool
    database: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    supported_labels: List[str]
    training_timestamp: Optional[str] = None
    metrics: dict = Field(default_factory=dict)


class DeletionResponse(BaseModel):
    subject_id: str
    deleted: dict
