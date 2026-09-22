"""FastAPI app: demo page at ``/``, prediction endpoints, and OpenAPI docs.

Raw descriptions are redacted before preprocessing, persistence, or logging.
"""

from __future__ import annotations

import logging
import os
from typing import List

from fastapi import Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from api.demo import demo_page
from api.dependencies import (
    clear_model_cache,
    get_cleaner_dep,
    get_model,
    get_model_info,
    get_redactor_dep,
    model_version,
)
from api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    CategoryPrediction,
    DeletionResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
)
from config import settings
from database import deletion as deletion_helpers
from database.models import FeedbackLabel, Prediction, RedactedTransaction
from database.session import get_db, init_db
from preprocessing.features import ID2LABEL, LABELS, build_metadata_features

logger = logging.getLogger(__name__)

app = FastAPI(
    title="UK Merchant Categoriser",
    version=settings.APP_VERSION,
    description=(
        "Privacy-first local ML merchant categorisation for messy UK bank "
        "transaction descriptions. Proof of concept on synthetic/sandbox data."
    ),
)


@app.on_event("startup")
def _startup() -> None:
    init_db()
    # Warm the model at startup so the first request does not pay the load
    # cost and deployments fail fast if the artifact cannot be read.
    model = get_model()
    logger.info(
        "UK Merchant Categoriser started (fixture_mode=%s, serving=%s)",
        settings.FIXTURE_MODE,
        getattr(model, "name", type(model).__name__),
    )


@app.get("/", include_in_schema=False)
def index():
    return demo_page()


def _predict_one(payload: PredictRequest, db: Session) -> PredictResponse:
    model = get_model()
    redactor = get_redactor_dep()
    cleaner = get_cleaner_dep()

    redaction = redactor.redact(payload.description)
    cleaned = cleaner.clean_redacted(redaction.redacted_text)

    meta = build_metadata_features(
        amount=payload.amount,
        direction=payload.direction,
        transaction_type=payload.transaction_type,
        mcc=payload.mcc,
        pii_entities=redaction.entity_types,
        raw_description=redaction.redacted_text,
    )

    prediction = model.predict([cleaned] if cleaned else ["[UNK]"], [meta])
    topk_indices = prediction["topk_indices"][0]
    topk_probs = prediction["topk_probs"][0]

    top_category = ID2LABEL[int(topk_indices[0])]
    confidence = float(topk_probs[0])
    requires_review = confidence < settings.CONFIDENCE_THRESHOLD
    top_3 = [
        CategoryPrediction(category=ID2LABEL[int(idx)], confidence=float(prob))
        for idx, prob in zip(topk_indices, topk_probs)
    ]


    transaction = RedactedTransaction(
        subject_id=payload.subject_id,
        transaction_id=payload.transaction_id,
        redacted_description=redaction.redacted_text,
        cleaned_description=cleaned,
        pii_entity_types=redaction.entity_types,
        amount=payload.amount,
        currency=payload.currency,
        direction=meta["direction"],
        transaction_type=payload.transaction_type,
        merchant_name=None,
        mcc=payload.mcc,
    )
    db.add(transaction)
    db.flush()
    db.add(
        Prediction(
            transaction_id=transaction.id,
            model_version=model_version(),
            predicted_category=top_category,
            confidence=confidence,
            top3_json=[item.model_dump() for item in top_3],
            requires_review=requires_review,
        )
    )


    logger.info(
        "Prediction: cleaned=%r category=%s confidence=%.3f review=%s",
        cleaned, top_category, confidence, requires_review,
    )

    return PredictResponse(
        predicted_category=top_category,
        confidence=round(confidence, 4),
        top_3_predictions=top_3,
        redacted_description=redaction.redacted_text,
        cleaned_description=cleaned,
        pii_entities_redacted=redaction.entity_types,
        requires_review=requires_review,
        model_version=model_version(),
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        model_version=model_version(),
        fixture_mode=settings.FIXTURE_MODE,
        database=settings.DATABASE_URL.split("://")[0],
    )


@app.get("/model", response_model=ModelInfoResponse, tags=["model"])
def model_info() -> ModelInfoResponse:
    info = get_model_info()
    return ModelInfoResponse(**info)


@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(payload: PredictRequest, db: Session = Depends(get_db)) -> PredictResponse:
    try:
        return _predict_one(payload, db)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Prediction failed: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Raw description was not logged.",
        ) from exc


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["prediction"])
def predict_batch(
    payload: BatchPredictRequest, db: Session = Depends(get_db)
) -> BatchPredictResponse:
    results: List[PredictResponse] = []
    for transaction in payload.transactions:
        try:
            results.append(_predict_one(transaction, db))
        except Exception as exc:  # noqa: BLE001
            logger.error("Batch item prediction failed: %s", type(exc).__name__)
            results.append(
                PredictResponse(
                    predicted_category="Other",
                    confidence=0.0,
                    top_3_predictions=[],
                    redacted_description="[REDACTION_ERROR]",
                    cleaned_description="",
                    pii_entities_redacted=[],
                    requires_review=True,
                    model_version=model_version(),
                )
            )
    return BatchPredictResponse(results=results, total=len(results))


@app.post("/feedback", response_model=FeedbackResponse, tags=["feedback"])
def submit_feedback(
    payload: FeedbackRequest, db: Session = Depends(get_db)
) -> FeedbackResponse:
    if payload.corrected_category not in LABELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"corrected_category must be one of {LABELS}",
        )
    feedback = FeedbackLabel(
        prediction_id=payload.prediction_id,
        redacted_description=payload.redacted_description,
        original_prediction=payload.original_prediction,
        corrected_category=payload.corrected_category,
        note=payload.note,
    )
    db.add(feedback)
    db.flush()
    logger.info("Feedback stored (id=%s, corrected=%s)", feedback.id, payload.corrected_category)
    return FeedbackResponse(id=feedback.id, stored="redacted")


@app.delete("/data/{subject_id}", response_model=DeletionResponse, tags=["privacy"])
def delete_subject(subject_id: str) -> DeletionResponse:
    counts = deletion_helpers.delete_subject_data(subject_id)
    return DeletionResponse(subject_id=subject_id, deleted=counts)


@app.post("/admin/reload-model", tags=["admin"])
def reload_model() -> dict:
    clear_model_cache()
    return {"status": "model cache cleared"}


def run() -> None:
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("api.main:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    run()
