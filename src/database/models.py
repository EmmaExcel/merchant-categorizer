"""SQLAlchemy models for the service database.

Production uses PostgreSQL (``UKMC_DATABASE_URL``); local tests use SQLite.
All persisted transaction text is redacted/cleaned — raw descriptions are
never stored.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RedactedTransaction(Base):
    __tablename__ = "redacted_transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)
    transaction_id: Mapped[Optional[str]] = mapped_column(String(128), index=True, nullable=True)

    # Privacy: only redacted and cleaned text is persisted.
    redacted_description: Mapped[str] = mapped_column(Text, nullable=False)
    cleaned_description: Mapped[str] = mapped_column(Text, nullable=False)
    pii_entity_types: Mapped[list] = mapped_column(JSON, default=list)

    amount: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    currency: Mapped[str] = mapped_column(String(8), default="GBP")
    direction: Mapped[str] = mapped_column(String(16), default="debit")
    transaction_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    merchant_name: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    mcc: Mapped[Optional[str]] = mapped_column(String(8), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    predictions: Mapped[list["Prediction"]] = relationship(
        back_populates="transaction", cascade="all, delete-orphan"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    transaction_id: Mapped[int] = mapped_column(
        ForeignKey("redacted_transactions.id", ondelete="CASCADE"), index=True
    )
    model_version: Mapped[str] = mapped_column(String(32), default="1.0.0")
    predicted_category: Mapped[str] = mapped_column(String(64), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    top3_json: Mapped[list] = mapped_column(JSON, default=list)
    requires_review: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    transaction: Mapped[RedactedTransaction] = relationship(back_populates="predictions")
    feedback: Mapped[list["FeedbackLabel"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class FeedbackLabel(Base):
    __tablename__ = "feedback_labels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("predictions.id", ondelete="SET NULL"), nullable=True, index=True
    )
    # Privacy: feedback stores redacted text only.
    redacted_description: Mapped[str] = mapped_column(Text, nullable=False)
    original_prediction: Mapped[str] = mapped_column(String(64), nullable=False)
    corrected_category: Mapped[str] = mapped_column(String(64), nullable=False)
    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    prediction: Mapped[Optional[Prediction]] = relationship(back_populates="feedback")


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(32), nullable=False)
    trained_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    artifact_path: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)


class TrainingRun(Base):
    __tablename__ = "training_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    model_type: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="started")
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
