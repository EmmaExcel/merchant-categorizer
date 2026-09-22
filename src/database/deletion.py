"""Data-deletion helpers.

Implements the right to erasure at the application level for a user/transaction
identifier. Only redacted data exists in this database, but pseudonymised data
can still be personal data, so a deletion function is provided and documented.
"""

from __future__ import annotations

import logging
from typing import Dict

from sqlalchemy import delete, select

from database.models import FeedbackLabel, Prediction, RedactedTransaction
from database.session import get_session_factory

logger = logging.getLogger(__name__)


def delete_subject_data(subject_id: str) -> Dict[str, int]:
    """Delete all stored rows belonging to ``subject_id``.

    Returns a count of deleted rows per table. Deletes cascade from
    redacted transactions through predictions to feedback labels.
    """
    factory = get_session_factory()
    with factory() as session:
        transaction_ids = select(RedactedTransaction.id).where(
            RedactedTransaction.subject_id == subject_id
        )
        prediction_ids = select(Prediction.id).where(
            Prediction.transaction_id.in_(transaction_ids)
        )

        feedback_deleted = session.execute(
            delete(FeedbackLabel).where(FeedbackLabel.prediction_id.in_(prediction_ids))
        ).rowcount
        predictions_deleted = session.execute(
            delete(Prediction).where(Prediction.transaction_id.in_(transaction_ids))
        ).rowcount
        transactions_deleted = session.execute(
            delete(RedactedTransaction).where(RedactedTransaction.subject_id == subject_id)
        ).rowcount

        session.commit()

    counts = {
        "redacted_transactions": transactions_deleted,
        "predictions": predictions_deleted,
        "feedback_labels": feedback_deleted,
    }
    logger.info("Deleted data for subject '%s': %s", subject_id, counts)
    return counts


def delete_transaction_data(transaction_id: str) -> Dict[str, int]:
    """Delete all stored rows for a single transaction identifier."""
    factory = get_session_factory()
    with factory() as session:
        tx_ids = select(RedactedTransaction.id).where(
            RedactedTransaction.transaction_id == transaction_id
        )
        prediction_ids = select(Prediction.id).where(
            Prediction.transaction_id.in_(tx_ids)
        )

        feedback_deleted = session.execute(
            delete(FeedbackLabel).where(FeedbackLabel.prediction_id.in_(prediction_ids))
        ).rowcount
        predictions_deleted = session.execute(
            delete(Prediction).where(Prediction.transaction_id.in_(tx_ids))
        ).rowcount
        transactions_deleted = session.execute(
            delete(RedactedTransaction).where(
                RedactedTransaction.transaction_id == transaction_id
            )
        ).rowcount
        session.commit()

    return {
        "redacted_transactions": transactions_deleted,
        "predictions": predictions_deleted,
        "feedback_labels": feedback_deleted,
    }
