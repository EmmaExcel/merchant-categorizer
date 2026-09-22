
from __future__ import annotations

import logging
from typing import Dict

from sqlalchemy import delete, select

from database.models import FeedbackLabel, Prediction, RedactedTransaction
from database.session import get_session_factory

logger = logging.getLogger(__name__)


def _delete_transaction_rows(session, transaction_ids) -> Dict[str, int]:
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
        delete(RedactedTransaction).where(RedactedTransaction.id.in_(transaction_ids))
    ).rowcount
    return {
        "redacted_transactions": transactions_deleted,
        "predictions": predictions_deleted,
        "feedback_labels": feedback_deleted,
    }


def delete_subject_data(subject_id: str) -> Dict[str, int]:
    factory = get_session_factory()
    with factory() as session:
        transaction_ids = select(RedactedTransaction.id).where(
            RedactedTransaction.subject_id == subject_id
        )
        counts = _delete_transaction_rows(session, transaction_ids)
        session.commit()

    logger.info("Deleted data for subject '%s': %s", subject_id, counts)
    return counts


def delete_transaction_data(transaction_id: str) -> Dict[str, int]:
    factory = get_session_factory()
    with factory() as session:
        transaction_ids = select(RedactedTransaction.id).where(
            RedactedTransaction.transaction_id == transaction_id
        )
        counts = _delete_transaction_rows(session, transaction_ids)
        session.commit()

    return counts
