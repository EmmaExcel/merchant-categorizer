"""initial tables

Revision ID: 0001
Revises:
Create Date: 2026-09-22
"""

from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "redacted_transactions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("subject_id", sa.String(length=128), nullable=True),
        sa.Column("transaction_id", sa.String(length=128), nullable=True),
        sa.Column("redacted_description", sa.Text(), nullable=False),
        sa.Column("cleaned_description", sa.Text(), nullable=False),
        sa.Column("pii_entity_types", sa.JSON(), nullable=True),
        sa.Column("amount", sa.Float(), nullable=True),
        sa.Column("currency", sa.String(length=8), nullable=True),
        sa.Column("direction", sa.String(length=16), nullable=True),
        sa.Column("transaction_type", sa.String(length=32), nullable=True),
        sa.Column("merchant_name", sa.String(length=256), nullable=True),
        sa.Column("mcc", sa.String(length=8), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index(
        "ix_redacted_transactions_subject_id",
        "redacted_transactions",
        ["subject_id"],
    )
    op.create_index(
        "ix_redacted_transactions_transaction_id",
        "redacted_transactions",
        ["transaction_id"],
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "transaction_id",
            sa.Integer(),
            sa.ForeignKey("redacted_transactions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(length=32), nullable=True),
        sa.Column("predicted_category", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("top3_json", sa.JSON(), nullable=True),
        sa.Column("requires_review", sa.Boolean(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_predictions_transaction_id", "predictions", ["transaction_id"])

    op.create_table(
        "feedback_labels",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column(
            "prediction_id",
            sa.Integer(),
            sa.ForeignKey("predictions.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("redacted_description", sa.Text(), nullable=False),
        sa.Column("original_prediction", sa.String(length=64), nullable=False),
        sa.Column("corrected_category", sa.String(length=64), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_feedback_labels_prediction_id", "feedback_labels", ["prediction_id"])

    op.create_table(
        "model_versions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("model_name", sa.String(length=64), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("trained_at", sa.DateTime(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("artifact_path", sa.String(length=512), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=True),
    )

    op.create_table(
        "training_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("run_id", sa.String(length=128), nullable=False),
        sa.Column("model_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("params", sa.JSON(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
    )
    op.create_index("ix_training_runs_run_id", "training_runs", ["run_id"], unique=True)


def downgrade() -> None:
    op.drop_table("training_runs")
    op.drop_table("model_versions")
    op.drop_table("feedback_labels")
    op.drop_table("predictions")
    op.drop_table("redacted_transactions")
