"""add cost-of-carry columns to pricing_run

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-11
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE pricing_run
            ADD COLUMN repo_rate      NUMERIC(9,6),
            ADD COLUMN carry_days     INT,
            ADD COLUMN coupon_income  NUMERIC(18,8),
            ADD COLUMN financing_cost NUMERIC(18,8),
            ADD COLUMN net_carry      NUMERIC(18,8)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE pricing_run
            DROP COLUMN IF EXISTS repo_rate,
            DROP COLUMN IF EXISTS carry_days,
            DROP COLUMN IF EXISTS coupon_income,
            DROP COLUMN IF EXISTS financing_cost,
            DROP COLUMN IF EXISTS net_carry
        """
    )
