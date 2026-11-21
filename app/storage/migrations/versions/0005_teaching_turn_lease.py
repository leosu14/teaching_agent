"""The teaching-turn lease (multi-replica readiness, issue #66 PR 1).

Two nullable columns on `teaching_sessions`: the token of the request producing the teacher turns the session owes,
and when its lease ends (database time, epoch milliseconds). Nothing to migrate: an existing session holds no lease.
The columns belong to a learner's session row, which their data deletion already erases.

Revision ID: 0005_teaching_turn_lease
Revises: 0004_class_assignments
Create Date: 2026-11-20 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0005_teaching_turn_lease'
down_revision = '0004_class_assignments'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database adopted from `create_all` (app/storage/migrate.py) may have them already.
    present = {c['name'] for c in sa.inspect(op.get_bind()).get_columns('teaching_sessions')}
    with op.batch_alter_table('teaching_sessions', schema=None) as batch_op:
        if 'turn_lease_token' not in present:
            batch_op.add_column(sa.Column('turn_lease_token', sa.String(length=64), nullable=True))
        if 'turn_lease_expires_ms' not in present:
            batch_op.add_column(sa.Column('turn_lease_expires_ms', sa.BigInteger(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table('teaching_sessions', schema=None) as batch_op:
        batch_op.drop_column('turn_lease_expires_ms')
        batch_op.drop_column('turn_lease_token')
