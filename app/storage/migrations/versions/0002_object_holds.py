"""Object holds and the object-deletion queue (MVP item 9: learner data deletion and shared media).

`object_holds`: a durable reference to a stored object by its owner (a task or a learning cycle), made before the
object is put or reused. `object_deletions`: objects a learner's data deletion queued in its own transaction, deleted
after it commits once nothing points at them (and retried by the sweeper). Both are new tables: nothing to migrate.

Revision ID: 0002_object_holds
Revises: 0001_baseline
Create Date: 2026-11-14 20:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0002_object_holds'
down_revision = '0001_baseline'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database adopted from `create_all` (app/storage/migrate.py) may have them already: it is stamped at the
    # baseline after `create_all` made every table of the current schema.
    present = set(sa.inspect(op.get_bind()).get_table_names())
    if 'object_deletions' not in present:
        _deletions()
    if 'object_holds' not in present:
        _holds()


def _deletions() -> None:
    op.create_table('object_deletions',
    sa.Column('deletion_id', sa.String(length=64), nullable=False),
    sa.Column('uri', sa.String(length=512), nullable=False),
    sa.Column('content_hash', sa.String(length=64), nullable=False),
    sa.Column('queued_at', sa.DateTime(timezone=True), nullable=False),
    sa.Column('attempts', sa.Integer(), server_default='0', nullable=False),
    sa.Column('next_attempt_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('deletion_id')
    )
    with op.batch_alter_table('object_deletions', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_object_deletions_next_attempt_at'), ['next_attempt_at'], unique=False)
        batch_op.create_index(batch_op.f('ix_object_deletions_uri'), ['uri'], unique=False)


def _holds() -> None:
    op.create_table('object_holds',
    sa.Column('uri', sa.String(length=512), nullable=False),
    sa.Column('owner_id', sa.String(length=64), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
    sa.PrimaryKeyConstraint('uri', 'owner_id')
    )
    with op.batch_alter_table('object_holds', schema=None) as batch_op:
        batch_op.create_index(batch_op.f('ix_object_holds_owner_id'), ['owner_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('object_holds', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_object_holds_owner_id'))
    op.drop_table('object_holds')
    with op.batch_alter_table('object_deletions', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_object_deletions_uri'))
        batch_op.drop_index(batch_op.f('ix_object_deletions_next_attempt_at'))
    op.drop_table('object_deletions')
