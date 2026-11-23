"""Shared state of several processes (multi-replica readiness, issue #66 PR 2).

Four new tables, nothing to migrate:
- `rate_limit_starts` and `rate_limit_slots`: a provider's requests-per-minute window and concurrency slots, counted
  across every process on the database (no learner data; kept by a learner's data deletion);
- `stream_slots`: the open progress streams, counted across every API process (a hash of the principal only; kept);
- `research_cache`: cached web search results per learner, 7 days (erased with the learner's data).

Revision ID: 0006_shared_state
Revises: 0005_teaching_turn_lease
Create Date: 2026-11-21 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0006_shared_state'
down_revision = '0005_teaching_turn_lease'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database adopted from `create_all` (app/storage/migrate.py) may have them already.
    present = set(sa.inspect(op.get_bind()).get_table_names())
    if 'rate_limit_starts' not in present:
        op.create_table(
            'rate_limit_starts',
            sa.Column('start_id', sa.String(length=64), nullable=False),
            sa.Column('limit_key', sa.String(length=160), nullable=False),
            sa.Column('started_ms', sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint('start_id'),
        )
        op.create_index('ix_rate_limit_starts_key', 'rate_limit_starts', ['limit_key', 'started_ms'], unique=False)
    if 'rate_limit_slots' not in present:
        op.create_table(
            'rate_limit_slots',
            sa.Column('slot_id', sa.String(length=64), nullable=False),
            sa.Column('limit_key', sa.String(length=160), nullable=False),
            sa.Column('expires_ms', sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint('slot_id'),
        )
        op.create_index('ix_rate_limit_slots_key', 'rate_limit_slots', ['limit_key', 'expires_ms'], unique=False)
    if 'stream_slots' not in present:
        op.create_table(
            'stream_slots',
            sa.Column('slot_id', sa.String(length=64), nullable=False),
            sa.Column('principal_hash', sa.String(length=64), nullable=False),
            sa.Column('expires_ms', sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint('slot_id'),
        )
        op.create_index('ix_stream_slots_principal', 'stream_slots', ['principal_hash', 'expires_ms'], unique=False)
        op.create_index('ix_stream_slots_expires', 'stream_slots', ['expires_ms'], unique=False)
    if 'research_cache' not in present:
        op.create_table(
            'research_cache',
            sa.Column('learner_id', sa.String(length=128), nullable=False),
            sa.Column('cache_key', sa.String(length=64), nullable=False),
            sa.Column('body', sa.Text(), nullable=False),
            sa.Column('created_ms', sa.BigInteger(), nullable=False),
            sa.Column('expires_ms', sa.BigInteger(), nullable=False),
            sa.PrimaryKeyConstraint('learner_id', 'cache_key'),
        )
        op.create_index('ix_research_cache_expires', 'research_cache', ['expires_ms'], unique=False)


def downgrade() -> None:
    op.drop_index('ix_research_cache_expires', table_name='research_cache')
    op.drop_table('research_cache')
    op.drop_index('ix_stream_slots_expires', table_name='stream_slots')
    op.drop_index('ix_stream_slots_principal', table_name='stream_slots')
    op.drop_table('stream_slots')
    op.drop_index('ix_rate_limit_slots_key', table_name='rate_limit_slots')
    op.drop_table('rate_limit_slots')
    op.drop_index('ix_rate_limit_starts_key', table_name='rate_limit_starts')
    op.drop_table('rate_limit_starts')
