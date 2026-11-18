"""Teacher review (Important item 2): lesson approval before release, and grade overrides.

`classes.lesson_approval`: whether the class's teacher approves lessons before its learners see them (off). New
tables `lesson_releases` (a lesson requested while approval was required, and the teacher's decision) and
`grade_overrides` (a teacher's outcome for an UNCERTAIN grade). Nothing to migrate: existing classes do not require
approval, and existing lessons have no release row (released).

Revision ID: 0003_teacher_review
Revises: 0002_object_holds
Create Date: 2026-11-18 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0003_teacher_review'
down_revision = '0002_object_holds'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database adopted from `create_all` (app/storage/migrate.py) may have them already.
    inspector = sa.inspect(op.get_bind())
    present = set(inspector.get_table_names())
    if 'lesson_approval' not in {c['name'] for c in inspector.get_columns('classes')}:
        with op.batch_alter_table('classes', schema=None) as batch_op:
            batch_op.add_column(sa.Column('lesson_approval', sa.Boolean(), server_default=sa.text('false'),
                                          nullable=False))
    if 'lesson_releases' not in present:
        op.create_table('lesson_releases',
        sa.Column('task_id', sa.String(length=64), nullable=False),
        sa.Column('learner_id', sa.String(length=128), nullable=False),
        sa.Column('class_id', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('requested_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('decided_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('decided_by', sa.String(length=128), nullable=True),
        sa.Column('note', sa.String(length=500), nullable=True),
        sa.PrimaryKeyConstraint('task_id')
        )
        with op.batch_alter_table('lesson_releases', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_lesson_releases_class_id'), ['class_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_lesson_releases_learner_id'), ['learner_id'], unique=False)
    if 'grade_overrides' not in present:
        op.create_table('grade_overrides',
        sa.Column('attempt_id', sa.String(length=128), nullable=False),
        sa.Column('learner_id', sa.String(length=128), nullable=False),
        sa.Column('class_id', sa.String(length=64), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.PrimaryKeyConstraint('attempt_id')
        )
        with op.batch_alter_table('grade_overrides', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_grade_overrides_class_id'), ['class_id'], unique=False)
            batch_op.create_index(batch_op.f('ix_grade_overrides_learner_id'), ['learner_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('grade_overrides', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_grade_overrides_learner_id'))
        batch_op.drop_index(batch_op.f('ix_grade_overrides_class_id'))
    op.drop_table('grade_overrides')
    with op.batch_alter_table('lesson_releases', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_lesson_releases_learner_id'))
        batch_op.drop_index(batch_op.f('ix_lesson_releases_class_id'))
    op.drop_table('lesson_releases')
    with op.batch_alter_table('classes', schema=None) as batch_op:
        batch_op.drop_column('lesson_approval')
