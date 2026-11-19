"""Class assignments (Important item 3, issue #62): a teacher assigns goals, course topics and deadlines to a class.

New tables `class_assignments` (teacher data: the assignment, naming no learner) and `assignment_learners` (learner
data: which learner got it, and their goal). Nothing to migrate: existing goals have no assignment, and keep behaving
as before. `LearningGoal.completed_at` lives in the goal's JSON body, so it needs no column; stored goals load with
none.

Revision ID: 0004_class_assignments
Revises: 0003_teacher_review
Create Date: 2026-11-19 12:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = '0004_class_assignments'
down_revision = '0003_teacher_review'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # A database adopted from `create_all` (app/storage/migrate.py) may have them already.
    present = set(sa.inspect(op.get_bind()).get_table_names())
    if 'class_assignments' not in present:
        op.create_table('class_assignments',
        sa.Column('assignment_id', sa.String(length=64), nullable=False),
        sa.Column('class_id', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=16), nullable=False),
        sa.Column('due_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('body', sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(['class_id'], ['classes.class_id'], ),
        sa.PrimaryKeyConstraint('assignment_id')
        )
        with op.batch_alter_table('class_assignments', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_class_assignments_class_id'), ['class_id'], unique=False)
    if 'assignment_learners' not in present:
        op.create_table('assignment_learners',
        sa.Column('assignment_id', sa.String(length=64), nullable=False),
        sa.Column('learner_id', sa.String(length=128), nullable=False),
        sa.Column('goal_id', sa.String(length=64), nullable=True),
        sa.Column('state', sa.String(length=16), nullable=False),
        sa.Column('assigned_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('source', sa.String(length=16), nullable=False),
        sa.ForeignKeyConstraint(['assignment_id'], ['class_assignments.assignment_id'], ),
        sa.PrimaryKeyConstraint('assignment_id', 'learner_id')
        )
        with op.batch_alter_table('assignment_learners', schema=None) as batch_op:
            batch_op.create_index(batch_op.f('ix_assignment_learners_learner_id'), ['learner_id'], unique=False)


def downgrade() -> None:
    with op.batch_alter_table('assignment_learners', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_assignment_learners_learner_id'))
    op.drop_table('assignment_learners')
    with op.batch_alter_table('class_assignments', schema=None) as batch_op:
        batch_op.drop_index(batch_op.f('ix_class_assignments_class_id'))
    op.drop_table('class_assignments')
