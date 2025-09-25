"""Add client_id and organization_id columns for multi-tenancy

Revision ID: 623072238987
Revises:
Create Date: 2025-09-25 14:48:59.527419

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '623072238987'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add client_id column to upload_sessions table
    op.add_column('upload_sessions',
        sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_upload_sessions_client_id', 'upload_sessions', ['client_id'])

    # Add organization_id column to upload_sessions table
    op.add_column('upload_sessions',
        sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_upload_sessions_organization_id', 'upload_sessions', ['organization_id'])

    # Add client_id column to training_runs table
    op.add_column('training_runs',
        sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_training_runs_client_id', 'training_runs', ['client_id'])

    # Add organization_id column to training_runs table
    op.add_column('training_runs',
        sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_training_runs_organization_id', 'training_runs', ['organization_id'])

    # Add client_id column to optimization_runs table
    op.add_column('optimization_runs',
        sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_optimization_runs_client_id', 'optimization_runs', ['client_id'])

    # Add organization_id column to optimization_runs table
    op.add_column('optimization_runs',
        sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index('ix_optimization_runs_organization_id', 'optimization_runs', ['organization_id'])


def downgrade() -> None:
    # Remove indexes and columns in reverse order
    op.drop_index('ix_optimization_runs_organization_id', 'optimization_runs')
    op.drop_column('optimization_runs', 'organization_id')
    op.drop_index('ix_optimization_runs_client_id', 'optimization_runs')
    op.drop_column('optimization_runs', 'client_id')

    op.drop_index('ix_training_runs_organization_id', 'training_runs')
    op.drop_column('training_runs', 'organization_id')
    op.drop_index('ix_training_runs_client_id', 'training_runs')
    op.drop_column('training_runs', 'client_id')

    op.drop_index('ix_upload_sessions_organization_id', 'upload_sessions')
    op.drop_column('upload_sessions', 'organization_id')
    op.drop_index('ix_upload_sessions_client_id', 'upload_sessions')
    op.drop_column('upload_sessions', 'client_id')