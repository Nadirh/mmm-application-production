"""Add client_id and organization_id fields for multi-tenancy

Revision ID: 001_add_client_tenant_fields
Revises:
Create Date: 2025-01-01 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_add_client_tenant_fields'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add client_id and organization_id to upload_sessions
    op.add_column('upload_sessions',
                  sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.add_column('upload_sessions',
                  sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index(op.f('ix_upload_sessions_client_id'), 'upload_sessions', ['client_id'], unique=False)
    op.create_index(op.f('ix_upload_sessions_organization_id'), 'upload_sessions', ['organization_id'], unique=False)

    # Add client_id and organization_id to training_runs
    op.add_column('training_runs',
                  sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.add_column('training_runs',
                  sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index(op.f('ix_training_runs_client_id'), 'training_runs', ['client_id'], unique=False)
    op.create_index(op.f('ix_training_runs_organization_id'), 'training_runs', ['organization_id'], unique=False)

    # Add client_id and organization_id to optimization_runs
    op.add_column('optimization_runs',
                  sa.Column('client_id', sa.String(), nullable=True, server_default='default'))
    op.add_column('optimization_runs',
                  sa.Column('organization_id', sa.String(), nullable=True, server_default='default'))
    op.create_index(op.f('ix_optimization_runs_client_id'), 'optimization_runs', ['client_id'], unique=False)
    op.create_index(op.f('ix_optimization_runs_organization_id'), 'optimization_runs', ['organization_id'], unique=False)


def downgrade() -> None:
    # Remove indexes and columns from optimization_runs
    op.drop_index(op.f('ix_optimization_runs_organization_id'), table_name='optimization_runs')
    op.drop_index(op.f('ix_optimization_runs_client_id'), table_name='optimization_runs')
    op.drop_column('optimization_runs', 'organization_id')
    op.drop_column('optimization_runs', 'client_id')

    # Remove indexes and columns from training_runs
    op.drop_index(op.f('ix_training_runs_organization_id'), table_name='training_runs')
    op.drop_index(op.f('ix_training_runs_client_id'), table_name='training_runs')
    op.drop_column('training_runs', 'organization_id')
    op.drop_column('training_runs', 'client_id')

    # Remove indexes and columns from upload_sessions
    op.drop_index(op.f('ix_upload_sessions_organization_id'), table_name='upload_sessions')
    op.drop_index(op.f('ix_upload_sessions_client_id'), table_name='upload_sessions')
    op.drop_column('upload_sessions', 'organization_id')
    op.drop_column('upload_sessions', 'client_id')