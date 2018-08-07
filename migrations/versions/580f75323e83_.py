"""empty message

Revision ID: 580f75323e83
Revises: bb61b40d54d8
Create Date: 2018-07-29 09:54:21.141418

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '580f75323e83'
down_revision = 'bb61b40d54d8'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('departments',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('name', sa.String(length=60), nullable=True),
    sa.Column('description', sa.String(length=200), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    op.add_column('employees', sa.Column('department_id', sa.Integer(), nullable=True))
    op.create_foreign_key(None, 'employees', 'departments', ['department_id'], ['id'])
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_constraint(None, 'employees', type_='foreignkey')
    op.drop_column('employees', 'department_id')
    op.drop_table('departments')
    # ### end Alembic commands ###
