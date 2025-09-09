#!/bin/bash
set -e

# Script to add a new MMM client to the infrastructure
# Usage: ./add-client.sh <client-id> <domain-name> [db-instance-class]

CLIENT_ID="${1:-}"
DOMAIN_NAME="${2:-}"
DB_INSTANCE="${3:-db.t3.small}"
ENVIRONMENT="${ENVIRONMENT:-production}"

if [ -z "$CLIENT_ID" ] || [ -z "$DOMAIN_NAME" ]; then
    echo "Usage: $0 <client-id> <domain-name> [db-instance-class]"
    echo "Example: $0 acme-corp acme.mmm.yourdomain.com db.t3.small"
    exit 1
fi

echo "🚀 Adding new MMM client: $CLIENT_ID"
echo "   Domain: $DOMAIN_NAME"
echo "   DB Instance: $DB_INSTANCE"
echo "   Environment: $ENVIRONMENT"
echo ""

# Create Terraform variables for the new client
TERRAFORM_DIR="../terraform"
VARS_FILE="$TERRAFORM_DIR/clients.auto.tfvars"

# Create or update the client configuration
if [ ! -f "$VARS_FILE" ]; then
    cat > "$VARS_FILE" << EOF
client_configs = {}
EOF
fi

# Add new client to configuration
python3 -c "
import json
import re

# Read existing vars file
with open('$VARS_FILE', 'r') as f:
    content = f.read()

# Extract existing client_configs
match = re.search(r'client_configs\s*=\s*({[^}]*})', content, re.DOTALL)
if match:
    config_str = match.group(1)
    # Simple parsing - in production use proper HCL parser
    if config_str.strip() == '{}':
        configs = {}
    else:
        # This is a simplified parser - use proper HCL library in production
        configs = {}
else:
    configs = {}

# Add new client
new_client = {
    'client_id': '$CLIENT_ID',
    'domain_name': '$DOMAIN_NAME', 
    'db_instance': '$DB_INSTANCE',
    'min_capacity': 1,
    'max_capacity': 3
}

# Write updated configuration
with open('$VARS_FILE', 'w') as f:
    f.write('client_configs = {\n')
    for name, config in configs.items():
        f.write(f'  \"{name}\" = {{\n')
        for key, value in config.items():
            if isinstance(value, str):
                f.write(f'    {key} = \"{value}\"\n')
            else:
                f.write(f'    {key} = {value}\n')
        f.write('  }\n')
    
    # Add new client
    f.write(f'  \"client_{len(configs) + 1}\" = {{\n')
    for key, value in new_client.items():
        if isinstance(value, str):
            f.write(f'    {key} = \"{value}\"\n')
        else:
            f.write(f'    {key} = {value}\n')
    f.write('  }\n')
    f.write('}\n')
"

echo "✅ Updated Terraform configuration: $VARS_FILE"

# Plan and apply Terraform changes
echo "📋 Planning Terraform changes..."
cd "$TERRAFORM_DIR"
terraform plan -var-file="clients.auto.tfvars" -out="client-$CLIENT_ID.plan"

echo ""
read -p "Do you want to apply these changes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Applying Terraform changes..."
    terraform apply "client-$CLIENT_ID.plan"
    
    # Create database URL parameter
    echo "🔐 Setting up database connection..."
    aws ssm put-parameter \
        --name "/mmm/$CLIENT_ID/database/url" \
        --value "postgresql://mmm_admin:$(aws ssm get-parameter --name "/mmm/$CLIENT_ID/database/password" --with-decryption --query 'Parameter.Value' --output text)@mmm-$CLIENT_ID-$ENVIRONMENT.$(aws rds describe-db-instances --db-instance-identifier "mmm-$CLIENT_ID-$ENVIRONMENT" --query 'DBInstances[0].Endpoint.Address' --output text):5432/mmm_$(echo $CLIENT_ID | tr '-' '_')" \
        --type "SecureString" \
        --overwrite
    
    # Deploy application
    echo "🚀 Deploying application for client $CLIENT_ID..."
    aws ecs update-service \
        --cluster "mmm-cluster-$ENVIRONMENT" \
        --service "mmm-$CLIENT_ID-$ENVIRONMENT" \
        --force-new-deployment
    
    echo "✅ Client $CLIENT_ID has been successfully added!"
    echo ""
    echo "📊 Client Details:"
    echo "   - Client ID: $CLIENT_ID"
    echo "   - Domain: $DOMAIN_NAME"
    echo "   - Database: mmm-$CLIENT_ID-$ENVIRONMENT"
    echo "   - S3 Bucket: mmm-$CLIENT_ID-$ENVIRONMENT-*"
    echo "   - ECS Service: mmm-$CLIENT_ID-$ENVIRONMENT"
    echo ""
    echo "🔗 Next steps:"
    echo "   1. Configure DNS for $DOMAIN_NAME"
    echo "   2. Set up SSL certificate"
    echo "   3. Test the deployment"
    echo "   4. Run database migrations"
else
    echo "❌ Deployment cancelled"
    rm -f "client-$CLIENT_ID.plan"
fi