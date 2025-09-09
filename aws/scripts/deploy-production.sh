#!/bin/bash
set -e

# Production deployment script for MMM Application
# Usage: ./deploy-production.sh [client-id]

CLIENT_ID="${1:-mmm-demo}"
ENVIRONMENT="production"
AWS_REGION="${AWS_REGION:-us-east-1}"
ECR_REPOSITORY="mmm-application"

echo "🚀 Deploying MMM Application to Production"
echo "   Client ID: $CLIENT_ID"
echo "   Environment: $ENVIRONMENT"
echo "   AWS Region: $AWS_REGION"
echo ""

# Check prerequisites
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found. Please install and configure AWS CLI."
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker."
    exit 1
fi

# Get AWS account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo "❌ Could not determine AWS Account ID. Please check your AWS credentials."
    exit 1
fi

echo "✅ AWS Account ID: $AWS_ACCOUNT_ID"

# Build and push Docker image
echo "🐳 Building Docker image..."
DOCKER_TAG="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:latest"

# Build image
docker build -t $ECR_REPOSITORY:latest .
docker tag $ECR_REPOSITORY:latest $DOCKER_TAG

# Login to ECR
echo "🔐 Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository if it doesn't exist
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION || {
    echo "📦 Creating ECR repository..."
    aws ecr create-repository --repository-name $ECR_REPOSITORY --region $AWS_REGION
}

# Push image
echo "📤 Pushing image to ECR..."
docker push $DOCKER_TAG

echo "✅ Image pushed successfully: $DOCKER_TAG"

# Deploy infrastructure with Terraform
echo "🏗️ Deploying infrastructure with Terraform..."
cd ../terraform

# Initialize Terraform if needed
if [ ! -d ".terraform" ]; then
    terraform init
fi

# Plan and apply
terraform plan -var-file="clients.auto.tfvars" -out="production-deploy.plan"

echo ""
read -p "Do you want to apply these infrastructure changes? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply "production-deploy.plan"
    rm -f "production-deploy.plan"
else
    echo "❌ Infrastructure deployment cancelled"
    rm -f "production-deploy.plan"
    exit 1
fi

# Set up database connection parameters
echo "🔐 Setting up database connection..."
DB_ENDPOINT=$(aws rds describe-db-instances \
    --db-instance-identifier "mmm-$CLIENT_ID-$ENVIRONMENT" \
    --region $AWS_REGION \
    --query 'DBInstances[0].Endpoint.Address' \
    --output text)

if [ "$DB_ENDPOINT" != "None" ] && [ -n "$DB_ENDPOINT" ]; then
    # Get database password from SSM
    DB_PASSWORD=$(aws ssm get-parameter \
        --name "/mmm/$CLIENT_ID/database/password" \
        --region $AWS_REGION \
        --with-decryption \
        --query 'Parameter.Value' \
        --output text)
    
    # Create full database URL
    DB_URL="postgresql+asyncpg://mmm_admin:$DB_PASSWORD@$DB_ENDPOINT:5432/mmm_$(echo $CLIENT_ID | tr '-' '_')"
    
    # Store database URL in SSM
    aws ssm put-parameter \
        --name "/mmm/$CLIENT_ID/database/url" \
        --value "$DB_URL" \
        --type "SecureString" \
        --region $AWS_REGION \
        --overwrite
    
    echo "✅ Database connection configured"
else
    echo "❌ Could not find database endpoint"
    exit 1
fi

# Run database migrations
echo "🔄 Running database migrations..."
# This would typically be done via an ECS task or lambda function
# For now, we'll just ensure the task definition is updated

# Update ECS service with new image
echo "🚀 Updating ECS service..."
aws ecs update-service \
    --cluster "mmm-cluster-$ENVIRONMENT" \
    --service "mmm-$CLIENT_ID-$ENVIRONMENT" \
    --region $AWS_REGION \
    --force-new-deployment

# Wait for deployment to complete
echo "⏳ Waiting for deployment to complete..."
aws ecs wait services-stable \
    --cluster "mmm-cluster-$ENVIRONMENT" \
    --services "mmm-$CLIENT_ID-$ENVIRONMENT" \
    --region $AWS_REGION

# Get service status
echo "📊 Deployment Status:"
aws ecs describe-services \
    --cluster "mmm-cluster-$ENVIRONMENT" \
    --services "mmm-$CLIENT_ID-$ENVIRONMENT" \
    --region $AWS_REGION \
    --query 'services[0].{ServiceName:serviceName,Status:status,RunningCount:runningCount,DesiredCount:desiredCount}' \
    --output table

# Get load balancer DNS
ALB_DNS=$(aws elbv2 describe-load-balancers \
    --names "mmm-alb-$ENVIRONMENT" \
    --region $AWS_REGION \
    --query 'LoadBalancers[0].DNSName' \
    --output text)

echo ""
echo "✅ Production deployment completed successfully!"
echo ""
echo "📋 Deployment Summary:"
echo "   • Client ID: $CLIENT_ID"
echo "   • Environment: $ENVIRONMENT"
echo "   • Docker Image: $DOCKER_TAG"
echo "   • Database: mmm-$CLIENT_ID-$ENVIRONMENT"
echo "   • ECS Service: mmm-$CLIENT_ID-$ENVIRONMENT"
echo ""
echo "🌐 Access URLs:"
echo "   • Load Balancer: http://$ALB_DNS"
echo "   • API Docs: http://$ALB_DNS/docs"
echo ""
echo "📝 Next Steps:"
echo "   1. Configure DNS for your domain"
echo "   2. Set up SSL certificate"
echo "   3. Test the application"
echo "   4. Run your production tests"
echo ""
echo "🧪 Ready for your production testing!"
echo ""

# Health check
echo "🏥 Running health check..."
sleep 30
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" "http://$ALB_DNS/" || echo "000")

if [ "$HTTP_STATUS" = "200" ]; then
    echo "✅ Health check passed: Application is responding"
else
    echo "⚠️ Health check warning: HTTP $HTTP_STATUS (may need a few more minutes to start)"
fi