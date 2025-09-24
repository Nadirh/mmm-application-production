#!/bin/bash
# Deployment script for MMM application with authentication enabled

set -e

echo "=== MMM Application Deployment with Authentication ==="
echo "======================================================="

# Configuration
VERSION="${1:-v1.9.81}"  # Use first argument or default to v1.9.81
ECR_REPO="727529935876.dkr.ecr.us-east-2.amazonaws.com/mmm-application"
AWS_REGION="us-east-2"

# Set production environment variables
export MMM_ENV="production"
export AUTH_USERNAME="${AUTH_USERNAME:-mmm_admin}"
export AUTH_PASSWORD="${AUTH_PASSWORD:-SecureMMM2024!@#}"
export SESSION_DURATION_HOURS="${SESSION_DURATION_HOURS:-8}"

echo ""
echo "1. Building Docker image with version $VERSION..."
env -u GITHUB_TOKEN docker build -t mmm-application:$VERSION .

echo ""
echo "2. Tagging for ECR..."
docker tag mmm-application:$VERSION $ECR_REPO:$VERSION
docker tag mmm-application:$VERSION $ECR_REPO:latest

echo ""
echo "3. Authenticating with ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO

echo ""
echo "4. Pushing to ECR..."
env -u GITHUB_TOKEN docker push $ECR_REPO:$VERSION
env -u GITHUB_TOKEN docker push $ECR_REPO:latest

echo ""
echo "5. Updating ECS service..."
# Update the task definition with new environment variables
aws ecs update-service \
    --cluster mmm-cluster-production \
    --service mmm-service-production \
    --force-new-deployment \
    --region $AWS_REGION

echo ""
echo "=== Deployment Complete ==="
echo "Version: $VERSION"
echo "Authentication: Enabled"
echo "Username: $AUTH_USERNAME"
echo "Session Duration: $SESSION_DURATION_HOURS hours"
echo ""
echo "Note: Password is set via environment variable AUTH_PASSWORD"
echo "Make sure to update the ECS task definition with the authentication environment variables:"
echo "  - MMM_ENV=production"
echo "  - AUTH_USERNAME=<your_username>"
echo "  - AUTH_PASSWORD=<your_secure_password>"
echo "  - SESSION_DURATION_HOURS=<hours>"
echo ""
echo "Production URL: http://mmm-alb-production-190214907.us-east-2.elb.amazonaws.com/"
echo "==========================="