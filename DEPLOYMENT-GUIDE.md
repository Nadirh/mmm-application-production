# 🚀 MMM Application Production Deployment Guide

## Prerequisites

**Required Tools:**
- AWS CLI v2.x configured with appropriate permissions
- Terraform v1.0+
- Docker
- Python 3.11+

**AWS Permissions Required:**
- EC2, ECS, RDS, ElastiCache, S3, IAM, CloudWatch
- VPC, ALB, Route53 (if using custom domain)
- Billing and Cost Management

## Step-by-Step Deployment

### 1. Clone and Setup

```bash
git clone <your-repo>
cd mmm-application

# Verify your AWS credentials
aws sts get-caller-identity
```

### 2. Configure Billing Alerts

```bash
cd aws/scripts
chmod +x setup-billing-alerts.sh
./setup-billing-alerts.sh nadirh@gmail.com
```

**Important**: You'll receive an email at **nadirh@gmail.com** to confirm SNS subscription.

### 3. Deploy Infrastructure and Application

```bash
cd aws/scripts
chmod +x deploy-production.sh
./deploy-production.sh mmm-demo
```

This will:
- 🏗️ Create complete AWS infrastructure (~10 minutes)
- 🐳 Build and push Docker image to ECR
- 🚀 Deploy MMM application to ECS
- 🔐 Configure database connections
- 🏥 Run health checks

### 4. Verify Deployment

The script will output:
```
✅ Production deployment completed successfully!

🌐 Access URLs:
   • Load Balancer: http://mmm-alb-production-XXXXXXX.us-east-1.elb.amazonaws.com
   • API Docs: http://mmm-alb-production-XXXXXXX.us-east-1.elb.amazonaws.com/docs
```

### 5. Run Production Tests

```bash
cd scripts
pip install httpx websockets pandas numpy

# Use your actual ALB DNS from step 4
python3 production-test-suite.py --url http://YOUR-ALB-DNS-HERE
```

## Expected Costs

**Month 1 (~$180-220):**
- Shared Infrastructure: ~$100
- Your Client Resources: ~$80-120
- **First $100 alert**: Within 15-20 days

## Troubleshooting

### If Terraform Fails:
```bash
cd aws/terraform
terraform init
terraform plan -var-file="terraform.tfvars"
# Fix any issues, then
terraform apply -var-file="terraform.tfvars"
```

### If Docker Build Fails:
```bash
# Check Docker is running
docker info

# Manual build and push
docker build -t mmm-application .
```

### If ECS Service Won't Start:
```bash
# Check ECS service logs
aws ecs describe-services --cluster mmm-cluster-production --services mmm-mmm-demo-production

# Check task logs in CloudWatch
```

## Security Notes

- Database passwords auto-generated and stored in AWS Systems Manager
- All traffic encrypted in transit
- VPC with private subnets for database/Redis
- Security groups restrict access appropriately

## Monitoring

**Billing Alerts Setup:**
- ✅ $100 threshold → **nadirh@gmail.com**
- ✅ $200 threshold → **nadirh@gmail.com** 
- ✅ $500 threshold → **nadirh@gmail.com**
- ✅ Monthly budget: $300 limit
- ✅ Anomaly detection: Unusual spending patterns

**Application Monitoring:**
- CloudWatch logs for all services
- ECS health checks and auto-restart
- RDS performance insights
- Redis cluster monitoring

## Next Steps After Deployment

1. **Confirm email subscription** (check spam folder)
2. **Test the application** with production test suite
3. **Set up custom domain** (optional)
4. **Configure SSL certificate** (recommended)
5. **Add more clients** using `./add-client.sh`

## Adding More Clients

```bash
cd aws/scripts
./add-client.sh client-name client.domain.com
```

Each client adds ~$140/month to your bill.

## Production Testing Results

The test suite will validate:
- ✅ Health endpoints and API docs
- ✅ Data upload (400-day realistic dataset)
- ✅ Model training with WebSocket updates
- ✅ Model results (MAPE, R², channel performance)
- ✅ Response curves generation
- ✅ Budget optimization with constraints
- ✅ Performance under load

**Success criteria**: All tests pass with reasonable performance metrics.

---

**🎉 Your MMM application will be production-ready with full AWS cloud-native infrastructure!**