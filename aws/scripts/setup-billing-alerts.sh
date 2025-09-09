#!/bin/bash
set -e

# Script to set up billing alerts for MMM Application
# Usage: ./setup-billing-alerts.sh <your-email@example.com>

EMAIL="${1:-}"
AWS_REGION="${AWS_REGION:-us-east-1}"

if [ -z "$EMAIL" ]; then
    echo "Usage: $0 <your-email@example.com>"
    echo "Example: $0 john@company.com"
    exit 1
fi

echo "🔔 Setting up billing alerts for MMM Application"
echo "   Email: $EMAIL"
echo "   Region: $AWS_REGION"
echo ""

# Validate email format
if [[ ! "$EMAIL" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
    echo "❌ Invalid email format: $EMAIL"
    exit 1
fi

# Check if billing alerts are enabled (they need to be enabled in us-east-1)
echo "📊 Checking billing preferences..."

# Enable billing alerts via AWS CLI (this needs to be done once per account)
aws ce put-preferences --region us-east-1 --receive-billing-alerts true || {
    echo "⚠️ Could not enable billing alerts via CLI. Please enable manually:"
    echo "   1. Go to AWS Console → Billing → Preferences"
    echo "   2. Enable 'Receive Billing Alerts'"
    echo "   3. Re-run this script"
}

# Create terraform.tfvars with billing configuration
TERRAFORM_DIR="../terraform"
VARS_FILE="$TERRAFORM_DIR/terraform.tfvars"

if [ ! -f "$VARS_FILE" ]; then
    echo "📝 Creating terraform.tfvars with billing configuration..."
    cp "$TERRAFORM_DIR/terraform.tfvars.example" "$VARS_FILE"
fi

# Update billing email in terraform.tfvars
echo "✏️ Updating billing email in terraform.tfvars..."
if grep -q "billing_alert_email" "$VARS_FILE"; then
    # Update existing email
    sed -i.bak "s/billing_alert_email = .*/billing_alert_email = \"$EMAIL\"/" "$VARS_FILE"
else
    # Add email configuration
    echo "" >> "$VARS_FILE"
    echo "# Billing alerts" >> "$VARS_FILE"
    echo "billing_alert_email = \"$EMAIL\"" >> "$VARS_FILE"
    echo "billing_thresholds = [100, 200, 500]" >> "$VARS_FILE"
fi

echo "✅ Updated terraform.tfvars with billing email: $EMAIL"

# Apply Terraform changes
echo "🏗️ Applying billing alert configuration..."
cd "$TERRAFORM_DIR"

terraform plan -var-file="terraform.tfvars" -target=module.billing -out="billing-alerts.plan" 2>/dev/null || {
    # If no module structure, plan normally
    terraform plan -var-file="terraform.tfvars" -out="billing-alerts.plan"
}

echo ""
echo "📋 Terraform will create:"
echo "   • CloudWatch billing alarms at \$100, \$200, \$500"
echo "   • AWS Budget with \$300 monthly limit"
echo "   • SNS topic for email notifications"
echo "   • Cost anomaly detection (alerts on unexpected spikes)"
echo ""

read -p "Apply billing alert configuration? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    terraform apply "billing-alerts.plan"
    
    echo ""
    echo "✅ Billing alerts configured successfully!"
    echo ""
    echo "📧 You will receive an email to confirm your subscription to:"
    echo "   $EMAIL"
    echo ""
    echo "🔔 Alert Summary:"
    echo "   • \$100 threshold: CloudWatch alarm + email"
    echo "   • \$200 threshold: CloudWatch alarm + email" 
    echo "   • \$500 threshold: CloudWatch alarm + email"
    echo "   • \$240 (80% of budget): Budget alert"
    echo "   • \$300 (100% of budget): Budget alert"
    echo "   • Anomaly detection: Unusual spending patterns"
    echo ""
    echo "💡 Pro tips:"
    echo "   • Check AWS Cost Explorer daily for the first week"
    echo "   • Set up AWS Cost and Usage Reports for detailed analysis"
    echo "   • Consider AWS Cost Optimization Hub for recommendations"
    echo ""
    
    # Clean up plan file
    rm -f "billing-alerts.plan"
    
    # Test SNS topic (optional)
    echo "🧪 Would you like to test the alert system? (sends test email)"
    read -p "Send test notification? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        TOPIC_ARN=$(terraform output -raw billing_alerts_topic_arn 2>/dev/null || echo "")
        if [ -n "$TOPIC_ARN" ]; then
            aws sns publish \
                --topic-arn "$TOPIC_ARN" \
                --message "Test alert from MMM Application billing setup. Your billing alerts are working correctly!" \
                --subject "MMM Billing Alert Test" \
                --region "$AWS_REGION"
            echo "📧 Test email sent! Check your inbox."
        else
            echo "⚠️ Could not get SNS topic ARN for testing"
        fi
    fi
    
else
    echo "❌ Billing alerts setup cancelled"
    rm -f "billing-alerts.plan"
fi