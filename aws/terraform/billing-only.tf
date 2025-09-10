# Minimal billing alerts setup
# This file can be used to deploy just billing alerts first

# Variables for billing alerts
variable "billing_alert_email_only" {
  description = "Email address for billing alerts"
  type        = string
  default     = "nadirh@gmail.com"
}

# SNS Topic for billing alerts
resource "aws_sns_topic" "billing_only" {
  name = "mmm-billing-alerts-production"

  tags = {
    Name = "mmm-billing-alerts-production"
  }
}

# SNS Topic Subscription (email)
resource "aws_sns_topic_subscription" "billing_only" {
  topic_arn = aws_sns_topic.billing_only.arn
  protocol  = "email"
  endpoint  = var.billing_alert_email_only
}

# CloudWatch Billing Alarms
resource "aws_cloudwatch_metric_alarm" "billing_100" {
  alarm_name          = "mmm-billing-alert-100-production"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 24 hours
  statistic           = "Maximum"
  threshold           = "100"
  alarm_description   = "Alert when AWS billing exceeds $100"
  alarm_actions       = [aws_sns_topic.billing_only.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name = "mmm-billing-alert-100"
  }
}

# Budget for cost control
resource "aws_budgets_budget" "billing_only" {
  name         = "mmm-monthly-budget-production"
  budget_type  = "COST"
  limit_amount = "100"  # $100 monthly limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80  # Alert at 80% of budget
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = [var.billing_alert_email_only]
  }
}

# Output the SNS topic ARN
output "billing_topic_arn" {
  description = "ARN of the SNS topic for billing alerts"
  value       = aws_sns_topic.billing_only.arn
}