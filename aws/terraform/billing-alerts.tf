# Billing Alerts and Cost Monitoring
# This sets up CloudWatch alarms for AWS billing

# Variables for billing alerts
variable "billing_alert_email" {
  description = "Email address for billing alerts"
  type        = string
  default     = ""  # Set this in terraform.tfvars
}

variable "billing_thresholds" {
  description = "List of billing thresholds for alerts"
  type        = list(number)
  default     = [100, 200, 500]  # Alert at $100, $200, $500
}

# SNS Topic for billing alerts
resource "aws_sns_topic" "billing_alerts" {
  name = "mmm-billing-alerts-${var.environment}"

  tags = {
    Name = "mmm-billing-alerts-${var.environment}"
  }
}

# SNS Topic Subscription (email)
resource "aws_sns_topic_subscription" "billing_email" {
  count = var.billing_alert_email != "" ? 1 : 0
  
  topic_arn = aws_sns_topic.billing_alerts.arn
  protocol  = "email"
  endpoint  = var.billing_alert_email
}

# CloudWatch Billing Alarms
resource "aws_cloudwatch_metric_alarm" "billing" {
  count = length(var.billing_thresholds)
  
  alarm_name          = "mmm-billing-alert-${var.billing_thresholds[count.index]}-${var.environment}"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "EstimatedCharges"
  namespace           = "AWS/Billing"
  period              = "86400"  # 24 hours
  statistic           = "Maximum"
  threshold           = var.billing_thresholds[count.index]
  alarm_description   = "Alert when AWS billing exceeds $${var.billing_thresholds[count.index]}"
  alarm_actions       = [aws_sns_topic.billing_alerts.arn]
  treat_missing_data  = "notBreaching"

  dimensions = {
    Currency = "USD"
  }

  tags = {
    Name = "mmm-billing-alert-${var.billing_thresholds[count.index]}"
  }
}

# Budget for cost control
resource "aws_budgets_budget" "mmm_monthly" {
  name         = "mmm-monthly-budget-${var.environment}"
  budget_type  = "COST"
  limit_amount = "300"  # $300 monthly limit
  limit_unit   = "USD"
  time_unit    = "MONTHLY"
  
  cost_filters = {
    Service = [
      "Amazon Elastic Container Service",
      "Amazon Relational Database Service", 
      "Amazon ElastiCache",
      "Amazon Elastic Load Balancing",
      "Amazon Simple Storage Service",
      "Amazon CloudWatch",
      "Amazon Virtual Private Cloud"
    ]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 80  # Alert at 80% of budget
    threshold_type            = "PERCENTAGE"
    notification_type         = "ACTUAL"
    subscriber_email_addresses = var.billing_alert_email != "" ? [var.billing_alert_email] : []
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 100  # Alert at 100% of budget
    threshold_type            = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = var.billing_alert_email != "" ? [var.billing_alert_email] : []
  }
  
  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                 = 90   # Forecast alert at 90%
    threshold_type            = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = var.billing_alert_email != "" ? [var.billing_alert_email] : []
  }
}

# Cost Anomaly Detection
resource "aws_ce_anomaly_detector" "mmm_service" {
  name         = "mmm-anomaly-detector-${var.environment}"
  monitor_type = "DIMENSIONAL"

  specification = jsonencode({
    Dimension = "SERVICE"
    MatchOptions = ["EQUALS"]
    Values = [
      "Amazon Elastic Container Service",
      "Amazon Relational Database Service",
      "Amazon ElastiCache"
    ]
  })
}

resource "aws_ce_anomaly_subscription" "mmm_anomaly_email" {
  count = var.billing_alert_email != "" ? 1 : 0
  
  name      = "mmm-anomaly-subscription-${var.environment}"
  frequency = "DAILY"
  
  monitor_arn_list = [
    aws_ce_anomaly_detector.mmm_service.arn
  ]
  
  subscriber {
    type    = "EMAIL"
    address = var.billing_alert_email
  }

  threshold_expression {
    and {
      dimension {
        key           = "ANOMALY_TOTAL_IMPACT_ABSOLUTE"
        values        = ["50"]  # Alert on anomalies > $50
        match_options = ["GREATER_THAN_OR_EQUAL"]
      }
    }
  }
}

# Output the SNS topic ARN
output "billing_alerts_topic_arn" {
  description = "ARN of the SNS topic for billing alerts"
  value       = aws_sns_topic.billing_alerts.arn
}

output "budget_name" {
  description = "Name of the AWS Budget"
  value       = aws_budgets_budget.mmm_monthly.name
}