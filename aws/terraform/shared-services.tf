# ECS Cluster (shared across all clients)
resource "aws_ecs_cluster" "main" {
  name = "mmm-cluster-${var.environment}"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Name = "mmm-cluster-${var.environment}"
  }
}

# Application Load Balancer (shared)
resource "aws_lb" "main" {
  name               = "mmm-alb-${var.environment}"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  enable_deletion_protection = var.environment == "production" ? true : false

  tags = {
    Name = "mmm-alb-${var.environment}"
  }
}

# ECR Repository
resource "aws_ecr_repository" "mmm_app" {
  name                 = "mmm-application"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle_policy {
    policy = jsonencode({
      rules = [
        {
          rulePriority = 1
          description  = "Keep last 30 production images"
          selection = {
            tagStatus     = "tagged"
            tagPrefixList = ["v"]
            countType     = "imageCountMoreThan"
            countNumber   = 30
          }
          action = {
            type = "expire"
          }
        },
        {
          rulePriority = 2
          description  = "Keep last 10 untagged images"
          selection = {
            tagStatus   = "untagged"
            countType   = "imageCountMoreThan"
            countNumber = 10
          }
          action = {
            type = "expire"
          }
        }
      ]
    })
  }
}

# Redis Cluster (shared across clients)
resource "aws_elasticache_subnet_group" "main" {
  name       = "mmm-cache-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id
}

resource "aws_elasticache_replication_group" "redis" {
  replication_group_id         = "mmm-redis-${var.environment}"
  description                  = "Redis cluster for MMM application"
  
  port                         = 6379
  parameter_group_name         = "default.redis7.cluster.on"
  node_type                    = "cache.t3.micro"
  num_cache_clusters           = 2
  
  engine_version               = "7.0"
  subnet_group_name            = aws_elasticache_subnet_group.main.name
  security_group_ids           = [aws_security_group.redis.id]
  
  at_rest_encryption_enabled   = true
  transit_encryption_enabled   = true
  
  # Enable automatic failover for high availability
  automatic_failover_enabled   = true
  multi_az_enabled            = true
  
  # Backup settings
  snapshot_retention_limit     = 7
  snapshot_window             = "03:00-05:00"
  maintenance_window          = "sun:05:00-sun:07:00"
  
  tags = {
    Name = "mmm-redis-${var.environment}"
  }
}

# CloudWatch Log Group
resource "aws_cloudwatch_log_group" "mmm" {
  name              = "/ecs/mmm-application-${var.environment}"
  retention_in_days = 7

  tags = {
    Name = "mmm-logs-${var.environment}"
  }
}

# IAM Role for ECS Tasks
resource "aws_iam_role" "ecs_task_execution" {
  name = "mmm-ecs-task-execution-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_task_execution" {
  role       = aws_iam_role.ecs_task_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

# IAM Role for ECS Tasks (runtime)
resource "aws_iam_role" "ecs_task" {
  name = "mmm-ecs-task-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

# Policy for ECS tasks to access S3, SSM, etc.
resource "aws_iam_role_policy" "ecs_task" {
  name = "mmm-ecs-task-policy-${var.environment}"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::mmm-*",
          "arn:aws:s3:::mmm-*/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:*:*:parameter/mmm/*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}