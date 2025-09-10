# This file defines the template for per-client resources
# Use this with Terraform modules or duplicate for each client

# Variables for client-specific configuration
variable "client_configs" {
  description = "Map of client configurations"
  type = map(object({
    client_id     = string
    domain_name   = string
    db_instance   = string
    min_capacity  = number
    max_capacity  = number
  }))
  default = {
    # Example client configuration
    # "client1" = {
    #   client_id     = "acme-corp"
    #   domain_name   = "acme.mmm.yourdomain.com"
    #   db_instance   = "db.t3.small"
    #   min_capacity  = 1
    #   max_capacity  = 3
    # }
  }
}

# RDS Subnet Group (shared across client databases)
resource "aws_db_subnet_group" "main" {
  name       = "mmm-db-subnet-${var.environment}"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "mmm-db-subnet-${var.environment}"
  }
}

# Per-client resources using for_each
# S3 Buckets for each client
resource "aws_s3_bucket" "client_data" {
  for_each = var.client_configs
  
  bucket = "mmm-${each.value.client_id}-${var.environment}-${random_id.bucket_suffix[each.key].hex}"
  
  tags = {
    Name     = "mmm-${each.value.client_id}-${var.environment}"
    ClientId = each.value.client_id
  }
}

resource "random_id" "bucket_suffix" {
  for_each = var.client_configs
  
  byte_length = 4
  keepers = {
    client_id = each.value.client_id
  }
}

resource "aws_s3_bucket_versioning" "client_data" {
  for_each = var.client_configs
  
  bucket = aws_s3_bucket.client_data[each.key].id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "client_data" {
  for_each = var.client_configs
  
  bucket = aws_s3_bucket.client_data[each.key].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# RDS PostgreSQL instances for each client
resource "aws_db_instance" "client_db" {
  for_each = var.client_configs
  
  identifier = "mmm-${each.value.client_id}-${var.environment}"
  
  # Database configuration
  engine                 = "postgres"
  engine_version         = "15.4"
  instance_class         = each.value.db_instance
  allocated_storage      = 20
  max_allocated_storage  = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  
  # Database settings
  db_name  = "mmm_${replace(each.value.client_id, "-", "_")}"
  username = "mmm_admin"
  password = random_password.db_password[each.key].result
  
  # Network and security
  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  
  # Backup and maintenance
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  # Performance and monitoring
  performance_insights_enabled = true
  monitoring_interval         = 60
  monitoring_role_arn        = aws_iam_role.rds_monitoring.arn
  
  # Deletion protection for production
  deletion_protection = var.environment == "production" ? true : false
  skip_final_snapshot = var.environment != "production"
  
  tags = {
    Name     = "mmm-${each.value.client_id}-db-${var.environment}"
    ClientId = each.value.client_id
  }
}

# Random passwords for databases
resource "random_password" "db_password" {
  for_each = var.client_configs
  
  length  = 16
  special = true
}

# Store database passwords in Systems Manager Parameter Store
resource "aws_ssm_parameter" "db_password" {
  for_each = var.client_configs
  
  name  = "/mmm/${each.value.client_id}/database/password"
  type  = "SecureString"
  value = random_password.db_password[each.key].result

  tags = {
    Name     = "mmm-${each.value.client_id}-db-password"
    ClientId = each.value.client_id
  }
}

# ALB Target Group for each client
resource "aws_lb_target_group" "client_app" {
  for_each = var.client_configs
  
  name     = "mmm-${each.value.client_id}-${var.environment}"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = aws_vpc.main.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name     = "mmm-${each.value.client_id}-tg-${var.environment}"
    ClientId = each.value.client_id
  }
}

# ECS Service for each client
resource "aws_ecs_service" "client_app" {
  for_each = var.client_configs
  
  name            = "mmm-${each.value.client_id}-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.client_app[each.key].arn
  desired_count   = each.value.min_capacity
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.private[*].id
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.client_app[each.key].arn
    container_name   = "mmm-app"
    container_port   = 8000
  }

  # Auto scaling configuration
  lifecycle {
    ignore_changes = [desired_count]
  }

  depends_on = [
    aws_lb_listener.main,
    aws_iam_role_policy_attachment.ecs_task_execution,
  ]

  tags = {
    Name     = "mmm-${each.value.client_id}-service-${var.environment}"
    ClientId = each.value.client_id
  }
}

# ECS Task Definition for each client
resource "aws_ecs_task_definition" "client_app" {
  for_each = var.client_configs
  
  family                   = "mmm-${each.value.client_id}-${var.environment}"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  execution_role_arn       = aws_iam_role.ecs_task_execution.arn
  task_role_arn           = aws_iam_role.ecs_task.arn

  container_definitions = jsonencode([
    {
      name  = "mmm-app"
      image = "${aws_ecr_repository.mmm_app.repository_url}:latest"
      
      portMappings = [
        {
          containerPort = 8000
          protocol      = "tcp"
        }
      ]
      
      environment = [
        {
          name  = "MMM_ENV"
          value = var.environment
        },
        {
          name  = "CLIENT_ID"
          value = each.value.client_id
        },
        {
          name  = "REDIS_URL"
          value = "redis://${aws_elasticache_replication_group.redis.primary_endpoint_address}:6379"
        }
      ]
      
      secrets = [
        {
          name      = "DATABASE_URL"
          valueFrom = "/mmm/${each.value.client_id}/database/url"
        }
      ]
      
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.mmm.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "ecs-${each.value.client_id}"
        }
      }
      
      healthCheck = {
        command = ["CMD-SHELL", "curl -f http://localhost:8000/ || exit 1"]
        interval = 30
        timeout = 5
        retries = 3
      }
    }
  ])

  tags = {
    Name     = "mmm-${each.value.client_id}-task-${var.environment}"
    ClientId = each.value.client_id
  }
}