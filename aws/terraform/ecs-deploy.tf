# Temporary ECS service deployment file
# This will deploy the ECS service using existing working infrastructure

# First, let's create a working target group
resource "aws_lb_target_group" "ecs_temp" {
  name        = "mmm-ecs-temp-${var.environment}"
  port        = 8000
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/api/health/"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }

  tags = {
    Name = "mmm-ecs-temp-tg-${var.environment}"
  }
}

# Deploy ECS service directly
resource "aws_ecs_service" "mmm_temp" {
  name            = "mmm-temp-${var.environment}"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.client_app["client_1"].arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    security_groups = [aws_security_group.ecs.id]
    subnets         = aws_subnet.private[*].id
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.ecs_temp.arn
    container_name   = "mmm-app"
    container_port   = 8000
  }

  depends_on = [aws_lb_listener.main]

  tags = {
    Name = "mmm-temp-service-${var.environment}"
  }
}

# Update listener to use new target group
resource "aws_lb_listener_rule" "ecs_temp" {
  listener_arn = aws_lb_listener.main.arn
  priority     = 100

  action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ecs_temp.arn
  }

  condition {
    path_pattern {
      values = ["/*"]
    }
  }
}