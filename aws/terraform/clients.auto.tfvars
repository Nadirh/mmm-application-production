# Client configurations for MMM application
# Add your clients here

client_configs = {
  "client_1" = {
    client_id     = "mmm-demo"
    domain_name   = "demo.mmm.yourdomain.com"  # Update with your actual domain
    db_instance   = "db.t3.small"
    min_capacity  = 1
    max_capacity  = 3
  }
  
  # Example for adding more clients:
  # "client_2" = {
  #   client_id     = "acme-corp"
  #   domain_name   = "acme.mmm.yourdomain.com"
  #   db_instance   = "db.t3.small"
  #   min_capacity  = 1
  #   max_capacity  = 3
  # }
}