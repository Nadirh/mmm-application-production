# Docker Deployment Guide

This guide covers deploying the MMM application using Docker containers with Redis integration.

## Quick Start

### Development Environment

1. **Start services with Docker Compose:**
   ```bash
   make docker-up-build
   ```

2. **Access the application:**
   - API: http://localhost:8000
   - WebSocket Test: http://localhost:8000/static/websocket_test.html
   - API Documentation: http://localhost:8000/docs

3. **View logs:**
   ```bash
   make docker-logs
   ```

4. **Stop services:**
   ```bash
   make docker-down
   ```

### Production Environment

1. **Start production services:**
   ```bash
   make docker-prod-build
   ```

2. **Access via Nginx (if enabled):**
   - HTTP: http://localhost:80
   - HTTPS: http://localhost:443 (requires SSL certificates)

3. **Monitor production logs:**
   ```bash
   make docker-prod-logs
   ```

## Architecture

### Services

1. **mmm-app**: Main application container
   - FastAPI web server
   - Model training engine
   - WebSocket support
   - SQLite database

2. **redis**: Cache and session storage
   - Redis 7.x Alpine
   - Persistent data storage
   - Memory optimization

3. **nginx** (production only): Reverse proxy
   - Load balancing
   - SSL termination
   - Rate limiting
   - Static file serving

### Volumes

- `app_data`: SQLite database and persistent data
- `app_logs`: Application logs
- `app_uploads`: Uploaded CSV files
- `redis_data`: Redis persistence

### Networks

- `mmm-network`: Internal bridge network for service communication

## Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MMM_ENV` | Environment (development/production) | development | No |
| `DATABASE_URL` | SQLite database URL | sqlite:///app/data/mmm_app.db | No |
| `REDIS_URL` | Redis connection URL | redis://redis:6379/0 | No |
| `HOST` | Server bind address | 0.0.0.0 | No |
| `PORT` | Server port | 8000 | No |
| `LOG_LEVEL` | Logging level | INFO | No |

### Volume Mounts

**Development:**
- Source code is mounted for hot reloading
- Database and logs are persisted

**Production:**
- Only data volumes are mounted
- Application code is baked into image

## Resource Requirements

### Minimum Requirements

- **CPU**: 1 core
- **Memory**: 2GB RAM
- **Storage**: 10GB disk space
- **Network**: 1Gbps (for large file uploads)

### Recommended for Production

- **CPU**: 2-4 cores
- **Memory**: 4-8GB RAM
- **Storage**: 50GB+ SSD
- **Network**: 1Gbps+ with low latency

### Resource Limits (Production)

```yaml
deploy:
  resources:
    limits:
      cpus: '2.0'
      memory: 4G
    reservations:
      cpus: '1.0'
      memory: 2G
```

## Database Management

### Initialize Database

```bash
# Start services
make docker-up

# Access container shell
make docker-shell

# Run migrations
alembic upgrade head
```

### Backup Database

```bash
# Copy database from container
docker cp mmm-app:/app/data/mmm_app.db ./backup/mmm_app_$(date +%Y%m%d).db

# Or use volume backup
docker run --rm -v mmm_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/mmm_data_$(date +%Y%m%d).tar.gz -C /data .
```

### Restore Database

```bash
# Stop services
make docker-down

# Restore from backup
docker run --rm -v mmm_data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/mmm_data_YYYYMMDD.tar.gz -C /data

# Start services
make docker-up
```

## Monitoring and Health Checks

### Health Endpoints

- **Application**: `GET /api/health/`
- **Detailed Health**: `GET /api/health/detailed`
- **WebSocket Stats**: `GET /ws/stats`

### Container Health Checks

```bash
# Check container status
make docker-ps

# View health check logs
docker inspect mmm-app --format='{{.State.Health.Status}}'

# Manual health check
curl -f http://localhost:8000/api/health/
```

### Log Monitoring

```bash
# Follow all logs
make docker-logs

# Application logs only
docker-compose logs -f mmm-app

# Redis logs only
docker-compose logs -f redis

# Filter error logs
docker-compose logs mmm-app | grep ERROR
```

## Security

### Network Security

- Services communicate over internal bridge network
- Redis is not exposed externally in production
- Nginx provides external access control

### Data Security

- SQLite database stored in persistent volumes
- Redis configured with memory limits
- File uploads restricted to designated volume

### SSL/TLS (Production)

1. **Obtain SSL certificates:**
   ```bash
   # Using Let's Encrypt
   certbot certonly --standalone -d your-domain.com
   ```

2. **Copy certificates:**
   ```bash
   mkdir -p nginx/ssl
   cp /etc/letsencrypt/live/your-domain.com/fullchain.pem nginx/ssl/cert.pem
   cp /etc/letsencrypt/live/your-domain.com/privkey.pem nginx/ssl/key.pem
   ```

3. **Update nginx configuration and restart:**
   ```bash
   make docker-prod-down
   make docker-prod-up
   ```

## Scaling and Performance

### Horizontal Scaling

For multiple clients, deploy separate container stacks:

```bash
# Client 1
docker-compose -f docker-compose.prod.yml -p mmm-client1 up -d

# Client 2  
docker-compose -f docker-compose.prod.yml -p mmm-client2 up -d
```

### Performance Tuning

1. **Redis Optimization:**
   - Adjust `maxmemory` based on available RAM
   - Monitor memory usage: `make docker-redis-cli` → `INFO memory`

2. **Application Tuning:**
   - Increase worker processes for CPU-intensive tasks
   - Adjust timeout values for long training sessions

3. **Storage Performance:**
   - Use SSD storage for database volumes
   - Consider separate volumes for different data types

## Troubleshooting

### Common Issues

1. **Container won't start:**
   ```bash
   # Check logs
   make docker-logs
   
   # Check resource usage
   docker stats
   
   # Verify port availability
   netstat -tlnp | grep 8000
   ```

2. **Redis connection errors:**
   ```bash
   # Test Redis connectivity
   make docker-redis-cli
   > ping
   
   # Check Redis logs
   docker-compose logs redis
   ```

3. **Database migration errors:**
   ```bash
   # Access container shell
   make docker-shell
   
   # Check migration status
   alembic current
   
   # Apply migrations
   alembic upgrade head
   ```

4. **Performance issues:**
   ```bash
   # Monitor resource usage
   docker stats
   
   # Check application metrics
   curl http://localhost:8000/api/health/detailed
   ```

### Reset and Clean Up

```bash
# Reset all data (WARNING: destroys all data)
make docker-reset

# Clean up unused resources
make docker-clean

# Remove specific volumes
docker volume rm mmm_app_data_prod
```

## Backup and Disaster Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backup/mmm"

mkdir -p $BACKUP_DIR

# Backup database
docker run --rm -v mmm_data:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/database_$DATE.tar.gz -C /data .

# Backup uploads
docker run --rm -v mmm_uploads:/data -v $BACKUP_DIR:/backup alpine \
  tar czf /backup/uploads_$DATE.tar.gz -C /data .

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

### Disaster Recovery

1. **Stop services**
2. **Restore from backup**
3. **Verify data integrity**
4. **Start services**
5. **Test functionality**

This containerized deployment provides a robust, scalable foundation for the MMM application with proper separation of concerns and production-ready configuration.