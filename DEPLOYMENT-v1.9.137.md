# Deployment v1.9.137 - Stable Release

## Date: September 25, 2025

## Status: ✅ Production Stable

## Overview
This deployment resolves critical production issues including database connection problems and JavaScript errors. This is a known-good version suitable for rollback.

## Critical Fixes Applied
1. **Database Connection Fixed**: Restored PostgreSQL RDS connection (was incorrectly using SQLite)
2. **Table Auto-Creation**: Added automatic database table creation on startup
3. **JavaScript Error Fixed**: Resolved fold parameters table rendering error
4. **Cache Busting**: Updated app.js version string to force browser cache refresh

## Deployment Details

### Docker Image
```
727529935876.dkr.ecr.us-east-2.amazonaws.com/mmm-application:v1.9.137
```

### ECS Configuration
- **Cluster**: mmm-cluster-production
- **Service**: mmm-temp-production
- **Task Definition**: mmm-mmm-demo-production:158
- **Region**: us-east-2

### Database
- **Type**: PostgreSQL RDS
- **Instance**: mmm-mmm-demo-production
- **Endpoint**: mmm-mmm-demo-production.cl4y84wusrzb.us-east-2.rds.amazonaws.com
- **Port**: 5432
- **Database Name**: mmm_mmm_demo

### Environment Variables
```bash
MMM_ENV=production
AUTH_USERNAME=mmm_admin
AUTH_PASSWORD=SecureMMM2024!@#
SESSION_DURATION_HOURS=8
DATABASE_URL=postgresql://[connection_string]
```

## Rollback Instructions

If you need to rollback to this stable version:

### Option 1: Quick Rollback (if recent)
```bash
# Update ECS service to use task definition 158
aws ecs update-service \
  --cluster mmm-cluster-production \
  --service mmm-temp-production \
  --task-definition mmm-mmm-demo-production:158 \
  --force-new-deployment \
  --region us-east-2
```

### Option 2: Full Redeploy
```bash
# 1. Checkout this version
git checkout v1.9.137-stable

# 2. Build and tag Docker image
docker build -t mmm-application:v1.9.137 .
docker tag mmm-application:v1.9.137 727529935876.dkr.ecr.us-east-2.amazonaws.com/mmm-application:v1.9.137

# 3. Push to ECR
aws ecr get-login-password --region us-east-2 | docker login --username AWS --password-stdin 727529935876.dkr.ecr.us-east-2.amazonaws.com
docker push 727529935876.dkr.ecr.us-east-2.amazonaws.com/mmm-application:v1.9.137

# 4. Deploy using task definition 158 or create new one
```

## Testing Checklist
- [x] Health endpoint responding
- [x] Authentication working
- [x] File uploads successful
- [x] Database tables created
- [x] JavaScript errors resolved
- [x] PostgreSQL connection verified

## Known Issues Resolved
- "no such table: upload_sessions" error - **FIXED**
- JavaScript fold parameters table error - **FIXED**
- Database persistence across deployments - **FIXED**

## Production URL
http://mmm-alb-production-190214907.us-east-2.elb.amazonaws.com/

## Git Information
- **Commit**: 7b30200
- **Tag**: v1.9.137-stable
- **Branch**: main

## Notes
This version includes defensive programming with automatic table creation that works with both PostgreSQL (production) and SQLite (local development). The critical fix was restoring the DATABASE_URL environment variable that was missing in v1.9.135-136.