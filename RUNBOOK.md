# Montage Production Runbook

## System Requirements

### Minimum Requirements
- CPU: 8 cores (Intel/AMD x64 or Apple Silicon)
- RAM: 16GB
- Storage: 100GB free space
- OS: macOS 12+, Ubuntu 20.04+, Windows 10+ with WSL2

### Recommended Requirements (Your M4 Max Config)
- CPU: 14-core Apple M4 Max ✅
- GPU: 32-core GPU (excellent for video processing) ✅
- RAM: 36GB unified memory (perfect for 4K video) ✅
- Storage: 1TB SSD ✅
- Neural Engine: 16-core (accelerates ML tasks) ✅

Your M4 Max is ideal for Montage - expect 2-3x faster processing than baseline.

## Pre-requisites

### 1. System Dependencies
```bash
# macOS (Apple Silicon optimized)
brew install ffmpeg@6 postgresql@15 redis python@3.11
brew install --cask docker

# Install ML dependencies optimized for Apple Silicon
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install tensorflow-macos tensorflow-metal  # For Neural Engine acceleration
```

### 2. Python Environment
```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On macOS/Linux

# Install requirements
pip install -r requirements.txt

# Apple Silicon optimizations
pip install pyannote.audio --extra-index-url https://pypi.ngc.nvidia.com
```

### 3. API Keys
Create `.env` file:
```bash
# Required
OPENAI_API_KEY=your_key_here
DEEPGRAM_API_KEY=your_key_here

# Optional but recommended
ANTHROPIC_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here  # For speaker diarization

# Security
JWT_SECRET_KEY=$(openssl rand -base64 32)
API_KEY=$(openssl rand -base64 32)

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=montage
DB_USER=montage
DB_PASS=secure_password_here

# Redis
REDIS_URL=redis://localhost:6379
```

### 4. Database Setup
```bash
# Start PostgreSQL
brew services start postgresql@15

# Create database
createdb montage
psql montage < infrastructure/postgres/init.sql

# Verify
psql montage -c "SELECT version();"
```

### 5. Model Downloads
```bash
# Download required models
python scripts/download_models.py

# This downloads:
# - Whisper models for transcription
# - PyAnnote models for speaker diarization
# - OpenCV face detection models
# - MMTracking models for object tracking
```

## Deployment

### Local Development
```bash
# Start services
brew services start postgresql@15
brew services start redis

# Run application
python main.py

# Or use Docker
docker-compose up -d
```

### Production Deployment

#### 1. Using Docker Compose
```bash
# Build and start all services
docker-compose up -d --build

# Scale workers based on your M4 Max (14 cores)
docker-compose up -d --scale celery-worker=6

# Monitor
docker-compose logs -f montage-app
docker stats
```

#### 2. Kubernetes Deployment
```yaml
# montage-deployment.yaml optimized for M4 Max
apiVersion: apps/v1
kind: Deployment
metadata:
  name: montage-app
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: montage
        image: montage:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            # Apple Silicon GPU access
            apple.com/gpu: 1
```

## Health Checks

### Application Health
```bash
# Basic health check
curl http://localhost:8000/health

# Expected response
{
  "status": "healthy",
  "timestamp": "2025-07-24T...",
  "services": {
    "database": "healthy",
    "redis": "healthy"
  }
}
```

### System Monitoring
```bash
# CPU/Memory for M4 Max
sudo powermetrics --samplers cpu_power,gpu_power --show-process-energy

# GPU usage (Metal Performance Shaders)
ioreg -l | grep "PerformanceStatistics"

# Disk I/O
iostat -d 1

# Process monitoring
htop
```

### Database Health
```bash
# Connection check
psql montage -c "SELECT 1;"

# Active queries
psql montage -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Table sizes
psql montage -c "SELECT relname, pg_size_pretty(pg_total_relation_size(relid)) 
FROM pg_catalog.pg_statio_user_tables ORDER BY pg_total_relation_size(relid) DESC;"
```

## Common Operations

### Process Video
```bash
# Basic processing
python -m montage.cli.run_pipeline video.mp4 --output highlights.mp4

# Optimized for M4 Max Neural Engine
python -m montage.cli.run_pipeline video.mp4 \
  --output highlights.mp4 \
  --use-neural-engine \
  --threads 14 \
  --gpu-acceleration metal

# Batch processing utilizing all cores
find ./videos -name "*.mp4" | parallel -j 14 \
  python -m montage.cli.run_pipeline {} --output-dir ./processed/
```

### Clear Cache
```bash
# Redis cache
redis-cli FLUSHDB

# Temporary files
rm -rf /tmp/video_processing/*
rm -rf output/temp/*

# Docker volumes
docker volume prune -f
```

### Database Operations
```bash
# Backup
pg_dump montage > backup_$(date +%Y%m%d).sql

# Restore
psql montage < backup_20250724.sql

# Vacuum and analyze
psql montage -c "VACUUM ANALYZE;"

# Reset sequences
psql montage -c "SELECT setval('jobs_id_seq', (SELECT MAX(id) FROM jobs));"
```

## Troubleshooting

### Memory Issues on M4 Max
Despite 36GB RAM, video processing can be memory-intensive:

```bash
# Monitor memory pressure
vm_stat 1

# Adjust memory limits
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7  # Limit Metal usage
export VIDEO_PROCESSING_MAX_MEMORY=24G

# Use memory-optimized settings
python -m montage.cli.run_pipeline video.mp4 \
  --memory-limit 24G \
  --processing-mode balanced
```

### GPU/Neural Engine Issues
```bash
# Verify Metal support
python -c "import torch; print(torch.backends.mps.is_available())"

# Check Neural Engine
python -c "import coremltools as ct; print(ct.utils.get_available_compute_units())"

# Fallback to CPU if needed
export FORCE_CPU_PROCESSING=1
```

### Performance Tuning for M4 Max
```bash
# Optimize for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export TF_ENABLE_MLCOMPUTE=1
export OMP_NUM_THREADS=14

# Video codec optimization
export FFMPEG_ENCODER="hevc_videotoolbox"  # Hardware encoding
export FFMPEG_DECODER="hevc_videotoolbox"  # Hardware decoding
```

### Common Errors

#### "FFmpeg process killed"
```bash
# Increase memory limits
ulimit -m unlimited
ulimit -v unlimited

# Use lower resolution processing
python -m montage.cli.run_pipeline video.mp4 --max-resolution 1080p
```

#### "Database connection refused"
```bash
# Check PostgreSQL status
brew services list | grep postgresql

# Restart if needed
brew services restart postgresql@15

# Check logs
tail -f /opt/homebrew/var/log/postgresql@15.log
```

#### "PyAnnote model download failed"
```bash
# Manual download with HuggingFace token
huggingface-cli login --token $HUGGINGFACE_TOKEN
huggingface-cli download pyannote/speaker-diarization
```

## Monitoring & Logs

### Application Logs
```bash
# Main application
tail -f logs/montage.log

# Worker logs
tail -f logs/celery_worker.log

# Error logs
grep ERROR logs/*.log | tail -50
```

### Metrics Endpoint
```bash
# Get metrics (requires API key)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/metrics

# Prometheus format
curl -H "X-API-Key: $API_KEY" http://localhost:8000/metrics?format=prometheus
```

### Performance Metrics
```bash
# Processing statistics
psql montage -c "SELECT 
  AVG(processing_time) as avg_time,
  MAX(processing_time) as max_time,
  COUNT(*) as total_jobs
FROM jobs WHERE created_at > NOW() - INTERVAL '24 hours';"
```

## Backup & Recovery

### Automated Backups
```bash
# Add to crontab
0 3 * * * pg_dump montage | gzip > /backups/montage_$(date +\%Y\%m\%d).sql.gz

# Backup processed videos
0 4 * * * rsync -av --progress output/ /backups/videos/
```

### Disaster Recovery
```bash
# Full system restore
1. Restore database: gunzip -c backup.sql.gz | psql montage
2. Restore Redis: redis-cli --rdb /backups/redis/dump.rdb
3. Restore videos: rsync -av /backups/videos/ output/
4. Restart services: docker-compose up -d
```

## Security Checklist

- [ ] API keys in environment variables (never in code)
- [ ] JWT_SECRET_KEY is unique and secure
- [ ] Database password is strong
- [ ] Redis is not exposed publicly
- [ ] File upload size limits configured
- [ ] Path traversal protection enabled
- [ ] Rate limiting active on all endpoints
- [ ] HTTPS enabled in production
- [ ] Regular security updates applied

## Performance Benchmarks (M4 Max)

| Operation | Expected Time | Memory Usage |
|-----------|--------------|--------------|
| 1080p 10min transcription | 45-60s | 2-3GB |
| 4K 10min transcription | 90-120s | 4-6GB |
| Face detection (1080p) | 15-20 fps | 1-2GB |
| Object tracking (1080p) | 10-15 fps | 2-3GB |
| Neural Engine inference | 50-100ms/frame | 1GB |
| Video encoding (HEVC) | 2-3x realtime | 2-4GB |

## Support

- GitHub Issues: https://github.com/your-org/montage/issues
- Logs: Check `/app/logs/` in containers
- Metrics: http://localhost:8000/metrics
- Health: http://localhost:8000/health

Remember: Your M4 Max with 36GB unified memory and Neural Engine provides exceptional performance for video AI tasks. Utilize Metal acceleration and Neural Engine whenever possible for optimal results.