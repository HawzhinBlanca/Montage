# Montage System Architecture

Complete architecture overview of the production video processing pipeline.

## System Overview

Montage is a production-ready video processing pipeline that transforms raw video into optimized highlights using real transcription technology and intelligent content analysis.

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Input Video   │───▶│   Video Analysis │───▶│ Highlight Gen   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │                          │
                              ▼                          ▼
                       ┌──────────────────┐    ┌─────────────────┐
                       │ Real Transcription│    │ Intelligent     │
                       │ (Whisper+Deepgram)│    │ Cropping        │
                       └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │ Professional    │
                                               │ Video Output    │
                                               └─────────────────┘
```

## Core Architecture Principles

### 1. Real Functionality Only
- No mock or fake AI components
- All features use production-ready technology
- Honest capability descriptions

### 2. Performance First
- FIFO-based streaming (zero temp files)
- Parallel processing with optimal worker counts
- Hardware acceleration when available
- < 1.2x processing ratio for 1080p video

### 3. Production Ready
- Comprehensive error handling and recovery
- Real-time monitoring and metrics
- Budget enforcement with hard caps
- Container-based deployment

### 4. Scalable Design
- Microservice-ready architecture
- Database-backed job management
- Horizontal scaling support
- Cloud-native deployment

## Component Architecture

### Transcription Layer

```
Audio Extraction ──▶ Whisper (faster-whisper) ──┐
                                                 ├──▶ ROVER Merge ──▶ Final Transcript
                 ──▶ Deepgram API ──────────────┘
```

**Technologies:**
- `faster-whisper`: Local ASR with word timestamps
- `Deepgram Nova-2`: Cloud ASR with high accuracy
- `pyannote.audio`: Speaker diarization
- ROVER algorithm: Multi-ASR output fusion

**Features:**
- Word-level timestamps for precise editing
- Speaker identification and turns
- Confidence scoring and quality metrics
- Cost tracking with budget enforcement

### Video Processing Layer

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│ Segment     │───▶│ Parallel     │───▶│ FIFO        │───▶│ Concatenation│
│ Selection   │    │ Extraction   │    │ Streaming   │    │ & Encoding   │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
```

**Pipeline Stages:**
1. **Validation**: File integrity and format verification
2. **Analysis**: Content scoring and segment selection  
3. **Extraction**: Parallel segment processing via FIFOs
4. **Concatenation**: Professional encoding with transitions
5. **Optimization**: Hardware acceleration and quality presets

### Intelligent Cropping System

```
Video Frames ──▶ Face Detection ──┐
                                  ├──▶ Crop Center ──▶ FFmpeg Filters
Motion Analysis ──▶ Motion Center ─┘
```

**Computer Vision Pipeline:**
- **Multi-method face detection**: Haar cascades + DNN models
- **Motion analysis**: Optical flow + background subtraction  
- **Content-aware positioning**: Rule of thirds + face weighting
- **Confidence scoring**: Quality assessment and fallbacks

### Data Management Layer

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ PostgreSQL  │    │ Redis       │    │ File System │
│ - Jobs      │    │ - Cache     │    │ - Videos    │
│ - Metadata  │    │ - Sessions  │    │ - Temp      │
│ - Costs     │    │ - Locks     │    │ - Output    │
└─────────────┘    └─────────────┘    └─────────────┘
```

**Persistence Strategy:**
- **PostgreSQL**: Job state, metadata, cost tracking
- **Redis**: Checkpointing, caching, session management
- **File System**: Video storage with automatic cleanup

## Service Architecture

### Core Services

#### 1. Video Analysis Service
```python
# src/core/analyze_video.py
class VideoAnalyzer:
    - Multi-provider transcription
    - Speaker diarization
    - Quality assessment
    - Caching and checkpointing
```

#### 2. Processing Service  
```python
# src/providers/video_processor.py
class VideoProcessor:
    - FIFO-based pipeline
    - Parallel extraction
    - Resource management
    - Performance monitoring
```

#### 3. Intelligence Service
```python
# src/utils/intelligent_crop.py
class IntelligentCropper:
    - Face detection
    - Motion analysis
    - Crop optimization
    - Visual quality assessment
```

#### 4. Cost Management Service
```python
# src/core/cost.py
class CostManager:
    - Budget enforcement
    - Real-time tracking
    - Hard cap protection
    - Cost allocation
```

### Supporting Services

#### Metrics & Monitoring
```python
# src/core/metrics.py
- Prometheus instrumentation
- Performance tracking
- Resource monitoring  
- Error classification
```

#### Configuration Management
```python
# src/core/config.py + src/utils/secret_loader.py
- AWS Secrets Manager integration
- Environment-based configuration
- LRU caching for performance
- Multi-environment support
```

#### Performance Optimization
```python
# src/core/performance.py
- Resource monitoring
- Automatic optimization
- Hardware acceleration
- Memory management
```

## Deployment Architecture

### Container Strategy

```
┌─────────────────────────────────────────────────────────────┐
│                    Kubernetes Cluster                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │   Montage   │  │ PostgreSQL  │  │ Redis       │         │
│  │   App Pod   │  │    Pod      │  │   Pod       │         │
│  │             │  │             │  │             │         │
│  │ + FFmpeg    │  │ + Persistent│  │ + Memory    │         │
│  │ + OpenCV    │  │   Volume    │  │   Cache     │         │
│  │ + Whisper   │  │             │  │             │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

### Infrastructure Components

#### Application Layer
- **Multi-stage Docker builds** for optimized images
- **Non-root containers** for security
- **Health checks** for reliability
- **Resource limits** for stability

#### Data Layer  
- **PostgreSQL 15** with connection pooling
- **Redis 7** for caching and sessions
- **Persistent volumes** for data durability
- **Backup strategies** for disaster recovery

#### Monitoring Layer
- **Prometheus** for metrics collection
- **Grafana** for visualization
- **Custom dashboards** for video processing KPIs
- **Alert rules** for system health

### Scaling Strategy

#### Horizontal Scaling
```
Load Balancer ──┬──▶ Montage Instance 1
                ├──▶ Montage Instance 2
                └──▶ Montage Instance N
                        │
                        ▼
                Shared Database Layer
```

#### Vertical Scaling
- **CPU scaling**: Based on processing load
- **Memory scaling**: Based on video complexity  
- **Storage scaling**: Based on throughput requirements
- **GPU scaling**: For hardware acceleration

## Security Architecture

### Data Protection
- **Secrets management**: AWS Secrets Manager + Kubernetes secrets
- **Network security**: TLS encryption, private subnets
- **Access control**: RBAC, service accounts
- **Data encryption**: At rest and in transit

### Runtime Security
- **Container security**: Non-root users, read-only filesystems
- **Resource isolation**: CPU/memory limits, network policies
- **Vulnerability scanning**: Automated image scanning
- **Security updates**: Automated dependency updates

## Performance Characteristics

### Processing Performance
- **Throughput**: < 1.2x processing ratio for 1080p
- **Latency**: < 30s queue time for typical jobs
- **Concurrency**: Up to 6 parallel video processing workers
- **Memory**: < 8GB per processing instance

### API Performance  
- **Response time**: < 2s for job status queries
- **Availability**: 99.9% uptime SLA
- **Cost tracking**: Real-time with < 1s latency
- **Budget enforcement**: Hard caps with immediate blocking

### Storage Performance
- **Video I/O**: Optimized for streaming workloads
- **Database**: Connection pooling, query optimization
- **Cache hit ratio**: > 90% for transcription results
- **Cleanup**: Automatic temporary file management

## Monitoring & Observability

### Key Metrics

#### Business Metrics
- Jobs processed per hour
- Success/failure rates
- Average processing time
- Cost per processed minute

#### Technical Metrics  
- CPU and memory utilization
- FFmpeg process count
- Database connection pools
- Cache hit rates

#### Quality Metrics
- Face detection accuracy
- Transcription confidence scores
- Audio quality measurements
- Output video quality

### Alerting Strategy
- **Critical**: Budget exceeded, system down
- **Warning**: High resource usage, quality degradation  
- **Info**: Job completion, performance milestones

## Future Architecture Considerations

### Scalability Improvements
- **Event-driven architecture** with message queues
- **Microservice decomposition** for independent scaling
- **Edge processing** for reduced latency
- **GPU cluster integration** for acceleration

### Technology Evolution
- **ML model updates** for improved accuracy
- **Codec improvements** for better compression
- **Cloud services integration** for managed scaling
- **Real-time processing** for live video streams

## Implementation Guidelines

### Development Practices
- **Test-driven development** with comprehensive coverage
- **Performance testing** with realistic workloads  
- **Security scanning** in CI/CD pipeline
- **Documentation-first** approach

### Operational Practices
- **Infrastructure as code** with Terraform
- **GitOps deployment** with automated rollbacks
- **Monitoring-driven operations** with SLO tracking
- **Incident response** with automated remediation

This architecture provides a solid foundation for production video processing with room for future enhancement and scaling.