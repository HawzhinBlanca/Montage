# Comprehensive Rate Limiting System

This document describes the comprehensive rate limiting system implemented to prevent API cost explosions in production.

## Overview

The rate limiting system provides multiple layers of protection against runaway API costs and service failures:

1. **Token Bucket Rate Limiter**: Smooth rate limiting with configurable burst capability
2. **Per-API Service Limits**: Different limits for different services (Deepgram, OpenAI, Anthropic, Gemini)
3. **Cost-Aware Rate Limiting**: Dynamic limits based on remaining budget and spending patterns
4. **Circuit Breaker Pattern**: Automatic failure detection and recovery
5. **Request Queuing**: Priority-based queuing with graceful backpressure
6. **Adaptive Limits**: Automatically adjust based on API success rates and health
7. **Production Monitoring**: Real-time metrics and alerting

## Architecture

```
┌─────────────────────┐
│   Application       │
│   Code             │
└─────┬───────────────┘
      │
┌─────▼───────────────┐
│   API Wrappers      │
│   (Service Layer)   │
└─────┬───────────────┘
      │
┌─────▼───────────────┐
│   Rate Limiter      │
│   - Token Bucket    │
│   - Circuit Breaker │
│   - Priority Queue  │
└─────┬───────────────┘
      │
┌─────▼───────────────┐
│   Monitoring &      │
│   Metrics          │
└─────────────────────┘
```

## Components

### 1. Core Rate Limiter (`src/core/rate_limiter.py`)

The heart of the system, providing:

- **TokenBucket**: Implements token bucket algorithm for smooth rate limiting
- **CircuitBreaker**: Protects against cascading failures
- **PriorityQueue**: Manages requests with different priority levels
- **RateLimiter**: Orchestrates all components for comprehensive protection

#### Usage Example:

```python
from src.core.rate_limiter import rate_limited, Priority
from decimal import Decimal

@rate_limited("my_service", Priority.HIGH, Decimal("0.01"))
def expensive_api_call(data):
    # Your API call here
    return api_response
```

### 2. API Service Wrappers (`src/core/api_wrappers.py`)

Provides rate-limited wrappers for all external API services:

- **DeepgramWrapper**: Audio transcription with chunking support
- **OpenAIWrapper**: GPT completions with token cost estimation
- **AnthropicWrapper**: Claude API with cost tracking
- **GeminiWrapper**: Google's Gemini with high throughput support

#### Usage Example:

```python
from src.core.api_wrappers import deepgram_service

# Automatically rate-limited
result = deepgram_service.transcribe_audio("audio.wav", job_id="video_123")
```

### 3. Environment Configuration (`src/core/rate_limit_config.py`)

Environment-specific configurations optimized for different deployment scenarios:

- **Development**: Generous limits for fast iteration
- **Testing**: Restrictive limits to prevent test suite costs
- **Staging**: Balanced limits for realistic testing
- **Production**: Conservative limits for cost control

#### Configuration Example:

```python
# Automatic environment detection
from src.core.rate_limit_config import get_service_config

config = get_service_config("deepgram")
# Returns environment-appropriate configuration
```

### 4. Monitoring and Alerting (`src/core/rate_limit_monitor.py`)

Real-time monitoring with alerting:

- **ServiceMetrics**: Track success rates, queue sizes, costs
- **RateLimitAlert**: Structured alerts for different conditions
- **RateLimitMonitor**: Continuous monitoring with configurable thresholds

#### Monitoring Example:

```python
from src.core.rate_limit_monitor import start_rate_limit_monitoring, log_alert_handler

# Start monitoring with alert handler
start_rate_limit_monitoring(alert_callback=log_alert_handler)
```

### 5. CLI Management Tool (`src/cli/rate_limit_cli.py`)

Command-line tool for monitoring and management:

```bash
# Check current status
python -m src.cli.rate_limit_cli status

# Start continuous monitoring
python -m src.cli.rate_limit_cli monitor

# Enable emergency mode (minimal limits)
python -m src.cli.rate_limit_cli emergency --enable

# Export metrics
python -m src.cli.rate_limit_cli export --output metrics.json
```

## Environment Configuration

Set the environment using the `MONTAGE_ENV` environment variable:

```bash
# Development (default)
export MONTAGE_ENV=development

# Testing
export MONTAGE_ENV=testing

# Staging
export MONTAGE_ENV=staging

# Production
export MONTAGE_ENV=production
```

### Environment Differences:

| Environment | Token Limits | Cost Limits | Recovery Time | Queue Size |
|-------------|--------------|-------------|---------------|------------|
| Development | Generous     | High        | Fast          | Large      |
| Testing     | Restrictive  | Very Low    | Fast          | Small      |
| Staging     | Moderate     | Medium      | Balanced      | Medium     |
| Production  | Conservative | Strict      | Slow          | Controlled |

## Integration with Existing Code

The rate limiting system is designed to integrate seamlessly with existing code:

### 1. Video Analysis Integration

```python
# Before: Direct API call
result = transcribe_deepgram(audio_path, job_id)

# After: Automatically rate-limited
result = transcribe_deepgram(audio_path, job_id)  # Same interface!
```

### 2. Highlight Selection Integration

```python
# Before: Direct Gemini call
story_beats = analyze_story_structure(transcript_segments)

# After: Rate-limited with fallback
story_beats = analyze_story_structure(transcript_segments)  # Same interface!
```

### 3. Graceful Degradation

The system provides automatic fallback mechanisms:

```python
from src.core.api_wrappers import graceful_api_call

result = graceful_api_call(
    service_func=expensive_ai_service,
    fallback_func=local_processing,
    service_name="ai_service"
)
```

## Cost Protection

Multiple layers of cost protection:

### 1. Budget Enforcement

```python
from src.core.cost import check_budget, BudgetExceededError

# Automatic budget checking
within_budget, remaining = check_budget(estimated_cost)
if not within_budget:
    raise BudgetExceededError("Insufficient budget")
```

### 2. Cost-Aware Rate Limiting

The system automatically adjusts rate limits based on:

- Current spending rate ($/minute)
- Remaining budget
- Historical cost patterns
- Service-specific cost profiles

### 3. Emergency Mode

When costs spike or budgets are exhausted:

```bash
# Enable emergency mode
python -m src.cli.rate_limit_cli emergency --enable
```

This reduces all limits to absolute minimums while maintaining basic functionality.

## Monitoring and Alerting

### Alert Types

1. **Circuit Breaker Open**: Service is failing repeatedly
2. **High Failure Rate**: Success rate below threshold
3. **Queue Backup**: Request queue is filling up
4. **Cost Spike**: Spending rate above normal
5. **Low Tokens**: Rate limiter running out of capacity

### Health Monitoring

```python
from src.core.rate_limit_monitor import get_rate_limit_health

health = get_rate_limit_health()
# Returns: overall_status, health_score, service details
```

### Metrics Export

```python
from src.core.rate_limit_monitor import export_rate_limit_metrics

metrics_json = export_rate_limit_metrics()
# Contains comprehensive system metrics
```

## Configuration Reference

### RateLimitConfig Parameters

```python
@dataclass
class RateLimitConfig:
    # Token bucket parameters
    max_tokens: int = 10              # Maximum tokens in bucket
    refill_rate: float = 1.0          # Tokens added per second
    burst_capacity: int = 5           # Additional burst tokens
    
    # Cost-based limiting
    max_cost_per_minute: Decimal = Decimal("1.00")  # Max spend per minute
    cost_per_request: Decimal = Decimal("0.01")     # Estimated cost per request
    
    # Circuit breaker parameters
    failure_threshold: int = 5        # Failures before opening circuit
    recovery_timeout: int = 30        # Seconds before attempting recovery
    success_threshold: int = 3        # Successes needed to close circuit
    
    # Queue parameters
    max_queue_size: int = 100         # Maximum queued requests
    request_timeout: int = 300        # Request timeout in seconds
    
    # Adaptive parameters
    min_interval: float = 0.1         # Minimum time between requests
    max_interval: float = 5.0         # Maximum time between requests
    backoff_multiplier: float = 1.5   # Exponential backoff multiplier
```

### Service-Specific Defaults

#### Deepgram (Audio Transcription)
- **Production**: 2 tokens, 0.5/sec refill, $1.50/min, 1min recovery
- **Development**: 10 tokens, 2.0/sec refill, $5.00/min, 30sec recovery

#### OpenAI (GPT Models)
- **Production**: 2 tokens, 0.33/sec refill, $1.00/min, 1min recovery
- **Development**: 5 tokens, 1.0/sec refill, $3.00/min, 30sec recovery

#### Anthropic (Claude)
- **Production**: 1 token, 0.25/sec refill, $1.25/min, 1.5min recovery
- **Development**: 3 tokens, 0.5/sec refill, $4.00/min, 30sec recovery

#### Gemini (Google AI)
- **Production**: 10 tokens, 2.0/sec refill, $0.50/min, 30sec recovery
- **Development**: 20 tokens, 5.0/sec refill, $1.00/min, 20sec recovery

## Best Practices

### 1. Error Handling

Always handle rate limiting errors gracefully:

```python
from src.core.api_wrappers import RateLimitExceededError, CircuitBreakerOpenError

try:
    result = rate_limited_api_call()
except RateLimitExceededError:
    # Implement backoff or queue for later
    pass
except CircuitBreakerOpenError:
    # Use fallback mechanism
    result = fallback_processing()
```

### 2. Priority Management

Use appropriate priorities for different request types:

```python
# Critical user-facing requests
@rate_limited("service", Priority.CRITICAL)

# Background processing
@rate_limited("service", Priority.LOW)
```

### 3. Cost Estimation

Provide accurate cost estimates when possible:

```python
# Estimate based on input size
estimated_cost = Decimal(str(len(input_text) / 1000 * 0.002))

result = service.process(input_text, cost_estimate=estimated_cost)
```

### 4. Monitoring Integration

Integrate monitoring into your application:

```python
# Start monitoring at application startup
start_rate_limit_monitoring(alert_callback=your_alert_handler)

# Check health periodically
health = get_rate_limit_health()
if health['overall_status'] in ['critical', 'degraded']:
    # Take corrective action
    enable_emergency_mode()
```

## Testing

Run the comprehensive test suite:

```bash
# Run all rate limiting tests
python -m pytest tests/test_rate_limiting.py -v

# Run specific test categories
python -m pytest tests/test_rate_limiting.py::TestTokenBucket -v
python -m pytest tests/test_rate_limiting.py::TestCircuitBreaker -v
python -m pytest tests/test_rate_limiting.py::TestIntegration -v
```

## Troubleshooting

### Common Issues

1. **High Queue Sizes**: Indicates demand exceeds capacity
   - Solution: Adjust rate limits or add more capacity

2. **Circuit Breaker Open**: API service is failing
   - Solution: Check service health, wait for recovery, or use fallback

3. **Cost Alerts**: Spending above normal levels
   - Solution: Review request patterns, enable emergency mode if needed

4. **Low Success Rates**: Requests frequently failing
   - Solution: Check API credentials, service status, network connectivity

### Debug Commands

```bash
# Check current system status
python -m src.cli.rate_limit_cli status

# Validate configuration
python -m src.cli.rate_limit_cli validate

# Export detailed metrics
python -m src.cli.rate_limit_cli export --output debug_metrics.json
```

### Log Analysis

The system provides structured logging for debugging:

```bash
# Filter rate limiting logs
grep "RATE_LIMIT" logs/application.log

# Monitor circuit breaker events
grep "Circuit breaker" logs/application.log

# Track cost alerts
grep "Cost spike" logs/application.log
```

## Performance Impact

The rate limiting system is designed for minimal performance impact:

- **Memory Usage**: ~10MB per service (token buckets, queues, metrics)
- **CPU Overhead**: <1% for typical request volumes
- **Latency Added**: ~1-10ms per request (depending on queue size)

### Optimization Tips

1. **Batch Processing**: Use batch endpoints when available
2. **Caching**: Cache results to reduce API calls
3. **Intelligent Queuing**: Use priorities appropriately
4. **Monitoring**: Monitor queue sizes and adjust limits as needed

## Security Considerations

1. **API Keys**: Never log or expose API keys in rate limiting code
2. **Cost Data**: Treat cost information as sensitive
3. **Circuit States**: Don't expose internal state to unauthorized users
4. **Queue Contents**: Ensure request data in queues is secure

## Future Enhancements

Planned improvements to the rate limiting system:

1. **Redis Backend**: Distributed rate limiting across multiple instances
2. **Machine Learning**: Predictive rate limiting based on usage patterns
3. **Advanced Metrics**: More detailed performance and cost analytics
4. **Custom Policies**: User-defined rate limiting policies
5. **Integration APIs**: REST APIs for external monitoring systems

---

For questions or support, please contact the development team or file an issue in the project repository.