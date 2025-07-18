"""Integration module for monitoring and alerting"""

import logging
from flask import Flask, request, jsonify
from prometheus_client import start_http_server
import threading
import json
from datetime import datetime
from typing import Dict, Any, List

from config import Config
from db import Database
from metrics import registry
import redis

logger = logging.getLogger(__name__)


class AlertHandler:
    """Handles incoming alerts from Alertmanager"""
    
    def __init__(self):
        self.db = Database()
        self.redis_client = redis.from_url(Config.REDIS_URL)
        self.app = Flask(__name__)
        self._setup_routes()
        
    def _setup_routes(self):
        """Setup webhook routes for alerts"""
        
        @self.app.route('/webhook/alerts', methods=['POST'])
        def handle_alerts():
            """General alert handler"""
            try:
                alerts = request.json
                logger.info(f"Received {len(alerts)} alerts")
                
                for alert in alerts:
                    self._process_alert(alert)
                
                return jsonify({"status": "processed"}), 200
            except Exception as e:
                logger.error(f"Alert processing failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/webhook/critical', methods=['POST'])
        def handle_critical():
            """Critical alert handler - immediate action required"""
            try:
                alert_data = request.json
                alerts = alert_data.get('alerts', [])
                
                for alert in alerts:
                    alert_name = alert['labels'].get('alertname')
                    
                    if alert_name == 'FFmpegCrashes':
                        self._handle_ffmpeg_crashes(alert)
                    elif alert_name == 'DatabaseConnectionPoolExhausted':
                        self._handle_db_exhaustion(alert)
                    elif alert_name == 'HighErrorRate':
                        self._handle_high_error_rate(alert)
                
                return jsonify({"status": "handled"}), 200
            except Exception as e:
                logger.error(f"Critical alert handling failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/webhook/budget', methods=['POST'])
        def handle_budget():
            """Budget alert handler - cost control"""
            try:
                alert_data = request.json
                alerts = alert_data.get('alerts', [])
                
                for alert in alerts:
                    if alert['labels'].get('alertname') == 'BudgetExceeded':
                        job_id = alert['labels'].get('job_id')
                        self._handle_budget_exceeded(job_id)
                
                return jsonify({"status": "budget_controlled"}), 200
            except Exception as e:
                logger.error(f"Budget alert handling failed: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat()
            }), 200
    
    def _process_alert(self, alert: Dict[str, Any]):
        """Process individual alert"""
        alert_name = alert['labels'].get('alertname')
        severity = alert['labels'].get('severity')
        
        # Log alert
        logger.warning(f"Alert: {alert_name} - Severity: {severity}")
        
        # Store in database for audit
        self.db.insert('alert_log', {
            'alert_name': alert_name,
            'severity': severity,
            'labels': json.dumps(alert['labels']),
            'annotations': json.dumps(alert['annotations']),
            'status': alert['status'],
            'starts_at': alert.get('startsAt'),
            'ends_at': alert.get('endsAt')
        })
    
    def _handle_ffmpeg_crashes(self, alert: Dict[str, Any]):
        """Handle FFmpeg crash alerts"""
        logger.critical("FFmpeg crashing frequently - pausing video processing")
        
        # Set circuit breaker in Redis
        self.redis_client.setex('ffmpeg_circuit_breaker', 300, 'open')
        
        # Mark all running jobs as failed
        running_jobs = self.db.find('video_job', {'status': 'processing'})
        for job in running_jobs:
            self.db.update('video_job', 
                          {'id': job['id']}, 
                          {
                              'status': 'failed',
                              'error': 'FFmpeg crash detected - processing paused'
                          })
    
    def _handle_db_exhaustion(self, alert: Dict[str, Any]):
        """Handle database connection pool exhaustion"""
        logger.critical("Database connection pool exhausted")
        
        # Set backpressure flag
        self.redis_client.setex('db_backpressure', 60, 'true')
        
        # Could implement connection pool expansion here
    
    def _handle_high_error_rate(self, alert: Dict[str, Any]):
        """Handle high error rate alerts"""
        error_rate = float(alert['annotations'].get('value', 0))
        logger.critical(f"High error rate detected: {error_rate:.1%}")
        
        # Reduce processing rate
        current_rate = int(self.redis_client.get('processing_rate_limit') or 10)
        new_rate = max(1, current_rate // 2)
        self.redis_client.setex('processing_rate_limit', 3600, new_rate)
        logger.info(f"Reduced processing rate limit to {new_rate} jobs/minute")
    
    def _handle_budget_exceeded(self, job_id: str):
        """Handle budget exceeded for a job"""
        logger.error(f"Budget exceeded for job {job_id}")
        
        # Immediately stop the job
        self.db.update('video_job',
                      {'id': job_id},
                      {
                          'status': 'failed',
                          'error': 'Budget limit exceeded'
                      })
        
        # Set flag to prevent further API calls
        self.redis_client.setex(f'job_budget_exceeded:{job_id}', 3600, 'true')
    
    def start(self, port: int = 8001):
        """Start the alert handler server"""
        self.app.run(host='0.0.0.0', port=port, threaded=True)


class MonitoringServer:
    """Main monitoring server that combines metrics and alerting"""
    
    def __init__(self):
        self.metrics_port = Config.METRICS_PORT or 8000
        self.alerts_port = Config.ALERTS_PORT or 8001
        self.alert_handler = AlertHandler()
    
    def start(self):
        """Start both metrics and alert servers"""
        # Start Prometheus metrics server
        logger.info(f"Starting metrics server on port {self.metrics_port}")
        start_http_server(self.metrics_port, registry=registry)
        
        # Start alert handler in separate thread
        alert_thread = threading.Thread(
            target=self.alert_handler.start,
            args=(self.alerts_port,),
            daemon=True
        )
        alert_thread.start()
        logger.info(f"Started alert handler on port {self.alerts_port}")


# Circuit breaker implementation
class CircuitBreaker:
    """Circuit breaker for external services"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def is_open(self, service: str) -> bool:
        """Check if circuit breaker is open for a service"""
        return self.redis.exists(f'{service}_circuit_breaker')
    
    def can_proceed(self, service: str) -> bool:
        """Check if we can proceed with the service call"""
        if self.is_open(service):
            logger.warning(f"Circuit breaker OPEN for {service}")
            return False
        return True


# Rate limiter implementation
class RateLimiter:
    """Rate limiter for job processing"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        # Get current rate limit (jobs per minute)
        rate_limit = int(self.redis.get('processing_rate_limit') or 10)
        
        # Count jobs in current minute
        current_minute = datetime.utcnow().strftime('%Y%m%d%H%M')
        key = f'job_count:{current_minute}'
        
        # Increment and check
        count = self.redis.incr(key)
        self.redis.expire(key, 120)  # Expire after 2 minutes
        
        if count > rate_limit:
            logger.warning(f"Rate limit exceeded: {count}/{rate_limit}")
            return False
        
        return True


# Example integration with main processing
def process_video_with_monitoring(job_id: str, video_path: str):
    """Example of processing video with monitoring integration"""
    redis_client = redis.from_url(Config.REDIS_URL)
    circuit_breaker = CircuitBreaker(redis_client)
    rate_limiter = RateLimiter(redis_client)
    
    # Check circuit breakers
    if not circuit_breaker.can_proceed('ffmpeg'):
        raise Exception("FFmpeg circuit breaker is open")
    
    # Check rate limits
    if not rate_limiter.check_rate_limit():
        raise Exception("Rate limit exceeded")
    
    # Check if budget exceeded for this job
    if redis_client.exists(f'job_budget_exceeded:{job_id}'):
        raise Exception("Budget already exceeded for this job")
    
    # Check database backpressure
    if redis_client.exists('db_backpressure'):
        logger.warning("Database backpressure detected - slowing down")
        import time
        time.sleep(5)
    
    # Proceed with processing...
    logger.info(f"Processing video {video_path} for job {job_id}")


# CLI for testing alerts
def send_test_alert(alert_type: str):
    """Send test alert to webhook"""
    import requests
    
    test_alerts = {
        'budget': {
            'alerts': [{
                'status': 'firing',
                'labels': {
                    'alertname': 'BudgetExceeded',
                    'severity': 'critical',
                    'job_id': 'test-job-001'
                },
                'annotations': {
                    'summary': 'Test budget exceeded',
                    'description': 'Job test-job-001 exceeded $5 budget'
                }
            }]
        },
        'ffmpeg': {
            'alerts': [{
                'status': 'firing',
                'labels': {
                    'alertname': 'FFmpegCrashes',
                    'severity': 'critical'
                },
                'annotations': {
                    'summary': 'FFmpeg crashing frequently',
                    'description': 'FFmpeg crashed 5 times in last 5 minutes'
                }
            }]
        }
    }
    
    alert_data = test_alerts.get(alert_type)
    if alert_data:
        response = requests.post(
            f'http://localhost:8001/webhook/{alert_type}',
            json=alert_data,
            headers={'Authorization': f'Bearer {Config.WEBHOOK_TOKEN}'}
        )
        print(f"Alert sent: {response.status_code} - {response.text}")
    else:
        print(f"Unknown alert type: {alert_type}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test-alert':
        # Test alert sending
        alert_type = sys.argv[2] if len(sys.argv) > 2 else 'budget'
        send_test_alert(alert_type)
    else:
        # Start monitoring server
        logging.basicConfig(level=logging.INFO)
        server = MonitoringServer()
        server.start()
        
        # Keep running
        try:
            while True:
                import time
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Shutting down monitoring server")