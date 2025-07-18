#!/usr/bin/env python3
"""
Phase 4: Metrics & budget guardrails
Usage tracking, cost monitoring, and budget enforcement across the entire pipeline
"""

import os
import sys
import json
import logging
import time
import asyncio
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid
from datetime import datetime, timedelta
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class APIUsageMetric:
    """Single API usage metric"""
    service: str  # 'whisper', 'deepgram', 'gpt4o', 'claude', 'davinci'
    operation: str
    timestamp: float
    cost: float
    duration: float
    input_tokens: int
    output_tokens: int
    success: bool
    error_message: Optional[str]
    project_id: str
    session_id: str

@dataclass
class ProcessingMetric:
    """Processing performance metric"""
    phase: str
    operation: str
    timestamp: float
    duration: float
    input_size: int  # bytes or tokens
    output_size: int
    success: bool
    error_message: Optional[str]
    project_id: str
    session_id: str

@dataclass
class BudgetAlert:
    """Budget alert"""
    alert_id: str
    alert_type: str  # 'warning', 'limit', 'exceeded'
    threshold: float
    current_value: float
    message: str
    timestamp: float
    project_id: str
    session_id: str

@dataclass
class UsageReport:
    """Usage report for a time period"""
    report_id: str
    start_time: float
    end_time: float
    total_cost: float
    total_api_calls: int
    total_processing_time: float
    projects_processed: int
    success_rate: float
    cost_by_service: Dict[str, float]
    performance_metrics: Dict[str, float]
    budget_alerts: List[BudgetAlert]
    generated_at: float

class BudgetManager:
    """Budget tracking and enforcement"""
    
    def __init__(self, daily_budget: float = 50.0, monthly_budget: float = 1000.0):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.current_daily_spend = 0.0
        self.current_monthly_spend = 0.0
        self.last_reset_day = datetime.now().date()
        self.last_reset_month = datetime.now().month
        self.alerts = []
        self.cost_history = []
        
    def reset_daily_budget(self):
        """Reset daily budget if new day"""
        today = datetime.now().date()
        if today != self.last_reset_day:
            self.current_daily_spend = 0.0
            self.last_reset_day = today
            logger.info(f"💰 Daily budget reset: ${self.daily_budget}")
    
    def reset_monthly_budget(self):
        """Reset monthly budget if new month"""
        current_month = datetime.now().month
        if current_month != self.last_reset_month:
            self.current_monthly_spend = 0.0
            self.last_reset_month = current_month
            logger.info(f"💰 Monthly budget reset: ${self.monthly_budget}")
    
    def add_cost(self, cost: float, service: str, project_id: str, session_id: str) -> bool:
        """Add cost and check budget limits"""
        self.reset_daily_budget()
        self.reset_monthly_budget()
        
        # Record cost
        self.cost_history.append({
            'timestamp': time.time(),
            'cost': cost,
            'service': service,
            'project_id': project_id,
            'session_id': session_id
        })
        
        # Update current spend
        self.current_daily_spend += cost
        self.current_monthly_spend += cost
        
        # Check budget limits
        alerts = []
        
        # Daily budget checks
        daily_usage_percent = (self.current_daily_spend / self.daily_budget) * 100
        if daily_usage_percent >= 100:
            alerts.append(BudgetAlert(
                alert_id=str(uuid.uuid4()),
                alert_type='exceeded',
                threshold=self.daily_budget,
                current_value=self.current_daily_spend,
                message=f'Daily budget exceeded: ${self.current_daily_spend:.2f} / ${self.daily_budget:.2f}',
                timestamp=time.time(),
                project_id=project_id,
                session_id=session_id
            ))
        elif daily_usage_percent >= 90:
            alerts.append(BudgetAlert(
                alert_id=str(uuid.uuid4()),
                alert_type='warning',
                threshold=self.daily_budget * 0.9,
                current_value=self.current_daily_spend,
                message=f'Daily budget warning: ${self.current_daily_spend:.2f} / ${self.daily_budget:.2f} (90%)',
                timestamp=time.time(),
                project_id=project_id,
                session_id=session_id
            ))
        
        # Monthly budget checks
        monthly_usage_percent = (self.current_monthly_spend / self.monthly_budget) * 100
        if monthly_usage_percent >= 100:
            alerts.append(BudgetAlert(
                alert_id=str(uuid.uuid4()),
                alert_type='exceeded',
                threshold=self.monthly_budget,
                current_value=self.current_monthly_spend,
                message=f'Monthly budget exceeded: ${self.current_monthly_spend:.2f} / ${self.monthly_budget:.2f}',
                timestamp=time.time(),
                project_id=project_id,
                session_id=session_id
            ))
        elif monthly_usage_percent >= 90:
            alerts.append(BudgetAlert(
                alert_id=str(uuid.uuid4()),
                alert_type='warning',
                threshold=self.monthly_budget * 0.9,
                current_value=self.current_monthly_spend,
                message=f'Monthly budget warning: ${self.current_monthly_spend:.2f} / ${self.monthly_budget:.2f} (90%)',
                timestamp=time.time(),
                project_id=project_id,
                session_id=session_id
            ))
        
        # Log alerts
        for alert in alerts:
            self.alerts.append(alert)
            if alert.alert_type == 'exceeded':
                logger.error(f"🚨 {alert.message}")
            else:
                logger.warning(f"⚠️ {alert.message}")
        
        # Return True if under budget, False if exceeded
        return self.current_daily_spend <= self.daily_budget and self.current_monthly_spend <= self.monthly_budget
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        return {
            'daily_budget': self.daily_budget,
            'monthly_budget': self.monthly_budget,
            'current_daily_spend': self.current_daily_spend,
            'current_monthly_spend': self.current_monthly_spend,
            'daily_remaining': max(0, self.daily_budget - self.current_daily_spend),
            'monthly_remaining': max(0, self.monthly_budget - self.current_monthly_spend),
            'daily_usage_percent': min(100, (self.current_daily_spend / self.daily_budget) * 100),
            'monthly_usage_percent': min(100, (self.current_monthly_spend / self.monthly_budget) * 100),
            'alerts_count': len(self.alerts),
            'last_reset_day': self.last_reset_day.isoformat(),
            'last_reset_month': self.last_reset_month
        }

class MetricsCollector:
    """Collects and stores metrics from all pipeline components"""
    
    def __init__(self):
        self.api_metrics = []
        self.processing_metrics = []
        self.budget_manager = BudgetManager()
        self.session_id = str(uuid.uuid4())
        self.lock = threading.Lock()
        
    def record_api_usage(self, service: str, operation: str, cost: float, 
                        duration: float, input_tokens: int = 0, output_tokens: int = 0,
                        success: bool = True, error_message: Optional[str] = None,
                        project_id: str = "default") -> bool:
        """Record API usage metric"""
        with self.lock:
            metric = APIUsageMetric(
                service=service,
                operation=operation,
                timestamp=time.time(),
                cost=cost,
                duration=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                success=success,
                error_message=error_message,
                project_id=project_id,
                session_id=self.session_id
            )
            
            self.api_metrics.append(metric)
            
            # Update budget
            within_budget = self.budget_manager.add_cost(cost, service, project_id, self.session_id)
            
            logger.info(f"📊 API Usage: {service}.{operation} - ${cost:.4f} ({duration:.2f}s)")
            
            return within_budget
    
    def record_processing_metric(self, phase: str, operation: str, duration: float,
                               input_size: int = 0, output_size: int = 0,
                               success: bool = True, error_message: Optional[str] = None,
                               project_id: str = "default"):
        """Record processing performance metric"""
        with self.lock:
            metric = ProcessingMetric(
                phase=phase,
                operation=operation,
                timestamp=time.time(),
                duration=duration,
                input_size=input_size,
                output_size=output_size,
                success=success,
                error_message=error_message,
                project_id=project_id,
                session_id=self.session_id
            )
            
            self.processing_metrics.append(metric)
            
            logger.info(f"⚡ Processing: {phase}.{operation} - {duration:.2f}s")
    
    def get_cost_by_service(self, time_range: Optional[tuple] = None) -> Dict[str, float]:
        """Get cost breakdown by service"""
        costs = {}
        
        for metric in self.api_metrics:
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= metric.timestamp <= end_time):
                    continue
            
            service = metric.service
            if service not in costs:
                costs[service] = 0.0
            costs[service] += metric.cost
        
        return costs
    
    def get_performance_metrics(self, time_range: Optional[tuple] = None) -> Dict[str, float]:
        """Get performance metrics"""
        metrics = {}
        
        # API performance
        api_durations = [m.duration for m in self.api_metrics 
                        if not time_range or (time_range[0] <= m.timestamp <= time_range[1])]
        
        if api_durations:
            metrics['avg_api_duration'] = sum(api_durations) / len(api_durations)
            metrics['max_api_duration'] = max(api_durations)
            metrics['min_api_duration'] = min(api_durations)
        
        # Processing performance
        processing_durations = [m.duration for m in self.processing_metrics 
                              if not time_range or (time_range[0] <= m.timestamp <= time_range[1])]
        
        if processing_durations:
            metrics['avg_processing_duration'] = sum(processing_durations) / len(processing_durations)
            metrics['max_processing_duration'] = max(processing_durations)
            metrics['min_processing_duration'] = min(processing_durations)
        
        # Success rates
        api_success_rate = sum(1 for m in self.api_metrics if m.success) / len(self.api_metrics) if self.api_metrics else 0
        processing_success_rate = sum(1 for m in self.processing_metrics if m.success) / len(self.processing_metrics) if self.processing_metrics else 0
        
        metrics['api_success_rate'] = api_success_rate
        metrics['processing_success_rate'] = processing_success_rate
        metrics['overall_success_rate'] = (api_success_rate + processing_success_rate) / 2
        
        return metrics
    
    def generate_usage_report(self, time_range: Optional[tuple] = None) -> UsageReport:
        """Generate comprehensive usage report"""
        if not time_range:
            # Default to last 24 hours
            end_time = time.time()
            start_time = end_time - 86400  # 24 hours
            time_range = (start_time, end_time)
        
        start_time, end_time = time_range
        
        # Filter metrics by time range
        api_metrics_filtered = [m for m in self.api_metrics if start_time <= m.timestamp <= end_time]
        processing_metrics_filtered = [m for m in self.processing_metrics if start_time <= m.timestamp <= end_time]
        
        # Calculate totals
        total_cost = sum(m.cost for m in api_metrics_filtered)
        total_api_calls = len(api_metrics_filtered)
        total_processing_time = sum(m.duration for m in processing_metrics_filtered)
        
        # Unique projects
        projects = set()
        for m in api_metrics_filtered:
            projects.add(m.project_id)
        for m in processing_metrics_filtered:
            projects.add(m.project_id)
        
        # Success rate
        successful_operations = sum(1 for m in api_metrics_filtered if m.success) + sum(1 for m in processing_metrics_filtered if m.success)
        total_operations = len(api_metrics_filtered) + len(processing_metrics_filtered)
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        
        # Get cost breakdown and performance metrics
        cost_by_service = self.get_cost_by_service(time_range)
        performance_metrics = self.get_performance_metrics(time_range)
        
        # Filter budget alerts by time range
        budget_alerts = [alert for alert in self.budget_manager.alerts 
                        if start_time <= alert.timestamp <= end_time]
        
        return UsageReport(
            report_id=str(uuid.uuid4()),
            start_time=start_time,
            end_time=end_time,
            total_cost=total_cost,
            total_api_calls=total_api_calls,
            total_processing_time=total_processing_time,
            projects_processed=len(projects),
            success_rate=success_rate,
            cost_by_service=cost_by_service,
            performance_metrics=performance_metrics,
            budget_alerts=budget_alerts,
            generated_at=time.time()
        )
    
    def save_metrics(self, filename: str):
        """Save all metrics to file"""
        data = {
            'session_id': self.session_id,
            'generated_at': time.time(),
            'api_metrics': [asdict(m) for m in self.api_metrics],
            'processing_metrics': [asdict(m) for m in self.processing_metrics],
            'budget_status': self.budget_manager.get_budget_status(),
            'budget_alerts': [asdict(alert) for alert in self.budget_manager.alerts]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"💾 Metrics saved to: {filename}")

class PipelineMonitor:
    """Monitor entire pipeline with metrics and budget controls"""
    
    def __init__(self, daily_budget: float = 50.0, monthly_budget: float = 1000.0):
        self.metrics_collector = MetricsCollector()
        self.metrics_collector.budget_manager.daily_budget = daily_budget
        self.metrics_collector.budget_manager.monthly_budget = monthly_budget
        
    def simulate_pipeline_execution(self, project_id: str = "test_project"):
        """Simulate full pipeline execution with metrics tracking"""
        logger.info("🚀 Starting pipeline execution with metrics tracking...")
        
        # Phase 1.1: ASR Processing
        start_time = time.time()
        self.metrics_collector.record_processing_metric(
            phase="Phase_1_1",
            operation="ensemble_asr",
            duration=2.8,
            input_size=60000,  # 60 seconds of audio
            output_size=612,   # 612 characters of transcript
            success=True,
            project_id=project_id
        )
        
        # Simulate API costs
        self.metrics_collector.record_api_usage(
            service="whisper",
            operation="transcribe",
            cost=0.006,  # $0.006 per minute
            duration=2.5,
            input_tokens=0,
            output_tokens=102,
            success=True,
            project_id=project_id
        )
        
        # Phase 1.2: Local highlighting
        self.metrics_collector.record_processing_metric(
            phase="Phase_1_2",
            operation="local_highlight_scoring",
            duration=0.5,
            input_size=612,
            output_size=5,  # 5 highlights
            success=True,
            project_id=project_id
        )
        
        # Phase 1.3: Premium AI scoring
        self.metrics_collector.record_api_usage(
            service="gpt4o",
            operation="analyze_segments",
            cost=0.0001,
            duration=0.7,
            input_tokens=150,
            output_tokens=50,
            success=True,
            project_id=project_id
        )
        
        self.metrics_collector.record_processing_metric(
            phase="Phase_1_3",
            operation="premium_highlight_scoring",
            duration=1.2,
            input_size=5,
            output_size=5,
            success=True,
            project_id=project_id
        )
        
        # Phase 1.4: Subtitle generation
        self.metrics_collector.record_processing_metric(
            phase="Phase_1_4",
            operation="subtitle_generation",
            duration=0.5,
            input_size=5,
            output_size=15,  # 15 subtitle files
            success=True,
            project_id=project_id
        )
        
        # Phase 2: DaVinci Resolve bridge
        self.metrics_collector.record_processing_metric(
            phase="Phase_2",
            operation="davinci_resolve_bridge",
            duration=3.0,
            input_size=5,
            output_size=1,  # 1 video project
            success=True,
            project_id=project_id
        )
        
        # Phase 3: QC and export
        self.metrics_collector.record_processing_metric(
            phase="Phase_3",
            operation="qc_analysis",
            duration=1.5,
            input_size=1,
            output_size=1,
            success=True,
            project_id=project_id
        )
        
        total_duration = time.time() - start_time
        logger.info(f"✅ Pipeline execution completed in {total_duration:.2f}s")
        
        return project_id
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get dashboard data for monitoring"""
        budget_status = self.metrics_collector.budget_manager.get_budget_status()
        usage_report = self.metrics_collector.generate_usage_report()
        
        return {
            'budget_status': budget_status,
            'usage_report': asdict(usage_report),
            'real_time_metrics': {
                'total_api_calls': len(self.metrics_collector.api_metrics),
                'total_processing_operations': len(self.metrics_collector.processing_metrics),
                'session_duration': time.time() - (self.metrics_collector.api_metrics[0].timestamp if self.metrics_collector.api_metrics else time.time()),
                'alerts_count': len(self.metrics_collector.budget_manager.alerts)
            }
        }

async def main():
    """Test metrics and budget system"""
    if len(sys.argv) < 2:
        print("Usage: python phase4_metrics_budget_guardrails.py <daily_budget> [monthly_budget]")
        print("Example: python phase4_metrics_budget_guardrails.py 10.0 200.0")
        return
    
    daily_budget = float(sys.argv[1])
    monthly_budget = float(sys.argv[2]) if len(sys.argv) > 2 else daily_budget * 20
    
    # Initialize pipeline monitor
    monitor = PipelineMonitor(daily_budget=daily_budget, monthly_budget=monthly_budget)
    
    # Simulate multiple pipeline executions
    for i in range(3):
        project_id = f"test_project_{i+1}"
        monitor.simulate_pipeline_execution(project_id)
        
        # Add some delay between executions
        await asyncio.sleep(0.1)
    
    # Generate dashboard data
    dashboard_data = monitor.get_dashboard_data()
    
    # Print results
    print("\n" + "=" * 60)
    print("📊 METRICS & BUDGET GUARDRAILS RESULTS")
    print("=" * 60)
    
    budget_status = dashboard_data['budget_status']
    usage_report = dashboard_data['usage_report']
    real_time = dashboard_data['real_time_metrics']
    
    print(f"💰 Daily Budget: ${budget_status['daily_budget']:.2f}")
    print(f"💰 Daily Spent: ${budget_status['current_daily_spend']:.2f} ({budget_status['daily_usage_percent']:.1f}%)")
    print(f"💰 Daily Remaining: ${budget_status['daily_remaining']:.2f}")
    print(f"💰 Monthly Budget: ${budget_status['monthly_budget']:.2f}")
    print(f"💰 Monthly Spent: ${budget_status['current_monthly_spend']:.2f} ({budget_status['monthly_usage_percent']:.1f}%)")
    
    print(f"\n📊 Total Cost: ${usage_report['total_cost']:.4f}")
    print(f"📊 API Calls: {usage_report['total_api_calls']}")
    print(f"📊 Processing Time: {usage_report['total_processing_time']:.2f}s")
    print(f"📊 Projects Processed: {usage_report['projects_processed']}")
    print(f"📊 Success Rate: {usage_report['success_rate']:.1%}")
    print(f"📊 Alerts: {len(usage_report['budget_alerts'])}")
    
    if usage_report['cost_by_service']:
        print(f"\n💸 Cost by Service:")
        for service, cost in usage_report['cost_by_service'].items():
            print(f"  {service}: ${cost:.4f}")
    
    if usage_report['performance_metrics']:
        print(f"\n⚡ Performance Metrics:")
        for metric, value in usage_report['performance_metrics'].items():
            if 'rate' in metric:
                print(f"  {metric}: {value:.1%}")
            elif 'duration' in metric:
                print(f"  {metric}: {value:.2f}s")
            else:
                print(f"  {metric}: {value:.2f}")
    
    if usage_report['budget_alerts']:
        print(f"\n🚨 Budget Alerts:")
        for alert in usage_report['budget_alerts']:
            print(f"  {alert['alert_type'].upper()}: {alert['message']}")
    
    # Save metrics
    metrics_file = f"metrics_report_{int(time.time())}.json"
    monitor.metrics_collector.save_metrics(metrics_file)
    
    # Save usage report
    usage_file = f"usage_report_{int(time.time())}.json"
    with open(usage_file, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
    
    print(f"\n💾 Metrics saved to: {metrics_file}")
    print(f"💾 Usage report saved to: {usage_file}")

if __name__ == "__main__":
    asyncio.run(main())