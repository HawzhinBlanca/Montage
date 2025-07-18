"""
User Success Metrics - Track what actually matters
Not vanity metrics, but real user satisfaction indicators
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import asyncio
import numpy as np

from db_secure import SecureDatabase
from metrics import metrics

logger = logging.getLogger(__name__)


@dataclass
class UserSession:
    """Track a single user session"""
    session_id: str
    user_id: Optional[str]
    started_at: float
    video_uploaded: bool = False
    video_processed: bool = False
    video_exported: bool = False
    processing_mode: Optional[str] = None
    regeneration_count: int = 0
    face_in_frame_percent: float = 0.0
    cost_usd: float = 0.0
    processing_seconds: float = 0.0
    error_message: Optional[str] = None
    abandoned_at_stage: Optional[str] = None
    satisfaction_score: Optional[float] = None  # From user feedback
    segments_kept: int = 0
    segments_total: int = 0


class UserSuccessMetrics:
    """
    Track real success metrics that indicate user satisfaction
    """
    
    def __init__(self):
        self.db = SecureDatabase()
        self._init_tables()
        self.active_sessions = {}
        
    def _init_tables(self):
        """Create metrics tables if not exists"""
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    started_at TIMESTAMP,
                    ended_at TIMESTAMP,
                    video_uploaded BOOLEAN DEFAULT FALSE,
                    video_processed BOOLEAN DEFAULT FALSE, 
                    video_exported BOOLEAN DEFAULT FALSE,
                    processing_mode TEXT,
                    regeneration_count INTEGER DEFAULT 0,
                    face_in_frame_percent REAL DEFAULT 0,
                    cost_usd REAL DEFAULT 0,
                    processing_seconds REAL DEFAULT 0,
                    error_message TEXT,
                    abandoned_at_stage TEXT,
                    satisfaction_score REAL,
                    segments_kept INTEGER DEFAULT 0,
                    segments_total INTEGER DEFAULT 0,
                    metadata JSONB
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS daily_metrics (
                    date DATE PRIMARY KEY,
                    videos_started INTEGER DEFAULT 0,
                    videos_completed INTEGER DEFAULT 0,
                    completion_rate REAL DEFAULT 0,
                    avg_retries REAL DEFAULT 0,
                    avg_face_coverage REAL DEFAULT 0,
                    avg_cost REAL DEFAULT 0,
                    median_processing_time REAL DEFAULT 0,
                    mode_distribution JSONB,
                    error_rate REAL DEFAULT 0,
                    satisfaction_avg REAL
                )
            """)
            
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS abandonment_analysis (
                    date DATE,
                    stage TEXT,
                    count INTEGER DEFAULT 0,
                    reasons JSONB,
                    PRIMARY KEY (date, stage)
                )
            """)
            
            # Create the view from Tasks.md
            self.db.execute("""
                CREATE OR REPLACE VIEW user_success_metrics AS
                SELECT 
                    DATE(started_at) as day,
                    COUNT(*) as videos_started,
                    COUNT(CASE WHEN video_exported THEN 1 END) as videos_completed,
                    AVG(CASE WHEN video_exported THEN 1 ELSE 0 END) as completion_rate,
                    AVG(regeneration_count) as avg_retries,
                    AVG(face_in_frame_percent) as avg_face_coverage,
                    AVG(cost_usd) as avg_cost,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY processing_seconds) as median_time
                FROM user_sessions
                GROUP BY DATE(started_at)
            """)
            
        except Exception as e:
            logger.error(f"Failed to init metrics tables: {e}")
    
    def start_session(self, session_id: str, user_id: Optional[str] = None) -> str:
        """Start tracking a new session"""
        session = UserSession(
            session_id=session_id,
            user_id=user_id,
            started_at=time.time()
        )
        
        self.active_sessions[session_id] = session
        
        # Record in database
        self.db.insert('user_sessions', {
            'session_id': session_id,
            'user_id': user_id,
            'started_at': datetime.fromtimestamp(session.started_at),
            'metadata': json.dumps({'user_agent': 'adaptive_pipeline'})
        })
        
        # Prometheus metric
        metrics.video_job_total.labels(status='started').inc()
        
        return session_id
    
    def track_upload(self, session_id: str, video_metadata: Dict[str, Any]):
        """Track video upload"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].video_uploaded = True
            
            self.db.update(
                'user_sessions',
                {'session_id': session_id},
                {
                    'video_uploaded': True,
                    'metadata': json.dumps(video_metadata)
                }
            )
    
    def track_processing(self, 
                        session_id: str,
                        mode: str,
                        segments: List[Dict[str, Any]],
                        cost: float = 0.0):
        """Track processing completion"""
        if session_id not in self.active_sessions:
            logger.warning(f"Unknown session: {session_id}")
            return
        
        session = self.active_sessions[session_id]
        session.video_processed = True
        session.processing_mode = mode
        session.cost_usd = cost
        session.segments_total = len(segments)
        
        # Calculate face coverage
        face_coverage = self._calculate_face_coverage(segments)
        session.face_in_frame_percent = face_coverage
        
        # Update database
        self.db.update(
            'user_sessions',
            {'session_id': session_id},
            {
                'video_processed': True,
                'processing_mode': mode,
                'cost_usd': cost,
                'segments_total': len(segments),
                'face_in_frame_percent': face_coverage
            }
        )
        
        # Prometheus metrics
        metrics.video_processing_complete.labels(mode=mode).inc()
        if cost > 0:
            metrics.api_cost_total.labels(api='adaptive').inc(cost)
    
    def track_export(self, session_id: str, segments_kept: int):
        """Track successful export"""
        if session_id not in self.active_sessions:
            return
        
        session = self.active_sessions[session_id]
        session.video_exported = True
        session.segments_kept = segments_kept
        session.processing_seconds = time.time() - session.started_at
        
        # Update database
        self.db.update(
            'user_sessions',
            {'session_id': session_id},
            {
                'video_exported': True,
                'segments_kept': segments_kept,
                'processing_seconds': session.processing_seconds,
                'ended_at': datetime.now()
            }
        )
        
        # Success metric
        metrics.video_job_total.labels(status='completed').inc()
        
        # Log success pattern
        logger.info(f"Success: {session.processing_mode} mode, "
                   f"{segments_kept}/{session.segments_total} segments kept, "
                   f"${session.cost_usd:.2f} cost, "
                   f"{session.processing_seconds:.1f}s total")
    
    def track_regeneration(self, session_id: str):
        """Track when user regenerates (indicates dissatisfaction)"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].regeneration_count += 1
            
            self.db.execute("""
                UPDATE user_sessions 
                SET regeneration_count = regeneration_count + 1
                WHERE session_id = %s
            """, (session_id,))
    
    def track_abandonment(self, session_id: str, stage: str, reason: Optional[str] = None):
        """Track where users abandon the process"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.abandoned_at_stage = stage
            
            self.db.update(
                'user_sessions',
                {'session_id': session_id},
                {
                    'abandoned_at_stage': stage,
                    'error_message': reason,
                    'ended_at': datetime.now()
                }
            )
            
            # Track abandonment patterns
            self._record_abandonment(stage, reason)
    
    def track_error(self, session_id: str, error: str):
        """Track processing errors"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].error_message = error
            
            self.db.update(
                'user_sessions',
                {'session_id': session_id},
                {
                    'error_message': error,
                    'ended_at': datetime.now()
                }
            )
            
            metrics.video_job_total.labels(status='failed').inc()
    
    def track_satisfaction(self, session_id: str, score: float):
        """Track user satisfaction (1-5 scale)"""
        if session_id in self.active_sessions:
            self.active_sessions[session_id].satisfaction_score = score
            
            self.db.update(
                'user_sessions',
                {'session_id': session_id},
                {'satisfaction_score': score}
            )
    
    def _calculate_face_coverage(self, segments: List[Dict[str, Any]]) -> float:
        """Calculate percentage of segments with good face coverage"""
        if not segments:
            return 0.0
        
        good_coverage_count = 0
        for segment in segments:
            # Check if segment has face data and good crop
            if segment.get('type') == 'face' and segment.get('crop_params'):
                # Assume good coverage if smart crop was applied
                good_coverage_count += 1
            elif segment.get('features', {}).get('face_count', 0) > 0:
                good_coverage_count += 1
        
        return (good_coverage_count / len(segments)) * 100
    
    def _record_abandonment(self, stage: str, reason: Optional[str] = None):
        """Record abandonment patterns"""
        today = datetime.now().date()
        
        # Update or insert abandonment record
        existing = self.db.fetch_one("""
            SELECT reasons FROM abandonment_analysis
            WHERE date = %s AND stage = %s
        """, (today, stage))
        
        if existing:
            reasons = json.loads(existing['reasons'] or '{}')
            reason_key = reason or 'unknown'
            reasons[reason_key] = reasons.get(reason_key, 0) + 1
            
            self.db.execute("""
                UPDATE abandonment_analysis
                SET count = count + 1, reasons = %s
                WHERE date = %s AND stage = %s
            """, (json.dumps(reasons), today, stage))
        else:
            self.db.insert('abandonment_analysis', {
                'date': today,
                'stage': stage,
                'count': 1,
                'reasons': json.dumps({reason or 'unknown': 1})
            })
    
    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily success metrics report"""
        # Get today's metrics
        today = datetime.now().date()
        
        metrics_data = self.db.fetch_one("""
            SELECT * FROM user_success_metrics
            WHERE day = %s
        """, (today,))
        
        if not metrics_data:
            return {'error': 'No data for today'}
        
        # Get mode distribution
        mode_dist = self.db.fetch_all("""
            SELECT processing_mode, COUNT(*) as count
            FROM user_sessions
            WHERE DATE(started_at) = %s AND video_processed = TRUE
            GROUP BY processing_mode
        """, (today,))
        
        mode_distribution = {row['processing_mode']: row['count'] for row in mode_dist}
        
        # Get abandonment data
        abandonment = self.db.fetch_all("""
            SELECT stage, count, reasons
            FROM abandonment_analysis
            WHERE date = %s
            ORDER BY count DESC
        """, (today,))
        
        # Calculate key insights
        insights = self._generate_insights(metrics_data, mode_distribution, abandonment)
        
        return {
            'date': str(today),
            'summary': {
                'videos_started': metrics_data['videos_started'],
                'videos_completed': metrics_data['videos_completed'],
                'completion_rate': f"{metrics_data['completion_rate']:.1%}",
                'avg_retries': round(metrics_data['avg_retries'], 1),
                'avg_face_coverage': f"{metrics_data['avg_face_coverage']:.1%}",
                'avg_cost': f"${metrics_data['avg_cost']:.2f}",
                'median_time': f"{metrics_data['median_time']:.1f}s"
            },
            'mode_distribution': mode_distribution,
            'abandonment_stages': [
                {
                    'stage': a['stage'],
                    'count': a['count'],
                    'top_reasons': json.loads(a['reasons'])
                }
                for a in abandonment
            ],
            'insights': insights
        }
    
    def _generate_insights(self, 
                          metrics: Dict[str, Any],
                          mode_dist: Dict[str, int],
                          abandonment: List[Dict]) -> List[str]:
        """Generate actionable insights from metrics"""
        insights = []
        
        # Completion rate insight
        if metrics['completion_rate'] < 0.7:
            insights.append(f"⚠️ Low completion rate ({metrics['completion_rate']:.1%}). "
                          f"Check abandonment stages.")
        elif metrics['completion_rate'] > 0.9:
            insights.append(f"✅ Excellent completion rate ({metrics['completion_rate']:.1%})!")
        
        # Retry insight
        if metrics['avg_retries'] > 1.5:
            insights.append(f"⚠️ Users averaging {metrics['avg_retries']:.1f} retries. "
                          f"Initial results may not be satisfying.")
        
        # Cost insight
        if metrics['avg_cost'] > 0.5:
            insights.append(f"💰 High average cost (${metrics['avg_cost']:.2f}). "
                          f"Consider optimizing expensive API usage.")
        
        # Mode distribution insight
        if mode_dist:
            most_used = max(mode_dist.items(), key=lambda x: x[1])
            insights.append(f"📊 Most used mode: {most_used[0]} ({most_used[1]} times)")
        
        # Face coverage insight
        if metrics['avg_face_coverage'] < 50:
            insights.append(f"📷 Low face coverage ({metrics['avg_face_coverage']:.1%}). "
                          f"Smart crop may need improvement.")
        
        # Processing time insight
        if metrics['median_time'] > 180:
            insights.append(f"⏱️ Long processing time ({metrics['median_time']:.0f}s). "
                          f"Users may be abandoning.")
        
        return insights
    
    def get_session_metrics(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific session"""
        if session_id in self.active_sessions:
            return asdict(self.active_sessions[session_id])
        
        # Try database
        session = self.db.fetch_one("""
            SELECT * FROM user_sessions WHERE session_id = %s
        """, (session_id,))
        
        return dict(session) if session else None
    
    def cleanup_old_sessions(self, hours: int = 24):
        """Clean up old sessions from memory"""
        cutoff = time.time() - (hours * 3600)
        
        to_remove = []
        for sid, session in self.active_sessions.items():
            if session.started_at < cutoff:
                to_remove.append(sid)
        
        for sid in to_remove:
            del self.active_sessions[sid]
        
        logger.info(f"Cleaned up {len(to_remove)} old sessions")


# Global instance
success_metrics = UserSuccessMetrics()


async def demo_metrics():
    """Demonstrate metrics tracking"""
    
    # Simulate a successful session
    session1 = success_metrics.start_session("demo_session_1", "user_123")
    
    # Upload
    success_metrics.track_upload(session1, {
        'duration': 300,
        'resolution': '1920x1080',
        'size_mb': 150
    })
    
    # Process
    success_metrics.track_processing(
        session1,
        mode='smart_enhanced',
        segments=[
            {'type': 'face', 'crop_params': {'x': 100}},
            {'type': 'motion'},
            {'type': 'face', 'crop_params': {'x': 200}}
        ],
        cost=0.15
    )
    
    # Export
    success_metrics.track_export(session1, segments_kept=3)
    
    # Simulate abandoned session
    session2 = success_metrics.start_session("demo_session_2", "user_456")
    success_metrics.track_upload(session2, {'duration': 600})
    success_metrics.track_abandonment(session2, 'processing', 'too_slow')
    
    # Generate report
    report = await success_metrics.generate_daily_report()
    
    print("\n📊 DAILY SUCCESS METRICS REPORT")
    print("=" * 50)
    print(json.dumps(report, indent=2))
    
    # Get specific session
    session_data = success_metrics.get_session_metrics(session1)
    print(f"\nSession {session1} metrics:")
    print(json.dumps(session_data, indent=2))


if __name__ == "__main__":
    asyncio.run(demo_metrics())