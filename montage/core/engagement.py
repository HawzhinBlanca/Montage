"""
Engagement Telemetry - Tasks.md Step 0  
GET /watch/<job_id>?t=<sec> and /complete endpoints storing in PostgreSQL table watch_events
"""
import time
from typing import Any, Dict

from ..core.db import Database


class EngagementTracker:
    """Track user engagement events for A/B testing analysis"""

    def __init__(self):
        self.db = Database()

    def record_watch_ping(self, job_id: str, timestamp_sec: int, variant: str) -> None:
        """Record watch ping - Tasks.md: GET /watch/<job_id>?t=<sec>"""

        # Store in PostgreSQL table watch_events as specified
        query = """
        INSERT INTO watch_events 
        (job_id, variant, event_type, timestamp_sec, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        self.db.execute(query, (
            job_id,
            variant,
            'watch_ping',
            timestamp_sec,
            time.time()
        ))

    def record_completion(self, job_id: str, variant: str) -> None:
        """Record completion - Tasks.md: /complete on 90% watch-through"""

        # Store in PostgreSQL table watch_events as specified
        query = """
        INSERT INTO watch_events
        (job_id, variant, event_type, timestamp_sec, created_at)
        VALUES (%s, %s, %s, %s, %s)
        """

        self.db.execute(query, (
            job_id,
            variant,
            'completion',
            None,  # No specific timestamp for completion
            time.time()
        ))

    def get_engagement_metrics(self, variant: str, hours: int = 24) -> Dict[str, Any]:
        """Get engagement metrics for A/B testing analysis from watch_events table"""

        since_timestamp = time.time() - (hours * 3600)

        # Watch ping events
        watch_query = """
        SELECT COUNT(*), COUNT(DISTINCT job_id)
        FROM watch_events 
        WHERE variant = %s AND event_type = 'watch_ping' AND created_at > %s
        """
        watch_results = self.db.fetchone(watch_query, (variant, since_timestamp))

        # Completion events
        complete_query = """
        SELECT COUNT(*), COUNT(DISTINCT job_id)
        FROM watch_events
        WHERE variant = %s AND event_type = 'completion' AND created_at > %s
        """
        complete_results = self.db.fetchone(complete_query, (variant, since_timestamp))

        return {
            'variant': variant,
            'hours_analyzed': hours,
            'watch_metrics': {
                'total_pings': watch_results[0] or 0,
                'unique_viewers': watch_results[1] or 0
            },
            'completion_metrics': {
                'total_completions': complete_results[0] or 0,
                'unique_completions': complete_results[1] or 0
            }
        }

    def create_watch_events_table(self) -> None:
        """Create watch_events table in PostgreSQL - Tasks.md specification"""
        query = """
        CREATE TABLE IF NOT EXISTS watch_events (
            id SERIAL PRIMARY KEY,
            job_id TEXT NOT NULL,
            variant TEXT NOT NULL,
            event_type TEXT NOT NULL,
            timestamp_sec INTEGER,
            created_at REAL NOT NULL
        )
        """
        self.db.execute(query)
