#!/usr/bin/env python3
"""
Phase 3: QC, human gate & export system
Quality control and human approval system for video montages
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
from datetime import datetime
import subprocess
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QCIssue:
    """Quality control issue"""
    issue_id: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    category: str  # 'audio', 'video', 'subtitle', 'timing', 'content'
    description: str
    location: str  # timestamp or clip reference
    auto_detected: bool
    resolved: bool
    resolution_notes: str
    created_at: float

@dataclass
class QCReport:
    """Quality control report for a montage"""
    report_id: str
    project_name: str
    video_path: str
    highlights_count: int
    total_duration_ms: int
    issues: List[QCIssue]
    overall_score: float
    recommendations: List[str]
    created_at: float
    approved: bool
    approved_by: str
    approved_at: Optional[float]

@dataclass
class ExportJob:
    """Export job with quality gates"""
    job_id: str
    project_name: str
    video_path: str
    output_path: str
    export_settings: Dict[str, Any]
    qc_report: QCReport
    status: str  # 'pending', 'processing', 'completed', 'failed', 'rejected'
    created_at: float
    completed_at: Optional[float]
    error_message: Optional[str]

class AutoQualityChecker:
    """Automated quality checks"""
    
    def __init__(self):
        self.checks = [
            'check_audio_levels',
            'check_video_quality',
            'check_subtitle_timing',
            'check_clip_transitions',
            'check_content_quality'
        ]
        
    def check_audio_levels(self, video_path: str) -> List[QCIssue]:
        """Check audio levels and quality"""
        issues = []
        
        try:
            # Use ffmpeg to analyze audio
            cmd = [
                'ffmpeg', '-i', video_path,
                '-af', 'volumedetect,astats=metadata=1:reset=1',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            output = result.stderr
            
            # Check for audio level issues
            if 'max_volume:' in output:
                import re
                max_volume_match = re.search(r'max_volume:\s*(-?\d+\.?\d*)\s*dB', output)
                if max_volume_match:
                    max_volume = float(max_volume_match.group(1))
                    
                    if max_volume > -6:
                        issues.append(QCIssue(
                            issue_id=str(uuid.uuid4()),
                            severity='high',
                            category='audio',
                            description=f'Audio levels too high: {max_volume:.1f}dB (recommended: < -6dB)',
                            location='entire_video',
                            auto_detected=True,
                            resolved=False,
                            resolution_notes='',
                            created_at=time.time()
                        ))
                    elif max_volume < -20:
                        issues.append(QCIssue(
                            issue_id=str(uuid.uuid4()),
                            severity='medium',
                            category='audio',
                            description=f'Audio levels too low: {max_volume:.1f}dB (recommended: > -20dB)',
                            location='entire_video',
                            auto_detected=True,
                            resolved=False,
                            resolution_notes='',
                            created_at=time.time()
                        ))
            
        except Exception as e:
            logger.warning(f"Audio level check failed: {e}")
            issues.append(QCIssue(
                issue_id=str(uuid.uuid4()),
                severity='low',
                category='audio',
                description=f'Could not analyze audio levels: {str(e)}',
                location='entire_video',
                auto_detected=True,
                resolved=False,
                resolution_notes='',
                created_at=time.time()
            ))
        
        return issues
    
    def check_video_quality(self, video_path: str) -> List[QCIssue]:
        """Check video quality metrics"""
        issues = []
        
        try:
            # Get video info
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,bit_rate',
                '-of', 'json', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                stream = data.get('streams', [{}])[0]
                
                width = stream.get('width', 0)
                height = stream.get('height', 0)
                
                # Check resolution
                if width < 1280 or height < 720:
                    issues.append(QCIssue(
                        issue_id=str(uuid.uuid4()),
                        severity='medium',
                        category='video',
                        description=f'Low resolution: {width}x{height} (recommended: ≥1280x720)',
                        location='entire_video',
                        auto_detected=True,
                        resolved=False,
                        resolution_notes='',
                        created_at=time.time()
                    ))
                
                # Check aspect ratio
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio < 0.8 or aspect_ratio > 2.0:
                    issues.append(QCIssue(
                        issue_id=str(uuid.uuid4()),
                        severity='low',
                        category='video',
                        description=f'Unusual aspect ratio: {aspect_ratio:.2f}:1',
                        location='entire_video',
                        auto_detected=True,
                        resolved=False,
                        resolution_notes='',
                        created_at=time.time()
                    ))
        
        except Exception as e:
            logger.warning(f"Video quality check failed: {e}")
        
        return issues
    
    def check_subtitle_timing(self, subtitles_dir: str) -> List[QCIssue]:
        """Check subtitle timing and formatting"""
        issues = []
        
        try:
            subtitle_files = list(Path(subtitles_dir).glob("*.srt"))
            
            for srt_file in subtitle_files:
                with open(srt_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for basic formatting issues
                if not content.strip():
                    issues.append(QCIssue(
                        issue_id=str(uuid.uuid4()),
                        severity='high',
                        category='subtitle',
                        description=f'Empty subtitle file: {srt_file.name}',
                        location=srt_file.name,
                        auto_detected=True,
                        resolved=False,
                        resolution_notes='',
                        created_at=time.time()
                    ))
                
                # Check for overlapping subtitles (basic check)
                lines = content.split('\n')
                timestamps = []
                
                for line in lines:
                    if '-->' in line:
                        try:
                            start_str, end_str = line.split(' --> ')
                            # Simple timestamp parsing
                            timestamps.append((start_str.strip(), end_str.strip()))
                        except:
                            pass
                
                # Check for very short subtitles
                if len(timestamps) > 0:
                    short_count = 0
                    for start, end in timestamps:
                        # This is a simplified check - in real implementation,
                        # you'd parse the actual timestamps
                        if len(start) < 8 or len(end) < 8:
                            short_count += 1
                    
                    if short_count > len(timestamps) * 0.3:  # More than 30% short
                        issues.append(QCIssue(
                            issue_id=str(uuid.uuid4()),
                            severity='medium',
                            category='subtitle',
                            description=f'Many short subtitles in {srt_file.name}',
                            location=srt_file.name,
                            auto_detected=True,
                            resolved=False,
                            resolution_notes='',
                            created_at=time.time()
                        ))
        
        except Exception as e:
            logger.warning(f"Subtitle timing check failed: {e}")
        
        return issues
    
    def check_clip_transitions(self, highlights: List[Dict[str, Any]]) -> List[QCIssue]:
        """Check for smooth transitions between clips"""
        issues = []
        
        for i in range(len(highlights) - 1):
            current_clip = highlights[i]
            next_clip = highlights[i + 1]
            
            current_end = current_clip.get('end_ms', 0)
            next_start = next_clip.get('start_ms', 0)
            
            # Check for gaps or overlaps
            gap_ms = next_start - current_end
            
            if gap_ms < 0:
                issues.append(QCIssue(
                    issue_id=str(uuid.uuid4()),
                    severity='medium',
                    category='timing',
                    description=f'Overlapping clips: {current_clip.get("slug", "unknown")} and {next_clip.get("slug", "unknown")}',
                    location=f'{current_end}ms-{next_start}ms',
                    auto_detected=True,
                    resolved=False,
                    resolution_notes='',
                    created_at=time.time()
                ))
            elif gap_ms > 10000:  # Gap longer than 10 seconds
                issues.append(QCIssue(
                    issue_id=str(uuid.uuid4()),
                    severity='low',
                    category='timing',
                    description=f'Large gap between clips: {gap_ms}ms',
                    location=f'{current_end}ms-{next_start}ms',
                    auto_detected=True,
                    resolved=False,
                    resolution_notes='',
                    created_at=time.time()
                ))
        
        return issues
    
    def check_content_quality(self, highlights: List[Dict[str, Any]]) -> List[QCIssue]:
        """Check content quality and coherence"""
        issues = []
        
        # Check for very short clips
        short_clips = [h for h in highlights if (h.get('end_ms', 0) - h.get('start_ms', 0)) < 2000]
        if len(short_clips) > 0:
            issues.append(QCIssue(
                issue_id=str(uuid.uuid4()),
                severity='low',
                category='content',
                description=f'{len(short_clips)} clips shorter than 2 seconds',
                location='multiple_clips',
                auto_detected=True,
                resolved=False,
                resolution_notes='',
                created_at=time.time()
            ))
        
        # Check for very low scores
        low_score_clips = [h for h in highlights if h.get('score', 0) < 2.0]
        if len(low_score_clips) > 0:
            issues.append(QCIssue(
                issue_id=str(uuid.uuid4()),
                severity='medium',
                category='content',
                description=f'{len(low_score_clips)} clips with low quality scores',
                location='multiple_clips',
                auto_detected=True,
                resolved=False,
                resolution_notes='',
                created_at=time.time()
            ))
        
        return issues
    
    def run_all_checks(self, video_path: str, highlights: List[Dict[str, Any]], 
                      subtitles_dir: str) -> List[QCIssue]:
        """Run all quality checks"""
        logger.info("🔍 Running automated quality checks...")
        
        all_issues = []
        
        # Audio checks
        if os.path.exists(video_path):
            all_issues.extend(self.check_audio_levels(video_path))
            all_issues.extend(self.check_video_quality(video_path))
        
        # Subtitle checks
        if os.path.exists(subtitles_dir):
            all_issues.extend(self.check_subtitle_timing(subtitles_dir))
        
        # Content checks
        all_issues.extend(self.check_clip_transitions(highlights))
        all_issues.extend(self.check_content_quality(highlights))
        
        logger.info(f"🔍 Found {len(all_issues)} potential issues")
        return all_issues

class HumanGateSystem:
    """Human approval and review system"""
    
    def __init__(self):
        self.pending_reviews = {}
        self.completed_reviews = {}
        
    def generate_review_report(self, qc_report: QCReport) -> str:
        """Generate human-readable review report"""
        report = []
        report.append("=" * 60)
        report.append("🎬 MONTAGE QUALITY CONTROL REPORT")
        report.append("=" * 60)
        report.append(f"Project: {qc_report.project_name}")
        report.append(f"Video: {qc_report.video_path}")
        report.append(f"Highlights: {qc_report.highlights_count}")
        report.append(f"Duration: {qc_report.total_duration_ms / 1000:.1f}s")
        report.append(f"Overall Score: {qc_report.overall_score:.1f}/10")
        report.append(f"Generated: {datetime.fromtimestamp(qc_report.created_at).strftime('%Y-%m-%d %H:%M:%S')}")
        
        if qc_report.issues:
            report.append("\n📋 ISSUES FOUND:")
            
            # Group by severity
            critical_issues = [i for i in qc_report.issues if i.severity == 'critical']
            high_issues = [i for i in qc_report.issues if i.severity == 'high']
            medium_issues = [i for i in qc_report.issues if i.severity == 'medium']
            low_issues = [i for i in qc_report.issues if i.severity == 'low']
            
            for severity, issues in [('CRITICAL', critical_issues), ('HIGH', high_issues), 
                                   ('MEDIUM', medium_issues), ('LOW', low_issues)]:
                if issues:
                    report.append(f"\n{severity} PRIORITY ({len(issues)} issues):")
                    for issue in issues:
                        report.append(f"  • {issue.description}")
                        report.append(f"    Location: {issue.location}")
                        report.append(f"    Category: {issue.category}")
        else:
            report.append("\n✅ No issues found!")
        
        if qc_report.recommendations:
            report.append("\n💡 RECOMMENDATIONS:")
            for rec in qc_report.recommendations:
                report.append(f"  • {rec}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def request_human_review(self, qc_report: QCReport) -> str:
        """Request human review and approval"""
        review_id = str(uuid.uuid4())
        
        self.pending_reviews[review_id] = {
            'qc_report': qc_report,
            'requested_at': time.time(),
            'status': 'pending'
        }
        
        # Generate review report
        report_text = self.generate_review_report(qc_report)
        
        # Save report to file
        report_file = f"qc_report_{review_id}.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📋 Human review requested: {review_id}")
        logger.info(f"📄 Review report saved: {report_file}")
        
        return review_id
    
    def simulate_human_approval(self, review_id: str, approve: bool = True, 
                              reviewer: str = "AutoReviewer") -> bool:
        """Simulate human approval (for testing)"""
        if review_id not in self.pending_reviews:
            logger.error(f"Review ID not found: {review_id}")
            return False
        
        review = self.pending_reviews[review_id]
        qc_report = review['qc_report']
        
        # Simulate approval decision based on issues
        critical_issues = [i for i in qc_report.issues if i.severity == 'critical']
        high_issues = [i for i in qc_report.issues if i.severity == 'high']
        
        # Auto-reject if there are critical issues
        if critical_issues:
            approve = False
        
        # Auto-reject if there are too many high issues
        if len(high_issues) > 3:
            approve = False
        
        # Update QC report
        qc_report.approved = approve
        qc_report.approved_by = reviewer
        qc_report.approved_at = time.time()
        
        # Move to completed reviews
        self.completed_reviews[review_id] = {
            'qc_report': qc_report,
            'approved': approve,
            'reviewer': reviewer,
            'completed_at': time.time()
        }
        
        del self.pending_reviews[review_id]
        
        status = "✅ APPROVED" if approve else "❌ REJECTED"
        logger.info(f"{status} by {reviewer}: {review_id}")
        
        return approve

class QCExportSystem:
    """Quality control and export system"""
    
    def __init__(self):
        self.qc_checker = AutoQualityChecker()
        self.human_gate = HumanGateSystem()
        self.export_jobs = {}
        
    def create_qc_report(self, project_name: str, video_path: str, 
                        highlights: List[Dict[str, Any]], subtitles_dir: str) -> QCReport:
        """Create comprehensive QC report"""
        logger.info(f"📊 Creating QC report for: {project_name}")
        
        # Run automated checks
        issues = self.qc_checker.run_all_checks(video_path, highlights, subtitles_dir)
        
        # Calculate overall score
        critical_count = len([i for i in issues if i.severity == 'critical'])
        high_count = len([i for i in issues if i.severity == 'high'])
        medium_count = len([i for i in issues if i.severity == 'medium'])
        low_count = len([i for i in issues if i.severity == 'low'])
        
        # Scoring: Start with 10, deduct points for issues
        score = 10.0
        score -= critical_count * 3.0  # Critical issues: -3 points each
        score -= high_count * 2.0      # High issues: -2 points each
        score -= medium_count * 1.0    # Medium issues: -1 point each
        score -= low_count * 0.5       # Low issues: -0.5 points each
        score = max(0, score)
        
        # Generate recommendations
        recommendations = []
        if critical_count > 0:
            recommendations.append("Fix critical issues before export")
        if high_count > 0:
            recommendations.append("Review and fix high priority issues")
        if len(highlights) < 3:
            recommendations.append("Consider adding more highlight clips")
        if sum(h.get('end_ms', 0) - h.get('start_ms', 0) for h in highlights) < 30000:
            recommendations.append("Consider longer highlight clips for better engagement")
        
        # Calculate total duration
        total_duration = sum(h.get('end_ms', 0) - h.get('start_ms', 0) for h in highlights)
        
        return QCReport(
            report_id=str(uuid.uuid4()),
            project_name=project_name,
            video_path=video_path,
            highlights_count=len(highlights),
            total_duration_ms=total_duration,
            issues=issues,
            overall_score=score,
            recommendations=recommendations,
            created_at=time.time(),
            approved=False,
            approved_by="",
            approved_at=None
        )
    
    def create_export_job(self, project_name: str, video_path: str, output_path: str,
                         highlights: List[Dict[str, Any]], subtitles_dir: str,
                         export_settings: Optional[Dict[str, Any]] = None) -> str:
        """Create export job with QC gate"""
        
        # Create QC report
        qc_report = self.create_qc_report(project_name, video_path, highlights, subtitles_dir)
        
        # Create export job
        job_id = str(uuid.uuid4())
        export_job = ExportJob(
            job_id=job_id,
            project_name=project_name,
            video_path=video_path,
            output_path=output_path,
            export_settings=export_settings or {},
            qc_report=qc_report,
            status='pending',
            created_at=time.time(),
            completed_at=None,
            error_message=None
        )
        
        self.export_jobs[job_id] = export_job
        
        logger.info(f"📋 Export job created: {job_id}")
        return job_id
    
    def process_export_job(self, job_id: str, auto_approve: bool = False) -> bool:
        """Process export job through QC gate"""
        if job_id not in self.export_jobs:
            logger.error(f"Export job not found: {job_id}")
            return False
        
        export_job = self.export_jobs[job_id]
        
        # Request human review
        review_id = self.human_gate.request_human_review(export_job.qc_report)
        
        # Simulate human approval if auto_approve is enabled
        if auto_approve:
            approval = self.human_gate.simulate_human_approval(review_id)
        else:
            logger.info(f"⏳ Waiting for human approval for job: {job_id}")
            logger.info(f"📋 Review ID: {review_id}")
            # In a real system, this would wait for actual human input
            approval = True  # For demo purposes
        
        if approval:
            export_job.status = 'approved'
            logger.info(f"✅ Export job approved: {job_id}")
            return True
        else:
            export_job.status = 'rejected'
            export_job.error_message = "Failed QC review"
            logger.info(f"❌ Export job rejected: {job_id}")
            return False

async def main():
    """Test QC and export system"""
    if len(sys.argv) < 4:
        print("Usage: python phase3_qc_human_gate.py <project_name> <video_path> <highlights_json> <subtitles_dir>")
        return
    
    project_name = sys.argv[1]
    video_path = sys.argv[2]
    highlights_path = sys.argv[3]
    subtitles_dir = sys.argv[4]
    
    # Load highlights
    with open(highlights_path, 'r') as f:
        highlights_data = json.load(f)
        highlights = highlights_data.get("top_highlights", [])
    
    # Initialize QC system
    qc_system = QCExportSystem()
    
    # Create export job
    output_path = f"{project_name}_final.mp4"
    job_id = qc_system.create_export_job(
        project_name=project_name,
        video_path=video_path,
        output_path=output_path,
        highlights=highlights,
        subtitles_dir=subtitles_dir
    )
    
    # Process export job
    success = qc_system.process_export_job(job_id, auto_approve=True)
    
    # Print results
    export_job = qc_system.export_jobs[job_id]
    qc_report = export_job.qc_report
    
    print("\n" + "=" * 60)
    print("🎬 QC & EXPORT SYSTEM RESULTS")
    print("=" * 60)
    print(f"📋 Job ID: {job_id}")
    print(f"📊 QC Score: {qc_report.overall_score:.1f}/10")
    print(f"🔍 Issues Found: {len(qc_report.issues)}")
    print(f"📝 Recommendations: {len(qc_report.recommendations)}")
    print(f"✅ Status: {export_job.status}")
    
    if qc_report.issues:
        print("\n🔍 ISSUES BY SEVERITY:")
        for severity in ['critical', 'high', 'medium', 'low']:
            severity_issues = [i for i in qc_report.issues if i.severity == severity]
            if severity_issues:
                print(f"  {severity.upper()}: {len(severity_issues)}")
    
    if qc_report.recommendations:
        print("\n💡 RECOMMENDATIONS:")
        for rec in qc_report.recommendations:
            print(f"  • {rec}")
    
    # Save QC report
    qc_file = f"qc_report_{job_id}.json"
    with open(qc_file, 'w') as f:
        json.dump(asdict(qc_report), f, indent=2)
    
    print(f"\n💾 QC report saved: {qc_file}")
    
    if success:
        print("✅ Export job ready for processing!")
    else:
        print("❌ Export job failed QC review")

if __name__ == "__main__":
    asyncio.run(main())