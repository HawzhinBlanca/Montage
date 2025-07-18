"""
Adaptive Quality Pipeline - ONE button, intelligent routing
No more user confusion about which track to choose
"""

import os
import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from video_probe import VideoProbe, ProbeResult
from smart_track import SmartTrack
from selective_enhancer import SelectiveEnhancer
from progressive_renderer import ProgressiveRenderer
from metrics import metrics
from db import Database
from budget_guard import budget_guard
from cleanup_manager import cleanup_manager
from user_success_metrics import success_metrics

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Internal processing modes - invisible to user"""
    FAST = "fast"  # <5min, 1 speaker, simple
    SMART = "smart"  # 5-30min, moderate complexity
    SMART_ENHANCED = "smart_enhanced"  # Smart + selective API usage
    SELECTIVE_PREMIUM = "selective_premium"  # Complex sections only


@dataclass
class UserConstraints:
    """User preferences/constraints"""
    max_budget: float = 1.00  # Default $1 max
    max_wait_time: float = 300  # Default 5 min max
    allows_cloud: bool = True  # Privacy preference
    target_duration: Optional[float] = 60  # Target output duration
    quality_preference: str = "balanced"  # speed vs quality


@dataclass
class ProcessingResult:
    """Result of adaptive processing"""
    video_path: str
    mode_used: ProcessingMode
    actual_cost: float
    processing_time: float
    quality_score: float
    segments: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AdaptiveQualityPipeline:
    """
    ONE button: 'Create Video'. System decides optimal path.
    No more user paralysis choosing between tracks.
    """
    
    def __init__(self):
        self.probe = VideoProbe()
        self.smart_track = SmartTrack()
        self.enhancer = SelectiveEnhancer()
        self.renderer = ProgressiveRenderer()
        self.db = Database()
        
    async def process(self, 
                     video_path: str, 
                     user_constraints: Optional[UserConstraints] = None,
                     progress_callback: Optional[callable] = None,
                     session_id: Optional[str] = None) -> ProcessingResult:
        """
        Main entry point - ONE method that handles everything
        User just uploads video and gets magic
        """
        start_time = time.time()
        
        if user_constraints is None:
            user_constraints = UserConstraints()
        
        # Start metrics tracking
        if not session_id:
            session_id = f"session_{int(time.time() * 1000)}"
        success_metrics.start_session(session_id)
        
        # Track upload
        success_metrics.track_upload(session_id, {
            'path': video_path,
            'constraints': {
                'max_budget': user_constraints.max_budget,
                'max_wait_time': user_constraints.max_wait_time,
                'allows_cloud': user_constraints.allows_cloud
            }
        })
        
        # Create job record
        job_id = self._create_job(video_path, user_constraints)
        
        try:
            # 1. Quick probe to understand video (10 seconds)
            logger.info("Analyzing video characteristics...")
            probe_result = await self._probe_video(video_path)
            
            if progress_callback:
                await progress_callback({
                    'stage': 'analysis',
                    'progress': 10,
                    'message': f"Video analyzed: {probe_result.complexity} complexity"
                })
            
            # 2. Choose optimal processing path automatically
            processing_mode = self._select_processing_mode(probe_result, user_constraints)
            logger.info(f"Selected processing mode: {processing_mode.value}")
            
            # 3. Process based on selected mode
            if processing_mode == ProcessingMode.FAST:
                result = await self._process_fast(video_path, probe_result, job_id, progress_callback)
                
            elif processing_mode == ProcessingMode.SMART:
                result = await self._process_smart(video_path, probe_result, job_id, progress_callback)
                
            elif processing_mode == ProcessingMode.SMART_ENHANCED:
                result = await self._process_smart_enhanced(video_path, probe_result, job_id, progress_callback)
                
            elif processing_mode == ProcessingMode.SELECTIVE_PREMIUM:
                result = await self._process_selective_premium(video_path, probe_result, job_id, progress_callback)
            
            # 4. Record metrics
            processing_time = time.time() - start_time
            self._record_metrics(job_id, result, processing_time)
            
            # Track processing success
            success_metrics.track_processing(
                session_id,
                mode=processing_mode.value,
                segments=result.get('segments', []),
                cost=result.get('cost', 0)
            )
            
            # Track export (assuming user exports all segments)
            success_metrics.track_export(
                session_id,
                segments_kept=len(result.get('segments', []))
            )
            
            return ProcessingResult(
                video_path=result['output_path'],
                mode_used=processing_mode,
                actual_cost=result.get('cost', 0),
                processing_time=processing_time,
                quality_score=result.get('quality_score', 0.8),
                segments=result.get('segments', []),
                metadata=result.get('metadata', {})
            )
            
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            self._update_job_status(job_id, 'failed', str(e))
            
            # Track error
            success_metrics.track_error(session_id, str(e))
            
            raise
    
    async def _probe_video(self, video_path: str) -> ProbeResult:
        """Quick video analysis"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.probe.quick_probe, video_path)
    
    def _select_processing_mode(self, probe: ProbeResult, constraints: UserConstraints) -> ProcessingMode:
        """
        Intelligent mode selection based on video characteristics and user constraints
        This is the KEY - removes user choice paralysis
        """
        # Short, simple videos = fast track
        if probe.duration < 300 and probe.speaker_count == 1 and not probe.has_music:
            if probe.estimated_time <= constraints.max_wait_time:
                return ProcessingMode.FAST
        
        # Complex videos needing premium features
        if (probe.speaker_count > 2 or probe.technical_density > 0.7 or 
            probe.complexity == 'high'):
            # But respect budget constraints
            if probe.estimated_cost <= constraints.max_budget and constraints.allows_cloud:
                return ProcessingMode.SELECTIVE_PREMIUM
            else:
                # Fall back to enhanced smart track
                return ProcessingMode.SMART_ENHANCED
        
        # Medium complexity
        if probe.complexity == 'medium':
            # Use enhanced if we have budget and cloud permission
            if constraints.allows_cloud and constraints.max_budget > 0.1:
                return ProcessingMode.SMART_ENHANCED
            else:
                return ProcessingMode.SMART
        
        # Default to smart track
        return ProcessingMode.SMART
    
    async def _process_fast(self, video_path: str, probe: ProbeResult, 
                           job_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Fast processing for simple videos"""
        logger.info("Processing in FAST mode")
        
        # Quick segment extraction
        segments = self.smart_track.extract_quick_highlights(video_path, probe)
        
        if progress_callback:
            await progress_callback({
                'stage': 'processing',
                'progress': 50,
                'message': 'Extracting highlights...'
            })
        
        # Basic processing
        output_path = await self.renderer.render_basic(video_path, segments)
        
        if progress_callback:
            await progress_callback({
                'stage': 'complete',
                'progress': 100,
                'message': 'Video ready!'
            })
        
        return {
            'output_path': output_path,
            'segments': segments,
            'cost': 0.0,
            'quality_score': 0.7
        }
    
    async def _process_smart(self, video_path: str, probe: ProbeResult,
                            job_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Smart processing with local ML only"""
        logger.info("Processing in SMART mode")
        
        # Smart highlight detection
        result = await self.smart_track.process(video_path, probe)
        
        if progress_callback:
            await progress_callback({
                'stage': 'processing',
                'progress': 70,
                'message': 'Creating video...'
            })
        
        # Render with transitions
        output_path = await self.renderer.render_with_transitions(
            video_path, 
            result['segments']
        )
        
        return {
            'output_path': output_path,
            'segments': result['segments'],
            'cost': 0.0,
            'quality_score': 0.8,
            'metadata': result.get('metadata', {})
        }
    
    async def _process_smart_enhanced(self, video_path: str, probe: ProbeResult,
                                     job_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Smart processing with selective API enhancement"""
        logger.info("Processing in SMART_ENHANCED mode")
        
        # Start with smart track
        smart_result = await self.smart_track.process(video_path, probe)
        
        if progress_callback:
            await progress_callback({
                'stage': 'enhancing',
                'progress': 60,
                'message': 'Enhancing top moments...'
            })
        
        # Enhance only the best segments
        enhanced_result = await self.enhancer.enhance_highlights_only(
            smart_result,
            budget=0.50,  # Max $0.50 for enhancements
            video_path=video_path
        )
        
        # Render final video
        output_path = await self.renderer.render_enhanced(
            video_path,
            enhanced_result['segments']
        )
        
        return {
            'output_path': output_path,
            'segments': enhanced_result['segments'],
            'cost': enhanced_result['cost'],
            'quality_score': 0.9,
            'metadata': enhanced_result.get('metadata', {})
        }
    
    async def _process_selective_premium(self, video_path: str, probe: ProbeResult,
                                        job_id: str, progress_callback: Optional[callable]) -> Dict[str, Any]:
        """Use expensive tools only where absolutely needed"""
        logger.info("Processing in SELECTIVE_PREMIUM mode")
        
        # Identify complex sections that need premium processing
        complex_sections = self._identify_complex_sections(probe)
        
        if progress_callback:
            await progress_callback({
                'stage': 'analyzing',
                'progress': 30,
                'message': f'Analyzing {len(complex_sections)} complex sections...'
            })
        
        # Process different sections with appropriate tools
        all_segments = []
        total_cost = 0.0
        
        # Smart track for simple sections
        smart_segments = await self.smart_track.process_sections(
            video_path, 
            probe,
            exclude_ranges=complex_sections
        )
        all_segments.extend(smart_segments['segments'])
        
        # Premium processing for complex sections only
        if complex_sections and budget_guard.check_budget(job_id)[0]:
            premium_segments = await self.enhancer.process_complex_sections(
                video_path,
                complex_sections,
                probe
            )
            all_segments.extend(premium_segments['segments'])
            total_cost += premium_segments['cost']
        
        # Sort and merge all segments
        all_segments.sort(key=lambda x: x['start_time'])
        
        # Intelligent segment selection
        final_segments = self._select_best_segments(all_segments, probe)
        
        # Render with premium quality
        output_path = await self.renderer.render_premium(
            video_path,
            final_segments
        )
        
        return {
            'output_path': output_path,
            'segments': final_segments,
            'cost': total_cost,
            'quality_score': 0.95,
            'metadata': {
                'complex_sections': len(complex_sections),
                'processing_mode': 'selective_premium'
            }
        }
    
    def _identify_complex_sections(self, probe: ProbeResult) -> List[Tuple[float, float]]:
        """Identify sections that need premium processing"""
        complex_sections = []
        
        # Multi-speaker sections need diarization
        if probe.speaker_count > 2:
            # For now, assume whole video needs it
            # In real implementation, would analyze audio to find multi-speaker parts
            complex_sections.append((0, min(probe.duration, 600)))  # Max 10 min
        
        # Technical sections need GPT
        if probe.technical_density > 0.7:
            # Process middle third where main content usually is
            start = probe.duration * 0.3
            end = probe.duration * 0.7
            complex_sections.append((start, min(end, start + 300)))  # Max 5 min
        
        return complex_sections
    
    def _select_best_segments(self, segments: List[Dict], probe: ProbeResult) -> List[Dict]:
        """Select best segments based on scores and constraints"""
        # Sort by score
        segments.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        # Select top segments up to target duration
        target_duration = 60  # Default 60 seconds
        selected = []
        total_duration = 0
        
        for segment in segments:
            seg_duration = segment['end_time'] - segment['start_time']
            if total_duration + seg_duration <= target_duration * 1.2:  # Allow 20% over
                selected.append(segment)
                total_duration += seg_duration
            
            if total_duration >= target_duration:
                break
        
        # Sort by time for final video
        selected.sort(key=lambda x: x['start_time'])
        
        return selected
    
    def _create_job(self, video_path: str, constraints: UserConstraints) -> str:
        """Create job record in database"""
        job_data = {
            'input_path': video_path,
            'status': 'processing',
            'constraints': json.dumps({
                'max_budget': constraints.max_budget,
                'max_wait_time': constraints.max_wait_time,
                'allows_cloud': constraints.allows_cloud,
                'target_duration': constraints.target_duration
            }),
            'created_at': time.time()
        }
        
        return self.db.insert('video_job', job_data, returning='id')
    
    def _update_job_status(self, job_id: str, status: str, error: Optional[str] = None):
        """Update job status"""
        update_data = {
            'status': status,
            'updated_at': time.time()
        }
        
        if error:
            update_data['error_message'] = error
        
        if status == 'completed':
            update_data['completed_at'] = time.time()
        
        self.db.update('video_job', {'id': job_id}, update_data)
    
    def _record_metrics(self, job_id: str, result: ProcessingResult, processing_time: float):
        """Record processing metrics"""
        metrics.video_job_total.labels(status='completed').inc()
        metrics.video_processing_time_ratio.set(processing_time / 60)  # Assume 60s output
        
        if result.actual_cost > 0:
            metrics.api_cost_total.labels(api='selective').inc(result.actual_cost)
        
        # Update job record
        self._update_job_status(job_id, 'completed')


async def main():
    """Example usage showing the simplicity"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python adaptive_quality_pipeline.py <video_path>")
        return
    
    video_path = sys.argv[1]
    
    # Create pipeline
    pipeline = AdaptiveQualityPipeline()
    
    # Define progress callback
    async def progress_update(update: Dict[str, Any]):
        print(f"[{update['progress']}%] {update['message']}")
    
    # Process with ONE method call - no user choice needed!
    try:
        result = await pipeline.process(
            video_path,
            user_constraints=UserConstraints(
                max_budget=1.00,
                max_wait_time=300,
                allows_cloud=True
            ),
            progress_callback=progress_update
        )
        
        print(f"\n✅ Video processed successfully!")
        print(f"Mode used: {result.mode_used.value}")
        print(f"Cost: ${result.actual_cost:.2f}")
        print(f"Time: {result.processing_time:.1f}s")
        print(f"Quality: {result.quality_score:.1%}")
        print(f"Output: {result.video_path}")
        
    except Exception as e:
        print(f"\n❌ Processing failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())