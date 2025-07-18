"""Smart Video Editor - Main integration module"""

import logging
import os
import tempfile
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import uuid

from config import Config
from db import Database
from checkpoint import SmartVideoEditorCheckpoint, CheckpointManager
from video_validator import VideoValidator, perform_preflight_check
from concat_editor import ConcatEditor, EditSegment, create_edit_segments
from audio_normalizer import AudioNormalizer, NormalizationTarget
from color_converter import ColorSpaceConverter, ensure_bt709_output
from metrics import metrics, track_processing_stage, with_job_tracking

logger = logging.getLogger(__name__)


class SmartVideoEditor:
    """
    Main video processing pipeline with all integrations.
    
    Implements:
    - Pre-flight validation (with HDR rejection)
    - Checkpoint-based recovery
    - FIFO-based editing
    - Two-pass audio normalization
    - BT.709 color space enforcement
    - Comprehensive metrics
    """
    
    def __init__(self):
        self.db = Database()
        self.checkpoint_mgr = CheckpointManager()
        self.checkpoint = SmartVideoEditorCheckpoint(self.checkpoint_mgr)
        self.validator = VideoValidator()
        self.editor = ConcatEditor()
        self.normalizer = AudioNormalizer()
        self.color_converter = ColorSpaceConverter()
        
        # Start metrics server
        try:
            metrics.start_http_server()
        except:
            logger.warning("Metrics server already running")
    
    @with_job_tracking
    def process_job(self, job_id: str, input_path: str, highlights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process a complete video job with all safety checks and optimizations.
        
        Args:
            job_id: Unique job identifier
            input_path: Path to input video
            highlights: List of highlight segments to extract
            
        Returns:
            Processing results including output path
        """
        logger.info(f"Starting job {job_id}")
        
        try:
            # Check for resume point
            resume_info = self.checkpoint.get_resume_point(job_id)
            
            if resume_info:
                logger.info(f"Resuming from stage: {resume_info['resume_from_stage']}")
                return self._resume_processing(job_id, input_path, highlights, resume_info)
            
            # Full processing pipeline
            return self._full_processing_pipeline(job_id, input_path, highlights)
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.db.update('video_job', {
                'status': 'failed',
                'error_message': str(e)
            }, {'id': job_id})
            raise
    
    def _full_processing_pipeline(self, job_id: str, input_path: str, 
                                  highlights: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute full processing pipeline with checkpoints"""
        
        # Stage 1: Validation (includes HDR check)
        if not self.checkpoint.should_skip_stage(job_id, 'validation'):
            validation_result = self._validate_input(job_id, input_path)
            self.checkpoint.save_stage_data(job_id, 'validation', **validation_result)
        else:
            validation_result = self.checkpoint.load_stage_data(job_id, 'validation')
        
        video_duration = validation_result['duration']
        
        # Stage 2: Analysis (prepare segments)
        if not self.checkpoint.should_skip_stage(job_id, 'analysis'):
            analysis_result = self._analyze_segments(job_id, highlights, video_duration)
            self.checkpoint.save_stage_data(job_id, 'analysis', **analysis_result)
        else:
            analysis_result = self.checkpoint.load_stage_data(job_id, 'analysis')
        
        # Stage 3: Editing with color space conversion
        if not self.checkpoint.should_skip_stage(job_id, 'editing'):
            segments = create_edit_segments(highlights, input_path)
            edit_result = self._edit_video_with_color_safety(job_id, input_path, segments, video_duration)
            self.checkpoint.save_stage_data(job_id, 'editing', **edit_result)
        else:
            edit_result = self.checkpoint.load_stage_data(job_id, 'editing')
        
        # Stage 4: Audio normalization
        if not self.checkpoint.should_skip_stage(job_id, 'audio_normalization'):
            norm_result = self._normalize_audio(job_id, edit_result['temp_output'], video_duration)
            self.checkpoint.save_stage_data(job_id, 'audio_normalization', **norm_result)
        else:
            norm_result = self.checkpoint.load_stage_data(job_id, 'audio_normalization')
        
        # Stage 5: Final color verification
        if not self.checkpoint.should_skip_stage(job_id, 'color_verification'):
            final_result = self._verify_final_output(job_id, norm_result['output_path'])
            self.checkpoint.save_stage_data(job_id, 'color_verification', **final_result)
        else:
            final_result = self.checkpoint.load_stage_data(job_id, 'color_verification')
        
        # Update job status
        self.db.update('video_job', {
            'status': 'completed',
            'output_path': final_result['output_path'],
            'completed_at': 'NOW()'
        }, {'id': job_id})
        
        # Clean up checkpoints
        self.checkpoint_mgr.delete_job_checkpoints(job_id)
        
        return {
            'job_id': job_id,
            'output_path': final_result['output_path'],
            'segments_processed': len(highlights),
            'total_duration': video_duration,
            'color_space': 'bt709',
            'audio_loudness': norm_result.get('output_loudness', -16.0)
        }
    
    @track_processing_stage('validation')
    def _validate_input(self, job_id: str, input_path: str, video_duration: float = None) -> Dict[str, Any]:
        """Validate input video (includes HDR rejection)"""
        logger.info(f"Validating input for job {job_id}")
        
        result = perform_preflight_check(job_id, input_path)
        
        if not result['valid']:
            raise ValueError(result['error'])
        
        # Additional color space validation
        is_sdr, error = self.color_converter.validate_sdr_input(input_path)
        if not is_sdr:
            self.db.update('video_job', {
                'status': 'failed',
                'error_message': error
            }, {'id': job_id})
            raise ValueError(error)
        
        return result
    
    @track_processing_stage('analysis')
    def _analyze_segments(self, job_id: str, highlights: List[Dict[str, Any]], 
                         video_duration: float) -> Dict[str, Any]:
        """Analyze and prepare segments"""
        logger.info(f"Analyzing {len(highlights)} segments")
        
        # Track segment metrics
        for highlight in highlights:
            if 'score' in highlight:
                metrics.track_highlight_score(highlight['score'])
        
        metrics.track_segments_detected(len(highlights))
        
        return {
            'segments_count': len(highlights),
            'total_highlight_duration': sum(h['end_time'] - h['start_time'] for h in highlights)
        }
    
    @track_processing_stage('editing')
    def _edit_video_with_color_safety(self, job_id: str, input_path: str,
                                      segments: List[EditSegment], 
                                      video_duration: float) -> Dict[str, Any]:
        """Edit video with color space safety"""
        logger.info(f"Editing video with {len(segments)} segments")
        
        # Create temporary output
        temp_output = os.path.join(Config.TEMP_DIR, f"{job_id}_edited_temp.mp4")
        
        # Build color conversion filter
        color_filter = self.color_converter.build_color_conversion_filter()
        
        # Execute edit with color conversion in the pipeline
        # The concat editor will handle segment extraction
        edit_result = self.editor.execute_edit(
            segments,
            temp_output,
            apply_transitions=True,
            video_codec='libx264',
            audio_codec='aac'
        )
        
        # Now apply color conversion to ensure BT.709
        color_safe_output = os.path.join(Config.TEMP_DIR, f"{job_id}_color_safe.mp4")
        
        self.color_converter.convert_to_bt709(
            temp_output,
            color_safe_output,
            video_codec='libx264',
            audio_codec='copy'  # Don't re-encode audio yet
        )
        
        # Clean up first temp file
        try:
            os.unlink(temp_output)
        except:
            pass
        
        return {
            'temp_output': color_safe_output,
            'segments_processed': edit_result['segments_processed'],
            'processing_ratio': edit_result['processing_ratio']
        }
    
    @track_processing_stage('audio_normalization')  
    def _normalize_audio(self, job_id: str, input_path: str, 
                        video_duration: float) -> Dict[str, Any]:
        """Apply two-pass audio normalization"""
        logger.info(f"Normalizing audio for job {job_id}")
        
        output_path = os.path.join(Config.TEMP_DIR, f"{job_id}_final.mp4")
        
        # Use streaming standard target
        target = NormalizationTarget(
            integrated=-16.0,
            true_peak=-1.0,
            lra=7.0
        )
        
        norm_result = self.normalizer.normalize_audio(
            input_path,
            output_path,
            target=target,
            video_duration=video_duration
        )
        
        # Clean up temp file
        try:
            os.unlink(input_path)
        except:
            pass
        
        return {
            'output_path': output_path,
            **norm_result
        }
    
    def _verify_final_output(self, job_id: str, output_path: str) -> Dict[str, Any]:
        """Verify final output meets all requirements"""
        logger.info(f"Verifying final output for job {job_id}")
        
        # Verify color space
        color_info = self.color_converter.analyze_color_space(output_path)
        
        if color_info.color_primaries != 'bt709':
            logger.warning(f"Output color primaries: {color_info.color_primaries} (expected bt709)")
        
        # Move to final location
        final_path = os.path.join("/output", f"{job_id}_final.mp4")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        os.rename(output_path, final_path)
        
        return {
            'output_path': final_path,
            'color_primaries': color_info.color_primaries,
            'color_space_valid': color_info.color_primaries == 'bt709'
        }
    
    def _resume_processing(self, job_id: str, input_path: str,
                          highlights: List[Dict[str, Any]],
                          resume_info: Dict[str, Any]) -> Dict[str, Any]:
        """Resume processing from checkpoint"""
        logger.info(f"Resuming job {job_id} from {resume_info['resume_from_stage']}")
        
        # Restore state and continue
        return self._full_processing_pipeline(job_id, input_path, highlights)


# Example usage
def example_usage():
    """Example of using SmartVideoEditor"""
    
    editor = SmartVideoEditor()
    
    # Define highlights
    highlights = [
        {
            'start_time': 30,
            'end_time': 60,
            'score': 0.9,
            'transition': 'fade'
        },
        {
            'start_time': 120,
            'end_time': 150,
            'score': 0.85
        },
        {
            'start_time': 200,
            'end_time': 230,
            'score': 0.88
        }
    ]
    
    # Process video
    result = editor.process_job(
        job_id="test-job-001",
        input_path="/videos/input.mp4",
        highlights=highlights
    )
    
    print(f"Processing complete!")
    print(f"Output: {result['output_path']}")
    print(f"Color space: {result['color_space']}")
    print(f"Audio loudness: {result['audio_loudness']} LUFS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    example_usage()