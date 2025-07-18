# Checkpoint System Usage Guide

## Overview

The checkpoint system provides crash recovery and job resumption capabilities for the video processing pipeline. It uses Redis for fast access with PostgreSQL backup, ensuring jobs can resume from the last successful stage after interruption.

## Key Features

- **Automatic Stage Tracking**: Tracks completion of each processing stage
- **Crash Recovery**: Resume processing from last successful stage
- **Data Persistence**: Dual storage in Redis (fast) and PostgreSQL (durable)
- **TTL Management**: Automatic cleanup of old checkpoints
- **Atomic Operations**: Ensures checkpoint consistency

## Basic Usage

### Setting Up Checkpoints

```python
from checkpoint import CheckpointManager, SmartVideoEditorCheckpoint

# Initialize checkpoint system
checkpoint_mgr = CheckpointManager()
editor_checkpoint = SmartVideoEditorCheckpoint(checkpoint_mgr)
```

### Saving Checkpoints

```python
# After completing validation stage
editor_checkpoint.save_stage_data(
    job_id='job-123',
    stage='validation',
    duration=120.5,
    codec='h264',
    resolution='1920x1080',
    fps=30,
    valid=True
)

# After analysis stage
editor_checkpoint.save_stage_data(
    job_id='job-123',
    stage='analysis',
    segments_found=15,
    highlights=[
        {'start': 30, 'end': 60, 'score': 0.9},
        {'start': 120, 'end': 180, 'score': 0.85}
    ],
    processing_time=45.2
)
```

### Resuming from Checkpoint

```python
# On startup, check if job should resume
resume_info = editor_checkpoint.get_resume_point('job-123')

if resume_info:
    print(f"Resuming from stage: {resume_info['resume_from_stage']}")
    print(f"Last completed: {resume_info['last_completed_stage']}")
    
    # Load previous stage data
    previous_data = resume_info['checkpoint_data']
    # Use previous_data to skip reprocessing
```

### Checking Stage Completion

```python
# Check if a stage was already completed
if editor_checkpoint.should_skip_stage('job-123', 'analysis'):
    print("Analysis already completed, loading results...")
    data = editor_checkpoint.load_stage_data('job-123', 'analysis')
    segments = data['segments_found']
else:
    # Run analysis
    segments = perform_analysis()
    # Save checkpoint
    editor_checkpoint.save_stage_data('job-123', 'analysis', segments_found=segments)
```

## SmartVideoEditor Integration Example

```python
class SmartVideoEditor:
    def __init__(self):
        self.checkpoint_mgr = CheckpointManager()
        self.checkpoint = SmartVideoEditorCheckpoint(self.checkpoint_mgr)
        self.db = Database()
    
    def run(self, job_id: str, input_path: str):
        """Main processing pipeline with checkpoint support"""
        
        # Check for resume point
        resume_info = self.checkpoint.get_resume_point(job_id)
        start_stage = 'validation'
        
        if resume_info:
            start_stage = resume_info['resume_from_stage']
            logger.info(f"Resuming job {job_id} from stage: {start_stage}")
            
            # Load any necessary data from previous stages
            if resume_info['checkpoint_data']:
                self.restore_state(resume_info['checkpoint_data'])
        
        # Process stages
        stages = {
            'validation': self.validate_input,
            'analysis': self.analyze_video,
            'transcription': self.transcribe_audio,
            'highlight_detection': self.detect_highlights,
            'editing': self.edit_video,
            'audio_normalization': self.normalize_audio,
            'color_correction': self.correct_colors,
            'export': self.export_final
        }
        
        # Start from appropriate stage
        stage_list = list(stages.keys())
        start_index = stage_list.index(start_stage)
        
        for stage in stage_list[start_index:]:
            if self.checkpoint.should_skip_stage(job_id, stage):
                logger.info(f"Skipping completed stage: {stage}")
                continue
            
            try:
                logger.info(f"Processing stage: {stage}")
                result = stages[stage](job_id, input_path)
                
                # Save checkpoint after successful completion
                self.checkpoint.save_stage_data(job_id, stage, **result)
                
                # Update job status in database
                self.db.update('video_job', 
                    {'status': f'completed_{stage}'}, 
                    {'id': job_id}
                )
                
            except Exception as e:
                logger.error(f"Failed at stage {stage}: {e}")
                self.db.update('video_job', {
                    'status': 'failed',
                    'error_message': str(e)
                }, {'id': job_id})
                raise
    
    def validate_input(self, job_id: str, input_path: str) -> dict:
        """Validate input video"""
        # Implementation...
        return {'valid': True, 'duration': 120.5, 'codec': 'h264'}
    
    def analyze_video(self, job_id: str, input_path: str) -> dict:
        """Analyze video for interesting segments"""
        # Can load validation data if needed
        validation_data = self.checkpoint.load_stage_data(job_id, 'validation')
        duration = validation_data['duration']
        
        # Implementation...
        return {'segments_found': 10, 'total_score': 0.85}
```

## Stage Progression

The system follows this stage order:

1. `validation` → Verify input file integrity
2. `analysis` → Analyze video content
3. `transcription` → Generate transcript
4. `highlight_detection` → Find interesting segments
5. `editing` → Create edited video
6. `audio_normalization` → Normalize audio levels
7. `color_correction` → Apply color corrections
8. `export` → Generate final output

## Atomic Operations

For critical operations that must complete fully:

```python
with checkpoint_mgr.atomic_checkpoint(job_id, 'critical_stage'):
    # Perform operations that must all succeed
    process_step_1()
    process_step_2()
    process_step_3()
    # Checkpoint only saved if all steps complete
```

## Monitoring Progress

```python
# Get detailed progress information
progress = checkpoint_mgr.get_job_progress(job_id)

print(f"Job: {progress['job_id']}")
print(f"Completed stages: {progress['completed_stages']}")
print(f"Last stage: {progress['last_stage']}")

for checkpoint in progress['checkpoints']:
    print(f"  {checkpoint['stage']}: expires at {checkpoint['expires_at']}")
```

## Best Practices

1. **Save After Each Major Stage**: Don't wait too long between checkpoints
2. **Include Sufficient Data**: Save enough data to resume processing
3. **Use Atomic Checkpoints**: For multi-step operations that must complete together
4. **Handle Missing Checkpoints**: Always check if checkpoint exists before loading
5. **Clean Up**: Delete checkpoints after job completion

```python
# After job completion
checkpoint_mgr.delete_job_checkpoints(job_id)
```

## Testing Crash Recovery

To test the crash recovery functionality:

```bash
# Run the checkpoint tests
pytest tests/test_checkpoint.py -v

# Specific crash recovery test
pytest tests/test_checkpoint.py::TestSmartVideoEditorCheckpoint::test_crash_recovery_simulation -v
```

## Configuration

Checkpoint behavior is configured via environment variables:

- `CHECKPOINT_TTL`: How long checkpoints persist (default: 86400 seconds / 24 hours)
- `REDIS_HOST`, `REDIS_PORT`: Redis connection settings
- `REDIS_DB`: Redis database number (default: 0)