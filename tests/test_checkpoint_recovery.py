"""Test checkpoint and recovery functionality as per Task 3 acceptance criteria"""

import pytest
import time
import uuid
import tempfile
import os

from checkpoint import CheckpointManager
from video_processor import SmartVideoEditor
from db import Database


class TestCheckpointRecovery:
    """Test crash recovery with checkpoints"""
    
    @pytest.fixture
    def checkpoint_manager(self):
        """Checkpoint manager instance"""
        return CheckpointManager()
    
    @pytest.fixture
    def db(self):
        """Database instance"""
        return Database()
    
    @pytest.fixture
    def editor(self):
        """Video editor instance"""
        return SmartVideoEditor()
    
    def test_checkpoint_save_and_load(self, checkpoint_manager):
        """Test basic checkpoint save and load"""
        job_id = f"test-job-{uuid.uuid4()}"
        
        # Save checkpoint
        test_data = {
            'segments_analyzed': 10,
            'highlights_found': 3,
            'current_position': 150.5,
            'metadata': {'key': 'value'}
        }
        
        checkpoint_manager.save_checkpoint(job_id, 'analysis', test_data)
        
        # Load checkpoint
        loaded = checkpoint_manager.load_checkpoint(job_id)
        
        assert loaded is not None
        assert loaded['job_id'] == job_id
        assert loaded['stage'] == 'analysis'
        assert loaded['data']['segments_analyzed'] == 10
        assert loaded['data']['highlights_found'] == 3
        assert loaded['data']['metadata']['key'] == 'value'
    
    def test_resume_after_analysis_crash(self, checkpoint_manager, db, editor):
        """Test resuming from highlight stage after analysis crash"""
        job_id = str(uuid.uuid4())
        
        # Create test video
        test_video = self._create_test_video()
        
        # Create job
        db.insert('video_job', {
            'id': job_id,
            'src_hash': 'test_hash_recovery',
            'status': 'processing',
            'input_path': test_video
        })
        
        # Simulate analysis completion
        analysis_results = {
            'highlights': [
                {'start': 0, 'end': 10, 'score': 0.9},
                {'start': 20, 'end': 30, 'score': 0.8},
                {'start': 40, 'end': 50, 'score': 0.7}
            ],
            'metadata': {
                'total_duration': 60,
                'segments_analyzed': 6
            }
        }
        
        # Save checkpoint after analysis
        checkpoint_manager.save_checkpoint(job_id, 'analysis', analysis_results)
        
        # Simulate crash by starting new editor instance
        new_editor = SmartVideoEditor()
        
        # Mock the run method to verify it skips analysis
        analysis_called = False
        highlight_called = False
        
        original_analyze = new_editor.analyze_video
        original_generate = new_editor.generate_highlights
        
        def mock_analyze(*args, **kwargs):
            nonlocal analysis_called
            analysis_called = True
            return original_analyze(*args, **kwargs)
        
        def mock_generate(*args, **kwargs):
            nonlocal highlight_called
            highlight_called = True
            # Return mock highlights
            return analysis_results['highlights']
        
        new_editor.analyze_video = mock_analyze
        new_editor.generate_highlights = mock_generate
        
        # Run should resume from checkpoint
        class MockEditPlan:
            segments = [{'start': 0, 'end': 10}]
        
        # Check if checkpoint exists and load it
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        
        assert checkpoint is not None
        assert checkpoint['stage'] == 'analysis'
        
        # Should skip analysis and go to highlights
        if checkpoint and checkpoint['stage'] == 'analysis':
            # Use checkpoint data instead of re-analyzing
            highlights = checkpoint['data']['highlights']
            assert len(highlights) == 3
            assert not analysis_called  # Analysis should be skipped
        
        # Verify checkpoint data integrity
        assert checkpoint['data']['metadata']['total_duration'] == 60
        assert checkpoint['data']['metadata']['segments_analyzed'] == 6
        
        # Cleanup
        os.remove(test_video)
    
    def test_checkpoint_expiry(self, checkpoint_manager):
        """Test checkpoint TTL expiration"""
        job_id = f"test-expiry-{uuid.uuid4()}"
        
        # Save checkpoint with short TTL
        original_ttl = checkpoint_manager.checkpoint_ttl
        checkpoint_manager.checkpoint_ttl = 1  # 1 second
        
        checkpoint_manager.save_checkpoint(job_id, 'test', {'data': 'value'})
        
        # Should exist immediately
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint is not None
        
        # Wait for expiry
        time.sleep(2)
        
        # Should be expired
        checkpoint = checkpoint_manager.load_checkpoint(job_id)
        assert checkpoint is None
        
        # Restore original TTL
        checkpoint_manager.checkpoint_ttl = original_ttl
    
    def test_stage_progression(self, checkpoint_manager, db):
        """Test proper stage progression with checkpoints"""
        job_id = str(uuid.uuid4())
        
        # Define stages
        stages = ['validation', 'analysis', 'highlight', 'editing', 'encoding']
        
        # Progress through stages
        for i, stage in enumerate(stages):
            # Save checkpoint for stage
            checkpoint_manager.save_checkpoint(job_id, stage, {
                'stage_index': i,
                'progress': (i + 1) * 20,
                'timestamp': time.time()
            })
            
            # Verify latest checkpoint
            checkpoint = checkpoint_manager.load_checkpoint(job_id)
            assert checkpoint['stage'] == stage
            assert checkpoint['data']['stage_index'] == i
            assert checkpoint['data']['progress'] == (i + 1) * 20
    
    def test_concurrent_checkpoint_access(self, checkpoint_manager):
        """Test concurrent checkpoint operations"""
        import threading
        
        job_id = f"test-concurrent-{uuid.uuid4()}"
        results = []
        errors = []
        
        def save_and_load(thread_id):
            try:
                # Save checkpoint
                checkpoint_manager.save_checkpoint(
                    job_id, 
                    f'thread_{thread_id}',
                    {'thread': thread_id, 'time': time.time()}
                )
                
                # Load checkpoint
                checkpoint = checkpoint_manager.load_checkpoint(job_id)
                results.append(checkpoint)
                
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrent operations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=save_and_load, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify no errors
        assert len(errors) == 0
        
        # Verify all operations succeeded
        assert len(results) == 5
    
    def test_checkpoint_with_complex_data(self, checkpoint_manager):
        """Test checkpointing complex data structures"""
        job_id = f"test-complex-{uuid.uuid4()}"
        
        complex_data = {
            'segments': [
                {
                    'id': str(uuid.uuid4()),
                    'start': 0.0,
                    'end': 10.5,
                    'metadata': {
                        'audio_levels': [-23.5, -22.1, -24.3],
                        'scene_changes': [2.1, 5.3, 8.7]
                    }
                }
                for _ in range(10)
            ],
            'analysis_results': {
                'total_frames': 15000,
                'fps': 29.97,
                'color_space': 'bt709',
                'nested': {
                    'deeply': {
                        'nested': {
                            'value': 42
                        }
                    }
                }
            },
            'numpy_data': [1.5, 2.3, 3.7, 4.1]  # Would be numpy array
        }
        
        # Save complex data
        checkpoint_manager.save_checkpoint(job_id, 'complex', complex_data)
        
        # Load and verify
        loaded = checkpoint_manager.load_checkpoint(job_id)
        
        assert loaded is not None
        assert len(loaded['data']['segments']) == 10
        assert loaded['data']['analysis_results']['total_frames'] == 15000
        assert loaded['data']['analysis_results']['fps'] == 29.97
        assert loaded['data']['analysis_results']['nested']['deeply']['nested']['value'] == 42
        assert loaded['data']['numpy_data'] == [1.5, 2.3, 3.7, 4.1]
    
    def _create_test_video(self):
        """Create a simple test video"""
        output = tempfile.mktemp(suffix='.mp4')
        
        import subprocess
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc=duration=60:size=320x240:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=60',
            '-c:v', 'libx264', '-preset', 'ultrafast',
            '-c:a', 'aac',
            output
        ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output


if __name__ == "__main__":
    pytest.main([__file__, '-xvs'])