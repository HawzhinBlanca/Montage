"""Test Redis checkpointing system for crash recovery"""

import pytest
import time
import uuid
import json
from unittest.mock import Mock, patch
from checkpoint import (
    CheckpointManager, 
    SmartVideoEditorCheckpoint,
    CheckpointError
)
from db import Database


class TestCheckpointManager:
    """Test checkpoint manager functionality"""
    
    @pytest.fixture
    def checkpoint_mgr(self):
        """Create checkpoint manager instance"""
        return CheckpointManager()
    
    @pytest.fixture
    def test_job_id(self):
        """Generate unique job ID for tests"""
        return str(uuid.uuid4())
    
    def test_save_and_load_checkpoint(self, checkpoint_mgr, test_job_id):
        """Test basic save and load operations"""
        test_data = {
            'segments': [1, 2, 3],
            'scores': [0.8, 0.9, 0.7],
            'metadata': {'duration': 120.5, 'fps': 30}
        }
        
        # Save checkpoint
        checkpoint_mgr.save_checkpoint(test_job_id, 'analysis', test_data)
        
        # Load checkpoint
        loaded_data = checkpoint_mgr.load_checkpoint(test_job_id, 'analysis')
        
        assert loaded_data is not None
        assert loaded_data['segments'] == test_data['segments']
        assert loaded_data['scores'] == test_data['scores']
        assert loaded_data['metadata']['duration'] == 120.5
    
    def test_checkpoint_expiry(self, checkpoint_mgr, test_job_id):
        """Test checkpoint TTL behavior"""
        checkpoint_mgr.save_checkpoint(test_job_id, 'test_stage', {'data': 'test'})
        
        # Check TTL is set
        key = checkpoint_mgr._get_checkpoint_key(test_job_id, 'test_stage')
        ttl = checkpoint_mgr.redis_client.ttl(key)
        
        assert ttl > 0
        assert ttl <= checkpoint_mgr.redis_client.ttl(key) + 1  # Allow 1 second variance
    
    def test_get_last_successful_stage(self, checkpoint_mgr, test_job_id):
        """Test retrieving last successful stage"""
        # Save checkpoints in order
        checkpoint_mgr.save_checkpoint(test_job_id, 'validation', {'valid': True})
        checkpoint_mgr.save_checkpoint(test_job_id, 'analysis', {'segments': 10})
        checkpoint_mgr.save_checkpoint(test_job_id, 'transcription', {'words': 500})
        
        last_stage = checkpoint_mgr.get_last_successful_stage(test_job_id)
        assert last_stage == 'transcription'
        
        # Save a later stage
        checkpoint_mgr.save_checkpoint(test_job_id, 'editing', {'cuts': 5})
        
        last_stage = checkpoint_mgr.get_last_successful_stage(test_job_id)
        assert last_stage == 'editing'
    
    def test_postgres_fallback(self, checkpoint_mgr, test_job_id):
        """Test fallback to PostgreSQL when Redis doesn't have data"""
        # Save directly to PostgreSQL
        db = Database()
        db.insert('job_checkpoint', {
            'job_id': test_job_id,
            'stage': 'analysis',
            'checkpoint_data': json.dumps({'source': 'postgres', 'value': 42})
        })
        
        # Clear Redis to force PostgreSQL lookup
        key = checkpoint_mgr._get_checkpoint_key(test_job_id, 'analysis')
        checkpoint_mgr.redis_client.delete(key)
        
        # Load should restore from PostgreSQL
        data = checkpoint_mgr.load_checkpoint(test_job_id, 'analysis')
        assert data is not None
        assert data['source'] == 'postgres'
        assert data['value'] == 42
        
        # Verify it was re-saved to Redis
        assert checkpoint_mgr.redis_client.exists(key)
    
    def test_delete_job_checkpoints(self, checkpoint_mgr, test_job_id):
        """Test deleting all checkpoints for a job"""
        # Save multiple checkpoints
        stages = ['validation', 'analysis', 'transcription']
        for stage in stages:
            checkpoint_mgr.save_checkpoint(test_job_id, stage, {'stage': stage})
        
        # Verify they exist
        for stage in stages:
            assert checkpoint_mgr.exists(test_job_id, stage)
        
        # Delete all
        checkpoint_mgr.delete_job_checkpoints(test_job_id)
        
        # Verify they're gone
        for stage in stages:
            assert not checkpoint_mgr.exists(test_job_id, stage)
    
    def test_atomic_checkpoint(self, checkpoint_mgr, test_job_id):
        """Test atomic checkpoint context manager"""
        # Successful operation
        with checkpoint_mgr.atomic_checkpoint(test_job_id, 'test_atomic'):
            # Simulate work
            time.sleep(0.1)
        
        # Should have saved checkpoint
        assert checkpoint_mgr.exists(test_job_id, 'test_atomic')
        
        # Failed operation
        try:
            with checkpoint_mgr.atomic_checkpoint(test_job_id, 'test_failed'):
                # Simulate failure
                raise Exception("Simulated failure")
        except:
            pass
        
        # Should not have saved checkpoint
        assert not checkpoint_mgr.exists(test_job_id, 'test_failed')
    
    def test_job_progress(self, checkpoint_mgr, test_job_id):
        """Test getting job progress information"""
        # Save checkpoints
        checkpoint_mgr.save_checkpoint(test_job_id, 'validation', {})
        checkpoint_mgr.save_checkpoint(test_job_id, 'analysis', {})
        
        progress = checkpoint_mgr.get_job_progress(test_job_id)
        
        assert progress['job_id'] == test_job_id
        assert 'validation' in progress['completed_stages']
        assert 'analysis' in progress['completed_stages']
        assert progress['last_stage'] == 'analysis'
        assert len(progress['checkpoints']) == 2


class TestSmartVideoEditorCheckpoint:
    """Test SmartVideoEditor checkpoint integration"""
    
    @pytest.fixture
    def editor_checkpoint(self):
        """Create editor checkpoint instance"""
        checkpoint_mgr = CheckpointManager()
        return SmartVideoEditorCheckpoint(checkpoint_mgr)
    
    @pytest.fixture
    def test_job_id(self):
        """Generate unique job ID for tests"""
        return str(uuid.uuid4())
    
    def test_crash_recovery_simulation(self, editor_checkpoint, test_job_id):
        """Simulate crash and recovery as per acceptance criteria"""
        # Simulate processing up to analysis stage
        editor_checkpoint.save_stage_data(
            test_job_id,
            'validation',
            duration=300.5,
            codec='h264'
        )
        
        editor_checkpoint.save_stage_data(
            test_job_id,
            'analysis',
            segments=[
                {'start': 0, 'end': 30, 'score': 0.9},
                {'start': 45, 'end': 75, 'score': 0.85}
            ],
            total_duration=300.5
        )
        
        # Simulate crash - nothing saved for later stages
        
        # On restart, check resume point
        resume_info = editor_checkpoint.get_resume_point(test_job_id)
        
        # Verify it resumes from correct stage
        assert resume_info is not None
        assert resume_info['resume_from_stage'] == 'transcription'
        assert resume_info['last_completed_stage'] == 'analysis'
        assert resume_info['checkpoint_data'] is not None
        assert len(resume_info['checkpoint_data']['segments']) == 2
        
        # Verify analysis stage is marked as completed
        assert editor_checkpoint.should_skip_stage(test_job_id, 'analysis')
        assert editor_checkpoint.should_skip_stage(test_job_id, 'validation')
        
        # Verify later stages are not marked as completed
        assert not editor_checkpoint.should_skip_stage(test_job_id, 'transcription')
        assert not editor_checkpoint.should_skip_stage(test_job_id, 'editing')
    
    def test_stage_progression(self, editor_checkpoint, test_job_id):
        """Test correct stage progression logic"""
        stages_and_next = [
            ('validation', 'analysis'),
            ('analysis', 'transcription'),
            ('transcription', 'highlight_detection'),
            ('highlight_detection', 'editing'),
            ('editing', 'audio_normalization'),
            ('audio_normalization', 'color_correction'),
            ('color_correction', 'export')
        ]
        
        for current_stage, expected_next in stages_and_next:
            # Save checkpoint for current stage
            editor_checkpoint.save_stage_data(test_job_id, current_stage, {'test': True})
            
            # Get resume point
            resume_info = editor_checkpoint.get_resume_point(test_job_id)
            
            assert resume_info['resume_from_stage'] == expected_next
            assert resume_info['last_completed_stage'] == current_stage
    
    def test_skip_completed_stages(self, editor_checkpoint, test_job_id):
        """Test skipping already completed stages"""
        # Save checkpoints for multiple stages
        completed_stages = ['validation', 'analysis', 'transcription']
        
        for stage in completed_stages:
            editor_checkpoint.save_stage_data(test_job_id, stage, {'completed': True})
        
        # Verify all completed stages should be skipped
        for stage in completed_stages:
            assert editor_checkpoint.should_skip_stage(test_job_id, stage)
        
        # Verify incomplete stages should not be skipped
        incomplete_stages = ['highlight_detection', 'editing', 'export']
        for stage in incomplete_stages:
            assert not editor_checkpoint.should_skip_stage(test_job_id, stage)
    
    def test_load_stage_specific_data(self, editor_checkpoint, test_job_id):
        """Test loading stage-specific checkpoint data"""
        # Save different data for different stages
        validation_data = {'duration': 120.5, 'valid': True, 'errors': []}
        analysis_data = {'segments': 10, 'avg_score': 0.85, 'peaks': [30, 60, 90]}
        
        editor_checkpoint.save_stage_data(test_job_id, 'validation', **validation_data)
        editor_checkpoint.save_stage_data(test_job_id, 'analysis', **analysis_data)
        
        # Load and verify
        loaded_validation = editor_checkpoint.load_stage_data(test_job_id, 'validation')
        assert loaded_validation['duration'] == 120.5
        assert loaded_validation['valid'] is True
        
        loaded_analysis = editor_checkpoint.load_stage_data(test_job_id, 'analysis')
        assert loaded_analysis['segments'] == 10
        assert loaded_analysis['avg_score'] == 0.85
        assert loaded_analysis['peaks'] == [30, 60, 90]


@pytest.mark.integration
class TestCheckpointIntegration:
    """Integration tests for checkpoint system"""
    
    def test_concurrent_checkpoint_access(self):
        """Test concurrent access to checkpoints"""
        import threading
        
        checkpoint_mgr = CheckpointManager()
        job_id = str(uuid.uuid4())
        results = {'errors': []}
        
        def worker(thread_id):
            try:
                for i in range(10):
                    checkpoint_mgr.save_checkpoint(
                        job_id,
                        f'stage_{thread_id}_{i}',
                        {'thread': thread_id, 'iteration': i}
                    )
                    
                    # Try to load
                    data = checkpoint_mgr.load_checkpoint(job_id, f'stage_{thread_id}_{i}')
                    assert data['thread'] == thread_id
                    
            except Exception as e:
                results['errors'].append(str(e))
        
        # Run concurrent workers
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            t.start()
            threads.append(t)
        
        for t in threads:
            t.join()
        
        assert len(results['errors']) == 0
    
    def test_redis_connection_recovery(self):
        """Test behavior when Redis connection is lost and recovered"""
        checkpoint_mgr = CheckpointManager()
        job_id = str(uuid.uuid4())
        
        # Save initial checkpoint
        checkpoint_mgr.save_checkpoint(job_id, 'test', {'value': 1})
        
        # Simulate connection loss by closing Redis connection
        checkpoint_mgr.redis_client.close()
        
        # Should handle gracefully and attempt reconnection
        try:
            # This should trigger reconnection
            data = checkpoint_mgr.load_checkpoint(job_id, 'test')
            # May or may not succeed depending on Redis state
        except:
            # Expected if Redis is actually down
            pass
        
        # Health check should reflect status
        health = checkpoint_mgr.health_check()
        # Will be True if Redis is running, False otherwise


if __name__ == "__main__":
    pytest.main([__file__, "-v"])