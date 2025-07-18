"""Test concurrent database operations as per Task 2 acceptance criteria"""

import pytest
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from db import Database


class TestConcurrentDatabase:
    """Test database concurrency with pytest-xdist"""
    
    @pytest.fixture
    def db(self):
        """Database instance for testing"""
        return Database()
    
    def test_concurrent_writes_no_deadlock(self, db):
        """Test that concurrent writes don't cause deadlocks"""
        num_threads = 4
        writes_per_thread = 10
        results = []
        errors = []
        
        def write_job(thread_id):
            """Write jobs from a thread"""
            thread_results = []
            try:
                for i in range(writes_per_thread):
                    job_id = str(uuid.uuid4())
                    src_hash = f"hash_{thread_id}_{i}_{time.time()}"
                    
                    # Insert job
                    db.insert('video_job', {
                        'id': job_id,
                        'src_hash': src_hash,
                        'status': 'processing',
                        'duration': 100.0 + i
                    })
                    
                    # Update job
                    db.update('video_job', 
                             {'id': job_id},
                             {'status': 'completed'})
                    
                    # Read job
                    job = db.find_one('video_job', {'id': job_id})
                    thread_results.append(job)
                    
                    # Small delay to increase contention
                    time.sleep(0.01)
                    
            except Exception as e:
                errors.append(f"Thread {thread_id}: {str(e)}")
                
            return thread_results
        
        # Run concurrent writes
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for thread_id in range(num_threads):
                future = executor.submit(write_job, thread_id)
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                result = future.result()
                results.extend(result)
        
        # Verify no errors
        assert len(errors) == 0, f"Errors occurred: {errors}"
        
        # Verify all writes succeeded
        expected_total = num_threads * writes_per_thread
        assert len(results) == expected_total
        
        # Verify all jobs are completed
        for job in results:
            assert job['status'] == 'completed'
    
    def test_connection_pool_limits(self, db):
        """Test that connection pool respects size limits"""
        # Get pool info
        pool = db.pool
        
        # Check pool configuration
        assert pool.minconn >= 1
        assert pool.maxconn <= 32  # Should be ~2x CPU cores
        
        # Try to acquire many connections
        connections = []
        try:
            for i in range(pool.maxconn + 5):
                conn = pool.getconn()
                connections.append(conn)
        except Exception as e:
            # Should fail when pool is exhausted
            assert "connection pool exhausted" in str(e).lower()
        finally:
            # Return connections
            for conn in connections:
                pool.putconn(conn)
    
    def test_transaction_isolation(self, db):
        """Test transaction isolation between threads"""
        job_id = str(uuid.uuid4())
        src_hash = f"isolation_test_{time.time()}"
        
        # Create initial job
        db.insert('video_job', {
            'id': job_id,
            'src_hash': src_hash,
            'status': 'pending',
            'total_cost': 0
        })
        
        def update_in_transaction(amount):
            """Update cost in a transaction"""
            with db.transaction() as conn:
                cursor = conn.cursor()
                
                # Read current value
                cursor.execute(
                    "SELECT total_cost FROM video_job WHERE id = %s FOR UPDATE",
                    (job_id,)
                )
                current = cursor.fetchone()['total_cost']
                
                # Simulate processing
                time.sleep(0.1)
                
                # Update value
                new_value = float(current or 0) + amount
                cursor.execute(
                    "UPDATE video_job SET total_cost = %s WHERE id = %s",
                    (new_value, job_id)
                )
        
        # Run concurrent updates
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(update_in_transaction, 1.0),
                executor.submit(update_in_transaction, 2.0),
                executor.submit(update_in_transaction, 3.0)
            ]
            
            for future in as_completed(futures):
                future.result()
        
        # Verify final value
        job = db.find_one('video_job', {'id': job_id})
        assert job['total_cost'] == 6.0  # 1 + 2 + 3
    
    @pytest.mark.parametrize("num_workers", [2, 4, 8])
    def test_concurrent_checkpoint_updates(self, db, num_workers):
        """Test concurrent checkpoint updates don't conflict"""
        job_id = str(uuid.uuid4())
        
        # Create job
        db.insert('video_job', {
            'id': job_id,
            'src_hash': f"checkpoint_test_{time.time()}",
            'status': 'processing'
        })
        
        def save_checkpoint(stage_num):
            """Save a checkpoint for a stage"""
            stage = f"stage_{stage_num}"
            
            # Save checkpoint
            db.upsert('job_checkpoint', 
                     {'job_id': job_id, 'stage': stage},
                     {
                         'job_id': job_id,
                         'stage': stage,
                         'checkpoint_data': {
                             'progress': stage_num * 10,
                             'timestamp': time.time()
                         }
                     })
            
            return stage
        
        # Run concurrent checkpoint saves
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            
            for i in range(num_workers):
                future = executor.submit(save_checkpoint, i)
                futures.append(future)
            
            stages = []
            for future in as_completed(futures):
                stage = future.result()
                stages.append(stage)
        
        # Verify all checkpoints saved
        checkpoints = db.find('job_checkpoint', {'job_id': job_id})
        assert len(checkpoints) == num_workers
        
        # Verify data integrity
        for checkpoint in checkpoints:
            stage_num = int(checkpoint['stage'].split('_')[1])
            expected_progress = stage_num * 10
            actual_progress = checkpoint['checkpoint_data']['progress']
            assert actual_progress == expected_progress


if __name__ == "__main__":
    # Run with: pytest -xvs tests/test_concurrent_db.py -n 4
    pytest.main([__file__, '-xvs', '-n', '4'])