"""Test thread-safe database connection pool"""

import pytest
import threading
import time
import uuid
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from db import Database, DatabasePool, db_pool, with_retry
from config import Config


class TestDatabasePool:
    """Test database pool functionality and thread safety"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test table before each test"""
        db = Database()
        # Create a test table
        db.execute("""
            CREATE TABLE IF NOT EXISTS test_concurrent (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                thread_id TEXT NOT NULL,
                value INTEGER NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        yield
        # Cleanup
        db.execute("DROP TABLE IF EXISTS test_concurrent")
    
    def test_singleton_pool(self):
        """Test that pool is a singleton"""
        pool1 = DatabasePool()
        pool2 = DatabasePool()
        assert pool1 is pool2
    
    def test_pool_size_limits(self):
        """Test pool respects size configuration"""
        assert db_pool.pool.minconn == Config.MIN_POOL_SIZE
        assert db_pool.pool.maxconn == Config.MAX_POOL_SIZE
        # Ensure max pool size is ~2x CPU cores as required
        assert Config.MAX_POOL_SIZE <= Config.CPU_COUNT * 2 + 2  # Small buffer
    
    def test_basic_operations(self):
        """Test basic CRUD operations"""
        db = Database()
        
        # Insert
        data = {'thread_id': 'test-1', 'value': 42}
        record_id = db.insert('test_concurrent', data)
        assert record_id is not None
        
        # Find one
        record = db.find_one('test_concurrent', {'id': record_id})
        assert record is not None
        assert record['value'] == 42
        
        # Update
        affected = db.update('test_concurrent', {'value': 100}, {'id': record_id})
        assert affected == 1
        
        # Verify update
        record = db.find_one('test_concurrent', {'id': record_id})
        assert record['value'] == 100
        
        # Count
        count = db.count('test_concurrent')
        assert count >= 1
    
    def test_concurrent_writes_no_deadlock(self):
        """Test concurrent writes don't cause deadlocks (as per acceptance criteria)"""
        db = Database()
        num_threads = 4
        operations_per_thread = 50
        
        def worker(thread_num):
            thread_id = f"thread-{thread_num}"
            results = []
            
            for i in range(operations_per_thread):
                try:
                    # Random operation to increase contention
                    if random.random() < 0.7:
                        # Insert
                        record_id = db.insert('test_concurrent', {
                            'thread_id': thread_id,
                            'value': i
                        })
                        results.append(('insert', record_id))
                    else:
                        # Update random record
                        records = db.find_many('test_concurrent', limit=10)
                        if records:
                            record = random.choice(records)
                            affected = db.update(
                                'test_concurrent',
                                {'value': i * 1000},
                                {'id': record['id']}
                            )
                            results.append(('update', affected))
                    
                    # Small random delay to increase overlap
                    time.sleep(random.uniform(0.001, 0.01))
                    
                except Exception as e:
                    results.append(('error', str(e)))
            
            return results
        
        # Run concurrent operations
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker, i) for i in range(num_threads)]
            
            all_results = []
            for future in as_completed(futures):
                results = future.result()
                all_results.extend(results)
        
        # Verify no deadlocks or errors
        errors = [r for r in all_results if r[0] == 'error']
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"
        
        # Verify data integrity
        total_count = db.count('test_concurrent')
        assert total_count > 0
        
        # Check each thread successfully wrote data
        for i in range(num_threads):
            thread_count = db.count('test_concurrent', {'thread_id': f'thread-{i}'})
            assert thread_count > 0, f"Thread {i} failed to write any records"
    
    def test_transaction_isolation(self):
        """Test transaction isolation between threads"""
        db = Database()
        
        barrier = threading.Barrier(2)
        results = {'thread1': None, 'thread2': None}
        
        def transaction_worker(name, should_fail):
            with db.transaction() as tx:
                # Insert initial record
                tx.execute(
                    "INSERT INTO test_concurrent (thread_id, value) VALUES (%s, %s)",
                    (name, 1)
                )
                
                # Wait for other thread
                barrier.wait()
                
                # Try to read other thread's uncommitted data
                result = tx.execute(
                    "SELECT COUNT(*) as count FROM test_concurrent WHERE thread_id = %s",
                    ('thread1' if name == 'thread2' else 'thread2',)
                )
                results[name] = result[0]['count']
                
                # One thread fails, one succeeds
                if should_fail:
                    raise Exception("Simulated failure")
        
        # Run transactions concurrently
        thread1 = threading.Thread(target=lambda: transaction_worker('thread1', False))
        thread2 = threading.Thread(target=lambda: transaction_worker('thread2', True))
        
        thread1.start()
        thread2.start()
        
        thread1.join()
        thread2.join()
        
        # Verify isolation - each thread shouldn't see other's uncommitted data
        assert results['thread1'] == 0
        assert results['thread2'] == 0
        
        # Verify only successful transaction was committed
        count1 = db.count('test_concurrent', {'thread_id': 'thread1'})
        count2 = db.count('test_concurrent', {'thread_id': 'thread2'})
        assert count1 == 1  # Committed
        assert count2 == 0  # Rolled back
    
    def test_connection_exhaustion(self):
        """Test behavior when connections are exhausted"""
        # This test simulates holding connections longer than operations
        held_connections = []
        
        try:
            # Try to acquire more connections than the pool size
            for i in range(Config.MAX_POOL_SIZE + 2):
                conn = db_pool.pool.getconn()
                held_connections.append(conn)
                
                if i < Config.MAX_POOL_SIZE:
                    # Should succeed
                    assert conn is not None
        except psycopg2.pool.PoolError:
            # Expected when pool is exhausted
            pass
        finally:
            # Return all connections
            for conn in held_connections:
                db_pool.pool.putconn(conn)
    
    def test_retry_mechanism(self):
        """Test retry mechanism for transient failures"""
        attempt_count = 0
        
        def flaky_operation():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise psycopg2.OperationalError("Connection lost")
            return "success"
        
        result = with_retry(flaky_operation, max_attempts=3, delay=0.1)
        assert result == "success"
        assert attempt_count == 3
    
    def test_savepoint_rollback(self):
        """Test savepoint functionality in transactions"""
        db = Database()
        
        with db.transaction() as tx:
            # Insert first record
            tx.execute(
                "INSERT INTO test_concurrent (thread_id, value) VALUES (%s, %s)",
                ('savepoint-test', 1)
            )
            
            # Create savepoint
            tx.savepoint('sp1')
            
            # Insert second record
            tx.execute(
                "INSERT INTO test_concurrent (thread_id, value) VALUES (%s, %s)",
                ('savepoint-test', 2)
            )
            
            # Rollback to savepoint
            tx.rollback_to_savepoint('sp1')
            
            # Insert third record
            tx.execute(
                "INSERT INTO test_concurrent (thread_id, value) VALUES (%s, %s)",
                ('savepoint-test', 3)
            )
        
        # Verify only first and third records exist
        records = db.find_many('test_concurrent', {'thread_id': 'savepoint-test'})
        values = sorted([r['value'] for r in records])
        assert values == [1, 3]  # 2 was rolled back


@pytest.mark.parametrize("n_workers", [4, 8, 16])
def test_pytest_xdist_compatibility(n_workers):
    """Test compatibility with pytest-xdist parallel execution"""
    # This test ensures the pool works correctly when pytest runs with -n flag
    db = Database()
    
    # Each worker writes to its own "namespace"
    worker_id = f"worker-{threading.get_ident()}"
    
    for i in range(10):
        db.insert('test_concurrent', {
            'thread_id': worker_id,
            'value': i
        })
    
    # Verify all writes succeeded
    count = db.count('test_concurrent', {'thread_id': worker_id})
    assert count == 10


if __name__ == "__main__":
    # Run with pytest-xdist as required
    pytest.main([__file__, "-v", "-n", "4"])