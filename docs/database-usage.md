# Database Module Usage Guide

## Overview

The `db.py` module provides a thread-safe connection pool and high-level database operations for the video processing pipeline. It implements:

- Thread-safe connection pooling with automatic sizing (2x CPU cores)
- Automatic connection management and cleanup
- Transaction support with savepoints
- Retry mechanism for transient failures
- High-level CRUD operations

## Basic Usage

### Simple Operations

```python
from db import Database

db = Database()

# Insert a record
job_id = db.insert('video_job', {
    'src_hash': 'abc123...',
    'status': 'queued',
    'input_path': '/path/to/video.mp4'
})

# Find a record
job = db.find_one('video_job', {'id': job_id})

# Update a record
db.update('video_job', {'status': 'processing'}, {'id': job_id})

# Find multiple records
pending_jobs = db.find_many('video_job', 
    where={'status': 'queued'}, 
    order_by='created_at', 
    limit=10
)

# Count records
queue_size = db.count('video_job', {'status': 'queued'})
```

### Transactions

```python
from db import Database

db = Database()

# Using transaction context manager
with db.transaction() as tx:
    # All operations in the transaction
    job_id = tx.execute(
        "INSERT INTO video_job (src_hash, status) VALUES (%s, %s) RETURNING id",
        ('hash123', 'queued')
    )[0]['id']
    
    # Create savepoint
    tx.savepoint('before_highlights')
    
    # Insert highlights
    for highlight in highlights:
        tx.execute(
            "INSERT INTO highlight (job_id, start_time, end_time, score) VALUES (%s, %s, %s, %s)",
            (job_id, highlight['start'], highlight['end'], highlight['score'])
        )
    
    # Rollback to savepoint if needed
    if error_condition:
        tx.rollback_to_savepoint('before_highlights')
```

### Raw Queries

```python
from db import Database

db = Database()

# Execute raw query
results = db.execute("""
    SELECT j.*, COUNT(h.id) as highlight_count
    FROM video_job j
    LEFT JOIN highlight h ON j.id = h.job_id
    WHERE j.status = %s
    GROUP BY j.id
    ORDER BY j.created_at DESC
    LIMIT %s
""", ('completed', 10))
```

### Connection Pool Direct Access

```python
from db import db_pool

# Get a connection directly from pool
with db_pool.get_connection() as conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_job")
    results = cursor.fetchall()
    cursor.close()

# Get a cursor with automatic cleanup
with db_pool.get_cursor() as cursor:
    cursor.execute("UPDATE video_job SET status = %s WHERE id = %s", ('failed', job_id))
```

### Retry Mechanism

```python
from db import with_retry, Database

db = Database()

# Retry a flaky operation
def update_with_external_check():
    # Check external service
    if not external_service.is_ready():
        raise psycopg2.OperationalError("Service not ready")
    
    return db.update('video_job', {'status': 'completed'}, {'id': job_id})

result = with_retry(update_with_external_check, max_attempts=3, delay=1.0)
```

## Thread Safety

The connection pool is fully thread-safe and tested with concurrent operations:

```python
import threading
from db import Database

db = Database()

def worker(thread_id):
    for i in range(100):
        db.insert('processing_metrics', {
            'job_id': job_id,
            'stage': f'thread_{thread_id}',
            'duration_ms': i * 100
        })

# Safe to use from multiple threads
threads = []
for i in range(4):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()
```

## Best Practices

1. **Use the high-level Database class** for most operations
2. **Use transactions** for multi-step operations that must be atomic
3. **Use savepoints** for partial rollbacks within large transactions
4. **Let the pool manage connections** - don't hold connections manually
5. **Use parameterized queries** to prevent SQL injection
6. **Handle exceptions** appropriately:

```python
from db import Database, DatabaseError

db = Database()

try:
    result = db.insert('video_job', data)
except DatabaseError as e:
    logger.error(f"Database operation failed: {e}")
    # Handle appropriately
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Handle appropriately
```

## Testing

Run the concurrent write tests to verify thread safety:

```bash
# Run with pytest-xdist as required by acceptance criteria
pytest tests/test_db_pool.py -v -n 4
```

## Performance Considerations

- Pool size is automatically set to 2x CPU cores
- Connections are reused to minimize overhead
- Use `execute_many()` for bulk inserts
- Use transactions for multiple related operations
- The pool handles connection recovery automatically