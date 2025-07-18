# Database Setup Guide

## Prerequisites

1. PostgreSQL 14+ installed and running
2. Redis 6+ installed and running
3. Python 3.8+ installed

## Setup Steps

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Copy the example environment file and update with your credentials:

```bash
cp .env.example .env
```

Edit `.env` and set:
- `POSTGRES_PASSWORD` - Your PostgreSQL password
- `OPENAI_API_KEY` - Your OpenAI API key (for later phases)
- Other configuration as needed

### 3. Run Migrations

Execute the migration script to create the database and tables:

```bash
python migrate.py
```

This will:
- Create the database if it doesn't exist
- Create all required tables
- Set up indexes and foreign keys
- Track applied migrations

### 4. Verify Setup

Run the database tests to ensure everything is configured correctly:

```bash
pytest tests/test_database_setup.py -v
```

## Database Schema

The schema includes the following tables:

- **video_job** - Main job tracking table
- **transcript_cache** - Caches ASR results by file hash
- **highlight** - Stores detected highlights with scores
- **job_checkpoint** - Redis backup for crash recovery
- **api_cost_log** - Tracks API costs per job
- **processing_metrics** - Performance monitoring data
- **schema_migrations** - Tracks applied migrations

## Connection Pool Configuration

The connection pool size is automatically set to 2x CPU cores as per requirements. This can be overridden via:
- `MAX_POOL_SIZE` - Maximum connections (default: 2x CPU cores)
- `MIN_POOL_SIZE` - Minimum connections (default: 2)

## Troubleshooting

### Connection Refused
- Ensure PostgreSQL is running: `sudo systemctl status postgresql`
- Check connection settings in `.env`

### Permission Denied
- Ensure your PostgreSQL user has CREATE DATABASE privileges
- Grant if needed: `GRANT CREATE ON DATABASE postgres TO videouser;`

### Migration Failures
- Check PostgreSQL logs for detailed errors
- Ensure no conflicting schema exists
- Run `python migrate.py` with elevated logging