-- Migration: Add user_id column to jobs table for authentication
-- Date: 2025-01-22

BEGIN;

-- Add user_id column to jobs table if it doesn't exist
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'jobs' AND column_name = 'user_id'
    ) THEN
        ALTER TABLE jobs ADD COLUMN user_id VARCHAR(100);
        
        -- Set default value for existing records
        UPDATE jobs SET user_id = 'system' WHERE user_id IS NULL;
        
        -- Make it not null after setting defaults
        ALTER TABLE jobs ALTER COLUMN user_id SET NOT NULL;
        
        -- Add index for performance
        CREATE INDEX idx_jobs_user_id ON jobs(user_id);
        
        -- Add composite index for common queries
        CREATE INDEX idx_jobs_user_status ON jobs(user_id, status);
        
        RAISE NOTICE 'Added user_id column to jobs table';
    ELSE
        RAISE NOTICE 'user_id column already exists in jobs table';
    END IF;
END $$;

COMMIT;