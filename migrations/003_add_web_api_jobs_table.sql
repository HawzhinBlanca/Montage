-- Migration: Add jobs table for web API
-- Version: 3
-- Description: Adds jobs table for FastAPI/Celery job tracking

-- Create jobs table for web API (separate from video_job for now)
CREATE TABLE IF NOT EXISTS jobs (
    job_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    input_path TEXT NOT NULL,
    output_path TEXT,
    mode TEXT NOT NULL DEFAULT 'smart' CHECK (mode IN ('smart', 'premium')),
    vertical BOOLEAN NOT NULL DEFAULT FALSE,
    error_message TEXT,
    metrics JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    -- Link to main video_job table if needed
    video_job_id UUID REFERENCES video_job(id) ON DELETE SET NULL
);

-- Indexes for jobs table
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_jobs_job_id ON jobs(job_id);

-- Trigger to update updated_at
CREATE TRIGGER update_jobs_updated_at BEFORE UPDATE ON jobs 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments
COMMENT ON TABLE jobs IS 'Web API job tracking for async processing';
COMMENT ON COLUMN jobs.job_id IS 'Public job ID exposed to API clients';
COMMENT ON COLUMN jobs.video_job_id IS 'Link to internal video_job table';

-- Version tracking
INSERT INTO schema_version (version, description) 
VALUES (3, 'Add jobs table for web API')
ON CONFLICT (version) DO NOTHING;