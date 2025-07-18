-- AI Video Processing Pipeline Database Schema
-- PostgreSQL with UUID support

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main job tracking table
CREATE TABLE IF NOT EXISTS video_job (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    options JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_video_job_status ON video_job(status);
CREATE INDEX idx_video_job_created ON video_job(created_at);

-- Job checkpoints for crash recovery
CREATE TABLE IF NOT EXISTS job_checkpoint (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(job_id, stage)
);

CREATE INDEX idx_checkpoint_job ON job_checkpoint(job_id);

-- API cost tracking
CREATE TABLE IF NOT EXISTS api_cost_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES video_job(id) ON DELETE CASCADE,
    api_name VARCHAR(100) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    cost DECIMAL(10, 4) NOT NULL,
    tokens INTEGER,
    duration_seconds DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_cost_job ON api_cost_log(job_id);
CREATE INDEX idx_cost_created ON api_cost_log(created_at);

-- Transcript cache
CREATE TABLE IF NOT EXISTS transcript_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    video_hash VARCHAR(64) NOT NULL UNIQUE,
    transcript JSONB NOT NULL,
    model VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE INDEX idx_transcript_hash ON transcript_cache(video_hash);
CREATE INDEX idx_transcript_expires ON transcript_cache(expires_at);

-- Video highlights
CREATE TABLE IF NOT EXISTS highlight (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
    start_time DECIMAL(10, 3) NOT NULL,
    end_time DECIMAL(10, 3) NOT NULL,
    score DECIMAL(5, 4) NOT NULL,
    type VARCHAR(50) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_highlight_job ON highlight(job_id);
CREATE INDEX idx_highlight_score ON highlight(score DESC);

-- Alert log for monitoring
CREATE TABLE IF NOT EXISTS alert_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    message TEXT NOT NULL,
    context JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(100)
);

CREATE INDEX idx_alert_type ON alert_log(alert_type);
CREATE INDEX idx_alert_severity ON alert_log(severity);
CREATE INDEX idx_alert_created ON alert_log(created_at);
CREATE INDEX idx_alert_ack ON alert_log(acknowledged);

-- Additional tables for complete functionality
CREATE TABLE IF NOT EXISTS job (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(50) NOT NULL DEFAULT 'pending',
    input_path TEXT NOT NULL,
    output_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS checkpoint (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL,
    stage VARCHAR(50) NOT NULL,
    data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cost_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID,
    api VARCHAR(100) NOT NULL,
    cost DECIMAL(10, 4) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create update trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_video_job_updated_at BEFORE UPDATE
    ON video_job FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();