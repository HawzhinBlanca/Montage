-- Complete database schema for video processing pipeline
-- Used by test fixtures to set up clean database state

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Video jobs table - central tracking of all processing jobs
CREATE TABLE video_job (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    src_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 of source file
    status VARCHAR(20) NOT NULL DEFAULT 'queued', -- queued, processing, completed, failed, cancelled
    input_path TEXT NOT NULL,
    output_path TEXT,
    metadata JSONB DEFAULT '{}',
    
    -- Video properties (populated after validation)
    duration FLOAT,
    codec VARCHAR(50),
    color_space VARCHAR(50),
    width INTEGER,
    height INTEGER,
    fps FLOAT,
    bitrate INTEGER,
    
    -- Processing tracking
    stage VARCHAR(50), -- validation, analysis, highlights, editing, encoding, finalization
    progress INTEGER DEFAULT 0, -- 0-100
    error_message TEXT,
    
    -- Cost tracking
    total_cost DECIMAL(10,4) DEFAULT 0.00,
    budget_limit DECIMAL(10,4) DEFAULT 5.00,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Constraints
    CONSTRAINT valid_status CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
    CONSTRAINT valid_progress CHECK (progress >= 0 AND progress <= 100),
    CONSTRAINT valid_cost CHECK (total_cost >= 0 AND budget_limit > 0)
);

-- Indexes for video_job
CREATE INDEX idx_video_job_status ON video_job(status);
CREATE INDEX idx_video_job_created_at ON video_job(created_at);
CREATE INDEX idx_video_job_src_hash ON video_job(src_hash);
CREATE INDEX idx_video_job_stage ON video_job(stage);

-- Video segments table - stores analyzed/highlighted segments
CREATE TABLE video_segment (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
    
    -- Timing
    start_time FLOAT NOT NULL,
    end_time FLOAT NOT NULL,
    duration FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
    
    -- Scoring and ranking
    score FLOAT NOT NULL DEFAULT 0.0,
    confidence FLOAT NOT NULL DEFAULT 0.0,
    rank INTEGER, -- 1 = highest scoring segment
    
    -- Content analysis
    transcript TEXT,
    summary TEXT,
    key_points JSONB DEFAULT '[]',
    emotions JSONB DEFAULT '{}',
    technical_concepts JSONB DEFAULT '[]',
    
    -- Speaker information
    speaker_count INTEGER DEFAULT 1,
    speakers JSONB DEFAULT '[]',
    speaker_changes INTEGER DEFAULT 0,
    
    -- Motion and visual analysis
    motion_intensity FLOAT DEFAULT 0.0,
    scene_changes INTEGER DEFAULT 0,
    face_detections JSONB DEFAULT '[]',
    
    -- Enhancement flags
    enhanced BOOLEAN DEFAULT FALSE,
    enhancement_cost DECIMAL(8,4) DEFAULT 0.00,
    enhancement_metadata JSONB DEFAULT '{}',
    
    -- Classification
    segment_type VARCHAR(50) DEFAULT 'highlight', -- highlight, transition, intro, outro, technical, dialogue
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_timing CHECK (start_time >= 0 AND end_time > start_time),
    CONSTRAINT valid_score CHECK (score >= 0 AND score <= 1),
    CONSTRAINT valid_confidence CHECK (confidence >= 0 AND confidence <= 1),
    CONSTRAINT valid_speaker_count CHECK (speaker_count >= 0),
    CONSTRAINT valid_motion CHECK (motion_intensity >= 0)
);

-- Indexes for video_segment
CREATE INDEX idx_video_segment_job_id ON video_segment(job_id);
CREATE INDEX idx_video_segment_score ON video_segment(score DESC);
CREATE INDEX idx_video_segment_timing ON video_segment(start_time, end_time);
CREATE INDEX idx_video_segment_type ON video_segment(segment_type);
CREATE INDEX idx_video_segment_enhanced ON video_segment(enhanced);

-- Checkpoints table - stores processing state for recovery
CREATE TABLE checkpoint (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
    stage VARCHAR(50) NOT NULL,
    data JSONB NOT NULL DEFAULT '{}',
    
    -- TTL and cleanup
    expires_at TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '24 hours'),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure one checkpoint per job per stage
    UNIQUE(job_id, stage)
);

-- Indexes for checkpoint
CREATE INDEX idx_checkpoint_job_id ON checkpoint(job_id);
CREATE INDEX idx_checkpoint_stage ON checkpoint(stage);
CREATE INDEX idx_checkpoint_expires_at ON checkpoint(expires_at);

-- Cost tracking table - detailed cost breakdown
CREATE TABLE cost_tracking (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
    
    -- Cost details
    service VARCHAR(50) NOT NULL, -- openai, deepgram, aws, etc
    operation VARCHAR(100) NOT NULL, -- transcription, gpt_analysis, etc
    amount DECIMAL(8,4) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    
    -- Usage metrics
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    duration_seconds FLOAT DEFAULT 0,
    api_calls INTEGER DEFAULT 1,
    
    -- Context
    stage VARCHAR(50),
    segment_id UUID REFERENCES video_segment(id) ON DELETE SET NULL,
    metadata JSONB DEFAULT '{}',
    
    -- Timestamp
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_amount CHECK (amount >= 0),
    CONSTRAINT valid_tokens CHECK (input_tokens >= 0 AND output_tokens >= 0),
    CONSTRAINT valid_duration CHECK (duration_seconds >= 0),
    CONSTRAINT valid_api_calls CHECK (api_calls >= 0)
);

-- Indexes for cost_tracking
CREATE INDEX idx_cost_tracking_job_id ON cost_tracking(job_id);
CREATE INDEX idx_cost_tracking_service ON cost_tracking(service);
CREATE INDEX idx_cost_tracking_created_at ON cost_tracking(created_at);

-- Processing metrics table - performance and usage tracking
CREATE TABLE processing_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    job_id UUID REFERENCES video_job(id) ON DELETE CASCADE,
    
    -- Metric identification
    metric_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(20) NOT NULL, -- counter, gauge, histogram, timer
    value FLOAT NOT NULL,
    unit VARCHAR(20),
    
    -- Context
    stage VARCHAR(50),
    component VARCHAR(50), -- ffmpeg, openai, database, etc
    labels JSONB DEFAULT '{}',
    
    -- Timestamp
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_metric_type CHECK (metric_type IN ('counter', 'gauge', 'histogram', 'timer'))
);

-- Indexes for processing_metrics
CREATE INDEX idx_processing_metrics_job_id ON processing_metrics(job_id);
CREATE INDEX idx_processing_metrics_name ON processing_metrics(metric_name);
CREATE INDEX idx_processing_metrics_timestamp ON processing_metrics(timestamp);
CREATE INDEX idx_processing_metrics_component ON processing_metrics(component);

-- User preferences table - configuration and settings
CREATE TABLE user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(100) NOT NULL, -- external user identifier
    
    -- Budget settings
    default_budget DECIMAL(8,4) DEFAULT 5.00,
    budget_warnings_enabled BOOLEAN DEFAULT TRUE,
    
    -- Quality preferences
    default_quality VARCHAR(20) DEFAULT 'enhanced', -- preview, basic, enhanced, final
    enable_smart_crop BOOLEAN DEFAULT TRUE,
    default_aspect_ratio VARCHAR(10) DEFAULT '9:16',
    
    -- Processing preferences
    max_highlight_duration INTEGER DEFAULT 60, -- seconds
    min_highlight_duration INTEGER DEFAULT 15, -- seconds
    enable_transitions BOOLEAN DEFAULT TRUE,
    enable_captions BOOLEAN DEFAULT FALSE,
    
    -- Notification preferences
    notify_on_completion BOOLEAN DEFAULT TRUE,
    webhook_url TEXT,
    
    -- Configuration
    preferences JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Ensure one record per user
    UNIQUE(user_id),
    
    -- Constraints
    CONSTRAINT valid_budget CHECK (default_budget > 0),
    CONSTRAINT valid_durations CHECK (min_highlight_duration > 0 AND max_highlight_duration > min_highlight_duration)
);

-- Indexes for user_preferences
CREATE INDEX idx_user_preferences_user_id ON user_preferences(user_id);

-- File metadata table - track processed files and deduplication
CREATE TABLE file_metadata (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- File identification
    file_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256
    file_path TEXT NOT NULL,
    file_size BIGINT NOT NULL,
    
    -- Video properties
    duration FLOAT NOT NULL,
    codec VARCHAR(50),
    format VARCHAR(50),
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fps FLOAT,
    bitrate INTEGER,
    color_space VARCHAR(50),
    
    -- Analysis results cache
    analysis_completed BOOLEAN DEFAULT FALSE,
    segments_generated INTEGER DEFAULT 0,
    last_processed_at TIMESTAMPTZ,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_file_size CHECK (file_size > 0),
    CONSTRAINT valid_duration CHECK (duration > 0),
    CONSTRAINT valid_dimensions CHECK (width > 0 AND height > 0),
    CONSTRAINT valid_segments CHECK (segments_generated >= 0)
);

-- Indexes for file_metadata
CREATE INDEX idx_file_metadata_hash ON file_metadata(file_hash);
CREATE INDEX idx_file_metadata_size ON file_metadata(file_size);
CREATE INDEX idx_file_metadata_duration ON file_metadata(duration);
CREATE INDEX idx_file_metadata_last_processed ON file_metadata(last_processed_at);

-- Create views for common queries

-- Active jobs view
CREATE VIEW active_jobs AS
SELECT 
    j.*,
    COUNT(s.id) as segment_count,
    AVG(s.score) as avg_segment_score,
    SUM(ct.amount) as current_cost
FROM video_job j
LEFT JOIN video_segment s ON j.id = s.job_id
LEFT JOIN cost_tracking ct ON j.id = ct.job_id
WHERE j.status IN ('queued', 'processing')
GROUP BY j.id;

-- Job summary view
CREATE VIEW job_summary AS
SELECT 
    j.id,
    j.status,
    j.input_path,
    j.output_path,
    j.duration,
    j.created_at,
    j.completed_at,
    COUNT(s.id) as total_segments,
    COUNT(CASE WHEN s.enhanced = TRUE THEN 1 END) as enhanced_segments,
    COALESCE(SUM(ct.amount), 0) as total_cost,
    j.budget_limit,
    CASE 
        WHEN j.completed_at IS NOT NULL THEN 
            EXTRACT(EPOCH FROM (j.completed_at - j.created_at))
        ELSE NULL 
    END as processing_duration_seconds
FROM video_job j
LEFT JOIN video_segment s ON j.id = s.job_id  
LEFT JOIN cost_tracking ct ON j.id = ct.job_id
GROUP BY j.id;

-- Top segments view (best highlights across all jobs)
CREATE VIEW top_segments AS
SELECT 
    s.*,
    j.input_path,
    j.status as job_status,
    RANK() OVER (PARTITION BY s.job_id ORDER BY s.score DESC) as job_rank,
    RANK() OVER (ORDER BY s.score DESC) as global_rank
FROM video_segment s
JOIN video_job j ON s.job_id = j.id
WHERE j.status = 'completed'
ORDER BY s.score DESC;

-- Cost analysis view
CREATE VIEW cost_analysis AS
SELECT 
    service,
    operation,
    COUNT(*) as call_count,
    SUM(amount) as total_cost,
    AVG(amount) as avg_cost_per_call,
    SUM(input_tokens) as total_input_tokens,
    SUM(output_tokens) as total_output_tokens,
    SUM(duration_seconds) as total_duration_seconds
FROM cost_tracking
GROUP BY service, operation
ORDER BY total_cost DESC;

-- Performance metrics view
CREATE VIEW performance_summary AS
SELECT 
    component,
    stage,
    metric_name,
    COUNT(*) as measurement_count,
    AVG(value) as avg_value,
    MIN(value) as min_value,
    MAX(value) as max_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY value) as p95_value
FROM processing_metrics
WHERE timestamp > NOW() - INTERVAL '24 hours'
GROUP BY component, stage, metric_name
ORDER BY component, stage, metric_name;

-- Functions for common operations

-- Function to clean up expired checkpoints
CREATE OR REPLACE FUNCTION cleanup_expired_checkpoints()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM checkpoint WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get job progress percentage
CREATE OR REPLACE FUNCTION get_job_progress(job_uuid UUID)
RETURNS INTEGER AS $$
DECLARE
    progress_pct INTEGER;
BEGIN
    SELECT progress INTO progress_pct 
    FROM video_job 
    WHERE id = job_uuid;
    
    RETURN COALESCE(progress_pct, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to calculate job cost efficiency (highlights per dollar)
CREATE OR REPLACE FUNCTION get_cost_efficiency(job_uuid UUID)
RETURNS DECIMAL AS $$
DECLARE
    segment_count INTEGER;
    total_cost DECIMAL;
    efficiency DECIMAL;
BEGIN
    SELECT 
        COUNT(s.id),
        COALESCE(SUM(ct.amount), 0)
    INTO segment_count, total_cost
    FROM video_job j
    LEFT JOIN video_segment s ON j.id = s.job_id
    LEFT JOIN cost_tracking ct ON j.id = ct.job_id
    WHERE j.id = job_uuid;
    
    IF total_cost > 0 THEN
        efficiency := segment_count::DECIMAL / total_cost;
    ELSE
        efficiency := segment_count::DECIMAL;
    END IF;
    
    RETURN efficiency;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic updates

-- Update video_job.total_cost when cost_tracking changes
CREATE OR REPLACE FUNCTION update_job_total_cost()
RETURNS TRIGGER AS $$
DECLARE
    job_uuid UUID;
    new_total DECIMAL;
BEGIN
    -- Get job_id from the affected row
    IF TG_OP = 'DELETE' THEN
        job_uuid := OLD.job_id;
    ELSE
        job_uuid := NEW.job_id;
    END IF;
    
    -- Calculate new total
    SELECT COALESCE(SUM(amount), 0) 
    INTO new_total
    FROM cost_tracking 
    WHERE job_id = job_uuid;
    
    -- Update video_job
    UPDATE video_job 
    SET total_cost = new_total 
    WHERE id = job_uuid;
    
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_job_cost
    AFTER INSERT OR UPDATE OR DELETE ON cost_tracking
    FOR EACH ROW
    EXECUTE FUNCTION update_job_total_cost();

-- Update user_preferences.updated_at on changes
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Create initial data

-- Insert default user preferences
INSERT INTO user_preferences (user_id, preferences) VALUES 
('default', '{"theme": "dark", "notifications": true}')
ON CONFLICT (user_id) DO NOTHING;

-- Performance optimization: analyze tables
ANALYZE video_job;
ANALYZE video_segment;
ANALYZE checkpoint;
ANALYZE cost_tracking;
ANALYZE processing_metrics;
ANALYZE user_preferences;
ANALYZE file_metadata;

-- Grant permissions (adjust as needed for your environment)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO your_app_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO your_app_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO your_app_user;