-- PRODUCTION DATABASE SCHEMA FOR MONTAGE VIDEO PROCESSING PIPELINE
-- Version: 2.0 - Complete schema with all required tables
-- Generated: 2025-07-21

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 1. VIDEO_JOB TABLE - Central tracking of all video processing jobs
CREATE TABLE IF NOT EXISTS video_job (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_hash        CHAR(64) NOT NULL UNIQUE,
  status          TEXT NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'processing', 'completed', 'failed', 'cancelled')),
  error_message   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ,
  -- Video metadata
  duration        FLOAT CHECK (duration >= 0),
  codec           TEXT,
  color_space     TEXT,
  resolution      TEXT,
  fps             FLOAT CHECK (fps > 0 AND fps <= 1000),
  -- Cost tracking
  total_cost      NUMERIC(10,4) DEFAULT 0 CHECK (total_cost >= 0),
  -- Additional metadata as JSON
  metadata        JSONB DEFAULT '{}'::jsonb
);

-- Indexes for video_job
CREATE INDEX IF NOT EXISTS idx_video_job_status ON video_job(status);
CREATE INDEX IF NOT EXISTS idx_video_job_created_at ON video_job(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_job_src_hash ON video_job(src_hash);
CREATE INDEX IF NOT EXISTS idx_video_job_completed_at ON video_job(completed_at DESC) WHERE completed_at IS NOT NULL;

-- 2. TRANSCRIPT_CACHE TABLE - Cache transcription results
CREATE TABLE IF NOT EXISTS transcript_cache (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  sha256          CHAR(64) NOT NULL,
  transcript      TEXT NOT NULL,
  provider        TEXT NOT NULL DEFAULT 'whisper' CHECK (provider IN ('whisper', 'deepgram', 'openai', 'assembly')),
  model           TEXT,
  word_count      INTEGER CHECK (word_count >= 0),
  confidence      FLOAT CHECK (confidence >= 0 AND confidence <= 1),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at      TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '30 days'),
  metadata        JSONB DEFAULT '{}'::jsonb,
  UNIQUE(sha256, provider, model)
);

-- Indexes for transcript_cache
CREATE INDEX IF NOT EXISTS idx_transcript_cache_sha256 ON transcript_cache(sha256);
CREATE INDEX IF NOT EXISTS idx_transcript_cache_expires_at ON transcript_cache(expires_at) WHERE expires_at IS NOT NULL;

-- 3. HIGHLIGHT TABLE - Store identified video highlights
CREATE TABLE IF NOT EXISTS highlight (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  start_time      FLOAT NOT NULL CHECK (start_time >= 0),
  end_time        FLOAT NOT NULL CHECK (end_time > start_time),
  duration        FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
  score           FLOAT NOT NULL CHECK (score >= 0),
  -- Scoring components
  tf_idf_score    FLOAT CHECK (tf_idf_score >= 0),
  audio_rms       FLOAT CHECK (audio_rms >= 0 AND audio_rms <= 1),
  visual_energy   FLOAT CHECK (visual_energy >= 0 AND visual_energy <= 1),
  face_count      INTEGER CHECK (face_count >= 0),
  motion_score    FLOAT CHECK (motion_score >= 0 AND motion_score <= 1),
  -- Content
  transcript      TEXT,
  title           TEXT,
  reason          TEXT,
  keywords        TEXT[],
  speaker_id      TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for highlight
CREATE INDEX IF NOT EXISTS idx_highlight_job_id ON highlight(job_id);
CREATE INDEX IF NOT EXISTS idx_highlight_score ON highlight(score DESC);
CREATE INDEX IF NOT EXISTS idx_highlight_times ON highlight(start_time, end_time);

-- 4. JOB_CHECKPOINT TABLE - Recovery checkpoints for long-running jobs
CREATE TABLE IF NOT EXISTS job_checkpoint (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  stage           TEXT NOT NULL,
  checkpoint_data JSONB NOT NULL,
  progress        FLOAT CHECK (progress >= 0 AND progress <= 100),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  expires_at      TIMESTAMPTZ DEFAULT (NOW() + INTERVAL '7 days'),
  UNIQUE(job_id, stage)
);

-- Indexes for job_checkpoint
CREATE INDEX IF NOT EXISTS idx_job_checkpoint_job_id ON job_checkpoint(job_id);
CREATE INDEX IF NOT EXISTS idx_job_checkpoint_expires_at ON job_checkpoint(expires_at) WHERE expires_at IS NOT NULL;

-- 5. API_COST_LOG TABLE - Track API usage and costs
CREATE TABLE IF NOT EXISTS api_cost_log (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  api_name        TEXT NOT NULL,
  endpoint        TEXT,
  model           TEXT,
  tokens_used     INTEGER CHECK (tokens_used >= 0),
  cost_usd        NUMERIC(10,6) NOT NULL CHECK (cost_usd >= 0),
  response_time   FLOAT CHECK (response_time >= 0),
  status_code     INTEGER,
  error_message   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for api_cost_log
CREATE INDEX IF NOT EXISTS idx_api_cost_log_job_id ON api_cost_log(job_id);
CREATE INDEX IF NOT EXISTS idx_api_cost_log_created_at ON api_cost_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_api_cost_log_api_name ON api_cost_log(api_name);

-- 6. VIDEO_SEGMENT TABLE - Store video edit segments
CREATE TABLE IF NOT EXISTS video_segment (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  start_time      FLOAT NOT NULL CHECK (start_time >= 0),
  end_time        FLOAT NOT NULL CHECK (end_time > start_time),
  duration        FLOAT GENERATED ALWAYS AS (end_time - start_time) STORED,
  transition_type TEXT DEFAULT 'fade' CHECK (transition_type IN ('cut', 'fade', 'dissolve', 'wipe', 'zoom')),
  transition_duration FLOAT DEFAULT 0.5 CHECK (transition_duration >= 0 AND transition_duration <= 5),
  segment_order   INTEGER NOT NULL CHECK (segment_order >= 0),
  crop_params     JSONB,
  effects         JSONB,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for video_segment
CREATE INDEX IF NOT EXISTS idx_video_segment_job_id ON video_segment(job_id);
CREATE INDEX IF NOT EXISTS idx_video_segment_order ON video_segment(job_id, segment_order);

-- 7. ALERT_LOG TABLE - System monitoring and alerts
CREATE TABLE IF NOT EXISTS alert_log (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  alert_name      TEXT NOT NULL,
  severity        TEXT NOT NULL CHECK (severity IN ('info', 'warning', 'error', 'critical')),
  component       TEXT,
  message         TEXT NOT NULL,
  labels          JSONB DEFAULT '{}'::jsonb,
  annotations     JSONB DEFAULT '{}'::jsonb,
  status          TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'resolved', 'acknowledged')),
  starts_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ends_at         TIMESTAMPTZ,
  acknowledged_at TIMESTAMPTZ,
  acknowledged_by TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for alert_log
CREATE INDEX IF NOT EXISTS idx_alert_log_created_at ON alert_log(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_alert_log_alert_name ON alert_log(alert_name);
CREATE INDEX IF NOT EXISTS idx_alert_log_severity ON alert_log(severity);
CREATE INDEX IF NOT EXISTS idx_alert_log_status ON alert_log(status) WHERE status = 'active';

-- 8. PERFORMANCE_METRICS TABLE - Track system performance
CREATE TABLE IF NOT EXISTS performance_metrics (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID REFERENCES video_job(id) ON DELETE CASCADE,
  metric_name     TEXT NOT NULL,
  metric_value    FLOAT NOT NULL,
  unit            TEXT,
  tags            JSONB DEFAULT '{}'::jsonb,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance_metrics
CREATE INDEX IF NOT EXISTS idx_performance_metrics_created_at ON performance_metrics(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_job_id ON performance_metrics(job_id) WHERE job_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_performance_metrics_name ON performance_metrics(metric_name);

-- SECURITY: Row-level security policies (when RLS is enabled)
-- Uncomment these when implementing multi-tenant support
-- ALTER TABLE video_job ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE highlight ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE api_cost_log ENABLE ROW LEVEL SECURITY;

-- MAINTENANCE: Automated cleanup functions
CREATE OR REPLACE FUNCTION cleanup_expired_data() RETURNS void AS $$
BEGIN
  -- Delete expired checkpoints
  DELETE FROM job_checkpoint WHERE expires_at < NOW();
  
  -- Delete old transcript cache entries
  DELETE FROM transcript_cache WHERE expires_at < NOW();
  
  -- Delete old alerts
  DELETE FROM alert_log WHERE status = 'resolved' AND ends_at < NOW() - INTERVAL '30 days';
  
  -- Delete old performance metrics
  DELETE FROM performance_metrics WHERE created_at < NOW() - INTERVAL '90 days';
END;
$$ LANGUAGE plpgsql;

-- TRIGGERS: Automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add updated_at columns and triggers where needed
ALTER TABLE video_job ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW();
CREATE TRIGGER update_video_job_updated_at BEFORE UPDATE ON video_job 
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- VIEWS: Useful aggregated views
CREATE OR REPLACE VIEW job_summary AS
SELECT 
  j.id,
  j.src_hash,
  j.status,
  j.created_at,
  j.completed_at,
  j.total_cost,
  COUNT(DISTINCT h.id) as highlight_count,
  COUNT(DISTINCT s.id) as segment_count,
  MIN(h.start_time) as first_highlight_start,
  MAX(h.end_time) as last_highlight_end
FROM video_job j
LEFT JOIN highlight h ON j.id = h.job_id
LEFT JOIN video_segment s ON j.id = s.job_id
GROUP BY j.id;

-- Grant appropriate permissions (adjust for your user)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO montage_user;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO montage_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO montage_user;

-- Add comments for documentation
COMMENT ON TABLE video_job IS 'Central table tracking all video processing jobs';
COMMENT ON TABLE transcript_cache IS 'Caches transcription results to avoid reprocessing';
COMMENT ON TABLE highlight IS 'Stores identified highlights with scoring metrics';
COMMENT ON TABLE job_checkpoint IS 'Recovery checkpoints for resuming failed jobs';
COMMENT ON TABLE api_cost_log IS 'Tracks API usage and costs for budget management';
COMMENT ON TABLE video_segment IS 'Defines video segments for final edit';
COMMENT ON TABLE alert_log IS 'System monitoring and alert history';
COMMENT ON TABLE performance_metrics IS 'Performance tracking for optimization';

-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
  version INTEGER PRIMARY KEY,
  applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  description TEXT
);

INSERT INTO schema_version (version, description) 
VALUES (2, 'Complete production schema with all tables and indexes')
ON CONFLICT (version) DO NOTHING;