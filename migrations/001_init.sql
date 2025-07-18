-- Initial schema for AI Video Processing Pipeline
-- As specified in Tasks.md

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE video_job (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_hash        CHAR(64) NOT NULL UNIQUE,
  status          TEXT NOT NULL DEFAULT 'queued',
  error_message   TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  completed_at    TIMESTAMPTZ,
  -- Additional metadata columns
  duration        FLOAT,
  codec           TEXT,
  color_space     TEXT,
  total_cost      NUMERIC(10,4) DEFAULT 0,
  metadata        JSONB
);

CREATE INDEX idx_video_job_status ON video_job(status);
CREATE INDEX idx_video_job_created_at ON video_job(created_at);
CREATE INDEX idx_video_job_src_hash ON video_job(src_hash);

-- Transcript cache table
CREATE TABLE transcript_cache (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  src_hash        CHAR(64) NOT NULL,
  transcript_data TEXT NOT NULL,
  provider        TEXT NOT NULL DEFAULT 'whisper',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(src_hash, provider)
);

CREATE INDEX idx_transcript_cache_src_hash ON transcript_cache(src_hash);

-- Highlight table
CREATE TABLE highlight (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  start_time      FLOAT NOT NULL,
  end_time        FLOAT NOT NULL,
  score           FLOAT NOT NULL,
  tf_idf_score    FLOAT,
  audio_rms       FLOAT,
  visual_energy   FLOAT,
  transcript      TEXT,
  reason          TEXT,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_highlight_job_id ON highlight(job_id);
CREATE INDEX idx_highlight_score ON highlight(score DESC);

-- Job checkpoint table for recovery
CREATE TABLE job_checkpoint (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  stage           TEXT NOT NULL,
  checkpoint_data JSONB NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE(job_id, stage)
);

CREATE INDEX idx_job_checkpoint_job_id ON job_checkpoint(job_id);

-- API cost tracking
CREATE TABLE api_cost_log (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  api_name        TEXT NOT NULL,
  cost_usd        NUMERIC(10,4) NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_api_cost_log_job_id ON api_cost_log(job_id);
CREATE INDEX idx_api_cost_log_created_at ON api_cost_log(created_at);

-- Video segments for edit plans
CREATE TABLE video_segment (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  job_id          UUID NOT NULL REFERENCES video_job(id) ON DELETE CASCADE,
  start_time      FLOAT NOT NULL,
  end_time        FLOAT NOT NULL,
  transition_type TEXT DEFAULT 'fade',
  segment_order   INTEGER NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_video_segment_job_id ON video_segment(job_id);

-- Alert log for monitoring
CREATE TABLE alert_log (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  alert_name      TEXT NOT NULL,
  severity        TEXT NOT NULL,
  labels          JSONB,
  annotations     JSONB,
  status          TEXT NOT NULL,
  starts_at       TIMESTAMPTZ,
  ends_at         TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_alert_log_created_at ON alert_log(created_at);
CREATE INDEX idx_alert_log_alert_name ON alert_log(alert_name);