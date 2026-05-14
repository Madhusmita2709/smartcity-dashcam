CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS videos (
  id SERIAL PRIMARY KEY,
  filename VARCHAR(255) NOT NULL,
  stored_path TEXT,
  config_json JSONB NOT NULL,
  upload_time TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  status VARCHAR(32) NOT NULL DEFAULT 'uploaded',
  audio_removed_path TEXT,
  processed_video_path TEXT,
  original_bucket_name VARCHAR(128),
  original_object_key TEXT,
  original_object_url TEXT,
  processed_bucket_name VARCHAR(128),
  processed_object_key TEXT,
  processed_object_url TEXT
);

CREATE TABLE IF NOT EXISTS processing_runs (
  id SERIAL PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP,
  status VARCHAR(32) NOT NULL DEFAULT 'queued',
  stage_logs JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS detections (
  id SERIAL PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  frame_index INTEGER NOT NULL,
  timestamp_seconds DOUBLE PRECISION NOT NULL,
  object_class VARCHAR(64) NOT NULL,
  confidence DOUBLE PRECISION NOT NULL,
  bbox JSONB NOT NULL,
  latitude DOUBLE PRECISION,
  longitude DOUBLE PRECISION,
  location geometry(Point, 4326),
  source_mode VARCHAR(32),
  extracted_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  tags TEXT[] NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS frame_images (
  id SERIAL PRIMARY KEY,
  video_id INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
  frame_index INTEGER NOT NULL,
  timestamp_seconds DOUBLE PRECISION NOT NULL,
  bucket_name VARCHAR(128) NOT NULL,
  object_key TEXT NOT NULL,
  object_url TEXT NOT NULL,
  content_type VARCHAR(128),
  size_bytes INTEGER,
  created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_detections_video_id ON detections(video_id);
CREATE INDEX IF NOT EXISTS idx_detections_object_class ON detections(object_class);
CREATE INDEX IF NOT EXISTS idx_detections_location ON detections USING GIST(location);
CREATE INDEX IF NOT EXISTS idx_frame_images_video_id ON frame_images(video_id);
CREATE INDEX IF NOT EXISTS idx_frame_images_frame_index ON frame_images(frame_index);
