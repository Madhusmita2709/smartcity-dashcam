from sqlalchemy import text

from backend.app.db.database import engine


def apply_schema_compatibility() -> None:
    statements = [
        """
        ALTER TABLE videos
        ADD COLUMN IF NOT EXISTS original_bucket_name VARCHAR(128),
        ADD COLUMN IF NOT EXISTS original_object_key TEXT,
        ADD COLUMN IF NOT EXISTS original_object_url TEXT,
        ADD COLUMN IF NOT EXISTS processed_bucket_name VARCHAR(128),
        ADD COLUMN IF NOT EXISTS processed_object_key TEXT,
        ADD COLUMN IF NOT EXISTS processed_object_url TEXT
        """,
        "ALTER TABLE videos ALTER COLUMN stored_path DROP NOT NULL",
        """
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
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_frame_images_video_id ON frame_images(video_id)",
        "CREATE INDEX IF NOT EXISTS idx_frame_images_frame_index ON frame_images(frame_index)",
    ]

    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
