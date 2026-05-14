from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    app_title: str = Field(default="Video-to-Map Analytics Web Application")
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/dashcam"
    )
    media_root: Path = Field(default=Path("media"))
    model_cache_dir: Path = Field(default=Path("models"))
    default_detection_model: str = Field(default="yolov8n")
    minio_endpoint: str = Field(default="127.0.0.1:9000")
    minio_api_url: str = Field(default="http://127.0.0.1:9000")
    minio_console_url: str = Field(default="http://127.0.0.1:9001")
    minio_access_key: str = Field(default="minioadmin")
    minio_secret_key: str = Field(default="minioadmin123")
    minio_secure: bool = Field(default=False)
    minio_originals_bucket: str = Field(default="dashcam-original-videos")
    minio_processed_bucket: str = Field(default="dashcam-processed-videos")
    minio_images_bucket: str = Field(default="dashcam-images")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.media_root.mkdir(parents=True, exist_ok=True)
    settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
    return settings
