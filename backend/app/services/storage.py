import json
import mimetypes
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from backend.app.core.config import get_settings

try:
    from minio import Minio
except ImportError:  # pragma: no cover - dependency presence is validated at runtime
    Minio = None


settings = get_settings()

@dataclass
class StoredObject:
    bucket_name: str
    object_key: str
    object_url: str
    content_type: str | None
    size_bytes: int | None


def ensure_video_dir(video_id: int) -> Path:
    directory = settings.media_root / str(video_id)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_work_subdir(video_id: int, name: str) -> Path:
    directory = ensure_video_dir(video_id) / name
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_upload_to_disk(video_id: int, upload: UploadFile) -> Path:
    destination_dir = ensure_work_subdir(video_id, "cache")
    safe_name = f"{uuid4().hex}_{Path(upload.filename or 'video').name}"
    destination = destination_dir / safe_name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)
    return destination


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def get_minio_client():
    if Minio is None:
        raise RuntimeError("MinIO dependency missing. Install the 'minio' Python package.")
    return Minio(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
    )


def ensure_buckets() -> None:
    client = get_minio_client()
    for bucket_name in (
        settings.minio_originals_bucket,
        settings.minio_processed_bucket,
        settings.minio_images_bucket,
    ):
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)


def build_object_url(bucket_name: str, object_key: str) -> str:
    return f"{settings.minio_api_url.rstrip('/')}/{bucket_name}/{object_key}"


def _guess_content_type(source: Path, fallback: str | None = None) -> str | None:
    guessed, _ = mimetypes.guess_type(source.name)
    return fallback or guessed or "application/octet-stream"


def upload_file(bucket_name: str, object_key: str, source: Path, content_type: str | None = None) -> StoredObject:
    client = get_minio_client()
    resolved_content_type = _guess_content_type(source, content_type)
    client.fput_object(
        bucket_name=bucket_name,
        object_name=object_key,
        file_path=str(source),
        content_type=resolved_content_type,
    )
    size_bytes = source.stat().st_size if source.exists() else None
    return StoredObject(
        bucket_name=bucket_name,
        object_key=object_key,
        object_url=build_object_url(bucket_name, object_key),
        content_type=resolved_content_type,
        size_bytes=size_bytes,
    )


def download_file(bucket_name: str, object_key: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    client = get_minio_client()
    client.fget_object(bucket_name, object_key, str(destination))
    return destination


def upload_original_video(video_id: int, filename: str, source: Path) -> StoredObject:
    object_key = f"videos/{video_id}/original/{source.name}"
    return upload_file(settings.minio_originals_bucket, object_key, source, "video/mp4")


def upload_processed_video(video_id: int, source: Path) -> StoredObject:
    object_key = f"videos/{video_id}/processed/{source.name}"
    return upload_file(settings.minio_processed_bucket, object_key, source, "video/mp4")


def upload_frame_image(video_id: int, frame_index: int, source: Path) -> StoredObject:
    object_key = f"videos/{video_id}/frames/frame_{frame_index:06d}{source.suffix.lower()}"
    return upload_file(settings.minio_images_bucket, object_key, source, "image/jpeg")
