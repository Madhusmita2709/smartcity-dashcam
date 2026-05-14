from datetime import datetime

from pydantic import BaseModel

from backend.app.schemas.config import ProcessingConfig


class UploadResponseItem(BaseModel):
    video_id: int
    filename: str
    status: str
    config: ProcessingConfig
    original_video_url: str | None = None


class UploadResponse(BaseModel):
    items: list[UploadResponseItem]


class ProcessResponse(BaseModel):
    video_id: int
    status: str
    stages: dict


class DetectionResponse(BaseModel):
    id: int
    frame_index: int
    timestamp_seconds: float
    object_class: str
    confidence: float
    bbox: dict
    latitude: float | None
    longitude: float | None
    source_mode: str | None
    extracted_at: datetime


class FrameImageResponse(BaseModel):
    id: int
    frame_index: int
    timestamp_seconds: float
    bucket_name: str
    object_key: str
    object_url: str
    content_type: str | None
    size_bytes: int | None


class ResultsResponse(BaseModel):
    video_id: int
    filename: str
    status: str
    config: ProcessingConfig
    original_video_url: str | None
    processed_video_url: str | None
    detections: list[DetectionResponse]
    frame_images: list[FrameImageResponse]


class HeatmapPoint(BaseModel):
    latitude: float
    longitude: float
    intensity: float
    object_class: str
    timestamp_seconds: float
    video_id: int
