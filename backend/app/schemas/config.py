from typing import Literal
from typing import List

from pydantic import BaseModel, Field, field_validator


FaceBlurMethod = Literal["gaussian", "pixelate", "blackout"]
FrameExtractionMethod = Literal["interval", "fps", "motion_based"]
GeoTaggingMode = Literal["metadata", "manual"]


class FaceBlurConfig(BaseModel):
    enabled: bool = False
    method: FaceBlurMethod = "gaussian"
    intensity: int = Field(default=25, ge=1, le=100)


class FrameExtractionConfig(BaseModel):
    method: FrameExtractionMethod = "interval"
    value: float = Field(default=5, gt=0)
    motion_threshold: float = Field(default=25.0, ge=0.0)


class ObjectDetectionConfig(BaseModel):
    model: str = "yolov8n"
    classes: list[str] = Field(default_factory=lambda: ["person", "car", "bike"])
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("classes")
    @classmethod
    def ensure_unique_classes(cls, value: list[str]) -> list[str]:
        unique = []
        seen = set()
        for item in value:
            normalized = item.strip().lower()
            if normalized and normalized not in seen:
                unique.append(normalized)
                seen.add(normalized)
        if not unique:
            raise ValueError("At least one detection class must be selected.")
        return unique


class GeoTaggingConfig(BaseModel):
    mode: GeoTaggingMode = "metadata"
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    @field_validator("longitude")
    @classmethod
    def validate_manual_coordinates(cls, value: float | None, info):
        mode = info.data.get("mode")
        latitude = info.data.get("latitude")
        if mode == "manual" and (latitude is None or value is None):
            raise ValueError("Manual geo-tagging requires latitude and longitude.")
        return value
    
    # class ViolationDetectionConfig(BaseModel):
    #     enabled:bool = False
    #     list_violations: List[str] = Field(default_factory=lambda:["triple_riding"])


class ProcessingConfig(BaseModel):
    audio_removal: bool = True
    face_blur: FaceBlurConfig = Field(default_factory=FaceBlurConfig)
    frame_extraction: FrameExtractionConfig = Field(default_factory=FrameExtractionConfig)
    object_detection: ObjectDetectionConfig = Field(default_factory=ObjectDetectionConfig)
    geo_tagging: GeoTaggingConfig = Field(default_factory=GeoTaggingConfig)

    # violation_detection: ViolationDetectionConfig = Field(default_factory=ViolationDetectionConfig)