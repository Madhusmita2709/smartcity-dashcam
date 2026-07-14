from sqlalchemy import Column, Integer, String, Float, ForeignKey, JSON, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from backend.app.db.database import Base
from geoalchemy2 import Geometry



class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    stored_path = Column(String, nullable=True)
    config_json = Column(JSON, nullable=True)
    upload_time = Column(DateTime, default=func.now(), nullable=False)
    status = Column(String(32), default="uploaded", nullable=False)

    audio_removed_path = Column(String, nullable=True)
    processed_video_path = Column(String, nullable=True)

    original_bucket_name = Column(String(128), nullable=True)
    original_object_key = Column(String, nullable=True)
    original_object_url = Column(String, nullable=True)

    processed_bucket_name = Column(String(128), nullable=True)
    processed_object_key = Column(String(128), nullable=True)
    processed_object_url = Column(String, nullable=True)

    frames = relationship("FrameImage", back_populates="video")


class FrameImage(Base):
    __tablename__ = "frame_images"

    id = Column(Integer, primary_key=True, index=True)
    frame_number = Column(Integer)
    image_path = Column(String)

    video_id = Column(Integer, ForeignKey("videos.id"))
    video = relationship("Video", back_populates="frames")
    frame_index = Column(Integer)


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    frame_index = Column(Integer, nullable=False)
    timestamp_seconds = Column(Float, nullable=False)
    object_class = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox = Column(JSON, nullable=False)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    source_mode = Column(String(32), nullable=True)

    extracted_at = Column(DateTime, nullable=False, server_default=func.now())
    location = Column(Geometry('POINT', srid=4326), nullable=True)

class ProcessingRun(Base):
    __tablename__ = "processing_runs"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"))

    started_at = Column(DateTime, nullable=False, default=func.now())
    completed_at = Column(DateTime, nullable=True)

    status = Column(String(32), default="queued", nullable=False)

    stage_logs = Column(JSON, nullable=False, default={})

# ==========================================================================
# DASHBOARD ANALYTICS TABLES (ADDED FOR PIPELINE COMPATIBILITY)
# ==========================================================================

class VideoRoute(Base):
    __tablename__ = "video_routes"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp_seconds = Column(Float, default=0.0)
    sequence_order = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())


class ProjectViolation(Base):
    __tablename__ = "project_violations"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)

    timestamp_seconds = Column(Float, nullable=False)

    plate_number = Column(String(64), default="UNKNOWN", nullable=True)

    violation_type = Column(String(64), nullable=False)

    image_url = Column(String, nullable=False)

    confidence = Column(Float, nullable=False)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    location = Column(Geometry("POINT", srid=4326), nullable=True)

    extracted_at = Column(DateTime, nullable=False, server_default=func.now())

# ==========================================================================
# VIDEO REGISTRY (Dashboard Timeline Support)
# ==========================================================================

class VideoRegistry(Base):
    __tablename__ = "video_registry"

    video_id = Column(
        Integer,
        ForeignKey("videos.id"),
        primary_key=True,
        nullable=False
    )

    processed_date = Column(
        DateTime,
        nullable=False,
        server_default=func.now()
    )