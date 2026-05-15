from datetime import datetime
from pathlib import Path

from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy.orm import Session

from backend.app.models.video import Detection, FrameImage, ProcessingRun, Video
from backend.app.schemas.config import ProcessingConfig
from backend.app.services.processors.audio import AudioRemovalProcessor
from backend.app.services.processors.face_blur import FaceBlurProcessor
from backend.app.services.processors.frame_extractor import FrameExtractionProcessor
from backend.app.services.processors.geotagger import GeoTaggingProcessor
from backend.app.services.processors.object_detector import ObjectDetectionProcessor
from backend.app.services.storage import (
    download_file,
    ensure_video_dir,
    upload_frame_image,
    upload_processed_video,
    write_json,
)


class VideoProcessingPipeline:
    def __init__(self) -> None:
        self.audio = AudioRemovalProcessor()
        self.face_blur = FaceBlurProcessor()
        self.frame_extractor = FrameExtractionProcessor()
        self.object_detector = ObjectDetectionProcessor()
        self.geotagger = GeoTaggingProcessor()

    def run(self, db: Session, video: Video, config: ProcessingConfig) -> dict:
        work_dir = ensure_video_dir(video.id)
        run = ProcessingRun(video_id=video.id, status="running", stage_logs={})
        db.add(run)
        video.status = "processing"
        db.commit()
        db.refresh(run)

        original_video = self._ensure_original_video_cache(video)
        current_video = original_video
        stage_logs: dict = {"started_at": datetime.utcnow().isoformat()}

        try:
            if config.audio_removal:
                current_video, stage_logs["audio_removal"] = self.audio.run(current_video, work_dir / "audio")
                video.audio_removed_path = str(current_video)
            else:
                stage_logs["audio_removal"] = {"status": "skipped", "reason": "disabled"}

            if config.face_blur.enabled:
                current_video, stage_logs["face_blur"] = self.face_blur.run(
                    current_video, config.face_blur, work_dir / "face_blur"
                )
                video.processed_video_path = str(current_video)
            else:
                stage_logs["face_blur"] = {"status": "skipped", "reason": "disabled"}
                video.processed_video_path = str(current_video)

            frames, stage_logs["frame_extraction"] = self.frame_extractor.run(
                current_video, config.frame_extraction, work_dir / "frames"
            )
            db.query(FrameImage).filter(FrameImage.video_id == video.id).delete()
            frame_records = []
            for frame in frames:
                stored_frame = upload_frame_image(video.id, frame["frame_index"], Path(frame["path"]))
                frame_records.append(
                    FrameImage(
                        video_id=video.id,
                        frame_index=frame["frame_index"],
                        frame_number=frame["frame_index"],
                        image_path=stored_frame.object_url,
                    )   
                )
            if frame_records:
                db.add_all(frame_records)
            stage_logs["frame_extraction"]["images_uploaded"] = len(frame_records)

            detections, stage_logs["object_detection"] = self.object_detector.run(frames, config.object_detection)
            geo_source = original_video if config.geo_tagging.mode == "metadata" else current_video
            location, stage_logs["geo_tagging"] = self.geotagger.resolve(geo_source, config.geo_tagging)

            processed_object = upload_processed_video(video.id, current_video)
            video.processed_bucket_name = processed_object.bucket_name
            video.processed_object_key = processed_object.object_key
            video.processed_object_url = processed_object.object_url
            stage_logs["processed_video_storage"] = {
                "status": "completed",
                "bucket_name": processed_object.bucket_name,
                "object_key": processed_object.object_key,
            }

            db.query(Detection).filter(Detection.video_id == video.id).delete()
            for item in detections:
                latitude = location["latitude"] if location else None
                longitude = location["longitude"] if location else None
                db.add(
                    Detection(
                        video_id=video.id,
                        frame_index=item["frame_index"],
                        timestamp_seconds=item["timestamp_seconds"],
                        object_class=item["object_class"],
                        confidence=item["confidence"],
                        bbox=item["bbox"],
                        latitude=latitude,
                        longitude=longitude,
                        source_mode=config.geo_tagging.mode if location else None,
                        location=self._build_location_value(db, latitude, longitude),
                        #tags=[item["object_class"]],
                    )
                )

            run.status = "completed"
            video.status = "processed"
        except Exception as exc:
            run.status = "failed"
            video.status = "failed"
            stage_logs["error"] = str(exc)
            raise
        finally:
            stage_logs["ended_at"] = datetime.utcnow().isoformat()
            write_json(work_dir / "artifacts" / "stage_logs.json", stage_logs)
            run.completed_at = datetime.utcnow()
            run.stage_logs = stage_logs
            db.commit()

        return stage_logs

    def _build_location_value(self, db: Session, latitude: float | None, longitude: float | None):
        if latitude is None or longitude is None:
            return None
        if db.bind and db.bind.dialect.name == "sqlite":
            return f"POINT({longitude} {latitude})"
        return from_shape(Point(longitude, latitude), srid=4326)

    def _ensure_original_video_cache(self, video: Video) -> Path:
        if video.stored_path:
            candidate = Path(video.stored_path)
            if candidate.exists():
                return candidate

        if not video.original_bucket_name or not video.original_object_key:
            raise RuntimeError("Original video object metadata is missing.")

        local_name = Path(video.original_object_key).name
        target = ensure_video_dir(video.id) / "cache" / local_name
        if target.exists():
            return target
        return download_file(video.original_bucket_name, video.original_object_key, target)
