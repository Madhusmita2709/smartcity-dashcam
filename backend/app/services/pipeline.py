from datetime import datetime, date
import time
import inspect
import json
from pathlib import Path

from geoalchemy2.shape import from_shape
from shapely.geometry import Point
from sqlalchemy.orm import Session
from sqlalchemy import text
from backend.app import db
from backend.app.models.video import (
    Detection,
    FrameImage,
    Video,
    ProcessingRun,
    VideoRoute,
    ProjectViolation,
)

from backend.app.schemas import config
from backend.app.schemas.config import ProcessingConfig
from backend.app.services.processors.audio import AudioRemovalProcessor
from backend.app.services.processors.face_blur import FaceBlurProcessor
from backend.app.services.processors.triple_riding import TripleRidingDetector
from backend.app.services.processors.lane_detector import detect_lanes_and_save_config
from backend.app.services.processors.wrong_way import WrongWayDetector
from backend.app.services.processors.vehicle_speed import VehicleSpeedEstimator
from backend.app.services.processors.no_number_plate import NoNumberPlateDetector
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

# Points directly to backend/app/services where default_mapping.json lives
ENGINE_CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_MAPPING_FILE = ENGINE_CONFIG_DIR / "default_mapping.json"
CUSTOM_MAPPING_FILE = ENGINE_CONFIG_DIR / "custom_mapping.json"

class VideoProcessingPipeline:

    def __init__(self) -> None:
        self.audio = AudioRemovalProcessor()
        self.face_blur = FaceBlurProcessor()
        self.triple_riding = TripleRidingDetector()
        self.wrong_way = WrongWayDetector()
        self.overspeed = VehicleSpeedEstimator()
        self.no_number_plate = NoNumberPlateDetector()
        self.frame_extractor = FrameExtractionProcessor()
        self.object_detector = ObjectDetectionProcessor()
        self.geotagger = GeoTaggingProcessor()

    def _get_active_model_mapping(self, use_custom: bool = False) -> dict:
        """
        Reads from default_mapping.json to construct a flat dictionary lookup table
        of current model overrides for runtime module consumption.
        """
        mapping_file = CUSTOM_MAPPING_FILE if use_custom else DEFAULT_MAPPING_FILE
        flat_overrides = {}
        
        if mapping_file.exists():
            try:
                with open(mapping_file, "r", encoding="utf-8") as f:
                    blueprints = json.load(f)
                    for violation_id, meta in blueprints.items():
                        flat_overrides[violation_id] = {
                            task["id"]: task["default"] for task in meta.get("tasks", [])
                        }
                    return flat_overrides
            except Exception as e:
                print(f"[Pipeline] Fallback to system defaults due to file fault: {e}")
                
        # Factory fallbacks if configuration file isn't initialized yet
        return {
            "triple_riding": {"vehicle_detection": "yolov8n.pt", "person_detection": "triple_riding.pt"},
            "wrong_way": {"vehicle_detection": "yolov8n.pt", "tracking": "bytetrack"},
            "no_number_plate": {"plate_detection": "license_plate.pt", "ocr": "model (1).pt"},
            "overspeed": {"vehicle_detection": "yolov8n.pt", "tracking": "bytetrack"}
        }

    def run( self, db: Session, video: Video, config: ProcessingConfig ) -> dict:
        total_start = time.perf_counter()
        timings = {}
        # Ingest active model blueprints maps directly from storage
        use_custom = False

        if hasattr(config, "violation_pipeline"):
            use_custom = (getattr(config.violation_pipeline, "orchestration_strategy", "default") == "custom")

        print("ORCHESTRATION =", config.violation_pipeline.orchestration_strategy)
        print("USE_CUSTOM =", use_custom)
        model_mappings = self._get_active_model_mapping(use_custom)
        print(f"[Pipeline] Active Runtime Model Blueprint Map: {json.dumps(model_mappings)}")

        print(
            f"[Pipeline] "
            f"face_blur.enabled={config.face_blur.enabled}, "
            f"method={config.face_blur.method}, "
            f"intensity={config.face_blur.intensity}"
        )

        work_dir = ensure_video_dir(video.id)

        run = ProcessingRun(
            video_id=video.id,
            status="running",
            stage_logs={}
        )

        db.add(run)
        video.status = "processing"
        db.commit()
        db.refresh(run)

        original_video = self._ensure_original_video_cache(video)
        current_video = original_video

        stage_logs: dict = {
            "started_at": datetime.utcnow().isoformat()
        }

        try:
            # AUDIO REMOVAL
            if config.audio_removal:
                current_video, stage_logs["audio_removal"] = (
                    self.audio.run(current_video, work_dir / "audio")
                )
                video.audio_removed_path = str(current_video)
            else:
                stage_logs["audio_removal"] = {
                    "status": "skipped",
                    "reason": "disabled"
                }

            print(config.violation_detection)
            
            # TRIPLE RIDING
            start = time.perf_counter()
            if (config.violation_detection.taskkillenabled and "triple_riding" in config.violation_detection.list_violations):
                models = model_mappings.get("triple_riding", {})
                sig = inspect.signature(self.triple_riding.run)
                
                if "vehicle_model" in sig.parameters and "person_model" in sig.parameters:
                    current_video, stage_logs["triple_riding"] = self.triple_riding.run(
                        current_video,
                        work_dir / "triple_riding",
                        video.id,
                        vehicle_model=models.get("vehicle_detection", "yolov8n.pt"),
                        person_model=models.get("person_detection", "triple_riding.pt")
                    )
                else:
                    current_video, stage_logs["triple_riding"] = self.triple_riding.run(
                        current_video, work_dir / "triple_riding", video.id
                    )
                video.processed_video_path = str(current_video)
            else:
                stage_logs["triple_riding"] = {"status": "skipped"}
            
            timings["Triple Riding"] = time.perf_counter() - start
            
            # WRONG WAY
            start = time.perf_counter()
            if (config.violation_detection.taskkillenabled and "wrong_way" in config.violation_detection.list_violations):
                models = model_mappings.get("wrong_way", {})
                print(f"[PIPELINE] Wrong Way mappings = {models}", flush=True)
                wrong_way_dir = work_dir / "wrong_way"
                wrong_way_dir.mkdir(parents=True, exist_ok=True)

                print("[PIPELINE] Running lane calibration...", flush=True)
                detect_lanes_and_save_config(current_video, wrong_way_dir / "config.json", wrong_way_dir / "detected_lanes.jpg")
                print("[PIPELINE] Lane calibration completed", flush=True)

                sig = inspect.signature(self.wrong_way.run)
                if "vehicle_model" in sig.parameters and "tracker_module" in sig.parameters:
                    print("USE_CUSTOM =", use_custom)
                    print("MODEL_MAPPINGS =", json.dumps(model_mappings, indent=2))
                    print("WRONG WAY MODELS =", models)
                    current_video, stage_logs["wrong_way"] = self.wrong_way.run(
                        current_video,
                        wrong_way_dir,
                        video.id,
                        vehicle_model=models.get("vehicle_detection", "yolov8n.pt"),
                        tracker_module=models.get("tracking", "bytetrack")
                    )
                else:
                    current_video, stage_logs["wrong_way"] = self.wrong_way.run(
                        current_video, wrong_way_dir, video.id
                    )
            else:
                stage_logs["wrong_way"] = {"status": "skipped"}
            timings["Wrong Way"] = time.perf_counter() - start
            
            # OVERSPEED
            start = time.perf_counter()
            if (config.violation_detection.taskkillenabled and "overspeed" in config.violation_detection.list_violations):
                models = model_mappings.get("overspeed", {})
                overspeed_dir = work_dir / "overspeed"
                overspeed_dir.mkdir(parents=True, exist_ok=True)

                detect_lanes_and_save_config(
                    current_video,
                    overspeed_dir / "config.json",
                    overspeed_dir / "detected_lanes.jpg"
                )

                sig = inspect.signature(self.overspeed.run)
                if "vehicle_model" in sig.parameters and "tracker_module" in sig.parameters:
                    current_video, stage_logs["overspeed"] = self.overspeed.run(
                        current_video,
                        overspeed_dir,
                        video.id,
                        overspeed_dir / "config.json",
                        vehicle_model=models.get("vehicle_detection", "yolov8n.pt"),
                        tracker_module=models.get("tracking", "bytetrack")
                    )
                else:
                    current_video, stage_logs["overspeed"] = self.overspeed.run(
                        current_video, overspeed_dir, video.id, overspeed_dir / "config.json"
                    )
            else:
                stage_logs["overspeed"] = {"status": "skipped"}
            timings["Overspeed"] = time.perf_counter() - start
            
            # NO NUMBER PLATE
            start = time.perf_counter()
            if (config.violation_detection.taskkillenabled and "no_number_plate" in config.violation_detection.list_violations):
                models = model_mappings.get("no_number_plate", {})
                no_number_plate_dir = work_dir / "no_number_plate"
                no_number_plate_dir.mkdir(parents=True, exist_ok=True)

                sig = inspect.signature(self.no_number_plate.run)
                if "plate_model" in sig.parameters and "ocr_model" in sig.parameters:
                    current_video, stage_logs["no_number_plate"] = self.no_number_plate.run(
                        current_video,
                        no_number_plate_dir,
                        video.id,
                        plate_model=models.get("plate_detection", "license_plate.pt"),
                        ocr_model=models.get("ocr", "model (1).pt")
                    )
                else:
                    current_video, stage_logs["no_number_plate"] = self.no_number_plate.run(
                        current_video, no_number_plate_dir, video.id
                    )
            else:
                stage_logs["no_number_plate"] = {"status": "skipped"}
            timings["No Number Plate"] = time.perf_counter() - start
            
            # FRAME EXTRACTION
            start = time.perf_counter()
            frames, stage_logs["frame_extraction"] = (
                self.frame_extractor.run(
                    current_video,
                    config.frame_extraction,
                    work_dir / "frames"
                )
            )
            gps_timeline = stage_logs.get("frame_extraction", {}).get("gps_timeline", [])
            db.query(VideoRoute).filter(VideoRoute.video_id == video.id).delete()

            for index, point in enumerate(gps_timeline):
                db.add(
                    VideoRoute(
                    video_id=video.id,
                    latitude=float(point["latitude"]),
                    longitude=float(point["longitude"]),
                    timestamp_seconds=float(point["timestamp"]),
                    sequence_order=index + 1,
                    )
                )

            stage_logs["frame_extraction"]["gps_points_saved"] = len(gps_timeline)
            timings["Frame Extraction"] = time.perf_counter() - start

            # FACE BLUR
            start = time.perf_counter()
            if config.face_blur.enabled:
                current_video, stage_logs["face_blur"] = (
                    self.face_blur.run(current_video, config.face_blur, work_dir / "face_blur")
                )
                video.processed_video_path = str(current_video)
            else:
                stage_logs["face_blur"] = {"status": "skipped", "reason": "disabled"}
                video.processed_video_path = str(current_video)
            timings["Face Blur"] = time.perf_counter() - start

            # CLEAN OLD FRAME RECORDS
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
            

            # OBJECT DETECTION
            start = time.perf_counter()
            detections, stage_logs["object_detection"] = (
                self.object_detector.run(frames, config.object_detection)
            )

            # GEO TAGGING
            # geo_source = original_video if config.geo_tagging.mode == "metadata" else current_video
            start = time.perf_counter()
            location, stage_logs["geo_tagging"] = (self.geotagger.resolve(current_video, config.geo_tagging, gps_timeline))
            timings["Geo Tagging"] = time.perf_counter() - start

            # PROCESSED VIDEO UPLOAD
            processed_object = upload_processed_video(video.id, current_video)
            video.processed_bucket_name = processed_object.bucket_name
            video.processed_object_key = processed_object.object_key
            video.processed_object_url = processed_object.object_url

            stage_logs["processed_video_storage"] = {
                "status": "completed",
                "bucket_name": processed_object.bucket_name,
                "object_key": processed_object.object_key,
            }

            # CLEAN OLD DETECTIONS
            db.query(Detection).filter(Detection.video_id == video.id).delete()

            # CLEAN OLD VIOLATIONS
            db.query(ProjectViolation).filter(ProjectViolation.video_id == video.id).delete()

            # SAVE DETECTIONS
            start = time.perf_counter()
            for item in detections:
                coords = self.geotagger.get_coordinate_for_timestamp(item["timestamp_seconds"],gps_timeline,)
                latitude = coords["latitude"] if coords else None
                longitude = coords["longitude"] if coords else None

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
                        location=self._build_location_value(db,latitude,longitude,),
                    )
                )
            timings["Detection Saving"] = time.perf_counter() - start

            # SAVE ALL VIOLATIONS
            start = time.perf_counter()
            for stage in ["triple_riding", "wrong_way", "overspeed", "no_number_plate"]:

                if stage not in stage_logs:
                    continue
                print(f"[DEBUG] Stage = {stage}")
                print(stage_logs.get(stage))
                violations = stage_logs[stage].get("violations", [])
                print(f"[DEBUG] Number of violations = {len(violations)}")

                for item in violations:
                    print(f"[DB INSERT] {item}")
                    coords = self.geotagger.get_coordinate_for_timestamp(item["timestamp_seconds"],gps_timeline,)
                    latitude = coords["latitude"] if coords else None
                    longitude = coords["longitude"] if coords else None

                    db.add(
                        ProjectViolation(
                            video_id=video.id,
                            timestamp_seconds=item["timestamp_seconds"],
                            plate_number=item.get("plate_number"),
                            violation_type=item["violation_type"],
                            confidence=item["confidence"],
                            image_url=item["image_url"],
                            latitude=latitude,
                            longitude=longitude,
                            location=self._build_location_value(db, latitude, longitude)
                        )
                    )
    
                stage_logs[stage]["violations_saved"] = len(violations)
                timings[f"{stage} Violations"] = time.perf_counter() - start
                
            run.status = "completed"
            video.status = "processed"
            # REGISTRY UPSERT
            start = time.perf_counter()
            try:
                registry_upsert_query = text("""
                    INSERT INTO public.video_registry
                        (processed_date, video_id)
                    VALUES
                        (:processed_date, :video_id)
                    ON CONFLICT (video_id)
                    DO UPDATE
                    SET processed_date = EXCLUDED.processed_date;
                """)

                db.execute(
                    registry_upsert_query,
                    {
                        "processed_date": date.today(),
                        "video_id": video.id
                    }
                )

                print(f"🔗 [Pipeline] Successfully synced Video ID {video.id} to public.video_registry.")

            except Exception as registry_err:
                db.rollback()   # IMPORTANT
                print(f"⚠️ [Pipeline] Warning: Could not log processing marker to video_registry: {registry_err}")

            run.status = "completed"
            video.status = "processed"
            timings["Registry Upsert"] = time.perf_counter() - start

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
            start = time.perf_counter()
            db.commit()
            timings["Database Commit"] = time.perf_counter() - start

        total_time = time.perf_counter() - total_start

        print("\n========== PIPELINE TIMING ==========")
        for k, v in timings.items():
            print(f"{k:<25}: {v:.2f} sec")
        print("-------------------------------------")
        print(f"TOTAL PIPELINE        : {total_time:.2f} sec")
        print("=====================================")
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