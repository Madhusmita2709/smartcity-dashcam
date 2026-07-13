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
from backend.app.services.processors.helmet import HelmetDetector 
# FrameStreamProcessor is frozen and imported here
from backend.app.services.processors.frame_extractor import FrameStreamProcessor
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
        self.no_helmet = HelmetDetector()  
        self.frame_stream = FrameStreamProcessor()  
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
                
        # Fix #3: Factory fallbacks standardized cleanly on full tracker filenames
        return {
            "triple_riding": {"vehicle_detection": "yolov8n.pt", "person_detection": "triple_riding.pt"},
            "wrong_way": {"vehicle_detection": "yolov8n.pt", "tracking": "bytetrack.yaml"},
            "no_number_plate": {"plate_detection": "license_plate.pt", "ocr": "model (1).pt"},
            "overspeed": {"vehicle_detection": "yolov8n.pt", "tracking": "bytetrack.yaml"},
            "no_helmet": {"vehicle_detection": "yolov8n.pt", "helmet_detection": "cnn_helmet_detection(best).pt", "tracking": "botsort.yaml"}
        }

    def run(self, db: Session, video: Video, config: ProcessingConfig) -> dict:
        total_start = time.perf_counter()
        timings = {}
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
            
            # =================================================================
            # 🚀 NEW SINGLE-PASS GENERATOR ENGINE LOOP (PHASE 2 MASTER PASS)
            # =================================================================
            loop_start = time.perf_counter()
            frames_dir = work_dir / "frames"
            
            nnp_enabled = (config.violation_detection.taskkillenabled and "no_number_plate" in config.violation_detection.list_violations)
            wrong_way_enabled = (config.violation_detection.taskkillenabled and "wrong_way" in config.violation_detection.list_violations)
            triple_enabled = (config.violation_detection.taskkillenabled and "triple_riding" in config.violation_detection.list_violations)
            overspeed_enabled = (config.violation_detection.taskkillenabled and "overspeed" in config.violation_detection.list_violations)
            helmet_enabled = (config.violation_detection.taskkillenabled and "no_helmet" in config.violation_detection.list_violations)

            # --- PRE-LOOP PREPROCESSING: Isolated Lane Detection Calibration ---
            lane_detector_dir = work_dir / "lane_detector"
            if wrong_way_enabled or overspeed_enabled:
                lane_detector_dir.mkdir(parents=True, exist_ok=True)
                print("[PIPELINE] Running independent lane calibration...", flush=True)
                detect_lanes_and_save_config(
                    current_video,
                    lane_detector_dir / "config.json",
                    lane_detector_dir / "detected_lanes.jpg"
                )
                print("[PIPELINE] Lane calibration completed", flush=True)

            # --- PRE-LOOP SESSION INITIALIZATIONS (PRESERVES UI MODEL CONFIGS) ---
            if nnp_enabled:
                self.no_number_plate.setup_session(work_dir / "no_number_plate", video.id)
            else:
                stage_logs["no_number_plate"] = {"status": "skipped"}

            # Fix #2: Standardized default runtime fallbacks to use full tracker extensions
            if wrong_way_enabled:
                ww_models = model_mappings.get("wrong_way", {})
                self.wrong_way.setup_session(
                    output_dir=work_dir / "wrong_way",
                    video_id=video.id,
                    config_path=lane_detector_dir / "config.json",
                    vehicle_model=ww_models.get("vehicle_detection", "yolov8n.pt"),
                    tracker_module=ww_models.get("tracking", "bytetrack.yaml")
                )
            else:
                stage_logs["wrong_way"] = {"status": "skipped"}

            if triple_enabled:
                tr_models = model_mappings.get("triple_riding", {})
                self.triple_riding.setup_session(
                    output_dir=work_dir / "triple_riding",
                    video_id=video.id,
                    person_model=tr_models.get("person_detection", "triple_riding.pt"),
                    vehicle_model=tr_models.get("vehicle_detection", "yolov8n.pt"),
                    fps=25.0  
                )
            else:
                stage_logs["triple_riding"] = {"status": "skipped"}

            if overspeed_enabled:
                os_models = model_mappings.get("overspeed", {})
                self.overspeed.setup_session(
                    output_dir=work_dir / "overspeed",
                    video_id=video.id,
                    config_path=lane_detector_dir / "config.json",
                    vehicle_model=os_models.get("vehicle_detection", "yolov8n.pt"),
                    tracker_module=os_models.get("tracking", "bytetrack.yaml"),
                )
            else:
                stage_logs["overspeed"] = {"status": "skipped"}  
                
            if helmet_enabled:
                helmet_models = model_mappings.get("no_helmet", {})
                self.no_helmet.setup_session(
                    output_dir=work_dir / "no_helmet",
                    video_id=video.id,
                    pipeline_config=config.violation_detection.model_dump(),  
                    vehicle_model=helmet_models.get("vehicle_detection", "yolov8n.pt"),
                    helmet_model=helmet_models.get("helmet_detection", "cnn_helmet_detection(best).pt"),
                    tracker_module=helmet_models.get("tracking", "botsort.yaml")
                )
            else:
                stage_logs["no_helmet"] = {"status": "skipped"}

            # High-precision individual accumulation buffers
            nnp_total_time = 0.0
            ww_total_time = 0.0
            tr_total_time = 0.0
            os_total_time = 0.0
            helmet_total_time = 0.0

            # --- RUNTIME SINGLE-PASS STREAM CONTEXT CONSUMPTION ---
            for context in self.frame_stream.run(current_video, config.frame_extraction, frames_dir):
                # 1. License Plate Empty/Blank Checker
                if nnp_enabled:
                    nnp_start = time.perf_counter()
                    self.no_number_plate.process_frame(context)
                    nnp_total_time += (time.perf_counter() - nnp_start)
                
                # 2. Kinematic Directional Wrong Way Module Pass
                if wrong_way_enabled:
                    ww_start = time.perf_counter()
                    self.wrong_way.process_frame(context)
                    ww_total_time += (time.perf_counter() - ww_start)
                
                # 3. Converted Passive Triple Riding Module Pass
                if triple_enabled:
                    tr_start = time.perf_counter()
                    self.triple_riding.process_frame(context)
                    tr_total_time += (time.perf_counter() - tr_start)

                # 4. Overspeed Detection Module Pass
                if overspeed_enabled:
                    os_start = time.perf_counter()
                    self.overspeed.process_frame(context)
                    os_total_time += (time.perf_counter() - os_start)

                # 5. Modular Plug-in No Helmet Pass
                if helmet_enabled:
                    helmet_start = time.perf_counter()
                    self.no_helmet.process_frame(context)
                    helmet_total_time += (time.perf_counter() - helmet_start)

            # --- POST-STREAM EXTRACTION MANIFEST HARVESTING ---
            if nnp_enabled:
                stage_logs["no_number_plate"] = self.no_number_plate.finish()

            if wrong_way_enabled:
                stage_logs["wrong_way"] = self.wrong_way.finish()

            if triple_enabled:
                stage_logs["triple_riding"] = self.triple_riding.finish()

            if overspeed_enabled:
                stage_logs["overspeed"] = self.overspeed.finish()

            if helmet_enabled:
                stage_logs["no_helmet"] = self.no_helmet.finish()
                
            # Extract historical sampled frames summary manifests
            sampled_frames, stage_logs["frame_extraction"] = self.frame_stream.get_summary(config.frame_extraction)
            
            # Isolated, accurate loop performance tracking profiling metrics
            timings["Frame Stream Engine Loop"] = time.perf_counter() - loop_start
            timings["No Number Plate (Inference)"] = nnp_total_time
            timings["Wrong Way (Inference)"] = ww_total_time
            timings["Triple Riding (Inference)"] = tr_total_time
            timings["Overspeed (Inference)"] = os_total_time
            timings["No Helmet (Inference)"] = helmet_total_time

            # =================================================================
            # POST-LOOP COMPILATION & STORAGE DISPATCH (UNCHANGED DEPENDENCIES)
            # =================================================================
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

            # FACE BLUR (Legacy Processing Pass)
            start = time.perf_counter()
            if config.face_blur.enabled:
                current_video, stage_logs["face_blur"] = (
                    self.face_blur.run(current_video, config.face_blur, work_dir / "face_blur")
                )
                video.processed_video_path = str(current_video)
            else:
                stage_logs["face_blur"] = {"status": "skipped", "reason": "disabled"}
                video.processed_video_path = str(current_video)
            timings["Face Blur (Legacy Pass)"] = time.perf_counter() - start

            # CLEAN OLD FRAME RECORDS
            db.query(FrameImage).filter(FrameImage.video_id == video.id).delete()

            frame_records = []
            for frame in sampled_frames:
                stored_frame = upload_frame_image(video.id, frame["frame_index"], frames_dir / f"frame_{frame['frame_index']:06d}.jpg")
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
            detections, stage_logs["object_detection"] = (
                self.object_detector.run(sampled_frames, config.object_detection)
            )

            # GEO TAGGING
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

            # CLEAN OLD DETECTIONS & VIOLATIONS FROM DATABASE
            db.query(Detection).filter(Detection.video_id == video.id).delete()
            db.query(ProjectViolation).filter(ProjectViolation.video_id == video.id).delete()

            # SAVE DETECTIONS
            start = time.perf_counter()
            for item in detections:
                coords = self.geotagger.get_coordinate_for_timestamp(item["timestamp_seconds"], gps_timeline)
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
                        location=self._build_location_value(db, latitude, longitude),
                    )
                )
            timings["Detection Saving"] = time.perf_counter() - start

            # SAVE ALL VIOLATIONS
            start = time.perf_counter()
            for stage in ["triple_riding", "wrong_way", "overspeed", "no_number_plate", "no_helmet"]:
                if stage not in stage_logs:
                    continue
                
                violations = stage_logs[stage].get("violations", [])
                for item in violations:
                    coords = self.geotagger.get_coordinate_for_timestamp(item["timestamp_seconds"], gps_timeline)
                    latitude = coords["latitude"] if coords else None
                    longitude = coords["longitude"] if coords else None

                    # Fix #1: Preserves explicit database NULL support; strips hardcoded string mapping defaults
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
                timings[f"{stage} Violations Saving"] = time.perf_counter() - start
                
            run.status = "completed"
            video.status = "processed"
            
            # REGISTRY UPSERT MARKER
            start = time.perf_counter()
            try:
                registry_upsert_query = text("""
                    INSERT INTO public.video_registry (processed_date, video_id)
                    VALUES (:processed_date, :video_id)
                    ON CONFLICT (video_id) DO UPDATE
                    SET processed_date = EXCLUDED.processed_date;
                """)

                db.execute(registry_upsert_query, {"processed_date": date.today(), "video_id": video.id})
                print(f"🔗 [Pipeline] Successfully synced Video ID {video.id} to public.video_registry.")
            except Exception as registry_err:
                db.rollback()
                print(f"⚠️ [Pipeline] Warning: Could not log processing marker to video_registry: {registry_err}")

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
            print(f"{k:<30}: {v:.2f} sec")
        print("-------------------------------------")
        print(f"TOTAL PIPELINE                : {total_time:.2f} sec")
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