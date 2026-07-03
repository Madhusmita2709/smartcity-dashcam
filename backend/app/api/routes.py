from pathlib import Path
import json
MODELS_DIR = Path(__file__).resolve().parents[1] / "services" / "models"

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, Body
from pydantic import ValidationError
from sqlalchemy import and_, select
from sqlalchemy.orm import Session

from backend.app.db.database import get_db
from backend.app.models.video import Detection, FrameImage, Video
from backend.app.schemas.api import (
    DetectionResponse,
    FrameImageResponse,
    HeatmapPoint,
    ProcessResponse,
    ResultsResponse,
    UploadResponse,
    UploadResponseItem,
)
from backend.app.schemas.config import (ProcessingConfig,CustomMappingRequest,)
from backend.app.services.pipeline import VideoProcessingPipeline
from backend.app.services.storage import save_upload_to_disk, upload_original_video, write_json


router = APIRouter()
pipeline = VideoProcessingPipeline()


def _parse_configs(
    config_json: str | None,
    config_json_list: list[str] | None,
    file_count: int,
) -> list[ProcessingConfig]:
    if config_json_list:
        if len(config_json_list) != file_count:
            raise HTTPException(status_code=400, detail="config_json_list length must match uploaded files.")
        try:
            return [ProcessingConfig.model_validate_json(item) for item in config_json_list]
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc

    if config_json:
        try:
            config = ProcessingConfig.model_validate_json(config_json)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return [config for _ in range(file_count)]

    default_config = ProcessingConfig()
    return [default_config for _ in range(file_count)]


@router.post("/upload", response_model=UploadResponse)
async def upload_videos(
    files: list[UploadFile] = File(...),
    config_json: str | None = Form(default=None),
    config_json_list: list[str] | None = Form(default=None),
    db: Session = Depends(get_db),
):
    configs = _parse_configs(config_json, config_json_list, len(files))
    response_items: list[UploadResponseItem] = []

    for upload, config in zip(files, configs, strict=True):
        video = Video(filename=upload.filename or "video.mp4", stored_path="pending", config_json=config.model_dump())
        db.add(video)
        db.commit()
        db.refresh(video)

        stored_path = save_upload_to_disk(video.id, upload)
        original_object = upload_original_video(video.id, video.filename, stored_path)
        video.stored_path = str(stored_path)
        video.original_bucket_name = original_object.bucket_name
        video.original_object_key = original_object.object_key
        video.original_object_url = original_object.object_url
        write_json(Path(stored_path).parent / "config.json", config.model_dump())
        db.commit()

        response_items.append(
            UploadResponseItem(
                video_id=video.id,
                filename=video.filename,
                status=video.status,
                config=config,
                original_video_url=video.original_object_url,
            )
        )

    return UploadResponse(items=response_items)


@router.post("/process/{video_id}", response_model=ProcessResponse)
def process_video(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")

    config = ProcessingConfig.model_validate(video.config_json)
    stage_logs = pipeline.run(db, video, config)
    db.refresh(video)
    return ProcessResponse(video_id=video.id, status=video.status, stages=stage_logs)


@router.get("/config/{video_id}", response_model=ProcessingConfig)
def get_config(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")
    return ProcessingConfig.model_validate(video.config_json)


@router.get("/results/{video_id}", response_model=ResultsResponse)
def get_results(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")

    detections = db.scalars(
        select(Detection).where(Detection.video_id == video_id).order_by(Detection.timestamp_seconds.asc())
    ).all()
    frame_images = db.scalars(
        select(FrameImage).where(FrameImage.video_id == video_id).order_by(FrameImage.frame_index.asc())
    ).all()
    return ResultsResponse(
        video_id=video.id,
        filename=video.filename,
        status=video.status,
        config=ProcessingConfig.model_validate(video.config_json),
        original_video_url=video.original_object_url,
        processed_video_url=video.processed_object_url,
        detections=[
            DetectionResponse(
                id=item.id,
                frame_index=item.frame_index,
                timestamp_seconds=item.timestamp_seconds,
                object_class=item.object_class,
                confidence=item.confidence,
                bbox=item.bbox,
                latitude=item.latitude,
                longitude=item.longitude,
                source_mode=item.source_mode,
                extracted_at=item.extracted_at,
            )
            for item in detections
        ],
        frame_images=[
            FrameImageResponse(
                id=item.id,
                frame_index=item.frame_index,
                #timestamp_seconds=item.timestamp_seconds,
                #bucket_name=item.bucket_name,
                #object_key=item.object_key,
                #object_url=item.object_url,
                #content_type=item.content_type,
                #size_bytes=item.size_bytes,
                frame_number=item.frame_number,
                image_path=item.image_path,
                video_id=item.video_id,
            )
            for item in frame_images
        ],
    )


@router.get("/heatmap", response_model=list[HeatmapPoint])
def get_heatmap(
    object_type: str | None = Query(default=None),
    start_time: float | None = Query(default=None, ge=0),
    end_time: float | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
):
    filters = [Detection.latitude.is_not(None), Detection.longitude.is_not(None)]
    if object_type:
        filters.append(Detection.object_class == object_type.lower())
    if start_time is not None:
        filters.append(Detection.timestamp_seconds >= start_time)
    if end_time is not None:
        filters.append(Detection.timestamp_seconds <= end_time)

    rows = db.scalars(select(Detection).where(and_(*filters))).all()
    return [
        HeatmapPoint(
            latitude=row.latitude,
            longitude=row.longitude,
            intensity=max(row.confidence, 0.1),
            object_class=row.object_class,
            timestamp_seconds=row.timestamp_seconds,
            video_id=row.video_id,
        )
        for row in rows
        if row.latitude is not None and row.longitude is not None
    ]

@router.get("/violations/{video_id}")
def get_violations(video_id: int, db: Session = Depends(get_db)):
    video = db.get(Video, video_id)
    if not video:
        raise HTTPException(status_code=404, detail="Video not found.")

    detections = db.scalars(
        select(Detection)
        .where(Detection.video_id == video_id)
        .order_by(Detection.frame_index.asc())
    ).all()

    from collections import defaultdict
    frames = defaultdict(list)
    for det in detections:
        frames[det.frame_index].append(det)

    violations = []
    for frame_index, dets in frames.items():
        persons = [d for d in dets if d.object_class == "person"]
        if len(persons) >= 3:
            violations.append({
                "violation_type": "triple_riding",
                "frame_index": frame_index,
                "timestamp_seconds": dets[0].timestamp_seconds,
                "person_count": len(persons),
                "confidence": max(d.confidence for d in persons),
                "checked": False
            })

    return {"video_id": video_id, "violations": violations}
# ==========================================================
# AI MODEL EXECUTION ENGINE
# ==========================================================

@router.get("/api/models")
def get_available_models():
    models = []

    if MODELS_DIR.exists():
        for file in MODELS_DIR.iterdir():
            if file.suffix == ".pt":
                models.append(file.name)

    return {
        "models": sorted(models)
    }


@router.get("/api/violations")
def get_available_violations():
    return {
        "violations": [
            "triple_riding",
            "wrong_way",
            "no_number_plate",
            "overspeed"
        ]
    }


@router.get("/api/default-mapping")
def get_default_engine_mappings():

    mapping_file = MODELS_DIR.parent / "default_mapping.json"

    if mapping_file.exists():
        try:
            with open(mapping_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass

    return {
        "triple_riding": {
            "name": "Triple Riding",
            "tasks": [
                {
                    "id": "vehicle_detection",
                    "name": "Vehicle Detection",
                    "type": "model",
                    "default": "yolov8n.pt"
                },
                {
                    "id": "person_detection",
                    "name": "Person Detection",
                    "type": "model",
                    "default": "triple_riding.pt"
                }
            ]
        },

        "wrong_way": {
            "name": "Wrong Way",
            "tasks": [
                {
                    "id": "vehicle_detection",
                    "name": "Vehicle Detection",
                    "type": "model",
                    "default": "yolov8n.pt"
                },
                {
                    "id": "tracking",
                    "name": "Tracking",
                    "type": "execution_module",
                    "default": "bytetrack",
                    "display_name": "ByteTrack"
                }
            ]
        },

        "no_number_plate": {
            "name": "No Number Plate",
            "tasks": [
                {
                    "id": "plate_detection",
                    "name": "Plate Detection",
                    "type": "model",
                    "default": "license_plate.pt"
                },
                {
                    "id": "ocr",
                    "name": "OCR",
                    "type": "model",
                    "default": "model (1).pt"
                }
            ]
        },

        "overspeed": {
            "name": "Overspeeding",
            "tasks": [
                {
                    "id": "vehicle_detection",
                    "name": "Vehicle Detection",
                    "type": "model",
                    "default": "yolov8n.pt"
                },
                {
                    "id": "tracking",
                    "name": "Tracking",
                    "type": "execution_module",
                    "default": "bytetrack",
                    "display_name": "ByteTrack"
                }
            ]
        }
    }


@router.post("/api/custom-mapping")
def save_custom_override_configuration(
    payload: CustomMappingRequest
):

    mapping_file = MODELS_DIR.parent / "default_mapping.json"

    if mapping_file.exists():
        with open(mapping_file, "r", encoding="utf-8") as f:
            current = json.load(f)
    else:
        current = get_default_engine_mappings()

    if payload.violation not in current:
        raise HTTPException(
            status_code=404,
            detail="Violation not found."
        )

    for task in current[payload.violation]["tasks"]:

        if task["id"] in payload.overrides:
            task["default"] = payload.overrides[task["id"]]

    with open(mapping_file, "w", encoding="utf-8") as f:
        json.dump(current, f, indent=2)

    return {
        "status": "success"
    }