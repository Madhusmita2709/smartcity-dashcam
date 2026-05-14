from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.schemas.config import ObjectDetectionConfig

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

settings = get_settings()


class ObjectDetectionProcessor:
    def __init__(self) -> None:
        self._models: dict[str, object] = {}

    def _load_model(self, model_name: str):
        if model_name not in self._models:
            if YOLO is None:
                self._models[model_name] = None
            else:
                try:
                    model_path = settings.model_cache_dir / f"{model_name}.pt"
                    self._models[model_name] = YOLO(str(model_path))
                except Exception:
                    self._models[model_name] = None
        return self._models[model_name]

    def run(self, frames: list[dict], config: ObjectDetectionConfig) -> tuple[list[dict], dict]:
        allowed = {item.lower() for item in config.classes}
        model = self._load_model(config.model)
        detections: list[dict] = []

        if model is None:
            return detections, {
                "status": "skipped",
                "message": "YOLO model unavailable in the current environment.",
                "classes": sorted(allowed),
                "model": config.model,
            }

        for frame in frames:
            image_path = Path(frame["path"])
            result = model.predict(
                source=str(image_path),
                conf=config.confidence_threshold,
                verbose=False,
                device="cpu",
            )[0]

            label_lookup = result.names
            for box in result.boxes:
                cls_idx = int(box.cls[0].item())
                label = str(label_lookup.get(cls_idx, cls_idx)).lower()
                if label not in allowed:
                    continue

                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]
                detections.append(
                    {
                        "frame_index": frame["frame_index"],
                        "timestamp_seconds": frame["timestamp_seconds"],
                        "object_class": label,
                        "confidence": float(box.conf[0].item()),
                        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                )

        return detections, {
            "status": "completed",
            "model": config.model,
            "classes": sorted(allowed),
            "detections": len(detections),
        }
