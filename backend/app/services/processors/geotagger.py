from pathlib import Path
from backend.app.schemas.config import GeoTaggingConfig


class GeoTaggingProcessor:
    def resolve(self,source: Path,config: GeoTaggingConfig,ocr_gps_timeline: list | None = None,) -> tuple[dict | None, dict]:
        if config.mode == "manual":
            location = {"latitude": config.latitude, "longitude": config.longitude}
            return location, {"status": "completed", "mode": "manual", **location}
        # OCR GPS timeline has highest priority for dashboard routing
        if ocr_gps_timeline:
            base_coords = {
                "latitude": ocr_gps_timeline[0]["latitude"],
                "longitude": ocr_gps_timeline[0]["longitude"],
            }

            return base_coords, {
                "status": "completed",
                "mode": "ocr_timeline",
                "route": ocr_gps_timeline,
                **base_coords,
            }

        return None, {
            "status": "skipped",
            "mode": "ocr_timeline",
            "message": "Metadata GPS not found; detections will be stored without coordinates.",
        }
    
    def get_coordinate_for_timestamp(self,timestamp_seconds: float,gps_timeline: list,) -> dict | None:
        """
        Return the nearest OCR GPS coordinate for a given timestamp.
        """
        if not gps_timeline:
            return None

        best_match = min(
            gps_timeline,
            key=lambda x: abs(
                x.get("timestamp_seconds", x.get("timestamp", 0.0)) - timestamp_seconds),
        )

        return {
            "latitude": best_match.get("latitude"),
            "longitude": best_match.get("longitude"),
        }
