import json
import subprocess
from pathlib import Path

from backend.app.schemas.config import GeoTaggingConfig


class GeoTaggingProcessor:
    def resolve(self, source: Path, config: GeoTaggingConfig) -> tuple[dict | None, dict]:
        if config.mode == "manual":
            location = {"latitude": config.latitude, "longitude": config.longitude}
            return location, {"status": "completed", "mode": "manual", **location}

        command = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", str(source)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            payload = json.loads(result.stdout or "{}")
            tags = payload.get("format", {}).get("tags", {})
            coords = self._parse_metadata_coordinates(tags)
            if coords:
                return coords, {"status": "completed", "mode": "metadata", **coords}
        except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
            pass

        return None, {
            "status": "skipped",
            "mode": "metadata",
            "message": "Metadata GPS not found; detections will be stored without coordinates.",
        }

    def _parse_metadata_coordinates(self, tags: dict) -> dict | None:
        for key in ("location", "com.apple.quicktime.location.ISO6709", "GPSCoordinates"):
            value = tags.get(key)
            if not value:
                continue

            if key == "GPSCoordinates" and "," in value:
                lat, lon = value.split(",", maxsplit=1)
                try:
                    return {"latitude": float(lat), "longitude": float(lon)}
                except ValueError:
                    continue

            cleaned = value.strip().replace("/", "")
            if cleaned.startswith(("+", "-")) and len(cleaned) > 8:
                midpoint = max(cleaned.rfind("+", 1), cleaned.rfind("-", 1))
                if midpoint > 0:
                    try:
                        return {
                            "latitude": float(cleaned[:midpoint]),
                            "longitude": float(cleaned[midpoint:]),
                        }
                    except ValueError:
                        continue

        return None
