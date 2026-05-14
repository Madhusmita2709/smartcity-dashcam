# Video-to-Map Analytics Web Application

Config-driven prototype that ingests one or more videos, stores a processing configuration JSON per video, runs a modular CPU-friendly pipeline, persists media artifacts in MinIO, and visualizes geo-tagged detections on a Leaflet map with a heatmap layer.

## Architecture

- `backend/app/main.py`: FastAPI entry point, static frontend hosting, and startup table creation.
- `backend/app/api/routes.py`: Upload, process, config, results, and heatmap endpoints.
- `backend/app/schemas/config.py`: Pydantic configuration model shared across API and pipeline.
- `backend/app/services/pipeline.py`: Dynamic orchestration layer.
- `backend/app/services/processors/*`: Stage-specific processors for audio removal, face blur, frame extraction, detection, and geo-tagging.
- `backend/app/static/*`: Plain HTML, CSS, and JavaScript frontend with upload/config controls, processing actions, Leaflet map, and live heatmap refresh.
- `database/schema.sql`: PostgreSQL + PostGIS schema for metadata and MinIO object references.
- `examples/processing-config.json`: Example per-video configuration payload.

## Example Config JSON

```json
{
  "audio_removal": true,
  "face_blur": {
    "enabled": true,
    "method": "gaussian",
    "intensity": 25
  },
  "frame_extraction": {
    "method": "interval",
    "value": 5,
    "motion_threshold": 25
  },
  "object_detection": {
    "model": "yolov8n",
    "classes": ["person", "car", "truck"],
    "confidence_threshold": 0.25
  },
  "geo_tagging": {
    "mode": "manual",
    "latitude": 20.2961,
    "longitude": 85.8245
  }
}
```

## End-to-End Workflow

1. `POST /upload` accepts one or more videos and either a shared `config_json` or per-file `config_json_list`.
2. The backend validates the config, stores it in `videos.config_json`, uploads the original video to a private MinIO bucket, and stores object metadata in PostgreSQL.
3. `POST /process/{video_id}` loads the stored config, downloads the original from MinIO if needed, conditionally runs audio removal, face blur, frame extraction, object detection, and geo-tagging.
4. The final preprocessed video is uploaded to a separate private MinIO bucket, extracted frames are uploaded to a separate private images bucket, and PostgreSQL stores all object keys and URLs.
5. Detections are stored in Postgres/PostGIS and served through `GET /results/{video_id}` and `GET /heatmap`.
6. The frontend shows job status, stage logs, detection summaries, map markers, and a filtered heatmap.

## Run Locally

1. Start PostGIS and MinIO with Docker.
2. Copy `.env.example` to `.env` and update the credentials or endpoints if needed.
3. Install dependencies with `pip install -r requirements.txt`.
4. Start the server with `uvicorn backend.app.main:app --reload`.
5. Open [http://localhost:8000](http://localhost:8000).

## One-Command PostGIS

If Docker Desktop is available, you can bring up PostgreSQL/PostGIS with one command:

```powershell
.\scripts\start-postgis.ps1
```

This uses [docker-compose.yml](D:\DashCam\docker-compose.yml) and starts a `postgis/postgis:17-3.5` container on `localhost:5432` with:

- database: `dashcam`
- user: `postgres`
- password: `postgres`

Stop it with:

```powershell
.\scripts\stop-postgis.ps1
```

## One-Command MinIO

Bring up MinIO with:

```powershell
.\scripts\start-minio.ps1
```

This starts a private local MinIO service on:

- API: [http://127.0.0.1:19000](http://127.0.0.1:19000)
- Console: [http://127.0.0.1:19001](http://127.0.0.1:19001)

Default credentials:

- access key: `minioadmin`
- secret key: `minioadmin123`

The application uses three separate private buckets:

- `dashcam-original-videos`
- `dashcam-processed-videos`
- `dashcam-images`

Stop it with:

```powershell
.\scripts\stop-minio.ps1
```

To start PostgreSQL, MinIO, and the FastAPI app together:

```powershell
.\scripts\start-all.ps1
```

Optional flags:

```powershell
.\scripts\start-all.ps1 -Port 8010
.\scripts\start-all.ps1 -NoReload
.\scripts\start-all.ps1 -BindHost 0.0.0.0 -Port 8010
```

## Notes

- The pipeline is CPU-first and degrades gracefully when `ffmpeg`, `ffprobe`, or YOLO runtime dependencies are unavailable.
- `media/<video_id>/` is now treated as a temporary local processing workspace; the durable originals, processed videos, and frame images are stored in MinIO.
- PostgreSQL stores processing metadata, geospatial detections, and MinIO object keys/URLs.
- The current prototype processes synchronously to keep the single-machine architecture simple and easy to extend later with a job queue.
