from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes import router
from backend.app.core.config import get_settings
from backend.app.db.bootstrap import apply_schema_compatibility
from backend.app.db.database import Base, engine
import backend.app.models.video  # noqa: F401
from backend.app.services.storage import ensure_buckets


settings = get_settings()
app = FastAPI(title=settings.app_title)
app.include_router(router)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
def on_startup() -> None:
    Base.metadata.create_all(bind=engine)
    apply_schema_compatibility()
    ensure_buckets()


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(static_dir / "index.html")
