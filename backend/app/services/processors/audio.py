from pathlib import Path
import shutil
import subprocess


class AudioRemovalProcessor:
    """Removes audio from a video when ffmpeg is available."""

    def run(self, source: Path, output_dir: Path) -> tuple[Path, dict]:
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / f"{source.stem}_no_audio{source.suffix}"
        command = [
            "ffmpeg",
            "-y",
            "-i",
            str(source),
            "-c:v",
            "copy",
            "-an",
            str(target),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True)
            return target, {"status": "completed", "method": "ffmpeg", "path": str(target)}
        except (FileNotFoundError, subprocess.CalledProcessError):
            shutil.copy2(source, target)
            return target, {
                "status": "skipped_with_fallback",
                "method": "copy",
                "path": str(target),
                "message": "ffmpeg unavailable; copied source video instead.",
            }
