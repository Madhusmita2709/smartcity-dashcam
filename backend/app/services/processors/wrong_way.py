from pathlib import Path


class WrongWayDetector:

    def __init__(self):
        print("[WRONG WAY INIT]", flush=True)

    def run(self, input_path, output_dir, video_id):

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[WRONG WAY DETECTOR STARTED]", flush=True)

        return input_path, {
            "status": "completed",
            "violations": []
        }