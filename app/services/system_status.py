import os
import json
from datetime import datetime

UPLOAD_DIR = "uploads"
STATUS_PATH = os.path.join(UPLOAD_DIR, "system_status.json")
os.makedirs(UPLOAD_DIR, exist_ok=True)

def _now_date():
    return datetime.now().strftime("%Y-%m-%d")

def read_status() -> dict:
    if not os.path.exists(STATUS_PATH):
        return {
            "lastUploadDate": None,
            "lastUploadedFilename": None,
            "fishCount": 0,
        }
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_status(status: dict) -> None:
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)

def update_last_upload(filename: str) -> dict:
    status = read_status()
    status["lastUploadDate"] = datetime.now().strftime("%Y-%m-%d")

    status["lastUploadedFilename"] = filename
    write_status(status)
    return status

def update_fish_count(fish_count: int) -> dict:
    status = read_status()
    status["fishCount"] = int(fish_count)
    write_status(status)
    return status
