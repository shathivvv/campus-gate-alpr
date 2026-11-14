"""All tunable settings in one place. Override any of them with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Detection (YOLO) ---
MODEL_PATH = os.getenv("ALPR_MODEL", str(BASE_DIR / "models" / "plate_yolo.pt"))
DET_CONF = float(os.getenv("ALPR_DET_CONF", "0.4"))
# 640 keeps small, far-away plates detectable in recorded footage; drop to 320 for speed.
DET_IMGSZ = int(os.getenv("ALPR_DET_IMGSZ", "640"))

# --- OCR (PaddleOCR) ---
OCR_THREADS = int(os.getenv("ALPR_OCR_THREADS", "2"))
OCR_WIDTH = int(os.getenv("ALPR_OCR_WIDTH", "256"))  # plate crops are resized to this width

# --- Voting: a plate is logged only after MIN_VOTES matching reads within VOTE_WINDOW seconds ---
MIN_VOTES = int(os.getenv("ALPR_MIN_VOTES", "3"))
VOTE_WINDOW = float(os.getenv("ALPR_VOTE_WINDOW", "4.0"))
LOG_COOLDOWN = float(os.getenv("ALPR_LOG_COOLDOWN", "15.0"))  # don't re-log the same plate within this

# --- Database (MySQL) ---
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "umpsa_alpr"),
}

# --- Dashboard ---
FACULTY_NAME = os.getenv("ALPR_FACULTY_NAME", "Fakulti Komputeran")
WELCOME_SECONDS = float(os.getenv("ALPR_WELCOME_SECONDS", "8"))  # how long the welcome banner stays up
