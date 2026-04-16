from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REF_DIR = DATA_DIR / "reference"
OUT_DIR = DATA_DIR / "output"
CACHE_DIR = DATA_DIR / "cache"

XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 22050
TARGET_DBFS = -20.0
MAX_CHARS_PER_CHUNK = 250
INTER_CHUNK_SILENCE_MS = 150
