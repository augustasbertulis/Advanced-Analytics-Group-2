# paths.py
from pathlib import Path

# Root of the project (the folder where paths.py lives)
PROJECT_ROOT = Path(__file__).parent

# Data folder (gitignored)
DATA_DIR = PROJECT_ROOT / "data"

# Subfolders
RAW_DATA_DIR = DATA_DIR / "raw data"
PROCESSED_DATA_DIR = DATA_DIR / "processed data"

# Make sure the folders exist (optional, but nice for avoiding errors)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)