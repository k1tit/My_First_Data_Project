"""Скачать файлы соревнования Home Credit через kagglehub (нужен Kaggle API token)."""
import shutil
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

path = Path(kagglehub.competition_download("home-credit-default-risk"))
print("Cache:", path)

DATA_DIR.mkdir(exist_ok=True)
for src in path.glob("*.csv"):
    dst = DATA_DIR / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        print("Copied ->", dst)
    else:
        print("Already in data/:", dst.name)

print("Done. Run: python scripts/dataset_quality.py")