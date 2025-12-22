from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))


@dataclass(frozen=True)
class TrackItem:
    track_id: str
    audio_path: Path
    lyrics_path: Path | None
    label: str | None


def infer_label_from_filename(path: Path) -> str | None:
    """Infer a coarse label from GTZAN-like filenames: genre.00000.au."""
    stem = path.name
    if "." in stem:
        return stem.split(".", 1)[0]
    return None
