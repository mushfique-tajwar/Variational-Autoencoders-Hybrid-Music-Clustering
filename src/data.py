from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .utils import TrackItem, infer_label_from_filename


AUDIO_EXTS = {".au", ".wav", ".mp3", ".flac", ".ogg"}
LYRICS_EXTS = {".txt", ".lrc"}


def list_audio_files(audio_dir: Path) -> list[Path]:
    files: list[Path] = []
    for p in audio_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
    return sorted(files)


def build_items(audio_dir: Path, lyrics_dir: Path | None = None) -> list[TrackItem]:
    audio_dir = audio_dir.expanduser().resolve()
    if lyrics_dir is not None:
        lyrics_dir = lyrics_dir.expanduser().resolve()

    audio_files = list_audio_files(audio_dir)
    items: list[TrackItem] = []
    for ap in audio_files:
        label = infer_label_from_filename(ap)
        lyrics_path: Path | None = None
        if lyrics_dir is not None and lyrics_dir.exists():
            # best-effort: match by stem prefix; if audio is genre.00012.au, lyrics could be genre.00012.txt
            cand = lyrics_dir / (ap.stem + ".txt")
            if cand.exists():
                lyrics_path = cand
        items.append(
            TrackItem(
                track_id=ap.stem,
                audio_path=ap,
                lyrics_path=lyrics_path,
                label=label,
            )
        )
    return items


@dataclass
class FeatureBatch:
    X: np.ndarray  # (n, d)
    ids: list[str]
    labels: list[str]  # may include "unknown"


def _pad_or_truncate_2d(mat: np.ndarray, target_frames: int) -> np.ndarray:
    # mat: (n_mels, t)
    n_mels, t = mat.shape
    if t == target_frames:
        return mat
    if t > target_frames:
        return mat[:, :target_frames]
    pad = np.zeros((n_mels, target_frames - t), dtype=mat.dtype)
    return np.concatenate([mat, pad], axis=1)


def extract_audio_features(
    items: Iterable[TrackItem],
    *,
    sample_rate: int = 22050,
    n_mels: int = 64,
    hop_length: int = 512,
    n_fft: int = 2048,
    target_frames: int = 256,
) -> FeatureBatch:
    """Extract log-mel spectrogram features.

    Output is flattened (n_mels * target_frames). This is intentionally simple for MLP-VAE.
    """

    import librosa  # local import to keep import-time requirements minimal

    feats: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[str] = []

    for it in items:
        y, sr = librosa.load(it.audio_path.as_posix(), sr=sample_rate, mono=True)
        S = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            power=2.0,
        )
        S = np.log10(1e-6 + S)
        S = _pad_or_truncate_2d(S, target_frames)
        feats.append(S.reshape(-1).astype(np.float32))
        ids.append(it.track_id)
        labels.append(it.label or "unknown")

    X = np.stack(feats, axis=0)
    # standardize per-feature
    X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)
    return FeatureBatch(X=X, ids=ids, labels=labels)


def extract_lyrics_features(
    items: Iterable[TrackItem],
    *,
    max_features: int = 5000,
) -> FeatureBatch:
    """Extract TF-IDF features from lyrics files if present; missing lyrics become empty strings."""

    from sklearn.feature_extraction.text import TfidfVectorizer

    texts: list[str] = []
    ids: list[str] = []
    labels: list[str] = []

    for it in items:
        text = ""
        if it.lyrics_path is not None and it.lyrics_path.exists():
            try:
                text = it.lyrics_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                text = ""
        texts.append(text)
        ids.append(it.track_id)
        labels.append(it.label or "unknown")

    vec = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return FeatureBatch(X=X, ids=ids, labels=labels)


def fuse_features(*batches: FeatureBatch) -> FeatureBatch:
    if not batches:
        raise ValueError("No batches to fuse")
    ids0 = batches[0].ids
    labels0 = batches[0].labels
    for b in batches[1:]:
        if b.ids != ids0:
            raise ValueError("Feature batches have mismatched ordering/ids")
    X = np.concatenate([b.X for b in batches], axis=1)
    return FeatureBatch(X=X.astype(np.float32), ids=ids0, labels=labels0)
