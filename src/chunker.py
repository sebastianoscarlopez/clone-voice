"""Split long text into TTS-friendly chunks; concat WAVs with silence."""
import re
from pathlib import Path

import numpy as np
import soundfile as sf

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def split_text(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            for i in range(0, len(sentence), max_chars):
                chunks.append(sentence[i : i + max_chars])
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def concat_wavs(parts: list[Path], out_path: Path, silence_ms: int) -> Path:
    if not parts:
        raise ValueError("no parts to concatenate")
    audios = []
    sr_ref = None
    for p in parts:
        data, sr = sf.read(str(p))
        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            raise ValueError(f"sample rate mismatch: {sr} vs {sr_ref} in {p}")
        audios.append(data)

    silence = np.zeros(int(sr_ref * silence_ms / 1000), dtype=audios[0].dtype)
    pieces: list[np.ndarray] = []
    for i, a in enumerate(audios):
        if i > 0:
            pieces.append(silence)
        pieces.append(a)
    combined = np.concatenate(pieces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), combined, sr_ref)
    return out_path
