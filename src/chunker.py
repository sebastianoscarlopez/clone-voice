"""Split long text into TTS-friendly chunks; concat WAVs with silence."""
from pathlib import Path


def split_text(text: str, max_chars: int) -> list[str]:
    raise NotImplementedError("Phase 5")


def concat_wavs(parts: list[Path], out_path: Path, silence_ms: int) -> Path:
    raise NotImplementedError("Phase 5")
