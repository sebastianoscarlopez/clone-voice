"""CLI: clone a voice from a reference audio and synthesize text."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice cloning via XTTS v2")
    parser.add_argument("--ref", type=Path, required=True, help="Reference audio file")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--out", type=Path, required=True, help="Output WAV path")
    parser.add_argument("--lang", type=str, default="es", help="Language code (es, en, ...)")
    args = parser.parse_args()

    raise NotImplementedError("Phase 6 — wire preprocess + clone + chunker")


if __name__ == "__main__":
    sys.exit(main())
