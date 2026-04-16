"""CLI: clone a voice from a reference audio and synthesize text."""
import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.chunker import concat_wavs, split_text
from src.clone import VoiceCloner
from src.config import INTER_CHUNK_SILENCE_MS, MAX_CHARS_PER_CHUNK
from src.preprocess import preprocess


def main() -> int:
    parser = argparse.ArgumentParser(description="Voice cloning via XTTS v2")
    parser.add_argument("--ref", type=Path, required=True, help="Reference audio (WAV)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Inline text to synthesize")
    group.add_argument("--text-file", type=Path, help="Path to a UTF-8 text file")
    parser.add_argument("--out", type=Path, required=True, help="Output WAV path")
    parser.add_argument("--lang", type=str, default="es", help="Language code (es, en, fr, ...)")
    parser.add_argument("--temperature", type=float, default=0.4, help="XTTS sampling temperature")
    parser.add_argument("--preprocess", action="store_true", help="Normalize + trim reference first")
    parser.add_argument(
        "--max-chars",
        type=int,
        default=MAX_CHARS_PER_CHUNK,
        help=f"Max chars per synthesis chunk (default {MAX_CHARS_PER_CHUNK})",
    )
    args = parser.parse_args()

    if not args.ref.exists():
        print(f"error: reference not found: {args.ref}", file=sys.stderr)
        return 1

    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        print("error: empty text", file=sys.stderr)
        return 1

    ref_path = args.ref
    if args.preprocess:
        processed = ref_path.with_name(f"{ref_path.stem}_processed.wav")
        preprocess(ref_path, processed)
        ref_path = processed
        print(f"preprocessed reference: {processed}")

    chunks = split_text(text, max_chars=args.max_chars)
    print(f"split into {len(chunks)} chunk(s)")

    cloner = VoiceCloner(ref_audio=ref_path, language=args.lang)

    if len(chunks) == 1:
        cloner.synthesize(chunks[0], args.out, temperature=args.temperature)
        print(f"wrote {args.out}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        for i, chunk in enumerate(chunks):
            p = tmp_dir / f"chunk_{i:02d}.wav"
            cloner.synthesize(chunk, p, temperature=args.temperature)
            parts.append(p)
        concat_wavs(parts, args.out, silence_ms=INTER_CHUNK_SILENCE_MS)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
