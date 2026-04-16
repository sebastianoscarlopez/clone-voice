"""Reference audio preprocessing: resample, trim silences, normalize loudness."""
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, TARGET_DBFS


def preprocess(input_path: Path, output_path: Path, top_db: float = 30.0) -> Path:
    audio, _ = librosa.load(str(input_path), sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio, top_db=top_db)

    rms = float(np.sqrt(np.mean(audio ** 2)))
    peak = float(np.max(np.abs(audio)))
    if rms > 0 and peak > 0:
        current_dbfs = 20.0 * np.log10(rms)
        rms_gain = 10.0 ** ((TARGET_DBFS - current_dbfs) / 20.0)
        peak_ceiling_gain = 0.891 / peak  # -1 dBFS headroom
        audio = audio * min(rms_gain, peak_ceiling_gain)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, SAMPLE_RATE, subtype="PCM_16")
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("usage: python -m src.preprocess <input> <output>", file=sys.stderr)
        sys.exit(1)
    out = preprocess(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"wrote {out}")
