"""XTTS v2 voice cloning + synthesis with conditioning-latent caching."""
import os
from pathlib import Path

os.environ.setdefault("COQUI_TOS_AGREED", "1")

import soundfile as sf
import torch
from TTS.api import TTS

from src.config import CACHE_DIR, XTTS_MODEL


class VoiceCloner:
    def __init__(self, ref_audio: Path, language: str = "es", use_fp16: bool = True):
        self.language = language
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        tts = TTS(XTTS_MODEL, progress_bar=True).to(self.device)
        self.model = tts.synthesizer.tts_model

        if use_fp16 and self.device == "cuda":
            self.model.half()

        self.output_sr = self.model.config.audio.output_sample_rate

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = CACHE_DIR / f"{ref_audio.stem}.latents.pt"
        if cache_path.exists():
            cache = torch.load(cache_path, map_location=self.device)
            self.gpt_cond_latent = cache["gpt_cond_latent"]
            self.speaker_embedding = cache["speaker_embedding"]
        else:
            with torch.inference_mode():
                self.gpt_cond_latent, self.speaker_embedding = (
                    self.model.get_conditioning_latents(audio_path=[str(ref_audio)])
                )
            torch.save(
                {
                    "gpt_cond_latent": self.gpt_cond_latent,
                    "speaker_embedding": self.speaker_embedding,
                },
                cache_path,
            )

    def synthesize(self, text: str, out_path: Path, temperature: float = 0.7) -> Path:
        with torch.inference_mode():
            result = self.model.inference(
                text=text,
                language=self.language,
                gpt_cond_latent=self.gpt_cond_latent,
                speaker_embedding=self.speaker_embedding,
                temperature=temperature,
            )
        wav = result["wav"]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wav, self.output_sr)
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return out_path


if __name__ == "__main__":
    import sys
    from src.config import OUT_DIR, REF_DIR

    ref = REF_DIR / "me_processed.wav"
    text = sys.argv[1] if len(sys.argv) > 1 else "Hola, esta es una prueba de mi voz clonada."
    out = OUT_DIR / "clone_test.wav"

    cloner = VoiceCloner(ref_audio=ref, language="es")
    cloner.synthesize(text, out)
    print(f"wrote {out}")
