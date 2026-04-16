# Voice Cloning Project Plan — RTX 3050 Ti (4 GB VRAM)

## Goal

Clone a voice from a short reference audio (~16s) and synthesize arbitrary text in that voice, running locally on a laptop with an NVIDIA RTX 3050 Ti (4 GB VRAM) on Windows 11.

## Strategy: Two-tier approach

- **Primary:** XTTS v2 (Coqui TTS) — best quality-per-VRAM for cloning, works with 6–20s reference audio, supports Spanish and English natively.
- **Fallback:** OpenVoice v2 or RVC + EdgeTTS if XTTS runs out of memory on long inputs.

## Phase 1 — Environment (Day 1)

- Install **Python 3.10** (XTTS/Coqui breaks on 3.12+).
- Create venv: `python -m venv .venv`
- Install PyTorch with CUDA 11.8:
  `pip install torch==2.1.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118`
- Install: `coqui-tts` (community fork of TTS, still maintained), `pydub`, `soundfile`, `librosa`.
- Verify: `torch.cuda.is_available()` → `True`, check `torch.cuda.get_device_properties(0).total_memory`.

## Phase 2 — Project Structure

```
clone-voice/
├── .venv/
├── data/
│   ├── reference/          # your 16s sample (WAV, 22050 Hz mono)
│   └── output/
├── src/
│   ├── preprocess.py       # clean + normalize reference audio
│   ├── clone.py            # load XTTS, generate speech
│   ├── chunker.py          # split long text into sentences
│   └── config.py           # paths, model params, VRAM limits
├── scripts/
│   └── run.py              # CLI entrypoint
└── requirements.txt
```

## Phase 3 — Reference Audio Prep (`preprocess.py`)

- Convert to **22050 Hz mono WAV**.
- Trim silences (`librosa.effects.trim`).
- Normalize loudness (pydub, target `-20 dBFS`).
- Target: 10–20s clean speech, no background noise, single speaker.

## Phase 4 — Cloning + Synthesis (`clone.py`)

Key VRAM tactics:

- Load XTTS with `torch.float16` on GPU.
- Call `model.get_conditioning_latents(audio_path=ref)` **once**, cache the tensor to disk (`.pt`). Reuse across all subsequent generations — this is the "voice embedding".
- Run inference inside `torch.inference_mode()` (no grad tracking).
- Call `torch.cuda.empty_cache()` between chunks.

## Phase 5 — Long-text Handling (`chunker.py`)

- Split input text on sentence boundaries (`. ! ?`), max ~250 chars per chunk.
- Generate each chunk separately.
- Concatenate WAVs with `pydub`, inserting 150ms silence between chunks.

## Phase 6 — CLI (`run.py`)

```
python scripts/run.py --ref data/reference/me.wav --text "Hola mundo" --out data/output/out.wav --lang es
```

## Phase 7 — Fallback Plan

If XTTS OOMs even with FP16 + chunking:

- Drop to **OpenVoice v2** (ToneColorConverter is ~1.5 GB).
- Or **RVC-only**: generate base audio with Edge-TTS (CPU, free, Microsoft neural voices), then pass it through RVC to apply your timbre. Total GPU load <2 GB.

## VRAM Budget Expectations

| Stage | Est. VRAM |
| --- | --- |
| XTTS v2 FP16 loaded | ~2.2 GB |
| Inference peak (short chunk) | +0.8 GB |
| **Headroom left** | ~1 GB |

Tight but feasible. Close Chrome, Discord, Edge (system tray) before running.

## Open Questions

1. Path to the 16s reference audio file.
2. Whether Python 3.10 is already installed.
