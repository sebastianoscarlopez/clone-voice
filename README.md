# clone-voice

Local voice cloning + text-to-speech via XTTS v2, targeted at laptops with
~4 GB VRAM (tested on an RTX 3050 Ti).

Given a short reference recording of a voice, generate arbitrary text spoken
in that voice. Supports Spanish, English, and other languages XTTS v2 was
trained on.

## Requirements

- Windows 10/11 (the install steps below assume it; Linux/macOS should work
  with equivalent package managers)
- Python 3.11
- NVIDIA GPU with ~4 GB VRAM and a driver that supports CUDA 12.x
- ffmpeg on PATH (only needed if your reference audio isn't already WAV)

## Install

```bash
# From the project root
py -3.11 -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

The first run will download the XTTS v2 model (~1.8 GB) into the Coqui TTS
cache (`%USERPROFILE%\AppData\Local\tts\` on Windows).

If you don't have ffmpeg:

```
winget install --id=Gyan.FFmpeg -e
```

Then open a new terminal so `ffmpeg` lands on PATH.

## Reference audio

XTTS clones best from **10–20 seconds of clean, loud speech** from a single
speaker. Target:

- Peak around -3 to -1 dBFS (loud but not clipped)
- No music, fan noise, or reverb
- Source bitrate at least 128 kbps if encoded (96 kbps AAC or lower blurs
  vocal detail and produces poor clones)
- Mic ~10 cm from mouth, normal conversation volume

The CLI accepts M4A/MP3/WAV (ffmpeg needed for non-WAV).

## Usage

Basic:

```bash
.venv/Scripts/python.exe scripts/run.py \
  --ref path/to/your_voice.m4a \
  --text "Hola, esta es mi voz clonada." \
  --out out.wav \
  --lang es \
  --preprocess
```

Long text from a file:

```bash
.venv/Scripts/python.exe scripts/run.py \
  --ref path/to/your_voice.wav \
  --text-file story.txt \
  --out story.wav \
  --lang en \
  --temperature 0.4
```

### Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--ref` | (required) | Reference audio file (WAV or ffmpeg-supported format) |
| `--text` / `--text-file` | (one required) | Inline text or path to a UTF-8 text file |
| `--out` | (required) | Output WAV path |
| `--lang` | `es` | Language code (`es`, `en`, `fr`, `de`, `it`, `pt`, ...) |
| `--temperature` | `0.4` | XTTS sampling temperature. Lower = more faithful to reference |
| `--preprocess` | off | Run silence-trim + loudness-normalize on the reference first |
| `--max-chars` | `250` | Chunk size for long text (split on sentence boundaries) |

## How it works

1. `src/preprocess.py` — optional reference cleanup: resample to 22050 Hz
   mono, trim leading/trailing silence, normalize loudness to ~-20 dBFS with
   peak headroom at -1 dBFS.
2. `src/clone.py` — `VoiceCloner` loads XTTS v2, extracts the speaker's
   conditioning latents once from the reference, and caches them to
   `data/cache/<name>.latents.pt`. Subsequent runs with the same reference
   reuse the cache instead of recomputing.
3. `src/chunker.py` — splits input text on sentence boundaries (`.?!`),
   greedily packs sentences into chunks of up to `--max-chars` characters,
   and concatenates per-chunk WAVs with 150 ms of silence between them.
4. `scripts/run.py` — CLI wiring the three together.

## VRAM notes

XTTS v2 runs in FP32 (the model has upstream FP16 incompatibilities in its
speaker encoder conv1d filter and GPT layer norms, so `.half()` fails). In
practice the model occupies ~2 GB and inference peaks at ~2.2 GB per chunk,
leaving ~1.8 GB headroom on a 4 GB card. Close Chrome, Discord, and other
GPU-using apps before running if you hit OOM.

Do not instantiate two `VoiceCloner` objects in the same process — that
loads the model twice and pushes peak past 4 GB.

## Model license

The XTTS v2 model is distributed under the Coqui Public Model License
(CPML) — **non-commercial use only**. See https://coqui.ai/cpml for terms.
Setting `COQUI_TOS_AGREED=1` (which this project does automatically)
constitutes acceptance.
