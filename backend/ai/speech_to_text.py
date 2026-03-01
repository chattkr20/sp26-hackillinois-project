"""
Modal.com serverless speech-to-text endpoint using OpenAI Whisper.

Pipeline:
  WebM bytes (browser MediaRecorder)
    → ffmpeg → WAV
    → Whisper-medium (or large-v3) transcription
    → structured inspection JSON

Output schema:
  {
    "transcript": str,            # raw transcribed speech
    "language": str,              # detected language code, e.g. "en"
    "duration_seconds": float,    # audio length
    "inspection_notes": [         # extracted observation items
      {
        "observation": str,       # what was said about this item
        "keywords": [str]         # matched inspection keywords
      }
    ]
  }

The `inspection_notes` field gives the next pipeline stage (LLM report generator)
pre-parsed regions of interest without requiring an additional LLM call here.
"""

import re
from typing import Any

import modal
import torch

# ---------------------------------------------------------------------------
# Modal app / image / volume
# ---------------------------------------------------------------------------

app = modal.App("cat-speech-to-text")

volume = modal.Volume.from_name("cat-audio-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "transformers==4.47.0",
        "accelerate",
        "numpy<2",
        "fastapi[standard]",
    )
    .add_local_file("backend/ai/utils.py", "/root/ai_utils.py")
)

# ---------------------------------------------------------------------------
# Inspection keyword vocabulary
# (used for lightweight pre-parsing of transcripts — no LLM needed here)
# ---------------------------------------------------------------------------

_INSPECTION_KEYWORDS: dict[str, list[str]] = {
    "track": ["track", "undercarriage", "belt", "link", "chain", "shoe"],
    "ladder": ["ladder", "step", "rung", "access", "climb"],
    "hydraulic": ["hydraulic", "cylinder", "hose", "leak", "fluid", "oil"],
    "engine": ["engine", "motor", "exhaust", "smoke", "coolant", "overheat"],
    "bucket": ["bucket", "teeth", "lip", "cutting edge", "blade"],
    "cab": ["cab", "window", "glass", "door", "mirror", "seat", "seatbelt"],
    "tire": ["tire", "tyre", "wheel", "rim", "inflation", "pressure"],
    "lights": ["light", "lamp", "beacon", "strobe", "headlight"],
    "body": ["body", "panel", "dent", "crack", "rust", "corrosion", "weld"],
    "electrical": ["wire", "cable", "connector", "battery", "electrical", "fuse"],
}

# ---------------------------------------------------------------------------
# Global model state (loaded once at cold start)
# ---------------------------------------------------------------------------

_pipe: Any = None  # transformers.pipeline


def _load_model() -> None:
    """Load Whisper model at cold-start and cache on GPU."""
    global _pipe
    from transformers import pipeline

    _pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        model_kwargs={"attn_implementation": "sdpa"},
    )


# ---------------------------------------------------------------------------
# Transcript post-processing
# ---------------------------------------------------------------------------

def _extract_inspection_notes(transcript: str) -> list[dict[str, Any]]:
    """
    Scan the transcript for sentences mentioning inspection-relevant keywords
    and return a list of structured observation objects.

    Args:
        transcript: Raw transcribed text.

    Returns:
        List of dicts with keys ``observation`` and ``keywords``.
    """
    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.!?])\s+|,\s*", transcript)

    notes: list[dict[str, Any]] = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        matched_categories: list[str] = []
        lower = sentence.lower()
        for category, kws in _INSPECTION_KEYWORDS.items():
            if any(kw in lower for kw in kws):
                matched_categories.append(category)
        if matched_categories:
            notes.append(
                {
                    "observation": sentence,
                    "keywords": matched_categories,
                }
            )

    return notes


# ---------------------------------------------------------------------------
# WebM → WAV helper (uses ffmpeg installed in the Modal image)
# ---------------------------------------------------------------------------

def _convert_webm(webm_bytes: bytes) -> bytes:
    """Inline import to avoid circular issues when this module used standalone."""
    import sys
    if '/root' not in sys.path:
        sys.path.insert(0, '/root')
    from ai_utils import webm_to_wav_bytes
    return webm_to_wav_bytes(webm_bytes)


# ---------------------------------------------------------------------------
# Modal web endpoint
# ---------------------------------------------------------------------------

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    min_containers=1,
    timeout=120,
)
@modal.asgi_app()
def transcribe():
    """
    ASGI endpoint with CORS — accepts raw audio bytes from browser MediaRecorder.
    URL: https://milindkumar1--cat-speech-to-text-transcribe.modal.run
    """
    from fastapi import FastAPI, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    import io
    import torchaudio
    import torchaudio.functional as FA

    fastapi_app = FastAPI()
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi_app.post("/")
    async def _transcribe(request: Request):
        global _pipe

        if _pipe is None:
            _load_model()

        audio = await request.body()
        if not audio:
            return JSONResponse({"error": "empty request body", "size": 0})

        print(f"[transcribe] received {len(audio)} bytes")
        # ---- Convert WebM → WAV -----------------------------------------
        try:
            wav_bytes = _convert_webm(audio)
        except Exception as exc:
            print(f"[transcribe] conversion error: {exc}")
            return JSONResponse({"error": f"audio conversion failed: {exc}"})

        # ---- Run Whisper ------------------------------------------------
        try:
            buf = io.BytesIO(wav_bytes)
            waveform, sr = torchaudio.load(buf)  # (C, T)
            duration = waveform.shape[-1] / sr

            if sr != 16_000:
                waveform = FA.resample(waveform, sr, 16_000)
            audio_np = waveform.squeeze(0).numpy()

            result: dict = _pipe(
                audio_np,
                return_timestamps=False,
                generate_kwargs={"language": "en", "task": "transcribe"},
            )
            transcript: str = result.get("text", "").strip()
        except Exception as exc:
            print(f"[transcribe] whisper error: {exc}")
            return JSONResponse({"error": f"transcription failed: {exc}"})

        # ---- Post-process -----------------------------------------------
        inspection_notes = _extract_inspection_notes(transcript)

        return {
            "transcript": transcript,
            "language": "en",
            "duration_seconds": round(duration, 2),
            "inspection_notes": inspection_notes,
        }

    return fastapi_app
