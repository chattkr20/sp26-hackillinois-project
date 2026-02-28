"""
Modal.com serverless GPU inference service for machine audio anomaly detection.

Model: facebook/wav2vec2-large fine-tuned on AnomalyMachine-50K
       (binary: normal vs anomalous; optional subtype head)

Pipeline:
  WebM bytes (browser MediaRecorder)
    → ffmpeg → WAV
    → resample 16 kHz mono
    → Wav2Vec2-Large feature extraction
    → binary classification head
    → JSON anomaly result

Output schema:
  {
    "status":          "normal" | "anomaly",
    "anomaly_score":   float,          # sigmoid probability of anomaly
    "threshold":       float,          # decision boundary
    "machine_type":    str | null,     # predicted machine type (if multi-head checkpoint)
    "anomaly_subtype": str | null,     # predicted subtype (if multi-head checkpoint)
    "confidence":      float           # max softmax confidence of machine_type prediction
  }

Checkpoint expected at: /vol/wav2vec2_anomaly.pt

Checkpoint format:
  {
    "model_state_dict": ...,
    "config": {
        "sample_rate": 16000,
        "clip_seconds": 10,
        "threshold": 0.5,
        "num_machine_types": 6,        # optional — omit for binary-only model
        "machine_type_labels": [...],  # optional
        "anomaly_subtype_labels": [...] # optional
    }
  }

If the checkpoint is not yet available (pre-training), the module falls back to
a zero-shot statistical detector using the mean norm of the Wav2Vec2 hidden states.
This allows the endpoint to be deployed and tested before fine-tuning is complete.
"""

import io
from typing import Any

import modal
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Modal app / image / volume
# ---------------------------------------------------------------------------

app = modal.App("cat-audio-anomaly")

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
)

# ---------------------------------------------------------------------------
# Label maps (must match training)
# ---------------------------------------------------------------------------

MACHINE_TYPE_LABELS: list[str] = [
    "fan",
    "pump",
    "compressor",
    "conveyor_belt",
    "electric_motor",
    "valve",
]

ANOMALY_SUBTYPE_LABELS: list[str] = [
    "none",
    "bearing_fault",
    "imbalance",
    "cavitation",
    "overheating",
    "obstruction",
]

# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------


class Wav2Vec2AnomalyModel(nn.Module):
    """
    Wav2Vec2-Large backbone with a binary anomaly head and optional
    machine-type and anomaly-subtype classification heads.

    The feature extractor is frozen; only the transformer encoder layers and
    classification heads are trained (see train_anomaly.py).
    """

    def __init__(
        self,
        num_machine_types: int = 6,
        num_subtypes: int = 6,
        hidden_size: int = 1024,  # wav2vec2-large hidden dim
    ) -> None:
        super().__init__()
        from transformers import Wav2Vec2Model

        self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        # Freeze the CNN feature extractor; transformer layers stay trainable
        self.wav2vec2.feature_extractor._freeze_parameters()

        # Binary anomaly head
        self.anomaly_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

        # Optional multi-class heads (trained jointly)
        self.machine_type_head = nn.Linear(hidden_size, num_machine_types)
        self.subtype_head = nn.Linear(hidden_size, num_subtypes)

    def _pool(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Mean-pool over the time dimension."""
        return hidden_states.mean(dim=1)  # (B, hidden_size)

    def forward(
        self, input_values: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_values: (B, T) float32 waveform at 16 kHz.

        Returns:
            Dict with:
              - anomaly_logit:       (B, 1)
              - machine_type_logit:  (B, num_machine_types)
              - subtype_logit:       (B, num_subtypes)
              - pooled:              (B, hidden_size)  for downstream use
        """
        outputs = self.wav2vec2(input_values=input_values, mask_time_indices=None)
        pooled = self._pool(outputs.last_hidden_state)  # (B, H)

        return {
            "anomaly_logit": self.anomaly_head(pooled),
            "machine_type_logit": self.machine_type_head(pooled),
            "subtype_logit": self.subtype_head(pooled),
            "pooled": pooled,
        }


# ---------------------------------------------------------------------------
# Zero-shot fallback (no fine-tuned checkpoint needed)
# ---------------------------------------------------------------------------

class ZeroShotAnomalyDetector:
    """
    Statistical anomaly detector using the raw Wav2Vec2 hidden-state norm.

    Normal machine audio produces consistent, lower-variance feature norms.
    Anomalous audio (impulses, cavitation bursts, overheating ramps) distorts
    the feature distribution. The decision is based on a z-score threshold over
    per-frame norms compared to an empirical baseline.

    This is deployed when the fine-tuned checkpoint is not yet available,
    allowing end-to-end testing before training is complete.
    """

    # Empirical baseline (will be overridden by checkpoint config if available)
    _BASELINE_MEAN: float = 8.5
    _BASELINE_STD: float = 1.2
    _Z_THRESHOLD: float = 2.5

    def __init__(self, wav2vec2_model: Any, threshold: float = 0.5) -> None:
        self.backbone = wav2vec2_model
        self.threshold = threshold

    @torch.no_grad()
    def __call__(self, input_values: torch.Tensor) -> dict[str, Any]:
        outputs = self.backbone(input_values=input_values, mask_time_indices=None)
        hidden = outputs.last_hidden_state  # (1, T, H)
        frame_norms = hidden.norm(dim=-1).squeeze(0)  # (T,)
        z_score = (frame_norms.mean().item() - self._BASELINE_MEAN) / self._BASELINE_STD
        score = torch.sigmoid(torch.tensor(z_score)).item()
        return {"anomaly_score": score, "threshold": self.threshold}


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_model: Wav2Vec2AnomalyModel | None = None
_zero_shot: ZeroShotAnomalyDetector | None = None
_config: dict[str, Any] = {}
_use_zero_shot: bool = False
CHECKPOINT_PATH = "/vol/wav2vec2_anomaly.pt"


def _load_model() -> None:
    """
    Attempt to load fine-tuned checkpoint from Modal Volume.
    Falls back to zero-shot statistical detector if checkpoint is absent.
    """
    global _model, _zero_shot, _config, _use_zero_shot
    import os
    from transformers import Wav2Vec2Model

    default_config = {
        "sample_rate": 16_000,
        "clip_seconds": 10,
        "threshold": 0.5,
        "num_machine_types": 6,
        "machine_type_labels": MACHINE_TYPE_LABELS,
        "anomaly_subtype_labels": ANOMALY_SUBTYPE_LABELS,
    }

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
        cfg = checkpoint.get("config", default_config)
        _config = {**default_config, **cfg}

        model = Wav2Vec2AnomalyModel(
            num_machine_types=_config.get("num_machine_types", 6),
            num_subtypes=len(_config.get("anomaly_subtype_labels", ANOMALY_SUBTYPE_LABELS)),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        model.to("cuda")
        _model = model
        _use_zero_shot = False
        print("[audio_anomaly] Fine-tuned checkpoint loaded.")
    else:
        # Fall back to zero-shot using raw Wav2Vec2-Large
        _config = default_config
        print("[audio_anomaly] Checkpoint not found — using zero-shot detector.")
        backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
        backbone.eval()
        backbone.to("cuda")
        _zero_shot = ZeroShotAnomalyDetector(backbone, threshold=default_config["threshold"])
        _use_zero_shot = True


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess(webm_bytes: bytes) -> torch.Tensor:
    """
    Convert WebM bytes → 16 kHz mono waveform of fixed length on CUDA.

    Args:
        webm_bytes: Raw browser MediaRecorder WebM buffer.

    Returns:
        Tensor of shape (1, clip_samples) on CUDA.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from ai.utils import webm_to_wav_bytes, load_waveform, pad_or_crop, normalise_waveform

    wav_bytes = webm_to_wav_bytes(webm_bytes)

    sr = _config["sample_rate"]
    clip_samples = sr * _config["clip_seconds"]

    waveform = load_waveform(wav_bytes, target_sr=sr, mono=True)  # (1, T)
    waveform = pad_or_crop(waveform, clip_samples)
    waveform = normalise_waveform(waveform)

    return waveform.to("cuda")  # (1, clip_samples)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def _run_fine_tuned(waveform: torch.Tensor) -> dict[str, Any]:
    """Run the fine-tuned multi-head model and return structured results."""
    cfg = _config
    threshold: float = cfg.get("threshold", 0.5)
    machine_labels: list[str] = cfg.get("machine_type_labels", MACHINE_TYPE_LABELS)
    subtype_labels: list[str] = cfg.get("anomaly_subtype_labels", ANOMALY_SUBTYPE_LABELS)

    outputs = _model(waveform)  # type: ignore[arg-type]

    anomaly_score: float = torch.sigmoid(outputs["anomaly_logit"]).item()
    status = "anomaly" if anomaly_score > threshold else "normal"

    # Machine type
    mt_probs = torch.softmax(outputs["machine_type_logit"], dim=-1).squeeze(0)
    mt_idx = mt_probs.argmax().item()
    machine_type = machine_labels[mt_idx] if mt_idx < len(machine_labels) else None
    confidence = mt_probs[mt_idx].item()

    # Anomaly subtype (only meaningful if anomaly detected)
    st_idx = torch.softmax(outputs["subtype_logit"], dim=-1).squeeze(0).argmax().item()
    anomaly_subtype = (
        subtype_labels[st_idx]
        if (status == "anomaly" and st_idx < len(subtype_labels) and subtype_labels[st_idx] != "none")
        else None
    )

    return {
        "status": status,
        "anomaly_score": round(anomaly_score, 4),
        "threshold": threshold,
        "machine_type": machine_type,
        "anomaly_subtype": anomaly_subtype,
        "confidence": round(confidence, 4),
    }


@torch.no_grad()
def _run_zero_shot(waveform: torch.Tensor) -> dict[str, Any]:
    """Run zero-shot statistical detector."""
    result = _zero_shot(waveform)  # type: ignore[misc]
    score = result["anomaly_score"]
    threshold = result["threshold"]
    return {
        "status": "anomaly" if score > threshold else "normal",
        "anomaly_score": round(score, 4),
        "threshold": threshold,
        "machine_type": None,
        "anomaly_subtype": None,
        "confidence": None,
        "mode": "zero_shot_statistical",
    }


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
@modal.fastapi_endpoint(method="POST")
def detect_anomaly(audio: bytes) -> dict[str, Any]:
    """
    POST endpoint — accepts raw WebM audio bytes (browser MediaRecorder).

    Args:
        audio: WebM audio bytes (body of the POST request).

    Returns:
        Structured JSON anomaly result.
        On error: ``{"error": str}``
    """
    global _model, _zero_shot

    if _model is None and _zero_shot is None:
        _load_model()

    try:
        waveform = _preprocess(audio)
    except Exception as exc:
        return {"error": f"audio preprocessing failed: {exc}"}

    try:
        if _use_zero_shot:
            return _run_zero_shot(waveform)
        return _run_fine_tuned(waveform)
    except Exception as exc:
        return {"error": f"inference failed: {exc}"}


# ---------------------------------------------------------------------------
# Local test entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) < 2:
        print("Usage: python audio_anomaly.py <path_to_webm_or_wav>")
        sys.exit(1)

    with open(sys.argv[1], "rb") as f:
        raw = f.read()

    result = detect_anomaly.local(raw)
    print(json.dumps(result, indent=2))
