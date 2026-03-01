"""
Modal.com endpoint for CAT equipment visual anomaly detection.

Inference pipeline (in priority order):
  1. Fine-tuned EfficientNet-B0 checkpoint (loaded from Modal Volume after training)
  2. Zero-shot CLIP fallback  (works immediately with no training)

POST /  — body: raw image bytes (JPEG / PNG / WEBP)
Returns:
  {
    "anomaly_score": float,      # 0.0 = normal, 1.0 = defective
    "status": "anomaly"|"normal",
    "confidence": float,         # how confident the model is
    "mode": "finetuned"|"zero_shot",
    "label": str                 # human-readable
  }
"""

from __future__ import annotations

import io
from typing import Any

import modal
from fastapi import Request

# ---------------------------------------------------------------------------
# Modal app / image / volume
# ---------------------------------------------------------------------------

app = modal.App("cat-image-anomaly")

volume = modal.Volume.from_name("cat-image-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.47.0",
        "Pillow",
        "numpy<2",
        "fastapi[standard]",
    )
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = "/vol/efficientnet_anomaly.pt"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
THRESHOLD = 0.5

# CLIP text prompts — tuned for CAT equipment inspection
NORMAL_PROMPTS = [
    "a normal machine part in good condition",
    "a clean undamaged equipment component",
    "a properly functioning industrial part",
]
ANOMALY_PROMPTS = [
    "a damaged or defective machine part",
    "a worn broken cracked industrial component",
    "a faulty equipment part with visible damage",
]

# ---------------------------------------------------------------------------
# EfficientNet-B0 architecture (must match train_image_anomaly.py)
# ---------------------------------------------------------------------------

def _build_efficientnet(num_classes: int = 2):
    import torchvision.models as models
    import torch.nn as nn

    m = models.efficientnet_b0(weights=None)
    # Replace classifier head
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return m


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_finetuned_model = None
_clip_model = None
_clip_processor = None
_mode = "zero_shot"

# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    gpu="T4",                   # T4 is cheaper; image models are lighter
    volumes={"/vol": volume},
    timeout=120,
    scaledown_window=300,
)
class ImageAnomalyDetector:

    @modal.enter()
    def load_models(self):
        global _finetuned_model, _clip_model, _clip_processor, _mode
        import os, torch

        # ── 1. Try fine-tuned checkpoint ────────────────────────────────────
        if os.path.exists(CHECKPOINT_PATH):
            try:
                checkpoint = torch.load(CHECKPOINT_PATH, map_location="cuda")
                model = _build_efficientnet(num_classes=2)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval().to("cuda")
                _finetuned_model = model
                _mode = "finetuned"
                print("[image_anomaly] Fine-tuned EfficientNet-B0 checkpoint loaded.")
            except Exception as e:
                print(f"[image_anomaly] Checkpoint load failed: {e} — falling back to CLIP.")

        # ── 2. Always load CLIP as fallback ─────────────────────────────────
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(
            "cuda" if _finetuned_model is None else "cpu"
        )
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()

        if _finetuned_model is None:
            print("[image_anomaly] No checkpoint — using CLIP zero-shot mode.")

    @modal.fastapi_endpoint(method="POST")
    async def detect_anomaly(self, request: Request) -> dict:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        # Read raw image bytes from request body
        body = await request.body()
        try:
            img = Image.open(io.BytesIO(body)).convert("RGB")
        except Exception as e:
            return {"error": f"Could not decode image: {e}"}

        if _mode == "finetuned" and _finetuned_model is not None:
            return self._finetuned_inference(img)
        else:
            return self._clip_inference(img)

    def _finetuned_inference(self, img) -> dict:
        import torch
        import torch.nn.functional as F
        import torchvision.transforms as T

        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        x = transform(img).unsqueeze(0).to("cuda")

        with torch.no_grad():
            logits = _finetuned_model(x)           # (1, 2)  [normal, anomaly]
            probs = F.softmax(logits, dim=-1)[0]   # (2,)
            anomaly_score = probs[1].item()         # class 1 = anomaly

        status = "anomaly" if anomaly_score >= THRESHOLD else "normal"
        label = (
            "Defect / damage detected"
            if status == "anomaly"
            else "No visible defect"
        )
        return {
            "anomaly_score": round(anomaly_score, 4),
            "status": status,
            "confidence": round(max(probs[0].item(), probs[1].item()), 4),
            "mode": "finetuned",
            "label": label,
        }

    def _clip_inference(self, img) -> dict:
        import torch
        import torch.nn.functional as F

        all_prompts = NORMAL_PROMPTS + ANOMALY_PROMPTS
        inputs = _clip_processor(
            text=all_prompts,
            images=img,
            return_tensors="pt",
            padding=True,
        )
        # Move to same device as clip model
        device = next(_clip_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = _clip_model(**inputs)
            logits = outputs.logits_per_image[0]        # (num_prompts,)
            probs = F.softmax(logits, dim=0)             # (num_prompts,)

        n = len(NORMAL_PROMPTS)
        normal_score = probs[:n].sum().item()
        anomaly_score = probs[n:].sum().item()

        # Renormalize to [0,1] anomaly probability
        total = normal_score + anomaly_score
        anomaly_prob = anomaly_score / total if total > 0 else 0.5

        status = "anomaly" if anomaly_prob >= THRESHOLD else "normal"
        label = (
            "Potential defect detected (visual)"
            if status == "anomaly"
            else "No visible defect (visual)"
        )
        return {
            "anomaly_score": round(anomaly_prob, 4),
            "status": status,
            "confidence": round(max(normal_score, anomaly_score) / total, 4),
            "mode": "zero_shot",
            "label": label,
        }
