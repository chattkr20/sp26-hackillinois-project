"""
Modal.com endpoint for CAT equipment visual anomaly detection.
POST /  body: JSON {"image_b64": "<base64-encoded image bytes>"}
Returns: {"anomaly_score", "status", "confidence", "mode", "label"}
"""

import base64, io, traceback
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import modal

app = modal.App("cat-image-anomaly")
volume = modal.Volume.from_name("cat-image-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "transformers==4.47.0",
        "Pillow>=10.0",
        "numpy==1.26.4",
        "fastapi[standard]",
        "python-multipart",
    )
)

CHECKPOINT_PATH = "/vol/efficientnet_anomaly.pt"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
THRESHOLD = 0.5

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

_finetuned_model = None
_clip_model = None
_clip_processor = None
_mode = "zero_shot"


def _build_efficientnet(num_classes=2):
    """Must match the classifier head built in train_image_anomaly.py exactly."""
    import torchvision.models as models
    import torch.nn as nn
    m = models.efficientnet_b0(weights=None)
    in_features = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes),
    )
    return m


def _load_models():
    global _finetuned_model, _clip_model, _clip_processor, _mode
    import os, torch

    cuda_ok = torch.cuda.is_available()
    dev = "cuda" if cuda_ok else "cpu"
    print(f"[image_anomaly] CUDA available: {cuda_ok}")

    # ── Try fine-tuned EfficientNet (classifier must match train_image_anomaly.py) ──
    if os.path.exists(CHECKPOINT_PATH):
        print(f"[image_anomaly] Checkpoint found at {CHECKPOINT_PATH}")
        try:
            ckpt = torch.load(CHECKPOINT_PATH, map_location=dev)
            m = _build_efficientnet()
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval().to(dev)
            _finetuned_model = m
            _mode = "finetuned"
            print("[image_anomaly] EfficientNet-B0 fine-tuned model loaded.")
        except Exception:
            print(f"[image_anomaly] Checkpoint load FAILED:\n{traceback.format_exc()}")
    else:
        print(f"[image_anomaly] No checkpoint at {CHECKPOINT_PATH} — will use CLIP zero-shot.")

    # ── Load CLIP (CPU to save VRAM when fine-tuned model is on GPU) ─────────
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_dev = "cpu" if (_finetuned_model and cuda_ok) else dev
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(clip_dev)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
        _clip_model.eval()
        print(f"[image_anomaly] CLIP loaded on {clip_dev}.")
    except Exception:
        print(f"[image_anomaly] CLIP load FAILED:\n{traceback.format_exc()}")

    print(f"[image_anomaly] _load_models complete. mode={_mode}")


def _finetuned_inference(img):
    import torch
    import torch.nn.functional as F
    import torchvision.transforms as T
    tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    x = tf(img).unsqueeze(0)
    if torch.cuda.is_available():
        x = x.to("cuda")
    with torch.no_grad():
        probs = F.softmax(_finetuned_model(x), dim=-1)[0]
    s = probs[1].item()
    status = "anomaly" if s >= THRESHOLD else "normal"
    return {
        "anomaly_score": round(s, 4),
        "status": status,
        "confidence": round(max(probs[0].item(), probs[1].item()), 4),
        "mode": "finetuned",
        "label": "Defect / damage detected" if status == "anomaly" else "No visible defect",
    }


def _clip_inference(img):
    import torch
    import torch.nn.functional as F
    prompts = NORMAL_PROMPTS + ANOMALY_PROMPTS
    inputs = _clip_processor(text=prompts, images=img, return_tensors="pt", padding=True)
    dev = next(_clip_model.parameters()).device
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(_clip_model(**inputs).logits_per_image[0], dim=0)
    n = len(NORMAL_PROMPTS)
    ns, as_ = probs[:n].sum().item(), probs[n:].sum().item()
    total = ns + as_
    ap = as_ / total if total > 0 else 0.5
    status = "anomaly" if ap >= THRESHOLD else "normal"
    return {
        "anomaly_score": round(ap, 4),
        "status": status,
        "confidence": round(max(ns, as_) / total, 4),
        "mode": "zero_shot",
        "label": "Potential defect detected (visual)" if status == "anomaly" else "No visible defect (visual)",
    }


@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    timeout=120,
    scaledown_window=300,
    min_containers=1,          # keep one warm container — model stays loaded
)
@modal.asgi_app()
def detect_anomaly():
    fapp = FastAPI()
    fapp.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    @fapp.get("/health")
    async def _health():
        return {
            "mode": _mode,
            "finetuned_loaded": _finetuned_model is not None,
            "clip_loaded": _clip_model is not None,
        }

    @fapp.post("/")
    async def _detect(request: Request):
        # Lazy-load models on first request (matching audio_anomaly.py pattern).
        if _clip_model is None:
            print("[image_anomaly] WARNING: models not loaded — loading now.")
            _load_models()
        try:
            body = await request.json()
            raw = base64.b64decode(body["image_b64"])
            from PIL import Image
            img = Image.open(io.BytesIO(raw)).convert("RGB")
        except Exception as e:
            return JSONResponse({"error": f"Bad request: {e}"}, status_code=400)
        try:
            if _mode == "finetuned" and _finetuned_model is not None:
                return _finetuned_inference(img)
            return _clip_inference(img)
        except Exception as e:
            print(f"[image_anomaly] inference error:\n{traceback.format_exc()}")
            return JSONResponse({"error": f"Inference failed: {e}"}, status_code=500)

    return fapp
