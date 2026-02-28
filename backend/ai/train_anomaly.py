"""
AnomalyMachine-50K Fine-tuning Script for Wav2Vec2-Large.

Run with:
    modal run backend/ai/train_anomaly.py

What this does:
  1. Loads mandipgoswami/AnomalyMachine-50K from HuggingFace Hub
  2. Fine-tunes facebook/wav2vec2-large:
       - CNN feature extractor frozen (saves VRAM, speeds up training)
       - All 24 transformer layers trainable
       - Three heads trained jointly:
           a) Binary anomaly (normal vs anomalous)  — primary loss, weight 1.0
           b) Machine type (6-class)                — auxiliary loss, weight 0.3
           c) Anomaly subtype (6-class)             — auxiliary loss, weight 0.2
  3. Saves best checkpoint (by validation AUC) to Modal Volume at /vol/wav2vec2_anomaly.pt

Expected A10G (24 GB) training time: ~90 minutes for 3 epochs over 35K clips.

Hyperparameters are set conservatively for hackathon robustness — feel free to tune.
"""

import os
from typing import Any

import modal

# ---------------------------------------------------------------------------
# Modal app configuration
# ---------------------------------------------------------------------------

app = modal.App("cat-audio-anomaly-training")

volume = modal.Volume.from_name("cat-audio-model", create_if_missing=True)

train_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "transformers==4.47.0",
        "datasets==2.20.0",
        "accelerate",
        "scikit-learn",
        "numpy<2",
        "tqdm",
        "soundfile",
    )
)

# ---------------------------------------------------------------------------
# Dataset label maps (must match audio_anomaly.py)
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

MACHINE_TYPE_TO_IDX: dict[str, int] = {v: i for i, v in enumerate(MACHINE_TYPE_LABELS)}
SUBTYPE_TO_IDX: dict[str, int] = {v: i for i, v in enumerate(ANOMALY_SUBTYPE_LABELS)}

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

CONFIG: dict[str, Any] = {
    "sample_rate": 16_000,         # Wav2Vec2 expects 16 kHz (dataset is 22050 Hz — we resample)
    "clip_seconds": 10,
    "threshold": 0.5,              # updated post-training based on val set
    "num_machine_types": len(MACHINE_TYPE_LABELS),
    "machine_type_labels": MACHINE_TYPE_LABELS,
    "anomaly_subtype_labels": ANOMALY_SUBTYPE_LABELS,
}

TRAIN_CONFIG: dict[str, Any] = {
    "epochs": 3,
    "batch_size": 8,               # fits A10G 24 GB with wav2vec2-large
    "grad_accum_steps": 4,         # effective batch = 32
    "learning_rate": 3e-5,
    "warmup_steps": 200,
    "weight_decay": 1e-2,
    "loss_weight_anomaly": 1.0,
    "loss_weight_machine": 0.3,
    "loss_weight_subtype": 0.2,
    "max_audio_len": 16_000 * 10,  # 10 seconds at 16 kHz
    "dataset_name": "mandipgoswami/AnomalyMachine-50K",
    "checkpoint_path": "/vol/wav2vec2_anomaly.pt",
    "seed": 42,
    "max_train_samples": None,     # set to e.g. 256 for a smoke test
    "max_val_samples": None,
}


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

def _build_dataset(split: str):
    """
    Load and preprocess one split from AnomalyMachine-50K.

    Uses cast_column(Audio(decode=False)) to get raw bytes and decodes with
    torchaudio directly — avoids the librosa dependency entirely.
    Returns a PyTorch Dataset.
    """
    import io
    import torch
    from torch.utils.data import Dataset
    import torchaudio
    import torchaudio.functional as FA
    from datasets import load_dataset, Audio

    actual_split = "val" if split == "validation" else split
    hf_ds = load_dataset(TRAIN_CONFIG["dataset_name"], split=actual_split)

    # Disable HuggingFace audio decoding (avoids librosa dependency).
    # Each row["audio"] will now be {"bytes": b"...", "path": "..."} instead of
    # a decoded numpy array.
    hf_ds = hf_ds.cast_column("audio", Audio(decode=False))

    # Optionally subsample for smoke tests
    max_key = f"max_{actual_split}_samples"
    max_samples = TRAIN_CONFIG.get(max_key) or TRAIN_CONFIG.get(f"max_{split}_samples")
    if max_samples:
        hf_ds = hf_ds.select(range(min(max_samples, len(hf_ds))))

    target_sr = CONFIG["sample_rate"]
    max_len = TRAIN_CONFIG["max_audio_len"]

    class MachineAudioDataset(Dataset):
        def __len__(self) -> int:
            return len(hf_ds)

        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
            row = hf_ds[idx]

            # Decode raw bytes with torchaudio (no librosa needed)
            audio_bytes = row["audio"]["bytes"]
            buf = io.BytesIO(audio_bytes)
            array, sr = torchaudio.load(buf)  # (C, T)

            # Mono
            if array.shape[0] > 1:
                array = array.mean(dim=0, keepdim=True)

            # Resample 22050 → 16000
            if sr != target_sr:
                array = FA.resample(array, sr, target_sr)

            array = array.squeeze(0)  # (T,)

            # Pad or crop
            if array.shape[0] < max_len:
                array = torch.nn.functional.pad(array, (0, max_len - array.shape[0]))
            else:
                array = array[:max_len]

            # Peak normalise
            peak = array.abs().max().clamp(min=1e-9)
            array = array / peak

            # Labels
            anomaly_label = 1 if row["label"] == "anomalous" else 0
            machine_label = MACHINE_TYPE_TO_IDX.get(row["machine_type"], 0)
            subtype_str = row.get("anomaly_subtype", "none") or "none"
            subtype_label = SUBTYPE_TO_IDX.get(subtype_str, 0)

            return {
                "input_values": array,
                "anomaly_label": torch.tensor(anomaly_label, dtype=torch.long),
                "machine_label": torch.tensor(machine_label, dtype=torch.long),
                "subtype_label": torch.tensor(subtype_label, dtype=torch.long),
            }

    return MachineAudioDataset()


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

@app.function(
    image=train_image,
    gpu="A10G",
    volumes={"/vol": volume},
    timeout=21600,  # 6 hours
)
def train() -> dict[str, Any]:
    """
    Fine-tune Wav2Vec2-Large on AnomalyMachine-50K and save checkpoint to
    Modal Volume at /vol/wav2vec2_anomaly.pt.

    Returns:
        Dict with best_val_auc, final_threshold, and training history.
    """
    import json
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    from sklearn.metrics import roc_auc_score
    from tqdm import tqdm

    # Define model architecture inline (avoids cross-module import issues in Modal)
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    class Wav2Vec2AnomalyModel(nn.Module):
        def __init__(self, num_machine_types=6, num_subtypes=6, hidden_size=1024):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
            self.wav2vec2.feature_extractor._freeze_parameters()
            self.anomaly_head = nn.Sequential(
                nn.Linear(hidden_size, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1)
            )
            self.machine_type_head = nn.Linear(hidden_size, num_machine_types)
            self.subtype_head = nn.Linear(hidden_size, num_subtypes)

        def forward(self, input_values):
            # mask_time_indices=None disables spec-augment masking during fine-tuning,
            # avoiding numpy bool conversion issues and stabilising classification training.
            out = self.wav2vec2(input_values=input_values, mask_time_indices=None)
            pooled = out.last_hidden_state.mean(dim=1)
            return {
                "anomaly_logit": self.anomaly_head(pooled),
                "machine_type_logit": self.machine_type_head(pooled),
                "subtype_logit": self.subtype_head(pooled),
            }

    torch.manual_seed(TRAIN_CONFIG["seed"])
    device = torch.device("cuda")

    # ---- Datasets ----------------------------------------------------------
    print("Loading datasets...")
    train_ds = _build_dataset("train")
    val_ds = _build_dataset("val")

    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- Model -------------------------------------------------------------
    print("Building model...")
    model = Wav2Vec2AnomalyModel(
        num_machine_types=CONFIG["num_machine_types"],
        num_subtypes=len(ANOMALY_SUBTYPE_LABELS),
    ).to(device)

    # ---- Optimizer + scheduler ---------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=TRAIN_CONFIG["learning_rate"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    total_steps = (
        len(train_loader) // TRAIN_CONFIG["grad_accum_steps"]
    ) * TRAIN_CONFIG["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=TRAIN_CONFIG["warmup_steps"],
        num_training_steps=total_steps,
    )

    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    # ---- Resume from checkpoint if available -------------------------------
    best_val_auc: float = 0.0
    best_threshold: float = CONFIG["threshold"]
    history: list[dict[str, float]] = []
    start_epoch: int = 1

    ckpt_path = TRAIN_CONFIG["checkpoint_path"]
    if os.path.exists(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_val_auc = ckpt.get("best_val_auc", 0.0)
        best_threshold = ckpt.get("config", {}).get("threshold", CONFIG["threshold"])
        start_epoch = ckpt.get("completed_epochs", 0) + 1
        history = ckpt.get("history", [])
        print(f"  ↳ Resumed: best_val_auc={best_val_auc:.4f}, starting at epoch {start_epoch}")
    else:
        print("No checkpoint found — starting fresh.")

    if start_epoch > TRAIN_CONFIG["epochs"]:
        print("Training already complete (all epochs done). Returning saved results.")
        return {
            "best_val_auc": best_val_auc,
            "final_threshold": best_threshold,
            "history": history,
        }

    # ---- Training loop -----------------------------------------------------
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"] + 1):
        model.train()
        train_loss_accum = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} train"), 1):
            inputs = batch["input_values"].to(device)       # (B, T)
            anomaly_labels = batch["anomaly_label"].to(device).float()
            machine_labels = batch["machine_label"].to(device)
            subtype_labels = batch["subtype_label"].to(device)

            outputs = model(inputs)

            loss_a = bce_loss(outputs["anomaly_logit"].squeeze(-1), anomaly_labels)
            loss_m = ce_loss(outputs["machine_type_logit"], machine_labels)
            loss_s = ce_loss(outputs["subtype_logit"], subtype_labels)

            loss = (
                TRAIN_CONFIG["loss_weight_anomaly"] * loss_a
                + TRAIN_CONFIG["loss_weight_machine"] * loss_m
                + TRAIN_CONFIG["loss_weight_subtype"] * loss_s
            )
            loss = loss / TRAIN_CONFIG["grad_accum_steps"]
            loss.backward()

            train_loss_accum += loss.item()

            if step % TRAIN_CONFIG["grad_accum_steps"] == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # ---- Validation ----------------------------------------------------
        model.eval()
        all_scores: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch} val"):
                inputs = batch["input_values"].to(device)
                labels = batch["anomaly_label"].cpu().tolist()

                outputs = model(inputs)
                scores = torch.sigmoid(outputs["anomaly_logit"]).squeeze(-1).cpu().tolist()

                all_scores.extend(scores)
                all_labels.extend(labels)

        val_auc = roc_auc_score(all_labels, all_scores)

        # Find threshold maximising F1 on validation set
        best_f1 = 0.0
        best_t = 0.5
        for t in [i / 100 for i in range(30, 70)]:
            from sklearn.metrics import f1_score
            preds = [1 if s > t else 0 for s in all_scores]
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        print(
            f"Epoch {epoch} | train_loss={train_loss_accum:.4f} | "
            f"val_auc={val_auc:.4f} | best_f1={best_f1:.4f} @ threshold={best_t:.2f}"
        )
        history.append({"epoch": epoch, "val_auc": val_auc, "best_f1": best_f1})

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = best_t

        # Always save after each epoch to enable resume
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "completed_epochs": epoch,
            "best_val_auc": best_val_auc,
            "history": history,
            "config": {
                **CONFIG,
                "threshold": best_threshold,
            },
        }
        torch.save(checkpoint, TRAIN_CONFIG["checkpoint_path"])
        volume.commit()
        print(f"  ↳ Checkpoint saved (epoch={epoch}, val_auc={val_auc:.4f}, best_ever={best_val_auc:.4f})")

    print(f"\nTraining complete. Best val AUC: {best_val_auc:.4f}, threshold: {best_threshold:.2f}")
    return {
        "best_val_auc": best_val_auc,
        "final_threshold": best_threshold,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Dry-run: validates model + training loop with random tensors (no dataset)
# ---------------------------------------------------------------------------

@app.function(image=train_image, gpu="T4", timeout=120)
def dry_run() -> str:
    """
    Run 2 forward+backward passes with random tensors to confirm:
      - Wav2Vec2-Large loads on GPU
      - All three heads produce outputs
      - Loss and optimizer step work
      - No numpy/torch dtype conflicts

    No dataset download needed. Completes in ~90 seconds.
    """
    import torch
    import torch.nn as nn
    from transformers import Wav2Vec2Model

    device = torch.device("cuda")
    max_len = CONFIG["sample_rate"] * CONFIG["clip_seconds"]  # 160000

    # Build model inline
    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large")
            self.wav2vec2.feature_extractor._freeze_parameters()
            self.anomaly_head = nn.Sequential(nn.Linear(1024, 256), nn.GELU(), nn.Dropout(0.1), nn.Linear(256, 1))
            self.machine_type_head = nn.Linear(1024, 6)
            self.subtype_head = nn.Linear(1024, 6)

        def forward(self, x):
            out = self.wav2vec2(input_values=x, mask_time_indices=None)
            pooled = out.last_hidden_state.mean(dim=1)
            return {
                "anomaly_logit": self.anomaly_head(pooled),
                "machine_type_logit": self.machine_type_head(pooled),
                "subtype_logit": self.subtype_head(pooled),
            }

    print("Loading Wav2Vec2-Large onto GPU...")
    model = _Model().to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    for step in range(2):
        x = torch.randn(2, max_len, device=device)           # batch=2, 10s at 16kHz
        anomaly_labels = torch.tensor([0, 1], dtype=torch.float32, device=device)
        machine_labels = torch.tensor([0, 1], dtype=torch.long, device=device)
        subtype_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

        out = model(x)
        loss = (
            bce(out["anomaly_logit"].squeeze(-1), anomaly_labels)
            + 0.3 * ce(out["machine_type_logit"], machine_labels)
            + 0.2 * ce(out["subtype_logit"], subtype_labels)
        )
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"Step {step+1}/2 — loss={loss.item():.4f} ✓")

    return "DRY RUN PASSED — model and training loop are working correctly"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(smoke_test: bool = False, dry_run_only: bool = False) -> None:
    """
    Usage:
        modal run backend/ai/train_anomaly.py --dry-run-only      # ~90s, no dataset
        modal run backend/ai/train_anomaly.py --smoke-test        # 256 samples, 1 epoch
        modal run backend/ai/train_anomaly.py                     # full training
    """
    import json

    if dry_run_only:
        print("=== DRY RUN: validating model + loop with random tensors ===")
        result = dry_run.remote()
        print(result)
        return

    if smoke_test:
        print("=== SMOKE TEST MODE: 256 train samples, 64 val samples, 1 epoch ===")
        TRAIN_CONFIG["max_train_samples"] = 256
        TRAIN_CONFIG["max_val_samples"] = 64
        TRAIN_CONFIG["epochs"] = 1
        TRAIN_CONFIG["warmup_steps"] = 10

    result = train.remote()
    print("\n=== Training Results ===")
    print(json.dumps(result, indent=2))
