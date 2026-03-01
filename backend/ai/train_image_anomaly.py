"""
Modal.com training script: fine-tunes EfficientNet-B0 on CAT equipment images.

Downloads Pass/Fail images from GitHub, trains with heavy augmentation,
saves checkpoint to the shared Modal Volume ('cat-image-model').

Run with:
    modal run backend/ai/train_image_anomaly.py

The checkpoint is then picked up automatically by image_anomaly.py at:
    /vol/efficientnet_anomaly.pt

Training strategy for small datasets (14 images):
  - EfficientNet-B0 backbone pretrained on ImageNet (frozen except block7)
  - New 2-class head trained from scratch
  - 300 epochs with very aggressive augmentation
  - Weighted sampler & weighted CE loss for class imbalance
  - Best-val-accuracy checkpoint saved
"""

import io
import urllib.request
import urllib.parse

import modal

app = modal.App("cat-image-train")

volume = modal.Volume.from_name("cat-image-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.2.2",
        "torchvision==0.17.2",
        "Pillow",
        "numpy<2",
    )
)

# ── GitHub raw image URLs ─────────────────────────────────────────────────────

BASE_URL = "https://raw.githubusercontent.com/ginocorrales/HackIL26-CATrack/main"

PASS_FILES = [
    "BrokenRimBolt1.jpg",
    "BrokenRimBolt2.jpg",
    "CoolantReservoir.jpg",
    "GoodStep.jpg",
    "HousingSeal.jpg",
    "HydraulicFluidFiltrationSystem.jpg",
    "HydraulicFluidTank.jpg",
    "HydraulicHose.jpg",
]

FAIL_FILES = [
    "CoolingSystemHose.jpg",
    "DamagedAccessLadder.jpg",
    "HydraulicFluidFiltration.jpg",
    "RustOnHydraulicComponentBracket.jpg",
    "StructuralDamage.jpg",
    "Tire ShowsSignsUnevenWear.jpg",
]

CHECKPOINT_PATH = "/vol/efficientnet_anomaly.pt"


# ── Training function ─────────────────────────────────────────────────────────

@app.function(
    image=image,
    gpu="T4",
    volumes={"/vol": volume},
    timeout=600,
)
def train():
    import random
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    import torchvision.models as models
    import torchvision.transforms as T
    from PIL import Image

    # ── 1. Download images ────────────────────────────────────────────────────
    print("=" * 60)
    print("Downloading training images from GitHub...")
    print("=" * 60)

    samples: list[tuple[Image.Image, int]] = []  # (image, label): 0=pass, 1=fail

    for split, files, label in [("Pass", PASS_FILES, 0), ("Fail", FAIL_FILES, 1)]:
        for fname in files:
            url = f"{BASE_URL}/{split}/{urllib.parse.quote(fname)}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                samples.append((img, label))
                print(f"  ✓ [{split}] {fname}  ({img.size[0]}×{img.size[1]})")
            except Exception as e:
                print(f"  ✗ [{split}] {fname}  ERROR: {e}")

    n_pass = sum(1 for _, l in samples if l == 0)
    n_fail = sum(1 for _, l in samples if l == 1)
    print(f"\nDownloaded: {len(samples)} total  ({n_pass} pass, {n_fail} fail)")

    if len(samples) < 4:
        raise RuntimeError(f"Not enough images ({len(samples)}) — need at least 4.")
    if n_pass == 0 or n_fail == 0:
        raise RuntimeError("Need at least one image per class.")

    # ── 2. Train/val split ────────────────────────────────────────────────────
    random.seed(42)
    # Stratified split: keep one of each class in val
    pass_imgs = [s for s in samples if s[1] == 0]
    fail_imgs = [s for s in samples if s[1] == 1]
    random.shuffle(pass_imgs)
    random.shuffle(fail_imgs)

    val_pass = pass_imgs[:1]
    val_fail = fail_imgs[:1]
    train_pass = pass_imgs[1:]
    train_fail = fail_imgs[1:]

    train_items = train_pass + train_fail
    val_items   = val_pass   + val_fail
    random.shuffle(train_items)

    print(f"Train: {len(train_items)} images  |  Val: {len(val_items)} images")

    # ── 3. Datasets ───────────────────────────────────────────────────────────
    # Aggressive augmentation for small dataset
    train_tf = T.Compose([
        T.Resize((256, 256)),
        T.RandomCrop(224),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=30),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.15),
        T.RandomGrayscale(p=0.1),
        T.RandomPerspective(distortion_scale=0.2, p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
        T.RandomErasing(p=0.2),  # occlusion robustness
    ])

    val_tf = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225]),
    ])

    class InMemoryDataset(Dataset):
        def __init__(self, items, transform):
            self.items = items
            self.transform = transform

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            img, label = self.items[i]
            return self.transform(img), label

    train_ds = InMemoryDataset(train_items, train_tf)
    val_ds   = InMemoryDataset(val_items,   val_tf)

    # Weighted sampler to handle class imbalance in mini-batches
    train_labels = [lbl for _, lbl in train_items]
    class_counts  = [train_labels.count(0), train_labels.count(1)]
    sample_weights = [1.0 / max(class_counts[lbl], 1) for lbl in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=4, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)

    # ── 4. Model ──────────────────────────────────────────────────────────────
    device = torch.device("cuda")

    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Freeze all backbone features
    for param in model.features.parameters():
        param.requires_grad = False

    # Unfreeze last feature block (block8 = AdaptiveAvgPool preceding classifier)
    # and block7 (the last MBConv block) for extra expressiveness
    for block_idx in [7, 8]:
        for param in model.features[block_idx].parameters():
            param.requires_grad = True

    # Replace classifier: 1280 → 2
    in_features = model.classifier[1].in_features  # 1280 for B0
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 256),
        nn.SiLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, 2),
    )
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: EfficientNet-B0  |  Trainable: {trainable:,} / {total_params:,} params")

    # ── 5. Optimizer & loss ───────────────────────────────────────────────────
    head_params   = list(model.classifier.parameters())
    block7_params = list(model.features[7].parameters())
    block8_params = list(model.features[8].parameters())

    optimizer = optim.AdamW([
        {"params": head_params,   "lr": 1e-3},
        {"params": block7_params, "lr": 5e-5},
        {"params": block8_params, "lr": 5e-5},
    ], weight_decay=1e-4)

    # Weighted CE loss: up-weight the minority class
    total = len(train_items)
    class_weights = torch.tensor([
        total / (2 * max(class_counts[0], 1)),
        total / (2 * max(class_counts[1], 1)),
    ], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    EPOCHS = 300
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)

    # ── 6. Training loop ──────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs...\n")

    best_val_acc  = 0.0
    best_state    = None

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = train_correct = train_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item() * imgs.size(0)
            train_correct += (out.argmax(1) == labels).sum().item()
            train_total   += imgs.size(0)

        scheduler.step()

        # Validate
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                val_correct += (out := model(imgs)).argmax(1).eq(labels).sum().item()
                val_total   += imgs.size(0)

        val_acc   = val_correct / max(val_total, 1)
        train_acc = train_correct / max(train_total, 1)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{EPOCHS}  "
                f"loss={train_loss/max(train_total,1):.4f}  "
                f"train={train_acc:.1%}  "
                f"val={val_acc:.1%}"
            )

        # Save best checkpoint by val accuracy
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── 7. Save checkpoint ────────────────────────────────────────────────────
    print(f"\nBest validation accuracy: {best_val_acc:.1%}")
    checkpoint = {
        "model_state_dict": best_state,
        "config": {
            "num_classes": 2,
            "threshold": 0.5,
            "labels": ["normal", "anomaly"],
            "architecture": "efficientnet_b0",
        },
        "metrics": {
            "best_val_acc": best_val_acc,
            "train_size": len(train_items),
            "val_size": len(val_items),
            "n_pass": n_pass,
            "n_fail": n_fail,
        },
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
    volume.commit()
    print(f"✓ Checkpoint saved → {CHECKPOINT_PATH}")
    print("Now run: modal deploy backend/ai/image_anomaly.py")


@app.local_entrypoint()
def main():
    train.remote()
