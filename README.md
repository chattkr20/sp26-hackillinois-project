# CATalyze

**Live Demo**: [https://hackillinois26.milindkumar.dev](https://hackillinois26.milindkumar.dev)

> **HackIllinois 2026 — Caterpillar Track** | *Best AI Inspection*
| **HackIllinois Prizes** | Most Creative, Best UI/UX Design |
| **Company Prizes** | Best Use of OpenAI API |

---

## What We Built

CATalyze is an AI-powered field inspection assistant that merges the capabilities of **CAT Inspect** and **CAT AI Assistant** into a single, voice-first, multimodal application. It lets a machine operator conduct a full equipment walkaround **hands-free** — speaking naturally, recording machine sounds, and snapping photos — while AI handles transcription, anomaly detection, and report generation in real time.

**The problem it solves:** Traditional CAT equipment inspections require inspectors to carry clipboards, manually notate findings, and later write structured reports. This is slow, error-prone, and forces skilled personnel to spend time on documentation instead of safety-critical observation. CATalyze eliminates the paperwork entirely.

**The output:** A fully structured, standards-aligned PDF inspection report — identical in format to CAT Inspect's standard reports — generated automatically from voice + audio + image inputs, with no manual data entry.

---

## AI Disclaimer

> **CATalyze is designed to assist trained CAT equipment inspectors — not replace them.**
>
> All AI-generated assessments (acoustic anomaly detection, image analysis, and report generation) are probabilistic and may produce false positives or false negatives. **Do not make safety-critical maintenance or operational decisions solely based on this tool's output.** All findings must be reviewed by a qualified technician before action is taken.
>
> Models were fine-tuned on limited training data and are intended for demonstration purposes at HackIllinois 2026.

---

## How It Addresses the CAT Track Challenge

| CAT Track Goal | CATalyze Approach |
|---|---|
| Reduce manual effort | Fully voice-driven — no typing, no forms, no clipboard |
| Improve safety | Acoustic anomaly detection catches faults the human ear may miss |
| Reduce on-site risk | Operators stay at safe distance; AI analyzes audio + image remotely |
| Structured inspection summary | PDF report aligned with CAT Inspect standard report format |
| Multimodal AI | Whisper (voice) + Wav2Vec2 (acoustic) + EfficientNet (visual) + Phi-3.5 (reasoning) |
| Visual parts identification | EfficientNet fine-tuned to detect visual defects from a single photo |
| Reduce SME overhead | Non-expert operators can conduct and document inspections with AI guidance |

---

## Inspection Flow

```
Operator puts on headset and opens CATalyze
        │
        ▼
1. Identifies part by voice → "Part name: fan"
        │
        ▼
2. Records machine sound → AI detects acoustic anomalies (imbalance, obstruction, etc.)
        │
        ▼
3. Describes the part verbally → Whisper transcribes in real time
        │
        ▼
4. Snaps a photo → EfficientNet detects visual defects
        │
        ▼
5. Says "submit" → All three signals sent to LLM reasoning layer
        │
        ▼
6. Phi-3.5-mini synthesizes findings → Structured PDF report generated
        │
        ▼
7. Report downloaded or shared instantly — no manual documentation
```

---

## AI Stack & Model Choices

### Why These Models

| Signal | Model | Why |
|---|---|---|
| Voice transcription | Whisper large-v3 | SOTA multilingual ASR; handles field noise, accents, technical vocabulary |
| Acoustic anomaly | Wav2Vec2-Large (fine-tuned) | Self-supervised audio rep; fine-tuned on CAT machine fault patterns to detect imbalance, bearing wear, obstruction |
| Visual anomaly | EfficientNet-B0 (fine-tuned) | Compact, fast CNN; fine-tuned binary classifier (normal vs. defect) on equipment imagery |
| Report reasoning | Phi-3.5-mini-instruct | Efficient 3.8B SLM on A10G; reasons over all three signals to produce structured, actionable findings |

### Fine-Tuning Approach

Both the acoustic and visual models were trained from pre-trained HuggingFace checkpoints using Modal A100 GPU compute:

- **Wav2Vec2**: Feature extractor frozen, classification head fine-tuned on audio samples labeled by fault type (normal, imbalance, obstruction, bearing fault)
- **EfficientNet-B0**: Convolutional base frozen, 4-layer classification head fine-tuned on equipment imagery (normal vs. anomaly) with dropout regularization

---

## Architecture

```
Browser (React 19 + TypeScript)
        │
        ├─► Whisper large-v3        (Modal · T4)   ← voice description
        ├─► Wav2Vec2-Large ft       (Modal · T4)   ← machine audio
        ├─► EfficientNet-B0 ft      (Modal · T4)   ← inspection photo
        └─► Phi-3.5-mini-instruct   (Modal · A10G) ← synthesize → PDF report
```

All inference runs on Modal serverless GPU containers — zero infrastructure to manage, cold-start under 5 seconds, scales to zero between inspections.

---

## Tech Stack

### Frontend

| Technology | Purpose |
|---|---|
| React 19 + TypeScript | UI framework |
| Vite 7 | Build tool |
| React Router v7 | Page routing |
| react-speech-recognition | Real-time voice command capture |
| MediaRecorder API | Native audio recording (WebM) |
| lucide-react | Icons |
| Cloudflare Pages | Hosting with auto-deploy on push |

### Backend (Modal Serverless GPU)

| Technology | Purpose |
|---|---|
| Modal | Serverless GPU container orchestration |
| OpenAI Whisper large-v3 | Speech-to-text transcription |
| Wav2Vec2-Large (fine-tuned) | Acoustic fault detection |
| EfficientNet-B0 (fine-tuned) | Visual defect classification |
| Microsoft Phi-3.5-mini-instruct | LLM reasoning + report generation |
| fpdf2 | PDF generation |
| FastAPI / ASGI | HTTP API layer |

### Training Infrastructure

| Technology | Purpose |
|---|---|
| PyTorch | Model training |
| Hugging Face Transformers | Pre-trained checkpoints |
| Modal A100 | Training compute |
| Modal Volumes | Checkpoint storage and serving |

---

## Project Structure

```
hackillinois26/
├── frontend/
│   └── src/pages/
│       ├── LoginPage.tsx        # Operator profile (name, ID)
│       ├── AudioRecording.tsx   # Hands-free voice + audio + photo capture
│       ├── LLM-Check.tsx        # Orchestrates all AI endpoint calls
│       └── ReportDisplay.tsx    # PDF preview + download
├── backend/
│   ├── prompt.txt               # LLM system prompt (baked into Modal container)
│   └── ai/
│       ├── speech_to_text.py    # Whisper endpoint
│       ├── audio_anomaly.py     # Wav2Vec2 acoustic anomaly endpoint
│       ├── image_anomaly.py     # EfficientNet visual anomaly endpoint
│       ├── report_generator.py  # Phi-3.5 reasoning + PDF generation endpoint
│       ├── train_anomaly.py     # Audio model training script
│       └── train_image_anomaly.py # Image model training script
└── README.md
```

---

## Running Locally

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Use **Chrome** (best MediaRecorder + SpeechRecognition API support).

### Backend

```bash
source venv/bin/activate

modal deploy backend/ai/speech_to_text.py
modal deploy backend/ai/audio_anomaly.py
modal deploy backend/ai/image_anomaly.py
modal deploy backend/ai/report_generator.py
```

Model checkpoints are stored in Modal Volumes (`cat-audio-model`, `cat-image-model`) and loaded lazily on first request.
