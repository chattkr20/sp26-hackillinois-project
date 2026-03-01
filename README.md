# CATalyze

**Live Demo**: [https://hackillinois26.milindkumar.dev](https://hackillinois26.milindkumar.dev)

> **HackIllinois 2026 — Caterpillar Track** | *Best AI Inspection*

| | |
|---|---|
| **HackIllinois Prizes** | Most Creative, Best UI/UX Design |
| **Company Prizes** | Best Use of OpenAI API |

---

## What We Built

CATalyze is the "baby" of **CAT Inspect** and **CAT AI Assistant** — a next-generation AI-powered field inspection assistant that merges the structured reporting of CAT Inspect with the conversational, multimodal intelligence of CAT AI Assistant into a single, voice-first application. An equipment operator can conduct a full walkaround **hands-free** — speaking naturally, recording machine audio, and snapping a photo — while AI handles transcription, anomaly detection, and report generation in real time.

**The problem it solves:** Traditional CAT equipment inspections require inspectors to carry clipboards, manually notate findings, and later write structured reports. This is slow, error-prone, and forces skilled personnel to spend time on documentation instead of on safety-critical observation. CATalyze eliminates the paperwork entirely.

**The output:** A fully structured, standards-aligned PDF inspection report — mirroring the format of CAT Inspect standard reports — generated automatically from voice + audio + image inputs, with no manual data entry. Each report includes an executive risk summary, color-coded urgency ratings per component, immediate action items, and verbatim operator transcript — all produced automatically in under 30 seconds.

---

## Key Innovations

> These are the novel aspects that distinguish CATalyze from a conventional inspection app.

**1. AI overrides operator blind spots.**
When acoustic or visual sensors detect an anomaly that the operator verbally reports as "normal," CATalyze flags the discrepancy and classifies the finding as IMMEDIATE risk — prioritizing instrument data over human subjectivity. This mimics how a trained CAT technician would interpret conflicting signals.

**2. Multimodal sensor fusion → structured reasoning.**
Three independent AI signals (voice transcript, acoustic anomaly score + subtype, visual defect score) are fused by a reasoning LLM (Phi-3.5-mini) that produces a *single* coherent inspection narrative — not just a list of raw scores. The LLM decides which signals to trust, how to weight them, and what the operator should do next.

**3. Fine-tuned models on CAT-relevant fault types.**
Rather than relying on generic pre-trained models, both the acoustic detector (Wav2Vec2) and visual classifier (EfficientNet-B0) were fine-tuned specifically on equipment fault patterns (imbalance, obstruction, bearing wear for audio; normal vs. defect for imagery). This raises precision on the exact fault types CAT inspectors encounter.

**4. Voice-first, hands-free UX built for field conditions.**
The entire inspection is conducted without touching a screen. Operators speak part names and observations; the system listens, transcribes, and captures audio. This reduces on-site risk by keeping operators' hands free and eyes on the equipment.

**5. Executive-grade inspection reports, automatically.**
Every generated PDF includes an overall risk rating (HIGH / MEDIUM / LOW), a 2-3 sentence executive summary, prioritized immediate action items, and a per-component breakdown with urgency classification. Output is structured to match CAT Inspect standard report conventions.

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
| Reduce manual effort | Fully voice-driven — no typing, no forms, no clipboard; report is auto-generated |
| Improve safety | Acoustic anomaly detection catches faults the human ear may miss; AI overrides operator under-reporting |
| Reduce on-site risk | Operators stay at safe distance; AI analyzes audio + image remotely on Modal GPU |
| Structured inspection summary | PDF report with risk rating, executive summary, immediate actions, per-part urgency — aligned to CAT Inspect standard report format |
| Multimodal AI | Whisper (voice) + Wav2Vec2 (acoustic) + EfficientNet (visual) + Phi-3.5 (LLM reasoning) |
| Visual parts identification | EfficientNet fine-tuned to detect visual defects from a single inspection photo |
| Reduce SME overhead | Non-expert operators conduct and document inspections with AI guidance; no specialist required on-site |
| Documentation quality | LLM synthesizes all signals into narrative findings, corrective actions, and urgency classifications — eliminating vague manual notes |
| AI reasoning (not just scoring) | Phi-3.5 reasons across all three signal sources to produce a coherent, actionable conclusion — not just raw anomaly scores |

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

## Inspection Report Output

Every CATalyze PDF report is structured to mirror a **CAT Inspect standard report**, with LLM-generated narrative insight layered on top:

| Report Section | Content |
|---|---|
| **Overall Risk Rating** | Color-coded badge: HIGH (red) / MEDIUM (amber) / LOW (green) — determined by fusing acoustic + visual sensor thresholds |
| **Executive Summary** | 2-3 sentence narrative synthesizing the most critical finding and its operational impact |
| **Immediate Actions Required** | Up to 3 numbered, urgency-ordered action items naming specific parts and required steps |
| **Inspection Details** | Operator name & ID, component, machine type, timestamp |
| **Acoustic Anomaly Analysis** | Anomaly score (%), fault subtype (imbalance / obstruction / bearing / none), detection confidence |
| **Visual Anomaly Analysis** | Defect score (%), model confidence, detection mode |
| **Inspection Photo** | Embedded image from field capture |
| **Part-by-Part Analysis** | Per-component table: urgency (IMMEDIATE / SCHEDULED / MONITOR, color-coded), AI findings, recommended corrective action |
| **Operator Transcript** | Verbatim voice transcript for audit trail |

**A key design decision:** When AI sensor data contradicts the operator's verbal assessment (e.g., operator says "fan is normal" while acoustic sensors detect 100% confidence imbalance), CATalyze treats the instrument data as authoritative and escalates the risk rating — the same workflow a trained CAT technician would follow.

---

## Judging Criteria Alignment

| Criterion | CATalyze |
|---|---|
| **Innovation & Novelty** | AI-driven contradiction detection (sensors vs. verbal report); multimodal fusion into a single LLM reasoning pass; voice-first hands-free UX for field conditions |
| **Research** | Fine-tuned Wav2Vec2 on CAT equipment fault types; fine-tuned EfficientNet-B0 on equipment imagery; CAT Inspect report format research for PDF schema design |
| **Technical Execution** | Four independent Modal GPU endpoints (Whisper T4, Wav2Vec2 T4, EfficientNet T4, Phi-3.5 A10G) orchestrated from a React frontend; zero-downtime serverless scaling |
| **Impact** | Eliminates manual inspection documentation; enables non-expert operators to generate professional-grade reports; catches faults the human ear and eye may miss |
| **Design** | Voice-first UX with real-time transcript display, visual recording status, and photo capture — designed for one-handed field use; PDF output structured to CAT Inspect conventions |
| **New AI Interfaces & Form Factors** | Hands-free voice command interface; acoustic anomaly detection as a real-time sensor; LLM reasoning layer that synthesizes three independent modalities into a coherent inspection document |

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
