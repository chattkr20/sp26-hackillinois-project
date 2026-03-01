# CATalyze

**Website**: [https://hackillinois26.milindkumar.dev](https://hackillinois26.milindkumar.dev)

## HackIllinois Info

| Field | Value |
|---|---|
| **Track** | Caterpillar |
| **HackIllinois Prizes** | Most Creative, Best UI/UX Design |
| **Company Prizes** | Best Use of OpenAI API |

---

## AI Disclaimer

> **This tool is designed to assist trained CAT equipment inspectors — not replace them.**
>
> All AI-generated assessments (acoustic anomaly detection, image analysis, and report generation) are probabilistic in nature and may produce false positives or false negatives. **Do not make safety-critical maintenance or operational decisions solely based on the output of this tool.** Always have findings reviewed by a qualified technician before taking action.
>
> Models used in this project were fine-tuned on limited training data and are intended for demonstration purposes at HackIllinois 2026.

---

## What It Does

CATalyze lets a CAT machine operator conduct a hands-free equipment inspection and generate a structured status report. The operator walks through each part of the machine, recording:

1. **Machine sound** — native audio of the part running (analyzed for acoustic anomalies)
2. **Description** — the operator speaking about the part (transcribed via Whisper)
3. **Photo** — a snapshot of the part (analyzed by a fine-tuned vision model)

All three signals are combined by an LLM to produce a per-part status (`PASS` / `MONITOR` / `FAIL`) and a full downloadable PDF report.

---

## Architecture

```
Browser (React + TypeScript)
        │
        ├─► /speech-to-text    (Modal · Whisper-large-v3 · T4)
        ├─► /audio-anomaly     (Modal · Wav2Vec2-Large fine-tuned · T4)
        ├─► /image-anomaly     (Modal · EfficientNet-B0 fine-tuned · T4)
        └─► /generate-report   (Modal · Phi-3.5-mini-instruct · A10G · fpdf2)
```

---

## Tech Stack

### Frontend

| Technology | Purpose |
|---|---|
| React 19 + TypeScript | UI framework |
| Vite 7 | Build tool / dev server |
| React Router v7 | Client-side routing |
| react-speech-recognition | Voice command capture |
| MediaRecorder API | Native audio recording (WebM) |
| lucide-react | Icons |
| Cloudflare Pages | Hosting (auto-deploys on push) |

### Backend (Modal Serverless)

| Technology | Purpose |
|---|---|
| Modal | Serverless GPU containers |
| OpenAI Whisper large-v3 | Speech-to-text transcription |
| Wav2Vec2-Large (fine-tuned) | Acoustic anomaly detection |
| EfficientNet-B0 (fine-tuned) | Visual anomaly detection |
| Microsoft Phi-3.5-mini-instruct | Report generation LLM |
| fpdf2 | PDF generation |
| FastAPI / ASGI | API layer inside Modal |

### Training

| Technology | Purpose |
|---|---|
| PyTorch | Model training |
| Hugging Face Transformers | Pre-trained model weights |
| Modal A100 | Training compute |

---

## Project Structure

```
hackillinois26/
├── frontend/
│   └── src/
│       └── pages/
│           ├── LoginPage.tsx        # Operator profile form
│           ├── AudioRecording.tsx   # Hands-free recording flow
│           ├── LLM-Check.tsx        # Sends data to AI endpoints
│           └── ReportDisplay.tsx    # PDF download + report view
├── backend/
│   ├── prompt.txt                   # LLM system prompt (baked into Modal image)
│   └── ai/
│       ├── speech_to_text.py        # Whisper endpoint
│       ├── audio_anomaly.py         # Wav2Vec2 acoustic endpoint
│       ├── image_anomaly.py         # EfficientNet visual endpoint
│       ├── report_generator.py      # Phi-3.5 + PDF endpoint
│       ├── train_anomaly.py         # Audio model training (reference)
│       └── train_image_anomaly.py   # Image model training (reference)
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

Use **Chrome** for best compatibility (MediaRecorder + SpeechRecognition APIs).

### Backend

All backend services run on Modal. Deploy each individually:

```bash
source venv/bin/activate

modal deploy backend/ai/speech_to_text.py
modal deploy backend/ai/audio_anomaly.py
modal deploy backend/ai/image_anomaly.py
modal deploy backend/ai/report_generator.py
```

Checkpoints are stored in Modal Volumes (`cat-audio-model`, `cat-image-model`) and loaded lazily on first request.


## Project Info

This web app allows a machine operator to create a status report for a machine, going through each part of the machine and providing a status by analyzing a recording of the operator speaking about the part and a recording of the part itself using AI.

Here's how the flow will work. When the user first begins using this web app, they fill out information about themselves (called Profile information). After that, every time the user wants to create a report about a machine, they press the microphone button and then can go hands-free until the report is created. The user then goes through each part of the machine and states the Part Name, records audio of the part (Machine Sound), and records themselves talking about the part (Description). The user uses voice commands to signal what they intend to record. After giving the phrase to signal submission, the Machine Sound is analyzed for discrepancies and concerning sounds. This analysis is combined with a transcription of the Description to create a final decision on each part consisting of the part's status, description, etc. These decisions about each part are then compiled into a report for the machine as a whole which the user can then view.

## Running the Project Locally

### Running Backend Locally

```
cd backend
```

#### Windows

```
./start.ps1
```

#### Mac / Linux

```
chmod +x start.sh
./start.sh
```

### Running Frontend Locally

```
cd frontend
npm i
npm run dev
```

It is recommended you use **Chrome** to run the localhost