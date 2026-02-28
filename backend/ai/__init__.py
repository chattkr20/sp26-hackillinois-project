"""
Modal AI inference components for the CAT Equipment Inspector.

Modules:
    utils            — shared WebM→WAV conversion and waveform loading
    speech_to_text   — Whisper-based voice transcription → structured inspection JSON
    audio_anomaly    — Wav2Vec2-based machine sound anomaly detection
    train_anomaly    — fine-tuning script for AnomalyMachine-50K
"""
