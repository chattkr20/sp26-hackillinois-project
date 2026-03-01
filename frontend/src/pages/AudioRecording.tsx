import Microphone from '../components/Mic';
import './AudioRecording.css'
import { useState, useEffect, useRef, useCallback } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useNavigate } from 'react-router-dom';

const ANOMALY_API = 'https://milindkumar1--cat-audio-anomaly-detect-anomaly.modal.run';
const STT_API = 'https://milindkumar1--cat-speech-to-text-transcribe.modal.run';

type ActiveRecording = 'machineTest' | 'description' | 'partName' | null;

// ── Native MediaRecorder helpers ─────────────────────────────────────────────
// We manage one MediaRecorder at a time so there is never a stream conflict.

function useNativeRecorder() {
    const mrRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);

    const start = useCallback(async (onStop: (blob: Blob) => void) => {
        // Stop any existing recorder first
        if (mrRef.current && mrRef.current.state !== 'inactive') {
            mrRef.current.stop();
        }
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mr = new MediaRecorder(stream);
        chunksRef.current = [];
        mr.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
        mr.onstop = () => {
            stream.getTracks().forEach(t => t.stop()); // release mic immediately
            const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' });
            onStop(blob);
        };
        mr.start(100); // collect chunks every 100ms for reliability
        mrRef.current = mr;
    }, []);

    const stop = useCallback(() => {
        if (mrRef.current && mrRef.current.state === 'recording') {
            mrRef.current.stop();
        }
    }, []);

    return { start, stop };
}

// ─────────────────────────────────────────────────────────────────────────────

export default function AudioRecording() {
    const navigate = useNavigate();

    useEffect(() => {
        const operator = localStorage.getItem('catInspectOperator');
        if (!operator) navigate('/login');
    }, []);

    const activeRecordingRef = useRef<ActiveRecording>(null);
    const [activeRecordingDisplay, setActiveRecordingDisplay] = useState<ActiveRecording>(null);
    const [partName, setPartName] = useState<string | null>(null);
    const [machineTestDone, setMachineTestDone] = useState(false);
    const [descriptionDone, setDescriptionDone] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [submitError, setSubmitError] = useState<string | null>(null);
    const [micError, setMicError] = useState<string | null>(null);
    const [recordingStarted, setRecordingStarted] = useState(false);

    // Store raw Blobs instead of blob URLs — avoids a second fetch round-trip
    const machineTestBlobRef = useRef<Blob | null>(null);
    const descriptionBlobRef = useRef<Blob | null>(null);

    const { start: nativeStart, stop: nativeStop } = useNativeRecorder();

    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition,
    } = useSpeechRecognition();

    const setActive = (val: ActiveRecording) => {
        activeRecordingRef.current = val;
        setActiveRecordingDisplay(val);
    };

    // ── Recording actions ────────────────────────────────────────────────────

    const beginPartName = useCallback(() => {
        if (activeRecordingRef.current !== null) return;
        setActive('partName');
        resetTranscript();
    }, [resetTranscript]);

    const endPartName = useCallback((currentTranscript: string) => {
        setPartName(currentTranscript.trim() || null);
        setActive(null);
        resetTranscript();
    }, [resetTranscript]);

    const beginMachineTest = useCallback(async () => {
        if (activeRecordingRef.current !== null) return;
        setMachineTestDone(false);
        setActive('machineTest');
        resetTranscript();
        try {
            await nativeStart((blob) => {
                machineTestBlobRef.current = blob;
                setMachineTestDone(true);
            });
        } catch (err: any) {
            setMicError(`Could not start machine test recording: ${err.message}`);
            setActive(null);
        }
    }, [nativeStart, resetTranscript]);

    const endMachineTest = useCallback(() => {
        nativeStop();
        setActive(null);
        resetTranscript();
    }, [nativeStop, resetTranscript]);

    const beginDescription = useCallback(async () => {
        if (activeRecordingRef.current !== null) return;
        setDescriptionDone(false);
        setActive('description');
        resetTranscript();
        try {
            await nativeStart((blob) => {
                descriptionBlobRef.current = blob;
                setDescriptionDone(true);
            });
        } catch (err: any) {
            setMicError(`Could not start description recording: ${err.message}`);
            setActive(null);
        }
    }, [nativeStart, resetTranscript]);

    const endDescription = useCallback(() => {
        nativeStop();
        setActive(null);
        resetTranscript();
    }, [nativeStop, resetTranscript]);

    const submitData = useCallback(async () => {
        if (!machineTestBlobRef.current || !descriptionBlobRef.current) {
            setSubmitError('Missing recordings — record both machine test and description first.');
            return;
        }
        setIsSubmitting(true);
        setSubmitError(null);
        SpeechRecognition.stopListening();
        setRecordingStarted(false);
        resetTranscript();

        try {
            const [anomalyRes, sttRes] = await Promise.all([
                fetch(ANOMALY_API, {
                    method: 'POST',
                    body: machineTestBlobRef.current,
                    headers: { 'Content-Type': 'application/octet-stream' },
                }),
                fetch(STT_API, {
                    method: 'POST',
                    body: descriptionBlobRef.current,
                    headers: { 'Content-Type': 'application/octet-stream' },
                }),
            ]);

            const [anomalyResult, sttResult] = await Promise.all([anomalyRes.json(), sttRes.json()]);

            if (anomalyResult.error || sttResult.error) {
                setSubmitError(`API error — anomaly: ${anomalyResult.error || 'ok'} | stt: ${sttResult.error || 'ok'}`);
            } else {
                const operator = JSON.parse(localStorage.getItem('catInspectOperator') || '{}');
                navigate('/llm-check', {
                    state: {
                        anomaly: anomalyResult,
                        stt: sttResult,
                        part_name: partName,
                        operator_name: operator.name,
                        operator_id: operator.id,
                    },
                });
            }
        } catch (err: any) {
            setSubmitError(`Submit failed: ${err.message}`);
        } finally {
            setIsSubmitting(false);
        }
    }, [partName, navigate, resetTranscript]);

    // ── Keep actionsRef fresh so voice command handler never has stale closures ──
    const actionsRef = useRef({ beginMachineTest, endMachineTest, beginDescription, endDescription, beginPartName, endPartName, submitData });
    useEffect(() => {
        actionsRef.current = { beginMachineTest, endMachineTest, beginDescription, endDescription, beginPartName, endPartName, submitData };
    });

    // ── Voice command handler ─────────────────────────────────────────────────
    // Debounce so we don't fire the same command multiple times as the transcript
    // grows word-by-word.
    const lastCommandRef = useRef('');
    useEffect(() => {
        if (!transcript) return;
        const t = transcript.toLowerCase();

        const fire = (key: string, fn: () => void) => {
            if (t.includes(key) && lastCommandRef.current !== key + t) {
                lastCommandRef.current = key + t;
                fn();
            }
        };

        if (t.includes('stop machine test')) {
            fire('stop machine test', () => actionsRef.current.endMachineTest());
        } else if (t.includes('start machine test')) {
            fire('start machine test', () => actionsRef.current.beginMachineTest());
        } else if (t.includes('stop description')) {
            fire('stop description', () => actionsRef.current.endDescription());
        } else if (t.includes('start description')) {
            fire('start description', () => actionsRef.current.beginDescription());
        } else if (t.includes('stop part name')) {
            const captured = transcript.substring(0, t.indexOf('stop part name')).trim();
            fire('stop part name', () => actionsRef.current.endPartName(captured));
        } else if (t.includes('start part name')) {
            fire('start part name', () => actionsRef.current.beginPartName());
        } else if (t.includes('confirm') || t.includes('submit')) {
            fire('confirm', () => actionsRef.current.submitData());
        }
    }, [transcript]);

    // ── Auto-restart SpeechRecognition if it drops while mic is on ───────────
    useEffect(() => {
        if (!listening && recordingStarted) {
            SpeechRecognition.startListening({ continuous: true, language: 'en-US' });
        }
    }, [listening, recordingStarted]);

    // ── Mic button — only activates voice commands, NOT audio capture ─────────
    const micPressed = async () => {
        setMicError(null);
        if (!recordingStarted) {
            if (!browserSupportsSpeechRecognition) {
                setMicError('Voice commands not supported on this browser — use the buttons below.');
                return;
            }
            try {
                await SpeechRecognition.startListening({ language: 'en-US', continuous: true });
                setRecordingStarted(true);
            } catch (err: any) {
                setMicError(`Could not start voice commands: ${err.message}`);
            }
        } else {
            SpeechRecognition.stopListening();
            setRecordingStarted(false);
            resetTranscript();
        }
    };

    const activeLabels: Record<string, string> = {
        machineTest: '🔴 Recording: Machine Test',
        description: '🔴 Recording: Description',
        partName: '🔴 Listening: Part Name',
    };

    return (
        <div id='recording-screen'>
            <div id='status-panel'>
                <h2>CAT Inspection Tool</h2>

                {micError && <div className='badge badge-error'>🚫 {micError}</div>}

                <div id='mic-row'>
                    <Microphone micPressed={micPressed} />
                    <span className={`listen-label ${recordingStarted ? 'active' : ''}`}>
                        {recordingStarted ? '🎙️ Listening for commands...' : 'Tap mic for voice commands'}
                    </span>
                </div>

                {recordingStarted && (
                    <div id='transcript-box'>
                        <span className='label'>Heard:</span>
                        {transcript || <em>say a command...</em>}
                    </div>
                )}

                {activeRecordingDisplay && (
                    <div className='badge badge-recording'>
                        {activeLabels[activeRecordingDisplay]}
                    </div>
                )}

                <div id='manual-controls'>
                    <div className='manual-row'>
                        <span className='manual-label'>Machine Test</span>
                        {activeRecordingDisplay === 'machineTest' ? (
                            <button className='btn-stop' onClick={endMachineTest}>⏹ Stop</button>
                        ) : (
                            <button className='btn-start' onClick={beginMachineTest} disabled={activeRecordingDisplay !== null}>▶ Start</button>
                        )}
                    </div>
                    <div className='manual-row'>
                        <span className='manual-label'>Description</span>
                        {activeRecordingDisplay === 'description' ? (
                            <button className='btn-stop' onClick={endDescription}>⏹ Stop</button>
                        ) : (
                            <button className='btn-start' onClick={beginDescription} disabled={activeRecordingDisplay !== null}>▶ Start</button>
                        )}
                    </div>
                    <button className='btn-submit' onClick={submitData} disabled={isSubmitting || !machineTestDone || !descriptionDone}>
                        {isSubmitting ? '⏳ Submitting…' : '✅ Submit'}
                    </button>
                </div>

                <div id='recordings-status'>
                    <div className={`recording-item ${machineTestDone ? 'done' : ''}`}>
                        {machineTestDone ? '✅' : '⬜'} Machine Test Audio
                    </div>
                    <div className={`recording-item ${descriptionDone ? 'done' : ''}`}>
                        {descriptionDone ? '✅' : '⬜'} Description Audio
                    </div>
                    {partName && (
                        <div className='recording-item done'>✅ Part Name: <strong>{partName}</strong></div>
                    )}
                </div>

                <div id='commands-hint'>
                    <strong>Voice commands:</strong>
                    <ul>
                        <li>"start machine test" → "stop machine test"</li>
                        <li>"start description" → "stop description"</li>
                        <li>"start part name" [say name] "stop part name"</li>
                        <li>"confirm" or "submit"</li>
                    </ul>
                </div>

                {submitError && <div className='badge badge-error'>{submitError}</div>}
                {isSubmitting && <div className='badge badge-submitting'>⏳ Submitting to AI...</div>}
            </div>
        </div>
    );
}
