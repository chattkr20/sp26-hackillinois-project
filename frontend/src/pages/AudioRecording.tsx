import { Mic } from 'lucide-react';
import './AudioRecording.css'
import { useState, useEffect, useRef, useCallback } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useNavigate } from 'react-router-dom';

const ANOMALY_API = 'https://milindkumar1--cat-audio-anomaly-detect-anomaly.modal.run';
const STT_API = 'https://milindkumar1--cat-speech-to-text-transcribe.modal.run';

type ActiveRecording = 'machineTest' | 'description' | 'partName' | null;

const log = (...args: any[]) => console.log('[CAT-REC]', ...args);
const err = (...args: any[]) => console.error('[CAT-REC]', ...args);

// ── Native MediaRecorder helpers ─────────────────────────────────────────────

function useNativeRecorder() {
    const mrRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const pendingStopRef = useRef(false);   // stop() called while getUserMedia was still pending
    const [recStatus, setRecStatus] = useState<string>('idle');

    const start = useCallback(async (onStop: (blob: Blob) => void) => {
        log('start() called');
        pendingStopRef.current = false;

        // Stop any existing recorder first
        if (mrRef.current && mrRef.current.state !== 'inactive') {
            mrRef.current.stop();
        }

        let stream: MediaStream;
        try {
            log('Requesting getUserMedia...');
            setRecStatus('requesting-mic');
            stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            log('getUserMedia OK, tracks:', stream.getAudioTracks().map(t => `${t.label} readyState=${t.readyState}`));
        } catch (e: any) {
            err('getUserMedia FAILED:', e.name, e.message);
            setRecStatus(`error: ${e.name}`);
            throw e;
        }

        // If stop() was called while we were waiting for mic permission, abort cleanly
        if (pendingStopRef.current) {
            log('Pending stop detected — aborting before start');
            stream.getTracks().forEach(t => t.stop());
            setRecStatus('idle');
            return;
        }

        // Pick best supported MIME type
        const mimeTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/mp4', ''];
        const mimeType = mimeTypes.find(m => m === '' || MediaRecorder.isTypeSupported(m)) ?? '';
        log('Using MIME type:', mimeType || '(browser default)');

        const mr = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
        log('MediaRecorder created, state:', mr.state, 'actual mimeType:', mr.mimeType);

        chunksRef.current = [];
        mr.ondataavailable = (e) => {
            log(`ondataavailable: size=${e.data.size}, total chunks=${chunksRef.current.length + (e.data.size > 0 ? 1 : 0)}`);
            if (e.data.size > 0) chunksRef.current.push(e.data);
        };
        mr.onerror = (e: any) => {
            err('MediaRecorder onerror:', e);
            setRecStatus(`mr-error: ${e.error?.message ?? String(e)}`);
        };
        mr.onstop = () => {
            log('onstop fired. chunks:', chunksRef.current.length, 'bytes:', chunksRef.current.reduce((s, b) => s + b.size, 0));
            stream.getTracks().forEach(t => t.stop());
            const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' });
            log('Final blob — size:', blob.size, 'type:', blob.type);
            setRecStatus(`done: ${blob.size}b`);
            onStop(blob);
        };

        mrRef.current = mr;
        mr.start(250);
        log('mr.start(250) called, state:', mr.state);
        setRecStatus('recording');
    }, []);

    const stop = useCallback(() => {
        log('stop() called. mrRef state:', mrRef.current?.state ?? 'none');
        if (mrRef.current && mrRef.current.state === 'recording') {
            mrRef.current.stop();
        } else {
            // getUserMedia may still be pending — flag it so start() aborts when it resolves
            log('Recorder not ready yet — setting pendingStop flag');
            pendingStopRef.current = true;
        }
    }, []);

    return { start, stop, recStatus };
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

    const { start: nativeStart, stop: nativeStop, recStatus } = useNativeRecorder();

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

    // Restart speech recognition after a native recording finishes (if voice mode is on)
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
        machineTest: 'REC — Machine Test Audio',
        description: 'REC — Description Audio',
        partName: 'LISTENING — Part Name',
    };

    return (
        <div className='rec-root'>
            <div className='login-bg-grid' />

            <header className='login-header'>
                <div className='login-app-title'>
                    <span className='title-cat'>CAT</span>
                    <span className='title-inspect'>INSPECT</span>
                </div>
                <span className='rec-step-label'>▸ STEP 2 · CAPTURE AUDIO</span>
            </header>

            <main className='rec-main'>
                <div className='rec-card'>
                    <div className='rec-card-header'>
                        <p className='rec-card-eyebrow'>▸ AUDIO CAPTURE</p>
                        <h1 className='rec-card-title'>Record Inspection</h1>
                    </div>

                    <div className='rec-card-body'>
                        {/* ── DIAGNOSTIC PANEL — remove before production ── */}
                        <pre style={{fontSize:'11px',background:'#111',color:'#0f0',padding:'8px',borderRadius:'4px',marginBottom:'12px',overflowX:'auto'}}>
{`recStatus       : ${recStatus}
machineTestDone : ${machineTestDone} | blob: ${machineTestBlobRef.current?.size ?? 'none'}b
descriptionDone : ${descriptionDone} | blob: ${descriptionBlobRef.current?.size ?? 'none'}b
partName        : ${partName ?? '(empty)'}
activeRecording : ${activeRecordingDisplay ?? 'none'}`}
                        </pre>
                        {/* ─────────────────────────────────────────────── */}
                        {micError && <div className='rec-badge rec-badge-error'>{micError}</div>}

                        {/* Mic button — activates voice command listening only */}
                        <div className='rec-mic-row'>
                            <div
                                className={`rec-mic-btn ${recordingStarted ? 'active' : ''}`}
                                onClick={micPressed}
                                role='button'
                                aria-label={recordingStarted ? 'Stop voice commands' : 'Start voice commands'}
                            >
                                <Mic color={recordingStarted ? '#FFCD11' : '#555'} size={34} />
                            </div>
                            <span className={`rec-listen-label ${recordingStarted ? 'active' : ''}`}>
                                {recordingStarted ? '● Listening for commands' : 'Tap to enable voice commands'}
                            </span>
                        </div>

                        {recordingStarted && (
                            <div className='rec-transcript'>
                                {transcript || <em style={{ color: 'var(--text-muted)' }}>say a command...</em>}
                            </div>
                        )}

                        {activeRecordingDisplay && (
                            <div className='rec-badge rec-badge-recording'>
                                {activeLabels[activeRecordingDisplay]}
                            </div>
                        )}

                        <p className='rec-section-label'>Recordings</p>

                        <div className='rec-row'>
                            <span className='rec-row-label'>Machine Test</span>
                            {activeRecordingDisplay === 'machineTest' ? (
                                <button className='rec-btn-stop' onClick={endMachineTest} disabled={recStatus === 'requesting-mic'}>⏹ Stop</button>
                            ) : (
                                <button className='rec-btn-start' onClick={beginMachineTest} disabled={activeRecordingDisplay !== null}>▶ Start</button>
                            )}
                        </div>

                        <div className='rec-row'>
                            <span className='rec-row-label'>Description</span>
                            {activeRecordingDisplay === 'description' ? (
                                <button className='rec-btn-stop' onClick={endDescription} disabled={recStatus === 'requesting-mic'}>⏹ Stop</button>
                            ) : (
                                <button className='rec-btn-start' onClick={beginDescription} disabled={activeRecordingDisplay !== null}>▶ Start</button>
                            )}
                        </div>

                        <div className='rec-status-list'>
                            <div className={`rec-status-item ${machineTestDone ? 'done' : ''}`}>
                                {machineTestDone ? '✓' : '○'} Machine Test Audio
                            </div>
                            <div className={`rec-status-item ${descriptionDone ? 'done' : ''}`}>
                                {descriptionDone ? '✓' : '○'} Description Audio
                            </div>
                            {partName && (
                                <div className='rec-status-item done'>✓ Part: <strong>{partName}</strong></div>
                            )}
                        </div>

                        <button className='rec-btn-submit' onClick={submitData} disabled={isSubmitting || !machineTestDone || !descriptionDone}>
                            <span>{isSubmitting ? 'Submitting…' : 'Submit Inspection'}</span>
                            <span>⟶</span>
                        </button>

                        {submitError && <div className='rec-error'>{submitError}</div>}
                        {isSubmitting && <div className='rec-badge rec-badge-submitting'>Sending to AI analysis...</div>}

                        <div className='rec-hint'>
                            <strong>Voice commands:</strong>
                            <ul>
                                <li>"start machine test" → "stop machine test"</li>
                                <li>"start description" → "stop description"</li>
                                <li>"start part name" [say name] "stop part name"</li>
                                <li>"confirm" or "submit"</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </main>

            <footer className='login-footer'>
                © {new Date().getFullYear()} Caterpillar Inc. · Internal Use Only
            </footer>
        </div>
    );
}
