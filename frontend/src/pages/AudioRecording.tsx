import { Mic } from 'lucide-react';
import './AudioRecording.css'
import { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useReactMediaRecorder } from 'react-media-recorder';
import { useNavigate } from 'react-router-dom';

const ANOMALY_API = 'https://milindkumar1--cat-audio-anomaly-detect-anomaly.modal.run';
const STT_API = 'https://milindkumar1--cat-speech-to-text-transcribe.modal.run';

type ActiveRecording = 'machineTest' | 'description' | 'partName' | null;


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

    const machineTestBlobUrlRef = useRef<string | null>(null);
    const descriptionBlobUrlRef = useRef<string | null>(null);
    // Which recording slot is this blob for — resolved when onstop fires
    const pendingSlotRef = useRef<'machineTest' | 'description' | null>(null);

    const {
        startRecording,
        stopRecording,
        mediaBlobUrl,
        status: recorderStatus,
    } = useReactMediaRecorder({ audio: true });

    // Route the finished blob to the correct slot
    useEffect(() => {
        if (!mediaBlobUrl) return;
        if (pendingSlotRef.current === 'machineTest') {
            machineTestBlobUrlRef.current = mediaBlobUrl;
            setMachineTestDone(true);
        } else if (pendingSlotRef.current === 'description') {
            descriptionBlobUrlRef.current = mediaBlobUrl;
            setDescriptionDone(true);
        }
        pendingSlotRef.current = null;
    }, [mediaBlobUrl]);

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

    const beginPartName = () => {
        if (activeRecordingRef.current !== null) return;
        setActive('partName');
        resetTranscript();
    };

    const endPartName = (currentTranscript: string) => {
        setPartName(currentTranscript.trim() || null);
        setActive(null);
        resetTranscript();
    };

    const beginMachineTest = () => {
        if (activeRecordingRef.current !== null) return;
        setMachineTestDone(false);
        pendingSlotRef.current = 'machineTest';
        startRecording();
        setActive('machineTest');
        resetTranscript();
    };

    const endMachineTest = () => {
        stopRecording();
        setActive(null);
        resetTranscript();
    };

    const beginDescription = () => {
        if (activeRecordingRef.current !== null) return;
        setDescriptionDone(false);
        pendingSlotRef.current = 'description';
        startRecording();
        setActive('description');
        resetTranscript();
    };

    const endDescription = () => {
        stopRecording();
        setActive(null);
        resetTranscript();
    };

    const submitData = async () => {
        if (!machineTestBlobUrlRef.current || !descriptionBlobUrlRef.current) {
            setSubmitError('Missing recordings — record both machine test and description first.');
            return;
        }
        setIsSubmitting(true);
        setSubmitError(null);
        SpeechRecognition.stopListening();
        setRecordingStarted(false);
        resetTranscript();

        try {
            const [machineTestAudio, descriptionAudio] = await Promise.all([
                fetch(machineTestBlobUrlRef.current).then(r => r.blob()),
                fetch(descriptionBlobUrlRef.current).then(r => r.blob()),
            ]);

            const [anomalyRes, sttRes] = await Promise.all([
                fetch(ANOMALY_API, {
                    method: 'POST',
                    body: machineTestAudio,
                    headers: { 'Content-Type': 'application/octet-stream' },
                }),
                fetch(STT_API, {
                    method: 'POST',
                    body: descriptionAudio,
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
        } catch (e: any) {
            setSubmitError(`Submit failed: ${e.message}`);
        } finally {
            setIsSubmitting(false);
        }
    };

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
                                <button className='rec-btn-stop' onClick={endMachineTest}>⏹ Stop</button>
                            ) : (
                                <button className='rec-btn-start' onClick={beginMachineTest} disabled={activeRecordingDisplay !== null}>▶ Start</button>
                            )}
                        </div>

                        <div className='rec-row'>
                            <span className='rec-row-label'>Description</span>
                            {activeRecordingDisplay === 'description' ? (
                                <button className='rec-btn-stop' onClick={endDescription}>⏹ Stop</button>
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
