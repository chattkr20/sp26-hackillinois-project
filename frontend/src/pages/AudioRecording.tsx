import Microphone from '../components/Mic';
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

    const machineTestFileRef = useRef<string | null>(null);
    const descriptionFileRef = useRef<string | null>(null);

    const { startRecording: startMachineTest, stopRecording: stopMachineTest, mediaBlobUrl: machineTestBlob, status: machineTestStatus } = useReactMediaRecorder({ audio: true });
    const { startRecording: startDescription, stopRecording: stopDescription, mediaBlobUrl: descriptionBlob, status: descriptionStatus } = useReactMediaRecorder({ audio: true });

    useEffect(() => {
        if (machineTestBlob) {
            machineTestFileRef.current = machineTestBlob;
            setMachineTestDone(true);
        }
    }, [machineTestBlob]);

    useEffect(() => {
        if (descriptionBlob) {
            descriptionFileRef.current = descriptionBlob;
            setDescriptionDone(true);
        }
    }, [descriptionBlob]);

    const setActive = (val: ActiveRecording) => {
        activeRecordingRef.current = val;
        setActiveRecordingDisplay(val);
    };

    const beginPartName = () => {
        if (activeRecordingRef.current !== null) return;
        setActive('partName');
        resetTranscript();
    }

    const endPartName = () => {
        setPartName(transcript);
        setActive(null);
        resetTranscript();
    }

    const beginMachineTest = () => {
        if (activeRecordingRef.current !== null) return;
        startMachineTest();
        setMachineTestDone(false);
        setActive('machineTest');
        resetTranscript();
    }

    const endMachineTest = () => {
        stopMachineTest();
        setActive(null);
        resetTranscript();
    }

    const beginDescription = () => {
        if (activeRecordingRef.current !== null) return;
        startDescription();
        setDescriptionDone(false);
        setActive('description');
        resetTranscript();
    }

    const endDescription = () => {
        stopDescription();
        setActive(null);
        resetTranscript();
    }

    const submitData = async () => {
        if (!machineTestFileRef.current || !descriptionFileRef.current) {
            setSubmitError('Missing recordings — record both machine test and description first.');
            return;
        }
        setIsSubmitting(true);
        setSubmitError(null);

        // Auto-stop mic immediately when submit is triggered
        SpeechRecognition.stopListening();
        setRecordingStarted(false);
        resetTranscript();

        try {
            const [machineTestAudio, descriptionAudio] = await Promise.all([
                fetch(machineTestFileRef.current).then(r => r.blob()),
                fetch(descriptionFileRef.current).then(r => r.blob()),
            ]);

            const [anomalyRes, sttRes] = await Promise.all([
                fetch(ANOMALY_API, { method: 'POST', body: machineTestAudio, headers: { 'Content-Type': 'application/octet-stream' } }),
                fetch(STT_API, { method: 'POST', body: descriptionAudio, headers: { 'Content-Type': 'application/octet-stream' } }),
            ]);

            const [anomalyResult, sttResult] = await Promise.all([anomalyRes.json(), sttRes.json()]);

            if (anomalyResult.error || sttResult.error) {
                setSubmitError(`API error — anomaly: ${anomalyResult.error || 'ok'} | stt: ${sttResult.error || 'ok'}`);
            } else {
                // Navigate to LLM check / report generation screen
                const operator = JSON.parse(localStorage.getItem('catInspectOperator') || '{}');
                navigate('/llm-check', {
                    state: {
                        anomaly: anomalyResult,
                        stt: sttResult,
                        part_name: partName,
                        operator_name: operator.name,
                        operator_id: operator.id,
                    }
                });
            }
        } catch (err: any) {
            setSubmitError(`Submit failed: ${err.message}`);
        } finally {
            setIsSubmitting(false);
        }
    }

    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition
    } = useSpeechRecognition();

    // Use refs so the polling useEffect always sees current callbacks without stale closures
    const actionsRef = useRef({
        beginMachineTest, endMachineTest,
        beginDescription, endDescription,
        beginPartName, endPartName,
        submitData,
    });
    useEffect(() => {
        actionsRef.current = { beginMachineTest, endMachineTest, beginDescription, endDescription, beginPartName, endPartName, submitData };
    });

    // Transcript polling — scans the live transcript for command keywords mid-stream.
    // Works even when commands are embedded in continuous speech.
    useEffect(() => {
        if (!transcript) return;
        const t = transcript.toLowerCase();

        if (t.includes('start machine test')) {
            actionsRef.current.beginMachineTest();
        } else if (t.includes('stop machine test')) {
            actionsRef.current.endMachineTest();
        } else if (t.includes('start description')) {
            actionsRef.current.beginDescription();
        } else if (t.includes('stop description')) {
            actionsRef.current.endDescription();
        } else if (t.includes('start part name')) {
            actionsRef.current.beginPartName();
        } else if (t.includes('stop part name')) {
            // Extract only the text before the stop keyword as the part name
            const captured = transcript.substring(0, t.indexOf('stop part name')).trim();
            setPartName(captured || null);
            setActive(null);
            resetTranscript();
        } else if (t.includes('confirm') || t.includes('done') || t.includes('submit')) {
            actionsRef.current.submitData();
        }
    }, [transcript]);

    const [recordingStarted, setRecordingStarted] = useState(false);

    useEffect(() => {
        if (!listening && recordingStarted) {
            SpeechRecognition.startListening({ continuous: true });
        }
    }, [listening, recordingStarted]);

    const micPressed = async () => {
        setMicError(null);
        if (!recordingStarted) {
            // Explicitly request mic permission first so we get a clear error if denied
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                stream.getTracks().forEach(t => t.stop()); // release immediately, just checking perms
            } catch (err: any) {
                const msg = err.name === 'NotAllowedError'
                    ? 'Microphone permission denied — allow mic access in your browser settings and try again.'
                    : `Mic error: ${err.message}`;
                setMicError(msg);
                return;
            }
            try {
                await SpeechRecognition.startListening({ language: 'en-US', continuous: true });
            } catch (err: any) {
                // Speech recognition may fail but media recording can still work
                console.warn('SpeechRecognition error:', err);
            }
            setRecordingStarted(true);
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

                {browserSupportsSpeechRecognition ? (
                    <>
                        <div id='mic-row'>
                            <Microphone micPressed={micPressed} />
                            <span className={`listen-label ${recordingStarted ? 'active' : ''}`}>
                                {recordingStarted ? '🎙️ Listening for commands...' : 'Tap mic to start'}
                            </span>
                        </div>
                        {recordingStarted && (
                            <div id='transcript-box'>
                                <span className='label'>Heard:</span>
                                {transcript || <em>say a command...</em>}
                            </div>
                        )}
                    </>
                ) : (
                    <div className='badge badge-warn'>⚠️ Voice commands not supported on this browser — use buttons below</div>
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
                    <div className='recorder-status'>
                        Machine: <code>{machineTestStatus}</code> · Description: <code>{descriptionStatus}</code>
                    </div>
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
                        <li>\"start machine test\" → \"stop machine test\"</li>
                        <li>\"start description\" → \"stop description\"</li>
                        <li>\"start part name\" → \"stop part name\"</li>
                        <li>\"confirm\" or \"done\"</li>
                    </ul>
                </div>

                {submitError && <div className='badge badge-error'>{submitError}</div>}
                {isSubmitting && <div className='badge badge-submitting'>⏳ Submitting to AI...</div>}
            </div>
        </div>
    );
}
