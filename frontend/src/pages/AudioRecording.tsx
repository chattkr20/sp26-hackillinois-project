import Microphone from '../components/Mic';
import './AudioRecording.css'
import { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useReactMediaRecorder } from 'react-media-recorder';

const ANOMALY_API = 'https://milindkumar1--cat-audio-anomaly-detect-anomaly.modal.run';
const STT_API = 'https://milindkumar1--cat-speech-to-text-transcribe.modal.run';

type ActiveRecording = 'machineTest' | 'description' | 'partName' | null;

export default function AudioRecording() {

    const activeRecordingRef = useRef<ActiveRecording>(null);
    const [activeRecordingDisplay, setActiveRecordingDisplay] = useState<ActiveRecording>(null);
    const [partName, setPartName] = useState<string | null>(null);
    const [machineTestDone, setMachineTestDone] = useState(false);
    const [descriptionDone, setDescriptionDone] = useState(false);
    const [isSubmitting, setIsSubmitting] = useState(false);
    const [results, setResults] = useState<{ anomaly: any; stt: any } | null>(null);
    const [submitError, setSubmitError] = useState<string | null>(null);

    const machineTestFileRef = useRef<string | null>(null);
    const descriptionFileRef = useRef<string | null>(null);

    const { startRecording: startMachineTest, stopRecording: stopMachineTest, mediaBlobUrl: machineTestBlob } = useReactMediaRecorder({ audio: true });
    const { startRecording: startDescription, stopRecording: stopDescription, mediaBlobUrl: descriptionBlob } = useReactMediaRecorder({ audio: true });

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
        setResults(null);

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
                setResults({ anomaly: anomalyResult, stt: sttResult });
            }
        } catch (err: any) {
            setSubmitError(`Submit failed: ${err.message}`);
        } finally {
            setIsSubmitting(false);
        }

        micPressed();
    }

    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition
    } = useSpeechRecognition({
        commands: [
            { command: ['start part name', 'start part names'], callback: beginPartName },
            { command: ['stop part name', 'stop part names'], callback: endPartName },
            { command: ['start machine test', 'start machine tests'], callback: beginMachineTest },
            { command: ['stop machine test', 'stop machine tests'], callback: endMachineTest },
            { command: ['start description', 'start descriptions'], callback: beginDescription },
            { command: ['stop description', 'stop descriptions'], callback: endDescription },
            { command: ['confirm', 'submit', 'done'], callback: submitData },
        ]
    });

    const [recordingStarted, setRecordingStarted] = useState(false);

    useEffect(() => {
        if (!listening && recordingStarted) {
            SpeechRecognition.startListening({ continuous: true });
        }
    }, [listening, recordingStarted]);

    if (!browserSupportsSpeechRecognition) {
        return <span>Browser doesn't support speech recognition.</span>;
    }

    const micPressed = () => {
        if (!recordingStarted) {
            SpeechRecognition.startListening({ language: 'en-US', continuous: true });
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

    const anomalyStatus = results?.anomaly?.status;

    return (
        <div id='recording-screen'>
            <div id='status-panel'>
                <h2>CAT Inspection Tool</h2>

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

                {activeRecordingDisplay && (
                    <div className='badge badge-recording'>
                        {activeLabels[activeRecordingDisplay]}
                    </div>
                )}

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

                {results && (
                    <div id='results-panel'>
                        <div className={`result-status ${anomalyStatus}`}>
                            {anomalyStatus === 'anomaly' ? '⚠️ ANOMALY DETECTED' : '✅ NORMAL'}
                        </div>
                        <div className='result-row'>
                            <span>Anomaly Score</span>
                            <strong>{(results.anomaly.anomaly_score * 100).toFixed(1)}%</strong>
                        </div>
                        {results.anomaly.machine_type && (
                            <div className='result-row'>
                                <span>Machine Type</span>
                                <strong>{results.anomaly.machine_type}</strong>
                            </div>
                        )}
                        {results.anomaly.anomaly_subtype && (
                            <div className='result-row'>
                                <span>Fault Type</span>
                                <strong>{results.anomaly.anomaly_subtype}</strong>
                            </div>
                        )}
                        {results.stt?.transcript && (
                            <div className='result-row'>
                                <span>Transcript</span>
                                <em>{results.stt.transcript}</em>
                            </div>
                        )}
                    </div>
                )}
            </div>
        </div>
    );
}
