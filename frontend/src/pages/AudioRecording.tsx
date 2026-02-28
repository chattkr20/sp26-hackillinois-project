import Microphone from '../components/Mic';
import './AudioRecording.css'
import { useState, useEffect, useRef } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';
import { useReactMediaRecorder } from 'react-media-recorder';

const ANOMALY_API = 'https://milindkumar1--cat-audio-anomaly-detect-anomaly.modal.run';
const STT_API = 'https://milindkumar1--cat-speech-to-text-transcribe.modal.run';

export default function AudioRecording() {

    const activeRecordingRef = useRef<'machineTest' | 'description' | 'partName' | null>(null);
    const [partName, setPartName] = useState<string | null>(null);
    const machineTestFileRef = useRef<string | null>(null);
    const descriptionFileRef = useRef<string | null>(null);

    const { startRecording: startMachineTest, stopRecording: stopMachineTest, mediaBlobUrl: machineTestBlob } = useReactMediaRecorder({ audio: true, mimeType: 'audio/wav' });
    const { startRecording: startDescription, stopRecording: stopDescription, mediaBlobUrl: descriptionBlob } = useReactMediaRecorder({ audio: true, mimeType: 'audio/wav' });

    useEffect(() => {
        if (machineTestBlob) {
            machineTestFileRef.current = machineTestBlob;
            console.log("Machine test saved:", machineTestBlob);
        }
    }, [machineTestBlob]);

    useEffect(() => {
        if (descriptionBlob) {
            descriptionFileRef.current = descriptionBlob;
            console.log("Description saved:", descriptionBlob);
        }
    }, [descriptionBlob]);

    const beginPartName = () => {
        if (activeRecordingRef.current !== null) {
            console.log("ERROR: Cannot start part name while another recording is active:", activeRecordingRef.current);
            resetTranscript();
            return;
        }
        activeRecordingRef.current = 'partName';
        console.log("BEGIN PART NAME");
        resetTranscript();
    }

    const endPartName = () => {
        setPartName(transcript);
        activeRecordingRef.current = null;
        console.log("END PART NAME, saved:", transcript);
        resetTranscript();
    }

    const beginMachineTest = () => {
        if (activeRecordingRef.current !== null) {
            console.log("ERROR: Cannot start machine test while another recording is active:", activeRecordingRef.current);
            resetTranscript();
            return;
        }
        startMachineTest();
        activeRecordingRef.current = 'machineTest';
        console.log("BEGIN MACHINE TEST");
        resetTranscript();
    }

    const endMachineTest = () => {
        stopMachineTest();
        activeRecordingRef.current = null;
        console.log("END MACHINE TEST");
        resetTranscript();
    }

    const beginDescription = () => {
        if (activeRecordingRef.current !== null) {
            console.log("ERROR: Cannot start description while another recording is active:", activeRecordingRef.current);
            resetTranscript();
            return;
        }
        startDescription();
        activeRecordingRef.current = 'description';
        console.log("BEGIN DESCRIPTION");
        resetTranscript();
    }

    const endDescription = () => {
        stopDescription();
        activeRecordingRef.current = null;
        console.log("END DESCRIPTION");
        resetTranscript();
    }

    const submitData = async () => {
        console.log("SUBMIT");

        if (!machineTestFileRef.current || !descriptionFileRef.current) {
            console.log("ERROR: Missing recordings — machineTest:", machineTestFileRef.current, "description:", descriptionFileRef.current);
            return;
        }

        try {
            const [machineTestAudio, descriptionAudio] = await Promise.all([
                fetch(machineTestFileRef.current).then(r => r.blob()),
                fetch(descriptionFileRef.current).then(r => r.blob()),
            ]);

            console.log("Submitting machine test audio, size:", machineTestAudio.size, "type:", machineTestAudio.type);
            console.log("Submitting description audio, size:", descriptionAudio.size, "type:", descriptionAudio.type);

            const [anomalyRes, sttRes] = await Promise.all([
                fetch(ANOMALY_API, { method: 'POST', body: machineTestAudio, headers: { 'Content-Type': 'application/octet-stream' } }),
                fetch(STT_API, { method: 'POST', body: descriptionAudio, headers: { 'Content-Type': 'application/octet-stream' } }),
            ]);

            const [anomalyResult, sttResult] = await Promise.all([anomalyRes.json(), sttRes.json()]);

            console.log("Anomaly result:", anomalyResult);
            console.log("STT result:", sttResult);
        } catch (err) {
            console.log("ERROR during submit:", err);
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
            { command: 'begin part name', callback: beginPartName },
            { command: 'end part name', callback: endPartName },
            { command: 'begin machine test', callback: beginMachineTest },
            { command: 'end machine test', callback: endMachineTest },
            { command: 'begin description', callback: beginDescription },
            { command: 'end description', callback: endDescription },
            { command: 'submit', callback: submitData },
        ]
    });

    const [recordingStarted, setRecordingStarted] = useState(false);

    useEffect(() => {
        if (!listening && recordingStarted) {
            SpeechRecognition.startListening({ continuous: true });
        }
    }, [listening, recordingStarted]);

    if (!browserSupportsSpeechRecognition) {
        console.log("BROWSER UNSUPPORTED");
        return <span>Browser doesn't support speech recognition.</span>;
    }

    const micPressed = () => {
        console.log("MIC PRESSED");
        if (!recordingStarted) {
            SpeechRecognition.startListening({ language: "en-US", continuous: true });
            setRecordingStarted(!recordingStarted);
            console.log("RECORDING STARTED");
        } else {
            SpeechRecognition.stopListening({ language: "en-US" });
            setRecordingStarted(!recordingStarted);
            console.log(transcript);
            resetTranscript();
            console.log("RECORDING ENDED");
        }
    };

    return (
        <div id='recording-screen'>
            <Microphone micPressed={micPressed} />
        </div>
    );
}
