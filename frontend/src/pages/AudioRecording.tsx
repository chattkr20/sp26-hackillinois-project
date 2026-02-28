import Microphone from '../components/Mic';
import './AudioRecording.css'
import { useState, useEffect } from 'react';
import SpeechRecognition, { useSpeechRecognition } from 'react-speech-recognition';

export default function AudioRecording () {

    const commands = [
        {
            command: 'testing commands',
            callback: () => {
                console.log("TEST SUCCESSFUL");
            }
        },
    ];

    const {
        transcript,
        listening,
        resetTranscript,
        browserSupportsSpeechRecognition
    } = useSpeechRecognition({ commands });
    
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
        if ( !recordingStarted ) {
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