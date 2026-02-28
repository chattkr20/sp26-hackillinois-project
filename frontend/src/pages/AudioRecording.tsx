import Microphone from '../components/Mic';
import './AudioRecording.css'
import { useState } from 'react';
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

    if (!browserSupportsSpeechRecognition) {
        console.log("BROWSER UNSUPPORTED");
        return <span>Browser doesn't support speech recognition.</span>;
    }

    const micPressed = () => {
        console.log("MIC PRESSED");
        if ( recordingStarted ) {
            SpeechRecognition.startListening({ language: "en-US" });
            setRecordingStarted(!recordingStarted);
            console.log("RECORDING STARTED");
        } else {
            SpeechRecognition.stopListening({ language: "en-US" });
            resetTranscript();
            setRecordingStarted(!recordingStarted);
            console.log("RECORDING ENDED");
        }
    };

    return (
        <div id='recording-screen'>
            <Microphone micPressed={micPressed} />
            <p>{ listening }</p>
            <p>{ transcript }</p>
        </div>
    );
}