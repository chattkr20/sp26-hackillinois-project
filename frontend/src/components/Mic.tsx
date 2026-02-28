import { Mic } from 'lucide-react'
import './Mic.css'
import { useState } from 'react';

interface MicProps {
    micPressed: () => void;
}

export default function Microphone({micPressed}: MicProps) {

    const [micOn, setMicOn] = useState(false);

    return (
        <div id='mic-circle' onClick={() => {console.log("MIC PRESSED"); micPressed(); setMicOn(!micOn)}}>
            <Mic color='green' size={100} />
        </div>
    );
}