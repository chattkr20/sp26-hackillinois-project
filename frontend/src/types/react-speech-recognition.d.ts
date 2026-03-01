declare module 'react-speech-recognition' {
  interface SpeechRecognitionHook {
    transcript: string;
    interimTranscript: string;
    finalTranscript: string;
    listening: boolean;
    resetTranscript: () => void;
    browserSupportsSpeechRecognition: boolean;
    isMicrophoneAvailable: boolean;
  }

  interface ListenOptions {
    continuous?: boolean;
    language?: string;
    interimResults?: boolean;
  }

  const SpeechRecognition: {
    startListening: (options?: ListenOptions) => Promise<void>;
    stopListening: () => void;
    abortListening: () => void;
    getRecognition: () => any;
  };

  export function useSpeechRecognition(): SpeechRecognitionHook;
  export default SpeechRecognition;
}
