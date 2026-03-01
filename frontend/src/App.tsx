import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import LoginPage from './pages/LoginPage';
import AudioRecording from './pages/AudioRecording';
import LLMCheck from './pages/LLM-Check';
import ReportDisplay from './pages/ReportDisplay';

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Navigate to="/login" replace />} />
        <Route path="/login" element={<LoginPage />} />
        <Route path="/recording" element={<AudioRecording />} />
        <Route path="/llm-check" element={<LLMCheck />} />
        <Route path="/report" element={<ReportDisplay />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;