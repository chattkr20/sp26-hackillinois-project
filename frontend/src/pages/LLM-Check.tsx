import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './LLM-Check.css';

const REPORT_API = 'https://milindkumar1--cat-report-generator-reportgenerator-gener-a5b1ab.modal.run';
const IMAGE_ANOMALY_API = 'https://milindkumar1--cat-image-anomaly-imageanomalydetector-det-8f11cb.modal.run';

export default function LLMCheck() {
    const navigate = useNavigate();
    const location = useLocation();
    const payload = location.state as Record<string, any> | null;

    const [stage, setStage] = useState<'analyzing' | 'imageAnalyzing' | 'generating' | 'error'>('analyzing');
    const [error, setError] = useState<string | null>(null);
    const [dot, setDot] = useState('');

    // Animated dots
    useEffect(() => {
        const t = setInterval(() => setDot(d => d.length >= 3 ? '' : d + '.'), 500);
        return () => clearInterval(t);
    }, []);

    useEffect(() => {
        if (!payload) { navigate('/login'); return; }

        let cancelled = false;

        const run = async () => {
            try {
                setStage('analyzing');
                await new Promise(r => setTimeout(r, 800));

                // ── Optional: visual anomaly from captured image ──────────
                let imageAnomalyResult: Record<string, any> | null = null;
                if (payload.imageDataUrl) {
                    setStage('imageAnalyzing');
                    try {
                        // Convert data URL → blob → raw bytes
                        const res = await fetch(payload.imageDataUrl);
                        const blob = await res.blob();
                        const imgRes = await fetch(IMAGE_ANOMALY_API, {
                            method: 'POST',
                            body: blob,
                            headers: { 'Content-Type': blob.type || 'image/jpeg' },
                        });
                        if (imgRes.ok) {
                            imageAnomalyResult = await imgRes.json();
                        }
                    } catch {
                        // Image analysis is optional — continue without it
                    }
                }

                setStage('generating');
                // Strip full data URL — send only the base64 portion to keep payload small
                // eslint-disable-next-line @typescript-eslint/no-unused-vars
                const { imageDataUrl: _imgDataUrl, ...payloadWithoutImage } = payload as Record<string, any>;
                const image_data_b64 = payload.imageDataUrl
                    ? (payload.imageDataUrl as string).split(',')[1] ?? ''
                    : '';
                const reportPayload = {
                    ...payloadWithoutImage,
                    ...(imageAnomalyResult ? { image_anomaly: imageAnomalyResult } : {}),
                    ...(image_data_b64 ? { image_data_b64 } : {}),
                };

                const res = await fetch(REPORT_API, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(reportPayload),
                });

                if (!res.ok) {
                    const msg = await res.text().catch(() => res.statusText);
                    throw new Error(`Report API ${res.status}: ${msg}`);
                }

                const data = await res.json();
                if (cancelled) return;

                if (!data.pdf_base64) throw new Error('No PDF returned from API');

                navigate('/report', { state: { ...data, ...reportPayload, imageDataUrl: payload.imageDataUrl } });
            } catch (err: any) {
                if (!cancelled) setError(err.message);
                setStage('error');
            }
        };

        run();
        return () => { cancelled = true; };
    }, []);

    const stageMessages: Record<string, string> = {
        analyzing: 'Analyzing anomaly results',
        imageAnalyzing: 'Analyzing inspection photo',
        generating: 'Generating inspection report with AI',
    };

    return (
        <div className='llm-root'>
            <div className='login-bg-grid' />

            <header className='login-header'>
                <div className='login-app-title'>
                    <span className='title-cat'>CAT</span>
                    <span className='title-inspect'>INSPECT</span>
                </div>
                <span className='rec-step-label'>▸ STEP 3 · GENERATING REPORT</span>
            </header>

            <main className='llm-main'>
            <div className='llm-check-card'>

                {stage !== 'error' ? (
                    <>
                        <div className='llm-spinner'>
                            <div className='spinner-ring' />
                        </div>
                        <p className='llm-stage'>{stageMessages[stage]}{dot}</p>
                        <p className='llm-sub'>This may take 15–30 seconds</p>

                        <div className='llm-steps'>
                            <div className={`llm-step ${stage === 'analyzing' ? 'active' : 'done'}`}>
                                <span className='step-icon'>{stage === 'analyzing' ? '⏳' : '✅'}</span>
                                Acoustic anomaly processed
                            </div>
                            {payload?.imageDataUrl && (
                                <div className={`llm-step ${
                                    stage === 'imageAnalyzing' ? 'active' : (stage === 'generating' ? 'done' : '')
                                }`}>
                                    <span className='step-icon'>{stage === 'imageAnalyzing' ? '⏳' : (stage === 'generating' ? '✅' : '⬜')}</span>
                                    Visual anomaly analysis
                                </div>
                            )}
                            <div className={`llm-step ${stage === 'generating' ? 'active' : ''}`}>
                                <span className='step-icon'>{stage === 'generating' ? '⏳' : '⬜'}</span>
                                AI report generation
                            </div>
                        </div>

                        {payload && (
                            <div className='llm-summary-preview'>
                                <div className='summary-row'>
                                    <span>Part</span>
                                    <strong>{payload.part_name || 'Not specified'}</strong>
                                </div>
                                <div className='summary-row'>
                                    <span>Anomaly</span>
                                    <strong className={payload.anomaly?.status === 'anomaly' ? 'red' : 'green'}>
                                        {payload.anomaly?.status === 'anomaly' ? '⚠ DETECTED' : '✓ NORMAL'}
                                    </strong>
                                </div>
                                <div className='summary-row'>
                                    <span>Score</span>
                                    <strong>{((payload.anomaly?.anomaly_score || 0) * 100).toFixed(1)}%</strong>
                                </div>
                            </div>
                        )}
                    </>
                ) : (
                    <div className='llm-error'>
                        <p className='error-title'>⚠ Report generation failed</p>
                        <p className='error-msg'>{error}</p>
                        <div className='error-actions'>
                            <button className='btn-retry' onClick={() => window.location.reload()}>Retry</button>
                            <button className='btn-back' onClick={() => navigate('/recording')}>← Back to Recording</button>
                        </div>
                    </div>
                )}
            </div>
            </main>

            <footer className='login-footer'>
                © {new Date().getFullYear()} Caterpillar Inc. · Internal Use Only
            </footer>
        </div>
    );
}

