
import { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import './ReportDisplay.css';

export default function ReportDisplay() {
    const navigate = useNavigate();
    const location = useLocation();
    const state = location.state as Record<string, any> | null;

    const [blobUrl, setBlobUrl] = useState<string | null>(null);

    const pdfBase64: string | null = state?.pdf_base64 ?? null;

    useEffect(() => {
        if (!pdfBase64) return;
        const bytes = atob(pdfBase64);
        const buf = new Uint8Array(bytes.length);
        for (let i = 0; i < bytes.length; i++) buf[i] = bytes.codePointAt(i) as number;
        const blob = new Blob([buf], { type: 'application/pdf' });
        const url = URL.createObjectURL(blob);
        setBlobUrl(url);
        return () => URL.revokeObjectURL(url);
    }, [pdfBase64]);

    const handleDownload = () => {
        if (!blobUrl) return;
        const a = document.createElement('a');
        a.href = blobUrl;
        a.download = `inspection-report-${Date.now()}.pdf`;
        a.click();
    };

    if (!state?.pdf_base64) {
        return (
            <div className='report-empty'>
                <p>No report data found.</p>
                <button className='report-btn-start-new' onClick={() => navigate('/login')}>
                    Start New Inspection
                </button>
            </div>
        );
    }

    const anomaly = state.anomaly ?? {};
    const parts: any[] = state.parts ?? [];
    const summary: string = state.summary ?? '';
    const partName: string = state.part_name ?? 'Not specified';
    const operatorName: string = state.operator_name ?? '—';
    const anomalyStatus: string = anomaly.status ?? 'unknown';
    const score: number = anomaly.anomaly_score ?? 0;
    const isAnomaly = anomalyStatus === 'anomaly';

    const partStatusColor: Record<string, string> = {
        FAIL: '#ef5350',
        MONITOR: '#ffb300',
        PASS: '#4caf50',
    };

    return (
        <div className='report-root'>
            {/* ── Sidebar ── */}
            <aside className='report-sidebar'>
                <div className='report-sidebar-header'>
                    <div className='report-sidebar-title'>
                        <span className='title-cat'>CAT</span>
                        <span className='title-inspect'>INSPECT</span>
                    </div>
                    <span className='report-sidebar-step'>▸ REPORT</span>
                </div>

                <div className='report-sidebar-body'>
                    <div className={`report-status-badge ${isAnomaly ? 'anomaly' : 'normal'}`}>
                        <span className={`report-status-label ${isAnomaly ? 'anomaly' : 'normal'}`}>
                            {isAnomaly ? '⚠ ANOMALY DETECTED' : '✓ NORMAL'}
                        </span>
                        <span className='report-score'>{(score * 100).toFixed(1)}% anomaly score</span>
                    </div>

                    <div className='report-info-block'>
                        <div className='report-info-row'>
                            <span className='report-info-label'>Part</span>
                            <span className='report-info-val'>{partName}</span>
                        </div>
                        <div className='report-info-row'>
                            <span className='report-info-label'>Operator</span>
                            <span className='report-info-val'>{operatorName}</span>
                        </div>
                        {anomaly.machine_type && (
                            <div className='report-info-row'>
                                <span className='report-info-label'>Machine</span>
                                <span className='report-info-val'>{anomaly.machine_type}</span>
                            </div>
                        )}
                        {anomaly.anomaly_subtype && (
                            <div className='report-info-row'>
                                <span className='report-info-label'>Fault Type</span>
                                <span className='report-info-val'>{anomaly.anomaly_subtype}</span>
                            </div>
                        )}
                    </div>

                    {summary && (
                        <>
                            <p className='report-section-label'>AI Summary</p>
                            <div className='report-summary-block'>
                                <p className='report-summary-text'>{summary}</p>
                            </div>
                        </>
                    )}

                    {parts.length > 0 && (
                        <>
                            <p className='report-section-label'>Parts Analyzed</p>
                            <div className='report-parts-block'>
                                {parts.map((p, i) => (
                                    <div key={i} className='report-part-row'>
                                        <span className='report-part-name'>{p.part_name}</span>
                                        <span
                                            className='report-part-status'
                                            style={{ color: partStatusColor[p.status?.toUpperCase()] ?? '#888' }}
                                        >
                                            {p.status}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </>
                    )}

                    <div className='report-actions'>
                        <button className='report-btn-download' onClick={handleDownload}>
                            ⬇ Download PDF
                        </button>
                        <button className='report-btn-new' onClick={() => navigate('/recording')}>
                            + New Recording
                        </button>
                        <button className='report-btn-logout' onClick={() => navigate('/login')}>
                            Log Out
                        </button>
                    </div>
                </div>
            </aside>

            {/* ── PDF viewer ── */}
            <div className='report-pdf-pane'>
                {blobUrl ? (
                    <iframe src={blobUrl} className='report-iframe' title='Inspection Report' />
                ) : (
                    <div className='report-pdf-loading'>Loading PDF…</div>
                )}
            </div>
        </div>
    );
}
