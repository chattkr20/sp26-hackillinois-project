
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
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: 16, fontFamily: 'sans-serif' }}>
                <p style={{ color: '#888' }}>No report data found.</p>
                <button onClick={() => navigate('/login')} style={{ padding: '10px 24px', background: '#FFCD11', border: 'none', borderRadius: 8, fontWeight: 700, cursor: 'pointer' }}>
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

    return (
        <div style={styles.root}>
            {/* Left sidebar */}
            <div style={styles.sidebar}>
                <div style={styles.sidebarHeader}>
                    <span style={styles.logoCat}>CAT</span>
                    <span style={styles.logoInspect}> INSPECT</span>
                </div>

                <div style={{ ...styles.statusBadge, background: anomalyStatus === 'anomaly' ? '#3d0a0a' : '#0a2e0a', borderColor: anomalyStatus === 'anomaly' ? '#ef5350' : '#4caf50' }}>
                    <span style={{ color: anomalyStatus === 'anomaly' ? '#ef5350' : '#4caf50', fontWeight: 700, fontSize: '1rem' }}>
                        {anomalyStatus === 'anomaly' ? '⚠ ANOMALY DETECTED' : '✓ NORMAL'}
                    </span>
                    <span style={styles.scoreText}>{(score * 100).toFixed(1)}% anomaly score</span>
                </div>

                <div style={styles.infoBlock}>
                    <div style={styles.infoRow}><span style={styles.infoLabel}>Part</span><span style={styles.infoVal}>{partName}</span></div>
                    <div style={styles.infoRow}><span style={styles.infoLabel}>Operator</span><span style={styles.infoVal}>{operatorName}</span></div>
                    {anomaly.machine_type && <div style={styles.infoRow}><span style={styles.infoLabel}>Machine</span><span style={styles.infoVal}>{anomaly.machine_type}</span></div>}
                    {anomaly.anomaly_subtype && <div style={styles.infoRow}><span style={styles.infoLabel}>Fault Type</span><span style={styles.infoVal}>{anomaly.anomaly_subtype}</span></div>}
                </div>

                {summary && (
                    <div style={styles.summaryBlock}>
                        <p style={styles.summaryLabel}>AI SUMMARY</p>
                        <p style={styles.summaryText}>{summary}</p>
                    </div>
                )}

                {parts.length > 0 && (
                    <div style={styles.partsBlock}>
                        <p style={styles.summaryLabel}>PARTS ANALYZED</p>
                        {parts.map((p, i) => {
                            const sc: Record<string, string> = { FAIL: '#ef5350', MONITOR: '#ffb300', PASS: '#4caf50' };
                            return (
                                <div key={i} style={styles.partRow}>
                                    <span style={styles.partName}>{p.part_name}</span>
                                    <span style={{ color: sc[p.status?.toUpperCase()] ?? '#aaa', fontWeight: 700, fontSize: '0.75rem' }}>{p.status}</span>
                                </div>
                            );
                        })}
                    </div>
                )}

                <div style={styles.actions}>
                    <button onClick={handleDownload} style={styles.btnDownload}>⬇ Download PDF</button>
                    <button onClick={() => navigate('/recording')} style={styles.btnNew}>+ New Recording</button>
                    <button onClick={() => navigate('/login')} style={styles.btnLogout}>Log Out</button>
                </div>
            </div>

            {/* PDF viewer */}
            <div style={styles.pdfPane}>
                {blobUrl ? (
                    <iframe
                        src={blobUrl}
                        style={styles.iframe}
                        title="Inspection Report"
                    />
                ) : (
                    <div style={styles.pdfLoading}>Loading PDF…</div>
                )}
            </div>
        </div>
    );
}

const styles: Record<string, React.CSSProperties> = {
    root: { display: 'flex', height: '100vh', width: '100vw', fontFamily: 'system-ui, sans-serif', background: '#0f1117', overflow: 'hidden' },
    sidebar: { width: 280, minWidth: 240, background: '#1a1d27', borderRight: '1px solid #2a2d3a', display: 'flex', flexDirection: 'column', gap: 16, padding: '24px 20px', overflowY: 'auto' },
    sidebarHeader: { fontWeight: 900, fontSize: '1.1rem', letterSpacing: '0.08em', marginBottom: 4 },
    logoCat: { color: '#FFCD11' },
    logoInspect: { color: '#fff' },
    statusBadge: { border: '1px solid', borderRadius: 10, padding: '10px 14px', display: 'flex', flexDirection: 'column', gap: 4 },
    scoreText: { color: '#888', fontSize: '0.78rem' },
    infoBlock: { display: 'flex', flexDirection: 'column', gap: 6, background: '#12141c', borderRadius: 10, padding: '10px 12px' },
    infoRow: { display: 'flex', justifyContent: 'space-between', fontSize: '0.82rem', gap: 8 },
    infoLabel: { color: '#666' },
    infoVal: { color: '#ddd', fontWeight: 600, textAlign: 'right', maxWidth: 140, wordBreak: 'break-word' },
    summaryBlock: { background: '#12141c', borderRadius: 10, padding: '10px 12px' },
    summaryLabel: { color: '#555', fontSize: '0.7rem', fontWeight: 700, letterSpacing: '0.1em', marginBottom: 6, margin: 0 },
    summaryText: { color: '#aaa', fontSize: '0.82rem', lineHeight: 1.5, margin: '6px 0 0' },
    partsBlock: { background: '#12141c', borderRadius: 10, padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: 6 },
    partRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.82rem' },
    partName: { color: '#ccc', flex: 1, marginRight: 8, wordBreak: 'break-word' },
    actions: { display: 'flex', flexDirection: 'column', gap: 8, marginTop: 'auto', paddingTop: 8 },
    btnDownload: { background: '#FFCD11', color: '#000', border: 'none', borderRadius: 8, padding: '10px 0', fontWeight: 700, cursor: 'pointer', fontSize: '0.9rem' },
    btnNew: { background: 'transparent', color: '#ddd', border: '1px solid #333', borderRadius: 8, padding: '9px 0', cursor: 'pointer', fontSize: '0.9rem' },
    btnLogout: { background: 'transparent', color: '#666', border: 'none', borderRadius: 8, padding: '8px 0', cursor: 'pointer', fontSize: '0.82rem' },
    pdfPane: { flex: 1, display: 'flex', alignItems: 'stretch', background: '#1a1a1a' },
    iframe: { width: '100%', height: '100%', border: 'none' },
    pdfLoading: { flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#555', fontSize: '0.9rem' },
};