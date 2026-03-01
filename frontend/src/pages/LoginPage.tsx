import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './LoginPage.css';

export default function LoginPage() {
  const navigate = useNavigate();

  // Pre-fill from cache if returning user
  const cached = (() => {
    try { return JSON.parse(localStorage.getItem('catInspectOperator') || 'null'); } catch { return null; }
  })();

  const [name, setName] = useState(cached?.name ?? '');
  const [id, setId] = useState(cached?.id ?? '');
  const [errors, setErrors] = useState<{ name?: string; id?: string }>({});

  const validate = () => {
    const e: { name?: string; id?: string } = {};
    if (!name.trim()) e.name = 'Required';
    if (!id.trim()) e.id = 'Required';
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit: React.SubmitEventHandler<HTMLFormElement> = (e) => {
    e.preventDefault();
    if (!validate()) return;

    localStorage.setItem('catInspectOperator', JSON.stringify({
      name: name.trim(),
      id: id.trim().toUpperCase(),
    }));

    navigate('/recording');
  };

  return (
    <div className="login-root">
      <div className="login-bg-grid" />

      <header className="login-header">
        <div className="login-app-title">
          <span className="title-cat">CAT</span>
          <span className="title-inspect">INSPECT</span>
        </div>
        <div className="login-logo">
          <svg viewBox="0 0 200 60" xmlns="http://www.w3.org/2000/svg">
            <polygon points="0,0 165,0 200,60 35,60" fill="#FFCD11" />
            <text x="28" y="48" fontFamily="Arial Black, Arial, sans-serif" fontWeight="900" fontSize="48" fill="#000"></text>
          </svg>
        </div>
      </header>

      <main className="login-main">
        <div className="login-card">
          <div className="login-card-body">
            <p className="login-card-eyebrow">▸ OPERATOR SIGN-IN</p>
            <h1 className="login-card-title">Begin Inspection</h1>
            <p className="login-card-sub">Enter your credentials to start a new session.</p>

            <form onSubmit={handleSubmit} className="login-form" noValidate>
              {cached && (
                <button
                  type="button"
                  className="login-btn login-btn-continue"
                  onClick={() => navigate('/recording')}
                >
                  CONTINUE AS {cached.name.toUpperCase()} <span className="login-btn-arrow">⟶</span>
                </button>
              )}
              <div className={`login-field ${errors.name ? 'has-error' : ''}`}>
                <label htmlFor="name">INSPECTOR NAME</label>
                <input
                  id="name"
                  type="text"
                  placeholder="e.g. John Martinez"
                  value={name}
                  onChange={(e) => { setName(e.target.value); setErrors(p => ({ ...p, name: undefined })); }}
                />
                {errors.name && <span className="login-error">{errors.name}</span>}
              </div>

              <div className={`login-field ${errors.id ? 'has-error' : ''}`}>
                <label htmlFor="id">INSPECTOR ID</label>
                <input
                  id="id"
                  type="text"
                  placeholder="e.g. OP-2841"
                  value={id}
                  onChange={(e) => { setId(e.target.value); setErrors(p => ({ ...p, id: undefined })); }}
                />
                {errors.id && <span className="login-error">{errors.id}</span>}
              </div>

              <button type="submit" className="login-btn">
                BEGIN INSPECTION <span className="login-btn-arrow">⟶</span>
              </button>
            </form>
          </div>
        </div>
      </main>

      <footer className="login-footer">
        © {new Date().getFullYear()} Caterpillar Inc. · Internal Use Only
      </footer>
    </div>
  );
}