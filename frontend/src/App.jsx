import { useState, useEffect, useRef, useCallback } from "react";
import { api } from "./api";
import "./index.css";

const EMPTY_STATE = {
  mode: "shapes", 
  current_url: null,
  original_url: null,
  can_undo: false,
  has_image: false,
  last_action: "",
  canny_low: 50,
  canny_high: 150,
  chain_code: null,
  perimeter: null,
  area: null,
};

export default function App() {
  const [state, setState] = useState(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");

  const [shapeType, setShapeType] = useState("canny");
  const [cannyLow, setCannyLow] = useState(50);
  const [cannyHigh, setCannyHigh] = useState(150);
  const [houghThresh, setHoughThresh] = useState(100);

  const fetchState = useCallback(async () => {
    try {
      const data = await api.getState();
      setState((prev) => ({ ...prev, ...data }));
      if (data.canny_low) setCannyLow(data.canny_low);
      if (data.canny_high) setCannyHigh(data.canny_high);
      if (data.hough_thresh !== undefined) setHoughThresh(data.hough_thresh);
    } catch { }
  }, []);

  useEffect(() => { fetchState(); }, [fetchState]);

  const withLoading = async (fn) => {
    setLoading(true);
    setError(null);
    try {
      const data = await fn();
      setState(prev => ({ ...prev, ...data }));
      
      // Force the local UI sliders to snap to the backend's historical values
      if (data.canny_low !== undefined) setCannyLow(data.canny_low);
      if (data.canny_high !== undefined) setCannyHigh(data.canny_high);
      if (data.hough_thresh !== undefined) setHoughThresh(data.hough_thresh);
      
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileSelect = (file) => {
    if (!file) return;
    setSelectedFile(file);
    setFileName(file.name);
  };

  const handleUpload = async () => {
    const file = selectedFile || fileInputRef.current?.files?.[0];
    if (!file) {
      setError("Please select an image file first.");
      return;
    }
    await withLoading(() => api.upload(file));
    setSelectedFile(null);
  };

  const handleApplyShapeDetection = () =>
    withLoading(() => api.detectShapes({ 
        shape_type: shapeType,
        canny_low: cannyLow,
        canny_high: cannyHigh,
        hough_thresh: houghThresh
    }));

  const handleRunSnake = () => withLoading(() => api.activeContour({}));
  const handleUndo = () => withLoading(api.undo);
  const handleReset = () => withLoading(api.reset);
  const handleSwitchMode = (mode) =>
    withLoading(() => api.switchMode(mode));

  const isShapesMode = state.mode === "shapes";

  return (
    <div className={`app-root ${!isShapesMode ? "mode-hist" : ""}`}>
      {/* Topbar */}
      <nav className="topbar">
        <div className="topbar-brand">
          <div className="brand-mark" />
          <span className="brand-name">Vision<span>Lab</span></span>
        </div>
        <div className="mode-switch">
          <button className={`mode-btn ${isShapesMode ? "active-filter" : ""}`} onClick={() => handleSwitchMode("shapes")}>
            Shape Detection
          </button>
          <button className={`mode-btn ${!isShapesMode ? "active-hist" : ""}`} onClick={() => handleSwitchMode("snake")}>
            Active Contour
          </button>
        </div>
      </nav>

      {/* Error Banner */}
      {error && (
        <div className="err-wrap">
          <div className="err-box">
            {error}
            <button className="err-close" onClick={() => setError(null)}>×</button>
          </div>
        </div>
      )}

      <div className="page">
        {/* Sidebar */}
        <aside className="sidebar">
          <div className="block">
            <div className="block-header">
              <span className="block-title">Upload Image</span>
            </div>
            <div className="block-body">
              <div className={`drop-zone ${selectedFile ? "has-file" : ""}`} onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => handleFileSelect(e.target.files[0])} />
                {selectedFile ? (
                  <p className="dz-label" style={{ color: "var(--cyan)" }}><strong>{fileName} Ready</strong></p>
                ) : (
                  <p className="dz-label"><strong>Click to browse</strong></p>
                )}
              </div>
              <button className={`btn btn-primary ${loading ? "loading" : ""}`} onClick={handleUpload} disabled={loading}>
                Upload
              </button>
            </div>
          </div>

          {isShapesMode ? (
            <div className="panel-filter block">
              <div className="block-header">
                <span className="block-title">Task 1: Detect Shapes</span>
              </div>
              <div className="block-body">
                <div className="fgroup show">
                  {[
                    { id: "canny", name: "Canny Edge Detector" },
                    { id: "lines", name: "Detect Lines" },
                    { id: "circles", name: "Detect Circles" },
                    { id: "ellipses", name: "Detect Ellipses" },
                  ].map(f => (
                    <div key={f.id} className={`fcard ${shapeType === f.id ? "sel" : ""}`} onClick={() => setShapeType(f.id)}>
                      <div className="fcard-info">
                        <span className="fcard-name">{f.name}</span>
                      </div>
                      <div className="sel-indicator" />
                    </div>
                  ))}
                </div>

                {/* Canny Specific Threshold Controls */}
                {shapeType === "canny" && (
                  <div className="canny-controls show" style={{marginTop: 15, padding: 10, background: "var(--surf2)", borderRadius: 8}}>
                    <div className="canny-title" style={{fontSize: "0.8rem", color: "var(--text3)", marginBottom: 10}}>Hysteresis Thresholds</div>
                    <div className="canny-row" style={{display: "flex", alignItems: "center", gap: 10, marginBottom: 10}}>
                      <label style={{fontSize: "0.8rem", width: 40}}>Low</label>
                      <input 
                        type="range" min="0" max="254" 
                        value={cannyLow} 
                        onChange={e => setCannyLow(Math.min(+e.target.value, cannyHigh - 1))} 
                        style={{flex: 1}}
                      />
                      <span className="canny-val" style={{fontSize: "0.8rem", width: 30}}>{cannyLow}</span>
                    </div>
                    <div className="canny-row" style={{display: "flex", alignItems: "center", gap: 10}}>
                      <label style={{fontSize: "0.8rem", width: 40}}>High</label>
                      <input 
                        type="range" min="1" max="255" 
                        value={cannyHigh} 
                        onChange={e => setCannyHigh(Math.max(+e.target.value, cannyLow + 1))} 
                        style={{flex: 1}}
                      />
                      <span className="canny-val" style={{fontSize: "0.8rem", width: 30}}>{cannyHigh}</span>
                    </div>
                  </div>
                )}

                {/* Hough Lines Threshold Control */}
                {shapeType === "lines" && (
                  <div className="canny-controls show" style={{marginTop: 15, padding: 10, background: "var(--surf2)", borderRadius: 8}}>
                    <div className="canny-title" style={{fontSize: "0.8rem", color: "var(--text3)", marginBottom: 10}}>Line Detection Threshold</div>
                    <div style={{fontSize: "0.7rem", color: "var(--text2)", marginBottom: 10}}>Minimum votes required in accumulator to form a line.</div>
                    <div className="canny-row" style={{display: "flex", alignItems: "center", gap: 10}}>
                      <label style={{fontSize: "0.8rem", width: 40}}>Votes</label>
                      <input 
                        type="range" min="10" max="300" 
                        value={houghThresh} 
                        onChange={e => setHoughThresh(+e.target.value)} 
                        style={{flex: 1}}
                      />
                      <span className="canny-val" style={{fontSize: "0.8rem", width: 30}}>{houghThresh}</span>
                    </div>
                  </div>
                )}

                <button
                  className={`btn btn-primary ${loading ? "loading" : ""}`}
                  disabled={!state.has_image || loading}
                  onClick={handleApplyShapeDetection}
                  style={{ marginTop: 20 }}
                >
                  Apply
                </button>
              </div>
            </div>
          ) : (
             <div className="panel-hist block">
              <div className="block-header">
                <span className="block-title">Task 2: Active Contour</span>
              </div>
              <div className="block-body">
                 <p style={{fontSize: "0.8rem", color: "var(--text3)", marginBottom: "20px"}}>
                   Currently set to initialize from center.
                 </p>
                <button className={`btn btn-green ${loading ? "loading" : ""}`} disabled={!state.has_image || loading} onClick={handleRunSnake}>
                  Initialize & Evolve Snake
                </button>
              </div>
            </div>
          )}
          
          <div className="act-row" style={{padding: "20px"}}>
             <button className="btn btn-secondary" disabled={!state.can_undo || loading} onClick={handleUndo}>Undo</button>
             <button className="btn btn-outline-red" disabled={!state.has_image || loading} onClick={handleReset}>Reset</button>
          </div>
        </aside>

        {/* Main Content */}
        <section className="right">
          <div className="right-filter">
            {state.current_url ? (
              <>
                <div className="img-grid">
                  <ImageBox label="Source" badge="b-src" badgeText="ORIGINAL" url={state.original_url || state.current_url} />
                  <ImageBox
                    label="Output"
                    badge="b-prev"
                    badgeText={state.last_action ? state.last_action.toUpperCase() : "PREVIEW"}
                    url={state.current_url}
                  />
                </div>
              </>
            ) : (
              <div className="empty-state">
                <div className="empty-title">No Image Loaded</div>
                <div className="empty-sub">Upload an image from the sidebar to get started</div>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
}

const MEDIA_BASE = "http://localhost:8000";
function resolveUrl(url) {
  if (!url) return null;
  if (url.startsWith("http") || url.startsWith("data:")) return url;
  return `${MEDIA_BASE}${url}`;
}

function ImageBox({ label, badge, badgeText, url }) {
  return (
    <div className="ibox">
      <div className="ibox-head">
        <span className="ibox-label">{label}</span>
        <span className={`badge ${badge}`}>{badgeText}</span>
      </div>
      <div className="ibox-frame">
        {url && <img src={resolveUrl(url)} alt={label} />}
      </div>
    </div>
  );
}