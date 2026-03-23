import { useState, useEffect, useRef, useCallback, forwardRef, useImperativeHandle } from "react";
import { api } from "./api";
import "./index.css";
import "./App.css";

const EMPTY_STATE = {
  mode: "shapes", current_url: null, original_url: null,
  can_undo: false, has_image: false, last_action: "",
  canny_low: 50, canny_high: 150, hough_thresh: 100,
  circle_min_r: 20, circle_max_r: 100, circle_thresh: 45,
  ellipse_min_area: 200, chain_code: null, perimeter: null, area: null,
};

export default function App() {
  const [state, setState] = useState(EMPTY_STATE);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [fileName, setFileName] = useState("");

  // Shape tab params
  const [shapeType, setShapeType] = useState("canny");
  const [cannyLow, setCannyLow] = useState(50);
  const [cannyHigh, setCannyHigh] = useState(150);
  const [houghThresh, setHoughThresh] = useState(100);
  const [circleMinR, setCircleMinR] = useState(20);
  const [circleMaxR, setCircleMaxR] = useState(100);
  const [circleThresh, setCircleThresh] = useState(45);
  const [ellipseMinArea, setEllipseMinArea] = useState(200);

  // Snake tab params
  const [snakeAlpha, setSnakeAlpha] = useState(0.1);
  const [snakeBeta, setSnakeBeta] = useState(0.1);
  const [snakeGamma, setSnakeGamma] = useState(0.5);
  const [snakeIter, setSnakeIter] = useState(200);

  // User-drawn contour (normalized [0,1] coords)
  const [drawnPoints, setDrawnPoints] = useState([]);
  const [hasDrawing, setHasDrawing] = useState(false);

  const fetchState = useCallback(async () => {
    try {
      const data = await api.getState();
      setState(prev => ({ ...prev, ...data }));
      if (data.canny_low != null) setCannyLow(data.canny_low);
      if (data.canny_high != null) setCannyHigh(data.canny_high);
      if (data.hough_thresh != null) setHoughThresh(data.hough_thresh);
      if (data.circle_min_r != null) setCircleMinR(data.circle_min_r);
      if (data.circle_max_r != null) setCircleMaxR(data.circle_max_r);
      if (data.circle_thresh != null) setCircleThresh(data.circle_thresh);
      if (data.ellipse_min_area != null) setEllipseMinArea(data.ellipse_min_area);
    } catch { }
  }, []);

  useEffect(() => { fetchState(); }, [fetchState]);

  const withLoading = async (fn) => {
    setLoading(true); setError(null);
    try {
      const data = await fn();
      setState(prev => ({ ...prev, ...data }));
      if (data.canny_low != null) setCannyLow(data.canny_low);
      if (data.canny_high != null) setCannyHigh(data.canny_high);
      if (data.hough_thresh != null) setHoughThresh(data.hough_thresh);
      if (data.circle_min_r != null) setCircleMinR(data.circle_min_r);
      if (data.circle_max_r != null) setCircleMaxR(data.circle_max_r);
      if (data.circle_thresh != null) setCircleThresh(data.circle_thresh);
      if (data.ellipse_min_area != null) setEllipseMinArea(data.ellipse_min_area);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const handleUpload = async () => {
    const file = selectedFile || fileInputRef.current?.files?.[0];
    if (!file) { setError("Please select an image file first."); return; }
    await withLoading(() => api.upload(file));
    setSelectedFile(null); setDrawnPoints([]); setHasDrawing(false);
  };

  const handleApplyShapeDetection = () => withLoading(() => api.detectShapes({
    shape_type: shapeType, canny_low: cannyLow, canny_high: cannyHigh,
    hough_thresh: houghThresh, circle_min_r: circleMinR, circle_max_r: circleMaxR,
    circle_thresh: circleThresh, ellipse_min_area: ellipseMinArea,
  }));

  const handleRunSnake = () => withLoading(() => api.activeContour({
    alpha: snakeAlpha, beta: snakeBeta, gamma: snakeGamma, iterations: snakeIter,
    init_points: hasDrawing && drawnPoints.length >= 3 ? drawnPoints : null,
  }));

  // Ref to the DrawableImageBox so we can clear its canvas on undo/reset
  const drawableBoxRef = useRef(null);

  const clearDrawing = useCallback(() => {
    setDrawnPoints([]);
    setHasDrawing(false);
    drawableBoxRef.current?.clearCanvas();
  }, []);

  const handleUndo = () => withLoading(async () => {
    const data = await api.undo();
    clearDrawing();
    return data;
  });

  const handleReset = () => withLoading(async () => {
    const data = await api.reset();
    clearDrawing();
    return data;
  });

  const handleSwitchMode = (mode) => {
    clearDrawing();
    withLoading(() => api.switchMode(mode));
  };

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
          <button className={`mode-btn ${isShapesMode ? "active-filter" : ""}`} onClick={() => handleSwitchMode("shapes")}>Shape Detection</button>
          <button className={`mode-btn ${!isShapesMode ? "active-hist" : ""}`} onClick={() => handleSwitchMode("snake")}>Active Contour</button>
        </div>
      </nav>

      {error && (
        <div className="err-wrap">
          <div className="err-box">{error}<button className="err-close" onClick={() => setError(null)}>×</button></div>
        </div>
      )}

      <div className="page">
        {/* Sidebar */}
        <aside className="sidebar">
          {/* Upload */}
          <div className="block">
            <div className="block-header"><span className="block-title">Upload Image</span></div>
            <div className="block-body">
              <div className={`drop-zone ${selectedFile ? "has-file" : ""}`} onClick={() => fileInputRef.current?.click()}>
                <input ref={fileInputRef} type="file" accept="image/*" style={{ display: "none" }} onChange={e => { setSelectedFile(e.target.files[0]); setFileName(e.target.files[0]?.name || ""); }} />
                {selectedFile
                  ? <p className="dz-label" style={{ color: "var(--cyan)" }}><strong>{fileName} Ready</strong></p>
                  : <p className="dz-label"><strong>Click to browse</strong></p>}
              </div>
              <button className={`btn btn-primary ${loading ? "loading" : ""}`} onClick={handleUpload} disabled={loading}>Upload</button>
            </div>
          </div>

          {isShapesMode ? (
            <div className="panel-filter block">
              <div className="block-header"><span className="block-title">Task 1: Detect Shapes</span></div>
              <div className="block-body">
                <div className="fgroup show">
                  {[
                    { id: "canny", name: "Canny Edge Detector" },
                    { id: "lines", name: "Detect Lines (Hough)" },
                    { id: "circles", name: "Detect Circles (Hough)" },
                    { id: "ellipses", name: "Detect Ellipses (Algebraic)" },
                  ].map(f => (
                    <div key={f.id} className={`fcard ${shapeType === f.id ? "sel" : ""}`} onClick={() => setShapeType(f.id)}>
                      <span className="fcard-name">{f.name}</span>
                      <div className="sel-indicator" />
                    </div>
                  ))}
                </div>

                {shapeType === "canny" && (
                  <div className="param-box">
                    <div className="param-title">Hysteresis Thresholds</div>
                    <SliderRow label="Low" min={0} max={254} value={cannyLow} onChange={v => setCannyLow(Math.min(v, cannyHigh - 1))} />
                    <SliderRow label="High" min={1} max={255} value={cannyHigh} onChange={v => setCannyHigh(Math.max(v, cannyLow + 1))} />
                  </div>
                )}
                {shapeType === "lines" && (
                  <div className="param-box">
                    <div className="param-title">Line Detection Threshold</div>
                    <div className="param-hint">Minimum accumulator votes to form a line.</div>
                    <SliderRow label="Votes" min={10} max={300} value={houghThresh} onChange={v => setHoughThresh(v)} />
                  </div>
                )}
                {shapeType === "circles" && (
                  <div className="param-box">
                    <div className="param-title">Circle Detection (Gradient-Constrained)</div>
                    <div className="param-hint">Uses CannyScratch edges + gradient voting direction.</div>
                    <SliderRow label="Min R" min={5} max={200} value={circleMinR} onChange={v => setCircleMinR(Math.min(v, circleMaxR - 1))} />
                    <SliderRow label="Max R" min={10} max={400} value={circleMaxR} onChange={v => setCircleMaxR(Math.max(v, circleMinR + 1))} />
                    <SliderRow label="Thresh %" min={10} max={90} value={circleThresh} onChange={v => setCircleThresh(v)} />
                  </div>
                )}
                {shapeType === "ellipses" && (
                  <div className="param-box">
                    <div className="param-title">Ellipse Detection (Direct Algebraic Fit)</div>
                    <div className="param-hint">Fitzgibbon 1996 — conic fitting without cv2.fitEllipse.</div>
                    <SliderRow label="Min Area" min={50} max={5000} value={ellipseMinArea} onChange={v => setEllipseMinArea(v)} />
                  </div>
                )}

                <button className={`btn btn-primary ${loading ? "loading" : ""}`}
                  disabled={!state.has_image || loading} onClick={handleApplyShapeDetection}
                  style={{ marginTop: 20 }}>Apply
                </button>
              </div>
            </div>
          ) : (
            <div className="panel-hist block">
              <div className="block-header"><span className="block-title">Task 2: Active Contour</span></div>
              <div className="block-body">
                <div className="param-box">
                  <div className="param-title">Snake Parameters</div>
                  <div className="param-hint">Draw a region on the image to set initial contour.</div>
                  <SliderRow label="α (elasticity)" min={1} max={100} value={Math.round(snakeAlpha * 100)}
                    onChange={v => setSnakeAlpha(v / 100)} displayVal={snakeAlpha.toFixed(2)} />
                  <SliderRow label="β (stiffness)" min={1} max={100} value={Math.round(snakeBeta * 100)}
                    onChange={v => setSnakeBeta(v / 100)} displayVal={snakeBeta.toFixed(2)} />
                  <SliderRow label="γ (step)" min={10} max={90} value={Math.round(snakeGamma * 100)}
                    onChange={v => setSnakeGamma(v / 100)} displayVal={snakeGamma.toFixed(2)} />
                  <SliderRow label="Iterations" min={50} max={500} value={snakeIter} onChange={v => setSnakeIter(v)} />
                </div>

                {hasDrawing && (
                  <div className="draw-status">
                    ✏️ Custom region drawn — snake will start from your path
                    <button className="clear-draw-btn" onClick={clearDrawing}>Clear</button>
                  </div>
                )}

                <button className={`btn btn-green ${loading ? "loading" : ""}`}
                  disabled={!state.has_image || loading} onClick={handleRunSnake}
                  style={{ marginTop: 16 }}>
                  {loading ? "⏳ Evolving…" : "▶ Initialize & Evolve Snake"}
                </button>

                {(state.chain_code || state.perimeter != null) && (
                  <div className="snake-results">
                    <div className="snake-results-title">📐 Contour Analysis</div>
                    <div className="snake-stat"><span className="snake-stat-label">Perimeter</span><span className="snake-stat-value">{state.perimeter?.toFixed(1) ?? "—"} px</span></div>
                    <div className="snake-stat"><span className="snake-stat-label">Area</span><span className="snake-stat-value">{state.area?.toFixed(1) ?? "—"} px²</span></div>
                    <div className="snake-chain-label">Freeman Chain Code (8-conn.)</div>
                    <div className="snake-chain-code">{state.chain_code ? state.chain_code.slice(0, 120) + (state.chain_code.length > 120 ? "…" : "") : "—"}</div>
                  </div>
                )}
              </div>
            </div>
          )}

          <div className="act-row" style={{ padding: "20px" }}>
            <button className="btn btn-secondary" disabled={!state.can_undo || loading} onClick={handleUndo}>Undo</button>
            <button className="btn btn-outline-red" disabled={!state.has_image || loading} onClick={handleReset}>Reset</button>
          </div>
        </aside>

        {/* Main Content */}
        <section className="right">
          <div className="right-filter">
            {state.current_url ? (
              <div className="img-grid">
                {/* Source image — drawable when in snake mode */}
                {!isShapesMode
                  ? <DrawableImageBox
                    ref={drawableBoxRef}
                    label="Source" badge="b-src" badgeText="DRAW REGION"
                    url={state.original_url || state.current_url}
                    onPointsChange={(pts) => { setDrawnPoints(pts); setHasDrawing(pts.length >= 3); }}
                  />
                  : <ImageBox label="Source" badge="b-src" badgeText="ORIGINAL" url={state.original_url || state.current_url} />
                }
                <ImageBox
                  label="Output" badge="b-prev"
                  badgeText={state.last_action ? state.last_action.toUpperCase() : "PREVIEW"}
                  url={state.current_url}
                />
              </div>
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

// ── Drawable Image Box (snake mode) ───────────────────────────────────────────
const DrawableImageBox = forwardRef(function DrawableImageBox({ label, badge, badgeText, url, onPointsChange }, ref) {
  const canvasRef = useRef(null);
  const imgRef = useRef(null);
  const isDrawingRef = useRef(false);
  const pointsRef = useRef([]);

  // Expose clearCanvas method to parent via ref
  useImperativeHandle(ref, () => ({
    clearCanvas() {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      pointsRef.current = [];
    }
  }));

  // Sync canvas size to match the rendered IMAGE element (not the whole frame)
  const syncCanvasSize = () => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return;
    const rect = img.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    // Position canvas exactly over the image
    canvas.style.left = `${img.offsetLeft}px`;
    canvas.style.top = `${img.offsetTop}px`;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
  };

  const redraw = (pts) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (pts.length < 2) return;
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    if (pts.length > 5) { ctx.closePath(); ctx.fillStyle = "rgba(0,255,128,0.12)"; ctx.fill(); }
    ctx.strokeStyle = "#00ff80";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#00ff80";
    ctx.shadowBlur = 6;
    ctx.stroke();
    ctx.shadowBlur = 0;
    // Start dot
    ctx.beginPath();
    ctx.arc(pts[0][0], pts[0][1], 5, 0, Math.PI * 2);
    ctx.fillStyle = "#00ff80";
    ctx.fill();
  };

  const getPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    // Cursor position relative to the canvas (which is now aligned to the image)
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  const onMouseDown = (e) => {
    e.preventDefault();
    syncCanvasSize();
    isDrawingRef.current = true;
    const p = getPos(e);
    pointsRef.current = [p];
    redraw([p]);
  };

  const onMouseMove = (e) => {
    if (!isDrawingRef.current) return;
    e.preventDefault();
    pointsRef.current.push(getPos(e));
    redraw(pointsRef.current);
  };

  const onMouseUp = (e) => {
    if (!isDrawingRef.current) return;
    isDrawingRef.current = false;
    const pts = pointsRef.current;
    redraw(pts);
    // Normalise to [0,1] range for backend
    const canvas = canvasRef.current;
    const norm = pts.map(([x, y]) => [x / canvas.width, y / canvas.height]);
    onPointsChange(norm);
  };

  return (
    <div className="ibox">
      <div className="ibox-head">
        <span className="ibox-label">{label}</span>
        <span className={`badge ${badge}`}>{badgeText}</span>
      </div>
      <div className="ibox-frame drawable-frame">
        {url && (
          <>
            <img ref={imgRef} src={resolveUrl(url)} alt={label} onLoad={syncCanvasSize} />
            <canvas
              ref={canvasRef}
              className="draw-canvas"
              onMouseDown={onMouseDown}
              onMouseMove={onMouseMove}
              onMouseUp={onMouseUp}
              onMouseLeave={onMouseUp}
              title="Draw a region to initialise the snake"
            />
          </>
        )}
        <div className="draw-hint">✏️ Draw to set initial contour</div>
      </div>
    </div>
  );
});

// ── Static Image Box ───────────────────────────────────────────────────────────
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

// ── Reusable Slider Row ────────────────────────────────────────────────────────
function SliderRow({ label, min, max, value, onChange, displayVal }) {
  return (
    <div className="slider-row">
      <label className="slider-label">{label}</label>
      <input type="range" min={min} max={max} value={value}
        onChange={e => onChange(+e.target.value)} className="slider-input" />
      <span className="slider-val">{displayVal !== undefined ? displayVal : value}</span>
    </div>
  );
}