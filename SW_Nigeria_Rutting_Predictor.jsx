import { useState, useRef, useEffect } from "react";

// ─── ACCURATE SURROGATE MODEL ────────────────────────────────────────────────
// Gradient Boosting Regressor trained on 500 PLAXIS 3D FEA simulations
// Linear approximation in log-space: R²=0.984 vs GBR outputs
// CV R²=0.9876 ± 0.0025 | RMSE=0.2273mm | MAPE=6.37%
// University of Ibadan · Apara Olalewa · 252070 · TCE 799

const MODEL = {
  intercept: 2.7429,
  coefs: {
    log_e50:    -0.5921,
    cref:       -0.0008,
    phi:        -0.0026,
    thickness:  -2.3419,
    load:        0.0006641,
    isMedium:    0.001783,
    isSoft:      0.001857,
    isStiff:    -0.000401,
  },
  cv_r2: 0.9876,
  cv_std: 0.0025,
  rmse: 0.2273,
  mape: 6.37,
  n_runs: 500,
};

function predictRut(inputs) {
  const { e50, cref, phi, thickness, load, soilGroup } = inputs;
  const log_e50   = Math.log10(Math.max(5000, e50));
  const isMedium  = soilGroup === "Medium_Laterite" ? 1 : 0;
  const isSoft    = soilGroup === "Soft_Laterite"   ? 1 : 0;
  const isStiff   = soilGroup === "Stiff_Laterite"  ? 1 : 0;
  const c = MODEL.coefs;
  const log_rut =
    MODEL.intercept +
    c.log_e50   * log_e50 +
    c.cref      * cref +
    c.phi       * phi +
    c.thickness * thickness +
    c.load      * load +
    c.isMedium  * isMedium +
    c.isSoft    * isSoft +
    c.isStiff   * isStiff;
  return Math.max(0.05, Math.pow(10, log_rut));
}

// CBR → E50Ref  (Powell et al. 1984)
function cbrToE50(cbr, soilGroup) {
  const e50_kPa = 17.6 * Math.pow(cbr, 0.64) * 1000;
  const CLAMP = {
    Lagos_Blue_Clay:  [5000,  15000],
    Soft_Laterite:    [15000, 40000],
    Medium_Laterite:  [40000, 80000],
    Stiff_Laterite:   [80000, 150000],
  };
  const [lo, hi] = CLAMP[soilGroup] || [5000, 150000];
  return Math.round(Math.min(hi, Math.max(lo, e50_kPa)));
}

// Typical cohesion and phi defaults per soil group
const SOIL_DEFAULTS = {
  Lagos_Blue_Clay:  { cref: 28, phi: 22 },
  Soft_Laterite:    { cref: 18, phi: 29 },
  Medium_Laterite:  { cref: 14, phi: 33 },
  Stiff_Laterite:   { cref: 10, phi: 37 },
};

const SOILS = [
  { id: "Lagos_Blue_Clay",  label: "Lagos Blue Clay",   sub: "Coastal zone · CBR 2–8%",   cbr:[2,8],   accent:"#60a5fa" },
  { id: "Soft_Laterite",    label: "Soft Laterite",     sub: "Transitional zone · CBR 8–20%", cbr:[8,20], accent:"#34d399" },
  { id: "Medium_Laterite",  label: "Medium Laterite",   sub: "Inland plateau · CBR 20–40%", cbr:[20,40], accent:"#fbbf24" },
  { id: "Stiff_Laterite",   label: "Stiff Laterite",    sub: "Uplands · CBR 40–80%",      cbr:[40,80], accent:"#f87171" },
];

const AXLE_TYPES = [
  { label: "Light vehicle",       load: 420,  icon: "🚗" },
  { label: "Medium truck",        load: 600,  icon: "🚚" },
  { label: "Standard axle (80kN)",load: 700,  icon: "🚛" },
  { label: "Heavy commercial",    load: 800,  icon: "⛟" },
  { label: "Overloaded HCV",      load: 900,  icon: "⚠️" },
];

const KNOWN_SITES = [
  { label: "Lagos-Ibadan Expressway km 90 (Akintayo & Ibrahim, 2024)",
    soilGroup: "Stiff_Laterite", cbr: 26, thickness: 150, load: 700 },
  { label: "Typical Lagos Island road (soft clay subgrade)",
    soilGroup: "Lagos_Blue_Clay", cbr: 4, thickness: 100, load: 650 },
  { label: "Ibadan bypass — laterite gravel (FHM standard)",
    soilGroup: "Medium_Laterite", cbr: 30, thickness: 150, load: 700 },
  { label: "Ondo upland — stiff laterite highway",
    soilGroup: "Stiff_Laterite", cbr: 55, thickness: 200, load: 700 },
];

// ─── COLOUR SCHEME ───────────────────────────────────────────────────────────
function rutStatus(rut) {
  if (rut < 3)  return { label: "Acceptable",  color: "#22c55e", bg: "#14532d22" };
  if (rut < 6)  return { label: "Moderate",    color: "#f59e0b", bg: "#78350f22" };
  if (rut < 10) return { label: "Severe",      color: "#ef4444", bg: "#7f1d1d22" };
  return           { label: "Critical",     color: "#a855f7", bg: "#3b0764aa" };
}

// ─── COMPONENTS ──────────────────────────────────────────────────────────────
function Label({ children, hint }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", marginBottom:5 }}>
      <span style={{ fontSize:12, color:"#94a3b8", fontFamily:"'DM Sans',sans-serif",
        textTransform:"uppercase", letterSpacing:"0.08em" }}>{children}</span>
      {hint && <span style={{ fontSize:11, color:"#475569", fontFamily:"monospace" }}>{hint}</span>}
    </div>
  );
}

function Slider({ min, max, step, value, onChange, color="#60a5fa" }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ position:"relative", height:6, borderRadius:3,
      background:"#1e293b", marginBottom:2 }}>
      <div style={{ position:"absolute", left:0, top:0, height:"100%",
        width:`${pct}%`, borderRadius:3, background:color,
        transition:"width 0.1s" }} />
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ position:"absolute", top:-7, left:0, width:"100%",
          height:20, opacity:0, cursor:"pointer", margin:0 }} />
    </div>
  );
}

function Card({ children, style={}, accent=false }) {
  return (
    <div style={{
      background: "#0f172a",
      border: `1px solid ${accent ? "#1e40af55" : "#1e293b"}`,
      borderRadius: 12, padding: 20, ...style
    }}>{children}</div>
  );
}

function SectionTitle({ children }) {
  return (
    <div style={{ fontSize:10, letterSpacing:"0.15em", textTransform:"uppercase",
      color:"#334155", marginBottom:14, fontFamily:"'DM Sans',sans-serif",
      paddingBottom:8, borderBottom:"1px solid #1e293b" }}>{children}</div>
  );
}

function ResultGauge({ rut, animating }) {
  const status = rutStatus(rut);
  const max = 15;
  const pct = Math.min(100, (rut / max) * 100);
  const sweep = (pct / 100) * 220;
  const r = 70, cx = 100, cy = 95;
  const toRad = d => (d - 200) * Math.PI / 180;
  const arcPath = (deg) => {
    const rad = toRad(deg);
    return `${cx + r * Math.cos(rad)},${cy + r * Math.sin(rad)}`;
  };
  const trackEnd = toRad(200 + 220);
  const fillEnd  = toRad(200 + sweep);
  return (
    <div style={{ textAlign:"center" }}>
      <svg viewBox="0 0 200 120" width="200" height="120" style={{ overflow:"visible" }}>
        {/* Track */}
        <path d={`M ${arcPath(0)} A ${r} ${r} 0 1 1 ${cx + r * Math.cos(trackEnd)},${cy + r * Math.sin(trackEnd)}`}
          fill="none" stroke="#1e293b" strokeWidth="12" strokeLinecap="round" />
        {/* Fill */}
        {!animating && (
          <path d={`M ${arcPath(0)} A ${r} ${r} 0 ${sweep > 180 ? 1 : 0} 1 ${arcPath(sweep)}`}
            fill="none" stroke={status.color} strokeWidth="12" strokeLinecap="round"
            style={{ filter:`drop-shadow(0 0 6px ${status.color}88)` }} />
        )}
        {/* Value */}
        <text x={cx} y={cy + 8} textAnchor="middle"
          style={{ fontSize:28, fontFamily:"monospace", fontWeight:700,
            fill: animating ? "#334155" : status.color }}>
          {animating ? "···" : rut.toFixed(2)}
        </text>
        <text x={cx} y={cy + 24} textAnchor="middle"
          style={{ fontSize:10, fill:"#475569", fontFamily:"sans-serif" }}>mm rut depth</text>
        {/* Scale labels */}
        <text x={cx - r - 8} y={cy + 8} textAnchor="end"
          style={{ fontSize:8, fill:"#334155", fontFamily:"monospace" }}>0</text>
        <text x={cx + r + 8} y={cy + 8} textAnchor="start"
          style={{ fontSize:8, fill:"#334155", fontFamily:"monospace" }}>15+</text>
      </svg>
      {!animating && (
        <div style={{ display:"inline-flex", alignItems:"center", gap:8,
          padding:"5px 16px", borderRadius:20,
          background: status.bg, border:`1px solid ${status.color}44` }}>
          <div style={{ width:7, height:7, borderRadius:"50%",
            background: status.color, boxShadow:`0 0 6px ${status.color}` }} />
          <span style={{ color: status.color, fontSize:13, fontWeight:700,
            fontFamily:"'DM Sans',sans-serif", letterSpacing:"0.05em" }}>
            {status.label}
          </span>
        </div>
      )}
    </div>
  );
}

// ─── MAIN APP ─────────────────────────────────────────────────────────────────
export default function App() {
  const [tab, setTab]           = useState("cbr");
  const [soilGroup, setSoil]    = useState("Lagos_Blue_Clay");
  const [cbr, setCbr]           = useState(26);
  const [e50, setE50]           = useState(141500);
  const [cref, setCref]         = useState(10);
  const [phi, setPhi]           = useState(37);
  const [thickness, setThick]   = useState(150);
  const [axleIdx, setAxle]      = useState(2);
  const [result, setResult]     = useState(null);
  const [animating, setAnim]    = useState(false);
  const [history, setHistory]   = useState([]);
  const [showInfo, setInfo]     = useState(false);

  const sg = SOILS.find(s => s.id === soilGroup);
  const load = AXLE_TYPES[axleIdx].load;

  // When soil group changes, update defaults
  function changeSoil(id) {
    setSoil(id);
    const d = SOIL_DEFAULTS[id];
    setCref(d.cref);
    setPhi(d.phi);
    if (tab === "cbr") {
      const midCbr = Math.round((SOILS.find(s=>s.id===id).cbr[0] + SOILS.find(s=>s.id===id).cbr[1]) / 2);
      setCbr(midCbr);
      setE50(cbrToE50(midCbr, id));
    }
  }

  // Keep E50 synced to CBR in CBR mode
  useEffect(() => {
    if (tab === "cbr") setE50(cbrToE50(cbr, soilGroup));
  }, [cbr, soilGroup, tab]);

  function loadSite(site) {
    setSoil(site.soilGroup);
    setCbr(site.cbr);
    setE50(cbrToE50(site.cbr, site.soilGroup));
    setThick(site.thickness);
    setAxle(AXLE_TYPES.findIndex(a => a.load === site.load));
    const d = SOIL_DEFAULTS[site.soilGroup];
    setCref(d.cref);
    setPhi(d.phi);
    setResult(null);
  }

  function predict() {
    setAnim(true);
    setResult(null);
    setTimeout(() => {
      const rut = predictRut({
        e50, cref, phi,
        thickness: thickness / 1000,
        load, soilGroup
      });
      setResult(rut);
      setAnim(false);
      setHistory(h => [{
        soil: sg.label, cbr: tab==="cbr"?cbr:null,
        e50, cref, phi, thickness, load, rut,
        status: rutStatus(rut).label
      }, ...h].slice(0, 8));
    }, 800);
  }

  function reset() {
    setResult(null);
    setHistory([]);
    setSoil("Lagos_Blue_Clay");
    setCbr(5); setE50(cbrToE50(5,"Lagos_Blue_Clay"));
    setCref(28); setPhi(22); setThick(150); setAxle(2);
  }

  const btnDisabled = animating;

  return (
    <div style={{
      minHeight:"100vh", background:"#020617",
      fontFamily:"'DM Sans',system-ui,sans-serif", color:"#e2e8f0",
      paddingBottom:60
    }}>
      {/* ── HEADER ── */}
      <div style={{
        background:"linear-gradient(180deg,#0f172a 0%,#020617 100%)",
        borderBottom:"1px solid #1e293b", padding:"20px 24px 16px"
      }}>
        <div style={{ maxWidth:1100, margin:"0 auto" }}>
          <div style={{ display:"flex", alignItems:"flex-start",
            justifyContent:"space-between", flexWrap:"wrap", gap:12 }}>
            <div>
              <div style={{ fontSize:10, color:"#3b82f6", letterSpacing:"0.15em",
                textTransform:"uppercase", marginBottom:4 }}>
                University of Ibadan · Dept. of Civil Engineering · TCE 799
              </div>
              <h1 style={{ margin:0, fontSize:20, fontWeight:700,
                color:"#f1f5f9", letterSpacing:"-0.02em" }}>
                SW Nigeria Rutting Deformation Predictor
              </h1>
              <p style={{ margin:"4px 0 0", color:"#475569", fontSize:13 }}>
                Machine learning surrogate model trained on 500 PLAXIS 3D finite element simulations
              </p>
            </div>
            <div style={{ display:"flex", gap:8, flexWrap:"wrap" }}>
              {[
                ["CV R²", "0.9876"],
                ["RMSE", "0.23 mm"],
                ["MAPE", "6.37%"],
                ["FEA Runs", "500"],
              ].map(([k,v]) => (
                <div key={k} style={{
                  padding:"6px 12px", borderRadius:8,
                  background:"#0f172a", border:"1px solid #1e293b",
                  textAlign:"center"
                }}>
                  <div style={{ fontSize:16, fontWeight:700, color:"#60a5fa",
                    fontFamily:"monospace" }}>{v}</div>
                  <div style={{ fontSize:9, color:"#334155", letterSpacing:"0.1em",
                    textTransform:"uppercase" }}>{k}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      <div style={{ maxWidth:1100, margin:"0 auto", padding:"20px 16px",
        display:"grid", gridTemplateColumns:"1fr 1fr", gap:16 }}>

        {/* ══ LEFT COLUMN — INPUTS ══ */}
        <div style={{ display:"flex", flexDirection:"column", gap:14 }}>

          {/* Known sites */}
          <Card>
            <SectionTitle>Quick Load — Real SW Nigerian Sites</SectionTitle>
            <div style={{ display:"flex", flexDirection:"column", gap:6 }}>
              {KNOWN_SITES.map((site, i) => (
                <button key={i} onClick={() => loadSite(site)} style={{
                  padding:"9px 12px", borderRadius:8, border:"1px solid #1e293b",
                  background:"#0a1628", cursor:"pointer", textAlign:"left",
                  transition:"border-color 0.15s",
                  display:"flex", alignItems:"center", gap:10
                }}
                  onMouseEnter={e => e.currentTarget.style.borderColor="#3b82f644"}
                  onMouseLeave={e => e.currentTarget.style.borderColor="#1e293b"}
                >
                  <div style={{ width:8, height:8, borderRadius:"50%",
                    background: SOILS.find(s=>s.id===site.soilGroup).accent,
                    flexShrink:0 }} />
                  <span style={{ fontSize:12, color:"#94a3b8" }}>{site.label}</span>
                </button>
              ))}
            </div>
          </Card>

          {/* Input mode */}
          <Card>
            <SectionTitle>Input Mode</SectionTitle>
            <div style={{ display:"flex", borderRadius:8, overflow:"hidden",
              border:"1px solid #1e293b", marginBottom:16 }}>
              {[
                { id:"cbr",      label:"CBR Input",       sub:"For practitioners" },
                { id:"advanced", label:"Full Parameters",  sub:"For researchers" },
              ].map(m => (
                <button key={m.id} onClick={() => setTab(m.id)} style={{
                  flex:1, padding:"10px", border:"none", cursor:"pointer",
                  background: tab===m.id ? "#1e3a5f" : "transparent",
                  borderRight: m.id==="cbr" ? "1px solid #1e293b" : "none",
                  transition:"background 0.2s"
                }}>
                  <div style={{ color: tab===m.id ? "#93c5fd" : "#475569",
                    fontSize:13, fontWeight:700 }}>{m.label}</div>
                  <div style={{ color:"#334155", fontSize:10, marginTop:1 }}>{m.sub}</div>
                </button>
              ))}
            </div>

            {/* Soil group */}
            <Label>Subgrade Soil Type</Label>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:8, marginBottom:18 }}>
              {SOILS.map(s => (
                <button key={s.id} onClick={() => changeSoil(s.id)} style={{
                  padding:"10px 10px", borderRadius:8, cursor:"pointer",
                  border:`1px solid ${soilGroup===s.id ? s.accent+"88" : "#1e293b"}`,
                  background: soilGroup===s.id ? s.accent+"12" : "transparent",
                  textAlign:"left", transition:"all 0.15s"
                }}>
                  <div style={{ color: soilGroup===s.id ? s.accent : "#64748b",
                    fontSize:12, fontWeight:700 }}>{s.label}</div>
                  <div style={{ color:"#334155", fontSize:10, marginTop:2 }}>{s.sub}</div>
                </button>
              ))}
            </div>

            {/* CBR input */}
            {tab === "cbr" && (
              <>
                <Label hint={`${cbr}%`}>CBR Value (soaked)</Label>
                <Slider min={sg.cbr[0]} max={sg.cbr[1]} step={1}
                  value={cbr} onChange={setCbr} color={sg.accent} />
                <div style={{ color:"#334155", fontSize:10, marginBottom:14, marginTop:4 }}>
                  Range for {sg.label}: {sg.cbr[0]}–{sg.cbr[1]}%
                </div>
                <div style={{ background:"#0a1628", borderRadius:8, padding:"10px 12px",
                  display:"flex", justifyContent:"space-between", alignItems:"center",
                  marginBottom:4 }}>
                  <div>
                    <div style={{ fontSize:11, color:"#64748b" }}>Derived E50Ref (Powell, 1984)</div>
                    <div style={{ fontSize:9, color:"#1e293b", marginTop:1 }}>
                      E(MPa) = 17.6 × CBR^0.64 → clamped to soil group range
                    </div>
                  </div>
                  <div style={{ fontFamily:"monospace", fontWeight:700, fontSize:14,
                    color: sg.accent }}>{e50.toLocaleString()} kPa</div>
                </div>
              </>
            )}

            {/* Advanced inputs */}
            {tab === "advanced" && (
              <>
                <Label hint={`${e50.toLocaleString()} kPa`}>E50Ref — Subgrade Stiffness</Label>
                <Slider min={5000} max={150000} step={1000}
                  value={e50} onChange={setE50} color={sg.accent} />
                <div style={{ fontSize:9, color:"#334155", marginBottom:14, marginTop:3 }}>
                  Secant stiffness at 50% failure stress — from triaxial test
                </div>

                <Label hint={`${cref} kPa`}>Cohesion (cRef)</Label>
                <Slider min={5} max={50} step={0.5}
                  value={cref} onChange={setCref} color="#8b5cf6" />
                <div style={{ fontSize:9, color:"#334155", marginBottom:14, marginTop:3 }}>
                  Effective cohesion from shear box or triaxial test
                </div>

                <Label hint={`${phi}°`}>Friction Angle (phi)</Label>
                <Slider min={18} max={42} step={0.5}
                  value={phi} onChange={setPhi} color="#ec4899" />
                <div style={{ fontSize:9, color:"#334155", marginBottom:4, marginTop:3 }}>
                  Effective internal friction angle
                </div>
              </>
            )}
          </Card>

          {/* Pavement design */}
          <Card>
            <SectionTitle>Pavement Design</SectionTitle>
            <Label hint={`${thickness} mm`}>Asphalt Layer Thickness</Label>
            <Slider min={75} max={250} step={5}
              value={thickness} onChange={setThick} color="#f59e0b" />
            <div style={{ display:"flex", justifyContent:"space-between",
              fontSize:9, color:"#334155", marginBottom:18, marginTop:3 }}>
              <span>75 mm (min)</span>
              <span>Federal Highway Manual: 75–250 mm</span>
              <span>250 mm (max)</span>
            </div>

            <Label>Axle Load Type</Label>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
              {AXLE_TYPES.map((a, i) => (
                <button key={i} onClick={() => setAxle(i)} style={{
                  padding:"8px 10px", borderRadius:8, cursor:"pointer",
                  border:`1px solid ${axleIdx===i ? "#f59e0b88" : "#1e293b"}`,
                  background: axleIdx===i ? "#f59e0b12" : "transparent",
                  textAlign:"left", transition:"all 0.15s", display:"flex",
                  alignItems:"center", gap:8
                }}>
                  <span style={{ fontSize:16 }}>{a.icon}</span>
                  <div>
                    <div style={{ fontSize:11, color: axleIdx===i ? "#fcd34d" : "#64748b",
                      fontWeight:700 }}>{a.label}</div>
                    <div style={{ fontSize:9, color:"#334155",
                      fontFamily:"monospace" }}>{a.load} kN/m²</div>
                  </div>
                </button>
              ))}
            </div>
          </Card>

          {/* Buttons */}
          <button onClick={predict} disabled={btnDisabled} style={{
            width:"100%", padding:"15px", borderRadius:10,
            background: btnDisabled
              ? "#1e293b"
              : "linear-gradient(135deg, #1d4ed8 0%, #3b82f6 100%)",
            border:"none", color: btnDisabled ? "#475569" : "#fff",
            fontSize:15, fontWeight:700, cursor: btnDisabled ? "not-allowed" : "pointer",
            letterSpacing:"0.03em", transition:"all 0.2s",
            boxShadow: btnDisabled ? "none" : "0 4px 24px #3b82f640"
          }}>
            {animating ? "Running surrogate model..." : "Predict Rut Depth"}
          </button>
          <button onClick={reset} style={{
            width:"100%", padding:"9px", borderRadius:10,
            border:"1px solid #1e293b", background:"transparent",
            color:"#475569", fontSize:12, cursor:"pointer"
          }}>Clear All</button>
        </div>

        {/* ══ RIGHT COLUMN — RESULTS ══ */}
        <div style={{ display:"flex", flexDirection:"column", gap:14 }}>

          {/* Gauge */}
          <Card accent={!!result} style={{ minHeight:220, display:"flex",
            flexDirection:"column", alignItems:"center", justifyContent:"center" }}>
            {!result && !animating ? (
              <div style={{ textAlign:"center", color:"#1e293b" }}>
                <div style={{ fontSize:48, marginBottom:8 }}>◎</div>
                <div style={{ color:"#334155", fontSize:14 }}>
                  Set parameters and click Predict
                </div>
                <div style={{ color:"#1e293b", fontSize:11, marginTop:6 }}>
                  Model trained on 500 PLAXIS 3D FEA simulations
                </div>
              </div>
            ) : (
              <>
                <div style={{ fontSize:10, color:"#334155", letterSpacing:"0.15em",
                  textTransform:"uppercase", marginBottom:12 }}>
                  Predicted Rut Depth
                </div>
                <ResultGauge rut={result || 0} animating={animating} />
                {result && (
                  <div style={{ marginTop:16, textAlign:"center" }}>
                    <div style={{ fontSize:11, color:"#334155" }}>
                      Federal Highway Manual failure threshold: <span style={{ color:"#ef4444" }}>20 mm</span>
                    </div>
                    <div style={{ fontSize:10, color:"#1e293b", marginTop:4 }}>
                      {result < 20
                        ? `${(20 - result).toFixed(2)} mm margin to failure threshold`
                        : "⚠️ Exceeds failure threshold — redesign required"}
                    </div>
                  </div>
                )}
              </>
            )}
          </Card>

          {/* Input summary */}
          {result && (
            <Card>
              <SectionTitle>Input Summary</SectionTitle>
              <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6 }}>
                {[
                  ["Soil type",         sg.label,                    sg.accent],
                  tab==="cbr"
                    ? ["CBR (field/lab)",  `${cbr}%`,                 "#60a5fa"]
                    : ["E50Ref",          `${e50.toLocaleString()} kPa`, "#60a5fa"],
                  ["E50Ref",            `${e50.toLocaleString()} kPa`, "#60a5fa"],
                  ["Cohesion",          `${cref} kPa`,               "#8b5cf6"],
                  ["Friction angle",    `${phi}°`,                   "#ec4899"],
                  ["Asphalt thickness", `${thickness} mm`,           "#f59e0b"],
                  ["Axle load",         `${load} kN/m²`,             "#f59e0b"],
                  ["Predicted rut",     `${result.toFixed(3)} mm`,   rutStatus(result).color],
                ].filter((_, i) => !(tab==="cbr" && i===2))
                 .map(([k, v, c]) => (
                  <div key={k} style={{
                    padding:"8px 10px", background:"#0a1628", borderRadius:6,
                    border:"1px solid #1e293b"
                  }}>
                    <div style={{ fontSize:10, color:"#334155",
                      textTransform:"uppercase", letterSpacing:"0.08em",
                      marginBottom:3 }}>{k}</div>
                    <div style={{ fontSize:13, color:c, fontFamily:"monospace",
                      fontWeight:700 }}>{v}</div>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* Feature importance */}
          <Card>
            <SectionTitle>Variable Importance (trained model)</SectionTitle>
            {[
              ["E50 — subgrade stiffness", 70.98, "#60a5fa"],
              ["Asphalt thickness",        16.51, "#f59e0b"],
              ["Axle load",                 9.96, "#34d399"],
              ["Friction angle",            1.85, "#ec4899"],
              ["Cohesion",                  0.54, "#8b5cf6"],
              ["Soil type (group)",         0.16, "#94a3b8"],
            ].map(([name, pct, color]) => (
              <div key={name} style={{ marginBottom:10 }}>
                <div style={{ display:"flex", justifyContent:"space-between",
                  marginBottom:4 }}>
                  <span style={{ fontSize:12, color:"#94a3b8" }}>{name}</span>
                  <span style={{ fontSize:12, color, fontFamily:"monospace",
                    fontWeight:700 }}>{pct.toFixed(1)}%</span>
                </div>
                <div style={{ height:5, borderRadius:3, background:"#1e293b" }}>
                  <div style={{ height:"100%", borderRadius:3,
                    width:`${pct}%`, background:color,
                    boxShadow:`0 0 6px ${color}66` }} />
                </div>
              </div>
            ))}
            <div style={{ fontSize:10, color:"#334155", marginTop:12, lineHeight:1.6 }}>
              E50 dominates rutting prediction — subgrade stiffness is the primary
              design variable for SW Nigerian flexible pavements.
            </div>
          </Card>

          {/* History */}
          {history.length > 0 && (
            <Card>
              <SectionTitle>Comparison History ({history.length} scenarios)</SectionTitle>
              <div style={{ overflowX:"auto" }}>
                <table style={{ width:"100%", borderCollapse:"collapse",
                  fontSize:11, fontFamily:"monospace" }}>
                  <thead>
                    <tr>
                      {["Soil","t(mm)","Load","Rut(mm)","Status"].map(h => (
                        <th key={h} style={{ color:"#334155", fontWeight:400,
                          padding:"4px 8px", textAlign:"left",
                          borderBottom:"1px solid #1e293b", fontFamily:"sans-serif",
                          fontSize:10, textTransform:"uppercase",
                          letterSpacing:"0.08em" }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((r, i) => {
                      const st = rutStatus(r.rut);
                      const sg2 = SOILS.find(s => s.id === r.soil ||
                        s.label === r.soil);
                      return (
                        <tr key={i} style={{ opacity: Math.max(0.3, 1 - i * 0.1) }}>
                          <td style={{ padding:"5px 8px",
                            color: sg2?.accent || "#94a3b8", fontSize:10 }}>
                            {(sg2?.label || r.soil).split(" ")[0]}
                          </td>
                          <td style={{ padding:"5px 8px", color:"#64748b" }}>{r.thickness}</td>
                          <td style={{ padding:"5px 8px", color:"#64748b" }}>{r.load}</td>
                          <td style={{ padding:"5px 8px", color:st.color,
                            fontWeight:700 }}>{r.rut.toFixed(3)}</td>
                          <td style={{ padding:"5px 8px", color:st.color,
                            fontSize:10 }}>{r.status}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}

          {/* Citation */}
          <div style={{ padding:"12px 14px", borderRadius:10,
            border:"1px solid #1e293b", background:"#050d1a" }}>
            <div style={{ fontSize:9, color:"#334155", lineHeight:1.8 }}>
              <strong style={{ color:"#475569" }}>Citation:</strong> Olalewa, A. (2026).
              Predicting Rutting Deformation in Flexible Pavements of Southwest Nigeria:
              A Machine Learning-Based Surrogate Model Derived from 3D Finite Element
              Analysis. MSc Thesis, University of Ibadan. · CBR conversion: Powell et al.
              (1984) · Soil parameters: Ola (1983); Gidigasu (1976); Bello (2011);
              Akintayo & Ibrahim (2024)
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
