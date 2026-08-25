// Appliance Fault Profiles for Mobile App
const APPLIANCE_FAULTS = {
  refrigerator: [
    { id: "compressor_bearing_wear", label: "Bearing Wear" },
    { id: "fan_blade_friction", label: "Fan Friction" },
    { id: "relay_chatter", label: "Relay Chatter" }
  ],
  washing_machine: [
    { id: "unbalance_tub", label: "Tub Unbalance" },
    { id: "belt_slip", label: "Belt Slip" },
    { id: "drain_cavitation", label: "Drain Cavitation" }
  ],
  water_pump: [
    { id: "impeller_cavitation", label: "Impeller Cavitation" },
    { id: "dry_running", label: "Dry Running" },
    { id: "motor_bearing_wear", label: "Motor Bearing" },
    { id: "pipe_water_hammer", label: "Water Hammer" }
  ]
};

let currentAppliance = "water_pump";
let activeFault = "normal";
let ws = null;

// Canvases
const audioCanvas = document.getElementById("audioCanvas");
const vibCanvas = document.getElementById("vibCanvas");
const specCanvas = document.getElementById("specCanvas");

const audioCtx = audioCanvas.getContext("2d");
const vibCtx = vibCanvas.getContext("2d");
const specCtx = specCanvas.getContext("2d");

const SPEC_HISTORY_LENGTH = 100;
let specHistory = [];

function resizeCanvases() {
  const dpr = window.devicePixelRatio || 1;
  [audioCanvas, vibCanvas, specCanvas].forEach(canvas => {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
  });
}
window.addEventListener("resize", resizeCanvases);
resizeCanvases();

// Spectrogram Color Gradient for Light Mode (Vibrant Turbo / Heat gradient)
function getHeatmapColorLight(val) {
  const clamped = Math.max(0, Math.min(100, val)) / 100;
  let r, g, b;
  if (clamped < 0.25) {
    const t = clamped / 0.25;
    r = Math.floor(245 - 60 * t);
    g = Math.floor(248 - 30 * t);
    b = Math.floor(255 - 10 * t);
  } else if (clamped < 0.5) {
    const t = (clamped - 0.25) / 0.25;
    r = Math.floor(59 + 180 * t);
    g = Math.floor(130 + 60 * t);
    b = Math.floor(246 - 200 * t);
  } else if (clamped < 0.75) {
    const t = (clamped - 0.5) / 0.25;
    r = Math.floor(239 + 10 * t);
    g = Math.floor(190 - 120 * t);
    b = Math.floor(46 - 40 * t);
  } else {
    const t = (clamped - 0.75) / 0.25;
    r = Math.floor(225 - 50 * t);
    g = Math.floor(29 - 20 * t);
    b = Math.floor(72 + 20 * t);
  }
  return `rgb(${r},${g},${b})`;
}

// Draw Audio Waveform (Crisp Electric Blue on Light Slate)
function drawAudioWaveform(samples) {
  if (!samples || samples.length === 0 || !audioCanvas) return;
  const w = audioCanvas.width;
  const h = audioCanvas.height;
  const cy = h / 2;

  audioCtx.fillStyle = "#f8fafc";
  audioCtx.fillRect(0, 0, w, h);

  // Center baseline
  audioCtx.strokeStyle = "#e2e8f0";
  audioCtx.lineWidth = 1;
  audioCtx.beginPath();
  audioCtx.moveTo(0, cy);
  audioCtx.lineTo(w, cy);
  audioCtx.stroke();

  // Waveform
  audioCtx.strokeStyle = "#2563eb";
  audioCtx.lineWidth = 2.2;
  audioCtx.beginPath();

  const sliceWidth = w / samples.length;
  for (let i = 0; i < samples.length; i++) {
    const x = i * sliceWidth;
    const y = cy - (samples[i] * cy * 0.88);
    if (i === 0) audioCtx.moveTo(x, y);
    else audioCtx.lineTo(x, y);
  }
  audioCtx.stroke();
}

// Draw 3-Axis Vibration Oscilloscope
function drawVibrationWaveform(vibX, vibY, vibZ) {
  if (!vibX || vibX.length === 0 || !vibCanvas) return;
  const w = vibCanvas.width;
  const h = vibCanvas.height;
  const cy = h / 2;

  vibCtx.fillStyle = "#f8fafc";
  vibCtx.fillRect(0, 0, w, h);

  vibCtx.strokeStyle = "#e2e8f0";
  vibCtx.lineWidth = 1;
  vibCtx.beginPath();
  vibCtx.moveTo(0, cy);
  vibCtx.lineTo(w, cy);
  vibCtx.stroke();

  const drawAxis = (data, color) => {
    vibCtx.strokeStyle = color;
    vibCtx.lineWidth = 2;
    vibCtx.beginPath();
    const sliceWidth = w / data.length;
    for (let i = 0; i < data.length; i++) {
      const x = i * sliceWidth;
      const y = cy - (data[i] * (cy * 0.42));
      if (i === 0) vibCtx.moveTo(x, y);
      else vibCtx.lineTo(x, y);
    }
    vibCtx.stroke();
  };

  drawAxis(vibX, "#0284c7"); // X: Blue
  drawAxis(vibY, "#059669"); // Y: Emerald
  drawAxis(vibZ, "#ea580c"); // Z: Coral Orange
}

// Draw Spectrogram Heatmap
function drawSpectrogram(melSlice) {
  if (!melSlice || melSlice.length === 0 || !specCanvas) return;

  const nBins = melSlice.length;
  const nTime = melSlice[0].length;
  const col = new Array(nBins);

  for (let i = 0; i < nBins; i++) {
    let sum = 0;
    for (let j = 0; j < nTime; j++) sum += melSlice[i][j];
    col[i] = sum / nTime;
  }

  specHistory.push(col);
  if (specHistory.length > SPEC_HISTORY_LENGTH) specHistory.shift();

  const w = specCanvas.width;
  const h = specCanvas.height;
  const colWidth = w / SPEC_HISTORY_LENGTH;
  const binHeight = h / nBins;

  specCtx.fillStyle = "#f8fafc";
  specCtx.fillRect(0, 0, w, h);

  for (let c = 0; c < specHistory.length; c++) {
    const currentColumn = specHistory[c];
    const x = c * colWidth;
    for (let b = 0; b < nBins; b++) {
      const y = h - (b + 1) * binHeight;
      const val = currentColumn[b];
      specCtx.fillStyle = getHeatmapColorLight(val);
      specCtx.fillRect(x, y, colWidth + 0.6, binHeight + 0.6);
    }
  }
}

// Update Mobile Dashboard UI State
function updateDashboardUI(data) {
  const pred = data.prediction || {};
  const safety = data.safety || {};
  const ap = data.autopilot || {};
  const inj = data.fault_injector || {};

  // 1. Health Radial Score Gauge
  const healthVal = pred.health_score !== undefined ? pred.health_score : 100.0;
  document.getElementById("healthScoreValue").innerText = `${Math.round(healthVal)}%`;

  // Update SVG Radial Circle (circumference = 314.159)
  const circle = document.getElementById("gaugeProgressCircle");
  if (circle) {
    const offset = 314.159 - (healthVal / 100) * 314.159;
    circle.style.strokeDashoffset = offset;
    if (healthVal >= 80) circle.style.stroke = "var(--accent-emerald)";
    else if (healthVal >= 55) circle.style.stroke = "var(--accent-amber)";
    else circle.style.stroke = "var(--accent-rose)";
  }

  // 2. Anomaly & Confidence
  const anomVal = pred.anomaly_score !== undefined ? pred.anomaly_score : 0.0;
  document.getElementById("anomalyScoreValue").innerText = anomVal.toFixed(3);
  document.getElementById("anomalyBarFill").style.width = `${Math.min(100, anomVal * 100)}%`;

  const confVal = (pred.fault_confidence || 0) * 100;
  document.getElementById("faultConfValue").innerText = `${confVal.toFixed(1)}%`;
  document.getElementById("confBarFill").style.width = `${confVal}%`;

  // 3. Fault Signature & Severity Pill
  const faultClass = pred.fault_class || "normal";
  document.getElementById("faultClassValue").innerText = faultClass.replace(/_/g, " ").toUpperCase();

  const sev = pred.severity || "low";
  const sevPill = document.getElementById("severityBadge");
  sevPill.innerText = `${sev.toUpperCase()} RISK`;
  sevPill.className = `severity-pill ${sev}`;

  // 4. Relay Badge & Trip Banner
  const relayState = safety.relay_state || "CLOSED";
  const relayBadge = document.getElementById("relayStatusBadge");
  const relayText = document.getElementById("relayStatusText");
  const tripBanner = document.getElementById("bannerTripWarning");

  if (relayState === "OPEN") {
    relayBadge.className = "relay-badge tripped";
    relayText.innerText = "TRIPPED (OFF)";
    tripBanner.classList.add("visible");
  } else {
    relayBadge.className = "relay-badge nominal";
    relayText.innerText = "230V ON";
    tripBanner.classList.remove("visible");
  }

  // 5. Autopilot Status Pill
  const allowOp = ap.allow_operation !== undefined ? ap.allow_operation : true;
  const apPill = document.getElementById("autopilotStatusPill");
  const apRec = document.getElementById("autopilotRec");

  if (allowOp) {
    apPill.className = "status-pill-green";
    apPill.innerText = "ALLOWED";
  } else {
    apPill.className = "status-pill-red";
    apPill.innerText = "BLOCKED";
  }
  apRec.innerText = ap.recommended_action || "Monitoring machine health profile.";

  // 6. Active Chip Highlight
  activeFault = inj.fault || "normal";
  document.querySelectorAll(".fault-chip").forEach(chip => {
    chip.classList.toggle("active-chip", chip.dataset.fault === activeFault && activeFault !== "normal");
  });

  // 7. Render Waves
  if (data.waveform) {
    drawAudioWaveform(data.waveform.audio);
    drawVibrationWaveform(data.waveform.vib_x, data.waveform.vib_y, data.waveform.vib_z);
  }
  if (data.spectrogram) {
    drawSpectrogram(data.spectrogram);
  }
}

// WebSocket Connection
function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => console.log("[WS] Connected to Mobile AcousticAI stream.");
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      updateDashboardUI(data);
    } catch (e) {
      console.error("[WS] Parse error:", e);
    }
  };
  ws.onclose = () => {
    setTimeout(connectWebSocket, 2000);
  };
}

// Populate Fault Chips
function renderFaultChips() {
  const container = document.getElementById("faultButtonsContainer");
  if (!container) return;
  container.innerHTML = "";
  const faults = APPLIANCE_FAULTS[currentAppliance] || [];

  faults.forEach(f => {
    const chip = document.createElement("button");
    chip.className = "fault-chip";
    chip.dataset.fault = f.id;
    chip.innerHTML = `<span>${f.label}</span><span style="font-size:0.8rem">⚡</span>`;
    chip.onclick = () => injectFault(f.id);
    container.appendChild(chip);
  });
}

// Tab Switching for Mobile
function switchNavTab(viewId, tabButton) {
  document.querySelectorAll(".app-view").forEach(v => v.classList.remove("active"));
  document.querySelectorAll(".nav-tab").forEach(t => t.classList.remove("active"));

  const targetView = document.getElementById(viewId);
  if (targetView) targetView.classList.add("active");
  if (tabButton) tabButton.classList.add("active");

  // Re-measure canvases when switching back to Monitor
  if (viewId === "viewMonitor") {
    setTimeout(resizeCanvases, 50);
  }
}

// API Calls
async function injectFault(faultType) {
  const intensity = parseFloat(document.getElementById("intensitySlider").value);
  const dur = parseFloat(document.getElementById("durationSelect").value);

  try {
    await fetch("/api/fault/inject", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ fault_type: faultType, intensity: intensity, duration_sec: dur })
    });
    appendLog(`[INJECT] Activated "${faultType}" (${Math.round(intensity * 100)}%)`, "warning");
  } catch (e) {
    console.error("Inject error:", e);
  }
}

async function clearFault() {
  try {
    await fetch("/api/fault/clear", { method: "POST" });
    appendLog("[CLEAR] Mechanical faults cleared. Baseline restored.", "normal");
  } catch (e) {
    console.error("Clear error:", e);
  }
}

async function resetRelay() {
  try {
    await fetch("/api/relay/reset", { method: "POST" });
    appendLog("[RELAY] Emergency cutoff reset. 230V restored.", "normal");
  } catch (e) {
    console.error("Relay error:", e);
  }
}

async function switchAppliance(applianceType) {
  currentAppliance = applianceType;
  document.querySelectorAll(".picker-tab").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.app === applianceType);
  });
  renderFaultChips();
  specHistory = [];

  try {
    await fetch("/api/appliance/select", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ appliance_type: applianceType })
    });
    appendLog(`[APPLIANCE] Switched to ${applianceType.replace('_', ' ').toUpperCase()}`, "normal");
  } catch (e) {
    console.error("Switch error:", e);
  }
}

function appendLog(msg, type = "normal") {
  const container = document.getElementById("eventLogContainer");
  if (!container) return;
  const item = document.createElement("div");
  item.className = `log-entry ${type}`;
  const timeStr = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  item.innerHTML = `<span class="log-timestamp">[${timeStr}]</span> ${msg}`;
  container.prepend(item);
  if (container.children.length > 30) container.removeChild(container.lastChild);
}

async function fetchRecentEvents() {
  try {
    const res = await fetch("/api/events?limit=10");
    const json = await res.json();
    if (json.events && json.events.length > 0) {
      const container = document.getElementById("eventLogContainer");
      if (!container) return;
      container.innerHTML = "";
      json.events.forEach(e => {
        const item = document.createElement("div");
        const type = e.severity === "critical" ? "critical" : (e.severity === "medium" ? "warning" : "normal");
        item.className = `log-entry ${type}`;
        const timeStr = new Date(e.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
        item.innerHTML = `<span class="log-timestamp">[${timeStr}]</span> ${e.description}`;
        container.appendChild(item);
      });
    }
  } catch (e) {
    console.error("Events fetch error:", e);
  }
}

// Init
document.addEventListener("DOMContentLoaded", () => {
  renderFaultChips();
  connectWebSocket();
  fetchRecentEvents();
  setInterval(fetchRecentEvents, 5000);

  const slider = document.getElementById("intensitySlider");
  if (slider) {
    slider.addEventListener("input", (e) => {
      document.getElementById("intensityValue").innerText = `${Math.round(e.target.value * 100)}%`;
    });
  }
});
