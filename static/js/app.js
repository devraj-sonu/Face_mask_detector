/* ═══════════════════════════════════════════════════════════════
   MaskGuard AI — Frontend Logic
   Handles: upload, webcam, API calls, UI rendering, sound, chart
   ═══════════════════════════════════════════════════════════════ */

const API_BASE = window.location.origin;  // same server

/* ══════════════════ MODE SWITCHING ══════════════════ */
function switchMode(mode) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`.tab[data-mode="${mode}"]`).classList.add('active');

  document.getElementById('uploadMode').style.display  = mode === 'upload'  ? '' : 'none';
  document.getElementById('webcamMode').style.display  = mode === 'webcam'  ? '' : 'none';

  if (mode !== 'webcam') stopWebcam();
  clearResult();
}

/* ══════════════════ UPLOAD ══════════════════ */
let selectedFile = null;

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) loadFilePreview(file);
}

function handleDragOver(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.add('drag-over');
}

function handleDrop(e) {
  e.preventDefault();
  document.getElementById('dropZone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadFilePreview(file);
}

function loadFilePreview(file) {
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = ev => {
    document.getElementById('previewImg').src = ev.target.result;
    document.getElementById('dropZone').style.display     = 'none';
    document.getElementById('imagePreview').style.display = '';
    document.getElementById('uploadDetectBtn').disabled   = false;
  };
  reader.readAsDataURL(file);
}

function clearUpload() {
  selectedFile = null;
  document.getElementById('fileInput').value = '';
  document.getElementById('dropZone').style.display     = '';
  document.getElementById('imagePreview').style.display = 'none';
  document.getElementById('uploadDetectBtn').disabled   = true;
  clearResult();
}

async function detectFromUpload() {
  if (!selectedFile) return;
  showLoader();

  const formData = new FormData();
  formData.append('image', selectedFile);

  try {
    const res  = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: formData });
    const data = await res.json();
    displayResult(data);
  } catch (err) {
    showError('API connection failed. Is the Flask server running?');
  }
}

/* ══════════════════ WEBCAM ══════════════════ */
let stream         = null;
let autoDetectTimer = null;

async function startWebcam() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    const video = document.getElementById('webcamVideo');
    video.srcObject = stream;
    document.getElementById('webcamOverlay').style.display = 'none';
    document.getElementById('startCamBtn').style.display  = 'none';
    document.getElementById('stopCamBtn').style.display   = '';
    document.getElementById('snapBtn').style.display      = '';
    document.getElementById('snapBtn').disabled           = false;
  } catch (err) {
    showError('Camera access denied. Please allow camera permission.');
  }
}

function stopWebcam() {
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  if (autoDetectTimer) { clearInterval(autoDetectTimer); autoDetectTimer = null; }
  document.getElementById('autoDetect').checked           = false;
  document.getElementById('webcamOverlay').style.display  = '';
  document.getElementById('startCamBtn').style.display    = '';
  document.getElementById('stopCamBtn').style.display     = 'none';
  document.getElementById('snapBtn').style.display        = 'none';
}

function snapAndDetect() {
  const video  = document.getElementById('webcamVideo');
  const canvas = document.getElementById('webcamCanvas');
  canvas.width  = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);

  canvas.toBlob(async blob => {
    showLoader();
    const formData = new FormData();
    formData.append('image', blob, 'webcam.jpg');
    try {
      const res  = await fetch(`${API_BASE}/api/detect`, { method: 'POST', body: formData });
      const data = await res.json();
      displayResult(data);
    } catch (err) {
      showError('API connection failed. Is the Flask server running?');
    }
  }, 'image/jpeg', 0.92);
}

function toggleAutoDetect() {
  const on = document.getElementById('autoDetect').checked;
  if (on) {
    autoDetectTimer = setInterval(() => {
      if (stream) snapAndDetect();
    }, 2000);
  } else {
    clearInterval(autoDetectTimer);
    autoDetectTimer = null;
  }
}

/* ══════════════════ RESULT RENDERING ══════════════════ */
function showLoader() {
  document.getElementById('resultPlaceholder').style.display = 'none';
  document.getElementById('resultContent').style.display     = 'none';
  document.getElementById('loader').style.display            = '';
}

function clearResult() {
  document.getElementById('resultPlaceholder').style.display = '';
  document.getElementById('resultContent').style.display     = 'none';
  document.getElementById('loader').style.display            = 'none';
  dismissAlert();
}

function showError(msg) {
  document.getElementById('loader').style.display = 'none';
  document.getElementById('resultPlaceholder').style.display = '';
  document.getElementById('resultPlaceholder').innerHTML = `
    <div class="placeholder-icon">⚠</div>
    <p style="color:var(--red)">${msg}</p>`;
}

function displayResult(data) {
  document.getElementById('loader').style.display = 'none';

  if (!data.success) {
    showError(data.error || 'Detection failed');
    return;
  }

  // Annotated image
  document.getElementById('annotatedImg').src = data.annotated_image;

  // Summary cards
  const preds     = data.predictions;
  const maskCnt   = preds.filter(p => p.label === 'Mask').length;
  const noMaskCnt = preds.filter(p => p.label === 'No Mask').length;

  document.getElementById('summaryCards').innerHTML = `
    <div class="summary-card">
      <div class="sv">${data.face_count}</div>
      <div class="sl">FACES</div>
    </div>
    <div class="summary-card green">
      <div class="sv" style="color:var(--green)">${maskCnt}</div>
      <div class="sl">WITH MASK</div>
    </div>
    <div class="summary-card red">
      <div class="sv" style="color:var(--red)">${noMaskCnt}</div>
      <div class="sl">NO MASK</div>
    </div>`;

  // Prediction list
  const predHTML = preds.length
    ? preds.map((p, i) => {
        const isMask = p.label === 'Mask';
        return `
          <div class="pred-item">
            <span class="pred-badge ${isMask ? 'mask' : 'nomask'}">${p.label.toUpperCase()}</span>
            <span style="color:var(--text2)">Face #${i + 1}</span>
            <div class="conf-bar-wrap">
              <div class="conf-bar ${isMask ? 'mask' : 'nomask'}" style="width:${p.confidence}%"></div>
            </div>
            <span class="pred-conf">${p.confidence}%</span>
          </div>`;
      }).join('')
    : `<div class="pred-item" style="justify-content:center;color:var(--text3)">No faces detected</div>`;

  document.getElementById('predList').innerHTML = predHTML;

  // Show result
  document.getElementById('resultPlaceholder').style.display = 'none';
  document.getElementById('resultContent').style.display     = '';

  // Alert for No Mask
  if (data.no_mask_alert) {
    playAlertSound();
    showAlert();
  }
}

/* ══════════════════ ALERT ══════════════════ */
function showAlert() {
  const banner = document.getElementById('alertBanner');
  banner.style.display = '';
  setTimeout(dismissAlert, 6000);
}

function dismissAlert() {
  document.getElementById('alertBanner').style.display = 'none';
}

/* ══════════════════ ALERT SOUND (Web Audio API) ══════════════════ */
function playAlertSound() {
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const freqs = [880, 660, 880];
    freqs.forEach((f, i) => {
      const osc  = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = 'square';
      osc.frequency.value = f;
      gain.gain.setValueAtTime(.18, ctx.currentTime + i * .18);
      gain.gain.exponentialRampToValueAtTime(.001, ctx.currentTime + i * .18 + .15);
      osc.start(ctx.currentTime + i * .18);
      osc.stop(ctx.currentTime + i * .18 + .15);
    });
  } catch (e) { /* Audio not available */ }
}

/* ══════════════════ DASHBOARD ══════════════════ */
let activityChart = null;

async function loadStats() {
  try {
    const res  = await fetch(`${API_BASE}/api/stats`);
    const data = await res.json();

    document.querySelector('#sc-total .stat-val').textContent  = data.total;
    document.querySelector('#sc-mask .stat-val').textContent   = data.mask;
    document.querySelector('#sc-nomask .stat-val').textContent = data.no_mask;
    const rate = data.total ? Math.round((data.mask / data.total) * 100) : 0;
    document.querySelector('#sc-rate .stat-val').textContent   = `${rate}%`;

    // Chart
    const labels = data.recent.map(r => r.hour.slice(11)); // HH
    const values = data.recent.map(r => r.count);

    if (activityChart) activityChart.destroy();
    const ctx = document.getElementById('activityChart').getContext('2d');
    activityChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'Detections',
          data: values,
          backgroundColor: 'rgba(0,229,160,.25)',
          borderColor: 'rgba(0,229,160,.8)',
          borderWidth: 1,
          borderRadius: 4,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: {
            ticks: { color: '#50556a', font: { family: 'Space Mono', size: 10 } },
            grid:  { color: '#1a1c21' }
          },
          y: {
            ticks: { color: '#50556a', font: { family: 'Space Mono', size: 10 } },
            grid:  { color: '#1a1c21' }
          }
        }
      }
    });
  } catch (e) { console.warn('Stats fetch failed:', e); }
}

/* ══════════════════ LOGS ══════════════════ */
async function loadLogs() {
  try {
    const res  = await fetch(`${API_BASE}/api/logs?n=50`);
    const logs = await res.json();
    const tbody = document.getElementById('logTableBody');

    if (!logs.length) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty-row">No logs yet</td></tr>';
      return;
    }

    tbody.innerHTML = logs.slice().reverse().map(entry => {
      const maskCnt   = entry.detections.filter(d => d.label === 'Mask').length;
      const noMaskCnt = entry.detections.filter(d => d.label === 'No Mask').length;
      const ts        = new Date(entry.timestamp).toLocaleString();
      const status    = noMaskCnt > 0
        ? `<span class="badge-bad">⚠ Alert</span>`
        : `<span class="badge-ok">✓ OK</span>`;
      return `
        <tr>
          <td>${ts}</td>
          <td>${entry.detections.length}</td>
          <td>${maskCnt}</td>
          <td>${noMaskCnt}</td>
          <td>${status}</td>
        </tr>`;
    }).join('');
  } catch (e) { console.warn('Logs fetch failed:', e); }
}

/* ══════════════════ NAV HIGHLIGHTING ══════════════════ */
const sections = ['detect', 'dashboard', 'logs'];
window.addEventListener('scroll', () => {
  let current = 'detect';
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el && window.scrollY >= el.offsetTop - 100) current = id;
  });
  document.querySelectorAll('.nav-link').forEach(a => {
    a.classList.toggle('active', a.getAttribute('href') === `#${current}`);
  });
});

/* ══════════════════ INIT ══════════════════ */
loadStats();
loadLogs();
