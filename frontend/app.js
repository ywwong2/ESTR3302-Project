/* ─────────────────────────────────────────────────────────────
   Image Ranking Simulation — Frontend Logic
───────────────────────────────────────────────────────────── */

const API = 'http://127.0.0.1:8000';

// ── DOM refs ──────────────────────────────────────────────────
const initOverlay  = document.getElementById('initOverlay');
const appEl        = document.getElementById('app');
const imageGrid    = document.getElementById('imageGrid');
const imageCount   = document.getElementById('imageCount');
const searchInput  = document.getElementById('searchInput');
const titleInput   = document.getElementById('titleInput');
const fileInput    = document.getElementById('fileInput');
const roundBadge   = document.getElementById('roundBadge');
const rewardBadge  = document.getElementById('rewardBadge');
const simRound     = document.getElementById('simRound');
const simReward    = document.getElementById('simReward');
const toggleBtn    = document.getElementById('toggleBtn');
const intervalSlider = document.getElementById('intervalSlider');
const intervalVal  = document.getElementById('intervalVal');
const weightBars   = document.getElementById('weightBars');
const modal        = document.getElementById('modal');
const modalBody    = document.getElementById('modalBody');
const debugLogEl   = document.getElementById('debugLog');

const WEIGHT_LABELS = ['cos', 'fresh', 'lr', 'ctr', 'awt', 'social'];
let currentItems = [];
let simRunning = false;
let logSince = 0;

// ── Init overlay ──────────────────────────────────────────────
async function initPool(mode) {
  try {
    await fetch(`${API}/sim/init`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ mode }),
    });
    showApp();
    await refreshAll();
  } catch (e) {
    alert('Cannot reach backend. Make sure uvicorn is running on port 8000.');
  }
}

function skipInit() {
  showApp();
  refreshAll();
  return false;
}

function showApp() {
  initOverlay.classList.add('hidden');
  appEl.classList.remove('hidden');
}

// Check if backend already has state → skip overlay
async function checkExistingState() {
  try {
    const res = await fetch(`${API}/sim/status`);
    if (!res.ok) return;
    const state = await res.json();
    if (state.pool_mode !== null) {
      showApp();
      await refreshAll();
    }
  } catch (e) { /* backend not up */ }
}

// ── Simulation control ────────────────────────────────────────
async function toggleSim() {
  const res = await fetch(`${API}/sim/toggle`, { method: 'POST' });
  const data = await res.json();
  simRunning = data.running;
  updateToggleBtn();
}

async function stepSim() {
  const res = await fetch(`${API}/sim/step`, { method: 'POST' });
  const data = await res.json();
  if (data.ok) {
    await refreshAll();
  }
}

async function resetSim() {
  if (!confirm('Reset simulation to round 0?')) return;
  await fetch(`${API}/sim/reset`, { method: 'POST' });
  await refreshAll();
}

function updateInterval(val) {
  intervalVal.textContent = `${val}s`;
  fetch(`${API}/sim/set_interval`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ interval: Number(val) }),
  }).catch(() => {});
}

function updateToggleBtn() {
  if (simRunning) {
    toggleBtn.textContent = '⏸ Pause';
    toggleBtn.className = 'btn-toggle btn-stop';
  } else {
    toggleBtn.textContent = '▶ Start';
    toggleBtn.className = 'btn-toggle btn-start';
  }
}

function renderWeights(weights) {
  if (!weights || !weights.length) return;
  weightBars.innerHTML = '';
  const maxW = Math.max(...weights, 0.01);
  weights.forEach((w, i) => {
    const row = document.createElement('div');
    row.className = 'weight-row';
    row.innerHTML = `
      <span class="weight-label">${WEIGHT_LABELS[i]}</span>
      <div class="weight-track"><div class="weight-fill" style="width:${(w/maxW)*100}%"></div></div>
      <span class="weight-val">${w.toFixed(3)}</span>`;
    weightBars.appendChild(row);
  });
}

// ── Search ────────────────────────────────────────────────────
async function doSearch(query) {
  const q = (query !== undefined ? query : searchInput.value).trim();
  try {
    const res = await fetch(`${API}/images?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error();
    const data = await res.json();
    currentItems = data.items || [];
    renderGrid(currentItems);
    updateSimStatus(data);
  } catch (e) {
    console.error('Search failed', e);
  }
}

searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

// ── Upload ────────────────────────────────────────────────────
async function doUpload() {
  const file = fileInput.files[0];
  if (!file) { alert('Select a file first.'); return; }
  if (!file.type.startsWith('image/')) { alert('Only image files supported.'); return; }

  const fd = new FormData();
  fd.append('file', file);
  fd.append('title', titleInput.value.trim());

  try {
    const res = await fetch(`${API}/images/upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Upload failed');
    titleInput.value = '';
    fileInput.value = '';
    await doSearch('');
  } catch (e) {
    alert('Upload failed. Check backend.');
  }
}

// ── Image Grid ────────────────────────────────────────────────
function renderGrid(items) {
  imageCount.textContent = items.length;
  imageGrid.innerHTML = '';

  if (!items.length) {
    const p = document.createElement('p');
    p.style.cssText = 'color:#64748b;padding:20px;';
    p.textContent = 'No images yet. Upload something or load the dataset.';
    imageGrid.appendChild(p);
    return;
  }

  items.forEach((item, idx) => {
    const card = document.createElement('div');
    card.className = 'card' + (item.status === 'PROCESSING' ? ' card-loading' : '');

    const rank = document.createElement('div');
    rank.className = 'rank-badge';
    rank.textContent = `#${item.rank || idx + 1}`;

    const thumb = document.createElement('div');
    thumb.className = 'thumb';
    if (item.filename && !item.is_dataset) {
      const img = document.createElement('img');
      img.src = item.url || '';
      img.alt = item.title;
      img.loading = 'lazy';
      img.addEventListener('click', () => viewImage(item));
      thumb.appendChild(img);
    } else {
      const emoji = document.createElement('span');
      emoji.className = 'no-img';
      const icons = { cat: '🐱', dog: '🐶', plane: '✈️', clothes: '👗', car: '🚗' };
      emoji.textContent = icons[item.category] || '🖼️';
      thumb.appendChild(emoji);
    }

    if (item.status === 'PROCESSING') {
      const ov = document.createElement('div');
      ov.className = 'status-overlay';
      ov.textContent = '⏳ Embedding…';
      card.appendChild(ov);
    }

    const body = document.createElement('div');
    body.className = 'card-body';

    const catTag = document.createElement('span');
    catTag.className = 'cat-tag';
    catTag.textContent = item.category || 'unknown';

    const titleEl = document.createElement('div');
    titleEl.className = 'card-title';
    titleEl.textContent = item.title || item.id;

    const meta = document.createElement('div');
    meta.className = 'card-meta';
    const ctr = (item.ctr || 0).toFixed(3);
    const lr  = (item.lr  || 0).toFixed(3);
    const awt = (item.awt || 0).toFixed(2);
    const fresh = (item.freshness || 0).toFixed(2);
    const qs  = item.query_score !== undefined ? ` │ score:${item.query_score}` : '';
    meta.textContent = `CTR:${ctr} │ LR:${lr} │ AWT:${awt} │ fresh:${fresh}${qs}`;

    const actions = document.createElement('div');
    actions.className = 'card-actions';

    const likeBtn = document.createElement('button');
    likeBtn.className = 'btn-primary';
    likeBtn.textContent = '👍 Like';
    likeBtn.addEventListener('click', () => interact(item.id, 'like'));

    const delBtn = document.createElement('button');
    delBtn.className = 'btn-danger';
    delBtn.textContent = '🗑';
    delBtn.addEventListener('click', () => deleteImage(item.id));

    actions.appendChild(likeBtn);
    if (!item.is_dataset) actions.appendChild(delBtn);

    body.appendChild(catTag);
    body.appendChild(titleEl);
    body.appendChild(meta);
    body.appendChild(actions);

    card.appendChild(rank);
    card.appendChild(thumb);
    card.appendChild(body);
    imageGrid.appendChild(card);
  });

  const processing = items.some(i => i.status === 'PROCESSING');
  if (processing) setTimeout(() => doSearch(''), 3000);
}

// ── Interactions ──────────────────────────────────────────────
async function viewImage(item) {
  await interact(item.id, 'view');
  modalBody.innerHTML = `
    <h3 style="margin-bottom:12px">${item.title}</h3>
    <img src="${item.url}" style="max-width:100%;border-radius:8px;" />
    <div style="margin-top:10px;font-size:12px;color:#64748b">
      Category: ${item.category} │ CTR: ${(item.ctr||0).toFixed(3)} │ LR: ${(item.lr||0).toFixed(3)}
    </div>`;
  modal.classList.remove('hidden');
}

async function interact(imageId, action) {
  await fetch(`${API}/interact`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ image_id: imageId, action }),
  }).catch(() => {});
}

async function deleteImage(imageId) {
  if (!confirm('Delete this image?')) return;
  await fetch(`${API}/images/${imageId}`, { method: 'DELETE' });
  await doSearch('');
}

function closeModal() {
  modal.classList.add('hidden');
}
modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });

// ── Status polling ────────────────────────────────────────────
function updateSimStatus(data) {
  const round = data.round ?? 0;
  const reward = data.last_reward;
  roundBadge.textContent = `Round ${round}`;
  simRound.textContent = round;
  simReward.textContent = reward !== undefined && reward !== null ? reward.toFixed(4) : '—';
  rewardBadge.textContent = reward !== undefined && reward !== null ? `Reward: ${reward.toFixed(4)}` : '';
  if (data.weights) renderWeights(data.weights);
}

async function pollStatus() {
  try {
    const res = await fetch(`${API}/sim/status`);
    if (!res.ok) return;
    const state = await res.json();
    simRunning = state.running;
    updateToggleBtn();
    updateSimStatus(state);
    intervalSlider.value = state.interval || 5;
    intervalVal.textContent = `${state.interval || 5}s`;
    if (simRunning) await doSearch('');
  } catch (e) {}
}

async function refreshAll() {
  await pollStatus();
  await doSearch('');
}

// ── Preset ────────────────────────────────────────────────────
async function openPreset() {
  modalBody.innerHTML = '<p>Loading preset…</p>';
  modal.classList.remove('hidden');
  try {
    const res = await fetch(`${API}/sim/preset`);
    const data = await res.json();
    const figs = data.figures || [];
    const summary = data.summary || {};

    let html = '<h2 style="margin-bottom:14px">Experiment Results (1000 rounds)</h2>';

    if (Object.keys(summary).length) {
      html += `<div class="preset-summary"><table>
        <tr><td>Total rounds</td><td><b>${summary.total_rounds}</b></td></tr>
        <tr><td>Final reward</td><td><b>${(summary.final_reward||0).toFixed(4)}</b></td></tr>
        <tr><td>Final CTR</td><td><b>${(summary.final_ctr||0).toFixed(4)}</b></td></tr>
        <tr><td>Final LR</td><td><b>${(summary.final_lr||0).toFixed(4)}</b></td></tr>
        <tr><td>Final AWT</td><td><b>${(summary.final_awt||0).toFixed(4)}</b></td></tr>
      </table></div>`;
    }

    if (figs.length) {
      html += '<div class="preset-grid">';
      figs.forEach(f => {
        html += `<div><p style="font-size:12px;margin-bottom:4px;color:#64748b">${f.name}</p>
                 <img src="${f.path}" alt="${f.name}" /></div>`;
      });
      html += '</div>';
    } else {
      html += '<p style="color:#64748b;margin-top:12px">No figures found. Run the simulation first.</p>';
    }

    modalBody.innerHTML = html;
  } catch (e) {
    modalBody.innerHTML = '<p>Failed to load preset data.</p>';
  }
}

async function loadPresetState() {
  if (!confirm('Load round-1000 weights and image stats? This will overwrite current simulation state.')) return;
  try {
    const res = await fetch(`${API}/sim/load_preset`, { method: 'POST' });
    const data = await res.json();
    if (data.ok) {
      alert(`Loaded! Round=${data.round}. You can now continue running the simulation.`);
      await refreshAll();
    } else {
      alert('Load failed: ' + (data.detail || 'unknown error'));
    }
  } catch (e) {
    alert('Failed to load preset state.');
  }
}

// ── Debug log ─────────────────────────────────────────────────
async function pollDebugLogs() {
  try {
    const res = await fetch(`${API}/debug/logs?since=${logSince}`);
    if (!res.ok) return;
    const data = await res.json();
    const entries = data.entries || [];
    if (!entries.length) return;
    entries.forEach(e => {
      const line = document.createElement('div');
      line.className = 'debug-line';
      const ts = document.createElement('span');
      ts.className = 'debug-ts';
      ts.textContent = e.ts;
      const msg = document.createElement('span');
      msg.className = (e.msg.includes('❌') || e.msg.includes('ERROR')) ? 'debug-error'
                    : (e.msg.includes('✅') || e.msg.includes('DONE'))  ? 'debug-success' : '';
      msg.textContent = ' ' + e.msg;
      line.appendChild(ts);
      line.appendChild(msg);
      debugLogEl.appendChild(line);
    });
    logSince = entries[entries.length - 1].i + 1;
    debugLogEl.scrollTop = debugLogEl.scrollHeight;
  } catch (e) {}
}

// ── Boot ──────────────────────────────────────────────────────
checkExistingState();
setInterval(pollStatus, 2000);
setInterval(pollDebugLogs, 1500);
