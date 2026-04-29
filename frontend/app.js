/* ─────────────────────────────────────────────────────────────
   Image Ranking Simulation — Frontend Logic
───────────────────────────────────────────────────────────── */

const API = 'http://127.0.0.1:8000';

// ── DOM refs ──────────────────────────────────────────────────
const imageGrid    = document.getElementById('imageGrid');
const imageCount   = document.getElementById('imageCount');
const searchInput  = document.getElementById('searchInput');
const titleInput   = document.getElementById('titleInput');
const fileInput    = document.getElementById('fileInput');
const roundBadge   = document.getElementById('roundBadge');
const rewardBadge  = document.getElementById('rewardBadge');
const simRound     = document.getElementById('simRound');
const simReward    = document.getElementById('simReward');
const runBtn       = document.getElementById('runBtn');
const intervalSlider = document.getElementById('intervalSlider');
const intervalVal  = document.getElementById('intervalVal');
const weightBars   = document.getElementById('weightBars');
const countdownEl  = document.getElementById('countdown');
const modal        = document.getElementById('modal');
const modalBody    = document.getElementById('modalBody');
const debugLogEl   = document.getElementById('debugLog');

const WEIGHT_LABELS = ['cos', 'fresh', 'lr', 'ctr', 'awt', 'social'];
let currentItems = [];
let simRunning = false;
let logSince = 0;
let lastKnownRound = -1;
let _intervalSecs = 5;
let _countdownSecs = 0;
let _countdownTimer = null;

// ── Data Pool ─────────────────────────────────────────────────
async function initPool(mode) {
  try {
    await fetch(`${API}/sim/init`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ mode }),
    });
    lastKnownRound = -1;
    await doSearch('');
    await pollStatus();
  } catch (e) {
    alert('Cannot reach backend. Make sure uvicorn is running on port 8000.');
  }
}

async function loadWithState() {
  try {
    await fetch(`${API}/sim/init`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ mode: 'dataset' }),
    });
    const res = await fetch(`${API}/sim/load_preset`, { method: 'POST' });
    const data = await res.json();
    if (!data.ok) alert('Preset load failed: ' + (data.detail || 'unknown error'));
    lastKnownRound = -1;
    await doSearch('');
    await pollStatus();
  } catch (e) {
    alert('Cannot reach backend. Make sure uvicorn is running on port 8000.');
  }
}

// ── Simulation control ────────────────────────────────────────
async function toggleSim() {
  const res = await fetch(`${API}/sim/toggle`, { method: 'POST' });
  const data = await res.json();
  simRunning = data.running;
  updateRunBtn();
  if (simRunning) resetCountdown();
}

async function resetSim() {
  if (!confirm('Reset simulation to round 0?')) return;
  await fetch(`${API}/sim/reset`, { method: 'POST' });
  lastKnownRound = -1;
  clearInterval(_countdownTimer);
  countdownEl.textContent = '';
  await refreshAll();
}

function updateRunBtn() {
  if (simRunning) {
    runBtn.textContent = '⏸ Pause';
    runBtn.className = 'btn-toggle btn-stop';
  } else {
    runBtn.textContent = '▶ Run';
    runBtn.className = 'btn-toggle btn-start';
    clearInterval(_countdownTimer);
    countdownEl.textContent = '';
  }
}

function updateInterval(val) {
  _intervalSecs = Number(val);
  intervalVal.textContent = `${val}s`;
  fetch(`${API}/sim/set_interval`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({ interval: _intervalSecs }),
  }).catch(() => {});
}

// ── Countdown ─────────────────────────────────────────────────
function resetCountdown() {
  clearInterval(_countdownTimer);
  _countdownSecs = _intervalSecs;
  countdownEl.textContent = `⟳ Next round in ${_countdownSecs}s`;
  _countdownTimer = setInterval(() => {
    if (!simRunning) { countdownEl.textContent = ''; clearInterval(_countdownTimer); return; }
    _countdownSecs = Math.max(0, _countdownSecs - 1);
    countdownEl.textContent = `⟳ Next round in ${_countdownSecs}s`;
  }, 1000);
}

// ── Behaviour params ──────────────────────────────────────────
function setParams() {
  fetch(`${API}/sim/set_params`, {
    method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({
      alpha_q:    parseFloat(document.getElementById('alphaqSlider').value),
      gamma_q:    parseFloat(document.getElementById('gammaqSlider').value),
      delta_q:    parseFloat(document.getElementById('deltaqSlider').value),
      batch_size: parseInt(document.getElementById('batchSlider').value),
      user_sigma: parseFloat(document.getElementById('sigmaSlider').value),
    }),
  }).catch(() => {});
}

// ── Weight bars (in-place update) ─────────────────────────────
function renderWeights(weights) {
  if (!weights || !weights.length) return;
  const maxW = Math.max(...weights, 0.01);

  if (weightBars.children.length !== weights.length) {
    weightBars.innerHTML = '';
    weights.forEach((w, i) => {
      const row = document.createElement('div');
      row.className = 'weight-row';
      row.innerHTML = `
        <span class="weight-label">${WEIGHT_LABELS[i]}</span>
        <div class="weight-track"><div class="weight-fill" style="width:${(w/maxW)*100}%"></div></div>
        <span class="weight-val">${w.toFixed(3)}</span>`;
      weightBars.appendChild(row);
    });
    return;
  }
  weights.forEach((w, i) => {
    const row = weightBars.children[i];
    if (!row) return;
    row.querySelector('.weight-fill').style.width = `${(w/maxW)*100}%`;
    row.querySelector('.weight-val').textContent = w.toFixed(3);
  });
}

// ── Sim status display ───────────────────────────────────────
function updateSimStatus(data) {
  const round = data.round ?? 0;
  const reward = data.last_reward;
  roundBadge.textContent = `Round ${round}`;
  simRound.textContent = round;
  simReward.textContent = reward !== undefined && reward !== null ? reward.toFixed(4) : '—';
  rewardBadge.textContent = reward !== undefined && reward !== null ? `Reward: ${reward.toFixed(4)}` : '';
  if (data.weights) renderWeights(data.weights);
}

// ── Search ────────────────────────────────────────────────────
async function doSearch(query) {
  const q = (query !== undefined ? query : searchInput.value).trim();
  // Highlight active chip
  document.querySelectorAll('.chip').forEach(c => {
    c.classList.toggle('active', c.dataset.q === q);
  });
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

function quickSearch(q) {
  searchInput.value = q;
  doSearch(q);
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
  const qv = document.getElementById('qualityInput').value;
  if (qv !== '') fd.append('quality', Math.min(1, Math.max(0, parseFloat(qv))));

  try {
    const res = await fetch(`${API}/images/upload`, { method: 'POST', body: fd });
    if (!res.ok) throw new Error('Upload failed');
    titleInput.value = '';
    fileInput.value = '';
    document.getElementById('qualityInput').value = '';
    await doSearch('');
  } catch (e) {
    alert('Upload failed. Check backend.');
  }
}

// ── Image Grid ────────────────────────────────────────────────
function qualityLabel(q) {
  const v = parseFloat(q);
  if (isNaN(v)) return null;
  if (v >= 0.75) return { text: 'Very High', cls: 'qtag-veryhigh' };
  if (v >= 0.5)  return { text: 'High',      cls: 'qtag-high' };
  if (v >= 0.3)  return { text: 'Medium',    cls: 'qtag-medium' };
  return                { text: 'Low',       cls: 'qtag-low' };
}

function metaText(item) {
  const ctr = (item.ctr || 0).toFixed(3);
  const lr  = (item.lr  || 0).toFixed(3);
  const awt = (item.awt || 0).toFixed(2);
  const fresh = (item.freshness || 0).toFixed(2);
  return `CTR:${ctr} │ LR:${lr} │ AWT:${awt} │ fresh:${fresh} │ Q:${(item.quality||0).toFixed(2)}`;
}

function buildCard(item, idx) {
  const card = document.createElement('div');
  card.className = 'card' + (item.status === 'PROCESSING' ? ' card-loading' : '');
  card.dataset.id = item.id;

  const rank = document.createElement('div');
  rank.className = 'rank-badge';
  rank.textContent = `#${item.rank || idx + 1}`;

  const thumb = document.createElement('div');
  thumb.className = 'thumb';
  if (item.url) {
    const img = document.createElement('img');
    img.src = item.url;
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

  const ql = qualityLabel(item.quality);
  const qTag = ql ? document.createElement('span') : null;
  if (qTag) { qTag.className = `qtag ${ql.cls}`; qTag.textContent = ql.text; }

  const titleEl = document.createElement('div');
  titleEl.className = 'card-title';
  titleEl.textContent = item.title || item.id;

  const meta = document.createElement('div');
  meta.className = 'card-meta';
  meta.textContent = metaText(item);

  // View + like counters
  const counters = document.createElement('div');
  counters.className = 'card-counters';
  counters.innerHTML =
    `<span class="counter-views">👁 <span class="views-count">${item.total_views || 0}</span></span>` +
    `<span class="counter-likes">❤️ <span class="likes-count">${item.total_likes || 0}</span></span>`;

  const actions = document.createElement('div');
  actions.className = 'card-actions';

  const delBtn = document.createElement('button');
  delBtn.className = 'btn-danger';
  delBtn.textContent = '🗑 Delete';
  delBtn.addEventListener('click', () => deleteImage(item.id));

  actions.appendChild(delBtn);

  body.appendChild(catTag);
  if (qTag) body.appendChild(qTag);
  body.appendChild(titleEl);
  body.appendChild(meta);
  body.appendChild(counters);
  body.appendChild(actions);

  card.appendChild(rank);
  card.appendChild(thumb);
  card.appendChild(body);
  return card;
}

let _lastExaminedIds = new Set();

function renderGrid(items) {
  const top10 = items.slice(0, 10);
  imageCount.textContent = top10.length;

  if (!top10.length) {
    imageGrid.innerHTML = '';
    const p = document.createElement('p');
    p.style.cssText = 'color:#64748b;padding:20px;';
    p.textContent = 'No images yet. Upload something or load the dataset.';
    imageGrid.appendChild(p);
    _lastExaminedIds = new Set();
    return;
  }

  // Remove any placeholder text (e.g. "No images yet" paragraph)
  [...imageGrid.children].forEach(el => { if (!el.dataset.id) el.remove(); });

  // Collect existing cards by id
  const existingCards = {};
  imageGrid.querySelectorAll('.card[data-id]').forEach(c => { existingCards[c.dataset.id] = c; });

  const newIds = new Set(top10.map(i => i.id));

  // Remove cards no longer in top10
  Object.keys(existingCards).forEach(id => {
    if (!newIds.has(id)) { existingCards[id].remove(); delete existingCards[id]; }
  });

  // Update existing cards in-place (NO appendChild → no reflow)
  top10.forEach((item, idx) => {
    const card = existingCards[item.id];
    if (card) {
      card.querySelector('.rank-badge').textContent = `#${item.rank || idx + 1}`;
      card.querySelector('.card-meta').textContent = metaText(item);
      const vc = card.querySelector('.views-count');
      const lc = card.querySelector('.likes-count');
      if (vc) vc.textContent = item.total_views || 0;
      if (lc) lc.textContent = item.total_likes || 0;
    } else {
      existingCards[item.id] = buildCard(item, idx);
    }
  });

  // Reorder DOM only when the order has actually changed
  const currentOrder = [...imageGrid.querySelectorAll('.card[data-id]')].map(c => c.dataset.id);
  const desiredOrder = top10.map(i => i.id);
  const orderChanged = currentOrder.length !== desiredOrder.length ||
    currentOrder.some((id, i) => id !== desiredOrder[i]);
  if (orderChanged) {
    desiredOrder.forEach(id => imageGrid.appendChild(existingCards[id]));
  }

  // Mark top-10 as examined once per new appearance
  top10.filter(i => !_lastExaminedIds.has(i.id)).forEach(i => interact(i.id, 'examine'));
  _lastExaminedIds = newIds;

  if (top10.some(i => i.status === 'PROCESSING')) setTimeout(() => doSearch(''), 3000);
}

// ── Interactions ──────────────────────────────────────────────
async function viewImage(item) {
  await interact(item.id, 'view');

  // Build modal with like button inside
  modalBody.innerHTML = '';

  const titleEl = document.createElement('h3');
  titleEl.style.marginBottom = '12px';
  titleEl.textContent = item.title;

  const imgEl = document.createElement('img');
  imgEl.src = item.url || '';
  imgEl.style.cssText = 'max-width:100%;border-radius:8px;display:block;margin:0 auto;';

  const metaEl = document.createElement('div');
  metaEl.style.cssText = 'margin-top:10px;font-size:12px;color:#64748b;';
  metaEl.textContent = `Category: ${item.category} │ CTR: ${(item.ctr||0).toFixed(3)} │ LR: ${(item.lr||0).toFixed(3)} │ AWT: ${(item.awt||0).toFixed(2)}`;

  const actionsEl = document.createElement('div');
  actionsEl.style.cssText = 'margin-top:14px;display:flex;align-items:center;gap:14px;';

  const likeBtn = document.createElement('button');
  likeBtn.className = 'btn-primary';
  likeBtn.style.fontSize = '15px';
  const likesCount = (item.total_likes || 0);
  likeBtn.innerHTML = `👍 Like &nbsp;<span class="modal-likes-count" style="font-weight:700">${likesCount}</span>`;
  likeBtn.addEventListener('click', () => {
    const span = likeBtn.querySelector('.modal-likes-count');
    if (span) span.textContent = parseInt(span.textContent || 0) + 1;
    // Also update card counter
    const card = imageGrid.querySelector(`.card[data-id="${item.id}"]`);
    if (card) {
      const lc = card.querySelector('.likes-count');
      const vc = card.querySelector('.views-count');
      if (lc) lc.textContent = parseInt(lc.textContent || 0) + 1;
      if (vc) vc.textContent = parseInt(vc.textContent || 0) + 1;
    }
    interact(item.id, 'like');
  });

  const viewsEl = document.createElement('span');
  viewsEl.style.cssText = 'font-size:13px;color:#64748b;';
  viewsEl.innerHTML = `👁 <span class="modal-views-count">${item.total_views || 0}</span> views`;

  actionsEl.appendChild(likeBtn);
  actionsEl.appendChild(viewsEl);

  modalBody.appendChild(titleEl);
  modalBody.appendChild(imgEl);
  modalBody.appendChild(metaEl);
  modalBody.appendChild(actionsEl);

  modal.classList.remove('hidden');
  // Refresh card stats in background
  doSearch(searchInput.value.trim());
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
async function pollStatus() {
  try {
    const res = await fetch(`${API}/sim/status`);
    if (!res.ok) return;
    const state = await res.json();
    const wasRunning = simRunning;
    simRunning = state.running;
    updateRunBtn();
    updateSimStatus(state);
    const storedInterval = state.interval || 5;
    _intervalSecs = storedInterval;
    intervalSlider.value = storedInterval;
    intervalVal.textContent = `${storedInterval}s`;
    const newRound = state.round ?? 0;
    if (newRound !== lastKnownRound) {
      lastKnownRound = newRound;
      if (simRunning) resetCountdown();
      await doSearch(searchInput.value.trim());
    } else if (simRunning && !wasRunning) {
      resetCountdown();
    }
  } catch (e) {}
}

async function refreshAll() {
  await pollStatus();
  if (!currentItems.length) await doSearch('');
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
                 <img src="${API}${f.path}" alt="${f.name}" /></div>`;
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
refreshAll();
setInterval(pollStatus, 2000);
setInterval(pollDebugLogs, 1500);
