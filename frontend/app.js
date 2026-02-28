const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const titleInput = document.getElementById('titleInput');
const fileInput = document.getElementById('fileInput');
const uploadBtn = document.getElementById('uploadBtn');
const textTitleInput = document.getElementById('textTitleInput');
const textArea = document.getElementById('textArea');
const textUploadBtn = document.getElementById('textUploadBtn');
const resultsEl = document.getElementById('results');
const modal = document.getElementById('modal');
const modalBody = document.getElementById('modalBody');
const closeModal = document.getElementById('closeModal');

const API_BASE = 'http://127.0.0.1:8000';
const mediaItems = [];

const MODEL_NAMES = {
  image: 'SigLIP',
  video: 'SigLIP',
  audio: 'CLAP',
  text: 'EmbeddingGemma',
};

function inferType(contentType) {
  if (!contentType) return 'text';
  if (contentType.startsWith('image/')) return 'image';
  if (contentType.startsWith('video/')) return 'video';
  if (contentType.startsWith('audio/')) return 'audio';
  return 'text';
}

function normalizeItem(serverItem) {
  return {
    id: serverItem.id,
    title: serverItem.title,
    type: serverItem.media_type || inferType(serverItem.content_type),
    url: serverItem.url,
    status: serverItem.status || 'UPLOADED',
    likes: Number(serverItem.likes || 0),
    views: Number(serverItem.views || 0),
    ctr: Number(serverItem.ctr || 0),
    avg_watch_time: Number(serverItem.avg_watch_time || 0),
    cosine_sim: serverItem.cosine_sim,
    final_score: serverItem.final_score,
    days_since_upload: serverItem.days_since_upload ?? null,
  };
}

function statusLabel(item) {
  if (item.status === 'INDEXED') return '';
  if (item.status === 'PROCESSING') return `⏳ Embedding with ${MODEL_NAMES[item.type] || 'model'}…`;
  if (item.status === 'UPLOADED') return '⏳ Queued…';
  if (item.status === 'FAILED') return '❌ Embedding failed';
  return '';
}

function createPreview(item, large = false) {
  const wrap = document.createElement('div');
  wrap.className = 'preview';

  if (item.type === 'image') {
    const img = document.createElement('img');
    img.src = item.url;
    img.alt = item.title;
    if (large) img.style.height = 'auto';
    wrap.appendChild(img);
  } else if (item.type === 'video') {
    const video = document.createElement('video');
    video.src = item.url;
    video.controls = true;
    if (!large) video.muted = true;
    wrap.appendChild(video);
  } else if (item.type === 'audio') {
    const audio = document.createElement('audio');
    audio.src = item.url;
    audio.controls = true;
    wrap.appendChild(audio);
  } else {
    const span = document.createElement('span');
    span.textContent = '📄 Text File';
    wrap.appendChild(span);
  }

  return wrap;
}

function openModal(item) {
  // Count a view
  fetch(`${API_BASE}/media/${item.id}/view`, { method: 'POST' }).catch(() => {});
  item.views += 1;

  modalBody.innerHTML = '';
  const title = document.createElement('h3');
  title.textContent = item.title;
  const preview = createPreview(item, true);
  preview.style.height = 'auto';
  modalBody.appendChild(title);
  modalBody.appendChild(preview);
  modal.classList.remove('hidden');
}

async function showInfoModal(item) {
  modalBody.innerHTML = '<p>Loading…</p>';
  modal.classList.remove('hidden');
  try {
    const res = await fetch(`${API_BASE}/media/${item.id}`);
    if (!res.ok) throw new Error('Not found');
    const data = await res.json();
    modalBody.innerHTML = '';
    const title = document.createElement('h3');
    title.textContent = `Info: ${data.title}`;
    modalBody.appendChild(title);
    const pre = document.createElement('pre');
    pre.className = 'info-json';
    pre.textContent = JSON.stringify(data, null, 2);
    modalBody.appendChild(pre);
  } catch (err) {
    modalBody.innerHTML = '<p>Failed to load info.</p>';
  }
}

function buildMetrics(item) {
  const el = document.createElement('div');
  el.className = 'metrics';

  const lines = [];
  if (item.cosine_sim !== null && item.cosine_sim !== undefined) {
    lines.push(`cos_sim: ${item.cosine_sim.toFixed(4)}`);
  }
  if (item.final_score !== null && item.final_score !== undefined) {
    lines.push(`score: ${item.final_score.toFixed(4)}`);
  }
  lines.push(`views: ${item.views}`);
  lines.push(`likes: ${item.likes}`);
  if (item.days_since_upload !== null && item.days_since_upload !== undefined) {
    lines.push(`days: ${item.days_since_upload}`);
  }
  lines.push(`ctr: ${item.ctr.toFixed(2)}`);
  lines.push(`avg_watch: ${item.avg_watch_time.toFixed(2)}`);

  el.textContent = lines.join(' │ ');
  return el;
}

function render(items) {
  resultsEl.innerHTML = '';

  if (!items.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = 'No content yet. Upload something and search.';
    resultsEl.appendChild(p);
    return;
  }

  items.forEach((item) => {
    const card = document.createElement('div');
    card.className = 'card';
    if (item.status !== 'INDEXED') card.classList.add('card-loading');

    const preview = createPreview(item);
    preview.addEventListener('click', () => openModal(item));

    // Status overlay
    const label = statusLabel(item);
    if (label) {
      const overlay = document.createElement('div');
      overlay.className = 'status-overlay';
      overlay.textContent = label;
      preview.style.position = 'relative';
      preview.appendChild(overlay);
    }

    const body = document.createElement('div');
    body.className = 'card-body';

    const title = document.createElement('p');
    title.className = 'title';
    title.textContent = item.title;

    // Metrics row
    const metrics = buildMetrics(item);

    const actions = document.createElement('div');
    actions.className = 'actions';

    const likeBtn = document.createElement('button');
    likeBtn.className = 'like-btn';
    likeBtn.textContent = `👍 ${item.likes}`;
    likeBtn.addEventListener('click', async () => {
      try {
        const res = await fetch(`${API_BASE}/media/${item.id}/like`, { method: 'POST' });
        if (res.ok) {
          const data = await res.json();
          item.likes = data.likes;
          likeBtn.textContent = `👍 ${item.likes}`;
        }
      } catch (e) { console.error(e); }
    });

    const infoBtn = document.createElement('button');
    infoBtn.className = 'info-btn';
    infoBtn.textContent = 'ℹ️';
    infoBtn.addEventListener('click', () => showInfoModal(item));

    const deleteBtn = document.createElement('button');
    deleteBtn.className = 'delete-btn';
    deleteBtn.textContent = '🗑️';
    deleteBtn.addEventListener('click', async () => {
      if (!confirm('Delete this item?')) return;
      try {
        const res = await fetch(`${API_BASE}/media/${item.id}`, { method: 'DELETE' });
        if (res.ok) await doSearch();
        else alert('Delete failed.');
      } catch (err) { console.error(err); alert('Delete failed.'); }
    });

    actions.appendChild(likeBtn);
    actions.appendChild(infoBtn);
    actions.appendChild(deleteBtn);

    body.appendChild(title);
    body.appendChild(metrics);
    body.appendChild(actions);

    card.appendChild(preview);
    card.appendChild(body);
    resultsEl.appendChild(card);
  });
}

// ── Polling ──────────────────────────────────────────────────
let pollTimer = null;
function startPolling() {
  stopPolling();
  pollTimer = setInterval(async () => {
    const pending = mediaItems.filter(i => i.status !== 'INDEXED' && i.status !== 'FAILED');
    if (!pending.length) { stopPolling(); return; }
    await doSearch(true);
  }, 3000);
}
function stopPolling() {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
}

// ── Search ───────────────────────────────────────────────────
async function doSearch(silent = false) {
  const q = searchInput.value.trim();
  const url = `${API_BASE}/search?q=${encodeURIComponent(q)}`;

  try {
    const res = await fetch(url);
    if (!res.ok) throw new Error('Search failed');

    const data = await res.json();
    mediaItems.length = 0;
    mediaItems.push(...(data.items || []).map(normalizeItem));
    render(mediaItems);

    if (mediaItems.some(i => i.status !== 'INDEXED' && i.status !== 'FAILED')) {
      startPolling();
    }
  } catch (err) {
    if (!silent) {
      console.error(err);
      alert('Cannot connect to backend. Please start the API first.');
    }
  }
}

// ── File upload ──────────────────────────────────────────────
uploadBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  if (!file) { alert('Please select a file first.'); return; }

  const formData = new FormData();
  formData.append('file', file);
  formData.append('title', titleInput.value.trim());

  try {
    const res = await fetch(`${API_BASE}/media/upload`, { method: 'POST', body: formData });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    mediaItems.unshift(normalizeItem(data.item));
    render(mediaItems);
    startPolling();
    titleInput.value = '';
    fileInput.value = '';
  } catch (err) {
    console.error(err);
    alert('Upload failed.');
  }
});

// ── Pure text upload ─────────────────────────────────────────
textUploadBtn.addEventListener('click', async () => {
  const text = textArea.value.trim();
  if (!text) { alert('Please enter some text first.'); return; }

  try {
    const res = await fetch(`${API_BASE}/media/upload/text`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, title: textTitleInput.value.trim() }),
    });
    if (!res.ok) throw new Error('Upload failed');
    const data = await res.json();
    mediaItems.unshift(normalizeItem(data.item));
    render(mediaItems);
    startPolling();
    textTitleInput.value = '';
    textArea.value = '';
  } catch (err) {
    console.error(err);
    alert('Text upload failed.');
  }
});

// ── Event listeners ──────────────────────────────────────────
searchBtn.addEventListener('click', () => doSearch());
searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSearch(); });
closeModal.addEventListener('click', () => modal.classList.add('hidden'));
modal.addEventListener('click', (e) => { if (e.target === modal) modal.classList.add('hidden'); });

doSearch();

// ── Debug Log Panel ──────────────────────────────────────────
const debugLogEl = document.getElementById('debugLog');
const clearLogBtn = document.getElementById('clearLogBtn');
let logSince = 0;

async function pollDebugLogs() {
  try {
    const res = await fetch(`${API_BASE}/debug/logs?since=${logSince}`);
    if (!res.ok) return;
    const data = await res.json();
    const entries = data.entries || [];
    if (entries.length === 0) return;

    entries.forEach((e) => {
      const line = document.createElement('div');
      line.className = 'debug-line';
      const ts = document.createElement('span');
      ts.className = 'debug-ts';
      ts.textContent = e.ts;
      const msg = document.createElement('span');
      msg.className = 'debug-msg';
      msg.textContent = e.msg;
      if (e.msg.includes('ERROR') || e.msg.includes('❌') || e.msg.includes('FAILED')) {
        msg.classList.add('debug-error');
      } else if (e.msg.includes('✅') || e.msg.includes('DONE')) {
        msg.classList.add('debug-success');
      }
      line.appendChild(ts);
      line.appendChild(msg);
      debugLogEl.appendChild(line);
    });
    logSince = entries[entries.length - 1].i + 1;
    debugLogEl.scrollTop = debugLogEl.scrollHeight;
  } catch (e) { /* ignore */ }
}

clearLogBtn.addEventListener('click', () => { debugLogEl.innerHTML = ''; });
setInterval(pollDebugLogs, 1000);
pollDebugLogs();
