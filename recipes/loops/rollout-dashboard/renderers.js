/* ==========================================================================
   training-explorer · renderers.js
   --------------------------------------------------------------------------
   Two parts:
   1. Main-page renderers (header, stat strip, rollouts section, metrics, config)
   2. Format-specific rollout renderers — one function per format, named exactly
      after the format. These are loaded by detail.html (the iframe) via a
      separate path; they're NOT used by the main index.html.

   See CUSTOMIZATION.md for the "if user asks for X, edit Y" map.
   ========================================================================== */

// ============================================================================
// PART 1 — Main page renderers (index.html only)
// ============================================================================

function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function fmt(n, places = 3) {
  if (n == null || typeof n !== 'number') return '—';
  if (Math.abs(n) >= 1000 || Math.abs(n) < 0.001 && n !== 0) {
    return n.toExponential(2);
  }
  return n.toFixed(places);
}

function scoreClass(v) {
  if (v == null) return 'unknown';
  if (v === 0) return 'zero';
  if (v < 0.25) return 's0';
  if (v < 0.50) return 's1';
  if (v < 0.85) return 's2';
  return 's3';
}

// ---- header + stat strip ----------------------------------------------------

function renderHeader() {
  document.getElementById('run-id').textContent = RUN.run_id || '(unnamed run)';
  const meta = [];
  if (RUN.last_updated) meta.push(`last update ${RUN.last_updated}`);
  if (RUN.started_at)   meta.push(`started ${RUN.started_at}`);
  document.getElementById('run-meta').textContent = meta.join(' · ');
}

function renderStatStrip() {
  const strip = document.getElementById('stat-strip');
  const stats = [];
  const totalRollouts = (RUN.rollout_sources || []).reduce((s, src) => s + (src.n_rollouts || 0), 0);
  if (totalRollouts) stats.push(['rollouts', totalRollouts.toLocaleString()]);

  // Score summary across the active source, if available
  const src = (RUN.rollout_sources || [])[activeSourceIdx];
  if (src && src._records && src.fields.score) {
    const scores = src._records.map(r => r[src.fields.score]).filter(v => v != null);
    if (scores.length) {
      const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
      const std  = Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length);
      stats.push([`${src.fields.score}`, `${fmt(mean)} ± ${fmt(std)}`]);
    }
  }

  if (RUN.metrics?.series?.length) stats.push(['metric series', RUN.metrics.series.length]);
  if (RUN.config) stats.push(['config keys', Object.keys(RUN.config.values || {}).length]);

  strip.innerHTML = stats.map((s, i) => `
    ${i > 0 ? '<div class="stat-divider"></div>' : ''}
    <div class="stat">
      <div class="stat-label">${esc(s[0])}</div>
      <div class="stat-value">${esc(s[1])}</div>
    </div>
  `).join('');
}

// ---- rollouts section ------------------------------------------------------

function renderRollouts() {
  const sources = RUN.rollout_sources || [];
  document.getElementById('rollouts-meta').textContent =
    sources.length ? `${sources.length} source${sources.length > 1 ? 's' : ''}` : 'none detected';

  // Source tabs
  const tabsEl = document.getElementById('source-tabs');
  if (sources.length > 1) {
    tabsEl.innerHTML = sources.map((s, i) => `
      <div class="source-tab ${i === activeSourceIdx ? 'active' : ''}" onclick="selectSource(${i})">
        ${esc(s.name)}<span class="tab-count">${(s.n_rollouts || 0).toLocaleString()}</span>
      </div>
    `).join('');
  } else {
    tabsEl.innerHTML = '';
  }

  const src = sources[activeSourceIdx];
  if (!src) {
    document.getElementById('rollout-display').innerHTML =
      '<div style="padding: 24px; color: var(--fg-muted);">No rollout sources detected. Edit run.json to add one manually.</div>';
    document.getElementById('filter-bar').innerHTML = '';
    document.getElementById('pagination').innerHTML = '';
    return;
  }

  renderFilters(src);
  renderRolloutDisplay(src);
}

function selectSource(idx) {
  activeSourceIdx = idx;
  activePage = 0;
  activeFilters = {step: null, group: null};
  renderRollouts();
  renderStatStrip();
}

function renderFilters(src) {
  const bar = document.getElementById('filter-bar');
  const parts = [];
  const recs = src._records || [];

  if (src.fields.step) {
    const steps = [...new Set(recs.map(r => r[src.fields.step]).filter(v => v != null))].sort((a, b) => a - b);
    if (steps.length > 1 && steps.length < 200) {
      parts.push(`
        <label>${esc(src.fields.step)}:</label>
        <select onchange="setFilter('step', this.value)">
          <option value="">all</option>
          ${steps.map(s => `<option value="${s}" ${activeFilters.step == s ? 'selected' : ''}>${s}</option>`).join('')}
        </select>
      `);
    }
  }

  if (src.fields.group) {
    const groups = [...new Set(recs.map(r => r[src.fields.group]).filter(v => v != null))];
    if (groups.length > 1 && groups.length < 100) {
      parts.push(`
        <label>${esc(src.fields.group)}:</label>
        <select onchange="setFilter('group', this.value)">
          <option value="">all</option>
          ${groups.map(g => `<option value="${esc(g)}" ${activeFilters.group == g ? 'selected' : ''}>${esc(g)}</option>`).join('')}
        </select>
      `);
    }
  }

  bar.innerHTML = parts.join('');
}

function setFilter(field, value) {
  activeFilters[field] = value === '' ? null : value;
  activePage = 0;
  renderRollouts();
}

function filteredRecords(src) {
  let recs = src._records || [];
  if (activeFilters.step != null && src.fields.step) {
    recs = recs.filter(r => String(r[src.fields.step]) === String(activeFilters.step));
  }
  if (activeFilters.group != null && src.fields.group) {
    recs = recs.filter(r => String(r[src.fields.group]) === String(activeFilters.group));
  }
  return recs;
}

function renderRolloutDisplay(src) {
  const display = document.getElementById('rollout-display');
  const recs = filteredRecords(src);

  // Decide layout: grouped grid if a group field exists AND score field exists.
  // Otherwise flat list. (Group without score is fine too, but the dot grid loses
  // some of its value without colors.)
  if (src.fields.group && recs.length > 4) {
    display.innerHTML = renderGroupGrid(src, recs);
    document.getElementById('pagination').innerHTML = '';
  } else {
    const paged = recs.slice(activePage * PAGE_SIZE, (activePage + 1) * PAGE_SIZE);
    display.innerHTML = renderFlatList(src, paged);
    renderPagination(recs.length);
  }
}

function renderGroupGrid(src, recs) {
  const groupKey = src.fields.group;
  const scoreKey = src.fields.score;
  const idKey    = src.fields.id;

  // Group records
  const groups = new Map();
  for (const r of recs) {
    const k = r[groupKey];
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }

  const cards = Array.from(groups.entries()).map(([label, items]) => {
    const scores = scoreKey ? items.map(r => r[scoreKey]).filter(v => v != null) : [];
    const mean = scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
    const std  = scores.length ? Math.sqrt(scores.reduce((a, b) => a + (b - mean) ** 2, 0) / scores.length) : null;

    const dotCap = 24;
    const dots = items.slice(0, dotCap).map(r => {
      const v = scoreKey ? r[scoreKey] : null;
      // Click target: synth id is always safe; fall back to qualified or natural id.
      const id = r._synth_id || r._qualified_id || (idKey && r[idKey]) || '';
      const label = v != null ? fmt(v, 2).replace(/^0\./, '.').replace(/^-0\./, '-.') : '';
      return `<div class="rollout-dot ${scoreClass(v)}" onclick="openPanel(${activeSourceIdx}, '${esc(id)}', '${esc(String(label || id))}')" title="${esc(String(id))}">${esc(label)}</div>`;
    }).join('');
    const more = items.length > dotCap ? `<span class="rollout-dot-more">+${items.length - dotCap}</span>` : '';

    const stats = scores.length
      ? `${items.length} rollouts · μ ${fmt(mean)} · σ ${fmt(std)}`
      : `${items.length} rollouts`;

    return `
      <div class="group-card">
        <div class="group-label">${esc(String(label))}</div>
        <div class="group-stats">${stats}</div>
        <div class="group-dots">${dots}${more}</div>
      </div>
    `;
  }).join('');

  return `<div class="group-grid">${cards}</div>`;
}

function renderFlatList(src, recs) {
  const idKey    = src.fields.id;
  const scoreKey = src.fields.score;
  const stepKey  = src.fields.step;

  const header = `
    <div class="rollout-row" style="background: var(--bg-hover); font-size: 11px; color: var(--fg-muted); text-transform: uppercase; letter-spacing: 0.04em; cursor: default;">
      <div>${scoreKey ? esc(scoreKey) : ''}</div>
      <div>${stepKey ? esc(stepKey) : ''}</div>
      <div>id</div>
      <div>preview</div>
    </div>
  `;

  const rows = recs.map(r => {
    const id = r._synth_id || r._qualified_id || (idKey && r[idKey]) || '';
    const score = scoreKey ? r[scoreKey] : null;
    const step = stepKey ? r[stepKey] : '';
    const preview = r._preview || '';
    return `
      <div class="rollout-row" onclick="openPanel(${activeSourceIdx}, '${esc(id)}', '${esc(String(id))}')">
        <div class="score-cell"><span class="rollout-dot ${scoreClass(score)}" style="width: auto; padding: 2px 6px;">${score != null ? fmt(score, 2) : '—'}</span></div>
        <div class="step-cell">${esc(String(step))}</div>
        <div class="id-cell">${esc(String(id))}</div>
        <div class="preview-cell">${esc(preview)}</div>
      </div>
    `;
  }).join('');

  return `<div class="rollout-list">${header}${rows}</div>`;
}

function renderPagination(total) {
  const el = document.getElementById('pagination');
  const totalPages = Math.ceil(total / PAGE_SIZE);
  if (totalPages <= 1) { el.innerHTML = ''; return; }
  const start = activePage * PAGE_SIZE + 1;
  const end = Math.min((activePage + 1) * PAGE_SIZE, total);
  el.innerHTML = `
    <button onclick="setPage(0)" ${activePage === 0 ? 'disabled' : ''}>« first</button>
    <button onclick="setPage(${activePage - 1})" ${activePage === 0 ? 'disabled' : ''}>‹ prev</button>
    <span>${start.toLocaleString()}–${end.toLocaleString()} of ${total.toLocaleString()}</span>
    <button onclick="setPage(${activePage + 1})" ${activePage >= totalPages - 1 ? 'disabled' : ''}>next ›</button>
    <button onclick="setPage(${totalPages - 1})" ${activePage >= totalPages - 1 ? 'disabled' : ''}>last »</button>
  `;
}

function setPage(p) {
  activePage = p;
  renderRollouts();
  window.scrollTo({top: document.querySelector('.rollouts-section').offsetTop, behavior: 'smooth'});
}

// ---- metrics section -------------------------------------------------------

const _charts = new Map();

function renderMetrics() {
  const m = RUN.metrics;
  if (!m || !m.series?.length) {
    document.getElementById('metrics-section').style.display = 'none';
    return;
  }
  document.getElementById('metrics-meta').textContent =
    `${m.series.length} series · ${m._records?.length || 0} steps`;

  const grid = document.getElementById('chart-grid');
  // Build or update cards
  for (const series of m.series) {
    let card = document.getElementById(`chart-card-${cssId(series)}`);
    if (!card) {
      card = document.createElement('div');
      card.className = 'chart-card';
      card.id = `chart-card-${cssId(series)}`;
      card.innerHTML = `
        <div class="chart-title" title="${esc(series)}">${esc(series)}</div>
        <div class="chart-wrap"><canvas></canvas></div>
      `;
      grid.appendChild(card);
    }
    updateChart(series);
  }
}

function cssId(s) { return s.replace(/[^a-zA-Z0-9]/g, '_'); }

function updateChart(series) {
  const m = RUN.metrics;
  const xAxis = m.x_axis;
  const recs = m._records || [];
  const xs = recs.map(r => r[xAxis]);
  const ys = recs.map(r => r[series]);

  const cardId = `chart-card-${cssId(series)}`;
  const canvas = document.querySelector(`#${cardId} canvas`);
  if (!canvas) return;

  if (_charts.has(series)) {
    const ch = _charts.get(series);
    ch.data.labels = xs;
    ch.data.datasets[0].data = ys;
    ch.update('none');
    return;
  }

  const ctx = canvas.getContext('2d');
  const ch = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xs,
      datasets: [{
        data: ys,
        borderColor: '#16a34a',
        backgroundColor: 'rgba(22,163,74,0.08)',
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 3,
        tension: 0.2,
        fill: true,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { display: true, ticks: { font: { size: 9 }, maxTicksLimit: 6 } },
        y: { display: true, ticks: { font: { size: 9 } } },
      },
    },
  });
  _charts.set(series, ch);
}

// ---- config section --------------------------------------------------------

function renderConfig() {
  const c = RUN.config;
  if (!c || !c.values) {
    document.getElementById('config-section').style.display = 'none';
    return;
  }
  document.getElementById('config-meta').textContent =
    `${Object.keys(c.values).length} keys${c.file ? ' · ' + c.file : ''}`;
  const t = document.getElementById('config-table');
  t.innerHTML = Object.entries(c.values).map(([k, v]) => `
    <div class="key">${esc(k)}</div>
    <div class="value">${esc(typeof v === 'object' ? JSON.stringify(v) : String(v))}</div>
  `).join('');
}

// ---- detection footer ------------------------------------------------------

function renderDetectionFooter() {
  const parts = [];
  const sources = RUN.rollout_sources || [];
  parts.push('<div class="det-list">');
  for (const s of sources) {
    parts.push(`<div>${esc(s.name)}</div><div>${esc(s.format)} · ${esc(s.file)} · ${(s.n_rollouts || 0).toLocaleString()} rollouts · fields: ${esc(JSON.stringify(s.fields))}</div>`);
  }
  if (RUN.metrics)  parts.push(`<div>metrics</div><div>${esc(RUN.metrics.file)} · x=${esc(RUN.metrics.x_axis)} · ${(RUN.metrics.series || []).length} series</div>`);
  if (RUN.config)   parts.push(`<div>config</div><div>${esc(RUN.config.file)} · ${Object.keys(RUN.config.values || {}).length} keys</div>`);
  for (const ig of (RUN.ignored || []))
    parts.push(`<div style="color: var(--fg-subtle)">ignored</div><div style="color: var(--fg-subtle)">${esc(ig.path)} — ${esc(ig.reason)}</div>`);
  if (RUN.tfevents?.length)
    parts.push(`<div>tfevents</div><div>${RUN.tfevents.length} file(s) — use <code>tensorboard --logdir .</code> to view</div>`);
  parts.push('</div>');
  document.getElementById('detection-body').innerHTML = parts.join('');
}

// ============================================================================
// PART 2 — Detail-view renderers (used by detail.html only)
// One function per format. Names match the `format` field in run.json exactly.
// If a user asks to change how a format is rendered, they want one of these.
// ============================================================================

const SPEAKER_VARS = ['--speaker-1','--speaker-2','--speaker-3','--speaker-4','--speaker-5','--speaker-6','--speaker-7','--speaker-8'];
function _speakerColors(speakers) {
  const cs = getComputedStyle(document.documentElement);
  const map = {};
  speakers.forEach((s, i) => { map[s] = cs.getPropertyValue(SPEAKER_VARS[i % SPEAKER_VARS.length]).trim(); });
  return map;
}

function _turnTimelineHtml(turns, header = '') {
  if (!turns.length) return '<div class="raw-text">No turns extracted.</div>';
  const speakers = [...new Set(turns.map(t => t.speaker))];
  const colors = _speakerColors(speakers);

  const nav = turns.map((t, i) => `
    <div class="turn-nav-item" data-idx="${i}" onclick="showTurn(${i})">
      <span class="turn-num">${t.turn ?? (i + 1)}</span>
      <span class="turn-preview">
        <span class="speaker" style="color: ${colors[t.speaker]}">${esc(t.speaker)}</span>${esc(String(t.text).slice(0, 80))}
      </span>
    </div>
  `).join('');

  const content = turns.map((t, i) => `
    <div class="turn-card ${t._kind ? 'event-' + t._kind : ''} ${t._has_error ? 'has-error' : ''}" id="turn-${i}">
      <div class="turn-header">
        <span class="speaker" style="color: ${colors[t.speaker]}">${esc(t.speaker)}</span>
        <span class="turn-num">turn ${t.turn ?? (i + 1)}</span>
        ${t._meta ? `<span class="turn-meta">${esc(t._meta)}</span>` : ''}
      </div>
      <div class="turn-body">${esc(t.text)}</div>
    </div>
  `).join('');

  return `
    ${header}
    <div class="two-pane">
      <div class="turn-nav">${nav}</div>
      <div class="turn-content">${content}</div>
    </div>
    <script>
      function showTurn(i) {
        document.querySelectorAll('.turn-nav-item').forEach(n => n.classList.toggle('active', +n.dataset.idx === i));
        const card = document.getElementById('turn-' + i);
        if (card) card.scrollIntoView({behavior: 'smooth', block: 'start'});
      }
      // Activate first turn on load
      if (document.querySelector('.turn-nav-item')) showTurn(0);
    </script>
  `;
}

// --- format: group_turns -----------------------------------------------------
// Used when a logtree group contains interleaved turns from N parallel
// trajectories. The server tags each message with {role, content, turn?}.
//
// Trajectories are reconstructed from the interleaved flat list using greedy
// slot assignment: Player+Answerer pairs are always consecutive in the stream,
// so for each pair at Turn N we assign it to the slot whose last completed
// turn was N-1. This gives 8 individual trajectories in the left nav, each
// showing its full Turn 1→20 back-and-forth in the right panel.

function _reconstructTrajectories(msgs) {
  // Each slot: { msgs: [], lastTurn: 0, won: null }
  const slots = [];
  const gameOvers = msgs.filter(m => (m.role || '').toLowerCase() === 'game over');

  let i = 0;
  while (i < msgs.length) {
    const msg = msgs[i];
    const role = (msg.role || '').toLowerCase();

    if (role === 'game over') { i++; continue; } // handled after

    // Expect Player immediately followed by Answerer for the same turn
    if (role === 'player' && i + 1 < msgs.length &&
        (msgs[i + 1].role || '').toLowerCase() === 'answerer') {
      const turn = msg.turn ?? 1;
      // Find slot whose lastTurn == turn - 1  (i.e. ready for this turn)
      let target = slots.find(s => s.lastTurn === turn - 1);
      if (!target) {
        target = { msgs: [], lastTurn: 0, won: null };
        slots.push(target);
      }
      target.msgs.push(msg, msgs[i + 1]);
      target.lastTurn = turn;
      i += 2;
    } else {
      i++;
    }
  }

  // Assign Game Over messages by matching Turns: N to the slot whose
  // lastTurn == N. Fall back to first unassigned slot if no match.
  gameOvers.forEach(go => {
    const m = go.content.match(/Turns:\s*(\d+)/);
    const goTurns = m ? parseInt(m[1]) : null;
    let target = null;
    if (goTurns !== null) {
      target = slots.find(s => s.lastTurn === goTurns && s.won === null);
    }
    if (!target) target = slots.find(s => s.won === null);
    if (target) {
      target.msgs.push(go);
      target.won = go.content.includes('\u2713') ? 1 : 0;
    }
  });

  return slots;
}

function renderGroupTurns(record, contentField) {
  const msgs = record[contentField] || [];
  if (!msgs.length) return '<div class="raw-text">No messages.</div>';

  const gameOverMsgs = msgs.filter(m => (m.role || '').toLowerCase() === 'game over');
  const wons   = gameOverMsgs.filter(m => m.content.includes('\u2713')).length;
  const total  = gameOverMsgs.length || 1;
  const secret = (() => {
    const raw = gameOverMsgs[0]?.content || '';
    const m = raw.match(/Secret:\s*([\w\s]+?)(?:,|$)/);
    return m ? m[1].trim() : (record.secret || null);
  })();

  const wonPct = Math.round(wons / total * 100);
  const wonColor = wons === 0 ? 'var(--score-0)' : wons === total ? 'var(--score-3)' : 'var(--score-2)';
  const header = `<div class="pc-meta">
    ${secret ? `<span><strong>Secret: ${esc(secret)}</strong></span>` : ''}
    <span style="color:${wonColor}">${wons}/${total} won (${wonPct}%)</span>
    <span style="color:var(--fg-muted)">Group ${record.group_idx ?? '?'} &middot; ${total} trajectories</span>
  </div>`;

  const cs = getComputedStyle(document.documentElement);
  const playerColor   = cs.getPropertyValue('--speaker-1').trim() || '#3b82f6';
  const answererColor = cs.getPropertyValue('--speaker-2').trim() || '#10b981';

  const trajectories = _reconstructTrajectories(msgs);

  // Nav: one item per trajectory
  const navHtml = trajectories.map((traj, i) => {
    const firstQ = traj.msgs.find(m => (m.role || '').toLowerCase() === 'player');
    const wonIcon = traj.won === 1 ? '\u2713' : traj.won === 0 ? '\u2717' : '?';
    const color   = traj.won === 1 ? 'var(--score-3)' : traj.won === 0 ? 'var(--score-0)' : '';
    return `<div class="turn-nav-item" data-gt="${i}" onclick="gtShow(${i})">
      <span class="turn-num" style="color:${color}">${wonIcon}</span>
      <span class="turn-preview">${esc((firstQ?.content || '').slice(0, 70))}</span>
    </div>`;
  }).join('');

  // Content: one panel per trajectory — show each turn as Q→A pair
  const contentHtml = trajectories.map((traj, i) => {
    const goMsg = traj.msgs.find(m => (m.role || '').toLowerCase() === 'game over');
    const turns = [];
    let j = 0;
    const trajMsgs = traj.msgs.filter(m => (m.role || '').toLowerCase() !== 'game over');
    while (j < trajMsgs.length) {
      const p = trajMsgs[j];
      const a = trajMsgs[j + 1];
      if (p && (p.role || '').toLowerCase() === 'player') {
        turns.push({ turn: p.turn, q: p.content, a: a?.content ?? null });
        j += 2;
      } else { j++; }
    }

    const rows = turns.map(t => `
      <div class="gt-pair">
        <span class="gt-turn-num">${t.turn ?? ''}</span>
        <div class="gt-exchange">
          <div class="gt-q"><span class="gt-role" style="color:${playerColor}">P</span>${esc(t.q)}</div>
          ${t.a != null ? `<div class="gt-a"><span class="gt-role" style="color:${answererColor}">A</span>${esc(t.a)}</div>` : ''}
        </div>
      </div>`).join('');

    const goRow = goMsg ? `<div class="gt-gameover" style="color:${traj.won ? 'var(--score-3)' : 'var(--score-0)'}">
      ${esc(goMsg.content)}</div>` : '';

    return `<div class="gt-panel" id="gt-${i}" style="display:none">${rows}${goRow}</div>`;
  }).join('');

  return `${header}
    <div class="two-pane">
      <div class="turn-nav">${navHtml}</div>
      <div class="turn-content">${contentHtml}</div>
    </div>
    <style>
      .gt-pair { display: flex; gap: 8px; border-bottom: 1px solid var(--border);
                 padding: 6px 0; }
      .gt-pair:last-of-type { border-bottom: none; }
      .gt-turn-num { flex: 0 0 22px; font-size: 11px; color: var(--fg-muted);
                     text-align: right; padding-top: 3px; }
      .gt-exchange { flex: 1; display: flex; flex-direction: column; gap: 2px; }
      .gt-q, .gt-a { display: flex; gap: 6px; font-size: 13px; line-height: 1.5; }
      .gt-a { color: var(--fg-muted); font-style: italic; }
      .gt-role { flex: 0 0 14px; font-size: 10px; font-weight: 700;
                 letter-spacing: .03em; padding-top: 3px; }
      .gt-gameover { margin-top: 12px; padding: 6px 8px; font-size: 12px;
                     border: 1px solid currentColor; border-radius: 4px;
                     opacity: 0.85; }
    </style>
    <script>
      (function() {
        function gtShow(i) {
          document.querySelectorAll('.turn-nav-item[data-gt]').forEach(n =>
            n.classList.toggle('active', +n.dataset.gt === i));
          document.querySelectorAll('.gt-panel').forEach(p => p.style.display = 'none');
          var p = document.getElementById('gt-' + i);
          if (p) { p.style.display = 'block'; }
        }
        window.gtShow = gtShow;
        gtShow(0);
      })();
    </script>`;
}

// --- format: trajectory ------------------------------------------------------
// One trajectory: Turn 1→N back-and-forth. Nav = game turns, content = Q→A pair.

function renderTrajectory(record, contentField) {
  const msgs = record[contentField] || [];
  if (!msgs.length) return '<div class="raw-text">No messages.</div>';

  const goMsg = msgs.find(m => (m.role || '').toLowerCase() === 'game over');
  const won   = goMsg ? goMsg.content.includes('\u2713') : null;
  const secret = (() => {
    const raw = goMsg?.content || record.secret || '';
    const m = raw.match ? raw.match(/Secret:\s*([\w\s]+?)(?:,|$)/) : null;
    return m ? m[1].trim() : (record.secret || null);
  })();

  const wonColor = won === true ? 'var(--score-3)' : won === false ? 'var(--score-0)' : 'var(--fg-muted)';
  const header = `<div class="pc-meta">
    ${secret ? `<span><strong>Secret: ${esc(secret)}</strong></span>` : ''}
    <span style="color:${wonColor}">${won === true ? '\u2713 Won' : won === false ? '\u2717 Lost' : ''}</span>
    <span style="color:var(--fg-muted)">Group ${record.group_idx ?? '?'} &middot; Traj ${(record.traj_idx ?? 0) + 1}</span>
  </div>`;

  // Build turn list: pair consecutive Player+Answerer messages
  const turns = [];
  let i = 0;
  const turnMsgs = msgs.filter(m => (m.role || '').toLowerCase() !== 'game over');
  while (i < turnMsgs.length) {
    const p = turnMsgs[i];
    const a = turnMsgs[i + 1];
    if (p && (p.role || '').toLowerCase() === 'player') {
      turns.push({ turn: p.turn ?? turns.length + 1, q: p.content, a: a?.content ?? null });
      i += 2;
    } else { i++; }
  }

  const cs = getComputedStyle(document.documentElement);
  const playerColor   = cs.getPropertyValue('--speaker-1').trim() || '#3b82f6';
  const answererColor = cs.getPropertyValue('--speaker-2').trim() || '#10b981';

  const navHtml = turns.map((t, i) =>
    `<div class="turn-nav-item" data-gt="${i}" onclick="gtShow(${i})">
      <span class="turn-num">${t.turn}</span>
      <span class="turn-preview">${esc((t.q || '').slice(0, 70))}</span>
    </div>`
  ).join('');

  const contentHtml = turns.map((t, i) =>
    `<div class="gt-panel" id="gt-${i}" style="display:none">
      <div class="gt-pair" style="border:none">
        <span class="gt-turn-num">${t.turn}</span>
        <div class="gt-exchange">
          <div class="gt-q"><span class="gt-role" style="color:${playerColor}">P</span>${esc(t.q)}</div>
          ${t.a != null ? `<div class="gt-a"><span class="gt-role" style="color:${answererColor}">A</span>${esc(t.a)}</div>` : ''}
        </div>
      </div>
    </div>`
  ).join('');

  const goHtml = goMsg ? `<div class="gt-gameover" style="color:${wonColor}">${esc(goMsg.content)}</div>` : '';

  return `${header}
    <div class="two-pane">
      <div class="turn-nav">${navHtml}</div>
      <div class="turn-content">${contentHtml}${goHtml}</div>
    </div>
    <style>
      .gt-pair { display:flex; gap:8px; padding:6px 0; }
      .gt-turn-num { flex:0 0 22px; font-size:11px; color:var(--fg-muted);
                     text-align:right; padding-top:3px; }
      .gt-exchange { flex:1; display:flex; flex-direction:column; gap:2px; }
      .gt-q, .gt-a { display:flex; gap:6px; font-size:13px; line-height:1.5; }
      .gt-a { color:var(--fg-muted); font-style:italic; }
      .gt-role { flex:0 0 14px; font-size:10px; font-weight:700;
                 letter-spacing:.03em; padding-top:3px; }
      .gt-gameover { margin:12px 8px 0; padding:6px 8px; font-size:12px;
                     border:1px solid currentColor; border-radius:4px; opacity:.85; }
    </style>
    <script>
      (function() {
        function gtShow(i) {
          document.querySelectorAll('.turn-nav-item[data-gt]').forEach(n =>
            n.classList.toggle('active', +n.dataset.gt === i));
          document.querySelectorAll('.gt-panel').forEach(p => p.style.display = 'none');
          var p = document.getElementById('gt-' + i);
          if (p) p.style.display = 'block';
        }
        window.gtShow = gtShow;
        gtShow(0);
      })();
    </script>`;
}

// --- format: conversation_list -----------------------------------------------

const ROLE_KEYS    = ['role', 'speaker', 'from', 'author', 'name'];
const CONTENT_KEYS = ['content', 'text', 'message', 'value', 'utterance'];

function renderConversationList(record, contentField) {
  const list = record[contentField] || [];
  const turns = list.map((item, i) => ({
    turn: i + 1,
    speaker: ROLE_KEYS.map(k => item[k]).find(v => v) || '?',
    text: String(CONTENT_KEYS.map(k => item[k]).find(v => v != null) || ''),
  }));
  return _turnTimelineHtml(turns);
}

// --- format: turn_text -------------------------------------------------------

const TURN_RE = /Turn\s+(\d+)\s*[-–:]\s*([^:\n]{1,40})\s*:\s*(.+)/gi;

function renderTurnText(record, contentField) {
  const text = String(record[contentField] || '');
  const turns = [];
  let m;
  // Reset lastIndex since regex is module-level
  TURN_RE.lastIndex = 0;
  while ((m = TURN_RE.exec(text)) !== null) {
    turns.push({turn: +m[1], speaker: m[2].trim(), text: m[3].trim()});
  }
  if (!turns.length) {
    return `<div class="raw-text">${esc(text)}</div>`;
  }
  return _turnTimelineHtml(turns);
}

// --- format: logtree_ast -----------------------------------------------------

function renderLogtreeAst(record, contentField) {
  const root = record[contentField];
  // Walk AST, collect text in DOM order
  const collected = [];
  (function walk(n) {
    if (typeof n === 'string') { collected.push(n); return; }
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n && typeof n === 'object') walk(n.children || []);
  })(root);
  const text = collected.join('\n');

  TURN_RE.lastIndex = 0;
  const turns = [];
  let m;
  while ((m = TURN_RE.exec(text)) !== null) {
    turns.push({turn: +m[1], speaker: m[2].trim(), text: m[3].trim()});
  }

  if (!turns.length) {
    // Fallback: render the AST as a tree
    return `<div class="tree">${_treeHtml(root, 0)}</div>`;
  }
  return _turnTimelineHtml(turns);
}

function _treeHtml(value, depth) {
  if (value === null || typeof value !== 'object') {
    return `<span class="leaf">${esc(JSON.stringify(value))}</span>`;
  }
  const entries = Array.isArray(value) ? value.map((v, i) => [i, v]) : Object.entries(value);
  const openAttr = depth < 2 ? 'open' : '';
  return entries.map(([k, v]) => {
    const summary = typeof v === 'object' && v !== null
      ? (Array.isArray(v) ? `[${v.length}]` : `{${Object.keys(v).length} keys}`)
      : esc(JSON.stringify(v));
    return `
      <details ${openAttr}>
        <summary><span class="key">${esc(String(k))}</span>: ${summary}</summary>
        ${typeof v === 'object' && v !== null ? _treeHtml(v, depth + 1) : ''}
      </details>
    `;
  }).join('');
}

// --- format: prompt_completion -----------------------------------------------

function renderPromptCompletion(record, promptField, completionField, scoreField) {
  const score = scoreField && record[scoreField] != null ? record[scoreField] : null;
  const meta = score != null
    ? `<div class="pc-meta"><span class="pc-score rollout-dot ${scoreClass(score)}" style="width: auto; padding: 2px 8px;">${fmt(score)}</span><span>${esc(scoreField)}</span></div>`
    : '';
  return `
    ${meta}
    <div class="pc-grid">
      <div class="pc-card">
        <div class="pc-label">${esc(promptField)}</div>
        <div class="pc-body">${esc(record[promptField] ?? '')}</div>
      </div>
      <div class="pc-card">
        <div class="pc-label">${esc(completionField)}</div>
        <div class="pc-body">${esc(record[completionField] ?? '')}</div>
      </div>
    </div>
  `;
}

// --- format: event_stream ----------------------------------------------------
// Records are events; multiple events share one rollout_id and form one rollout.
// The server pre-groups them — `record` is actually a list of events here.

function renderEventStream(record, contentField) {
  const events = contentField ? record[contentField] : record;
  if (!Array.isArray(events) || !events.length) {
    return '<div class="raw-text">No events for this rollout.</div>';
  }
  // Best-effort header — normalize type/event key
  const evType = e => e.type || e.event || '';
  const startEv  = events.find(e => evType(e) === 'rollout_start') || {};
  const endEv    = events.find(e => evType(e) === 'rollout_end')   || {};

  const meta = [];
  if (startEv.task)    meta.push(`task: ${startEv.task}`);
  if (endEv.reward != null)    meta.push(`reward: ${fmt(endEv.reward)}`);
  if (endEv.stop_condition)    meta.push(`stop: ${endEv.stop_condition}`);
  if (endEv.num_turns != null) meta.push(`turns: ${endEv.num_turns}`);
  const header = meta.length
    ? `<div class="pc-meta">${meta.map(m => `<span>${esc(m)}</span>`).join('')}</div>`
    : '';

  const turns = [];
  let tIdx = 1;
  for (const e of events) {
    const t = evType(e) || 'event';
    let kind = null, speaker = t, text = '', mTag = '', hasError = false;
    if (t === 'assistant_turn') {
      kind = 'assistant'; speaker = 'assistant';
      text = e.content || e.preview || '';
      mTag = e.turn != null ? `t${e.turn}` : '';
    } else if (t === 'tool_call') {
      kind = 'tool_call'; speaker = e.tool || 'tool_call';
      text = typeof e.args === 'object' ? JSON.stringify(e.args, null, 2) : String(e.args || '');
      mTag = e.turn != null ? `t${e.turn}` : '';
    } else if (t === 'tool_result') {
      kind = 'tool_result'; speaker = (e.tool || 'tool') + ' result';
      text = e.result || e.preview || '';
      mTag = e.error_kind ? `error: ${e.error_kind}` : (e.turn != null ? `t${e.turn}` : '');
      hasError = !!e.error;
    } else if (t === 'rollout_start' || t === 'rollout_end' || t === 'rollout_complete') {
      continue; // already in header
    } else {
      // generic event
      text = JSON.stringify(e);
    }
    turns.push({turn: tIdx++, speaker, text, _kind: kind, _meta: mTag, _has_error: hasError});
  }
  return _turnTimelineHtml(turns, header);
}

// --- format: raw_text --------------------------------------------------------

function renderRawText(record, contentField) {
  if (contentField && record[contentField] != null) {
    return `<div class="raw-text">${esc(String(record[contentField]))}</div>`;
  }
  // No content field — render the whole record as a tree
  return `<div class="tree">${_treeHtml(record, 0)}</div>`;
}

// --- main detail dispatch (used by detail.html) ------------------------------

function renderDetail(detail) {
  // detail: { source, record } where source has .format and .fields
  const f = detail.source.fields || {};
  // Per-record format override (set when a logtree got split into conversation records).
  const fmt = detail.record._format_override || detail.source.format;

  // For split logtree records, show the title + metadata at the top.
  let header = '';
  if (detail.record._conversation_title) {
    const meta = detail.record._conversation_title;
    const sf = detail.record._source_file;
    header = `<div class="pc-meta">
      <span><strong>${esc(meta)}</strong></span>
      ${sf ? `<span style="color: var(--fg-muted)">${esc(sf)}</span>` : ''}
    </div>`;
  }

  switch (fmt) {
    case 'group_turns':
      return renderGroupTurns(detail.record, f.content);
    case 'trajectory':
      return renderTrajectory(detail.record, f.content);
    case 'conversation_list':
      return header + renderConversationList(detail.record, f.content);
    case 'turn_text':
      return header + renderTurnText(detail.record, f.content);
    case 'logtree_ast':
      return header + renderLogtreeAst(detail.record, f.content);
    case 'prompt_completion':
      return renderPromptCompletion(detail.record, f.content_prompt, f.content_completion, f.score);
    case 'event_stream':
      return renderEventStream(detail.record, f.content);
    case 'raw_text':
    default:
      return header + renderRawText(detail.record, f.content);
  }
}
