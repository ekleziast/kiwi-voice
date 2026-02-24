/**
 * Kiwi Voice Dashboard - Main Application
 *
 * Vanilla JavaScript client for the Kiwi Voice REST API and WebSocket events.
 * No build tools or external dependencies required.
 */

// ============================================================
// Configuration
// ============================================================

const API_BASE = window.location.origin + '/api';
const WS_URL = `ws://${window.location.host}/api/events`;

// ============================================================
// State
// ============================================================

let ws = null;
let wsReconnectTimer = null;
let eventLog = [];
const MAX_EVENT_LOG = 100;
let statusPollInterval = null;
let startTime = null;       // Service start time (from API)
let uptimeInterval = null;
let apiAvailable = false;   // Track whether the API responded at least once

// ============================================================
// Initialization
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    addEventLogEntry('SYSTEM', 'Dashboard initialized', 'system');
    fetchStatus();
    fetchLanguages();
    fetchSpeakers();
    connectWebSocket();

    // Poll status every 5 seconds as a fallback
    statusPollInterval = setInterval(fetchStatus, 5000);

    // Update uptime display every second
    uptimeInterval = setInterval(updateUptimeDisplay, 1000);
});

// ============================================================
// API Calls
// ============================================================

/**
 * Fetch current service status from /api/status
 */
async function fetchStatus() {
    try {
        const resp = await fetch(`${API_BASE}/status`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        updateStatus(data);
        showApiConnected();
    } catch (err) {
        // If the API is not reachable, silently wait for it
        if (apiAvailable) {
            addEventLogEntry('ERROR', `Status fetch failed: ${err.message}`, 'error');
        }
    }
}

/**
 * Fetch available languages from /api/languages
 */
async function fetchLanguages() {
    try {
        const resp = await fetch(`${API_BASE}/languages`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        updateLanguages(data);
    } catch (err) {
        // Silently ignore on first load
    }
}

/**
 * Fetch known speakers from /api/speakers
 */
async function fetchSpeakers() {
    try {
        const resp = await fetch(`${API_BASE}/speakers`);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        updateSpeakers(data);
    } catch (err) {
        // Silently ignore on first load
    }
}

/**
 * Switch language via POST /api/language
 */
async function applyLanguage() {
    const select = document.getElementById('language-select');
    const lang = select.value;
    if (!lang) return;

    try {
        const resp = await fetch(`${API_BASE}/language`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ language: lang })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        showToast(`Language changed to ${lang}`, 'success');
        addEventLogEntry('SYSTEM', `Language changed to ${lang}`, 'system');
        fetchStatus();
    } catch (err) {
        showToast(`Failed to change language: ${err.message}`, 'error');
        addEventLogEntry('ERROR', `Language change failed: ${err.message}`, 'error');
    }
}

/**
 * Test TTS via POST /api/tts/test
 */
async function testTTS() {
    const input = document.getElementById('tts-test-input');
    const text = input.value.trim();
    if (!text) {
        showToast('Enter some text first', 'info');
        return;
    }

    const btn = document.getElementById('tts-test-btn');
    btn.disabled = true;
    btn.textContent = 'Speaking...';

    try {
        const resp = await fetch(`${API_BASE}/tts/test`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast('TTS test sent', 'success');
        addEventLogEntry('TTS', `Test: "${text}"`, 'tts');
    } catch (err) {
        showToast(`TTS test failed: ${err.message}`, 'error');
        addEventLogEntry('ERROR', `TTS test failed: ${err.message}`, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Speak';
    }
}

/**
 * Stop current playback via POST /api/stop
 */
async function stopService() {
    try {
        const resp = await fetch(`${API_BASE}/stop`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast('Playback stopped', 'success');
        addEventLogEntry('SYSTEM', 'Stop command sent', 'system');
    } catch (err) {
        showToast(`Stop failed: ${err.message}`, 'error');
        addEventLogEntry('ERROR', `Stop failed: ${err.message}`, 'error');
    }
}

/**
 * Reset conversation context via POST /api/reset-context
 */
async function resetContext() {
    try {
        const resp = await fetch(`${API_BASE}/reset-context`, { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast('Context reset', 'success');
        addEventLogEntry('SYSTEM', 'Context reset', 'system');
    } catch (err) {
        showToast(`Reset failed: ${err.message}`, 'error');
        addEventLogEntry('ERROR', `Context reset failed: ${err.message}`, 'error');
    }
}

/**
 * Block a speaker via POST /api/speakers/:id/block
 */
async function blockSpeaker(id) {
    if (!confirm(`Block speaker "${id}"?`)) return;
    try {
        const resp = await fetch(`${API_BASE}/speakers/${encodeURIComponent(id)}/block`, {
            method: 'POST'
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast(`Speaker "${id}" blocked`, 'success');
        addEventLogEntry('SPEAKER', `Blocked: ${id}`, 'speaker');
        fetchSpeakers();
    } catch (err) {
        showToast(`Block failed: ${err.message}`, 'error');
    }
}

/**
 * Unblock a speaker via POST /api/speakers/:id/unblock
 */
async function unblockSpeaker(id) {
    try {
        const resp = await fetch(`${API_BASE}/speakers/${encodeURIComponent(id)}/unblock`, {
            method: 'POST'
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast(`Speaker "${id}" unblocked`, 'success');
        addEventLogEntry('SPEAKER', `Unblocked: ${id}`, 'speaker');
        fetchSpeakers();
    } catch (err) {
        showToast(`Unblock failed: ${err.message}`, 'error');
    }
}

/**
 * Delete a speaker via DELETE /api/speakers/:id
 */
async function deleteSpeaker(id) {
    if (!confirm(`Delete speaker "${id}"? This cannot be undone.`)) return;
    try {
        const resp = await fetch(`${API_BASE}/speakers/${encodeURIComponent(id)}`, {
            method: 'DELETE'
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        showToast(`Speaker "${id}" deleted`, 'success');
        addEventLogEntry('SPEAKER', `Deleted: ${id}`, 'speaker');
        fetchSpeakers();
    } catch (err) {
        showToast(`Delete failed: ${err.message}`, 'error');
    }
}

// ============================================================
// WebSocket
// ============================================================

/**
 * Connect to the WebSocket endpoint for real-time events.
 * Automatically reconnects on close.
 */
function connectWebSocket() {
    // Clear any pending reconnect timer
    if (wsReconnectTimer) {
        clearTimeout(wsReconnectTimer);
        wsReconnectTimer = null;
    }

    try {
        ws = new WebSocket(WS_URL);
    } catch (err) {
        scheduleWsReconnect();
        return;
    }

    ws.onopen = () => {
        addEventLogEntry('SYSTEM', 'WebSocket connected', 'system');
        updateConnectionDot(true);
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            handleWsEvent(data);
        } catch (err) {
            // Non-JSON message, ignore
        }
    };

    ws.onerror = () => {
        // onerror is usually followed by onclose, so we handle reconnect there
    };

    ws.onclose = () => {
        updateConnectionDot(false);
        scheduleWsReconnect();
    };
}

function scheduleWsReconnect() {
    if (wsReconnectTimer) return;
    wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectWebSocket();
    }, 3000);
}

/**
 * Handle an incoming WebSocket event object.
 */
function handleWsEvent(data) {
    const eventType = data.type || data.event || 'UNKNOWN';
    const payload = data.payload || data.data || data;

    // Determine CSS class for the log entry
    const cssClass = getEventCssClass(eventType);
    const displayData = typeof payload === 'string' ? payload : JSON.stringify(payload);
    addEventLogEntry(eventType, displayData, cssClass);

    // Live status updates from events
    updateStatusFromEvent(eventType, payload);
}

/**
 * Map event type strings to CSS class names for coloring.
 */
function getEventCssClass(eventType) {
    const upper = String(eventType).toUpperCase();
    if (upper.includes('STATE')) return 'state';
    if (upper.includes('SPEECH') || upper.includes('WAKE') || upper.includes('COMMAND') || upper.includes('LISTEN')) return 'speech';
    if (upper.includes('TTS') || upper.includes('SPEAK')) return 'tts';
    if (upper.includes('ERROR')) return 'error';
    if (upper.includes('SPEAKER') || upper.includes('IDENTIFY')) return 'speaker';
    if (upper.includes('LLM') || upper.includes('THINK')) return 'llm';
    if (upper.includes('VAD')) return 'vad';
    return 'system';
}

// ============================================================
// UI Update Functions
// ============================================================

/**
 * Update dashboard status from API response.
 * Expected shape: { state, is_speaking, is_processing, active_speaker, tts_provider, uptime, language, ... }
 */
function updateStatus(status) {
    // Dialogue state
    const state = (status.state || 'idle').toLowerCase();
    const stateEl = document.getElementById('status-state');
    stateEl.innerHTML = getStateBadge(state);

    // State dot in header
    const dot = document.getElementById('state-dot');
    dot.className = 'state-dot ' + state;
    const label = document.getElementById('state-label');
    label.textContent = state.charAt(0).toUpperCase() + state.slice(1);

    // Speaking / Processing
    const speakingEl = document.getElementById('status-speaking');
    speakingEl.innerHTML = status.is_speaking
        ? '<span class="indicator-on">Yes</span>'
        : '<span class="indicator-off">No</span>';

    const processingEl = document.getElementById('status-processing');
    processingEl.innerHTML = status.is_processing
        ? '<span class="indicator-on">Yes</span>'
        : '<span class="indicator-off">No</span>';

    // Active speaker
    const speakerEl = document.getElementById('status-speaker');
    speakerEl.textContent = status.active_speaker || '--';

    // TTS provider
    const ttsEl = document.getElementById('status-tts-provider');
    ttsEl.textContent = status.tts_provider || '--';

    // Language
    const langEl = document.getElementById('current-language');
    langEl.textContent = status.language || '--';

    // Uptime
    if (status.uptime !== undefined) {
        startTime = Date.now() - (status.uptime * 1000);
    }
}

/**
 * Update language dropdown from API response.
 * Expected shape: { current, available: [{code, name}] } or { current, available: ["ru", "en"] }
 */
function updateLanguages(data) {
    const select = document.getElementById('language-select');
    select.innerHTML = '';

    const current = data.current || '';
    const available = data.available || data.languages || [];

    if (available.length === 0) {
        const opt = document.createElement('option');
        opt.value = '';
        opt.textContent = 'No languages available';
        select.appendChild(opt);
        return;
    }

    for (const lang of available) {
        const opt = document.createElement('option');
        if (typeof lang === 'object') {
            opt.value = lang.code || lang.id || '';
            opt.textContent = lang.name || lang.code || '';
        } else {
            opt.value = lang;
            opt.textContent = lang;
        }
        if (opt.value === current) {
            opt.selected = true;
        }
        select.appendChild(opt);
    }

    // Update current language display
    const langEl = document.getElementById('current-language');
    langEl.textContent = current || '--';
}

/**
 * Update speakers table from API response.
 * Expected shape: { speakers: { id: { name, priority, is_blocked, samples, last_seen } } }
 * or an array: [{ id, name, priority, ... }]
 */
function updateSpeakers(data) {
    const tbody = document.getElementById('speakers-tbody');

    // Normalize to array
    let speakers = [];
    if (Array.isArray(data)) {
        speakers = data;
    } else if (data.speakers && typeof data.speakers === 'object') {
        if (Array.isArray(data.speakers)) {
            speakers = data.speakers;
        } else {
            speakers = Object.entries(data.speakers).map(([id, info]) => ({ id, ...info }));
        }
    }

    if (speakers.length === 0) {
        tbody.innerHTML = '<tr><td colspan="5" class="empty-state">No speakers registered</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    for (const sp of speakers) {
        const id = sp.id || sp.speaker_id || '';
        const name = sp.name || sp.display_name || id;
        const priority = sp.priority || 'guest';
        const isBlocked = sp.is_blocked || false;
        const samples = sp.samples !== undefined ? sp.samples : '--';
        const lastSeen = sp.last_seen ? formatLastSeen(sp.last_seen) : '--';

        const effectivePriority = isBlocked ? 'blocked' : priority;

        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${escapeHtml(name)}</td>
            <td>${getPriorityBadge(effectivePriority)}</td>
            <td>${samples}</td>
            <td>${lastSeen}</td>
            <td class="speaker-actions">
                ${getActionsHtml(id, isBlocked, priority)}
            </td>
        `;
        tbody.appendChild(tr);
    }
}

/**
 * Update live status from a WebSocket event.
 */
function updateStatusFromEvent(eventType, payload) {
    const upper = String(eventType).toUpperCase();

    // State change events
    if (upper.includes('STATE_CHANGED') || upper === 'STATE') {
        const newState = (payload.state || payload.new_state || '').toLowerCase();
        if (newState) {
            const stateEl = document.getElementById('status-state');
            stateEl.innerHTML = getStateBadge(newState);

            const dot = document.getElementById('state-dot');
            dot.className = 'state-dot ' + newState;
            const label = document.getElementById('state-label');
            label.textContent = newState.charAt(0).toUpperCase() + newState.slice(1);
        }
    }

    // TTS events
    if (upper.includes('TTS_STARTED')) {
        document.getElementById('status-speaking').innerHTML = '<span class="indicator-on">Yes</span>';
    }
    if (upper.includes('TTS_ENDED')) {
        document.getElementById('status-speaking').innerHTML = '<span class="indicator-off">No</span>';
    }

    // LLM events
    if (upper.includes('LLM_THINKING_STARTED')) {
        document.getElementById('status-processing').innerHTML = '<span class="indicator-on">Yes</span>';
    }
    if (upper.includes('LLM_THINKING_ENDED') || upper.includes('LLM_RESPONSE_COMPLETE')) {
        document.getElementById('status-processing').innerHTML = '<span class="indicator-off">No</span>';
    }

    // Speaker identification
    if (upper.includes('SPEAKER_IDENTIFIED')) {
        const name = payload.name || payload.speaker_name || payload.display_name || '';
        if (name) {
            document.getElementById('status-speaker').textContent = name;
        }
    }
}

// ============================================================
// Event Log
// ============================================================

/**
 * Add an entry to the on-screen event log.
 */
function addEventLogEntry(type, data, cssClass) {
    const logEl = document.getElementById('event-log');
    const now = new Date();
    const time = formatTimeHMS(now);

    const entry = document.createElement('div');
    entry.className = `event-entry event-${cssClass || 'system'}`;
    entry.innerHTML = `
        <span class="event-time">${time}</span>
        <span class="event-type">${escapeHtml(String(type))}</span>
        <span class="event-data">${escapeHtml(String(data))}</span>
    `;

    logEl.appendChild(entry);
    eventLog.push(entry);

    // Trim old entries
    while (eventLog.length > MAX_EVENT_LOG) {
        const oldest = eventLog.shift();
        if (oldest && oldest.parentNode) {
            oldest.parentNode.removeChild(oldest);
        }
    }

    // Auto-scroll to bottom
    logEl.scrollTop = logEl.scrollHeight;
}

/**
 * Clear the event log.
 */
function clearEventLog() {
    const logEl = document.getElementById('event-log');
    logEl.innerHTML = '';
    eventLog = [];
    addEventLogEntry('SYSTEM', 'Log cleared', 'system');
}

// ============================================================
// Helpers
// ============================================================

/**
 * Show API connected state (hide overlay if it was shown).
 */
function showApiConnected() {
    if (!apiAvailable) {
        apiAvailable = true;
        const overlay = document.getElementById('connection-overlay');
        overlay.style.display = 'none';
    }
}

/**
 * Update the header connection dot for WebSocket state.
 */
function updateConnectionDot(connected) {
    const dot = document.getElementById('state-dot');
    if (!connected && !apiAvailable) {
        dot.className = 'state-dot disconnected';
        document.getElementById('state-label').textContent = 'Disconnected';
    }
}

/**
 * Update the uptime display every second.
 */
function updateUptimeDisplay() {
    if (!startTime) return;
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    document.getElementById('uptime').textContent = `Uptime: ${formatDuration(elapsed)}`;
}

/**
 * Format seconds into HH:MM:SS.
 */
function formatDuration(totalSeconds) {
    const h = Math.floor(totalSeconds / 3600);
    const m = Math.floor((totalSeconds % 3600) / 60);
    const s = totalSeconds % 60;
    return [
        String(h).padStart(2, '0'),
        String(m).padStart(2, '0'),
        String(s).padStart(2, '0')
    ].join(':');
}

/**
 * Format a Date into HH:MM:SS.mmm
 */
function formatTimeHMS(date) {
    return [
        String(date.getHours()).padStart(2, '0'),
        String(date.getMinutes()).padStart(2, '0'),
        String(date.getSeconds()).padStart(2, '0')
    ].join(':') + '.' + String(date.getMilliseconds()).padStart(3, '0');
}

/**
 * Format an ISO date string or timestamp into a human-readable "last seen" string.
 */
function formatLastSeen(isoString) {
    try {
        const date = new Date(isoString);
        if (isNaN(date.getTime())) return isoString;
        const now = new Date();
        const diff = Math.floor((now - date) / 1000);

        if (diff < 60) return 'just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return date.toLocaleDateString();
    } catch {
        return isoString;
    }
}

/**
 * Return an HTML state badge for the given state string.
 */
function getStateBadge(state) {
    const s = String(state).toLowerCase();
    const label = s.charAt(0).toUpperCase() + s.slice(1);
    return `<span class="state-badge state-${s}">${label}</span>`;
}

/**
 * Return an HTML priority badge.
 */
function getPriorityBadge(priority) {
    const p = String(priority).toLowerCase();
    const labelMap = {
        owner: 'Owner',
        friend: 'Friend',
        guest: 'Guest',
        blocked: 'Blocked',
        self: 'Self'
    };
    const label = labelMap[p] || p.charAt(0).toUpperCase() + p.slice(1);
    return `<span class="priority-badge priority-${p}">${label}</span>`;
}

/**
 * Return action button HTML for a speaker row.
 */
function getActionsHtml(id, isBlocked, priority) {
    // OWNER cannot be blocked or deleted
    const p = String(priority).toLowerCase();
    if (p === 'owner') {
        return '<span style="color: var(--text-muted); font-size: 0.8rem;">Protected</span>';
    }

    const blockBtn = isBlocked
        ? `<button class="btn btn-small btn-secondary" onclick="unblockSpeaker('${escapeAttr(id)}')">Unblock</button>`
        : `<button class="btn btn-small btn-danger" onclick="blockSpeaker('${escapeAttr(id)}')">Block</button>`;

    const deleteBtn = `<button class="btn btn-small btn-secondary" onclick="deleteSpeaker('${escapeAttr(id)}')">Delete</button>`;

    return blockBtn + deleteBtn;
}

/**
 * Escape HTML entities to prevent XSS.
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/**
 * Escape a string for safe use inside an HTML attribute (single-quoted).
 */
function escapeAttr(text) {
    return String(text).replace(/\\/g, '\\\\').replace(/'/g, "\\'").replace(/"/g, '&quot;');
}

/**
 * Show a toast notification.
 * @param {string} message - Toast message
 * @param {'success'|'error'|'info'} type - Toast type
 * @param {number} duration - Display duration in ms
 */
function showToast(message, type = 'info', duration = 3000) {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;

    container.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-out');
        toast.addEventListener('animationend', () => {
            if (toast.parentNode) toast.parentNode.removeChild(toast);
        });
    }, duration);
}
