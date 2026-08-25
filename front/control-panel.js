const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const state = {
    token: sessionStorage.getItem('gcli-control-token') || '',
    page: 'overview',
    credentialMode: 'geminicli',
    connectMode: 'geminicli',
    credentialOffset: 0,
    credentialLimit: 50,
    credentialResponse: null,
    selectedCredentials: new Set(),
    quotaCredentials: [],
    quotaCache: new Map(),
    quotaRequests: new Map(),
    quotaObserver: null,
    oauthProjectId: '',
    logSocket: null,
    logPaused: false,
    mihomo: null,
    proxies: {},
    activeGroup: '',
};

const pages = {
    overview: ['总览', 'CONTROL PLANE'],
    credentials: ['凭证管理', 'ACCOUNT POOL'],
    quota: ['额度中心', 'MODEL CAPACITY'],
    connect: ['添加凭证', 'ONBOARDING'],
    mihomo: ['Mihomo', 'NETWORK EGRESS'],
    settings: ['系统配置', 'SYSTEM'],
    logs: ['实时日志', 'OBSERVABILITY'],
    about: ['关于', 'PROJECT'],
};

const endpointDefinitions = [
    ['OpenAI', '/v1/chat/completions'],
    ['Gemini', '/v1beta/models/{model}:generateContent'],
    ['Antigravity', '/antigravity/v1/chat/completions'],
];

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#039;');
}

function icon(name) {
    return `<svg aria-hidden="true"><use href="#i-${name}"/></svg>`;
}

async function api(path, options = {}) {
    const headers = new Headers(options.headers || {});
    if (state.token) headers.set('Authorization', `Bearer ${state.token}`);
    if (options.body && !(options.body instanceof FormData) && !headers.has('Content-Type')) {
        headers.set('Content-Type', 'application/json');
    }
    const response = await fetch(path, { ...options, headers });
    if (response.status === 401 || response.status === 403) {
        if (!path.includes('/auth/login')) logout(false);
    }
    const contentType = response.headers.get('content-type') || '';
    const data = contentType.includes('json') ? await response.json() : await response.text();
    if (!response.ok && data?.success !== true) {
        const message = data?.detail || data?.message || data?.error || String(data) || `HTTP ${response.status}`;
        const error = new Error(typeof message === 'string' ? message : JSON.stringify(message));
        error.status = response.status;
        error.data = data;
        throw error;
    }
    return data;
}

async function downloadAuthenticated(path, fallbackName) {
    const response = await fetch(path, { headers: { Authorization: `Bearer ${state.token}` } });
    if (!response.ok) {
        let message = `下载失败：HTTP ${response.status}`;
        try { message = (await response.json()).detail || message; } catch {}
        throw new Error(message);
    }
    const blob = await response.blob();
    const disposition = response.headers.get('content-disposition') || '';
    const matched = disposition.match(/filename="?([^";]+)"?/i);
    const name = matched?.[1] || fallbackName;
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    setTimeout(() => URL.revokeObjectURL(url), 500);
}

function toast(message, type = 'success', duration = 4200) {
    const node = document.createElement('div');
    node.className = `toast ${type}`;
    node.innerHTML = `<i></i><span>${escapeHtml(message)}</span><button aria-label="关闭">×</button>`;
    $('button', node).addEventListener('click', () => node.remove());
    $('#toastRegion').append(node);
    setTimeout(() => node.remove(), duration);
}

function setBusy(button, busy, label = '处理中…') {
    if (!button) return;
    if (busy) {
        button.dataset.previousHtml = button.innerHTML;
        button.disabled = true;
        button.textContent = label;
    } else {
        button.disabled = false;
        if (button.dataset.previousHtml) button.innerHTML = button.dataset.previousHtml;
    }
}

function openModal(title, eyebrow, body) {
    $('#modalTitle').textContent = title;
    $('#modalEyebrow').textContent = eyebrow;
    $('#modalBody').innerHTML = body;
    $('#modal').classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeModal() {
    $('#modal').classList.add('hidden');
    document.body.style.overflow = '';
}

function formatTime(value) {
    if (!value) return '—';
    const numeric = Number(value);
    const date = Number.isFinite(numeric)
        ? new Date(numeric > 1e12 ? numeric : numeric * 1000)
        : new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return new Intl.DateTimeFormat('zh-CN', {
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
    }).format(date);
}

function renderEndpoints(target) {
    const origin = window.location.origin;
    $(target).innerHTML = endpointDefinitions.map(([name, path]) => `
        <button class="endpoint" data-copy="${escapeHtml(origin + path)}">
            <span><b>${name}</b><code>${escapeHtml(origin + path)}</code></span>${icon('copy')}
        </button>`).join('');
}

async function copyText(value) {
    try {
        await navigator.clipboard.writeText(value);
        toast('地址已复制');
    } catch {
        const input = document.createElement('textarea');
        input.value = value;
        document.body.append(input);
        input.select();
        document.execCommand('copy');
        input.remove();
        toast('地址已复制');
    }
}

function logout(showMessage = true) {
    state.token = '';
    sessionStorage.removeItem('gcli-control-token');
    disconnectLogs();
    $('#appView').classList.add('hidden');
    $('#loginView').classList.remove('hidden');
    if (showMessage) toast('已退出登录');
}

async function login(password) {
    const result = await api('/auth/login', {
        method: 'POST',
        body: JSON.stringify({ password }),
    });
    state.token = result.token;
    sessionStorage.setItem('gcli-control-token', state.token);
    await enterApp();
}

async function enterApp() {
    $('#loginView').classList.add('hidden');
    $('#appView').classList.remove('hidden');
    renderEndpoints('#endpointList');
    renderEndpoints('#aboutEndpoints');
    updateClock();
    navigate('overview');
}

function closeSidebar() {
    $('#sidebar').classList.remove('open');
    $('#sidebarScrim').classList.remove('open');
}

function navigate(page) {
    if (!pages[page]) return;
    if (state.page === 'logs' && page !== 'logs') disconnectLogs();
    state.page = page;
    $$('.page').forEach((item) => item.classList.toggle('active', item.id === `page-${page}`));
    $$('.nav-item').forEach((item) => item.classList.toggle('active', item.dataset.page === page));
    $('#pageTitle').textContent = pages[page][0];
    $('#pageEyebrow').textContent = pages[page][1];
    closeSidebar();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    loadPage(page).catch((error) => toast(error.message, 'error', 6500));
}

function loadPage(page, force = false) {
    const loaders = {
        overview: loadOverview,
        credentials: loadCredentials,
        quota: loadQuotaPage,
        mihomo: loadMihomo,
        settings: loadSettings,
        logs: connectLogs,
        about: loadAbout,
    };
    return loaders[page]?.(force) || Promise.resolve();
}

function updateClock() {
    $('#clock').textContent = new Intl.DateTimeFormat('zh-CN', {
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    }).format(new Date());
}

function credentialQuery(mode, overrides = {}) {
    const params = new URLSearchParams({
        mode,
        offset: String(overrides.offset ?? 0),
        limit: String(overrides.limit ?? 20),
        status_filter: overrides.status ?? 'all',
        error_code_filter: overrides.error ?? 'all',
        cooldown_filter: overrides.cooldown ?? 'all',
        preview_filter: overrides.preview ?? 'all',
        tier_filter: overrides.tier ?? 'all',
    });
    return `/creds/status?${params}`;
}

async function loadOverview() {
    $('#overviewStats').innerHTML = '<div class="loading">加载状态中…</div>';
    const [gcli, antigravity, mihomo] = await Promise.all([
        api(credentialQuery('geminicli')),
        api(credentialQuery('antigravity')),
        api('/mihomo/status').catch(() => ({ data: null })),
    ]);
    const total = gcli.total + antigravity.total;
    const enabled = (gcli.stats?.normal || 0) + (antigravity.stats?.normal || 0);
    const disabled = (gcli.stats?.disabled || 0) + (antigravity.stats?.disabled || 0);
    const network = mihomo.data;
    $('#navCredentialCount').textContent = String(total);
    $('#mihomoNavDot').classList.toggle('online', Boolean(network?.running));
    $('#overviewStats').innerHTML = [
        ['凭证总数', total, `${gcli.total} GCLI · ${antigravity.total} AG`, 'var(--green)'],
        ['可用凭证', enabled, total ? `${Math.round(enabled / total * 100)}% 健康` : '等待导入', 'var(--cyan)'],
        ['停用凭证', disabled, disabled ? '需要检查或手动启用' : '当前无停用项', 'var(--red)'],
        ['网络出口', network?.running ? '已接管' : '直连', network?.running ? network.current_profile : 'Mihomo 未运行', 'var(--violet)'],
    ].map(([label, value, note, accent]) => `
        <article class="stat-card" style="--accent:${accent}"><span>${label}</span><strong>${escapeHtml(value)}</strong><small>${escapeHtml(note)}</small></article>
    `).join('');
    $('#healthOverview').innerHTML = [
        ['Gemini CLI', gcli], ['Antigravity', antigravity],
    ].map(([label, data]) => {
        const good = data.stats?.normal || 0;
        const bad = data.stats?.disabled || 0;
        const ratio = data.total ? Math.round(good / data.total * 100) : 0;
        const unavailableRatio = data.total ? 100 - ratio : 0;
        return `<div class="health-block"><div class="health-title"><span>${label}</span><b>${good}/${data.total}</b></div><div class="health-track"><i style="width:${ratio}%"></i><i class="bad" style="width:${unavailableRatio}%"></i></div><div class="health-meta"><span>${ratio}% 可用</span><span>${bad} 停用</span></div></div>`;
    }).join('');
}

async function loadCredentials() {
    const container = $('#credentialTable');
    container.innerHTML = '<div class="loading">读取凭证池…</div>';
    const response = await api(credentialQuery(state.credentialMode, {
        offset: state.credentialOffset,
        limit: state.credentialLimit,
        status: $('#statusFilter').value,
        tier: $('#tierFilter').value,
    }));
    state.credentialResponse = response;
    const validNames = new Set(response.items.map((item) => item.filename));
    state.selectedCredentials = new Set([...state.selectedCredentials].filter((name) => validNames.has(name)));
    renderCredentials();
}

function cooldownInfo(cooldowns) {
    if (!cooldowns || typeof cooldowns !== 'object') return ['无冷却', ''];
    const active = Object.entries(cooldowns).filter(([, value]) => {
        const until = typeof value === 'object' ? value?.until || value?.timestamp : value;
        const timestamp = Number(until);
        return timestamp ? timestamp * (timestamp < 1e12 ? 1000 : 1) > Date.now() : Boolean(value);
    });
    if (!active.length) return ['无冷却', ''];
    return [`${active.length} 个模型`, active.map(([name]) => name).slice(0, 2).join(' · ')];
}

function inlineQuotaPlaceholder(filename) {
    return `<div class="inline-quota is-loading" data-inline-quota="${escapeHtml(filename)}" role="button" tabindex="0" title="点击查看完整额度">
        <div class="inline-quota-row"><span>Gemini</span><i><b></b></i><strong>—</strong><small>读取中</small></div>
        <div class="inline-quota-row"><span>Claude</span><i><b></b></i><strong>—</strong><small>读取中</small></div>
    </div>`;
}

function renderCredentials() {
    state.quotaObserver?.disconnect();
    state.quotaObserver = null;
    const response = state.credentialResponse;
    if (!response) return;
    const query = $('#credentialSearch').value.trim().toLowerCase();
    const items = response.items.filter((item) => !query ||
        item.filename.toLowerCase().includes(query) ||
        String(item.user_email || '').toLowerCase().includes(query));
    $('#credentialStats').innerHTML = [
        ['当前结果', response.total],
        ['可用', response.stats?.normal || 0],
        ['停用', response.stats?.disabled || 0],
        ['本页已选', state.selectedCredentials.size],
    ].map(([label, value]) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`).join('');

    if (!items.length) {
        $('#credentialTable').innerHTML = '<div class="empty-state"><h3>没有匹配的凭证</h3><p>调整筛选条件，或先添加新的凭证。</p></div>';
    } else {
        const rows = items.map((item) => {
            const [cooldown, cooldownNote] = cooldownInfo(item.model_cooldowns);
            const errors = Array.isArray(item.error_codes) ? item.error_codes : [];
            const capability = state.credentialMode === 'geminicli'
                ? `<span class="badge ${item.preview ? 'violet' : ''}">${item.preview ? 'Preview' : 'Stable'}</span>`
                : `<span class="badge ${item.enable_credit ? 'amber' : ''}">${item.enable_credit ? 'Credit' : 'Quota'}</span>`;
            const antigravity = state.credentialMode === 'antigravity';
            return `<div class="table-row ${antigravity ? 'antigravity-row' : ''}" data-filename="${escapeHtml(item.filename)}">
                <input class="check credential-check" type="checkbox" aria-label="选择 ${escapeHtml(item.filename)}" ${state.selectedCredentials.has(item.filename) ? 'checked' : ''}>
                <div class="identity"><b>${escapeHtml(item.user_email || '邮箱未获取')}</b><small>${escapeHtml(item.filename)}</small></div>
                <span class="badge ${item.disabled ? 'bad' : 'good'}">${item.disabled ? 'Disabled' : 'Enabled'}</span>
                <div class="cell-stack"><span class="badge">${escapeHtml(item.tier || 'pro')}</span>${capability}</div>
                ${antigravity ? inlineQuotaPlaceholder(item.filename) : ''}
                <div class="cell-stack"><span>${errors.length ? `<span class="badge bad">HTTP ${escapeHtml(errors.join('/'))}</span>` : cooldown}</span><small>${escapeHtml(cooldownNote)}</small></div>
                <div class="cell-stack"><span>${formatTime(item.last_success)}</span><small>最近成功</small></div>
                <div class="row-menu"><button data-menu="${escapeHtml(item.filename)}" aria-label="凭证操作">${icon('more')}</button></div>
            </div>`;
        }).join('');
        const antigravity = state.credentialMode === 'antigravity';
        $('#credentialTable').innerHTML = `<div class="table-row header ${antigravity ? 'antigravity-row' : ''}"><input id="selectPage" class="check" type="checkbox" aria-label="选择本页"><span>账号</span><span>状态</span><span>等级 / 能力</span>${antigravity ? '<span>模型额度</span>' : ''}<span>错误 / 冷却</span><span>最近成功</span><span></span></div>${rows}`;
        $('#selectPage').checked = items.length > 0 && items.every((item) => state.selectedCredentials.has(item.filename));
        if (antigravity) observeInlineQuotas();
    }
    const first = response.total ? response.offset + 1 : 0;
    const last = Math.min(response.offset + response.limit, response.total);
    $('#pageInfo').textContent = `${first}–${last} / ${response.total}`;
    $('#prevPage').disabled = response.offset <= 0;
    $('#nextPage').disabled = !response.has_more;
    renderBulkBar();
}

function renderBulkBar() {
    const count = state.selectedCredentials.size;
    $('#bulkBar').classList.toggle('hidden', count === 0);
    $('#selectedCount').textContent = String(count);
}

function credentialActionItems(filename) {
    const item = state.credentialResponse?.items.find((entry) => entry.filename === filename);
    const mode = state.credentialMode;
    return [
        ['detail', '查看详情'],
        ['test', '测试可用性'],
        ['verify', '重新校验 Project ID'],
        ['email', '获取邮箱'],
        ...(Array.isArray(item?.error_codes) && item.error_codes.length ? [['errors', '查看错误记录']] : []),
        ...(mode === 'antigravity' ? [['quota', '查看额度'], [item?.enable_credit ? 'disable_credit' : 'enable_credit', item?.enable_credit ? '关闭 Credit' : '开启 Credit']] : [['preview', '配置 Preview 通道']]),
        ['download', '下载 JSON'],
        ['separator', ''],
        [item?.disabled ? 'enable' : 'disable', item?.disabled ? '启用凭证' : '停用凭证'],
        ['delete', '删除凭证'],
    ];
}

function showCredentialMenu(button, filename) {
    $('.action-popover')?.remove();
    const popover = document.createElement('div');
    popover.className = 'action-popover';
    popover.innerHTML = credentialActionItems(filename).map(([action, label]) => action === 'separator'
        ? '<hr>'
        : `<button data-credential-action="${action}" data-filename="${escapeHtml(filename)}" class="${action === 'delete' ? 'danger-text' : ''}">${escapeHtml(label)}<span>›</span></button>`).join('');
    document.body.append(popover);
    const rect = button.getBoundingClientRect();
    const left = Math.min(rect.right - 190, window.innerWidth - 200);
    const estimatedHeight = 390;
    const top = rect.bottom + estimatedHeight > window.innerHeight ? Math.max(10, rect.top - estimatedHeight) : rect.bottom + 5;
    popover.style.left = `${Math.max(10, left)}px`;
    popover.style.top = `${top}px`;
}

async function runCredentialAction(action, filename) {
    $('.action-popover')?.remove();
    const mode = state.credentialMode;
    const suffix = `?mode=${mode}`;
    if (action === 'detail') return showCredentialDetail(filename, mode);
    if (action === 'quota') return showQuota(filename, true);
    if (action === 'errors') return showCredentialErrors(filename, mode);
    if (action === 'download') {
        await downloadAuthenticated(`/creds/download/${encodeURIComponent(filename)}${suffix}`, filename);
        return;
    }
    if (action === 'delete' && !confirm(`确定删除 ${filename}？此操作无法撤销。`)) return;
    const endpointMap = {
        test: `/creds/test/${encodeURIComponent(filename)}${suffix}`,
        verify: `/creds/verify-project/${encodeURIComponent(filename)}${suffix}`,
        email: `/creds/fetch-email/${encodeURIComponent(filename)}${suffix}`,
        preview: `/creds/configure-preview/${encodeURIComponent(filename)}${suffix}`,
    };
    if (endpointMap[action]) {
        try {
            const result = await api(endpointMap[action], { method: 'POST' });
            toast(result.message || '操作完成');
        } catch (error) {
            if (action === 'test') await loadCredentials();
            throw error;
        }
    } else {
        const result = await api(`/creds/action${suffix}`, {
            method: 'POST',
            body: JSON.stringify({ filename, action }),
        });
        toast(result.message || '操作完成');
    }
    await loadCredentials();
}

async function showCredentialDetail(filename, mode) {
    openModal(filename, 'CREDENTIAL DETAIL', '<div class="loading">读取详情…</div>');
    try {
        const detail = await api(`/creds/detail/${encodeURIComponent(filename)}?mode=${mode}`);
        const projectId = detail.content?.project_id || '—';
        $('#modalBody').innerHTML = `<div class="detail-grid">
            <div class="detail-item"><span>邮箱</span><b>${escapeHtml(detail.user_email || '未获取')}</b></div>
            <div class="detail-item"><span>Project ID</span><b>${escapeHtml(projectId)}</b></div>
            <div class="detail-item"><span>存储后端</span><b>${escapeHtml(detail.backend_type)}</b></div>
            <div class="detail-item"><span>状态</span><b>${detail.status?.disabled ? '已停用' : '已启用'}</b></div>
        </div><p class="eyebrow" style="margin-top:20px">RAW CREDENTIAL</p><pre class="json-view">${escapeHtml(JSON.stringify(detail.content, null, 2))}</pre>`;
    } catch (error) {
        $('#modalBody').innerHTML = `<div class="empty-state"><h3>无法读取详情</h3><p>${escapeHtml(error.message)}</p></div>`;
    }
}

async function showCredentialErrors(filename, mode) {
    openModal('错误记录', 'CREDENTIAL DIAGNOSTICS', '<div class="loading">读取错误…</div>');
    try {
        const errors = await api(`/creds/errors/${encodeURIComponent(filename)}?mode=${mode}`);
        $('#modalBody').innerHTML = `<pre class="json-view">${escapeHtml(JSON.stringify(errors, null, 2))}</pre>`;
    } catch (error) {
        $('#modalBody').innerHTML = `<div class="empty-state"><h3>无法读取错误</h3><p>${escapeHtml(error.message)}</p></div>`;
    }
}

async function runBatch(action) {
    const filenames = [...state.selectedCredentials];
    if (!filenames.length) return;
    if (action === 'delete' && !confirm(`确定删除选中的 ${filenames.length} 个凭证？`)) return;
    const result = await api(`/creds/batch-action?mode=${state.credentialMode}`, {
        method: 'POST',
        body: JSON.stringify({ action, filenames }),
    });
    toast(result.message || '批量操作完成');
    state.selectedCredentials.clear();
    await loadCredentials();
}

function showCredentialTools() {
    openModal('凭证工具', 'BATCH UTILITIES', `<div class="detail-grid">
        <button class="quick-action detail-item" data-tool="refresh-emails"><span><b>刷新缺失邮箱</b><small>只处理尚无邮箱的凭证</small></span></button>
        <button class="quick-action detail-item" data-tool="deduplicate"><span><b>按邮箱去重</b><small>删除相同邮箱的重复项</small></span></button>
        <button class="quick-action detail-item" data-tool="download-all"><span><b>下载全部</b><small>打包当前类型的所有凭证</small></span></button>
    </div>`);
}

async function runCredentialTool(tool) {
    const mode = state.credentialMode;
    if (tool === 'download-all') {
        await downloadAuthenticated(`/creds/download-all?mode=${mode}`, `${mode}-credentials.zip`);
        closeModal();
        return;
    }
    if (tool === 'deduplicate' && !confirm('按邮箱去重会删除重复凭证，确定继续？')) return;
    const endpoint = tool === 'deduplicate' ? 'deduplicate-by-email' : 'refresh-all-emails';
    const result = await api(`/creds/${endpoint}?mode=${mode}`, { method: 'POST' });
    toast(result.message || '操作完成', 'success', 6500);
    closeModal();
    await loadCredentials();
}

async function listAllCredentials(mode) {
    const items = [];
    let offset = 0;
    while (true) {
        const response = await api(credentialQuery(mode, { offset, limit: 1000 }));
        items.push(...response.items);
        if (!response.has_more) break;
        offset += response.limit;
    }
    return items;
}

async function loadQuotaPage(force = false) {
    const select = $('#quotaCredential');
    if (force || !state.quotaCredentials.length) {
        select.innerHTML = '<option>读取账号中…</option>';
        state.quotaCredentials = await listAllCredentials('antigravity');
        select.innerHTML = state.quotaCredentials.length
            ? state.quotaCredentials.map((item) => `<option value="${escapeHtml(item.filename)}">${escapeHtml(item.user_email || item.filename)}</option>`).join('')
            : '<option value="">暂无 Antigravity 凭证</option>';
    }
    if (select.value) await showQuota(select.value, false, force);
}

function quotaGroup(modelName) {
    const name = modelName.toLowerCase();
    if (name.includes('claude')) return 'Claude';
    if (name.includes('gemini')) return 'Gemini';
    if (name.includes('imagen') || name.includes('image')) return 'Image';
    return 'Other';
}

function summarizeQuotaPool(models, pool) {
    const entries = Object.entries(models || {}).filter(([name]) => {
        const isClaude = name.toLowerCase().includes('claude');
        return pool === 'Claude' ? isClaude : !isClaude;
    });
    if (!entries.length) return null;
    let remaining = 1;
    let resetTime = '—';
    let resetRaw = '';
    let resetMs = Number.POSITIVE_INFINITY;
    for (const [, quota] of entries) {
        remaining = Math.min(remaining, Math.max(0, Math.min(1, Number(quota?.remaining || 0))));
        const parsed = Date.parse(quota?.resetTimeRaw || '');
        if (Number.isFinite(parsed) && parsed < resetMs) {
            resetMs = parsed;
            resetRaw = quota.resetTimeRaw;
            resetTime = quota.resetTime || '—';
        } else if (!resetRaw && quota?.resetTime && resetTime === '—') {
            resetTime = quota.resetTime;
        }
    }
    return { remaining, resetRaw, resetTime };
}

function resetCountdown(raw, fallback) {
    const resetAt = Date.parse(raw || '');
    if (!Number.isFinite(resetAt)) return fallback && fallback !== 'N/A' ? `R: ${fallback}` : 'R: —';
    const minutes = Math.max(0, Math.ceil((resetAt - Date.now()) / 60000));
    if (minutes === 0) return '即将重置';
    const days = Math.floor(minutes / 1440);
    const hours = Math.floor((minutes % 1440) / 60);
    const mins = minutes % 60;
    const parts = [];
    if (days) parts.push(`${days}d`);
    if (hours) parts.push(`${hours}h`);
    if (!days && mins) parts.push(`${mins}m`);
    return `R: ${parts.join(' ')}`;
}

function inlineQuotaRow(label, summary) {
    if (!summary) return `<div class="inline-quota-row is-empty"><span>${label}</span><i><b></b></i><strong>—</strong><small>暂无</small></div>`;
    const percent = Math.round(summary.remaining * 100);
    return `<div class="inline-quota-row" title="${label} · ${percent}% · 重置 ${escapeHtml(summary.resetTime)}"><span>${label}</span><i style="--quota-color:${quotaColor(summary.remaining)}"><b style="width:${percent}%"></b></i><strong>${percent}%</strong><small>${escapeHtml(resetCountdown(summary.resetRaw, summary.resetTime))}</small></div>`;
}

function renderInlineQuota(target, data) {
    target.classList.remove('is-loading', 'has-error');
    target.innerHTML = inlineQuotaRow('Gemini', summarizeQuotaPool(data.models, 'Gemini'))
        + inlineQuotaRow('Claude', summarizeQuotaPool(data.models, 'Claude'));
}

async function getQuotaData(filename, force = false) {
    if (!force && state.quotaCache.has(filename)) return state.quotaCache.get(filename);
    if (!force && state.quotaRequests.has(filename)) return state.quotaRequests.get(filename);
    const request = api(`/creds/quota/${encodeURIComponent(filename)}?mode=antigravity`)
        .then((data) => {
            state.quotaCache.set(filename, data);
            return data;
        })
        .finally(() => state.quotaRequests.delete(filename));
    state.quotaRequests.set(filename, request);
    return request;
}

async function loadInlineQuota(target, force = false) {
    const filename = target.dataset.inlineQuota;
    try {
        renderInlineQuota(target, await getQuotaData(filename, force));
    } catch (error) {
        target.classList.remove('is-loading');
        target.classList.add('has-error');
        target.innerHTML = `<span>额度读取失败</span><small>${escapeHtml(error.message)}</small>`;
    }
}

function observeInlineQuotas() {
    const targets = $$('[data-inline-quota]', $('#credentialTable'));
    const pending = targets.filter((target) => {
        const cached = state.quotaCache.get(target.dataset.inlineQuota);
        if (cached) renderInlineQuota(target, cached);
        return !cached;
    });
    if (!pending.length) return;
    if (!('IntersectionObserver' in window)) {
        pending.forEach((target) => loadInlineQuota(target));
        return;
    }
    state.quotaObserver = new IntersectionObserver((entries, observer) => {
        for (const entry of entries) {
            if (!entry.isIntersecting) continue;
            observer.unobserve(entry.target);
            loadInlineQuota(entry.target);
        }
    }, { rootMargin: '180px 0px' });
    pending.forEach((target) => state.quotaObserver.observe(target));
}

function quotaColor(value) {
    if (value <= .2) return 'var(--red)';
    if (value <= .5) return 'var(--amber)';
    return 'var(--green)';
}

function quotaMarkup(data) {
    const entries = Object.entries(data.models || {}).sort(([a], [b]) => a.localeCompare(b));
    if (!entries.length) return '<div class="empty-state"><h3>没有可展示的模型额度</h3><p>上游没有返回 quotaInfo。</p></div>';
    const average = entries.reduce((sum, [, item]) => sum + Number(item.remaining || 0), 0) / entries.length;
    const groups = Object.groupBy
        ? Object.groupBy(entries, ([name]) => quotaGroup(name))
        : entries.reduce((result, entry) => { (result[quotaGroup(entry[0])] ||= []).push(entry); return result; }, {});
    const groupMarkup = Object.entries(groups).map(([group, models]) => `<section class="quota-group">
        <div class="quota-group-head"><h3>${escapeHtml(group)}</h3><span class="badge">${models.length} models</span></div>
        <div class="quota-models">${models.map(([name, item]) => {
            const remaining = Math.max(0, Math.min(1, Number(item.remaining || 0)));
            const percent = Math.round(remaining * 100);
            return `<div class="quota-model"><div class="quota-model-top"><b title="${escapeHtml(name)}">${escapeHtml(name)}</b><strong>${percent}%</strong></div><div class="quota-bar" style="--quota-color:${quotaColor(remaining)}"><i style="width:${percent}%"></i></div><div class="quota-model-bottom"><span>剩余额度</span><span>重置 ${escapeHtml(item.resetTime || 'N/A')}</span></div></div>`;
        }).join('')}</div></section>`).join('');
    return `<div class="quota-summary"><article class="quota-ring-card"><div class="quota-ring" style="--value:${Math.round(average * 100)}"><strong>${Math.round(average * 100)}%</strong></div><p>${entries.length} 个模型的平均余量</p></article><div class="quota-groups">${groupMarkup}</div></div>`;
}

async function showQuota(filename, modal = false, force = false) {
    const target = modal ? null : $('#quotaContent');
    if (modal) openModal('模型额度', 'ANTIGRAVITY QUOTA', '<div class="loading">向上游查询额度…</div>');
    else target.innerHTML = '<div class="loading">向上游查询额度…</div>';
    try {
        const data = await getQuotaData(filename, force);
        const html = quotaMarkup(data);
        if (modal) $('#modalBody').innerHTML = html;
        else {
            target.className = 'quota-content';
            target.innerHTML = html;
            const account = state.quotaCredentials.find((item) => item.filename === filename);
            $('#quotaAccountMeta').innerHTML = account ? `<span class="badge ${account.disabled ? 'bad' : 'good'}">${account.disabled ? 'Disabled' : 'Enabled'}</span><span class="badge">${escapeHtml(account.tier || 'pro')}</span>` : '';
        }
    } catch (error) {
        const html = `<div class="empty-state"><h3>额度读取失败</h3><p>${escapeHtml(error.message)}</p></div>`;
        if (modal) $('#modalBody').innerHTML = html;
        else target.innerHTML = html;
    }
}

async function startOAuth() {
    const button = $('#startOAuth');
    const authWindow = window.open('', '_blank');
    if (authWindow) authWindow.opener = null;
    setBusy(button, true, '生成链接…');
    try {
        state.oauthProjectId = $('#oauthProjectId').value.trim();
        const result = await api('/auth/start', {
            method: 'POST',
            body: JSON.stringify({ project_id: state.oauthProjectId || null, mode: state.connectMode }),
        });
        if (authWindow) authWindow.location.replace(result.auth_url);
        else window.open(result.auth_url, '_blank', 'noopener,noreferrer');
        $('#oauthStepTwo').classList.remove('hidden');
        toast('授权链接已在新窗口打开');
    } catch (error) {
        authWindow?.close();
        throw error;
    } finally {
        setBusy(button, false);
    }
}

async function finishOAuthWithWait() {
    const button = $('#waitOAuth');
    setBusy(button, true, '等待回调…');
    try {
        const result = await api('/auth/callback', {
            method: 'POST',
            body: JSON.stringify({ project_id: state.oauthProjectId || null, mode: state.connectMode }),
        });
        toast(result.message || '认证成功');
        $('#oauthStepTwo').classList.add('hidden');
    } finally {
        setBusy(button, false);
    }
}

async function finishOAuthWithUrl() {
    const button = $('#submitCallback');
    const callbackUrl = $('#callbackUrl').value.trim();
    if (!callbackUrl) return toast('请先粘贴回调 URL', 'error');
    setBusy(button, true, '提交中…');
    try {
        const result = await api('/auth/callback-url', {
            method: 'POST',
            body: JSON.stringify({ callback_url: callbackUrl, project_id: state.oauthProjectId || null, mode: state.connectMode }),
        });
        toast(result.message || '认证成功');
        $('#callbackUrl').value = '';
        $('#oauthStepTwo').classList.add('hidden');
    } finally {
        setBusy(button, false);
    }
}

function updateFileSelection() {
    const files = [...$('#credentialFiles').files];
    $('#fileSelection').textContent = files.length
        ? `${files.length} 个文件 · ${files.map((file) => file.name).slice(0, 3).join('、')}${files.length > 3 ? '…' : ''}`
        : '尚未选择文件';
    $('#uploadCredentials').disabled = files.length === 0;
}

async function uploadCredentials() {
    const button = $('#uploadCredentials');
    const files = [...$('#credentialFiles').files];
    if (!files.length) return;
    const form = new FormData();
    files.forEach((file) => form.append('files', file));
    setBusy(button, true, '导入中…');
    try {
        const result = await api(`/creds/upload?mode=${state.connectMode}`, { method: 'POST', body: form });
        const box = $('#uploadResult');
        box.classList.remove('hidden');
        box.innerHTML = `<b>${escapeHtml(result.message)}</b><br>${result.results.map((item) => `${escapeHtml(item.filename)} · ${escapeHtml(item.status)}`).join('<br>')}`;
        toast(result.message || '导入完成');
        $('#credentialFiles').value = '';
        updateFileSelection();
    } finally {
        setBusy(button, false);
    }
}

const settingSections = [
    {
        title: '服务与访问',
        description: '监听地址、端口和控制面/API 的访问凭据。监听项保存后通常需要重启服务生效。',
        fields: [
            ['host', '监听地址', 'text'],
            ['port', '监听端口', 'number'],
            ['api_password', 'API 访问密码', 'password'],
            ['panel_password', '控制台密码', 'password'],
            ['credentials_dir', '凭证目录', 'text'],
        ],
    },
    {
        title: '上游与代理',
        description: '代理所有外部请求，或替换 Google 服务端点。使用 Mihomo 接管时会暂时覆盖“出站代理”。',
        fields: [
            ['proxy', '出站代理', 'text'],
            ['code_assist_endpoint', 'Code Assist Endpoint', 'url'],
            ['antigravity_api_url', 'Antigravity Endpoint', 'url'],
            ['oauth_proxy_url', 'OAuth Endpoint', 'url'],
            ['googleapis_proxy_url', 'Google APIs Endpoint', 'url'],
            ['resource_manager_api_url', 'Resource Manager Endpoint', 'url'],
            ['service_usage_api_url', 'Service Usage Endpoint', 'url'],
        ],
    },
    {
        title: '重试与容错',
        description: '控制上游限流后的重试、流式抗截断，以及哪些错误会自动停用凭证。',
        fields: [
            ['retry_429_enabled', '启用 429 重试', 'boolean', '遇到限流时按配置重试。'],
            ['retry_429_max_retries', '429 最大重试次数', 'number'],
            ['retry_429_interval', '429 重试间隔（秒）', 'number-step'],
            ['anti_truncation_max_attempts', '抗截断最大尝试次数', 'number'],
            ['auto_ban_enabled', '启用自动停用', 'boolean', '凭证返回指定错误码时自动停用。'],
            ['auto_ban_error_codes', '自动停用错误码', 'list'],
        ],
    },
    {
        title: '响应行为',
        description: '控制消息兼容、思考内容和 Antigravity 的响应转换方式。',
        fields: [
            ['compatibility_mode_enabled', '兼容模式', 'boolean', '将 system 消息转换为 user，提高部分客户端兼容性。'],
            ['return_thoughts_to_frontend', '返回思考内容', 'boolean', '在响应中保留上游返回的思考片段。'],
            ['antigravity_stream2nostream', '流式聚合非流响应', 'boolean', '用上游流式接口收集完整的非流式响应。'],
            ['antigravity_switch_credential_enabled', '重试时切换凭证', 'boolean', 'Antigravity 请求重试时允许换用其他凭证。'],
            ['keepalive_url', '保活 URL', 'url'],
            ['keepalive_interval', '保活间隔（秒）', 'number'],
        ],
    },
];

function renderField([key, label, type, description], value, locked) {
    if (type === 'boolean') {
        return `<label class="toggle-field"><span><b>${escapeHtml(label)}</b><small class="${locked ? 'locked-note' : ''}">${escapeHtml(locked ? '由环境变量锁定' : (description || ''))}</small></span><span class="switch"><input name="${key}" type="checkbox" ${value ? 'checked' : ''} ${locked ? 'disabled' : ''}><i></i></span></label>`;
    }
    const inputType = type === 'password' ? 'password' : type === 'url' ? 'url' : type.startsWith('number') ? 'number' : 'text';
    const displayValue = type === 'list' && Array.isArray(value) ? value.join(', ') : value ?? '';
    return `<label class="field"><span>${escapeHtml(label)}${locked ? ' · <small class="locked-note">环境变量锁定</small>' : ''}</span><input name="${key}" type="${inputType}" value="${escapeHtml(displayValue)}" ${type === 'number-step' ? 'step="0.01"' : ''} ${locked ? 'disabled' : ''}></label>`;
}

async function loadSettings() {
    const form = $('#settingsForm');
    form.innerHTML = '<div class="loading">读取系统配置…</div>';
    const result = await api('/config/get');
    const config = result.config || {};
    const locked = new Set(result.env_locked || []);
    form.dataset.lockedKeys = JSON.stringify([...locked]);
    form.innerHTML = settingSections.map((section) => `<section class="panel settings-section"><div class="settings-section-head"><h3>${escapeHtml(section.title)}</h3><p>${escapeHtml(section.description)}</p></div><div class="settings-fields">${section.fields.map((field) => renderField(field, config[field[0]], locked.has(field[0]))).join('')}</div></section>`).join('');
}

async function saveSystemSettings() {
    const button = $('#saveSettings');
    const form = $('#settingsForm');
    const locked = new Set(JSON.parse(form.dataset.lockedKeys || '[]'));
    const config = {};
    for (const section of settingSections) {
        for (const [key, , type] of section.fields) {
            if (locked.has(key)) continue;
            const input = form.elements[key];
            if (!input) continue;
            if (type === 'boolean') config[key] = input.checked;
            else if (type === 'number') config[key] = Number.parseInt(input.value, 10);
            else if (type === 'number-step') config[key] = Number.parseFloat(input.value);
            else if (type === 'list') config[key] = input.value.split(',').map((value) => Number.parseInt(value.trim(), 10)).filter(Number.isFinite);
            else config[key] = input.value.trim();
        }
    }
    setBusy(button, true, '保存中…');
    try {
        const result = await api('/config/save', { method: 'POST', body: JSON.stringify({ config }) });
        toast(result.message || '配置已保存');
        if (config.panel_password && config.panel_password !== state.token) {
            state.token = config.panel_password;
            sessionStorage.setItem('gcli-control-token', state.token);
        }
        await loadSettings();
    } finally {
        setBusy(button, false);
    }
}

function setLogStatus(label, online = false) {
    $('#logStatus').textContent = label;
    $('#logStatusDot').style.background = online ? 'var(--green)' : 'var(--red)';
}

function appendLog(line) {
    const output = $('#logOutput');
    if (output.textContent === '等待日志连接…') output.textContent = '';
    const lines = `${output.textContent}${output.textContent ? '\n' : ''}${line}`.split('\n');
    output.textContent = lines.slice(-1000).join('\n');
    if (!state.logPaused) output.scrollTop = output.scrollHeight;
}

function connectLogs() {
    if (state.logSocket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(state.logSocket.readyState)) return Promise.resolve();
    setLogStatus('连接中…');
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const socket = new WebSocket(`${protocol}//${location.host}/logs/stream?token=${encodeURIComponent(state.token)}`);
    state.logSocket = socket;
    socket.onopen = () => setLogStatus('实时连接', true);
    socket.onmessage = (event) => appendLog(event.data);
    socket.onerror = () => setLogStatus('连接异常');
    socket.onclose = () => {
        setLogStatus('连接已断开');
        if (state.page === 'logs') setTimeout(() => state.page === 'logs' && connectLogs(), 3000);
    };
    return Promise.resolve();
}

function disconnectLogs() {
    if (state.logSocket) {
        state.logSocket.onclose = null;
        state.logSocket.close();
        state.logSocket = null;
    }
}

async function clearLogs() {
    if (!confirm('确定清空当前日志文件？')) return;
    const result = await api('/logs/clear', { method: 'POST' });
    $('#logOutput').textContent = '--- 日志已清空 ---';
    toast(result.message || '日志已清空');
}

async function loadAbout(force = false) {
    if (!force && $('#versionInfo').dataset.loaded) return;
    const result = await api(`/version/info${force ? '?check_update=true' : ''}`);
    $('#versionInfo').dataset.loaded = 'true';
    $('#versionInfo').textContent = result.success
        ? `${result.version} · ${result.date || 'local build'}${result.has_update ? ' · 有新版本' : ''}`
        : '开发版本';
}

const mihomoFields = [
    ['enabled', '启用 Mihomo', 'boolean', '允许控制台启动代理核心。'],
    ['auto_start', '服务启动时自动运行', 'boolean', '恢复上次 Profile 与节点选择。'],
    ['set_as_project_proxy', '接管项目出站代理', 'boolean', '运行时让 gcli2api 的上游请求走混合端口。'],
    ['mixed_port', 'Mixed Port', 'number'],
    ['controller_host', 'Controller Host', 'text'],
    ['controller_port', 'Controller Port', 'number'],
    ['profile', '默认 Profile', 'text'],
    ['bin_path', 'Mihomo 核心路径', 'text'],
    ['health_check_url', '延迟测试 URL', 'url'],
    ['startup_timeout_ms', '启动等待（毫秒）', 'number'],
    ['download_timeout_ms', '订阅下载等待（毫秒）', 'number'],
    ['max_subscription_size_bytes', '订阅大小上限（bytes）', 'number'],
    ['secret', 'Controller Secret', 'password'],
];

async function loadMihomo() {
    $('#mihomoStatus').innerHTML = '<div class="loading">读取运行状态…</div>';
    const result = await api('/mihomo/status');
    state.mihomo = result.data;
    renderMihomoStatus();
    renderProfiles();
    renderMihomoConfig();
    $('#mihomoNavDot').classList.toggle('online', state.mihomo.running);
    if (state.mihomo.running) await loadProxies();
    else {
        state.proxies = {};
        $('#proxyWorkspace').className = 'proxy-workspace empty-state';
        $('#proxyWorkspace').innerHTML = '<p>启动 Mihomo 后显示策略组。</p>';
    }
}

function renderMihomoStatus() {
    const data = state.mihomo;
    if (!data) return;
    $('#mihomoStatus').innerHTML = `<div class="runtime-primary"><span class="runtime-orb ${data.running ? 'online' : ''}">${icon('bolt')}</span><span><b>${data.running ? 'Mihomo 正在运行' : 'Mihomo 已停止'}</b><small>${data.last_error ? escapeHtml(data.last_error) : (data.binary_available ? '核心已就绪' : '尚未找到核心文件')}</small></span></div>
        <div class="runtime-metric"><span>Profile</span><b>${escapeHtml(data.current_profile || '—')}</b></div>
        <div class="runtime-metric"><span>Mixed Port</span><b>${escapeHtml(data.mixed_port)}</b></div>
        <div class="runtime-metric"><span>Controller</span><b>${escapeHtml(data.controller)}</b></div>
        <div class="runtime-metric"><span>Project Proxy</span><b>${data.proxy_active ? 'Active' : 'Inactive'}</b></div>`;
    $('#mihomoStart').innerHTML = data.running ? `${icon('refresh')}重启` : `${icon('bolt')}启动`;
    $('#mihomoStop').disabled = !data.running;
}

function renderProfiles() {
    const profiles = state.mihomo?.profiles || [];
    $('#profileList').innerHTML = profiles.length ? profiles.map((profile) => {
        const active = profile.name === state.mihomo.current_profile;
        return `<div class="profile-item ${active ? 'active' : ''}" data-profile="${escapeHtml(profile.name)}"><b>${escapeHtml(profile.name)}</b><div class="profile-actions"><button data-profile-activate="${escapeHtml(profile.name)}">${active ? '当前' : '启用'}</button>${active ? '' : `<button data-profile-delete="${escapeHtml(profile.name)}" class="danger-text">删除</button>`}</div><small>${escapeHtml(profile.source || 'local')} · ${formatTime(profile.updated_at)}</small></div>`;
    }).join('') : '<div class="empty-state"><h3>还没有 Profile</h3><p>从订阅 URL 或 YAML 导入。</p></div>';
}

function renderMihomoConfig() {
    const form = $('#mihomoConfigForm');
    const data = state.mihomo || {};
    form.innerHTML = mihomoFields.map((field) => renderField(field, field[0] === 'secret' ? '' : data[field[0]], false)).join('') + '<div class="form-actions"><button class="button primary" type="submit">保存运行设置</button></div>';
}

async function saveMihomoConfig(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const config = {};
    for (const [key, , type] of mihomoFields) {
        const input = form.elements[key];
        if (type === 'boolean') config[key] = input.checked;
        else if (type === 'number') config[key] = Number.parseInt(input.value, 10);
        else if (key === 'secret' && !input.value) continue;
        else config[key] = input.value.trim();
    }
    const button = $('button[type="submit"]', form);
    setBusy(button, true, '保存中…');
    try {
        const result = await api('/mihomo/config', { method: 'PUT', body: JSON.stringify({ config }) });
        state.mihomo = result.data.status;
        renderMihomoStatus();
        renderProfiles();
        renderMihomoConfig();
        toast('Mihomo 配置已保存');
        if (state.mihomo.running) await loadProxies();
    } finally {
        setBusy(button, false);
    }
}

async function toggleMihomo() {
    const button = $('#mihomoStart');
    const action = state.mihomo?.running ? 'restart' : 'start';
    setBusy(button, true, action === 'restart' ? '重启中…' : '启动中…');
    try {
        const result = await api(`/mihomo/${action}`, { method: 'POST', body: JSON.stringify({}) });
        state.mihomo = result.data;
        renderMihomoStatus();
        renderProfiles();
        $('#mihomoNavDot').classList.toggle('online', state.mihomo.running);
        await loadProxies();
        toast(action === 'restart' ? 'Mihomo 已重启' : 'Mihomo 已启动');
    } finally {
        setBusy(button, false);
    }
}

async function stopMihomo() {
    const button = $('#mihomoStop');
    setBusy(button, true, '停止中…');
    try {
        const result = await api('/mihomo/stop', { method: 'POST' });
        state.mihomo = result.data;
        renderMihomoStatus();
        $('#mihomoNavDot').classList.remove('online');
        $('#proxyWorkspace').className = 'proxy-workspace empty-state';
        $('#proxyWorkspace').innerHTML = '<p>启动 Mihomo 后显示策略组。</p>';
        toast('Mihomo 已停止');
    } finally {
        setBusy(button, false);
    }
}

function openProfileImport() {
    openModal('导入 Profile', 'MIHOMO PROFILE', `<div class="segment profile-import-tabs"><button class="active" data-import-tab="url">订阅 URL</button><button data-import-tab="yaml">YAML</button></div>
        <form id="profileImportForm" class="profile-import-pane"><input type="hidden" name="type" value="url"><label class="field"><span>Profile 名称 <small>可选</small></span><input name="name" placeholder="例如 HK Premium"></label><label class="field import-url-field"><span>订阅 URL</span><input name="url" type="url" placeholder="https://..."></label><label class="field import-yaml-field hidden"><span>YAML 内容</span><textarea name="content" rows="12" placeholder="proxies: ..."></textarea></label><button class="button primary" type="submit">导入并设为当前 Profile</button></form>`);
}

function switchProfileImportTab(type) {
    $$('[data-import-tab]', $('#modalBody')).forEach((button) => button.classList.toggle('active', button.dataset.importTab === type));
    $('[name="type"]', $('#modalBody')).value = type;
    $('.import-url-field', $('#modalBody')).classList.toggle('hidden', type !== 'url');
    $('.import-yaml-field', $('#modalBody')).classList.toggle('hidden', type !== 'yaml');
}

async function importProfile(event) {
    event.preventDefault();
    const form = event.target;
    const type = form.elements.type.value;
    const body = type === 'url'
        ? { name: form.elements.name.value, url: form.elements.url.value }
        : { name: form.elements.name.value, content: form.elements.content.value };
    const button = $('button[type="submit"]', form);
    setBusy(button, true, '导入中…');
    try {
        const result = await api(`/mihomo/profiles/import-${type}`, { method: 'POST', body: JSON.stringify(body) });
        state.mihomo = result.data.status;
        closeModal();
        renderMihomoStatus();
        renderProfiles();
        renderMihomoConfig();
        if (state.mihomo.running) await loadProxies();
        toast(`Profile ${result.data.profile.name} 已导入`);
    } finally {
        setBusy(button, false);
    }
}

async function activateProfile(name) {
    if (name === state.mihomo?.current_profile && state.mihomo.running) return;
    const result = state.mihomo?.running
        ? await api('/mihomo/restart', { method: 'POST', body: JSON.stringify({ profile: name }) })
        : await api('/mihomo/config', { method: 'PUT', body: JSON.stringify({ config: { profile: name } }) });
    state.mihomo = result.data.status || result.data;
    renderMihomoStatus();
    renderProfiles();
    renderMihomoConfig();
    await loadProxies();
    toast(`已切换到 ${name}`);
}

async function deleteProfile(name) {
    if (!confirm(`确定删除 Profile “${name}”？`)) return;
    await api(`/mihomo/profiles/${encodeURIComponent(name)}`, { method: 'DELETE' });
    toast(`Profile ${name} 已删除`);
    await loadMihomo();
}

const groupTypes = new Set(['Selector', 'URLTest', 'Fallback', 'LoadBalance', 'Relay', 'Smart']);

async function loadProxies() {
    const workspace = $('#proxyWorkspace');
    workspace.className = 'proxy-workspace loading';
    workspace.textContent = '读取策略组…';
    try {
        const result = await api('/mihomo/proxies');
        state.proxies = result.data?.proxies || {};
        const groups = Object.values(state.proxies).filter((proxy) => groupTypes.has(proxy.type));
        if (!groups.some((group) => group.name === state.activeGroup)) state.activeGroup = groups[0]?.name || '';
        renderProxyWorkspace();
    } catch (error) {
        workspace.className = 'proxy-workspace empty-state';
        workspace.innerHTML = `<h3>策略组读取失败</h3><p>${escapeHtml(error.message)}</p>`;
    }
}

function nodeDelay(node) {
    const history = Array.isArray(node?.history) ? node.history : [];
    const last = history[history.length - 1];
    return Number(last?.delay || 0);
}

function delayClass(delay) {
    if (!delay) return '';
    if (delay < 250) return 'good';
    if (delay < 700) return 'medium';
    return 'bad';
}

function renderProxyWorkspace() {
    const workspace = $('#proxyWorkspace');
    const groups = Object.values(state.proxies).filter((proxy) => groupTypes.has(proxy.type));
    if (!groups.length) {
        workspace.className = 'proxy-workspace empty-state';
        workspace.innerHTML = '<h3>没有策略组</h3><p>当前 Profile 未返回可选择的 proxy-groups。</p>';
        return;
    }
    const group = state.proxies[state.activeGroup] || groups[0];
    state.activeGroup = group.name;
    const query = $('#nodeSearch').value.trim().toLowerCase();
    const names = (group.all || []).filter((name) => !query || name.toLowerCase().includes(query));
    const selectable = group.type === 'Selector' || group.name === 'GLOBAL';
    workspace.className = 'proxy-workspace';
    workspace.innerHTML = `<div class="group-tabs">${groups.map((item) => `<button class="group-tab ${item.name === group.name ? 'active' : ''}" data-group="${escapeHtml(item.name)}">${escapeHtml(item.name)}</button>`).join('')}</div><div class="group-context"><span>${escapeHtml(group.type)} · ${names.length} nodes</span><span>当前：${escapeHtml(group.now || '—')}</span></div><div class="node-grid">${names.map((name) => {
        const node = state.proxies[name];
        const delay = nodeDelay(node);
        return `<button class="node-card ${name === group.now ? 'active' : ''}" data-node="${escapeHtml(name)}" ${selectable ? '' : 'data-readonly="true"'}><span class="node-card-top"><i></i><b title="${escapeHtml(name)}">${escapeHtml(name)}</b></span><span class="node-meta"><span>${escapeHtml(node?.type || (groupTypes.has(node?.type) ? 'Group' : 'Proxy'))}</span><span class="delay ${delayClass(delay)}" data-delay-node="${escapeHtml(name)}">${delay ? `${delay} ms` : '测速'}</span></span></button>`;
    }).join('')}</div>`;
}

async function selectNode(name) {
    const group = state.proxies[state.activeGroup];
    if (!group || (!['Selector'].includes(group.type) && group.name !== 'GLOBAL')) return toast('当前策略组由 Mihomo 自动选择', 'error');
    await api(`/mihomo/proxies/${encodeURIComponent(group.name)}`, { method: 'PUT', body: JSON.stringify({ name }) });
    group.now = name;
    renderProxyWorkspace();
    toast(`${group.name} → ${name}`);
}

async function testNodeDelay(name) {
    const result = await api(`/mihomo/proxies/${encodeURIComponent(name)}/delay`);
    const delay = Number(result.data?.delay || 0);
    const node = state.proxies[name];
    if (node) {
        node.history ||= [];
        node.history.push({ delay, time: new Date().toISOString() });
    }
    renderProxyWorkspace();
    toast(delay ? `${name}: ${delay} ms` : `${name}: 测速失败`, delay ? 'success' : 'error');
}

function bindEvents() {
    $('#loginForm').addEventListener('submit', async (event) => {
        event.preventDefault();
        const button = $('button[type="submit"]', event.currentTarget);
        $('#loginError').textContent = '';
        setBusy(button, true, '验证中…');
        try { await login($('#loginPassword').value); }
        catch (error) { $('#loginError').textContent = error.message; }
        finally { setBusy(button, false); }
    });
    $('#logoutButton').addEventListener('click', () => logout());
    $('#menuButton').addEventListener('click', () => { $('#sidebar').classList.add('open'); $('#sidebarScrim').classList.add('open'); });
    $('#sidebarScrim').addEventListener('click', closeSidebar);
    $$('.nav-item').forEach((button) => button.addEventListener('click', () => navigate(button.dataset.page)));
    document.addEventListener('click', (event) => {
        const go = event.target.closest('[data-go]');
        if (go) navigate(go.dataset.go);
        const copy = event.target.closest('[data-copy]');
        if (copy) copyText(copy.dataset.copy);
        if (event.target.closest('[data-close-modal]')) closeModal();
        if (!event.target.closest('.row-menu') && !event.target.closest('.action-popover')) $('.action-popover')?.remove();
    });
    document.addEventListener('keydown', (event) => { if (event.key === 'Escape') { closeModal(); $('.action-popover')?.remove(); closeSidebar(); } });
    $('#globalRefresh').addEventListener('click', () => loadPage(state.page, true).catch((error) => toast(error.message, 'error')));

    $('#credentialMode').addEventListener('click', (event) => {
        const button = event.target.closest('[data-mode]');
        if (!button) return;
        state.credentialMode = button.dataset.mode;
        state.credentialOffset = 0;
        state.selectedCredentials.clear();
        $$('button', $('#credentialMode')).forEach((item) => item.classList.toggle('active', item === button));
        loadCredentials().catch((error) => toast(error.message, 'error'));
    });
    $('#credentialSearch').addEventListener('input', renderCredentials);
    $('#statusFilter').addEventListener('change', () => { state.credentialOffset = 0; loadCredentials().catch((error) => toast(error.message, 'error')); });
    $('#tierFilter').addEventListener('change', () => { state.credentialOffset = 0; loadCredentials().catch((error) => toast(error.message, 'error')); });
    $('#prevPage').addEventListener('click', () => { state.credentialOffset = Math.max(0, state.credentialOffset - state.credentialLimit); loadCredentials(); });
    $('#nextPage').addEventListener('click', () => { state.credentialOffset += state.credentialLimit; loadCredentials(); });
    $('#credentialTools').addEventListener('click', showCredentialTools);
    $('#clearSelection').addEventListener('click', () => { state.selectedCredentials.clear(); renderCredentials(); });
    $('#bulkBar').addEventListener('click', (event) => { const button = event.target.closest('[data-bulk]'); if (button) runBatch(button.dataset.bulk).catch((error) => toast(error.message, 'error')); });
    $('#credentialTable').addEventListener('change', (event) => {
        if (event.target.id === 'selectPage') {
            state.credentialResponse.items.forEach((item) => event.target.checked ? state.selectedCredentials.add(item.filename) : state.selectedCredentials.delete(item.filename));
            renderCredentials();
            return;
        }
        if (event.target.classList.contains('credential-check')) {
            const filename = event.target.closest('[data-filename]').dataset.filename;
            event.target.checked ? state.selectedCredentials.add(filename) : state.selectedCredentials.delete(filename);
            renderBulkBar();
        }
    });
    $('#credentialTable').addEventListener('click', (event) => {
        const quota = event.target.closest('[data-inline-quota]');
        if (quota) {
            showQuota(quota.dataset.inlineQuota, true).catch((error) => toast(error.message, 'error'));
            return;
        }
        const button = event.target.closest('[data-menu]');
        if (button) showCredentialMenu(button, button.dataset.menu);
    });
    $('#credentialTable').addEventListener('keydown', (event) => {
        const quota = event.target.closest('[data-inline-quota]');
        if (quota && ['Enter', ' '].includes(event.key)) {
            event.preventDefault();
            showQuota(quota.dataset.inlineQuota, true).catch((error) => toast(error.message, 'error'));
        }
    });
    document.body.addEventListener('click', (event) => {
        const action = event.target.closest('[data-credential-action]');
        if (action) runCredentialAction(action.dataset.credentialAction, action.dataset.filename).catch((error) => toast(error.message, 'error', 6500));
        const tool = event.target.closest('[data-tool]');
        if (tool) runCredentialTool(tool.dataset.tool).catch((error) => toast(error.message, 'error'));
    });

    $('#quotaCredential').addEventListener('change', (event) => event.target.value && showQuota(event.target.value));
    $('#refreshQuota').addEventListener('click', () => $('#quotaCredential').value && showQuota($('#quotaCredential').value, false, true));
    $('#connectMode').addEventListener('click', (event) => {
        const button = event.target.closest('[data-mode]');
        if (!button) return;
        state.connectMode = button.dataset.mode;
        $$('button', $('#connectMode')).forEach((item) => item.classList.toggle('active', item === button));
    });
    $('#startOAuth').addEventListener('click', () => startOAuth().catch((error) => toast(error.message, 'error', 6500)));
    $('#waitOAuth').addEventListener('click', () => finishOAuthWithWait().catch((error) => toast(error.message, 'error', 6500)));
    $('#submitCallback').addEventListener('click', () => finishOAuthWithUrl().catch((error) => toast(error.message, 'error', 6500)));
    $('#credentialFiles').addEventListener('change', updateFileSelection);
    $('#uploadCredentials').addEventListener('click', () => uploadCredentials().catch((error) => toast(error.message, 'error', 6500)));
    for (const eventName of ['dragenter', 'dragover']) $('#uploadDropzone').addEventListener(eventName, (event) => { event.preventDefault(); event.currentTarget.classList.add('dragging'); });
    for (const eventName of ['dragleave', 'drop']) $('#uploadDropzone').addEventListener(eventName, (event) => { event.preventDefault(); event.currentTarget.classList.remove('dragging'); });
    $('#uploadDropzone').addEventListener('drop', (event) => { $('#credentialFiles').files = event.dataTransfer.files; updateFileSelection(); });

    $('#saveSettings').addEventListener('click', () => saveSystemSettings().catch((error) => toast(error.message, 'error', 6500)));
    $('#downloadLogs').addEventListener('click', () => downloadAuthenticated('/logs/download', 'gcli2api-logs.txt').catch((error) => toast(error.message, 'error')));
    $('#clearLogs').addEventListener('click', () => clearLogs().catch((error) => toast(error.message, 'error')));
    $('#pauseLogs').addEventListener('click', (event) => { state.logPaused = !state.logPaused; event.currentTarget.textContent = state.logPaused ? '继续滚动' : '暂停滚动'; });

    $('#mihomoStart').addEventListener('click', () => toggleMihomo().catch((error) => toast(error.message, 'error', 7000)));
    $('#mihomoStop').addEventListener('click', () => stopMihomo().catch((error) => toast(error.message, 'error')));
    $('#openProfileImport').addEventListener('click', openProfileImport);
    $('#mihomoConfigForm').addEventListener('submit', (event) => saveMihomoConfig(event).catch((error) => toast(error.message, 'error', 7000)));
    $('#profileList').addEventListener('click', (event) => {
        const activate = event.target.closest('[data-profile-activate]');
        const remove = event.target.closest('[data-profile-delete]');
        if (activate) activateProfile(activate.dataset.profileActivate).catch((error) => toast(error.message, 'error', 7000));
        if (remove) deleteProfile(remove.dataset.profileDelete).catch((error) => toast(error.message, 'error'));
    });
    $('#nodeSearch').addEventListener('input', renderProxyWorkspace);
    $('#proxyWorkspace').addEventListener('click', (event) => {
        const group = event.target.closest('[data-group]');
        if (group) { state.activeGroup = group.dataset.group; renderProxyWorkspace(); return; }
        const delay = event.target.closest('[data-delay-node]');
        if (delay) { event.stopPropagation(); testNodeDelay(delay.dataset.delayNode).catch((error) => toast(error.message, 'error')); return; }
        const node = event.target.closest('[data-node]');
        if (node) selectNode(node.dataset.node).catch((error) => toast(error.message, 'error'));
    });
    $('#modalBody').addEventListener('click', (event) => {
        const tab = event.target.closest('[data-import-tab]');
        if (tab) switchProfileImportTab(tab.dataset.importTab);
    });
    $('#modalBody').addEventListener('submit', (event) => {
        if (event.target.id === 'profileImportForm') importProfile(event).catch((error) => toast(error.message, 'error', 7000));
    });
}

async function initialize() {
    bindEvents();
    setInterval(updateClock, 1000);
    if (!state.token) return;
    try {
        await api('/config/get');
        await enterApp();
    } catch {
        logout(false);
    }
}

initialize();
