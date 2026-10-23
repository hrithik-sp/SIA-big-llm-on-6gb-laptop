"use strict";
// Sprout UI: talks only to its own backend (same origin). See app/server.py.

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const icon = (id, cls) => {
  const s = document.createElementNS("http://www.w3.org/2000/svg", "svg"), u = document.createElementNS("http://www.w3.org/2000/svg", "use");
  u.setAttribute("href", "#" + id); s.append(u); if (cls) s.setAttribute("class", cls); return s;
};
const MAX_DOC_CHARS = 45000;      // ~11k tokens: leaves room for the answer in the 16k context
const MAX_IMG_SIDE = 1600;

const S = {
  config: null, modes: {}, mode: "chat", model: null, chat: null, chats: [],
  attachments: [], abort: null, streaming: false, status: null, failures: 0, lastWake: 0, lastSpeed: null, settings: null,
};

// ------------------------------------------------------------------ helpers
function toast(msg, ms = 3000) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), ms);
}
async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
const newId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
const modelName = (m) => (S.config.names || {})[m] || (S.config.models[m] || {}).name || m;
const canSee = () => !!(S.config.models[S.model] || {}).vision;
const busy = (on) => { $("#loader").hidden = !on; document.body.classList.toggle("working", !!S.streaming); };

// ------------------------------------------------------------------ markdown + math
marked.setOptions({ gfm: true, breaks: false });
function renderMarkdown(text) {
  const math = [];
  const stash = (expr, display) => {
    let html;
    try { html = katex.renderToString(expr, { displayMode: display, throwOnError: false, trust: false }); }
    catch { return display ? `$$${expr}$$` : `$${expr}$`; }
    math.push(html); return `@@MATH${math.length - 1}@@`;
  };
  // leave code untouched; protect math from Markdown (underscores, asterisks)
  const parts = text.split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/g);
  for (let i = 0; i < parts.length; i += 2) {
    parts[i] = parts[i]
      .replace(/\$\$([\s\S]+?)\$\$/g, (_, e) => stash(e, true))
      .replace(/\\\[([\s\S]+?)\\\]/g, (_, e) => stash(e, true))
      .replace(/\\\(([\s\S]+?)\\\)/g, (_, e) => stash(e, false))
      .replace(/\$(?![\s$])([^$\n]+?)(?<!\s)\$(?!\d)/g, (_, e) => stash(e, false));
  }
  const html = DOMPurify.sanitize(marked.parse(parts.join("")));
  return html.replace(/@@MATH(\d+)@@/g, (_, i) => math[+i]);
}
function decorateCode(container) {
  container.querySelectorAll("pre code").forEach((c) => {
    if (c.dataset.done) return;
    c.dataset.done = "1";
    try { hljs.highlightElement(c); } catch { /* unknown language */ }
    const b = el("button", "copy"); b.append(icon("i-copy"), el("span", null, "Copy"));
    b.onclick = () => { navigator.clipboard.writeText(c.innerText); $("span", b).textContent = "Copied"; setTimeout(() => ($("span", b).textContent = "Copy"), 1200); };
    c.parentElement.appendChild(b);
  });
}

// ------------------------------------------------------------------ menus & panels
function openPopover(pop, anchor, { above = true } = {}) {
  closePopovers();
  pop.hidden = false;
  const r = anchor.getBoundingClientRect(), w = pop.offsetWidth, h = pop.offsetHeight;
  const left = Math.max(8, Math.min(r.left, innerWidth - w - 8));
  pop.style.left = left + "px";
  pop.style.top = (above && r.top - h - 8 > 8 ? r.top - h - 8 : r.bottom + 8) + "px";
  const first = pop.querySelector("button"); if (first) first.focus({ preventScroll: true });
}
function closePopovers() { $$(".menu").forEach((m) => (m.hidden = true)); }
document.addEventListener("mousedown", (e) => { if (!e.target.closest(".menu, #plus, #mode-btn, #status-pill, #open-settings")) closePopovers(); });

function renderModeMenu() {
  const menu = $("#menu-mode"); menu.innerHTML = "";
  for (const m of S.config.modes) {
    const b = el("button"); b.setAttribute("role", "menuitemradio"); b.setAttribute("aria-checked", m.id === S.mode);
    const t = el("span", "mt"); t.append(el("span", null, m.name), el("small", null, m.blurb));
    b.append(icon(m.icon), t);
    if (m.id === S.mode) b.append(icon("i-check", "on"));
    b.onclick = () => { closePopovers(); setMode(m.id); $("#input").focus(); };
    menu.append(b);
  }
}
function renderModeChips() {
  const box = $("#mode-chips"); box.innerHTML = "";
  for (const m of S.config.modes) {
    const b = el("button", "chip" + (m.id === S.mode ? " active" : "")); b.title = m.blurb;
    b.append(icon(m.icon), el("span", null, m.name));
    b.onclick = () => { setMode(m.id); $("#input").focus(); };
    box.append(b);
  }
}
function setMode(id, { keepModel = false } = {}) {
  S.mode = id;
  const m = S.modes[id];
  if (!keepModel) S.model = m.model === "gemma" ? (S.status && S.status.default_model === "gptoss" && !m.requires_image ? "gptoss" : "gemma") : m.model;
  $("#think").checked = !!m.think;
  $("#mode-name").textContent = m.name;
  $("#mode-btn .mi use").setAttribute("href", "#" + m.icon);
  $("#input").placeholder = m.requires_image ? "Add an image (+, paste or drop), then ask about it…"
    : m.requires_file ? "Add a PDF or text file (+), then ask about it…"
    : m.rag ? "Ask about the files in your Library…"
    : m.style === "messenger" ? "Say something…" : `Ask Sprout · ${m.name}…`;
  renderModeMenu(); renderModeChips(); renderModelSelect();
}
function renderModelSelect() {
  const sel = $("#model"); sel.innerHTML = "";
  for (const [id, info] of Object.entries(S.config.models)) {
    const o = el("option", null, info.name); o.value = id; o.title = info.about; if (id === S.model) o.selected = true;
    sel.append(o);
  }
}

// ------------------------------------------------------------------ settings
async function loadSettings() { S.settings = await api("/api/settings").catch(() => S.settings); renderSettings(); }
async function saveSettings(upd) { S.settings = await post("/api/settings", upd).catch((e) => { toast(e.message); return S.settings; }); renderSettings(); }
function renderSettings() {
  const st = S.settings; if (!st) return;
  $("#set-laptop").checked = !!st.laptop_access;
  $("#set-actions").checked = !!st.laptop_actions;
  $("#set-actions").disabled = !st.laptop_access;
  if (document.activeElement !== $("#set-name")) $("#set-name").value = st.user_name || "";
  const al = $("#set-allowed"); al.innerHTML = "";
  for (const id of st.always_allow) {
    const tool = st.tools.find((t) => t.id === id); if (!tool) continue;
    const c = el("span", "al", "Always allowed: " + tool.label);
    const x = el("button"); x.title = "Ask me again next time"; x.append(icon("i-x"));
    x.onclick = () => saveSettings({ always_allow: st.always_allow.filter((t) => t !== id) });
    c.append(x); al.append(c);
  }
}

// ------------------------------------------------------------------ status
async function pollStatus() {
  try {
    const s = await api("/api/status"); S.status = s; S.failures = 0;
    // Sprout was restarted with new code: reload the window between answers so it matches
    if (S.boot && s.boot && s.boot !== S.boot && !S.streaming) { location.reload(); return; }
    S.boot = S.boot || s.boot;
    $("#offline").hidden = true; $("#send").disabled = false;
    const entries = Object.entries(s.models || {});
    const loaded = entries.find(([, v]) => v === "loaded"), loading = entries.find(([, v]) => v === "loading");
    const sleeping = entries.find(([, v]) => v === "sleeping"), failed = entries.find(([, v]) => v === "error");
    const dot = $("#status-pill .dot"), txt = $("#status-text");
    if (S.streaming) { dot.className = "dot busy"; txt.textContent = "Working"; }
    else if (failed) { dot.className = "dot err"; txt.textContent = "Model failed to load"; }
    else if (loading) { dot.className = "dot busy"; txt.textContent = `Waking ${modelName(loading[0])}`; }
    else if (loaded) { dot.className = "dot ok"; txt.textContent = `${modelName(loaded[0])} ready`; }
    else if (sleeping) { dot.className = "dot"; txt.textContent = "Resting"; }
    else { dot.className = "dot"; txt.textContent = "Resting"; }
    busy(S.streaming || !!loading);
    $("#s-model").textContent = loaded ? modelName(loaded[0]) : loading ? "loading…" : sleeping ? "resting (unloaded)" : "not loaded";
    $("#s-speed").textContent = S.lastSpeed ? `${S.lastSpeed.toFixed(1)} tok/s` : "–";
    $("#s-ram").textContent = `${s.ram_avail_gib} of ${s.ram_total_gib} GB`;
    $("#s-ram-bar").style.width = Math.min(100, 100 * s.ram_avail_gib / s.ram_total_gib) + "%";
    $("#s-ram-bar").parentElement.classList.toggle("low", s.ram_avail_gib < 3);
    $("#s-vram").textContent = s.vram_free_gib != null ? `${s.vram_free_gib} of ${s.vram_total_gib} GB` : "–";
    $("#s-temp").textContent = s.gpu_temp_c != null ? `${s.gpu_temp_c} °C` : "–";
    $("#s-power").textContent = s.battery == null ? "–" : (s.on_ac ? `plugged in · ${Math.round(s.battery)}%` : `battery ${Math.round(s.battery)}% · slower`);
    if (s.engine_error) $("#s-model").textContent = s.engine_error;
    renderVitals(s);
    for (const item of s.inbox || []) receiveInbox(item);  // files sent from Explorer ("Ask Sprout")
  } catch {
    if (++S.failures >= 2) {
      $("#offline").hidden = false; $("#send").disabled = true;
      $("#status-pill .dot").className = "dot err"; $("#status-text").textContent = "Offline";
      busy(false);
    }
  }
}
function renderVitals(s) {
  const bar = (id, frac, hot) => { const b = $(id); b.style.height = Math.max(0, Math.min(100, 100 * frac)) + "%"; b.classList.toggle("hot", !!hot); };
  if (s.cpu != null) { $("#v-cpu").textContent = Math.round(s.cpu) + "%"; bar("#vb-cpu", s.cpu / 100, s.cpu > 85); }
  const used = s.ram_total_gib - s.ram_avail_gib;
  $("#v-ram").textContent = used.toFixed(1) + "G"; bar("#vb-ram", used / s.ram_total_gib, s.ram_avail_gib < 2);
  if (s.gpu_temp_c != null) { $("#v-gpu").textContent = s.gpu_temp_c + "°"; bar("#vb-gpu", (s.gpu_temp_c - 30) / 60, s.gpu_temp_c >= 83); }
  if (s.vram_free_gib != null) {
    const vu = s.vram_total_gib - s.vram_free_gib;
    $("#v-vram").textContent = vu.toFixed(1) + "G"; bar("#vb-vram", vu / s.vram_total_gib, false);
  }
}
function wake() {  // the window became visible: get the model ready before the first message
  if (document.hidden || Date.now() - S.lastWake < 30000 || !S.model) return;  // never from a hidden window
  S.lastWake = Date.now();
  post("/api/load", { model: S.model }).catch(() => {});
}

// ------------------------------------------------------------------ chats
async function loadChats() { S.chats = await api("/api/chats").catch(() => []); renderChatList(); }
function renderChatList() {
  const box = $("#chat-list"), q = $("#search").value.trim().toLowerCase();
  box.innerHTML = "";
  const today = new Date().setHours(0, 0, 0, 0), week = today - 6 * 864e5;
  let lastGroup = null;
  for (const c of S.chats.filter((c) => !q || c.title.toLowerCase().includes(q))) {
    const t = c.updated * 1000, g = t >= today ? "Today" : t >= week ? "This week" : "Earlier";
    if (g !== lastGroup) { box.append(el("div", "group-label", g)); lastGroup = g; }
    const b = el("button", "chat-item" + (S.chat && S.chat.id === c.id ? " active" : ""));
    b.append(el("span", "t", c.title));
    const del = el("span", "del"); del.title = "Delete chat"; del.append(icon("i-x"));
    del.onclick = async (e) => {
      e.stopPropagation();
      if (!confirm(`Delete "${c.title}"?`)) return;
      await fetch(`/api/chats/${c.id}`, { method: "DELETE" });
      if (S.chat && S.chat.id === c.id) newChat();
      loadChats();
    };
    b.append(del);
    b.onclick = () => openChat(c.id);
    box.append(b);
  }
}
function showHome(home) { $("#main").classList.toggle("home", home); $("#title").textContent = home ? "" : (S.chat ? S.chat.title : ""); }
function newChat() {
  if (S.streaming) stop();
  S.chat = null; S.attachments = [];
  $("#messages").innerHTML = ""; showHome(true);
  renderAttachments(); setMode(S.mode); renderChatList(); $("#input").focus();
}
async function openChat(id) {
  if (S.streaming) stop();
  const c = await api(`/api/chats/${id}`);
  S.chat = c; S.attachments = [];
  setMode(c.mode || "chat", { keepModel: true }); S.model = c.model || S.model; renderModelSelect();
  showHome(false); $("#messages").innerHTML = "";
  for (const m of c.messages) {
    if (m.role === "user") addUserMessage(m);
    else { const v = addAssistantMessage(); fillAssistant(v, m, true); }
  }
  renderAttachments(); renderChatList(); scrollDown(true);
}
async function saveChat() {
  if (!S.chat) return;
  await fetch(`/api/chats/${S.chat.id}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S.chat) });
  loadChats();
}

// ------------------------------------------------------------------ attachments
function addImage(url, name) {
  if (!canSee()) { S.model = "gemma"; renderModelSelect(); toast("Switched to Gemma: it can see images."); }
  S.attachments.push({ kind: "image", name: name || "image", url });
  if (!S.modes[S.mode].images) setMode("see");
  renderAttachments();
  // start loading the vision setup now, while the question is still being typed
  if (!S.status || S.status.models["gemma-vision"] !== "loaded") post("/api/load", { model: "gemma-vision" }).catch(() => {});
}
async function addFiles(files) {
  for (const f of files) {
    if (f.type.startsWith("image/")) {
      try { addImage(await imageToDataUrl(f), f.name); } catch (e) { toast(e.message); }
      continue;
    }
    try {
      const r = await fetch("/api/extract", { method: "POST", headers: { "X-Filename": encodeURIComponent(f.name) }, body: f });
      const d = await r.json();
      if (!r.ok) throw new Error(d.error);
      if (!d.text.trim()) throw new Error(`${f.name}: no text found. For scanned pages, add them as images.`);
      S.attachments.push({ kind: "doc", name: f.name, text: d.text, chars: d.chars, pages: d.pages });
      if (d.chars > MAX_DOC_CHARS) toast(`${f.name} is long. Sprout reads about ${MAX_DOC_CHARS / 1000}k characters at once; for long files, use the Library.`, 6000);
      if (!S.modes[S.mode].files) setMode("read");
    } catch (e) { toast(e.message, 5000); }
  }
  renderAttachments();
}
async function receiveInbox(item) {
  try {
    const blob = await (await fetch(`/api/inbox/${item.token}`)).blob();
    const ext = item.name.split(".").pop().toLowerCase();
    const type = ["png", "jpg", "jpeg", "gif", "bmp", "webp"].includes(ext) ? "image/" + (ext === "jpg" ? "jpeg" : ext) : "";
    await addFiles([new File([blob], item.name, { type })]);
    $("#input").focus();
  } catch (e) { toast(`Couldn't open ${item.name}: ${e.message}`); }
}
function imageToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const img = new Image(), url = URL.createObjectURL(file);
    img.onload = () => {
      const scale = Math.min(1, MAX_IMG_SIDE / Math.max(img.width, img.height));
      if (scale === 1 && file.size < 1.5e6) {  // small enough: keep the original bytes
        const fr = new FileReader(); fr.onload = () => resolve(fr.result); fr.onerror = reject; fr.readAsDataURL(file);
      } else {
        const c = document.createElement("canvas");
        c.width = Math.round(img.width * scale); c.height = Math.round(img.height * scale);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        resolve(c.toDataURL("image/jpeg", 0.9));
      }
      URL.revokeObjectURL(url);
    };
    img.onerror = () => reject(new Error("could not read image"));
    img.src = url;
  });
}
function renderAttachments() {
  const box = $("#attachments"); box.innerHTML = "";
  S.attachments.forEach((a, i) => {
    const d = el("div", "att");
    if (a.kind === "image") { const im = el("img"); im.src = a.url; d.append(im); }
    else { const ic = el("span", "ic"); ic.append(icon("i-doc")); d.append(ic); }
    const n = el("span", "n", a.name);
    if (a.kind === "doc") n.append(el("small", null, a.pages > 1 ? `${a.pages} pages` : `${Math.round(a.chars / 1000)}k characters`));
    d.append(n);
    const x = el("button", "x"); x.title = "Remove"; x.append(icon("i-x")); x.onclick = () => { S.attachments.splice(i, 1); renderAttachments(); };
    d.append(x); box.append(d);
  });
}
async function snipScreen() {
  toast("Drag over the part of the screen Sprout should see…", 6000);
  try {
    const r = await post("/api/snip");
    if (r.image) { addImage(r.image, "Screen snip"); $("#input").focus(); } else toast("No snip captured.");
  } catch (e) { toast("Snip failed: " + e.message); }
}
async function fromClipboard() {
  try {
    const r = await post("/api/clipboard");
    if (r.image) { addImage(r.image, "Clipboard image"); return; }
    if (r.files && r.files.length) { for (const f of r.files) await receiveInbox(f); return; }
  } catch { /* fall through to text */ }
  try {
    const t = await navigator.clipboard.readText();
    if (t && t.trim()) { const i = $("#input"); i.value = (i.value ? i.value + "\n\n" : "") + t; autosize(); i.focus(); return; }
  } catch { /* permission denied */ }
  toast("Nothing on the clipboard Sprout can use.");
}

// ------------------------------------------------------------------ messages
function scrollDown(force) {
  const c = $("#conversation");
  if (force || c.scrollHeight - c.scrollTop - c.clientHeight < 160) c.scrollTop = c.scrollHeight;
}
function addUserMessage(m) {
  const d = el("div", "msg user");
  if (m.images && m.images.length) {
    const imgs = el("div", "imgs");
    for (const u of m.images) { const im = el("img"); im.src = u; imgs.append(im); }
    d.append(imgs);
  }
  for (const doc of m.docs || []) { const c = el("span", "doc-chip"); c.append(icon("i-doc"), el("span", null, doc.name)); d.append(c); }
  if (m.content) d.append(el("div", "bubble", m.content));
  $("#messages").append(d);
}
function addAssistantMessage() {
  const d = el("div", "msg assistant");
  const mark = el("img", "mark"); mark.src = "/static/logo.svg"; mark.alt = "";
  const sources = el("div", "sources"), thinking = el("details", "thinking"), answer = el("div", "answer"), meta = el("div", "meta");
  thinking.hidden = true; thinking.append(el("summary", null, "Reasoning…"), el("div", "body"));
  d.append(mark, sources, thinking, answer, meta);
  $("#messages").append(d);
  return { d, sources, thinking, answer, meta };
}
function fillAssistant(v, m, final) {
  if (m.sources && m.sources.length && !v.sources.childElementCount) {
    m.sources.forEach((s, i) => {
      const b = el("span", "source", `[${i + 1}] ${s.file}${s.page ? " · p." + s.page : ""}`);
      b.title = s.text.slice(0, 600); v.sources.append(b);
    });
  }
  if (final && m.tools && m.tools.length && !v.d.querySelector(".tool-note")) for (const t of m.tools) toolNote(v, t);
  if (m.reasoning) {
    v.thinking.hidden = false;
    $(".body", v.thinking).textContent = m.reasoning;
    $("summary", v.thinking).textContent = final ? `Reasoned for ${m.think_s ? m.think_s.toFixed(0) + " s" : "a while"}` : "Reasoning…";
  }
  v.answer.innerHTML = m.answer ? renderMarkdown(m.answer) : "";
  if (final) {
    decorateCode(v.answer);
    v.meta.innerHTML = "";
    if (m.error) { v.meta.append(el("span", "error", m.error)); v.meta.classList.add("show"); }
    const copy = el("button"); copy.append(icon("i-copy"), el("span", null, "Copy"));
    copy.onclick = () => { navigator.clipboard.writeText(m.answer || ""); toast("Copied"); };
    v.meta.append(copy);
    if (m.stats) v.meta.append(el("span", null, m.stats));
  }
}

// ------------------------------------------------------------------ send / stream
function buildApiMessages() {
  return S.chat.messages.map((m) => {
    if (m.role === "assistant") return { role: "assistant", content: m.answer || "" };
    let text = (m.docs || []).map((d) => `<document name="${d.name.replace(/[<>"]/g, "")}">\n${d.text}\n</document>`).join("\n\n");
    text = text ? text + "\n\n" + (m.content || "Please read the document.") : m.content;
    if (!m.images || !m.images.length) return { role: "user", content: text };
    return { role: "user", content: [{ type: "text", text: text || "Look at this image." },
      ...m.images.map((u) => ({ type: "image_url", image_url: { url: u } }))] };
  });
}
async function send() {
  if (S.streaming || $("#send").disabled) return;
  const m = S.modes[S.mode];
  const text = $("#input").value.trim();
  const images = S.attachments.filter((a) => a.kind === "image");
  const docs = S.attachments.filter((a) => a.kind === "doc");
  if (!text && !images.length && !docs.length) return;
  const had = (k) => S.chat && S.chat.messages.some((x) => x[k] && x[k].length);
  if (m.requires_image && !images.length && !had("images")) { toast("Add an image first: +, paste, or drop it here."); return; }
  if (m.requires_file && !docs.length && !had("docs")) { toast("Add a PDF or text file first (+)."); return; }
  if (images.length && !canSee()) { toast(`${modelName(S.model)} can't see images; choose Gemma.`); return; }
  const docChars = docs.reduce((n, d) => n + d.chars, 0);
  if (docChars > MAX_DOC_CHARS && !confirm(`The attached text is longer than Sprout can read at once (~${MAX_DOC_CHARS / 1000}k characters), so parts may be missed. Continue? For long files, the Library works better.`)) return;

  if (!S.chat) S.chat = { id: newId(), title: (text || docs[0]?.name || "Image").slice(0, 60), mode: S.mode, model: S.model, created: Date.now(), messages: [] };
  S.chat.mode = S.mode; S.chat.model = S.model;
  const um = { role: "user", time: Date.now(), content: text, images: images.map((a) => a.url), docs: docs.map((d) => ({ name: d.name, text: d.text })) };
  S.chat.messages.push(um);
  showHome(false);
  addUserMessage(um);
  $("#input").value = ""; autosize(); S.attachments = []; renderAttachments();
  await stream();
}
async function stream() {
  const v = addAssistantMessage();
  const am = { role: "assistant", answer: "", reasoning: "", sources: [] };
  scrollDown(true);
  S.streaming = true; $("#send").hidden = true; $("#stop").hidden = false; busy(true);
  $("#status-pill .dot").className = "dot busy"; $("#status-text").textContent = "Working";
  S.abort = new AbortController();
  const t0 = performance.now(); let tFirst = null, tThinkEnd = null, timings = null, model = S.model;
  const wait = el("div", "waiting"); const dots = el("span", "typing"); dots.append(el("i"), el("i"), el("i"));
  wait.append(dots, el("span", null, "")); v.answer.before(wait);
  const waitTimer = setInterval(() => {
    const st = S.status && Object.entries(S.status.models).find(([, x]) => x === "loading");
    const secs = (performance.now() - t0) / 1000;
    $("span:last-child", wait).textContent = st ? `Waking ${modelName(st[0])} (up to a minute after a rest)` : secs > 5 ? `Reading… ${secs.toFixed(0)} s` : "";
  }, 500);
  let pending = false, finished = false;
  const paint = () => {
    if (pending || finished) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; if (!finished) { fillAssistant(v, am, false); scrollDown(); } });
  };
  v.d.classList.add("live"); document.body.classList.add("working");
  am.tools = [];
  try {
    const r = await fetch("/api/chat", {
      method: "POST", headers: { "Content-Type": "application/json" }, signal: S.abort.signal,
      body: JSON.stringify({ mode: S.mode, model: S.model, think: $("#think").checked, messages: buildApiMessages(), library: "library" }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || r.statusText);
    const reader = r.body.getReader(), dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim(); buf = buf.slice(i + 2);
        if (!line.startsWith("data: ")) continue;
        const ev = JSON.parse(line.slice(6));
        if (ev.type === "meta") { model = ev.model; am.sources = ev.sources || []; paint(); continue; }
        if (ev.type === "error") throw new Error(ev.error);
        if (ev.type === "tool_request") { wait.remove(); askPermission(v, ev); continue; }
        if (ev.type === "tool") { am.tools.push({ label: ev.label, status: ev.status }); toolNote(v, ev); continue; }
        if (ev.type === "done") continue;
        if (ev.timings) timings = ev.timings;
        const delta = ev.choices && ev.choices[0] && ev.choices[0].delta;
        if (!delta) continue;
        if (tFirst == null && (delta.content || delta.reasoning_content)) { tFirst = performance.now(); wait.remove(); }
        if (delta.reasoning_content) am.reasoning += delta.reasoning_content;
        if (delta.content) { if (am.reasoning && tThinkEnd == null) tThinkEnd = performance.now(); am.answer += delta.content; }
        paint();
      }
    }
  } catch (e) {
    am.error = e.name === "AbortError" ? "Stopped." : e.message === "Failed to fetch" ? "Sprout's engine isn't reachable. Reopen Sprout and try again." : e.message;
  } finally {
    finished = true;
    clearInterval(waitTimer); wait.remove();
    v.d.classList.remove("live");
    S.streaming = false; $("#send").hidden = false; $("#stop").hidden = true; busy(false);
  }
  if (am.reasoning) am.think_s = ((tThinkEnd || performance.now()) - (tFirst || t0)) / 1000;
  const parts = [modelName(model)];
  if (timings && timings.predicted_per_second) { S.lastSpeed = timings.predicted_per_second; parts.push(`${timings.predicted_per_second.toFixed(1)} tok/s`); }
  parts.push(`${((performance.now() - t0) / 1000).toFixed(1)} s`);
  am.stats = parts.join(" · ");
  am.time = Date.now();
  fillAssistant(v, am, true);
  S.chat.messages.push(am);
  $("#title").textContent = S.chat.title;
  saveChat(); pollStatus(); scrollDown();
}
function stop() { if (S.abort) S.abort.abort(); }
function askPermission(v, ev) {
  const card = el("div", "tool-card");
  const h = el("div", "tc-h"); h.append(icon("i-laptop"), el("span", null, `Sprout wants to: ${ev.label}`));
  const r = el("div", "tc-r", ev.action
    ? `This changes your laptop: it ${ev.does}.${ev.danger ? " You'll be asked every time for this one." : ""}`
    : `Reads ${ev.reads}. Read-only: nothing on your laptop is changed.`);
  if (ev.action) card.classList.add("action");
  const b = el("div", "tc-b");
  const decide = async (decision) => {
    $$("button", b).forEach((x) => (x.disabled = true));
    await post("/api/tool-decision", { id: ev.id, decision }).catch(() => {});
    card.remove();
    if (decision === "always") loadSettings();
  };
  const once = el("button", "yes", "Allow once"); once.onclick = () => decide("once");
  const always = el("button", null, "Always allow"); always.onclick = () => decide("always");
  const deny = el("button", null, "Deny"); deny.onclick = () => decide("deny");
  if (ev.danger) b.append(once, deny); else b.append(once, always, deny);
  card.append(h, r, b);
  v.answer.before(card); scrollDown(true); once.focus();
}
function toolNote(v, ev) {
  const ok = ev.status === "done";
  const text = ok ? ev.label : ev.status === "denied" ? `Declined: ${ev.label}` : ev.status === "off" ? "Laptop access is off (Settings)" : ev.status === "actions_off" ? "Actions are off (Settings)" : `Couldn't: ${ev.label}`;
  const n = el("div", "tool-note " + (ok ? "ok" : "no")); n.append(icon(ok ? "i-check" : "i-x"), el("span", null, text));
  v.answer.before(n);
}

// ------------------------------------------------------------------ library
function renderLibrary(libs) {
  const lib = (libs || []).find((l) => l.name === "library");
  const box = $("#lib-files"); box.innerHTML = "";
  $("#lib-count").textContent = lib && lib.files.length ? lib.files.length : "";
  $("#lib-clear").hidden = !lib;
  if (!lib) { box.append(el("div", "empty", "No files yet.")); return; }
  for (const f of lib.files) { const r = el("div", "f"); r.append(icon("i-doc"), el("span", null, f)); box.append(r); }
}
async function addToLibrary(files) {
  let libs = null, n = 0;
  for (const f of files) {
    toast(`Adding ${f.name} (${++n} of ${files.length})…`, 60000);
    try {
      const r = await fetch("/api/kb/add", { method: "POST", headers: { "X-Filename": encodeURIComponent(f.name), "X-Library": "library" }, body: f });
      const d = await r.json(); if (!r.ok) throw new Error(d.error);
      libs = d.libraries;
    } catch (e) { toast(`${f.name}: ${e.message}`, 6000); await new Promise((res) => setTimeout(res, 2000)); }
  }
  if (libs) { renderLibrary(libs); toast("Library updated. Ask about it in Library mode."); }
}

// ------------------------------------------------------------------ wiring
function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 240) + "px"; }
function wire() {
  const input = $("#input");
  input.addEventListener("input", () => {
    autosize();
    if (input.value === "/") { input.value = ""; openPopover($("#menu-mode"), $("#mode-btn")); }  // quick mode switch
  });
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } });
  $("#send").onclick = send;
  $("#stop").onclick = stop;
  $("#new-chat").onclick = newChat;
  $("#plus").onclick = (e) => { e.stopPropagation(); openPopover($("#menu-plus"), $("#plus")); };
  $("#mode-btn").onclick = (e) => { e.stopPropagation(); openPopover($("#menu-mode"), $("#mode-btn")); };
  $("#open-settings").onclick = (e) => { e.stopPropagation(); loadSettings(); openPopover($("#panel-settings"), $("#open-settings")); };
  $("#set-laptop").onchange = (e) => saveSettings({ laptop_access: e.target.checked });
  $("#set-actions").onchange = (e) => saveSettings({ laptop_actions: e.target.checked });
  $("#set-name").onchange = (e) => saveSettings({ user_name: e.target.value });
  $("#status-pill").onclick = (e) => { e.stopPropagation(); pollStatus(); openPopover($("#panel-status"), $("#status-pill"), { above: false }); };
  $("#menu-plus").onclick = (e) => {
    const act = e.target.closest("button")?.dataset.act; if (!act) return;
    closePopovers();
    if (act === "upload") $("#file-input").click();
    if (act === "snip") snipScreen();
    if (act === "clip") fromClipboard();
    if (act === "lib") $("#lib-input").click();
  };
  $("#file-input").onchange = (e) => { addFiles([...e.target.files]); e.target.value = ""; };
  $("#lib-input").onchange = (e) => { addToLibrary([...e.target.files]); e.target.value = ""; };
  $("#open-library").onclick = () => { $("#library").hidden = false; };
  $("#close-library").onclick = () => { $("#library").hidden = true; };
  $("#lib-add").onclick = () => $("#lib-input").click();
  $("#lib-clear").onclick = async () => {
    if (!confirm("Remove all files from the Library? Your original files aren't touched.")) return;
    renderLibrary((await post("/api/kb/delete", { name: "library" })).libraries);
  };
  $("#unload").onclick = async () => { closePopovers(); await post("/api/unload").catch(() => {}); toast("Model unloaded. Memory is free; it wakes on your next message."); pollStatus(); };
  $("#search").oninput = renderChatList;
  $("#think").onchange = () => {};
  $("#model").onchange = (e) => {
    S.model = e.target.value;
    if (S.chat) S.chat.model = S.model;
    post("/api/load", { model: S.model }).catch(() => {});
  };
  $("#collapse").onclick = () => $("#app").classList.add("collapsed");
  $("#expand").onclick = () => $("#app").classList.toggle("collapsed");
  $("#quit").onclick = async () => {
    if (!confirm("Quit Sprout completely? (Closing the window just hides it; Ctrl+Alt+Space brings it back.)")) return;
    await fetch("/api/quit", { method: "POST" }).catch(() => {});
    document.body.innerHTML = '<div style="display:grid;place-items:center;height:100vh;color:#9097a7;font:15px Segoe UI">Sprout has stopped. You can close this window.</div>';
  };
  document.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); newChat(); }
    if (e.key === "Escape") { if ($$(".menu").some((m) => !m.hidden)) closePopovers(); else if (!$("#library").hidden) $("#library").hidden = true; else if (S.streaming) stop(); }
  });
  document.addEventListener("paste", (e) => {
    const files = [...((e.clipboardData || {}).files || [])];
    if (files.length) { e.preventDefault(); addFiles(files); }
  });
  let depth = 0;
  document.addEventListener("dragenter", (e) => { if (e.dataTransfer.types.includes("Files")) { depth++; $("#drop").hidden = false; } });
  document.addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; $("#drop").hidden = true; } });
  document.addEventListener("dragover", (e) => e.preventDefault());
  document.addEventListener("drop", (e) => { e.preventDefault(); depth = 0; $("#drop").hidden = true; addFiles([...e.dataTransfer.files]); });
  document.addEventListener("visibilitychange", () => { if (!document.hidden) { wake(); $("#input").focus(); } });
  window.addEventListener("focus", wake);
}

// ------------------------------------------------------------------ start
(async function init() {
  wire();
  try { S.config = await api("/api/config"); }
  catch { $("#offline").hidden = false; $("#send").disabled = true; return; }
  for (const m of S.config.modes) S.modes[m.id] = m;
  S.status = S.config.status;
  S.model = S.config.status.default_model;
  setMode("chat", { keepModel: true });
  renderLibrary(S.config.libraries);
  if (S.model !== "gemma") toast("Memory is tight right now, so Sprout is using gpt-oss, which copes better.", 6000);
  loadChats(); pollStatus(); setInterval(pollStatus, 2000); loadSettings();
  wake();
  $("#input").title = "Enter to send · Shift+Enter new line · type / to switch mode · Ctrl+K new chat";
  const deep = location.hash.match(/^#chat=([A-Za-z0-9_-]{6,64})$/);
  if (deep) openChat(deep[1]).catch(() => {});
  $("#input").focus();
})();
