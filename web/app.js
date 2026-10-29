(function () {
  "use strict";

  const DEFAULT_LINE = [0.1, 0.6, 0.9, 0.6];
  const MAX_FILES = 5;
  const POLL_MS = 1000;
  const MAX_POLL_FAILURES = 5;
  const HEX_ID = /^[0-9a-f]{32}$/;
  const CLASSES = ["car", "bus", "truck", "motorcycle"];
  const RELIABILITY = ["HIGH", "MEDIUM", "LOW", "UNKNOWN"];
  const FLAG_KINDS = [
    "olası kaçırılan araç", "olası çift sayım", "kararsız geçiş",
    "zayıf tespit", "sınıf belirsiz", "kalabalık geçiş",
  ];
  const MISSED_KIND = "olası kaçırılan araç";
  const STATUSES = ["QUEUED", "RUNNING", "DONE", "ERROR"];
  const PHASES = ["loading", "processing", "encoding"];
  const LANG_KEY = "visionflow.lang";
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = {
    lang: "tr",
    videos: [], // {id, name, width, height, line:[x1,y1,x2,y2], lineSet}
    selectedId: null,
    jobs: new Map(), // jobId -> job record
    pollTimer: null,
    polling: false, // a pollOnce run is in flight; together with pollTimer: one loop only
    pollFailures: 0,
    busy: false, // starting jobs
    uploading: false,
    modelTouched: false, // the user picked a model; never override it with the server default
  };

  const $ = (id) => document.getElementById(id);
  const dom = {
    lang: $("lang"), alerts: $("alerts"), live: $("live"),
    previewEmpty: $("preview-empty"), editor: $("editor"), stage: $("stage"),
    frame: $("frame"), overlay: $("overlay"), flip: $("flip"), reset: $("reset"),
    lineState: $("line-state"), videos: $("videos"), videosEmpty: $("videos-empty"),
    model: $("model"), modelGpu: $("model-gpu"), conf: $("conf"), confRange: $("conf-range"),
    drop: $("drop"), files: $("files"), start: $("start"), jobs: $("jobs"),
  };

  /* ---------- helpers ---------- */

  function t(key, params) {
    const table = window.I18N[state.lang] || window.I18N.tr;
    let text = Object.prototype.hasOwnProperty.call(table, key) ? table[key] : key;
    if (params) {
      text = text.replace(/\{(\w+)\}/g, (m, name) =>
        Object.prototype.hasOwnProperty.call(params, name) ? String(params[name]) : m);
    }
    return text;
  }

  function el(tag, props, children) {
    const node = document.createElement(tag);
    if (props) {
      for (const [k, v] of Object.entries(props)) {
        if (k === "text") node.textContent = v;
        else if (k === "class") node.className = v;
        else node.setAttribute(k, v);
      }
    }
    for (const child of children || []) node.append(child);
    return node;
  }

  function svgEl(tag, attrs, text) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) node.setAttribute(k, String(v));
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
  }

  function pad2(n) {
    return String(n).padStart(2, "0");
  }

  function formatTime(seconds) {
    const total = Math.max(0, Math.floor(Number(seconds) || 0));
    return pad2(Math.floor(total / 60)) + ":" + pad2(total % 60);
  }

  function isId(value) {
    return typeof value === "string" && HEX_ID.test(value);
  }

  function safeGet(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }

  function safeSet(key, value) {
    try { window.localStorage.setItem(key, value); } catch (e) { /* storage unavailable */ }
  }

  function announce(text) {
    dom.live.textContent = text;
  }

  function showAlert(message) {
    const text = el("span", { text: message });
    const close = el("button", { type: "button", text: t("alerts.dismiss") });
    const box = el("div", { class: "alert" }, [text, close]);
    close.addEventListener("click", () => box.remove());
    dom.alerts.append(box);
    while (dom.alerts.children.length > 5) dom.alerts.firstElementChild.remove();
  }

  class ApiError extends Error {
    constructor(message, status) {
      super(message);
      this.status = status;
    }
  }

  async function api(path, options) {
    const url = path + (path.includes("?") ? "&" : "?") + "lang=" + state.lang;
    let response;
    try {
      response = await fetch(url, options);
    } catch (e) {
      throw new ApiError(t("err.network"), 0);
    }
    if (!response.ok) {
      let message = t("err.generic");
      try {
        const body = await response.json();
        if (body && typeof body.detail === "string") message = body.detail;
      } catch (e) { /* keep generic message */ }
      throw new ApiError(message, response.status);
    }
    return response;
  }

  /* ---------- static i18n ---------- */

  function applyStaticText() {
    document.documentElement.lang = state.lang;
    document.title = t("app.title");
    for (const node of document.querySelectorAll("[data-i18n]")) {
      node.textContent = t(node.getAttribute("data-i18n"));
    }
    for (const node of document.querySelectorAll("[data-i18n-alt]")) {
      node.setAttribute("alt", t(node.getAttribute("data-i18n-alt")));
    }
    for (const node of document.querySelectorAll("[data-i18n-aria-label]")) {
      node.setAttribute("aria-label", t(node.getAttribute("data-i18n-aria-label")));
    }
    dom.lang.value = state.lang;
  }

  function setLang(lang) {
    state.lang = lang === "en" ? "en" : "tr";
    safeSet(LANG_KEY, state.lang);
    applyStaticText();
    renderVideos();
    updateEditorTexts();
    drawLine();
    renderAllJobs();
    updateStartButton();
  }

  function initialLang() {
    const stored = safeGet(LANG_KEY);
    if (stored === "tr" || stored === "en") return stored;
    const nav = String(navigator.language || "").toLowerCase();
    return nav.startsWith("en") ? "en" : "tr";
  }

  /* ---------- videos and selection ---------- */

  function selectedVideo() {
    return state.videos.find((v) => v.id === state.selectedId) || null;
  }

  async function addFiles(fileList) {
    const files = Array.from(fileList || []);
    if (!files.length || state.uploading) return;
    if (state.videos.length + files.length > MAX_FILES) {
      showAlert(t("err.tooMany"));
      return;
    }
    const form = new FormData();
    for (const file of files) form.append("files", file, file.name);
    state.uploading = true;
    updateStartButton();
    announce(t("settings.uploading"));
    try {
      const response = await api("/api/videos", { method: "POST", body: form });
      const items = await response.json();
      for (const item of Array.isArray(items) ? items : []) {
        if (!item || !isId(item.video_id)) continue;
        const width = Number(item.width);
        const height = Number(item.height);
        if (!(width > 0) || !(height > 0)) continue;
        state.videos.push({
          id: item.video_id, name: String(item.name), width, height,
          line: DEFAULT_LINE.slice(), lineSet: false,
        });
        if (state.selectedId === null) state.selectedId = item.video_id;
      }
    } catch (e) {
      showAlert(e.message);
    } finally {
      state.uploading = false;
      renderVideos();
      renderEditor();
      updateStartButton();
    }
  }

  function selectVideo(id) {
    state.selectedId = id;
    renderVideos();
    renderEditor();
  }

  function removeVideo(id) {
    const index = state.videos.findIndex((v) => v.id === id);
    if (index < 0) return;
    state.videos.splice(index, 1);
    // Free the uploaded file on the server too; a failure only means the start-up sweep
    // removes it later, so it is not worth an alert.
    api("/api/videos/" + id, { method: "DELETE" }).catch(() => {});
    if (state.selectedId === id) {
      const next = state.videos[Math.min(index, state.videos.length - 1)];
      state.selectedId = next ? next.id : null;
    }
    renderVideos();
    renderEditor();
    updateStartButton();
  }

  function renderVideos() {
    dom.videos.replaceChildren();
    dom.videosEmpty.hidden = state.videos.length > 0;
    for (const video of state.videos) {
      const selected = video.id === state.selectedId;
      const note = el("span", {
        class: "badge-note" + (video.lineSet ? "" : " warn"),
        text: (video.lineSet ? "✓ " : "⚠ ") + t(video.lineSet ? "line.set" : "line.notSet"),
      });
      const pick = el("button", {
        type: "button",
        "aria-pressed": selected ? "true" : "false",
        "aria-label": t("videos.selectLabel", { name: video.name }),
        text: t(selected ? "videos.selected" : "videos.select"),
      });
      pick.addEventListener("click", () => selectVideo(video.id));
      const remove = el("button", {
        type: "button",
        "aria-label": t("videos.removeLabel", { name: video.name }),
        text: t("videos.remove"),
      });
      remove.addEventListener("click", () => removeVideo(video.id));
      const item = el("li", { class: selected ? "selected" : "" }, [
        el("span", { class: "name", text: video.name }), note, pick, remove,
      ]);
      dom.videos.append(item);
    }
  }

  /* ---------- line editor ---------- */

  const editor = { loadedId: null, parts: null, dragging: null };

  function buildOverlay() {
    const svg = dom.overlay;
    svg.replaceChildren();
    const parts = {
      lineUnder: svgEl("line", { class: "under" }),
      lineTop: svgEl("line", { class: "ln" }),
      arrowUnder: svgEl("path", { class: "under", fill: "none" }),
      arrowTop: svgEl("path", { class: "arrow", fill: "none" }),
      label1: svgEl("text", { class: "lbl", "text-anchor": "middle" }),
      label2: svgEl("text", { class: "lbl", "text-anchor": "middle" }),
      handles: [],
    };
    svg.append(parts.lineUnder, parts.lineTop, parts.arrowUnder, parts.arrowTop,
      parts.label1, parts.label2);
    for (let index = 0; index < 2; index += 1) {
      const hit = svgEl("circle", { class: "hit" });
      const ring = svgEl("circle", { class: "ring" });
      const dot = svgEl("circle", { class: "dot" });
      const group = svgEl("g", { class: "handle", tabindex: "0", role: "button" });
      group.append(hit, ring, dot);
      attachHandle(group, index);
      svg.append(group);
      parts.handles.push({ group, hit, ring, dot });
    }
    editor.parts = parts;
  }

  function pointFromEvent(event) {
    const rect = dom.overlay.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;
    return [
      clamp((event.clientX - rect.left) / rect.width, 0, 1),
      clamp((event.clientY - rect.top) / rect.height, 0, 1),
    ];
  }

  function setPoint(index, x, y) {
    const video = selectedVideo();
    if (!video) return;
    const nx = clamp(x, 0, 1);
    const ny = clamp(y, 0, 1);
    const other = index === 0 ? 2 : 0;
    if (nx === video.line[other] && ny === video.line[other + 1]) return; // endpoints must differ
    video.line[index * 2] = nx;
    video.line[index * 2 + 1] = ny;
    markLineSet(video);
    drawLine();
  }

  function markLineSet(video) {
    if (!video.lineSet) {
      video.lineSet = true;
      renderVideos();
    }
    updateEditorTexts();
  }

  function attachHandle(group, index) {
    group.addEventListener("touchstart", (e) => e.preventDefault(), { passive: false });
    group.addEventListener("pointerdown", (event) => {
      if (event.button !== undefined && event.button > 0) return;
      event.preventDefault();
      editor.dragging = index;
      try { group.setPointerCapture(event.pointerId); } catch (e) { /* not capturable */ }
      group.focus({ preventScroll: true });
    });
    group.addEventListener("pointermove", (event) => {
      if (editor.dragging !== index) return;
      const point = pointFromEvent(event);
      if (point) setPoint(index, point[0], point[1]);
    });
    const stop = (event) => {
      if (editor.dragging !== index) return;
      editor.dragging = null;
      try { group.releasePointerCapture(event.pointerId); } catch (e) { /* already released */ }
    };
    group.addEventListener("pointerup", stop);
    group.addEventListener("pointercancel", stop);
    group.addEventListener("keydown", (event) => {
      const video = selectedVideo();
      if (!video) return;
      const step = event.shiftKey ? 0.05 : 0.01;
      const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
      const move = moves[event.key];
      if (!move) return;
      event.preventDefault();
      setPoint(index, video.line[index * 2] + move[0], video.line[index * 2 + 1] + move[1]);
    });
  }

  function drawLine() {
    const video = selectedVideo();
    const parts = editor.parts;
    if (!video || !parts) return;
    const w = video.width;
    const h = video.height;
    const rect = dom.overlay.getBoundingClientRect();
    const s = rect.width > 0 ? w / rect.width : 1; // viewBox units per CSS pixel
    const [nx1, ny1, nx2, ny2] = video.line;
    const x1 = nx1 * w, y1 = ny1 * h, x2 = nx2 * w, y2 = ny2 * h;
    for (const node of [parts.lineUnder, parts.lineTop]) {
      node.setAttribute("x1", x1); node.setAttribute("y1", y1);
      node.setAttribute("x2", x2); node.setAttribute("y2", y2);
    }
    parts.lineUnder.setAttribute("stroke-width", 6 * s);
    parts.lineTop.setAttribute("stroke-width", 3 * s);

    // Arrow: perpendicular to the line, towards the positive side (Direction 1).
    // Positive side = (-dy, dx) in image coordinates, matching Line.signed_distance.
    const len = Math.hypot(x2 - x1, y2 - y1) || 1;
    const dx = (x2 - x1) / len, dy = (y2 - y1) / len;
    const nx = -dy, ny = dx;
    const mx = (x1 + x2) / 2, my = (y1 + y2) / 2;
    const tipX = mx + nx * 56 * s, tipY = my + ny * 56 * s;
    const barb = (sign) =>
      (tipX - nx * 12 * s + sign * dx * 8 * s) + " " + (tipY - ny * 12 * s + sign * dy * 8 * s);
    const arrow = "M" + mx + " " + my + " L" + tipX + " " + tipY +
      " M" + barb(1) + " L" + tipX + " " + tipY + " L" + barb(-1);
    parts.arrowUnder.setAttribute("d", arrow);
    parts.arrowTop.setAttribute("d", arrow);
    parts.arrowUnder.setAttribute("stroke-width", 7 * s);
    parts.arrowTop.setAttribute("stroke-width", 4 * s);
    parts.arrowUnder.setAttribute("stroke-linecap", "round");
    parts.arrowTop.setAttribute("stroke-linecap", "round");

    const fontSize = 14 * s;
    const place = (node, text, px, py) => {
      node.textContent = text;
      node.setAttribute("font-size", fontSize);
      node.setAttribute("stroke-width", 3.5 * s);
      node.setAttribute("x", clamp(px, 28 * s, w - 28 * s));
      node.setAttribute("y", clamp(py, 16 * s, h - 6 * s));
    };
    place(parts.label1, t("line.dir1"), tipX + nx * 16 * s, tipY + ny * 16 * s + 5 * s);
    place(parts.label2, t("line.dir2"), mx - nx * 26 * s, my - ny * 26 * s + 5 * s);

    parts.handles.forEach((handle, index) => {
      const cx = video.line[index * 2] * w;
      const cy = video.line[index * 2 + 1] * h;
      handle.hit.setAttribute("r", 24 * s);
      handle.ring.setAttribute("r", 11 * s);
      handle.dot.setAttribute("r", 7 * s);
      handle.ring.setAttribute("stroke-width", 2.5 * s);
      handle.dot.setAttribute("stroke-width", 2 * s);
      for (const node of [handle.hit, handle.ring, handle.dot]) {
        node.setAttribute("cx", cx); node.setAttribute("cy", cy);
      }
      const label = t(index === 0 ? "handle.start" : "handle.end") + ": " +
        Math.round(video.line[index * 2] * 100) + "%, " +
        Math.round(video.line[index * 2 + 1] * 100) + "%";
      handle.group.setAttribute("aria-label", label);
    });
  }

  function updateEditorTexts() {
    const video = selectedVideo();
    dom.overlay.setAttribute("aria-label", t("preview.svgLabel"));
    if (!video) return;
    dom.lineState.textContent =
      (video.lineSet ? "✓ " : "⚠ ") + t(video.lineSet ? "line.set" : "line.notSet");
    dom.lineState.classList.toggle("warn", !video.lineSet);
  }

  function renderEditor() {
    const video = selectedVideo();
    dom.previewEmpty.hidden = video !== null;
    dom.editor.hidden = video === null;
    if (!video) {
      editor.loadedId = null;
      dom.frame.removeAttribute("src");
      return;
    }
    if (editor.loadedId !== video.id) {
      editor.loadedId = video.id;
      dom.stage.style.aspectRatio = video.width + " / " + video.height;
      dom.overlay.setAttribute("viewBox", "0 0 " + video.width + " " + video.height);
      dom.frame.src = "/api/videos/" + video.id + "/preview?lang=" + state.lang;
      buildOverlay();
    }
    updateEditorTexts();
    drawLine();
  }

  function flipSelected() {
    const video = selectedVideo();
    if (!video) return;
    const [x1, y1, x2, y2] = video.line;
    video.line = [x2, y2, x1, y1];
    markLineSet(video);
    drawLine();
  }

  function resetSelected() {
    const video = selectedVideo();
    if (!video) return;
    video.line = DEFAULT_LINE.slice();
    video.lineSet = false;
    renderVideos();
    updateEditorTexts();
    drawLine();
  }

  /* ---------- settings ---------- */

  function readConfidence() {
    const raw = parseFloat(String(dom.conf.value).replace(",", "."));
    const value = Number.isFinite(raw) ? raw : 0.25;
    return Math.round(clamp(value, 0.05, 0.95) * 100) / 100;
  }

  function syncConfidenceFromNumber() {
    const value = readConfidence();
    dom.conf.value = String(value);
    dom.confRange.value = String(value);
  }

  function updateStartButton() {
    dom.start.disabled = state.busy || state.uploading;
    dom.start.textContent = t(state.busy ? "start.running" : "start.button");
  }

  /* ---------- jobs ---------- */

  async function startCounting() {
    if (state.busy || state.uploading) return;
    if (!state.videos.length) {
      showAlert(t("err.noVideo"));
      return;
    }
    const unset = state.videos.filter((v) => !v.lineSet);
    if (unset.length &&
        !window.confirm(t("start.confirmNotSet", { names: unset.map((v) => v.name).join(", ") }))) {
      return;
    }
    state.busy = true;
    updateStartButton();
    const body = {
      videos: state.videos.map((v) => ({
        video_id: v.id,
        line: v.line.map((n) => Math.round(clamp(n, 0, 1) * 1e6) / 1e6),
      })),
      model_name: dom.model.value,
      confidence: readConfidence(),
    };
    try {
      const response = await api("/api/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const created = await response.json();
      for (const item of Array.isArray(created) ? created : []) {
        if (!item || !isId(item.job_id) || !isId(item.video_id)) continue;
        const video = state.videos.find((v) => v.id === item.video_id);
        if (!state.jobs.has(item.job_id)) {
          state.jobs.set(item.job_id, newJob(item.job_id, video ? video.name : item.job_id));
        }
        // The server consumes the uploaded file, so the video leaves the pending list.
        state.videos = state.videos.filter((v) => v.id !== item.video_id);
      }
      if (!state.videos.some((v) => v.id === state.selectedId)) {
        state.selectedId = state.videos.length ? state.videos[0].id : null;
      }
      renderVideos();
      renderEditor();
      renderAllJobs();
      startPolling();
    } catch (e) {
      showAlert(e.message);
    } finally {
      state.busy = false;
      updateStartButton();
    }
  }

  function newJob(id, name, status) {
    return {
      id, name, status: STATUSES.includes(status) ? status : "QUEUED",
      progress: 0, phase: null, error: null, result: null, fetching: false, resultFailed: false,
      el: null, videoEl: null, announced: -1,
    };
  }

  // Jobs that already exist on the server (earlier visits, other tabs, a server restart):
  // listed in API order (oldest first), so cards keep the same order on every reload.
  async function loadExistingJobs() {
    let list;
    try {
      const response = await api("/api/jobs");
      list = await response.json();
    } catch (e) {
      showAlert(t("err.jobsLoadFailed") + " " + e.message);
      return;
    }
    const pageCards = Array.from(state.jobs.values()).map((job) => job.el).filter(Boolean);
    for (const data of Array.isArray(list) ? list : []) {
      if (!data || !isId(data.job_id) || state.jobs.has(data.job_id)) continue;
      const name = typeof data.video_name === "string" && data.video_name
        ? data.video_name : data.job_id;
      const job = newJob(data.job_id, name, data.status);
      job.announced = Math.floor(clamp(Number(data.progress) || 0, 0, 1) * 4); // no announce
      state.jobs.set(job.id, job);
      applyStatus(job, data); // renders the card; a DONE job fetches its result
    }
    // Cards of jobs started on this page while the list was loading are newer: keep them last.
    for (const card of pageCards) dom.jobs.append(card);
    if (Array.from(state.jobs.values()).some(isActive)) startPolling();
  }

  async function loadSystemInfo() {
    let info;
    try {
      const response = await api("/api/system");
      info = await response.json();
    } catch (e) {
      return; // keep the page defaults
    }
    if (!info || typeof info !== "object") return;
    const model = info.default_model;
    const known = Array.from(dom.model.options).some((option) => option.value === model);
    if (!state.modelTouched && typeof model === "string" && known) dom.model.value = model;
    dom.modelGpu.hidden = info.cuda !== false;
  }

  function isActive(job) {
    return job.status === "QUEUED" || job.status === "RUNNING";
  }

  function startPolling() {
    if (state.pollTimer !== null || state.polling) return;
    state.pollFailures = 0;
    state.pollTimer = window.setTimeout(pollOnce, POLL_MS);
  }

  async function pollOnce() {
    state.pollTimer = null;
    state.polling = true;
    try {
      await pollActive();
    } finally {
      state.polling = false;
    }
  }

  async function pollActive() {
    const active = Array.from(state.jobs.values()).filter(isActive);
    if (!active.length) return;
    let failed = false;
    await Promise.all(active.map(async (job) => {
      try {
        const response = await api("/api/jobs/" + job.id);
        applyStatus(job, await response.json());
      } catch (e) {
        if (e.status === 404) {
          job.status = "ERROR";
          job.error = e.message;
          renderJob(job);
        } else {
          failed = true;
        }
      }
    }));
    state.pollFailures = failed ? state.pollFailures + 1 : 0;
    if (failed && state.pollFailures >= MAX_POLL_FAILURES) {
      showAlert(t("err.pollFailed"));
      return; // stop; starting another job (or reloading) resumes polling
    }
    if (Array.from(state.jobs.values()).some(isActive)) {
      state.pollTimer = window.setTimeout(pollOnce, POLL_MS);
    }
  }

  function applyStatus(job, data) {
    const previous = job.status;
    job.status = STATUSES.includes(data.status) ? data.status : job.status;
    job.progress = clamp(Number(data.progress) || 0, 0, 1);
    job.phase = PHASES.includes(data.phase) ? data.phase : null;
    job.error = typeof data.error === "string" && data.error ? data.error : null;
    if (typeof data.video_name === "string" && data.video_name) job.name = data.video_name;
    const bucket = Math.floor(job.progress * 4);
    if (job.status !== previous || (job.status === "RUNNING" && bucket !== job.announced)) {
      job.announced = bucket;
      announce(t("status.line", {
        name: job.name, status: t("status." + job.status),
        percent: Math.round(job.progress * 100),
      }));
    }
    renderJob(job);
    if (job.status === "DONE") fetchResult(job);
  }

  async function fetchResult(job) {
    if (job.result || job.fetching) return;
    job.fetching = true;
    job.resultFailed = false;
    renderJob(job);
    try {
      const response = await api("/api/jobs/" + job.id + "/result");
      job.result = await response.json();
      job.resultFailed = false;
    } catch (e) {
      job.resultFailed = true;
      showAlert(t("err.resultFailed", { name: job.name }) + " " + e.message);
    } finally {
      job.fetching = false;
      renderJob(job);
    }
  }

  async function deleteJob(job) {
    if (!window.confirm(t("result.deleteConfirm"))) return;
    // Release the MP4 first: on Windows an open file cannot be deleted by the server.
    const video = job.videoEl;
    if (video) {
      video.pause();
      video.removeAttribute("src");
      video.load();
    }
    await new Promise((resolve) => window.setTimeout(resolve, 0));
    try {
      await api("/api/jobs/" + job.id, { method: "DELETE" });
    } catch (e) {
      if (e.status !== 404) { // 404: already gone on the server, just drop the card
        if (video) video.setAttribute("src", "/api/jobs/" + job.id + "/video"); // still there
        showAlert(e.message);
        return;
      }
    }
    if (job.el) job.el.remove();
    state.jobs.delete(job.id);
  }

  function deleteButton(job) {
    const del = el("button", { type: "button", class: "danger", text: t("result.delete") });
    del.addEventListener("click", () => deleteJob(job));
    return del;
  }

  /* ---------- result rendering ---------- */

  function renderAllJobs() {
    for (const job of state.jobs.values()) renderJob(job);
  }

  function renderJob(job) {
    if (!job.el) {
      job.el = el("article", { class: "card" });
      dom.jobs.append(job.el);
    }
    const card = job.el;
    card.replaceChildren();
    const done = job.status === "DONE" && job.result;
    const title = done && typeof job.result.video_name === "string" ? job.result.video_name : job.name;
    card.setAttribute("aria-label", t("result.card", { name: title }));
    card.append(el("h3", { text: title }));

    if (job.status === "ERROR") {
      card.append(el("p", { class: "err", role: "alert", text: job.error || t("err.generic") }));
      card.append(el("div", { class: "row" }, [deleteButton(job)]));
      return;
    }
    if (job.status !== "DONE") {
      const percent = Math.round(job.progress * 100);
      card.append(
        el("p", { class: "muted", text: t("status.line", {
          name: title, status: t("status." + job.status), percent }) }),
        el("progress", {
          max: "100", value: String(percent), "aria-label": t("progress.label", { name: title }),
        }),
      );
      if (job.status === "RUNNING" && job.phase) {
        card.append(el("p", { class: "muted small", text: t("phase." + job.phase) }));
      }
      return;
    }
    if (!done) {
      if (job.resultFailed && !job.fetching) {
        const retry = el("button", { type: "button", text: t("result.retry") });
        retry.addEventListener("click", () => fetchResult(job));
        card.append(el("p", { class: "err", text: t("err.resultFailed", { name: title }) }), retry);
      } else {
        card.append(el("p", { class: "muted", text: t("status.DONE") }));
      }
      return;
    }
    appendResult(card, job, job.result, title);
  }

  function totalText(result) {
    const texts = result.total_text;
    if (texts && typeof texts === "object") {
      const own = texts[state.lang];
      if (typeof own === "string" && own) return own;
      const other = texts.tr || texts.en;
      if (typeof other === "string") return other;
    }
    return String(Number(result.total_count) || 0);
  }

  function badgeIcon(level) {
    const svg = svgEl("svg", { viewBox: "0 0 16 16", "aria-hidden": "true", focusable: "false" });
    if (level === "HIGH") {
      svg.append(svgEl("path", { d: "M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-1.2 10L3.8 8l1.2-1.2 1.8 1.8 4.2-4.2L12.2 5.6z" }));
    } else if (level === "MEDIUM") {
      svg.append(svgEl("path", { d: "M8 1.5l7 12.5H1zM7.2 6h1.6v4H7.2zm0 5h1.6v1.6H7.2z", "fill-rule": "evenodd" }));
    } else if (level === "LOW") {
      svg.append(svgEl("path", { d: "M3 1h10l2 2v10l-2 2H3l-2-2V3zm1.6 3.2L8 7.6l3.4-3.4 1.2 1.2L9.2 8.8l3.4 3.4-1.2 1.2L8 10 4.6 13.4 3.4 12.2 6.8 8.8 3.4 5.4z", "fill-rule": "evenodd" }));
    } else {
      svg.append(svgEl("path", { d: "M8 1a7 7 0 1 0 0 14A7 7 0 0 0 8 1zm-.8 10h1.6v1.6H7.2zM8 3.6A2.6 2.6 0 0 1 10.6 6.2c0 1.4-1.4 1.8-1.8 2.6v.4H7.2v-.6c0-1.2 1.2-1.6 1.6-2.2A1 1 0 1 0 7 6.2H5.4A2.6 2.6 0 0 1 8 3.6z", "fill-rule": "evenodd" }));
    }
    return svg;
  }

  function directionTotal(result, name) {
    const totals = result.totals && typeof result.totals === "object" ? result.totals : {};
    return Number(totals[name]) || 0;
  }

  function keyValueList(rows) {
    const dl = el("dl", { class: "kv" });
    for (const [label, value] of rows) {
      dl.append(el("dt", { text: label }), el("dd", { text: String(value) }));
    }
    return dl;
  }

  function appendResult(card, job, result, title) {
    card.append(el("p", { class: "total", text: totalText(result) }));

    const level = RELIABILITY.includes(result.reliability) ? result.reliability : "UNKNOWN";
    card.append(el("p", {}, [el("span", { class: "badge " + level }, [
      badgeIcon(level),
      document.createTextNode(t("rel.badge", { level: t("rel." + level) })),
    ])]));

    const duration = Number(result.duration_s);
    const pfps = Number(result.processing_fps);
    card.append(el("p", { class: "muted small", text: t("result.meta", {
      model: String(result.model_name || ""),
      duration: Number.isFinite(duration) ? formatTime(duration) : "-",
      fps: Number.isFinite(pfps) ? pfps.toFixed(1) : "-",
    }) }));

    card.append(
      el("h4", { text: t("result.byDirection") }),
      keyValueList([
        [t("line.dir1"), directionTotal(result, "Yön 1")],
        [t("line.dir2"), directionTotal(result, "Yön 2")],
        [t("result.total"), Number(result.total_count) || 0],
      ]),
    );

    const byClass = result.by_class && typeof result.by_class === "object" ? result.by_class : {};
    card.append(
      el("h4", { text: t("result.byClass") }),
      keyValueList(CLASSES.map((c) => [t("class." + c), Number(byClass[c]) || 0])),
    );

    card.append(el("h4", { text: t("result.timeline") }), buildChart(result.timeline, title));
    card.append(buildLegend());

    const flags = Array.isArray(result.flags) ? result.flags : [];
    card.append(buildFlags(job, flags));
    card.append(buildVideo(job, title));

    const actions = el("div", { class: "row" });
    actions.append(
      el("a", { class: "button", href: "/api/jobs/" + job.id + "/video", download: "",
        text: t("result.downloadVideo") }),
      el("a", { class: "button", href: "/api/jobs/" + job.id + "/csv", download: "",
        text: t("result.downloadCsv") }),
      el("a", { class: "button", href: "/api/jobs/" + job.id + "/summary", download: "",
        text: t("result.downloadSummary") }),
    );
    actions.append(deleteButton(job));
    card.append(actions);
  }

  function buildVideo(job, title) {
    if (!job.videoEl) {
      job.videoEl = el("video", {
        controls: "", preload: "metadata", src: "/api/jobs/" + job.id + "/video",
      });
    }
    job.videoEl.setAttribute("aria-label", t("result.videoLabel", { name: title }));
    return job.videoEl;
  }

  function buildFlags(job, flags) {
    const details = el("details");
    details.append(el("summary", { text: t("flags.heading", { count: flags.length }) }));
    if (!flags.length) {
      details.append(el("p", { class: "muted", text: t("flags.none") }));
      return details;
    }
    details.append(el("p", { class: "muted small", text: t("flags.help") }));
    const list = el("ul", { class: "flags" });
    for (const flag of flags) {
      const kind = String(flag.kind);
      const kindLabel = FLAG_KINDS.includes(kind) ? t("flag." + kind) : kind;
      const seconds = Number(flag.time_s) || 0;
      const time = formatTime(seconds);
      const button = el("button", {
        type: "button",
        "aria-label": t("flags.seek", { time, kind: kindLabel }),
      }, [
        el("span", { class: "time", text: time }),
        el("span", { text: kindLabel }),
      ]);
      if (typeof flag.detail === "string" && flag.detail) {
        button.append(el("span", { class: "detail", text: flag.detail }));
      }
      if (flag.counted === false || kind === MISSED_KIND) {
        button.append(el("span", { class: "nc", text: t("flags.notCounted") }));
      }
      button.addEventListener("click", () => seekTo(job, seconds));
      list.append(el("li", {}, [button]));
    }
    details.append(list);
    return details;
  }

  function seekTo(job, seconds) {
    const video = job.videoEl;
    if (!video) return;
    try { video.currentTime = seconds; } catch (e) { /* metadata not ready yet */ }
    const started = video.play();
    if (started && typeof started.catch === "function") started.catch(() => {});
    video.scrollIntoView({ block: "nearest" });
  }

  function buildLegend() {
    const item = (cls, key) => el("li", {}, [
      el("span", { class: "sw " + cls }), document.createTextNode(t(key)),
    ]);
    return el("ul", { class: "legend" }, [item("s1", "line.dir1"), item("s2", "line.dir2")]);
  }

  function buildChart(timeline, title) {
    const bins = (Array.isArray(timeline) ? timeline : []).map((b, i) => ({
      minute: Number.isFinite(Number(b.minute)) ? Number(b.minute) : i,
      one: Math.max(0, Number(b.one) || 0),
      two: Math.max(0, Number(b.two) || 0),
    }));
    const W = 600, H = 230;
    const left = 52, right = 10, top = 14, bottom = 170;
    const svg = svgEl("svg", {
      class: "chart", viewBox: "0 0 " + W + " " + H, role: "img",
    });
    svg.append(svgEl("title", {}, t("chart.title") + " - " + title));
    svg.append(svgEl("desc", {}, t("chart.desc")));
    if (!bins.length) {
      svg.append(svgEl("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, t("chart.empty")));
      return svg;
    }
    const maxValue = Math.max(1, ...bins.map((b) => Math.max(b.one, b.two)));
    const step = Math.max(1, Math.ceil(maxValue / 4));
    const topValue = step * Math.ceil(maxValue / step);
    const plotW = W - left - right;
    const plotH = bottom - top;
    const yOf = (v) => bottom - (v / topValue) * plotH;

    for (let v = 0; v <= topValue; v += step) {
      const y = yOf(v);
      svg.append(svgEl("line", { class: v === 0 ? "axis" : "grid", x1: left, x2: W - right, y1: y, y2: y }));
      svg.append(svgEl("text", { x: left - 6, y: y + 4, "text-anchor": "end" }, String(v)));
    }
    svg.append(svgEl("line", { class: "axis", x1: left, x2: left, y1: top, y2: bottom }));

    const group = plotW / bins.length;
    const barW = Math.max(2, Math.min(22, group * 0.36));
    const labelEvery = Math.max(1, Math.ceil(bins.length / 12));
    bins.forEach((bin, i) => {
      const center = left + group * (i + 0.5);
      const minuteNo = bin.minute + 1;
      const bars = [
        ["bar1", center - barW - 0.5, bin.one, "line.dir1"],
        ["bar2", center + 0.5, bin.two, "line.dir2"],
      ];
      for (const [cls, x, value, dirKey] of bars) {
        const barHeight = bottom - yOf(value);
        const rect = svgEl("rect", {
          class: cls, x, y: yOf(value), width: barW, height: Math.max(barHeight, 0),
        });
        rect.append(svgEl("title", {}, t("chart.bar", {
          minute: minuteNo, dir: t(dirKey), count: value,
        })));
        svg.append(rect);
        if (value > 0 && bins.length <= 12) {
          svg.append(svgEl("text", { x: x + barW / 2, y: yOf(value) - 3, "text-anchor": "middle" }, String(value)));
        }
      }
      if (i % labelEvery === 0) {
        svg.append(svgEl("text", { x: center, y: bottom + 16, "text-anchor": "middle" }, String(minuteNo)));
      }
    });
    svg.append(svgEl("text", { x: left + plotW / 2, y: bottom + 40, "text-anchor": "middle" }, t("chart.minute")));
    svg.append(svgEl("text", {
      x: 0, y: 0, "text-anchor": "middle",
      transform: "translate(12 " + (top + plotH / 2) + ") rotate(-90)",
    }, t("chart.count")));
    return svg;
  }

  /* ---------- wiring ---------- */

  function init() {
    state.lang = initialLang();
    applyStaticText();

    dom.lang.addEventListener("change", () => setLang(dom.lang.value));
    dom.flip.addEventListener("click", flipSelected);
    dom.reset.addEventListener("click", resetSelected);
    dom.start.addEventListener("click", startCounting);
    dom.files.addEventListener("change", () => {
      const picked = Array.from(dom.files.files);
      dom.files.value = "";
      addFiles(picked);
    });
    dom.model.addEventListener("change", () => { state.modelTouched = true; });
    dom.conf.addEventListener("change", syncConfidenceFromNumber);
    dom.confRange.addEventListener("input", () => { dom.conf.value = dom.confRange.value; });

    dom.drop.addEventListener("dragover", (e) => {
      e.preventDefault();
      dom.drop.classList.add("over");
    });
    dom.drop.addEventListener("dragleave", () => dom.drop.classList.remove("over"));
    dom.drop.addEventListener("drop", (e) => {
      e.preventDefault();
      dom.drop.classList.remove("over");
      if (e.dataTransfer) addFiles(e.dataTransfer.files);
    });
    // A file dropped outside the drop zone must not navigate the page away.
    window.addEventListener("dragover", (e) => e.preventDefault());
    window.addEventListener("drop", (e) => e.preventDefault());

    dom.frame.addEventListener("load", drawLine);
    dom.frame.addEventListener("error", () => {
      if (editor.loadedId) showAlert(t("err.previewFailed"));
    });
    if (typeof ResizeObserver === "function") {
      new ResizeObserver(() => drawLine()).observe(dom.stage);
    } else {
      window.addEventListener("resize", drawLine);
    }

    renderVideos();
    renderEditor();
    updateStartButton();
    loadSystemInfo();
    loadExistingJobs();
  }

  init();
}());
