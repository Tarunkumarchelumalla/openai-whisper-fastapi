const API = "/api/v1";
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

let cfg = { extensions: [".wav", ".mp3", ".mp4"], max_upload_mb: 500, models: [], default_model: "small" };
let pending = [];          // files picked but not yet uploaded
let jobs = [];
let filter = "";
const UPLOAD_CONCURRENCY = 3;

// ---------------- helpers ----------------
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.add("show");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => t.classList.remove("show"), 2600);
}
function fmtBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1048576) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1048576).toFixed(1)} MB`;
}
function fmtDur(sec) {
  if (sec == null) return "";
  sec = Math.round(sec);
  const m = Math.floor(sec / 60), s = sec % 60;
  return m ? `${m}m ${String(s).padStart(2, "0")}s` : `${s}s`;
}
function fmtClock(sec) {
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
}
function ago(ts) {
  const d = Date.now() / 1000 - ts;
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)}m ago`;
  if (d < 86400) return `${Math.floor(d / 3600)}h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}
const ICONS = {
  audio: '<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M12 3v10.6A4 4 0 1 0 14 17V7h4V3z"/></svg>',
  video: '<svg viewBox="0 0 24 24" width="18" height="18"><path fill="currentColor" d="M4 5h11a1 1 0 0 1 1 1v3.5l4-3v11l-4-3V18a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1"/></svg>',
};

// ---------------- file picking ----------------
const drop = $("#drop");
$("#file").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => addFiles(e.dataTransfer.files));

function addFiles(list) {
  for (const f of list) {
    const ext = "." + f.name.split(".").pop().toLowerCase();
    let error = null;
    if (!cfg.extensions.includes(ext)) error = `Unsupported type (${cfg.extensions.join(", ")})`;
    else if (f.size > cfg.max_upload_mb * 1048576) error = `Over ${cfg.max_upload_mb} MB limit`;
    else if (f.size === 0) error = "File is empty";
    pending.push({ id: crypto.randomUUID(), file: f, progress: 0, state: error ? "invalid" : "ready", msg: error });
  }
  renderPending();
}

function renderPending() {
  $("#pending").innerHTML = pending.map((p) => {
    const cls = p.state === "done" ? "done" : p.state === "error" || p.state === "invalid" ? "fail" : "";
    const meta = p.msg ? `<div class="qmeta err">${esc(p.msg)}</div>` :
      `<span class="qmeta">${p.state === "uploading" ? Math.round(p.progress * 100) + "%" : p.state === "done" ? "queued ✓" : fmtBytes(p.file.size)}</span>`;
    return `<li><div class="qrow"><span class="qname" title="${esc(p.file.name)}">${esc(p.file.name)}</span>${p.msg ? "" : meta}</div>
      ${p.msg ? meta : `<div class="bar ${cls}"><i style="width:${p.progress * 100}%"></i></div>`}</li>`;
  }).join("");
  const ready = pending.filter((p) => p.state === "ready").length;
  $("#uploadBtn").disabled = !ready;
  $("#uploadBtn").textContent = ready > 1 ? `Upload & transcribe (${ready})` : "Upload & transcribe";
  $("#clearBtn").hidden = !pending.length;
}
$("#clearBtn").addEventListener("click", () => {
  pending = pending.filter((p) => p.state === "uploading");
  renderPending();
});

// ---------------- uploading (parallel, one request per file for per-file progress) ----------------
function uploadOne(p) {
  return new Promise((resolve) => {
    const fd = new FormData();
    fd.append("files", p.file);
    fd.append("model", $("#model").value);
    const lang = $("#language").value.trim();
    if (lang) fd.append("language", lang);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API}/transcriptions`);
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { p.progress = e.loaded / e.total; renderPending(); } };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch {}
      if (xhr.status === 202) {
        p.state = "done"; p.progress = 1;
        if (body.jobs?.[0]?.deduplicated) p.msg = "Already transcribed — reused existing result";
      } else {
        p.state = "error";
        p.msg = body.errors?.[0]?.error || body.detail || `Upload failed (${xhr.status})`;
      }
      renderPending(); refresh(); resolve();
    };
    xhr.onerror = () => { p.state = "error"; p.msg = "Network error"; renderPending(); resolve(); };
    p.state = "uploading";
    renderPending();
    xhr.send(fd);
  });
}
$("#uploadBtn").addEventListener("click", async () => {
  const todo = pending.filter((p) => p.state === "ready");
  const workers = Array.from({ length: Math.min(UPLOAD_CONCURRENCY, todo.length) }, async () => {
    while (todo.length) await uploadOne(todo.shift());
  });
  await Promise.all(workers);
  setTimeout(() => { pending = pending.filter((p) => p.state !== "done" || p.msg); renderPending(); }, 2500);
});

// ---------------- jobs list ----------------
document.querySelectorAll("#filters .chip").forEach((c) =>
  c.addEventListener("click", () => {
    filter = c.dataset.f;
    document.querySelectorAll("#filters .chip").forEach((x) => x.classList.toggle("on", x === c));
    renderJobs();
  })
);

function statusPill(j) {
  let label = j.status;
  if (j.status === "processing" && j.stage) label = j.stage;
  if (j.status === "retrying" && j.next_run_at) {
    const s = Math.max(0, Math.round(j.next_run_at - Date.now() / 1000));
    label = `retrying in ${s}s`;
  }
  return `<span class="pill ${j.status}"><span class="dot"></span>${esc(label)}</span>`;
}

function renderJobs() {
  const shown = jobs.filter((j) =>
    !filter ? true : filter === "processing" ? ["queued", "processing", "retrying"].includes(j.status) : j.status === filter
  );
  if (!shown.length) {
    $("#jobList").innerHTML = `<p class="empty">${jobs.length ? "Nothing in this view." : "No jobs yet. Upload a file to start."}</p>`;
    return;
  }
  $("#jobList").innerHTML = shown.map((j) => {
    const meta = [
      fmtBytes(j.size_bytes),
      j.model,
      j.duration_sec ? `${fmtDur(j.duration_sec)} media` : "",
      j.processing_sec ? `took ${fmtDur(j.processing_sec)}` : "",
      j.attempts > 1 || j.status === "retrying" ? `attempt ${j.attempts}/${j.max_attempts}` : "",
      ago(j.created_at),
    ].filter(Boolean).map(esc).join(" · ");
    const actions = [
      j.status === "completed" ? `<button class="btn sm" data-act="view" data-id="${j.id}">View</button>` : "",
      j.status === "failed" ? `<button class="btn sm" data-act="retry" data-id="${j.id}">Retry</button>` : "",
      j.status !== "processing" ? `<button class="btn sm ghost danger" data-act="del" data-id="${j.id}" aria-label="Delete">Delete</button>` : "",
    ].join("");
    return `<div class="job">
      <div class="kind" title="${j.media_kind}">${ICONS[j.media_kind]}</div>
      <div class="min0">
        <div class="jname" title="${esc(j.filename)}">${esc(j.filename)}</div>
        <div class="jmeta">${statusPill(j)}<span>${meta}</span></div>
        ${j.last_error && j.status !== "completed" ? `<div class="jerr" title="${esc(j.last_error)}">${esc(j.last_error)}</div>` : ""}
      </div>
      <div class="jactions">${actions}</div>
    </div>`;
  }).join("");
}

$("#jobList").addEventListener("click", async (e) => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const { act, id } = b.dataset;
  if (act === "view") return openDetail(id);
  if (act === "retry") {
    const r = await fetch(`${API}/transcriptions/${id}/retry`, { method: "POST" });
    toast(r.ok ? "Re-queued" : "Could not retry");
  }
  if (act === "del") {
    if (!confirm("Delete this job, its media and transcript?")) return;
    const r = await fetch(`${API}/transcriptions/${id}`, { method: "DELETE" });
    toast(r.ok ? "Deleted" : "Could not delete");
  }
  refresh();
});

function renderStats(s, workers) {
  const items = [
    ["queued", "Queued", "var(--muted)"],
    ["processing", "Processing", "var(--info)"],
    ["retrying", "Retrying", "var(--warn)"],
    ["completed", "Completed", "var(--ok)"],
    ["failed", "Failed", "var(--err)"],
  ];
  $("#stats").innerHTML = items.map(([k, label, c]) =>
    `<div class="stat"><b>${s[k] ?? 0}</b><span><i class="dot" style="color:${c}"></i>${label}</span></div>`
  ).join("") + `<div class="stat"><b>${workers}</b><span>Workers</span></div>`;
}

let workers = 0;
async function refresh() {
  try {
    const r = await fetch(`${API}/transcriptions?limit=200`);
    const body = await r.json();
    // Only touch the DOM when something changed (retry countdowns tick, so always redraw those).
    const sig = JSON.stringify(body);
    if (sig === refresh.last && !body.jobs.some((j) => j.status === "retrying")) return;
    refresh.last = sig;
    jobs = body.jobs;
    renderStats(body.stats, workers);
    renderJobs();
  } catch { /* server restarting; next poll will catch up */ }
}

// ---------------- detail ----------------
const dlg = $("#detail");
$("#dClose").addEventListener("click", () => dlg.close());
dlg.addEventListener("close", () => { $("#dPlayer").innerHTML = ""; });
dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });

async function openDetail(id) {
  const job = jobs.find((j) => j.id === id);
  const t = await (await fetch(`${API}/transcriptions/${id}/transcript`)).json();
  const base = `${API}/transcriptions/${id}`;
  $("#dTitle").textContent = job.filename;
  $("#dMeta").textContent = [
    `Language: ${t.language || "?"}`, `Model: ${t.model}`, t.duration_sec ? `Duration ${fmtDur(t.duration_sec)}` : "",
    `Processed in ${fmtDur(t.processing_sec)} on ${t.device}`, `${t.segments.length} segments`,
  ].filter(Boolean).join(" · ");

  const media = job.media_kind === "video"
    ? `<video controls preload="metadata" src="${base}/media"><track default kind="captions" srclang="${esc(t.language || "en")}" label="Transcript" src="${base}/transcript?format=vtt"></video>`
    : `<audio controls preload="metadata" src="${base}/media"></audio>`;
  $("#dPlayer").innerHTML = media;

  $("#dActions").innerHTML = ["txt", "srt", "vtt", "json"].map((f) =>
    `<a class="btn sm" href="${base}/transcript?format=${f}" download="${esc(job.filename.replace(/\.[^.]+$/, ""))}.${f}">⬇ ${f.toUpperCase()}</a>`
  ).join("") + `<button class="btn sm" id="copyBtn">Copy text</button>`;
  $("#copyBtn").onclick = () => navigator.clipboard.writeText(t.text).then(() => toast("Copied"));

  $("#dSegments").innerHTML = t.segments.length
    ? t.segments.map((s) => `<div class="seg" data-start="${s.start}" data-end="${s.end}"><time>${fmtClock(s.start)}</time><span>${esc(s.text)}</span></div>`).join("")
    : `<div class="seg-empty">${esc(t.text) || "No speech detected."}</div>`;

  const player = $("#dPlayer").firstElementChild;
  $("#dSegments").onclick = (e) => {
    const seg = e.target.closest(".seg");
    if (seg) { player.currentTime = +seg.dataset.start; player.play(); }
  };
  player.ontimeupdate = () => {
    const now = player.currentTime;
    document.querySelectorAll(".seg").forEach((el) => {
      const on = now >= +el.dataset.start && now < +el.dataset.end;
      if (on && !el.classList.contains("now")) el.scrollIntoView({ block: "nearest" });
      el.classList.toggle("now", on);
    });
  };
  dlg.showModal();
}

// ---------------- boot ----------------
(async () => {
  try {
    cfg = await (await fetch(`${API}/config`)).json();
    workers = (await (await fetch(`${API}/health`)).json()).workers;
  } catch {}
  $("#model").innerHTML = cfg.models.map((m) => `<option ${m === cfg.default_model ? "selected" : ""}>${m}</option>`).join("");
  $("#limits").textContent = `${cfg.extensions.map((e) => e.slice(1).toUpperCase()).join(", ")} · up to ${cfg.max_upload_mb} MB each · multiple files OK`;
  refresh();
  setInterval(refresh, 2000);
})();
