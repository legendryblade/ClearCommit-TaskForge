/* ClearCommit TaskForge frontend. Renders only what Python sends. */
"use strict";

const $ = (id) => document.getElementById(id);
const feed = $("feed");
const clarityLists = [$("clarity-list"), $("clarity-list-2")];
const pendingByUid = new Map();   // uid -> caption element awaiting translation
const orphanTrans = new Map();    // uid -> translation that arrived before its original
const findings = new Map();
let sessionState = "idle";

/* ---------- event plumbing (Python pushes here) ---------- */
window.__push = function (payload) {
  try {
    switch (payload.type) {
      case "caption": return onCaption(payload);
      case "status": return onStatus(payload);
      case "clarity": return onClarity(payload.finding);
      case "clarity_resolved": return onClarityResolved(payload);
      case "error": return showError(payload.message);
      case "hello": return onHello(payload);
    }
  } catch (err) {
    console.error("event handling failed", err);
  }
};

function onHello(payload) {
  const creds = payload.credentials || {};
  const label = (v, name) => (v ? "✓ " : "✗ ") + name;
  $("credentials-note").textContent =
    label(creds.groq, "Groq") + " · " + label(creds.deepgram, "Deepgram") +
    " · " + label(creds.elevenlabs, "ElevenLabs");
  for (const name of ["groq", "deepgram", "elevenlabs", "openrouter"]) {
    const el = $("st-" + name);
    if (el) el.textContent = creds[name] ? "— set ✓" : "— not set";
    if (el) el.style.color = creds[name] ? "var(--green)" : "var(--dim)";
  }
  $("clarity-off").classList.toggle("hidden", !!payload.clarity_provider);
  $("clarity-off-2").classList.toggle("hidden", !!payload.clarity_provider);
  const p1 = $("p1-lang"), ul = $("user-lang");
  // Preserve the user's selections across hello refreshes; only default on
  // the very first population.
  const keepP1 = p1.value, keepUl = ul.value;
  if (!p1.options.length) {
    p1.innerHTML = ""; ul.innerHTML = "";
    for (const code of payload.languages || []) {
      p1.add(new Option(code, code));
      ul.add(new Option(code, code));
    }
    p1.add(new Option("Auto-detect source speech", "auto"));
  }
  const restore = (sel, value, fallback) => {
    const exists = Array.from(sel.options).some((o) => o.value === value);
    sel.value = exists ? value : fallback;
  };
  restore(p1, keepP1 || "auto", "auto");
  restore(ul, keepUl || "en", "en");
  updateLangPair();
}

function onStatus(payload) {
  sessionState = payload.state;
  const pill = $("status-pill");
  pill.dataset.state = payload.state;
  const text = { idle: "Idle", starting: "Starting…", running: "Listening",
      paused: "View paused", stopped: "Stopped",
      error: "Error" }[payload.state] || payload.state;
  pill.textContent = text;
  $("side-dot").dataset.state = payload.state;
  $("side-state").textContent = text;
  $("side-status").textContent =
    payload.message || { idle: "Idle — pick devices, then start.",
      running: "Listening — speak naturally.",
      paused: "View paused — audio continues.",
      stopped: "Stopped." }[payload.state] || "";
  $("start-btn").disabled = payload.state === "running" ||
    payload.state === "starting";
  $("stop-btn").disabled = payload.state === "idle" ||
    payload.state === "stopped";
  $("pause-btn").disabled = payload.state !== "running" &&
    payload.state !== "paused";
  $("pause-btn").textContent = payload.state === "paused"
    ? "▶ Resume view" : "⏸ Pause view";
  if (payload.state === "error" && payload.message) {
    showError(payload.message);  // always visible, from any view
  }
}

function showError(message) {
  const toast = $("toast");
  toast.textContent = message;
  toast.classList.remove("hidden");
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => toast.classList.add("hidden"), 9000);
}

/* ---------- captions ---------- */
function onCaption(ev) {
  $("empty-state").classList.add("hidden");
  if (ev.kind === "orig") {
    const el = document.createElement("article");
    el.className = "cap " + ev.side;
    el.dataset.uid = String(ev.uid);
    el.innerHTML =
      '<div class="cap-head"><span class="cap-who"></span>' +
      '<span class="cap-meta"></span><span class="cap-ts"></span></div>' +
      '<p class="cap-orig"></p><p class="cap-trans hidden"></p>';
    el.querySelector(".cap-who").textContent =
      ev.side === "you" ? "You" : "Person 1";
    el.querySelector(".cap-meta").textContent = ev.meta || "";
    el.querySelector(".cap-ts").textContent =
      new Date().toLocaleTimeString();
    el.querySelector(".cap-orig").textContent = ev.text;
    feed.appendChild(el);
    const early = orphanTrans.get(ev.uid);
    if (early) {   // translation arrived first (parallel workers) — merge now
      const t = el.querySelector(".cap-trans");
      t.textContent = early.text;
      t.classList.remove("hidden");
      orphanTrans.delete(ev.uid);
    } else {
      pendingByUid.set(ev.uid, el);
    }
    feed.scrollTop = feed.scrollHeight;
  } else if (ev.kind === "trans") {
    const el = pendingByUid.get(ev.uid);
    if (!el) {
      // Never drop a translation silently: hold it until the original lands.
      orphanTrans.set(ev.uid, ev);
      if (orphanTrans.size > 100) {
        orphanTrans.delete(orphanTrans.keys().next().value);
      }
      return;
    }
    const t = el.querySelector(".cap-trans");
    t.textContent = ev.text;
    t.classList.remove("hidden");
    pendingByUid.delete(ev.uid);
    feed.scrollTop = feed.scrollHeight;
  }
}

/* ---------- clarity (rendered into both placements) ---------- */
function onClarity(f) {
  findings.set(f.finding_id, f);
  $("clarity-empty").classList.add("hidden");
  $("clarity-empty-2").classList.add("hidden");
  const plain = [];
  for (const list of clarityLists) {
    const existing = list.querySelector('[data-id="' + f.finding_id + '"]');
    if (existing) existing.remove();
    const card = document.createElement("article");
    card.className = "finding";
    card.dataset.id = f.finding_id;
    card.innerHTML =
      '<div class="speaker"></div>' +
      '<div class="quote"></div>' +
      '<div class="why"></div>' +
      '<ul class="interps"></ul>' +
      '<div class="conf"><span>model confidence</span>' +
      '<span class="conf-bar"><span class="conf-fill"></span></span>' +
      '<span class="conf-num"></span></div>' +
      '<label>Suggested clarification (editable)</label>' +
      '<textarea rows="3"></textarea>' +
      '<div class="actions"><button class="btn primary approve">Approve</button>' +
      '<button class="btn dismiss">Dismiss</button></div>';
    card.querySelector(".speaker").textContent =
      "From " + (f.speaker === "you" ? "you" : "Person 1") +
      " — “" + f.quote + "”";
    card.querySelector(".why").textContent = f.why;
    const ul = card.querySelector(".interps");
    for (const interp of f.interpretations || []) {
      const li = document.createElement("li");
      li.textContent = interp;
      ul.appendChild(li);
    }
    card.querySelector(".conf-fill").style.width =
      Math.round(f.confidence * 100) + "%";
    card.querySelector(".conf-num").textContent =
      Math.round(f.confidence * 100) + "%";
    card.querySelector("textarea").value = f.suggested_clarification;
    list.prepend(card);
    plain.push(card);
  }
  updateBadge();
}

function onClarityResolved(p) {
  for (const card of document.querySelectorAll(
      '.finding[data-id="' + p.finding_id + '"]')) {
    if (p.status === "approved") {
      card.classList.add("approved");
      const note = document.createElement("div");
      note.className = "approved-note";
      note.textContent = "Approved — shown here only, not sent or spoken.";
      const actions = card.querySelector(".actions");
      if (actions) actions.replaceWith(note);
    } else {
      card.remove();
    }
  }
  const f = findings.get(p.finding_id);
  if (f) f.status = p.status;
  updateBadge();
}

function updateBadge() {
  let n = 0;
  findings.forEach((f) => { if (f.status === "pending") n += 1; });
  const badge = $("clarity-badge");
  badge.textContent = String(n);
  badge.classList.toggle("hidden", n === 0);
}

/* delegated clicks work in both clarity placements */
document.addEventListener("click", (ev) => {
  const card = ev.target.closest(".finding");
  if (!card || !window.pywebview || !window.pywebview.api) return;
  const id = card.dataset.id;
  if (ev.target.classList.contains("approve")) {
    pywebview.api.clarify_approve(id, card.querySelector("textarea").value);
  } else if (ev.target.classList.contains("dismiss")) {
    pywebview.api.clarify_dismiss(id);
  }
});

/* ---------- controls ---------- */
async function refreshDevices() {
  if (!window.pywebview || !window.pywebview.api) {
    setTimeout(refreshDevices, 200); return;
  }
  try {
    const data = JSON.parse(await pywebview.api.list_audio_devices());
    const inSel = $("input-device"), outSel = $("output-device");
    inSel.innerHTML = ""; outSel.innerHTML = "";
    if (!(data.inputs || []).length) {
      inSel.add(new Option("No input devices found", "-1"));
      showError("No microphone/capture devices were found. " +
                (data.error || "Check your audio drivers."));
    }
    for (const d of data.inputs || []) inSel.add(new Option(d.name, d.index));
    for (const d of data.outputs || []) outSel.add(new Option(d.name, d.index));
    if (data.default_input >= 0) inSel.value = String(data.default_input);
    if (data.default_output >= 0) outSel.value = String(data.default_output);
    $("device-status").textContent =
      (data.inputs || []).length + " input · " +
      (data.outputs || []).length + " output devices" +
      (data.error ? " — " + data.error : "");
  } catch (err) {
    showError("Could not enumerate audio devices: " + err);
  }
}

async function startSession() {
  if (!window.pywebview || !window.pywebview.api) {
    showError("The Python bridge is not ready yet. Try again in a moment.");
    return;
  }
  onStatus({ state: "starting", message: "Opening audio devices…" });
  try {
    const res = JSON.parse(await pywebview.api.start_session({
      person1_lang: $("p1-lang").value,
      user_lang: $("user-lang").value,
      input_device: Number($("input-device").value),
      output_device: Number($("output-device").value),
      stt_backend: $("stt-backend").value,
      tts_mode: $("tts-mode").value,
      monitor_tts: $("monitor-tts").checked,
    }));
    if (!res.ok) showError(res.error || "Could not start the session.");
  } catch (err) {
    showError("Start failed: " + err);
    onStatus({ state: "error", message: String(err) });
  }
}

function updateLangPair() {
  const p1 = $("p1-lang").value === "auto"
    ? "auto-detect" : $("p1-lang").value;
  $("lang-pair").textContent = ($("user-lang").value || "—") + " ⇄ " + p1;
}

/* ---------- navigation ---------- */
document.querySelectorAll(".nav-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach((b) => {
      b.classList.remove("active"); b.removeAttribute("aria-current");
    });
    btn.classList.add("active"); btn.setAttribute("aria-current", "page");
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.remove("active"));
    $("view-" + btn.dataset.view).classList.add("active");
  });
});

$("empty-start-btn").addEventListener("click", () =>
  document.querySelector('.nav-btn[data-view="audio"]').click());
$("refresh-devices").addEventListener("click", refreshDevices);
$("start-btn").addEventListener("click", startSession);
$("stop-btn").addEventListener("click", async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  const res = JSON.parse(await pywebview.api.stop_session());
  if (!res.ok) showError(res.error || "Stop failed.");
});
$("pause-btn").addEventListener("click", async () => {
  if (!window.pywebview || !window.pywebview.api) return;
  await pywebview.api.set_paused(sessionState !== "paused");
});
$("p1-lang").addEventListener("change", updateLangPair);
$("user-lang").addEventListener("change", updateLangPair);

async function saveKeys() {
  if (!window.pywebview || !window.pywebview.api) return;
  const payload = {};
  for (const name of ["groq", "deepgram", "elevenlabs", "openrouter"]) {
    const v = $("key-" + name).value.trim();
    if (v) payload[name] = v;
  }
  if (!Object.keys(payload).length) {
    $("keys-help").textContent = "No new keys entered — nothing to save.";
    return;
  }
  try {
    const res = JSON.parse(await pywebview.api.save_keys(payload));
    if (res.ok) {
      for (const name of Object.keys(payload)) $("key-" + name).value = "";
      $("keys-help").textContent = res.message || "Keys saved.";
    } else {
      $("keys-help").textContent = res.error || "Could not save keys.";
    }
  } catch (err) {
    $("keys-help").textContent = "Save failed: " + err;
  }
}
$("save-keys-btn").addEventListener("click", saveKeys);

/* ---------- boot ---------- */
async function boot() {
  let tries = 0;
  while ((!window.pywebview || !window.pywebview.api) && tries < 100) {
    await new Promise((r) => setTimeout(r, 120));
    tries += 1;
  }
  if (!window.pywebview || !window.pywebview.api) {
    showError("Could not reach the Python bridge. Restart the app.");
    return;
  }
  try {
    window.__push(JSON.parse(await pywebview.api.hello()));
  } catch (err) {
    showError("Bridge hello failed: " + err);
  }
  refreshDevices();
  onStatus({ state: "idle" });
}
boot();
