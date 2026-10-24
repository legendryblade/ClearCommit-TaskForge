/* ClearCommit TaskForge frontend. Renders only what Python sends. */
"use strict";

const $ = (id) => document.getElementById(id);
const feed = $("feed");
const clarityList = $("clarity-list");
const pendingByUid = new Map();   // uid -> caption element awaiting translation
const findingsById = new Map();
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
  const bits = [];
  bits.push((creds.groq ? "✓" : "✗") + " Groq key");
  bits.push((creds.deepgram ? "✓" : "✗") + " Deepgram key");
  $("credentials-note").textContent = bits.join(" · ");
  $("clarity-off").classList.toggle("hidden", !!payload.clarity_provider);
  const p1 = $("p1-lang"), ul = $("user-lang");
  p1.innerHTML = ""; ul.innerHTML = "";
  for (const code of payload.languages || []) {
    p1.add(new Option(code, code));
    ul.add(new Option(code, code));
  }
  p1.value = "es"; ul.value = "en";
  updateLangPair();
}

function onStatus(payload) {
  sessionState = payload.state;
  const pill = $("status-pill");
  pill.dataset.state = payload.state;
  pill.textContent =
    { idle: "Idle", starting: "Starting…", running: "Listening",
      paused: "View paused", stopped: "Stopped",
      error: "Error" }[payload.state] || payload.state;
  $("start-btn").disabled = payload.state === "running" ||
    payload.state === "starting";
  $("stop-btn").disabled = payload.state === "idle" ||
    payload.state === "stopped";
  $("pause-btn").disabled = payload.state !== "running" &&
    payload.state !== "paused";
  $("pause-btn").textContent = payload.state === "paused"
    ? "Resume view" : "Pause view";
  if (payload.message) showError(payload.message);
}

function showError(message) {
  const banner = $("banner");
  banner.textContent = message;
  banner.classList.remove("hidden");
}

/* ---------- captions ---------- */
function onCaption(ev) {
  $("empty-state").classList.add("hidden");
  if (ev.kind === "orig") {
    const el = document.createElement("article");
    el.className = "cap " + ev.side;
    const who = ev.side === "you" ? "You" : "Person 1";
    el.innerHTML =
      '<div class="cap-head"><span class="cap-who"></span>' +
      '<span class="cap-meta"></span><span class="cap-ts"></span></div>' +
      '<p class="cap-orig"></p><p class="cap-trans hidden"></p>';
    el.querySelector(".cap-who").textContent = who;
    el.querySelector(".cap-meta").textContent = ev.meta || "";
    el.querySelector(".cap-ts").textContent =
      new Date().toLocaleTimeString();
    el.querySelector(".cap-orig").textContent = ev.text;
    feed.appendChild(el);
    pendingByUid.set(ev.uid, el);
    feed.scrollTop = feed.scrollHeight;
  } else if (ev.kind === "trans") {
    const el = pendingByUid.get(ev.uid);
    if (!el) return;
    const t = el.querySelector(".cap-trans");
    t.textContent = ev.text;
    t.classList.remove("hidden");
    pendingByUid.delete(ev.uid);
    feed.scrollTop = feed.scrollHeight;
  }
}

/* ---------- clarity ---------- */
function onClarity(finding) {
  findingsById.set(finding.finding_id, finding);
  $("clarity-empty").classList.add("hidden");
  const card = document.createElement("article");
  card.className = "finding";
  card.dataset.id = finding.finding_id;
  const li = (finding.interpretations || [])
    .map((x) => '<li class="interp"></li>').join("");
  card.innerHTML =
    '<div class="quote"></div>' +
    '<div class="why"></div>' +
    (li ? '<ul>' + li + '</ul>' : "") +
    '<div class="conf"><span>model confidence</span>' +
    '<span class="conf-bar"><span class="conf-fill"></span></span>' +
    '<span class="conf-num"></span></div>' +
    '<label for="clar-' + finding.finding_id + '">Suggested clarification (editable)</label>' +
    '<textarea id="clar-' + finding.finding_id + '"></textarea>' +
    '<div class="actions"><button class="btn primary approve">Approve</button>' +
    '<button class="btn dismiss">Dismiss</button></div>';
  card.querySelector(".quote").textContent = "“" + finding.quote + "”";
  card.querySelector(".why").textContent = finding.why;
  const items = card.querySelectorAll(".interp");
  (finding.interpretations || []).forEach((x, i) => {
    items[i].textContent = x;
  });
  card.querySelector(".conf-fill").style.width =
    Math.round(finding.confidence * 100) + "%";
  card.querySelector(".conf-num").textContent =
    Math.round(finding.confidence * 100) + "%";
  card.querySelector("textarea").value = finding.suggested_clarification;
  card.querySelector(".approve").addEventListener("click", () => {
    pywebview.api.clarify_approve(finding.finding_id,
      card.querySelector("textarea").value);
  });
  card.querySelector(".dismiss").addEventListener("click", () => {
    pywebview.api.clarify_dismiss(finding.finding_id);
  });
  clarityList.prepend(card);
  updateBadge();
}

function onClarityResolved(payload) {
  const card = clarityList.querySelector(
    '.finding[data-id="' + payload.finding_id + '"]');
  if (!card) return;
  if (payload.status === "approved") {
    card.classList.add("approved");
    const note = document.createElement("div");
    note.className = "approved-note";
    note.textContent = "Approved — shown here only, not sent or spoken.";
    card.querySelector(".actions").replaceWith(note);
  } else {
    card.remove();
  }
  updateBadge();
}

function updateBadge() {
  let n = 0;
  clarityList.querySelectorAll(".finding:not(.approved)").forEach((el) => {
    const f = findingsById.get(el.dataset.id);
    if (f && f.status === "pending") n += 1;
  });
  const badge = $("clarity-badge");
  badge.textContent = String(n);
  badge.classList.toggle("hidden", n === 0);
}

/* ---------- controls ---------- */
async function refreshDevices() {
  try {
    const data = JSON.parse(await pywebview.api.list_audio_devices());
    const inSel = $("input-device"), outSel = $("output-device");
    inSel.innerHTML = ""; outSel.innerHTML = "";
    if (!(data.inputs || []).length) {
      inSel.add(new Option("No input devices found", "-1"));
    }
    for (const d of data.inputs || []) inSel.add(new Option(d.name, d.index));
    for (const d of data.outputs || []) outSel.add(new Option(d.name, d.index));
    if (data.default_input >= 0) inSel.value = String(data.default_input);
    if (data.default_output >= 0) outSel.value = String(data.default_output);
    $("device-status").textContent =
      (data.inputs || []).length + " input · " +
      (data.outputs || []).length + " output devices detected" +
      (data.error ? " — " + data.error : "");
  } catch (err) {
    showError("Could not enumerate audio devices: " + err);
  }
}

async function startSession() {
  const res = JSON.parse(await pywebview.api.start_session({
    person1_lang: $("p1-lang").value,
    user_lang: $("user-lang").value,
    input_device: Number($("input-device").value),
    output_device: Number($("output-device").value),
  }));
  if (!res.ok) showError(res.error || "Could not start the session.");
}

function updateLangPair() {
  $("lang-pair").textContent =
    ($("user-lang").value || "—") + " ⇄ " + ($("p1-lang").value || "—");
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
    if (btn.dataset.view === "conversation") updateBadge();
  });
});

$("empty-start-btn").addEventListener("click", () =>
  document.querySelector('.nav-btn[data-view="audio"]').click());
$("refresh-devices").addEventListener("click", refreshDevices);
$("start-btn").addEventListener("click", startSession);
$("stop-btn").addEventListener("click", () => pywebview.api.stop_session());
$("pause-btn").addEventListener("click", () =>
  pywebview.api.set_paused(sessionState !== "paused"));
$("p1-lang").addEventListener("change", updateLangPair);
$("user-lang").addEventListener("change", updateLangPair);

/* ---------- boot ---------- */
function boot() {
  if (!window.pywebview || !window.pywebview.api) {
    setTimeout(boot, 120);
    return;
  }
  pywebview.api.hello().then((h) => window.__push(JSON.parse(h)));
  refreshDevices();
  onStatus({ state: "idle" });
}
boot();
