"use strict";
// Agent Guard page. Normal mode: scripted customers and "try a refund". Demo mode (?demo=1): a recorded story plays itself.

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
const DEMO = params.get("demo") === "1";
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
let pace = Number(params.get("speed")) || 1;
const wait = (ms) => new Promise((r) => setTimeout(r, REDUCED ? 0 : ms / pace));

const STAMP = { blocked: "Blocked", allowed: "Allowed", person: "To a person" };
const FROM_GUARD = { block: "blocked", allow: "allowed", person: "person" };
const PROB = { supported: "supported", violation: "violation", insufficient_information: "insufficient info", inconsistent_context: "inconsistent" };
const REASON = { change_of_mind: "change of mind", faulty: "faulty item", not_delivered: "did not arrive" };
const EXPECTED_NOTE = {
  allow: "The answer key says this refund should be paid.",
  block: "The answer key says this refund should be blocked.",
  person: "The answer key says a person should check this refund.",
};
const WHAT_HAPPENS = {
  blocked: "The action does not run. The agent tells the customer it cannot do this.",
  allowed: "The action runs.",
  person: "Nothing runs yet. A person checks it first.",
};
const SYMBOL = { GBP: "£", USD: "$", EUR: "€" };

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
function money(value, currency = "GBP", decimals = 0) {
  const n = Number(value).toLocaleString("en-GB", { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
  return SYMBOL[currency] ? SYMBOL[currency] + n : `${n} ${currency}`;
}
async function getJSON(url, options) {
  const res = await fetch(url, options);
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail ? (typeof data.detail === "string" ? data.detail : "Check the form and try again.") : res.statusText);
  return data;
}
const postJSON = (url, body) => getJSON(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

function fit() {
  document.documentElement.style.setProperty("--top-h", document.querySelector(".top").offsetHeight + "px");
}

// ---------- money meters ----------
const shown = { "m-alone": 0, "m-through": 0 };
let currency = "GBP";
function countTo(id, to, ms = 900) {
  const el = $(id), from = shown[id], start = performance.now();
  shown[id] = to;
  if (from !== to) { el.classList.remove("bump"); void el.offsetWidth; el.classList.add("bump"); }
  const step = (now) => {
    const t = REDUCED ? 1 : Math.min(1, (now - start) / (ms / pace));
    el.textContent = money(from + (to - from) * (1 - Math.pow(1 - t, 3)), currency);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------- chat ----------
function addEl(cls, html) {
  const el = document.createElement("div");
  el.className = cls;
  el.innerHTML = html;
  $("chat").appendChild(el);
  $("chat").scrollTop = $("chat").scrollHeight;
  return el;
}
const bubble = (cls, who, text) => addEl(`bubble ${cls}`, `<span class="who">${esc(who)}</span><span class="t">${esc(text)}</span>`);
const dots = () => addEl("dots", "<i></i><i></i><i></i>");
async function typeBubble(cls, who, text, cps = 48) {
  const el = bubble(cls, who, "");
  const t = el.querySelector(".t");
  t.classList.add("caret");
  for (let i = 0; i < text.length; i += 2) {
    t.textContent = text.slice(0, i + 2);
    $("chat").scrollTop = $("chat").scrollHeight;
    await wait(2000 / cps);
  }
  t.classList.remove("caret");
  return el;
}
async function typeText(el, text, cps = 110) {
  el.textContent = "";
  for (let i = 0; i < text.length; i += 3) {
    el.textContent = text.slice(0, i + 3);
    await wait(3000 / cps);
  }
}
function callText(c) {
  if (c.call) return c.call;
  if (c.kind === "refund") {
    const a = c.attempt;
    return `issue_refund(order_id="${a.order_id}", amount_gbp=${a.amount_gbp}, reason="${a.reason}")`;
  }
  const args = Object.entries(c.action.args || {}).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
  return `${c.action.name}(${args})`;
}

// ---------- the gate ----------
const status = (text) => { $("gate-status").textContent = text; };
function resetGate() {
  $("slot-empty").hidden = false;
  $("cheque").hidden = true;
  $("key").hidden = true;
  $("stamp").className = "stamp";
  $("stamp").textContent = "";
  $("g-claim").textContent = "";
  $("g-facts").innerHTML = "";
  $("g-rules").innerHTML = "";
  $("g-verdict").innerHTML = "";
  $("g-bars").innerHTML = Object.keys(PROB).map((k) => `
    <span class="name" data-k="${k}">${PROB[k]}</span>
    <span class="track"><span class="fill ${k}" data-k="${k}"></span></span>
    <span class="num" data-k="${k}">0.00</span>`).join("");
  $("inspect").classList.add("idle");
}
function keyVerdict(truth, outcome) {
  if (truth === "wrong") {
    if (outcome === "allowed") return ["miss", "Answer key: the agent was wrong. The guard missed it."];
    return ["good", outcome === "blocked" ? "Answer key: the agent was wrong. The guard stopped it." : "Answer key: the agent was wrong. A person checks before anything runs."];
  }
  if (outcome === "allowed") return ["good", "Answer key: the agent was right. The guard let it through."];
  if (outcome === "person") return ["slow", "Answer key: the agent was right. The guard sent it to a person: safe, but slow."];
  return ["miss", "Answer key: the agent was right. The guard wrongly blocked it."];
}
async function runGate(c) {
  $("slot-empty").hidden = true;
  $("ch-shop").textContent = c.shop || "Action slip";
  $("ch-by").textContent = c.by || "Written by the agent · not run yet";
  $("ch-do").textContent = c.action.plain || c.action.name;
  $("ch-amt").textContent = c.amount ? money(c.amount.value, c.amount.currency, 2) : "";
  $("ch-micr").textContent = c.micr || callText(c);
  const cheque = $("cheque");
  cheque.hidden = false;
  cheque.classList.remove("enter", "hit"); void cheque.offsetWidth; cheque.classList.add("enter");
  status("The action stops at the gate");
  await wait(900);

  $("inspect").classList.remove("idle");
  status("Building one claim. The chat stays behind.");
  await typeText($("g-claim"), c.claim);
  await wait(250);
  status("Reading the record");
  for (const fact of c.evidence || []) {
    const li = document.createElement("li");
    li.className = "in"; li.textContent = fact; li.title = fact;
    $("g-facts").appendChild(li);
    await wait(120);
  }
  status("Adding the written policy");
  for (const line of c.policy_lines || []) {
    const m = line.match(/^(R\d+)\.\s*(.*)$/);
    const div = document.createElement("div");
    div.className = "rule in"; div.title = line;
    div.innerHTML = m ? `<b>${esc(m[1])}</b>${esc(m[2])}` : esc(line);
    $("g-rules").appendChild(div);
    await wait(110);
  }

  status("Banyan Sapling is checking");
  const scan = $("scan");
  scan.style.setProperty("--scan-to", $("inspect").offsetHeight + "px");
  scan.classList.remove("run"); void scan.offsetWidth; scan.classList.add("run");
  await wait(950);
  const p = c.sapling.probabilities;
  for (const k of Object.keys(PROB)) {
    $("g-bars").querySelector(`.fill[data-k="${k}"]`).style.width = Math.round((p[k] || 0) * 100) + "%";
    countNum($("g-bars").querySelector(`.num[data-k="${k}"]`), p[k] || 0);
  }
  await wait(1200);
  const top = c.sapling.raw_label;
  const topName = $("g-bars").querySelector(`.name[data-k="${top}"]`);
  if (topName) topName.classList.add("lead");
  $("g-verdict").innerHTML = `Top score: <b>${esc(PROB[top] || top)} ${(p[top] || 0).toFixed(2)}</b>. Rule: block on violation or inconsistent, allow on supported, else a person.${c.note ? " " + esc(c.note) : ""}`;

  const stamp = $("stamp");
  stamp.textContent = STAMP[c.outcome];
  stamp.className = `stamp ${c.outcome}`; void stamp.offsetWidth; stamp.classList.add("slam");
  await wait(150);
  cheque.classList.remove("enter"); void cheque.offsetWidth; cheque.classList.add("hit");
  const flash = document.createElement("div");
  flash.className = `flash ${c.outcome}`;
  document.body.appendChild(flash);
  setTimeout(() => flash.remove(), 900);
  status(`${STAMP[c.outcome]} · ${PROB[top] || top} ${(p[top] || 0).toFixed(2)}`);
  await wait(500);

  const key = $("key");
  if (c.truth) {
    const [cls, text] = keyVerdict(c.truth, c.outcome);
    key.className = `key ${cls}`;
    $("key-verdict").textContent = text;
    $("key-note").textContent = c.truth_note || "";
    $("key-note").hidden = !c.truth_note;
  } else {
    key.className = "key";
    $("key-verdict").textContent = `${STAMP[c.outcome]} by the guard`;
    $("key-note").hidden = true;
  }
  $("key-src").textContent = c.source ? `Source: ${c.source}` : "";
  key.hidden = false;
}
function countNum(el, to) {
  const start = performance.now();
  const step = (now) => {
    const t = REDUCED ? 1 : Math.min(1, (now - start) / (1100 / pace));
    el.textContent = (to * (1 - Math.pow(1 - t, 3))).toFixed(2);
    if (t < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

// ---------- ledger ----------
let ledgerCount = 0;
function addLedger(c) {
  const list = $("ledger");
  const empty = list.querySelector(".ledger-empty");
  if (empty) empty.remove();
  const li = document.createElement("li");
  li.className = `entry ${c.outcome}`;
  const tag = c.truth ? (c.truth === "right" ? "agent was right" : "agent was wrong") : "your request";
  li.innerHTML = `<span class="bar"></span><span class="what" title="${esc(c.label)}">${esc(c.label)}</span>
    <span class="amt">${c.amount ? esc(money(c.amount.value, c.amount.currency)) : ""}</span>
    <span class="res"><b>${STAMP[c.outcome]}</b> · ${esc(tag)}</span>`;
  list.prepend(li);
  ledgerCount += 1;
  $("ledger-count").textContent = `${ledgerCount} checked`;
}

// ---------- one turn, after the customer has spoken ----------
async function afterAgent(c, { typing = false } = {}) {
  if (c.agent) {
    if (typing) await typeBubble("agent", "Support agent", c.agent, 60);
    else bubble("agent", "Support agent", c.agent);
  }
  if (!c.action) {
    if (!c.agent) bubble("agent", "Support agent", c.reply || "(no reply)");
    status("No action tried. Nothing to check.");
    return;
  }
  addEl("toolcall", `<span class="who">${esc(c.callWho || "Agent calls a tool")}</span>${esc(callText(c))}`);
  await wait(700);
  resetGate();
  await runGate(c);
  addLedger(c);
  await wait(300);
  bubble(`outcome ${c.outcome}`, c.kind === "refund" && c.reply ? "What the customer sees" : "What happens", c.reply || WHAT_HAPPENS[c.outcome]);
}

// A /api/chat or /api/try answer, in the same shape as a story case.
function fromGuard(data, extra) {
  const a = data.attempt, g = data.guard;
  const c = { kind: "refund", customer: extra.customer, agent: data.agent_reply || "", reply: data.reply, attempt: a, label: extra.label };
  if (!a || !g) return c;
  c.action = { name: "issue_refund", args: a, plain: `Refund ${a.amount_gbp.toFixed(2)} GBP on order ${a.order_id}` };
  c.amount = { value: a.amount_gbp, currency: "GBP" };
  c.shop = "Quillmoor · refund slip";
  c.micr = `order ${a.order_id} · ${a.amount_gbp.toFixed(2)} GBP · reason read as: ${REASON[g.reason] || g.reason}`;
  c.claim = g.claim;
  c.evidence = g.facts || [g.evidence];
  c.policy_lines = g.rules.map((r) => r.text);
  c.sapling = g.sapling;
  c.outcome = FROM_GUARD[g.outcome];
  c.note = g.note;
  if (extra.expected) {
    c.truth = extra.expected === "allow" ? "right" : "wrong";
    c.truth_note = EXPECTED_NOTE[extra.expected];
  }
  const ms = g.sapling.model_ms ? ` in ${g.sapling.model_ms} ms` : "";
  const how = g.note ? "recorded check" : g.sapling.cached ? "cached Sapling answer" : `checked live by Sapling${ms}`;
  c.source = extra.source ? `${extra.source}, ${how}` : how;
  return c;
}

// ---------- normal mode ----------
let busy = false;
function lock(on) {
  busy = on;
  document.querySelectorAll(".picker button, #try-go, .compose button").forEach((b) => { b.disabled = on; });
}
const caseTag = (html) => addEl("case-tag", html);

async function runScripted(c) {
  if (busy) return;
  lock(true);
  $("chat").innerHTML = "";
  caseTag(`Scripted customer · <b>${esc(c.order_id)}</b> · ${esc(c.tactic === "none" ? "honest request" : c.tactic)}`);
  bubble("customer", "Customer", c.message);
  const d = dots();
  try {
    const data = await postJSON("/api/chat", { messages: [{ role: "user", content: c.message }], case_id: c.case_id });
    d.remove();
    const label = `${c.order_id} · ${c.tactic === "none" ? "honest request" : c.tactic}`;
    await afterAgent(fromGuard(data, { customer: c.message, expected: c.expected, label, source: `Quillmoor case ${c.case_id}, agent turn recorded` }));
  } catch (err) {
    d.remove();
    bubble("error", "Error", err.message);
  }
  lock(false);
}

let history = [];
async function sendLive(text) {
  if (busy) return;
  lock(true);
  history.push({ role: "user", content: text });
  bubble("customer", "Customer", text);
  const d = dots();
  try {
    const data = await postJSON("/api/chat", { messages: history, case_id: null });
    d.remove();
    history.push({ role: "assistant", content: data.reply || "" });
    const label = data.attempt ? `${data.attempt.order_id} · live agent` : "live agent";
    await afterAgent(fromGuard(data, { customer: text, label, source: "Live agent turn" }));
  } catch (err) {
    d.remove();
    history.pop();
    bubble("error", "Error", err.message);
  }
  lock(false);
}

async function tryRefund(e) {
  e.preventDefault();
  if (busy) return;
  const text = $("try-text").value.trim();
  if (!text) { $("try-text").focus(); status("Write the request first"); return; }
  const orderId = $("try-order").value;
  const amount = parseFloat($("try-amount").value);
  lock(true);
  $("chat").innerHTML = "";
  caseTag(`Your request · order <b>${esc(orderId)}</b> · straight to the guard`);
  bubble("customer", "Customer", text);
  const d = dots();
  status("Sending to Sapling");
  try {
    const data = await postJSON("/api/try", { order_id: orderId, amount_gbp: Number.isFinite(amount) ? amount : null, request: text });
    d.remove();
    const c = fromGuard(data, { customer: text, label: `${orderId} · your request`, source: "Your request" });
    c.callWho = "No agent here. The form makes the refund slip";
    c.call = c.micr;
    c.by = "Written by the form · not run yet";
    await afterAgent(c);
  } catch (err) {
    d.remove();
    bubble("error", "Could not check", err.message);
    status("Check failed");
  }
  lock(false);
}

async function normal() {
  const meta = await getJSON("/api/meta");
  const a = meta.score && meta.score.agent_vs_guard;
  if (a) {
    countTo("m-alone", a.bad_refunds_gbp, 1400);
    countTo("m-through", a.bad_allowed_gbp, 1400);
    $("m-alone-sub").textContent = `${a.bad_refunds_tried} bad refunds the agent tried`;
    $("m-through-sub").textContent = `${a.bad_blocked} blocked · ${a.bad_to_person} to a person · ${a.bad_allowed} got through`;
  }
  const byId = {};
  for (const c of meta.customers) {
    byId[c.case_id] = c;
    const b = document.createElement("button");
    b.type = "button";
    b.innerHTML = `<b>${esc(c.order_id)}</b>${esc(c.tactic === "none" ? "honest request" : c.tactic)}`;
    b.title = c.message;
    b.onclick = () => runScripted(c);
    $("picker").appendChild(b);
  }
  if (meta.live_agent) {
    $("compose").hidden = false;
    $("compose").onsubmit = (e) => { e.preventDefault(); const t = $("input").value.trim(); if (t) { $("input").value = ""; sendLive(t); } };
  } else {
    const select = $("try-order");
    for (const o of meta.orders) {
      const opt = document.createElement("option");
      opt.value = o.order_id;
      opt.textContent = `${o.order_id} · ${o.item} · £${o.paid_gbp}`;
      opt.dataset.paid = o.paid_gbp;
      select.appendChild(opt);
    }
    select.value = "5515";
    const syncAmount = () => { $("try-amount").value = select.selectedOptions[0].dataset.paid; };
    select.onchange = syncAmount;
    syncAmount();
    $("try").hidden = false;
    $("try").onsubmit = tryRefund;
  }
  fit();
  // ?case=c15 or ?case=c27,c15 plays scripted customers on load (used for headless screenshots).
  for (const id of (params.get("case") || "").split(",").filter(Boolean)) {
    const c = byId[id] || (await fetch(`/api/case/${encodeURIComponent(id)}`).then((r) => (r.ok ? r.json() : null)));
    if (c) await runScripted(c);
  }
}

// ---------- demo mode ----------
async function demo() {
  document.body.classList.add("demo");
  const set = params.get("set");
  const story = await getJSON("/api/story" + (set ? `?set=${encodeURIComponent(set)}` : ""));
  const cases = story.cases.filter((c) => c.action && c.sapling && c.outcome);
  currency = (cases.find((c) => c.amount) || { amount: { currency: "GBP" } }).amount.currency;
  $("m-alone").textContent = money(0, currency);
  $("m-through").textContent = money(0, currency);
  $("meters-eyebrow").textContent = "Running total in this demo";
  $("m-alone-sub").textContent = "0 wrong actions the agent tried";
  $("m-through-sub").textContent = "0 wrong actions got through";
  fit();
  // About 90 seconds whatever the number of cases: about 14 s each at pace 1.
  pace *= 14.5 / Math.min(15, Math.max(9, 84 / Math.max(cases.length, 1)));

  const intro = $("intro");
  intro.querySelector(".eyebrow").textContent = story.title || "Banyan Sapling policy challenge";
  intro.hidden = false;
  await wait(4600);
  intro.classList.add("out");
  await wait(500);
  intro.hidden = true;

  let alone = 0, through = 0, wrongTried = 0, wrongThrough = 0;
  for (const [i, c] of cases.entries()) {
    $("chat").innerHTML = "";
    resetGate();
    status("Waiting for the agent");
    caseTag(`Case ${i + 1} of ${cases.length} · <b>${esc(c.id)}</b>`);
    await typeBubble("customer", "Customer", c.customer || "");
    await wait(350);
    const d = dots();
    await wait(900);
    d.remove();
    await afterAgent({ ...c, label: c.action.plain || c.id }, { typing: true });
    if (c.truth === "wrong" && c.amount) {
      wrongTried += 1;
      alone += c.amount.value;
      if (c.outcome === "allowed") { wrongThrough += 1; through += c.amount.value; }
      countTo("m-alone", alone);
      countTo("m-through", through);
    } else if (c.truth === "wrong") {
      wrongTried += 1;
      if (c.outcome === "allowed") wrongThrough += 1;
    }
    $("m-alone-sub").textContent = `${wrongTried} wrong action${wrongTried === 1 ? "" : "s"} the agent tried`;
    $("m-through-sub").textContent = `${wrongThrough} wrong action${wrongThrough === 1 ? "" : "s"} got through`;
    await wait(2700);
  }
  showScore(story);
}

function showScore(story) {
  const s = story.score || {};
  const rows = (s.rows || []).map((r, i) => `<div class="sc-row" style="animation-delay:${i * 120}ms"><div class="v">${esc(r.value)}</div><div class="k">${esc(r.label)}</div></div>`).join("");
  $("sc-inner").innerHTML = `
    <p class="eyebrow">Measured result</p>
    <h2 class="sc-head">${esc(s.headline || "")}</h2>
    <div class="sc-rows" style="margin-top:28px">${rows}</div>
    <p class="sc-foot">${esc(story.source_note || "")} <b>Banyan Sapling checks one claim against the written policy and the record. Never the chat.</b></p>
    <p class="credit big">Policy check by <b>Banyan Sapling</b> from <b>Mubit</b></p>
    <div class="sc-links"><button type="button" id="replay">Play again</button><a href="/">Try it yourself</a><a href="https://github.com/faveh/agent-guard">github.com/faveh/agent-guard</a></div>`;
  $("scorecard").hidden = false;
  $("replay").onclick = () => location.reload();
}

addEventListener("resize", fit);
document.fonts && document.fonts.ready.then(fit);
fit();
resetGate();
(DEMO ? demo() : normal()).catch((err) => {
  status("Could not load");
  bubble("error", "Error", err.message);
});
