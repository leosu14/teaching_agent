/* Teaching Agent teacher dashboard (MVP item 7): a single page over the authenticated JSON API (same origin, no
dependencies), next to the student app and with the same rules: everything shown comes from the API, text is always set
with textContent, never parsed as HTML, and identity comes from the deployment (behind the auth gateway the browser's
session is enough; in local development a token is pasted and kept for this tab only, in sessionStorage).

The page only reads learners' data: the class dashboard and UNCERTAIN grades (GET /classes/{id}/dashboard and
/uncertain-grades) and the learner routes a teacher may read (profile, progress, goals, curricula, cycle history). The
only changes it sends are the teacher's own: creating a class and rotating its join code. */
"use strict";

const TOKEN_KEY = "ta.token";
const SOURCE_LABELS = { api: "Assessment", teaching_session: "Practice session", evaluation: "Check" };
const STATUS_LABELS = {
  RUNNING: "working", WAITING: "waiting for the learner", BLOCKED: "stopped, can resume",
  COMPLETED: "completed", FAILED: "failed", CANCELLED: "cancelled",
};
const ERROR_MESSAGES = {
  401: "Your sign-in has expired. Reload the page to sign in again.",
  403: "This needs a teacher account.",
  404: "That could not be found, or it is not one of your classes.",
  422: "That was not accepted. Check what you entered and try again.",
};

const app = { me: null, view: 0 };

// --- small DOM helpers ---------------------------------------------------------------------------------------------

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "on") for (const [ev, fn] of Object.entries(v)) el.addEventListener(ev, fn);
    else if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else el.setAttribute(k, v === true ? "" : String(v));
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function show(...nodes) {
  document.getElementById("main").replaceChildren(...nodes.flat().filter(Boolean));
  window.scrollTo(0, 0);
}

// A page paints through `painter()`: once the teacher has moved on (a newer route), a slow response paints nothing.
function painter() {
  const gen = app.view;
  return (...nodes) => { if (gen === app.view) show(...nodes); };
}

function busy(text) {
  return h("p", { class: "muted", "data-testid": "busy" }, h("span", { class: "spinner" }), text);
}

function errorBox(err) {
  const status = err && err.status;
  const text = ERROR_MESSAGES[status] || (status ? `Something went wrong (${status}).` : "The service could not be reached.");
  return h("div", { class: "error", role: "alert", "data-testid": "error" }, text);
}

function percent(x) {
  return `${Math.round((Number(x) || 0) * 100)}%`;
}

function meter(fraction, good) {
  const pct = Math.max(0, Math.min(100, Math.round((Number(fraction) || 0) * 100)));
  const bar = h("span");
  bar.style.width = `${pct}%`; // through the CSSOM: the page's CSP allows no inline style attribute
  return h("div", { class: good ? "meter good" : "meter", role: "img", "aria-label": `${pct}%` }, bar);
}

function plural(n, word) {
  return `${n} ${word}${n === 1 ? "" : "s"}`;
}

function fmtDate(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return isNaN(d) ? "" : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function store(key, value) {
  try {
    if (value === undefined) return window.sessionStorage.getItem(key);
    if (value === null) window.sessionStorage.removeItem(key);
    else window.sessionStorage.setItem(key, value);
  } catch (_) { /* storage can be unavailable: the page still works for this load */ }
  return null;
}

function statusBadge(status) {
  if (!status) return h("span", { class: "muted small" }, "no sessions yet");
  const cls = status === "COMPLETED" ? "badge good" : status === "FAILED" ? "badge bad"
    : status === "BLOCKED" ? "badge warn" : "badge";
  return h("span", { class: cls }, STATUS_LABELS[status] || status.toLowerCase());
}

// --- API -----------------------------------------------------------------------------------------------------------

class ApiError extends Error {
  constructor(status, body) {
    super(`HTTP ${status}`);
    this.status = status;
    this.body = body;
  }
}

async function api(method, path, body, { allow = [] } = {}) {
  const headers = { Accept: "application/json" };
  const token = store(TOKEN_KEY);
  if (token) headers.Authorization = `Bearer ${token}`;
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(path, {
      method, headers, credentials: "same-origin", cache: "no-store",
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  } catch (_) {
    throw new ApiError(0, null);
  }
  let data = null;
  const type = response.headers.get("content-type") || "";
  if (type.includes("application/json")) data = await response.json().catch(() => null);
  if (response.status === 401 && path !== "/me") {
    renderSignIn(true);
    throw new ApiError(401, data);
  }
  if (!response.ok && !allow.includes(response.status)) throw new ApiError(response.status, data);
  return { status: response.status, data };
}

const get = (path) => api("GET", path).then((r) => r.data);
const post = (path, body) => api("POST", path, body).then((r) => r.data);
const enc = encodeURIComponent;

// --- start: who is this ----------------------------------------------------------------------------------------------

async function boot() {
  document.getElementById("sign-out").addEventListener("click", () => {
    store(TOKEN_KEY, null);
    location.hash = "#/";
    location.reload();
  });
  window.addEventListener("hashchange", () => route());
  let me;
  try {
    me = await api("GET", "/me", undefined, { allow: [401] });
  } catch (err) {
    return show(h("h1", { text: "Teaching Agent" }), errorBox(err));
  }
  if (me.status === 401) return renderSignIn(false);
  app.me = me.data;
  if (!(app.me.roles || []).includes("teacher")) return renderNotTeacher();
  document.getElementById("nav").hidden = false;
  route();
}

function renderSignIn(expired) {
  document.getElementById("nav").hidden = true;
  const input = h("input", { type: "password", id: "token", autocomplete: "off", "data-testid": "token" });
  const form = h("form", {
    on: {
      submit: (e) => {
        e.preventDefault();
        if (!input.value.trim()) return;
        store(TOKEN_KEY, input.value.trim());
        location.reload();
      },
    },
  }, h("label", { for: "token", text: "Development access token" }), input,
  h("p", { class: "muted small" }, "Kept for this browser tab only."),
  h("button", { type: "submit", "data-testid": "sign-in" }, "Sign in"));
  show(
    h("h1", { text: "Sign in" }),
    expired ? h("div", { class: "error", role: "alert" }, ERROR_MESSAGES[401]) : null,
    h("div", { class: "card" },
      h("p", {}, "This dashboard signs you in through your school's sign-in page. If you reached this screen there, reload the page."),
      h("p", { class: "muted small" }, "Running the app locally? Paste a token from the development token file."),
      form));
}

function renderNotTeacher() {
  show(h("h1", { text: "Not a teacher account" }), h("div", { class: "card", "data-testid": "not-teacher" },
    h("p", {}, "This dashboard is for teachers. Your account does not have the teacher role."),
    (app.me.learner_ids || []).length ? h("p", {}, h("a", { href: "/app/" }, "Go to the student app")) : null));
}

// --- routing -------------------------------------------------------------------------------------------------------

function route() {
  if (!app.me) return;
  app.view += 1;
  const [page, classId, sub, learnerId] = (location.hash.replace(/^#\/?/, "") || "").split("/").map(decodeURIComponent);
  for (const a of document.querySelectorAll("nav a")) a.classList.toggle("active", !page);
  if (page === "class" && classId && sub === "learner" && learnerId) return renderLearner(classId, learnerId);
  if (page === "class" && classId) return renderClass(classId);
  return renderClasses();
}

// --- classes: the list, and a new class ----------------------------------------------------------------------------

async function renderClasses() {
  const show = painter();
  show(busy("Loading…"));
  let classes;
  try {
    classes = await get("/classes");
  } catch (err) {
    return show(h("h1", { text: "Your classes" }), errorBox(err));
  }
  const rows = classes.map((c) => h("tr", { "data-testid": "class-row" },
    h("td", {}, h("a", { href: `#/class/${enc(c.class_id)}` }, c.name)),
    h("td", { class: "small muted" }, plural(c.learners.length, "learner")),
    h("td", { class: "small muted" }, fmtDate(c.created_at))));
  show(h("h1", { text: "Your classes" }),
    h("div", { class: "card", "data-testid": "classes" }, rows.length
      ? h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, rows)))
      : h("p", { class: "muted" }, "No classes yet. Create one below and share its join code with your learners.")),
    newClassForm());
}

function newClassForm() {
  const name = h("input", { type: "text", id: "class-name", maxlength: 200, required: true, "data-testid": "class-name" });
  const feedback = h("div");
  const form = h("form", {
    on: {
      submit: async (e) => {
        e.preventDefault();
        const button = form.querySelector("button");
        if (!name.value.trim()) return;
        button.disabled = true;
        try {
          const made = await post("/classes", { name: name.value.trim() });
          app.joinCode = made; // shown once on the class page: the code is not stored in a readable form
          location.hash = `#/class/${enc(made.class_id)}`;
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
          button.disabled = false;
        }
      },
    },
  }, feedback, h("label", { for: "class-name", text: "Class name" }), name, h("p", {}),
  h("button", { type: "submit", "data-testid": "create-class" }, "Create class"));
  return h("div", { class: "card" }, h("h2", { text: "New class" }), form);
}

function joinCodeCard(made) {
  return h("div", { class: "card highlight", "data-testid": "join-code-card" },
    h("h3", { text: "Join code" }),
    h("p", { class: "code", "data-testid": "join-code" }, made.join_code),
    h("p", { class: "muted small" },
      "Give this code to your learners: they enter it in the student app to join, which lets you see their progress. " +
      "It is shown only now. If you lose it, make a new one (the old one stops working)."));
}

// --- one class: the roster at a glance and the grades to review ----------------------------------------------------

async function renderClass(classId, opts = {}) {
  const show = painter();
  show(busy("Loading the class…"));
  const settled = Boolean(opts.settled);
  let board, uncertain;
  try {
    [board, uncertain] = await Promise.all([
      get(`/classes/${enc(classId)}/dashboard`),
      get(`/classes/${enc(classId)}/uncertain-grades${settled ? "?include_resolved=true" : ""}`)]);
  } catch (err) {
    return show(h("p", {}, h("a", { href: "#/" }, "← Classes")), h("h1", { text: "Class" }), errorBox(err));
  }
  const code = app.joinCode && app.joinCode.class_id === classId ? app.joinCode : null;
  app.joinCode = null;
  const rotate = h("button", {
    type: "button", class: "secondary", "data-testid": "rotate-code",
    on: {
      click: async () => {
        if (!window.confirm("Make a new join code? The current code stops working; learners already in the class stay.")) return;
        rotate.disabled = true;
        try {
          app.joinCode = await post(`/classes/${enc(classId)}/join-code`);
          renderClass(classId, opts);
        } catch (err) {
          rotate.disabled = false;
          rotate.after(errorBox(err));
        }
      },
    },
  }, "New join code");
  const learners = board.learners;
  const summary = h("div", { class: "card", "data-testid": "class-summary" },
    h("div", { class: "row spread" },
      h("span", {}, plural(learners.length, "learner")),
      h("span", {}, board.uncertain_open
        ? h("a", { href: "#review", class: "badge warn", "data-testid": "review-count" }, `${board.uncertain_open} to review`)
        : h("span", { class: "badge good", "data-testid": "review-count" }, "nothing to review"))),
    h("p", { class: "muted small" }, `As of ${fmtDate(board.as_of)}`));
  const roster = learners.length ? h("div", { class: "card", "data-testid": "roster" }, h("div", { class: "table-wrap" },
    h("table", {},
      h("thead", {}, h("tr", {}, ["Learner", "Goal progress", "Mastery", "Needs work", "Due", "Now", "Last active", "Review"]
        .map((t) => h("th", { class: "small" }, t)))),
      h("tbody", {}, learners.map(learnerRow.bind(null, classId))))))
    : h("div", { class: "card", "data-testid": "roster" }, h("p", { class: "muted" },
      "No learners yet. They join with the class's join code in the student app."));
  const toggle = h("label", { class: "choice small" },
    h("input", { type: "checkbox", checked: settled, "data-testid": "show-settled",
      on: { change: (e) => renderClass(classId, { settled: e.target.checked }) } }),
    "Also show answers a later attempt settled");
  show(h("p", {}, h("a", { href: "#/" }, "← Classes")),
    h("div", { class: "row spread" }, h("h1", { text: board.name }), rotate),
    code ? joinCodeCard(code) : null,
    summary, roster,
    h("h2", { id: "review", text: "Answers to review" }),
    h("p", { class: "muted small" },
      "The grader could not decide these answers, so they did not count for or against the learner. Learners may answer again; " +
      "a later graded answer settles one."),
    toggle,
    uncertainList(uncertain, settled, true));
}

function learnerRow(classId, o) {
  const goal = o.goals.find((g) => g.status === "ACTIVE" && g.planned) || o.goals.find((g) => g.planned) || o.goals[0];
  const goalCell = !goal ? h("span", { class: "muted small" }, "no goal yet")
    : !goal.planned ? h("span", { class: "small" }, h("span", { text: goal.title }), h("br"), h("span", { class: "muted" }, "no path planned"))
      : h("div", { class: "small" }, h("span", { text: goal.title }), meter(goal.percent_complete / 100, goal.goal_complete),
        h("span", { class: "muted" }, `${goal.objectives_mastered} of ${goal.objectives_required} objectives`));
  return h("tr", { "data-testid": "learner-row", "data-learner": o.learner_id },
    h("td", {}, h("a", { href: `#/class/${enc(classId)}/learner/${enc(o.learner_id)}`, "data-testid": "learner-link" }, o.display_name)),
    h("td", { class: "goal-cell" }, goalCell),
    h("td", {}, percent(o.average_mastery)),
    h("td", {}, o.weak ? h("span", { class: "badge bad" }, String(o.weak)) : "0"),
    h("td", {}, o.due_for_review ? h("span", { class: "badge warn" }, String(o.due_for_review)) : "0"),
    h("td", {}, statusBadge(o.current_cycle_status || o.last_cycle_status)),
    h("td", { class: "small muted" }, fmtDate(o.last_active_at)),
    h("td", {}, o.uncertain_open ? h("span", { class: "badge warn", "data-testid": "learner-review" }, String(o.uncertain_open)) : "0"));
}

function uncertainList(items, settled, withLearner) {
  if (!items.length) {
    return h("div", { class: "card", "data-testid": "uncertain" }, h("p", { class: "muted" },
      settled ? "No answer of these learners was left undecided." : "Nothing to review. Every answer was graded."));
  }
  return h("div", { "data-testid": "uncertain" }, items.map((u) => h("div", { class: "card", "data-testid": "uncertain-item" },
    h("div", { class: "row spread" },
      h("strong", { text: withLearner ? u.display_name : u.concept_name }),
      u.resolved ? h("span", { class: "badge good" }, "settled by a later attempt") : h("span", { class: "badge warn" }, "to review")),
    h("p", { class: "muted small" }, [withLearner ? u.concept_name : null, SOURCE_LABELS[u.source] || u.source,
      `attempt ${u.attempt_number}`, fmtDate(u.submitted_at)].filter(Boolean).join(" · ")),
    h("p", { class: "question-text", text: u.prompt }),
    u.choices.length ? h("ul", { class: "plain small" }, u.choices.map((c) => h("li", { text: c }))) : null,
    h("div", { class: "answer" }, h("span", { class: "who" }, "Learner's answer"), h("span", { "data-testid": "learner-answer", text: u.learner_answer })),
    u.later_attempts && !u.resolved ? h("p", { class: "muted small" }, `${plural(u.later_attempts, "later attempt")}, none graded yet.`) : null)));
}

// --- one learner of the class --------------------------------------------------------------------------------------

async function renderLearner(classId, learnerId) {
  const show = painter();
  show(busy("Loading…"));
  const L = `/learners/${enc(learnerId)}`;
  const back = h("p", {}, h("a", { href: `#/class/${enc(classId)}` }, "← Class"));
  let profile, progress, goals, cycles, uncertain;
  try {
    [profile, progress, goals, cycles, uncertain] = await Promise.all([get(L), get(`${L}/progress`), get(`${L}/goals`),
      get(`${L}/learning-cycles`), get(`/classes/${enc(classId)}/uncertain-grades?include_resolved=true`)]);
  } catch (err) {
    return show(back, h("h1", { text: "Learner" }), errorBox(err));
  }
  const curricula = await Promise.all(goals.filter((g) => g.status !== "CANCELLED").map((g) =>
    api("GET", `/goals/${enc(g.goal_id)}/curriculum`, undefined, { allow: [404] })
      .then((r) => [g, r.status === 404 ? null : r.data]).catch(() => [g, null])));
  const mastered = new Set(progress.mastered), weak = new Set(progress.weak), due = new Set(progress.due_for_review);
  const concepts = Object.values(profile.concepts).sort((a, b) => a.mastery - b.mastery || a.name.localeCompare(b.name));
  const label = (id) => (mastered.has(id) ? h("span", { class: "badge good" }, "mastered")
    : weak.has(id) ? h("span", { class: "badge bad" }, "needs work") : h("span", { class: "badge" }, "developing"));
  const nodes = [back, h("h1", { text: profile.display_name || learnerId }),
    h("div", { class: "card", "data-testid": "learner-summary" },
      h("div", { class: "row spread" }, h("strong", {}, "Average mastery"), h("span", {}, percent(progress.average_mastery))),
      meter(progress.average_mastery),
      h("p", { class: "muted small" }, `${plural(progress.lessons_completed, "lesson")} completed · ${plural(progress.assessments_taken, "check")} taken · ` +
        `${progress.mastered.length} mastered · ${progress.weak.length} need work · ${progress.due_for_review.length} due for review`)),
    h("h2", { text: "Mastery by concept" }),
    h("div", { class: "card", "data-testid": "concepts" }, concepts.length ? h("div", { class: "table-wrap" }, h("table", {},
      h("tbody", {}, concepts.map((c) => h("tr", {},
        h("td", {}, h("div", { text: c.name }), meter(c.mastery, mastered.has(c.concept_id))),
        h("td", { class: "small" }, label(c.concept_id), due.has(c.concept_id) ? h("span", { class: "badge warn" }, "due") : null),
        h("td", { class: "small muted" }, plural(c.evidence_count, "answer")))))))
      : h("p", { class: "muted" }, "No evidence yet."))];
  for (const [g, cur] of curricula) {
    const card = h("div", { class: "card", "data-testid": "learner-goal" },
      h("div", { class: "row spread" }, h("h3", { text: g.title }), h("span", { class: g.status === "COMPLETED" ? "badge good" : "badge" }, g.status.toLowerCase())),
      h("p", { class: "muted small" }, [g.domain, g.target_level, g.target_date ? `by ${fmtDate(g.target_date)}` : null].filter(Boolean).join(" · ")));
    if (!cur) card.append(h("p", { class: "muted small" }, "No path planned yet."));
    else {
      const p = cur.progress;
      card.append(meter(p.percent_complete / 100, p.goal_complete), h("p", { class: "small muted" }, `${p.mastered} of ${p.required} objectives mastered`),
        h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, cur.objectives.map((o) => h("tr", {},
          h("td", {}, h("div", { text: o.name }), meter(o.current_mastery / (o.target_mastery || 1), o.status === "MASTERED")),
          h("td", { class: "small" }, o.status.replace(/_/g, " ").toLowerCase())))))));
    }
    nodes.push(card);
  }
  if (!curricula.length) nodes.push(h("div", { class: "card" }, h("p", { class: "muted" }, "No goal set yet.")));
  const recent = cycles.slice().sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).slice(0, 10);
  nodes.push(h("h2", { text: "Recent learning sessions" }), h("div", { class: "card", "data-testid": "learner-cycles" }, recent.length
    ? h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {}, recent.map((c) => h("tr", {},
      h("td", { class: "small muted", text: fmtDate(c.created_at) }),
      h("td", { text: c.action.objective_description || c.action.action }),
      h("td", {}, statusBadge(c.status)))))))
    : h("p", { class: "muted" }, "No learning sessions yet.")));
  nodes.push(h("h2", { text: "Answers to review" }), uncertainList(uncertain.filter((u) => u.learner_id === learnerId), true, false));
  show(nodes);
}

document.addEventListener("DOMContentLoaded", boot);
