/* Teaching Agent student app: a single page over the authenticated JSON API (same origin, no dependencies).

Everything shown comes from the API; text is always set with textContent, never parsed as HTML. Identity comes from
the deployment: behind the auth gateway the browser's session is enough, and the gateway signs each request; in local
development (TA_AUTH_MODE=static) the learner pastes a development token, kept for this tab only (sessionStorage). */
"use strict";

const TOKEN_KEY = "ta.token";
const LEARNER_KEY = "ta.learner";
const ACTIVE = new Set(["RUNNING", "WAITING", "BLOCKED"]);
const ACTION_LABELS = {
  LEARN: "Learn something new",
  REVIEW: "Review",
  PRACTICE: "Practise",
  EVALUATE: "Check what you know",
  WAIT: "Nothing due right now",
  COMPLETE: "Goal complete",
};
const CEFR = { framework: "cefr", levels: ["A1", "A2", "B1", "B2", "C1", "C2"] };
const STEP_LABELS = { LESSON: "Lesson", TEACHING_SESSION: "Practice with the teacher", EVALUATION: "Check" };
const RESULT_LABELS = {
  TAUGHT: "You finished the lesson and the practice.",
  EVALUATED: "Your check is graded.",
  GOAL_COMPLETED: "You reached your goal!",
  NOTHING_DUE: "Nothing was due.",
};
const ERROR_MESSAGES = {
  401: "Your sign-in has expired. Reload the page to sign in again.",
  404: "That could not be found.",
  409: "Something changed in the meantime. The page has been refreshed.",
  422: "That was not accepted. Check your answers and try again.",
  429: "The spending limit for now has been reached. Try again later or ask your teacher.",
  503: "The teacher is not available right now. Try again in a moment.",
};

const app = { me: null, learnerId: null, profile: null, urls: [], pending: {}, polling: null, view: 0 };

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
  stopPolling();
  for (const url of app.urls) URL.revokeObjectURL(url);
  app.urls = [];
  const main = document.getElementById("main");
  main.replaceChildren(...nodes.flat().filter(Boolean));
  window.scrollTo(0, 0);
}

// A page's renderer paints through `painter()`: once the learner has moved to another page (a newer route), a slow
// response of the old page paints nothing.
function painter() {
  const gen = app.view;
  return (...nodes) => { if (gen === app.view) show(...nodes); };
}

function go(hash) {
  if (location.hash === hash) route();
  else location.hash = hash; // hashchange routes
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

function newId(prefix) {
  const raw = (window.crypto && crypto.randomUUID) ? crypto.randomUUID()
    : `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`;
  return `${prefix}-${raw}`;
}

function store(kind, key, value) {
  try {
    if (value === undefined) return window[kind].getItem(key);
    if (value === null) window[kind].removeItem(key);
    else window[kind].setItem(key, value);
  } catch (_) { /* storage can be unavailable: the app still works for this page load */ }
  return null;
}

function stopPolling() {
  if (app.polling) clearTimeout(app.polling);
  app.polling = null;
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
  const token = store("sessionStorage", TOKEN_KEY);
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

const get = (path, opts) => api("GET", path, undefined, opts).then((r) => (opts ? r : r.data));
const post = (path, body, opts) => api("POST", path, body, opts).then((r) => (opts ? r : r.data));

async function blobUrl(path) {
  const headers = {};
  const token = store("sessionStorage", TOKEN_KEY);
  if (token) headers.Authorization = `Bearer ${token}`;
  const response = await fetch(path, { headers, credentials: "same-origin", cache: "no-store" });
  if (!response.ok) throw new ApiError(response.status, null);
  const url = URL.createObjectURL(await response.blob());
  app.urls.push(url);
  return url;
}

const L = () => `/learners/${encodeURIComponent(app.learnerId)}`;

// --- start: who is this, and for which learner ---------------------------------------------------------------------

async function boot() {
  document.getElementById("sign-out").addEventListener("click", () => {
    store("sessionStorage", TOKEN_KEY, null);
    store("sessionStorage", LEARNER_KEY, null);
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
  const ids = app.me.learner_ids;
  const remembered = store("sessionStorage", LEARNER_KEY);
  if (ids.length === 0) return renderNoLearner();
  if (ids.length > 1 && !ids.includes(remembered)) return renderLearnerPicker(ids);
  await useLearner(ids.includes(remembered) ? remembered : ids[0]);
}

async function useLearner(learnerId) {
  app.learnerId = learnerId;
  store("sessionStorage", LEARNER_KEY, learnerId);
  const found = await api("GET", L(), undefined, { allow: [404] }).catch((err) => ({ error: err }));
  if (found.error) return show(errorBox(found.error));
  document.getElementById("nav").hidden = false;
  if (found.status === 404) return renderOnboarding();
  app.profile = found.data;
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
        store("sessionStorage", TOKEN_KEY, input.value.trim());
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
      h("p", {}, "This app signs you in through your school's sign-in page. If you reached this screen there, reload the page."),
      h("p", { class: "muted small" }, "Running the app locally? Paste a token from the development token file."),
      form));
}

function renderNoLearner() {
  show(h("h1", { text: "No learner profile" }), h("div", { class: "card" },
    h("p", {}, "This account is not linked to a learner, so there is nothing to study here."),
    (app.me.roles || []).includes("teacher")
      ? h("p", { class: "muted" }, "You are signed in as a teacher. The teacher dashboard is a separate app.") : null));
}

function renderLearnerPicker(ids) {
  show(h("h1", { text: "Who is learning today?" }), h("div", { class: "card" },
    ids.map((id) => h("p", {}, h("button", { type: "button", class: "secondary", on: { click: () => useLearner(id) } }, id)))));
}

function renderOnboarding() {
  const name = h("input", { type: "text", id: "display-name", maxlength: 80, required: true, "data-testid": "display-name" });
  const minutes = h("select", { id: "minutes" }, [15, 30, 45, 60].map((m) =>
    h("option", { value: m, selected: m === 30 }, `${m} minutes`)));
  const feedback = h("div");
  const form = h("form", {
    on: {
      submit: async (e) => {
        e.preventDefault();
        const button = form.querySelector("button");
        button.disabled = true;
        try {
          app.profile = await api("PUT", L(), {
            display_name: name.value.trim(),
            preferences: { session_minutes: Number(minutes.value) },
          }).then((r) => r.data);
          go("#/goal/new");
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
          button.disabled = false;
        }
      },
    },
  }, feedback, h("label", { for: "display-name", text: "What should we call you?" }), name,
  h("label", { for: "minutes", text: "How long is a usual study session?" }), minutes,
  h("p", {}), h("button", { type: "submit", "data-testid": "save-profile" }, "Continue"));
  show(h("h1", { text: "Welcome" }),
    h("p", { class: "muted" }, "A few details first. Your level is found with a short check in your first lesson."),
    h("div", { class: "card" }, form));
}

// --- routing -------------------------------------------------------------------------------------------------------

function route() {
  if (!app.learnerId) return;
  app.view += 1;
  const parts = (location.hash.replace(/^#\/?/, "") || "").split("/").map(decodeURIComponent);
  const [page, id] = parts;
  for (const a of document.querySelectorAll("nav a")) {
    a.classList.toggle("active", a.dataset.nav === (page || "today"));
  }
  const pages = {
    "": renderToday, progress: renderProgress, history: renderHistory,
    goal: renderGoalForm, cycle: () => renderCycle(id), lesson: () => renderLesson(id),
  };
  (pages[page] || renderToday)();
}

// --- today: the next action, the active cycle, the goals -----------------------------------------------------------

async function renderToday() {
  const show = painter();
  show(busy("Loading…"));
  let goals, cycles, next;
  try {
    [goals, cycles, next] = await Promise.all([get(`${L()}/goals`), get(`${L()}/learning-cycles`), get(`${L()}/next-action`)]);
  } catch (err) {
    return show(h("h1", { text: "Today" }), errorBox(err));
  }
  const name = app.profile && app.profile.display_name;
  const heading = h("h1", { text: name ? `Hello, ${name}` : "Today" });
  const activeGoals = goals.filter((g) => g.status === "ACTIVE");
  if (goals.length === 0) {
    return show(heading, h("div", { class: "card highlight", "data-testid": "no-goal" },
      h("h2", { text: "Set your first goal" }),
      h("p", {}, "Tell us what you want to reach. We plan a path of lessons for it."),
      h("a", { class: "button", href: "#/goal/new" }, "Set a goal")));
  }
  const curricula = await Promise.all(activeGoals.map((g) =>
    get(`/goals/${encodeURIComponent(g.goal_id)}/curriculum`, { allow: [404] }).then((r) => (r.status === 404 ? null : r.data))
      .catch(() => null)));
  const active = cycles.find((c) => ACTIVE.has(c.status));
  const nodes = [heading];
  if (active) nodes.push(activeCard(active));
  else nodes.push(nextActionCard(next, activeGoals, curricula));
  nodes.push(h("div", { class: "row spread" }, h("h2", { text: "Your goals" }), h("a", { href: "#/goal/new" }, "Add a goal")));
  goals.forEach((g) => {
    const i = activeGoals.indexOf(g);
    nodes.push(goalCard(g, i >= 0 ? curricula[i] : null));
  });
  show(nodes);
}

function activeCard(cycle) {
  const what = cycle.action.objective_description || ACTION_LABELS[cycle.action.action];
  return h("div", { class: "card highlight", "data-testid": "active-cycle" },
    h("span", { class: "badge" }, cycle.status === "BLOCKED" ? "Paused by a problem" : "In progress"),
    h("h2", { text: ACTION_LABELS[cycle.action.action] || cycle.action.action }),
    h("p", { text: what }),
    h("a", { class: "button", href: `#/cycle/${encodeURIComponent(cycle.cycle_id)}`, "data-testid": "continue" }, "Continue"));
}

function nextActionCard(next, goals, curricula) {
  const card = h("div", { class: "card highlight", "data-testid": "next-action" });
  const unplanned = goals.filter((_, i) => !curricula[i]);
  if (next.action === "WAIT" && unplanned.length) {
    const g = unplanned[0];
    const feedback = h("div");
    const button = h("button", {
      type: "button", "data-testid": "plan",
      on: {
        click: async () => {
          button.disabled = true;
          feedback.replaceChildren(busy("Planning your path…"));
          try {
            await planCurriculum(g.goal_id);
            renderToday();
          } catch (err) {
            feedback.replaceChildren(errorBox(err));
            button.disabled = false;
          }
        },
      },
    }, "Plan my path");
    card.append(h("h2", { text: `Plan the path to "${g.title}"` }),
      h("p", {}, "We turn your goal into a sequence of lessons, prerequisites first."), feedback, button);
    return card;
  }
  card.append(h("span", { class: "badge" }, "Next"), h("h2", { text: ACTION_LABELS[next.action] || next.action }));
  if (next.objective_description) card.append(h("p", { "data-testid": "objective", text: next.objective_description }));
  if (next.action === "WAIT") {
    card.append(h("p", { class: "muted" }, next.next_review_at
      ? `Your next review is due ${fmtDate(next.next_review_at)}.` : "Come back later, or add another goal."));
  } else if (next.action === "COMPLETE") {
    card.append(h("p", {}, "You have reached every objective of this goal. Well done!"));
  } else {
    const feedback = h("div");
    const button = h("button", { type: "button", "data-testid": "start", on: { click: () => startCycle(button, feedback) } }, "Start");
    card.append(feedback, button);
  }
  card.append(h("details", {}, h("summary", {}, "Why this?"), h("p", { class: "muted small", text: next.reason })));
  return card;
}

async function startCycle(button, feedback) {
  button.disabled = true;
  app.pending.start = app.pending.start || newId("web");
  feedback.replaceChildren(busy("Preparing your lesson. This can take a minute…"));
  try {
    const cycle = await post(`${L()}/learning-cycles`, { idempotency_key: app.pending.start });
    app.pending.start = null;
    go(`#/cycle/${encodeURIComponent(cycle.cycle_id)}`);
  } catch (err) {
    if (err.status === 409) app.pending.start = null;
    feedback.replaceChildren(errorBox(err));
    button.disabled = false;
  }
}

function goalCard(goal, curriculum) {
  const card = h("div", { class: "card", "data-testid": "goal" },
    h("div", { class: "row spread" }, h("h3", { text: goal.title }),
      h("span", { class: goal.status === "COMPLETED" ? "badge good" : "badge" }, goal.status.toLowerCase())),
    h("p", { class: "muted small" }, [goal.domain, goal.target_level].filter(Boolean).join(" · ")));
  if (curriculum) {
    const p = curriculum.progress;
    card.append(meter(p.percent_complete / 100, p.goal_complete),
      h("p", { class: "small muted" }, `${p.mastered} of ${p.required} objectives mastered`));
  } else if (goal.status === "ACTIVE") {
    card.append(h("p", { class: "small muted" }, "No path planned yet."));
  }
  return card;
}

async function planCurriculum(goalId) {
  const r = await api("POST", `/goals/${encodeURIComponent(goalId)}/curriculum`, undefined, { allow: [422] });
  if (r.status === 422) throw new ApiError(422, r.data);
  return r.data;
}

// --- goal form -----------------------------------------------------------------------------------------------------

async function renderGoalForm() {
  const show = painter();
  show(busy("Loading…"));
  let courses;
  try {
    courses = await get("/courses");
  } catch (err) {
    return show(h("h1", { text: "New goal" }), errorBox(err));
  }
  const subject = courses.length
    ? h("select", { id: "subject", "data-testid": "subject" }, courses.map((c) => h("option", { value: c.course_id }, c.title)))
    : h("input", { type: "text", id: "subject", value: "spanish", required: true, "data-testid": "subject" });
  const level = h("select", { id: "level", "data-testid": "level" });
  const title = h("input", { type: "text", id: "title", maxlength: 200, "data-testid": "goal-title" });
  const date = h("input", { type: "date", id: "date" });
  // Without courses (the fixture knowledge base) a goal is a language goal on the CEFR scale.
  const course = () => courses.find((c) => c.course_id === subject.value) || CEFR;
  const fillLevels = () => {
    const c = course();
    level.replaceChildren(...c.levels.map((lv, i, all) => h("option", { value: lv, selected: i === all.length - 1 }, lv)));
    suggest();
  };
  const levelValue = () => level.value;
  const domain = () => (courses.length ? course().subject : subject.value.trim().toLowerCase());
  let typed = false;
  const suggest = () => {
    if (typed) return;
    const s = domain();
    title.value = levelValue() && s ? `Reach ${levelValue()} ${s.charAt(0).toUpperCase()}${s.slice(1)}` : "";
  };
  title.addEventListener("input", () => { typed = true; });
  subject.addEventListener("change", fillLevels);
  subject.addEventListener("input", suggest);
  level.addEventListener("change", suggest);
  fillLevels();
  const feedback = h("div");
  app.pending.goal = app.pending.goal || newId("goal");
  const form = h("form", {
    on: {
      submit: async (e) => {
        e.preventDefault();
        const button = form.querySelector("button");
        if (!levelValue() || !domain() || !title.value.trim()) {
          feedback.replaceChildren(h("div", { class: "error", role: "alert" }, "Fill in the subject, the level and a title."));
          return;
        }
        button.disabled = true;
        feedback.replaceChildren(busy("Saving your goal and planning your path…"));
        const body = {
          title: title.value.trim(), domain: domain(), target_level: levelValue(), idempotency_key: app.pending.goal,
          metadata: { framework_id: course().framework },
        };
        if (date.value) body.target_date = new Date(`${date.value}T23:59:00`).toISOString();
        try {
          const goal = await post(`${L()}/goals`, body);
          app.pending.goal = null;
          // A failed plan leaves the goal saved: today's page offers "Plan my path" again.
          await planCurriculum(goal.goal_id).catch(() => null);
          go("#/");
        } catch (err) {
          if (err.status) app.pending.goal = newId("goal"); // refused: a corrected goal is a new request
          feedback.replaceChildren(errorBox(err));
          button.disabled = false;
        }
      },
    },
  }, feedback,
  h("label", { for: "subject", text: courses.length ? "Course" : "Subject" }), subject,
  h("label", { for: "level", text: "Level you want to reach" }), level,
  h("label", { for: "title", text: "Goal" }), title,
  h("label", { for: "date", text: "Target date (optional)" }), date,
  h("p", {}), h("button", { type: "submit", "data-testid": "save-goal" }, "Save goal"));
  show(h("h1", { text: "New goal" }), h("div", { class: "card" }, form));
}

// --- a learning cycle: lesson, diagnostic, session, evaluation -----------------------------------------------------

async function renderCycle(cycleId, view) {
  if (!location.hash.startsWith(`#/cycle/${encodeURIComponent(cycleId)}`)) return; // the learner moved on
  const show = painter();
  if (!view) show(busy("Loading…"));
  let cycle = view;
  try {
    cycle = cycle || await get(`/learning-cycles/${encodeURIComponent(cycleId)}`);
  } catch (err) {
    return show(h("h1", { text: "Learning session" }), errorBox(err));
  }
  const header = [
    h("p", { class: "muted small" }, ACTION_LABELS[cycle.action.action] || cycle.action.action),
    h("h1", { text: cycle.action.objective_description || "Learning session" }),
    stepper(cycle),
  ];
  const lesson = cycle.steps.find((s) => s.kind === "LESSON" && s.status === "COMPLETED" && s.child_id);
  const lessonLink = lesson
    ? h("a", { class: "button secondary", href: `#/lesson/${encodeURIComponent(lesson.child_id)}`, "data-testid": "read-lesson" }, "Read the lesson")
    : null;
  const body = await cycleBody(cycle, lessonLink);
  const footer = ACTIVE.has(cycle.status) ? h("p", { class: "small" }, h("button", {
    type: "button", class: "link", "data-testid": "cancel",
    on: {
      click: async () => {
        if (!window.confirm("Stop this learning session? Your answers so far are kept.")) return;
        try { renderCycle(cycleId, await post(`/learning-cycles/${encodeURIComponent(cycleId)}/cancel`)); } catch (err) { show(header, errorBox(err)); }
      },
    },
  }, "Stop this session")) : null;
  show(header, body, footer);
  if (cycle.status === "RUNNING") {
    app.polling = setTimeout(() => (cycle.busy_until ? renderCycle(cycleId)
      : post(`/learning-cycles/${encodeURIComponent(cycleId)}/resume`).then((v) => renderCycle(cycleId, v))
        .catch((err) => show(header, errorBox(err)))), cycle.busy_until ? 2000 : 300);
  }
}

function stepper(cycle) {
  const current = cycle.steps.find((s) => s.status !== "COMPLETED");
  return h("ol", { class: "steps", "aria-label": "Steps" }, cycle.steps.map((s) =>
    h("li", { class: s.status === "COMPLETED" ? "done" : s === current ? "current" : "" }, STEP_LABELS[s.kind] || s.kind)));
}

async function cycleBody(cycle, lessonLink) {
  const id = cycle.cycle_id;
  if (cycle.status === "RUNNING") return h("div", { class: "card" }, busy("Working on it…"));
  if (cycle.status === "BLOCKED") {
    const feedback = h("div");
    const button = h("button", {
      type: "button", "data-testid": "retry",
      on: {
        click: async () => {
          button.disabled = true;
          feedback.replaceChildren(busy("Trying again…"));
          try { renderCycle(id, await post(`/learning-cycles/${encodeURIComponent(id)}/resume`)); } catch (err) { feedback.replaceChildren(errorBox(err)); button.disabled = false; }
        },
      },
    }, "Try again");
    return h("div", { class: "card" }, h("p", { class: "notice", text: (cycle.failure && cycle.failure.message) || "Something went wrong." }), feedback, button);
  }
  if (cycle.status === "FAILED") {
    return h("div", { class: "card" }, h("div", { class: "error", text: (cycle.failure && cycle.failure.message) || "This session could not be completed." }),
      h("a", { class: "button", href: "#/" }, "Back to today"));
  }
  if (cycle.status === "CANCELLED") {
    return h("div", { class: "card" }, h("p", {}, "This session was stopped."), h("a", { class: "button", href: "#/" }, "Back to today"));
  }
  if (cycle.status === "COMPLETED") return outcomeCard(cycle, lessonLink);
  const prompt = cycle.prompt;
  if (!prompt) return h("div", { class: "card" }, busy("Waiting…"));
  if (prompt.kind === "SESSION_QUESTION") return sessionView(cycle, lessonLink);
  return sheetForm(cycle, prompt, lessonLink);
}

function respondKey(cycle) {
  const ref = cycle.waiting ? cycle.waiting.ref : cycle.cycle_id;
  app.pending[ref] = app.pending[ref] || newId("resp");
  return ref;
}

async function respond(cycle, payload, feedback, controls) {
  const ref = respondKey(cycle);
  controls.forEach((c) => { c.disabled = true; });
  feedback.replaceChildren(busy("Sending…"));
  try {
    const view = await post(`/learning-cycles/${encodeURIComponent(cycle.cycle_id)}/responses`,
      { client_response_id: app.pending[ref], ...payload });
    delete app.pending[ref];
    renderCycle(cycle.cycle_id, view);
  } catch (err) {
    if (err.status === 409) { delete app.pending[ref]; return renderCycle(cycle.cycle_id); }
    if (err.status === 422) delete app.pending[ref];
    feedback.replaceChildren(errorBox(err));
    controls.forEach((c) => { c.disabled = false; });
  }
}

function sheetForm(cycle, prompt, lessonLink) {
  const diagnostic = prompt.kind === "DIAGNOSTIC_QUESTIONS";
  const fields = [];
  const questions = prompt.questions.map((q, i) => {
    const name = `q${i}`;
    let input;
    if (Array.isArray(q.choices) && q.choices.length) {
      input = h("div", { role: "radiogroup" }, q.choices.map((choice, j) =>
        h("label", { class: "choice" }, h("input", { type: "radio", name, value: choice, id: `${name}-${j}` }), choice)));
      fields.push({ q, read: () => (input.querySelector("input:checked") || {}).value || "" });
    } else {
      input = h("input", { type: "text", id: name, autocomplete: "off", "data-testid": "sheet-answer" });
      fields.push({ q, read: () => input.value.trim() });
    }
    return h("div", { class: "question" }, h("p", {}, `${i + 1}. `, q.prompt), input);
  });
  const feedback = h("div");
  const submit = h("button", { type: "submit", "data-testid": "submit-sheet" }, "Send answers");
  const form = h("form", {
    on: {
      submit: (e) => {
        e.preventDefault();
        const answers = fields.map(({ q, read }) => ({ question_id: q.question_id, answer: read() }));
        if (answers.some((a) => !a.answer)) {
          feedback.replaceChildren(h("div", { class: "error", role: "alert" }, "Answer every question. A guess is fine: it helps us find your level."));
          return;
        }
        respond(cycle, { answers }, feedback, [submit]);
      },
    },
  }, questions, feedback, submit);
  return h("div", { class: "card", "data-testid": diagnostic ? "diagnostic" : "evaluation" },
    h("h2", { text: diagnostic ? "Quick check: where are you starting from?" : "Show what you learned" }),
    h("p", { class: "muted" }, diagnostic
      ? "Answer what you can. Your answers decide what the lesson focuses on."
      : "Your answers are graded and update your progress."),
    !diagnostic && lessonLink ? h("p", { class: "muted small" }, "The lesson stays available, but answers you are asked for are hidden while you answer.") : null,
    form);
}

async function sessionView(cycle, lessonLink) {
  const prompt = cycle.prompt;
  const step = cycle.steps.find((s) => s.kind === "TEACHING_SESSION");
  let turns = prompt.teacher_turns;
  if (step && step.child_id) {
    try { turns = (await get(`/teaching-sessions/${encodeURIComponent(step.child_id)}`)).turns; } catch (_) { /* the prompt's turns are enough */ }
  }
  const chat = h("div", { class: "chat", "data-testid": "chat" }, turns.map((t) =>
    h("div", { class: t.speaker === "LEARNER" ? "bubble learner" : "bubble teacher" },
      h("span", { class: "who" }, t.speaker === "LEARNER" ? "You" : "Teacher"), t.content)));
  const q = prompt.question;
  const feedback = h("div");
  const text = h("input", { type: "text", id: "reply", autocomplete: "off", maxlength: 2000, "data-testid": "reply" });
  const send = h("button", { type: "submit", "data-testid": "send" }, "Send answer");
  const ask = h("button", { type: "button", class: "secondary", "data-testid": "ask" }, "Ask the teacher");
  const stop = h("button", { type: "button", class: "link", "data-testid": "stop" }, "I want to stop here");
  const choices = q && q.choices && q.choices.length ? h("div", { class: "row" }, q.choices.map((c) =>
    h("button", { type: "button", class: "secondary", "data-testid": "choice", on: { click: () => respond(cycle, { answer: c, kind: "answer" }, feedback, controls) } }, c))) : null;
  const controls = [send, ask, stop, ...(choices ? [...choices.children] : [])];
  ask.addEventListener("click", () => {
    if (!text.value.trim()) { text.focus(); text.placeholder = "Type your question for the teacher"; return; }
    respond(cycle, { answer: text.value.trim(), kind: "question" }, feedback, controls);
  });
  stop.addEventListener("click", () => {
    if (window.confirm("End the practice now? It counts what you answered so far.")) respond(cycle, { answer: "stop", kind: "stop" }, feedback, controls);
  });
  const form = h("form", {
    on: {
      submit: (e) => {
        e.preventDefault();
        if (!text.value.trim()) return;
        respond(cycle, { answer: text.value.trim(), kind: "answer" }, feedback, controls);
      },
    },
  }, choices, h("label", { for: "reply", text: choices ? "Or type your answer, or a question" : "Your answer, or a question for the teacher" }), text,
  h("p", {}), h("div", { class: "row" }, send, ask, stop));
  return h("div", {},
    h("div", { class: "card" }, chat, q ? h("p", { class: "muted small", "data-testid": "question" },
      `Question at difficulty ${q.difficulty}${q.hint_level ? `, hint ${q.hint_level}` : ""}`) : null, feedback, form),
    lessonLink ? h("p", {}, lessonLink) : null);
}

function outcomeCard(cycle, lessonLink) {
  const outcome = cycle.outcome || {};
  const names = (app.profile && app.profile.concepts) || {};
  const changes = (outcome.mastery_changes || []).map((c) => h("li", {},
    `${(names[c.concept_id] && names[c.concept_id].name) || c.concept_id}: ${percent(c.before)} → ${percent(c.after)}`));
  const next = outcome.next_action;
  return h("div", { class: "card highlight", "data-testid": "outcome" },
    h("span", { class: "badge good" }, "Done"),
    h("h2", { text: RESULT_LABELS[outcome.result] || "Finished" }),
    changes.length ? [h("h3", { text: "Your mastery" }), h("ul", { class: "plain" }, changes)] : null,
    next ? h("p", { class: "muted" }, "Next up: ", next.objective_description || ACTION_LABELS[next.action]) : null,
    h("div", { class: "row" }, h("a", { class: "button", href: "#/", "data-testid": "home" }, "Back to today"), lessonLink));
}

// --- the lesson (learner-safe reading) -----------------------------------------------------------------------------

async function renderLesson(lessonId) {
  const show = painter();
  show(busy("Loading the lesson…"));
  const r = await api("GET", `/lessons/${encodeURIComponent(lessonId)}/reading`, undefined, { allow: [409] }).catch((err) => ({ error: err }));
  if (r.error) return show(h("h1", { text: "Lesson" }), errorBox(r.error));
  if (r.status === 409) return show(h("h1", { text: "Lesson" }), h("p", { class: "notice" }, "This lesson is still being prepared."));
  const d = r.data;
  const article = h("article", { class: "lesson", "data-testid": "lesson" },
    h("p", { class: "muted small" }, d.level ? `Level ${d.level}` : ""),
    h("h1", { text: d.title || "Lesson" }),
    d.withheld ? h("p", { class: "notice small", "data-testid": "withheld" }, d.question_open
      ? "Some lines are hidden while you answer a question. Come back after answering to read them."
      : "A few lines are hidden because they give away answers you will be asked for.") : null,
    d.objectives.length ? [h("h2", { text: "You will learn to" }), h("ul", { class: "plain" }, d.objectives.map((o) => h("li", { text: o })))] : null,
    d.introduction.length ? h("p", { text: d.introduction.join(" ") }) : null);
  for (const s of d.sections) {
    const section = h("section", { "data-testid": "section" }, s.heading ? h("h2", { text: s.heading }) : null,
      s.paragraphs.length ? h("p", { text: s.paragraphs.join(" ") }) : null,
      s.examples.length ? [h("h3", { text: "Examples" }), h("ul", { class: "plain" }, s.examples.map((x) => h("li", { text: x })))] : null,
      s.analogy ? h("p", { class: "analogy", text: s.analogy }) : null);
    for (const img of s.images) {
      const el = h("img", { alt: img.description || "", "data-testid": "image" });
      section.append(el);
      blobUrl(`/artifacts/${encodeURIComponent(img.artifact_id)}/content`).then((url) => { el.src = url; }).catch(() => el.remove());
    }
    article.append(section);
  }
  if (d.summary.length) article.append(h("h2", { text: "Summary" }), h("p", { text: d.summary.join(" ") }));
  if (d.references.length) {
    article.append(h("h2", { text: "Sources" }), h("ul", { class: "plain small" }, d.references.map((ref) => h("li", {},
      ref.url ? h("a", { href: ref.url, target: "_blank", rel: "noopener noreferrer" }, ref.title) : ref.title,
      ref.publisher ? ` · ${ref.publisher}` : ""))));
  }
  show(h("p", {}, h("button", { type: "button", class: "link small", on: { click: () => history.back() } }, "← Back")),
    h("div", { class: "card" }, article));
}

// --- progress and history ------------------------------------------------------------------------------------------

async function renderProgress() {
  const show = painter();
  show(busy("Loading…"));
  let progress, profile, goals;
  try {
    [progress, profile, goals] = await Promise.all([get(`${L()}/progress`), get(L()), get(`${L()}/goals`)]);
  } catch (err) {
    return show(h("h1", { text: "Progress" }), errorBox(err));
  }
  app.profile = profile;
  const name = (id) => (profile.concepts[id] && profile.concepts[id].name) || id;
  const list = (title, ids, testid) => h("div", { class: "card", "data-testid": testid }, h("h3", { text: title }),
    ids.length ? h("ul", { class: "plain" }, ids.map((id) => h("li", { text: name(id) }))) : h("p", { class: "muted small" }, "None yet."));
  const curricula = await Promise.all(goals.filter((g) => g.status !== "CANCELLED").map((g) =>
    get(`/goals/${encodeURIComponent(g.goal_id)}/curriculum`, { allow: [404] }).then((r) => [g, r.status === 404 ? null : r.data]).catch(() => [g, null])));
  const nodes = [h("h1", { text: "Progress" }),
    h("div", { class: "card", "data-testid": "summary" },
      h("div", { class: "row spread" }, h("strong", {}, "Average mastery"), h("span", {}, percent(progress.average_mastery))),
      meter(progress.average_mastery),
      h("p", { class: "muted small" }, `${plural(progress.lessons_completed, "lesson")} completed · ${plural(progress.assessments_taken, "check")} taken`)),
    list("Mastered", progress.mastered, "mastered"), list("Needs work", progress.weak, "weak"),
    list("Due for review", progress.due_for_review, "due")];
  for (const [g, cur] of curricula) {
    if (!cur) continue;
    const statusClass = (s) => (s === "MASTERED" ? "badge good" : s === "IN_PROGRESS" ? "badge warn" : s === "BLOCKED" ? "badge bad" : "badge");
    nodes.push(h("h2", { text: g.title }), h("div", { class: "card", "data-testid": "curriculum" }, h("table", {},
      h("tbody", {}, cur.objectives.map((o) => h("tr", {},
        h("td", {}, h("div", { text: o.name }), meter(o.current_mastery / (o.target_mastery || 1), o.status === "MASTERED")),
        h("td", {}, h("span", { class: statusClass(o.status) }, o.status.replace(/_/g, " ").toLowerCase()))))))));
  }
  show(nodes);
}

async function renderHistory() {
  const show = painter();
  show(busy("Loading…"));
  let cycles;
  try {
    cycles = await get(`${L()}/learning-cycles`);
  } catch (err) {
    return show(h("h1", { text: "History" }), errorBox(err));
  }
  const rows = cycles.slice().sort((a, b) => (a.created_at < b.created_at ? 1 : -1)).map((c) => h("tr", {},
    h("td", { class: "small muted", text: fmtDate(c.created_at) }),
    h("td", {}, h("a", { href: `#/cycle/${encodeURIComponent(c.cycle_id)}` }, c.action.objective_description || ACTION_LABELS[c.action.action])),
    h("td", {}, h("span", { class: c.status === "COMPLETED" ? "badge good" : ACTIVE.has(c.status) ? "badge" : "badge bad" }, c.status.toLowerCase()))));
  show(h("h1", { text: "History" }), h("div", { class: "card", "data-testid": "history" }, rows.length
    ? h("table", {}, h("tbody", {}, rows)) : h("p", { class: "muted" }, "No learning sessions yet.")));
}

document.addEventListener("DOMContentLoaded", boot);
