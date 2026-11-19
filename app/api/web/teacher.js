/* Teaching Agent teacher dashboard (MVP item 7): a single page over the authenticated JSON API (same origin, no
dependencies), next to the student app and with the same rules: everything shown comes from the API, text is always set
with textContent, never parsed as HTML, and identity comes from the deployment (behind the auth gateway the browser's
session is enough; in local development a token is pasted and kept for this tab only, in sessionStorage).

The page reads learners' data: the class dashboard and UNCERTAIN grades (GET /classes/{id}/dashboard and
/uncertain-grades) and the learner routes a teacher may read (profile, progress, goals, curricula, cycle history). The
changes it sends are the teacher's own (creating a class, rotating its join code, the lesson-approval setting) and the
teacher reviews of Important item 2, all on class routes: approving or rejecting a lesson after previewing it, and
deciding an answer the grader could not (an override of an UNCERTAIN grade). Class assignments (Important item 3) are
the teacher's too: a goal or course topics with a deadline for the whole class, its deadline changed or the assignment
cancelled, and each learner's progress on it read. */
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
  const raw = err && err.body && (status === 409 || status === 422) ? err.body.detail : null;
  // the API's reason, when it gives one: a sentence, or the first validation message
  const detail = typeof raw === "string" ? raw
    : Array.isArray(raw) && raw[0] && typeof raw[0].msg === "string" ? raw[0].msg.replace(/^Value error, /, "") : null;
  const text = detail || ERROR_MESSAGES[status] || (status ? `Something went wrong (${status}).` : "The service could not be reached.");
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
const patch = (path, body) => api("PATCH", path, body).then((r) => r.data);
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
  if (page === "class" && classId && sub === "assignment" && learnerId) return renderAssignment(classId, learnerId);
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
  let board, uncertain, info, pending, assignments, courses;
  try {
    [board, uncertain, info, pending, assignments, courses] = await Promise.all([
      get(`/classes/${enc(classId)}/dashboard`),
      get(`/classes/${enc(classId)}/uncertain-grades${settled ? "?include_resolved=true" : ""}`),
      get(`/classes/${enc(classId)}`),
      get(`/classes/${enc(classId)}/lessons?status=PENDING`),
      get(`/classes/${enc(classId)}/assignments`),
      get("/courses").catch(() => [])]);
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
    h("h2", { id: "assignments", text: "Assignments" }),
    assignmentList(classId, assignments),
    newAssignmentForm(classId, courses, () => renderClass(classId, opts)),
    approvalCard(classId, info, opts),
    info.lesson_approval || pending.length ? h("h2", { id: "lessons", text: "Lessons to approve" }) : null,
    info.lesson_approval || pending.length ? pendingLessons(classId, pending, opts) : null,
    h("h2", { id: "review", text: "Answers to review" }),
    h("p", { class: "muted small" },
      "The grader could not decide these answers, so they did not count for or against the learner. Learners may answer again; " +
      "a later graded answer settles one."),
    toggle,
    uncertainList(uncertain, settled, true, classId, () => renderClass(classId, opts)));
}

// --- class assignments (Important item 3) ------------------------------------------------------------------------

const ASSIGNMENT_STATUS = {
  COMPLETED: ["badge good", "completed"], IN_PROGRESS: ["badge", "in progress"], NOT_STARTED: ["badge", "not started"],
  OVERDUE: ["badge bad", "overdue"], NOT_ASSIGNED: ["badge warn", "not assigned"], CANCELLED: ["badge", "cancelled"],
};

function assignmentBadge(status) {
  const [cls, label] = ASSIGNMENT_STATUS[status] || ["badge", String(status).toLowerCase()];
  return h("span", { class: cls, "data-testid": "assignment-status" }, label);
}

// A deadline the teacher picks in their own time zone; the API is sent the instant, in UTC.
function localInput(iso) {
  const d = iso ? new Date(iso) : new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function defaultDue() {
  const d = new Date();
  d.setDate(d.getDate() + 7);
  d.setHours(23, 59, 0, 0);
  return localInput(d.toISOString());
}

function assignmentTotals(t) {
  const parts = [`${t.completed} of ${plural(t.learners, "learner")} completed`];
  if (t.overdue) parts.push(`${t.overdue} overdue`);
  if (t.not_assigned) parts.push(`${t.not_assigned} not assigned`);
  return parts.join(" · ");
}

function assignmentList(classId, items) {
  if (!items.length) {
    return h("div", { class: "card", "data-testid": "assignments" }, h("p", { class: "muted" },
      "No assignments yet. Assign course topics or a level, with a due date, to the whole class below."));
  }
  return h("div", { class: "card", "data-testid": "assignments" }, h("div", { class: "table-wrap" }, h("table", {},
    h("tbody", {}, items.map((a) => h("tr", { "data-testid": "assignment-row", "data-assignment": a.assignment_id },
      h("td", {}, h("a", { href: `#/class/${enc(classId)}/assignment/${enc(a.assignment_id)}`, "data-testid": "assignment-link" }, a.title),
        h("div", { class: "muted small" }, a.topics.length ? a.topics.join(", ") : `${a.domain} · level ${a.target_level}`)),
      h("td", { class: "small" }, h("span", { "data-testid": "assignment-due" }, `Due ${fmtDate(a.due_at)}`)),
      h("td", {}, a.status === "CANCELLED" ? assignmentBadge("CANCELLED")
        : a.overdue ? h("span", { class: "badge bad" }, "past due") : h("span", { class: "badge" }, "open")),
      h("td", { class: "small muted", "data-testid": "assignment-totals" }, assignmentTotals(a.totals))))))));
}

function newAssignmentForm(classId, courses, done) {
  const feedback = h("div");
  const title = h("input", { type: "text", id: "assignment-title", maxlength: 200, required: true, "data-testid": "assignment-title" });
  const description = h("textarea", { id: "assignment-description", maxlength: 2000, rows: 2, "data-testid": "assignment-description" });
  const due = h("input", { type: "datetime-local", id: "assignment-due", required: true, value: defaultDue(), "data-testid": "assignment-due-input" });
  const targets = h("div", { "data-testid": "assignment-targets" });
  const kind = h("select", { id: "assignment-kind", "data-testid": "assignment-kind" },
    h("option", { value: "topics" }, "Course topics"), h("option", { value: "level" }, "A level"));
  let subject;
  if (courses.length) {
    subject = h("select", { id: "assignment-subject", "data-testid": "assignment-subject" },
      courses.map((c) => h("option", { value: c.subject }, `${c.title} (${c.subject})`)));
  } else {
    subject = h("input", { type: "text", id: "assignment-subject", maxlength: 128, required: true, "data-testid": "assignment-subject" });
  }
  const course = () => courses.find((c) => c.subject === subject.value) || null;
  const fillTargets = () => {
    const c = course();
    if (kind.value === "level") {
      targets.replaceChildren(h("label", { for: "assignment-level", text: "Level" }), c
        ? h("select", { id: "assignment-level", "data-testid": "assignment-level" }, c.levels.map((l) => h("option", { value: l }, l)))
        : h("input", { type: "text", id: "assignment-level", maxlength: 32, "data-testid": "assignment-level" }));
    } else if (c) {
      targets.replaceChildren(h("p", { class: "small" }, "Topics"),
        c.topics.map((t) => h("label", { class: "choice" }, h("input", { type: "checkbox", value: t, "data-testid": "assignment-topic" }), t)));
    } else {
      targets.replaceChildren(h("label", { for: "assignment-topics", text: "Topics (separated by commas)" }),
        h("input", { type: "text", id: "assignment-topics", maxlength: 2000, "data-testid": "assignment-topics" }));
    }
  };
  kind.addEventListener("change", fillTargets);
  if (courses.length) subject.addEventListener("change", fillTargets); // a course's own topics and levels
  fillTargets();
  const form = h("form", {
    on: {
      submit: async (e) => {
        e.preventDefault();
        const button = form.querySelector("button[type=submit]");
        const when = new Date(due.value); // read in the teacher's own time zone
        if (!title.value.trim() || isNaN(when)) return;
        const body = { title: title.value.trim(), description: description.value.trim(), domain: subject.value.trim(),
          due_at: when.toISOString() };
        if (kind.value === "level") {
          body.target_level = targets.querySelector("#assignment-level").value.trim();
          if (course()) body.framework_id = course().framework;
        } else {
          const boxes = [...targets.querySelectorAll("input[type=checkbox]:checked")].map((b) => b.value);
          const typed = targets.querySelector("#assignment-topics");
          body.topics = typed ? typed.value.split(",").map((t) => t.trim()).filter(Boolean) : boxes;
        }
        button.disabled = true;
        try {
          await post(`/classes/${enc(classId)}/assignments`, body);
          done();
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
          button.disabled = false;
        }
      },
    },
  }, feedback,
  h("label", { for: "assignment-title", text: "Title" }), title,
  h("label", { for: "assignment-description", text: "Instructions (optional)" }), description,
  h("label", { for: "assignment-subject", text: "Course" }), subject,
  h("label", { for: "assignment-kind", text: "Assign" }), kind, targets,
  h("label", { for: "assignment-due", text: "Due (your time)" }), due,
  h("p", { class: "muted small" }, "Every learner in the class gets it as a goal, and so does anyone who joins later, with this due date. " +
    "Past the due date it shows as overdue; nothing is locked or cancelled."),
  h("button", { type: "submit", "data-testid": "create-assignment" }, "Assign to the class"));
  return h("div", { class: "card", "data-testid": "new-assignment" }, h("h3", { text: "New assignment" }), form);
}

async function renderAssignment(classId, assignmentId) {
  const show = painter();
  show(busy("Loading…"));
  const back = h("p", {}, h("a", { href: `#/class/${enc(classId)}` }, "← Class"));
  let detail;
  try {
    detail = await get(`/classes/${enc(classId)}/assignments/${enc(assignmentId)}`);
  } catch (err) {
    return show(back, h("h1", { text: "Assignment" }), errorBox(err));
  }
  const a = detail.assignment;
  const open = a.status === "ACTIVE";
  const feedback = h("div");
  const due = h("input", { type: "datetime-local", id: "change-due", value: localInput(a.due_at), "data-testid": "change-due-input" });
  const change = h("form", {
    on: {
      submit: async (e) => {
        e.preventDefault();
        const when = new Date(due.value);
        if (isNaN(when)) return;
        try {
          await patch(`/classes/${enc(classId)}/assignments/${enc(assignmentId)}`, { due_at: when.toISOString() });
          renderAssignment(classId, assignmentId);
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
        }
      },
    },
  }, h("label", { for: "change-due", text: "Due (your time)" }), due,
  h("p", { class: "muted small" }, "Every learner's goal gets the new date. Their plans stay as they are."),
  h("button", { type: "submit", class: "secondary", "data-testid": "change-due" }, "Change due date"));
  const cancel = h("button", {
    type: "button", class: "secondary", "data-testid": "cancel-assignment",
    on: {
      click: async () => {
        if (!window.confirm("Cancel this assignment? Learners' goals for it that are not completed are cancelled. This cannot be undone.")) return;
        try {
          await post(`/classes/${enc(classId)}/assignments/${enc(assignmentId)}/cancel`);
          renderAssignment(classId, assignmentId);
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
        }
      },
    },
  }, "Cancel assignment");
  const rows = detail.learners.map((l) => h("tr", { "data-testid": "assignment-learner", "data-learner": l.learner_id },
    h("td", {}, h("a", { href: `#/class/${enc(classId)}/learner/${enc(l.learner_id)}` }, l.display_name),
      l.source === "joined" ? h("div", { class: "muted small" }, "joined later") : null),
    h("td", {}, assignmentBadge(l.status), l.paused ? h("span", { class: "badge", "data-testid": "paused" }, "paused") : null,
      l.completed_late ? h("span", { class: "badge warn", "data-testid": "completed-late" }, "completed late") : null),
    h("td", { class: "small" }, l.planned ? [meter(l.percent_complete / 100, l.status === "COMPLETED"),
      h("span", { class: "muted" }, `${l.objectives_mastered} of ${l.objectives_required} objectives`)]
      : h("span", { class: "muted" }, l.status === "NOT_ASSIGNED" ? "the course no longer has these topics" : "no path planned yet")),
    h("td", { class: "small muted" }, l.topics.map((t) => `${t.topic}: ${t.mastered}/${t.total}`).join(" · ")),
    h("td", { class: "small muted" }, l.completed_at ? `Completed ${fmtDate(l.completed_at)}` : "")));
  show(back,
    h("div", { class: "row spread" }, h("h1", { text: a.title }), !open ? assignmentBadge("CANCELLED")
      : a.overdue ? h("span", { class: "badge bad" }, "past due") : h("span", { class: "badge" }, "open")),
    h("div", { class: "card", "data-testid": "assignment-summary" },
      a.description ? h("p", { text: a.description }) : null,
      h("p", { class: "small" }, a.topics.length ? `Topics: ${a.topics.join(", ")}` : `${a.domain} · level ${a.target_level}`),
      h("p", { class: "small", "data-testid": "assignment-due" }, `Due ${fmtDate(a.due_at)}`),
      h("p", { class: "muted small", "data-testid": "assignment-totals" }, assignmentTotals(a.totals))),
    feedback,
    open ? h("div", { class: "card" }, change, h("p", {}), cancel) : null,
    h("h2", { text: "Learners" }),
    h("div", { class: "card", "data-testid": "assignment-learners" }, rows.length
      ? h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, ["Learner", "Status", "Progress", "Topics", "Completed"].map((t) => h("th", { class: "small" }, t)))),
        h("tbody", {}, rows)))
      : h("p", { class: "muted" }, "No learners in the class.")));
}

// --- lesson approval (Important item 2) ----------------------------------------------------------------------------

function approvalCard(classId, info, opts) {
  const feedback = h("div");
  const box = h("input", {
    type: "checkbox", checked: info.lesson_approval, "data-testid": "lesson-approval",
    on: {
      change: async (e) => {
        box.disabled = true;
        try {
          await patch(`/classes/${enc(classId)}`, { lesson_approval: e.target.checked });
          renderClass(classId, opts);
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
          box.disabled = false;
        }
      },
    },
  });
  return h("div", { class: "card", "data-testid": "approval-setting" },
    h("label", { class: "choice" }, box, "Approve lessons before learners see them"),
    h("p", { class: "muted small" },
      "New lessons for this class's learners wait for you: you preview each one and release it or not. " +
      "Lessons requested before you turn this on are not affected."), feedback);
}

function pendingLessons(classId, items, opts) {
  if (!items.length) {
    return h("div", { class: "card", "data-testid": "pending-lessons" }, h("p", { class: "muted" }, "No lesson is waiting for you."));
  }
  return h("div", { "data-testid": "pending-lessons" }, items.map((l) => {
    const feedback = h("div");
    const preview = h("div", { "data-testid": "lesson-preview" });
    const ready = l.task_status === "COMPLETED";
    const decide = (verb) => async () => {
      let note = null;
      if (verb === "reject") {
        note = window.prompt("Reject this lesson? It is not shown to the learner, and their session stops. Optional note:", "");
        if (note === null) return;
      }
      try {
        await post(`/classes/${enc(classId)}/lessons/${enc(l.task_id)}/${verb}`, { note: note || null });
        renderClass(classId, opts);
      } catch (err) {
        feedback.replaceChildren(errorBox(err));
      }
    };
    const show = async () => {
      preview.replaceChildren(busy("Loading the lesson…"));
      try {
        preview.replaceChildren(readingNode(await get(`/classes/${enc(classId)}/lessons/${enc(l.task_id)}/preview`)));
      } catch (err) {
        preview.replaceChildren(errorBox(err));
      }
    };
    return h("div", { class: "card", "data-testid": "pending-lesson", "data-task": l.task_id },
      h("div", { class: "row spread" }, h("strong", { text: l.topic }), h("span", { class: "badge warn" }, "waiting for you")),
      h("p", { class: "muted small" }, [l.display_name, l.subject, fmtDate(l.requested_at)].filter(Boolean).join(" · ")),
      ready ? null : h("p", { class: "muted small" }, "Still being prepared. You can preview it once it is ready."),
      h("div", { class: "row" },
        h("button", { type: "button", class: "secondary", disabled: !ready, "data-testid": "preview", on: { click: show } }, "Preview"),
        h("button", { type: "button", disabled: !ready, "data-testid": "approve", on: { click: decide("approve") } }, "Approve"),
        h("button", { type: "button", class: "secondary", disabled: !ready, "data-testid": "reject", on: { click: decide("reject") } }, "Reject")),
      feedback, preview);
  }));
}

function readingNode(r) {
  const parts = [h("h3", { text: r.title || "Lesson" })];
  (r.introduction || []).forEach((p) => parts.push(h("p", { text: p })));
  (r.sections || []).forEach((s) => {
    if (s.heading) parts.push(h("h4", { text: s.heading }));
    (s.paragraphs || []).forEach((p) => parts.push(h("p", { text: p })));
    if (s.examples && s.examples.length) parts.push(h("ul", { class: "plain small" }, s.examples.map((e) => h("li", { text: e }))));
    if (s.analogy) parts.push(h("p", { class: "muted", text: s.analogy }));
  });
  (r.summary || []).forEach((p) => parts.push(h("p", { class: "muted", text: p })));
  if (r.withheld) parts.push(h("p", { class: "muted small" }, `${plural(r.withheld, "passage")} held back because it states an answer.`));
  return h("div", { class: "reading" }, parts);
}

const OVERRIDE_LABELS = { CORRECT: "Correct", PARTIAL: "Partly correct", INCORRECT: "Incorrect" };

function overrideButtons(classId, u, done) {
  const feedback = h("div");
  const buttons = Object.entries(OVERRIDE_LABELS).map(([outcome, label]) => h("button", {
    type: "button", class: "secondary", "data-testid": `override-${outcome.toLowerCase()}`,
    on: {
      click: async () => {
        if (!window.confirm(`Mark this answer as "${label}"? It then counts for the learner's progress, once.`)) return;
        try {
          await post(`/classes/${enc(classId)}/attempts/${enc(u.attempt_id)}/override`, { outcome });
          done();
        } catch (err) {
          feedback.replaceChildren(errorBox(err));
        }
      },
    },
  }, label));
  return h("div", {}, h("div", { class: "row" }, buttons), feedback);
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

function uncertainList(items, settled, withLearner, classId, done) {
  if (!items.length) {
    return h("div", { class: "card", "data-testid": "uncertain" }, h("p", { class: "muted" },
      settled ? "No answer of these learners was left undecided." : "Nothing to review. Every answer was graded."));
  }
  return h("div", { "data-testid": "uncertain" }, items.map((u) => h("div", { class: "card", "data-testid": "uncertain-item" },
    h("div", { class: "row spread" },
      h("strong", { text: withLearner ? u.display_name : u.concept_name }),
      u.override ? h("span", { class: "badge good", "data-testid": "overridden" }, `marked ${OVERRIDE_LABELS[u.override.outcome] || u.override.outcome}`)
        : u.resolved ? h("span", { class: "badge good" }, "settled by a later attempt") : h("span", { class: "badge warn" }, "to review")),
    h("p", { class: "muted small" }, [withLearner ? u.concept_name : null, SOURCE_LABELS[u.source] || u.source,
      `attempt ${u.attempt_number}`, fmtDate(u.submitted_at)].filter(Boolean).join(" · ")),
    h("p", { class: "question-text", text: u.prompt }),
    u.choices.length ? h("ul", { class: "plain small" }, u.choices.map((c) => h("li", { text: c }))) : null,
    h("div", { class: "answer" }, h("span", { class: "who" }, "Learner's answer"), h("span", { "data-testid": "learner-answer", text: u.learner_answer })),
    u.later_attempts && !u.resolved ? h("p", { class: "muted small" }, `${plural(u.later_attempts, "later attempt")}, none graded yet.`) : null,
    !u.resolved && classId ? overrideButtons(classId, u, done) : null)));
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
  nodes.push(h("h2", { text: "Answers to review" }), uncertainList(uncertain.filter((u) => u.learner_id === learnerId), true, false, classId,
    () => renderLearner(classId, learnerId)));
  show(nodes);
}

document.addEventListener("DOMContentLoaded", boot);
