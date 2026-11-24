# Local development on macOS: live preview

Edit the student app and the teacher dashboard in VS Code (with Claude Code) on one side of the screen and watch them
in Chrome on the other. With live reload on, an open page reloads by itself a moment after an HTML, CSS or JS file in
`app/api/web/` changes. You don't need to touch a file, press Cmd+R or restart the server. There is no build step:
the apps are plain HTML, CSS and dependency-free JavaScript served by the API.

## One-time setup

```bash
brew install python@3.11 ffmpeg       # ffmpeg is used by lesson videos (or see TA_VIDEO_COMPOSER below)
cd teaching_agent
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Create development sign-in tokens, kept **outside** the repository. Make two random tokens:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(24))"   # run twice: one for the student, one for the teacher
```

Put them in `~/.teaching-agent-tokens.json`:

```json
{
  "<student token>": {"user_id": "dev-student", "learner_ids": ["demo-learner"]},
  "<teacher token>": {"user_id": "dev-teacher", "roles": ["teacher"]}
}
```

## Start the server

In a VS Code terminal, from the repository root:

```bash
source .venv/bin/activate
TA_AUTH_MODE=static \
TA_AUTH_STATIC_TOKENS_FILE="$HOME/.teaching-agent-tokens.json" \
TA_WORKER_EMBEDDED=true \
TA_DATA_DIR="$HOME/.teaching-agent-dev" \
TA_LOG_JSON=false \
TA_DEV_LIVE_RELOAD=true \
uvicorn app.api.main:app --reload --port 8000
```

- `TA_DEV_LIVE_RELOAD=true` turns live reload on. Leave it out (or set `false`) to serve the apps exactly as in
  production.
- `--reload` restarts the server when a Python file changes. Frontend edits don't restart it: they are served
  straight from disk, so a lesson being generated is not interrupted.
- `TA_DATA_DIR` keeps the preview's SQLite database and files in their own folder. Delete that folder to start over.
- If `ffmpeg -version` fails, add `TA_VIDEO_COMPOSER=mock`.
- The providers are offline mocks by default, so nothing calls a paid API.

## Preview URLs

| App | URL | Sign in with |
|---|---|---|
| Student web app | http://localhost:8000/app/ | the student token |
| Teacher dashboard | http://localhost:8000/app/teacher.html | the teacher token |
| Privacy notice | http://localhost:8000/app/privacy.html | (public) |

Open them in Chrome, not in VS Code's built-in browser pane. The apps send `X-Frame-Options: DENY`, so they cannot
be shown inside an embedded frame. Put VS Code on the left half of the screen and Chrome on the right (hold the
green window button, or drag each window to a screen edge on macOS 15+).

Each tab keeps its own sign-in (sessionStorage), so one tab can be the student and another the teacher. Open the
second tab by typing its URL, because "Duplicate tab" copies the first tab's sign-in. To see a learner in the dashboard,
create a class in the teacher tab, then enter its join code on the student app's Progress page.

## How live reload behaves

- Saving `app.css`, `app.js`, `teacher.js`, `index.html`, `teacher.html` or `privacy.html` reloads every open app
  page within about a second, once the server is serving the new file.
- The page keeps its route (`#/progress`, a class, ...) and its sign-in. Anything typed but not sent is lost, as with
  any reload.
- Each edit reloads a page once. A server restart (a Python edit) does not reload the page: it waits for the server
  to come back.
- The version polls (`/app/dev-reload/version`) are left out of the access log.

To turn it off, stop the server and start it without `TA_DEV_LIVE_RELOAD=true`. With it off (the default), the
reload script and its endpoint don't exist, and `TA_DEPLOYMENT=production` refuses to start with it on. It never
changes the Content-Security-Policy, the security headers or authentication. The script is a same-origin file, and
the version endpoint returns only a hash of the app files.
