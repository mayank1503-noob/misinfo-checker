# VeriLens frontend (React + Vite)

The browser client. The backend is untouched by it: `backend/api/flask_app.py`
already serves `frontend/` as static files, so **this project builds into
`../frontend/`** and Flask picks it up with no change to Python.

```
web/            <- source (this directory)
frontend/       <- build output, served by Flask. Do not edit by hand.
```

## Run it

Production shape — one server, one origin, no CORS:

```bash
npm install
npm run build                       # writes ../frontend
python -m backend.api.flask_app     # http://127.0.0.1:5000
```

Dev, with hot reload:

```bash
python -m backend.api.flask_app     # keep this running in another terminal
npm run dev                         # http://127.0.0.1:5173
```

The dev server proxies `/api` to `http://127.0.0.1:5000`, so the client code is
identical in both modes — it always talks to a same-origin `/api`. Point the
proxy somewhere else with `API_ORIGIN=http://host:port npm run dev`.

`npm run build` empties `../frontend` first, so anything you want to keep lives
here, not there.

## What talks to what

| Page | Endpoint |
| --- | --- |
| Multimodal analysis | `POST /api/check/{text,link,image,video}` |
| Sidebar status | `GET /api/health` |

`?graph=1` and `?agentic=1` are the two checkboxes on the analysis page; both
are query flags the API already understands. Image and video go up as
`multipart/form-data` with `file` and an optional `caption`; text and link go up
as JSON. The response shape is `backend/api/shape.py` — the same JSON the bot's
FastAPI door returns.

Everything else on the page (the radar, the review queue, the pre-run evidence
view) is illustrative and says so on screen; it is not fed by the API.

## Layout

```
src/api.js            fetch wrapper, API base, label/percent helpers
src/ui.jsx            toast, reveal-on-scroll card, animated bar, counter, ripple
src/App.jsx           sidebar, page switching, last-result state, health probe
src/pages/*.jsx       one file per sidebar entry
src/styles.css        the original stylesheet, unchanged
```

Every animation checks `prefers-reduced-motion` and settles to its final state
when it is set.
