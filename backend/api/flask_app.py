"""
The browser's front door.

`main.py` (FastAPI) is what the bot talks to: it hands over a *path* to
a file it has already saved on the server. A browser cannot do that — it
has bytes in a form — so this app exists to close that gap and to serve
the frontend itself:

    python -m backend.api.flask_app          # http://127.0.0.1:5000

Serving `frontend/index.html` from the same origin as `/api/*` is the
reason there is no CORS configuration anywhere in this file. The page
and the API are one origin, so the browser never preflights.

Everything below is a thin shell. The endpoints validate the request,
save an upload if there is one, call the same `backend.pipeline`
functions the FastAPI app calls, and shape the result with the same
`shape.respond`. No checking logic lives here, and none should: if the
two front doors ever disagree about a verdict, that is a bug in one of
these shells, not a difference of opinion.

Failures inside the pipeline are not HTTP failures. A dead retriever or
a missing model produces `unverified` with an explanation, which is a
useful answer; only a genuinely unreadable request is a 4xx.
"""

import logging
import os
import re
import shutil
import tempfile

from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

from ..pipeline import analyze, analyze_image, analyze_link, analyze_text, analyze_video
from .shape import respond


log = logging.getLogger(__name__)


# Matches the FastAPI app, so a message that is too long is too long at
# either door.
MAX_TEXT = 10000

# A hard ceiling on an upload, enforced by Werkzeug before we ever read
# the body: the point is to reject a 2GB video at the socket, not after
# spooling it to disk.
MAX_UPLOAD_BYTES = 64 * 1024 * 1024

FRONTEND = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

TITLE = "VeriLens — AI-03 Multimodal Misinformation Intelligence"

# The built page is the manifest. Vite writes the entry bundle and the
# stylesheet into `frontend/index.html` with a content hash in each name,
# so rather than keep a second list of those names in sync (or guess by
# globbing `assets/`, which would also pick up lazy chunks the entry
# loads for itself), the template is handed whatever that file links.
_SCRIPT = re.compile(r"""<script[^>]+src=["']([^"']+)["']""", re.I)
_STYLE = re.compile(
    r"""<link[^>]+rel=["']stylesheet["'][^>]+href=["']([^"']+)["']""", re.I
)


class Invalid(Exception):
    """A request we can't read. Carries the status it deserves."""

    def __init__(self, message, status=422):
        super().__init__(message)
        self.status = status


def page_assets(frontend):
    """
    (stylesheets, scripts) for the built page, as URLs this app serves.

    An unbuilt or unreadable `frontend/` gives two empty lists, which the
    template renders as instructions rather than a blank page: a missing
    bundle is a setup step, not a server error.
    """
    try:
        with open(os.path.join(frontend, "index.html"), encoding="utf-8") as handle:
            built = handle.read()
    except OSError:
        return [], []

    def urls(pattern):
        # Vite writes "./assets/x" so the bundle also works opened off
        # disk; over HTTP that has to be rooted, or a URL one level deep
        # would resolve against the wrong directory.
        return [
            "/" + url.lstrip("./") if not url.startswith(("http", "/")) else url
            for url in pattern.findall(built)
        ]

    return urls(_STYLE), urls(_SCRIPT)


def create_app(frontend=FRONTEND):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

    frontend = os.path.abspath(frontend)

    # ---- the page -------------------------------------------------

    @app.get("/")
    def index():
        """
        The SPA shell, rendered from `templates/index.html`.

        The asset URLs come out of the build and the config comes out of
        this module, so the page never has to guess where the API is or
        what it will refuse.
        """
        styles, scripts = page_assets(frontend)

        return render_template(
            "index.html",
            title=TITLE,
            styles=styles,
            scripts=scripts,
            config={
                "apiBase": "/api",
                "maxTextChars": MAX_TEXT,
                "maxUploadBytes": MAX_UPLOAD_BYTES,
            },
        )

    @app.get("/<path:asset>")
    def asset(asset):
        """Any other static file the page asks for (css, js, images)."""
        return send_from_directory(frontend, asset)

    # ---- errors ---------------------------------------------------

    @app.errorhandler(Invalid)
    def _invalid(error):
        return jsonify(detail=str(error)), error.status

    @app.errorhandler(413)
    def _too_big(error):
        return jsonify(
            detail=(
                "Upload is too large. Maximum is "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
            )
        ), 413

    @app.errorhandler(404)
    def _missing(error):
        return jsonify(detail="Not found"), 404

    # ---- the API --------------------------------------------------

    @app.get("/api/health")
    def health():
        """Liveness, plus which optional retrievers and models are on."""
        from ..evidence.retrievers import RETRIEVERS
        from ..images import consistency, local_index
        from ..stance import nli, rank

        return jsonify(
            status="ok",
            service="misinformation-detection-api",
            api="flask",
            retrievers={
                name: module.available() for name, module in RETRIEVERS.items()
            },
            models={
                "embedder": rank.available(),
                "nli": nli.available(),
                "clip": consistency.available(),
            },
            image_index=local_index.available(),
            cache_dir=os.getenv("CACHE_DIR", "cache"),
        )

    @app.post("/api/check/text")
    def check_text():
        text = _field("text")

        if not text:
            raise Invalid("Text cannot be empty")

        if len(text) > MAX_TEXT:
            raise Invalid(
                f"Text is too long. Maximum length is {MAX_TEXT:,} characters.", 413
            )

        return _run(analyze_text, text)

    @app.post("/api/check/link")
    def check_link():
        url = _field("url")

        if not url:
            raise Invalid("URL cannot be empty")

        if not url.startswith(("http://", "https://")):
            raise Invalid("URL must start with http:// or https://")

        return _run(analyze_link, url)

    @app.post("/api/check/image")
    def check_image():
        return _media(analyze_image)

    @app.post("/api/check/video")
    def check_video():
        return _media(analyze_video)

    @app.post("/api/check/packet")
    def check_packet():
        """
        Check an already-built packet.

        Useful when ingestion happened elsewhere — a worker that did the
        OCR, a cached packet — and only the checking needs doing.
        """
        packet = request.get_json(silent=True)

        if not isinstance(packet, dict) or not packet.get("input_type"):
            raise Invalid("A packet needs at least an input_type")

        # Temp-path bookkeeping is ours, never the client's: a packet
        # posted with one would otherwise name directories for us to
        # delete.
        from ..analyzers.packet import TEMP_PATHS

        packet.pop(TEMP_PATHS, None)

        return _run(analyze, packet)

    return app


# ---- the shared plumbing ------------------------------------------


def _field(name, default=""):
    """
    Read one field from a JSON body or a form.

    The browser posts a form when there is a file attached and JSON when
    there isn't, and neither endpoint should care which.
    """
    body = request.get_json(silent=True)

    if isinstance(body, dict) and name in body:
        value = body.get(name)
    else:
        value = request.form.get(name, default)

    if value is None:
        return ""

    return value.strip() if isinstance(value, str) else value


def _flag(name):
    """A query flag, as the browser spells booleans."""
    return request.args.get(name, "").lower() in ("1", "true", "yes", "on")


def _media(function):
    """
    Run a media endpoint from either an upload or a server-side path.

    An upload is saved to a temp directory that we own and delete in
    `finally` — including when the pipeline raises. A path is checked to
    exist and is *not* deleted, because it isn't ours.
    """
    upload = request.files.get("file")
    caption = _field("caption")

    if upload and upload.filename:
        workdir = tempfile.mkdtemp(prefix="misinfo-upload-")

        try:
            # secure_filename can return "" for a name that is entirely
            # separators or dots, which would make us write the directory
            # itself; "upload" is the fallback.
            path = os.path.join(workdir, secure_filename(upload.filename) or "upload")
            upload.save(path)

            return _run(function, path, caption)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    path = _field("path")

    if not path:
        raise Invalid("Attach a file or give a path")

    if not os.path.isfile(path):
        raise Invalid(f"No such file: {path}", 404)

    return _run(function, path, caption)


def _run(function, *args, **kwargs):
    """Call a pipeline entry point, turning only real failures into 5xx."""
    graph = _flag("graph")

    try:
        result = function(*args, graph_json=graph, agentic=_flag("agentic"), **kwargs)
    except Exception as error:
        log.exception("pipeline failed")

        return jsonify(detail=f"Analysis failed: {error}"), 500

    return jsonify(respond(result, include_graph=graph))


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=os.getenv("DEBUG", "").lower() in ("1", "true"),
    )
