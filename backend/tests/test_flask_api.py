"""
The Flask front door.

`test_pipeline.py` already covers the FastAPI app, and the two share
`shape.respond`, so there is no point asserting the response shape twice.
What is tested here is what this app does that the other one cannot:
it serves the page, and it takes an *upload* rather than a server-side
path — which means it owns a temp file and has to delete it.

The pipeline itself is stubbed almost everywhere. These are tests of a
shell: validation, file handling, status codes. Whether the verdict is
right is `test_pipeline.py`'s job, and running the real thing here would
cost the suite a minute of torch imports per case.
"""

import io
import json
import os

import pytest


pytest.importorskip("flask")

from backend.api import flask_app                             # noqa: E402


SCAM = "SBI is giving Rs 5,000 cashback, forward to 10 people"

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c6360000002000100ffff03000006000557bfabd4000000"
    "0049454e44ae426082"
)


@pytest.fixture
def client():
    from backend.api.flask_app import create_app

    app = create_app()
    app.config["TESTING"] = True

    return app.test_client()


@pytest.fixture
def stub_pipeline(monkeypatch):
    """
    Replace the pipeline with something that records how it was called.

    Returns the list of calls, so a test can assert on the path and
    caption that reached it.
    """
    calls = []

    def fake(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})

        return {
            # `graph` only when it was asked for, exactly as the real
            # pipeline does it.
            **({"graph": {"nodes": [], "edges": []}} if kwargs.get("graph_json") else {}),
            "verdict": {
                "label": "unverified",
                "confidence": 0.1,
                "summary": "stub",
                "counts": {},
                "claims": [],
            },
            "packet": {"input_type": "text", "text": ""},
            "claims": [],
            "stages": {},
            "ms": 1.0,
        }

    for name in ("analyze", "analyze_text", "analyze_link",
                 "analyze_image", "analyze_video"):
        monkeypatch.setattr(f"backend.api.flask_app.{name}", fake)

    return calls


# --- the page ---------------------------------------------------------------


def test_it_serves_the_frontend_at_the_root(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.mimetype == "text/html"
    # The page is what makes same-origin work; if this ever 404s the
    # browser falls back to a cross-origin fetch and CORS bites.
    assert b"<html" in response.data.lower()
    assert b'<div id="root">' in response.data


def test_the_page_carries_the_built_assets_and_the_server_config(client):
    """
    The shell is a template, so these two things can go wrong silently:
    the asset names can stop matching the build (a blank page), and the
    config can stop being injected (the client falls back to guessing).
    """
    built = _built_asset_names()

    if not built:
        pytest.skip("no bundle in frontend/ - run `npm run build` in web/")

    body = client.get("/").get_data(as_text=True)

    for name in built:
        assert "/assets/" + name in body, name

    config = json.loads(
        body.split("window.__VERILENS__ = ", 1)[1].split(";</script>", 1)[0]
    )

    assert config["apiBase"] == "/api"
    # Not a copy of the numbers: the point of injecting them is that the
    # page and the server cannot drift.
    assert config["maxTextChars"] == flask_app.MAX_TEXT
    assert config["maxUploadBytes"] == flask_app.MAX_UPLOAD_BYTES


def test_an_unbuilt_frontend_explains_itself_instead_of_serving_nothing(tmp_path):
    """
    A clone without `npm run build` should say so. Rendering a bare
    `<div id="root">` would look like a broken server instead of a
    missing step, and 500ing would take the API down with the page.
    """
    app = flask_app.create_app(frontend=str(tmp_path))
    app.config["TESTING"] = True

    response = app.test_client().get("/")

    assert response.status_code == 200
    assert b"npm run build" in response.data
    assert b"<script type=\"module\"" not in response.data


def test_relative_asset_urls_are_rooted(tmp_path):
    """
    Vite writes "./assets/x" so the bundle also works opened off disk.
    Left alone, a URL one level deep would resolve against the wrong
    directory, so the server roots them on the way into the page.
    """
    (tmp_path / "index.html").write_text(
        '<script type="module" src="./assets/a.js"></script>'
        '<link rel="stylesheet" href="./assets/a.css">',
        encoding="utf-8",
    )

    styles, scripts = flask_app.page_assets(str(tmp_path))

    assert styles == ["/assets/a.css"]
    assert scripts == ["/assets/a.js"]


def _built_asset_names():
    assets = os.path.join(flask_app.FRONTEND, "assets")

    if not os.path.isdir(assets):
        return []

    return [
        name for name in os.listdir(assets) if name.endswith((".js", ".css"))
    ]


def test_a_missing_asset_is_json_not_an_html_error_page(client):
    response = client.get("/nope.js")

    assert response.status_code == 404
    assert response.get_json()["detail"]


# --- validation -------------------------------------------------------------


def test_check_text_validates_input(client, stub_pipeline):
    assert client.post("/api/check/text", json={"text": "   "}).status_code == 422
    assert client.post("/api/check/text", json={"text": "x" * 10001}).status_code == 413
    assert client.post("/api/check/text", json={}).status_code == 422


def test_check_link_validates_the_scheme(client, stub_pipeline):
    assert client.post("/api/check/link", json={"url": "notaurl"}).status_code == 422
    assert client.post("/api/check/link", json={"url": ""}).status_code == 422
    assert client.post("/api/check/link", json={"url": "https://x.test"}).status_code == 200


def test_media_needs_a_file_or_a_real_path(client, stub_pipeline):
    assert client.post("/api/check/image", json={"path": ""}).status_code == 422
    assert client.post("/api/check/image", json={"path": "no/such.jpg"}).status_code == 404
    assert client.post("/api/check/video", json={"path": "no/such.mp4"}).status_code == 404


def test_a_packet_needs_an_input_type(client, stub_pipeline):
    assert client.post("/api/check/packet", json={"nope": 1}).status_code == 422
    assert client.post("/api/check/packet", json={}).status_code == 422


def test_a_posted_packet_cannot_name_temp_paths_for_us_to_delete(client, stub_pipeline):
    from backend.analyzers.packet import TEMP_PATHS

    client.post(
        "/api/check/packet",
        json={"input_type": "text", "text": "hi", TEMP_PATHS: ["C:/Windows"]},
    )

    packet = stub_pipeline[0]["args"][0]

    assert TEMP_PATHS not in packet


# --- uploads: the reason this app exists ------------------------------------


def test_an_uploaded_image_reaches_the_pipeline_as_a_real_file(client, monkeypatch):
    seen = {}

    def fake(path, caption="", **kwargs):
        seen["path"] = path
        seen["caption"] = caption
        seen["existed"] = os.path.isfile(path)
        seen["bytes"] = open(path, "rb").read()

        return {
            "verdict": {"label": "unverified", "confidence": 0.1,
                        "summary": "stub", "counts": {}, "claims": []},
            "packet": {"input_type": "image"}, "claims": [],
            "stages": {}, "ms": 1.0,
        }

    monkeypatch.setattr("backend.api.flask_app.analyze_image", fake)

    response = client.post(
        "/api/check/image",
        data={"file": (io.BytesIO(PNG), "flood.png"), "caption": "flooding today"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert seen["existed"], "the pipeline must see a file that is actually on disk"
    assert seen["bytes"] == PNG, "the bytes must survive the round trip intact"
    assert seen["caption"] == "flooding today"


def test_the_temp_file_is_deleted_afterwards(client, monkeypatch):
    seen = {}

    def fake(path, caption="", **kwargs):
        seen["path"] = path

        return {
            "verdict": {"label": "unverified", "confidence": 0.1,
                        "summary": "stub", "counts": {}, "claims": []},
            "packet": {"input_type": "image"}, "claims": [],
            "stages": {}, "ms": 1.0,
        }

    monkeypatch.setattr("backend.api.flask_app.analyze_image", fake)

    client.post(
        "/api/check/image",
        data={"file": (io.BytesIO(PNG), "flood.png")},
        content_type="multipart/form-data",
    )

    assert not os.path.exists(seen["path"])
    assert not os.path.exists(os.path.dirname(seen["path"]))


def test_the_temp_file_is_deleted_even_when_the_pipeline_raises(client, monkeypatch):
    seen = {}

    def explode(path, caption="", **kwargs):
        seen["path"] = path

        raise RuntimeError("model missing")

    monkeypatch.setattr("backend.api.flask_app.analyze_image", explode)

    response = client.post(
        "/api/check/image",
        data={"file": (io.BytesIO(PNG), "flood.png")},
        content_type="multipart/form-data",
    )

    # A dead stage is a 500, but it must not also leak a file.
    assert response.status_code == 500
    assert response.get_json()["detail"]
    assert not os.path.exists(seen["path"])


def test_an_upload_cannot_escape_its_temp_directory(client, monkeypatch):
    """A filename is attacker-controlled; it must not steer the write."""
    seen = {}

    def fake(path, caption="", **kwargs):
        seen["path"] = path

        return {
            "verdict": {"label": "unverified", "confidence": 0.1,
                        "summary": "stub", "counts": {}, "claims": []},
            "packet": {"input_type": "image"}, "claims": [],
            "stages": {}, "ms": 1.0,
        }

    monkeypatch.setattr("backend.api.flask_app.analyze_image", fake)

    client.post(
        "/api/check/image",
        data={"file": (io.BytesIO(PNG), "../../../../evil.png")},
        content_type="multipart/form-data",
    )

    directory = os.path.dirname(seen["path"])

    assert os.path.basename(directory).startswith("misinfo-upload-")
    assert ".." not in seen["path"]


def test_a_path_given_by_the_caller_is_not_deleted(client, monkeypatch, tmp_path):
    """We delete what we saved. A file that was already there is not ours."""
    kept = tmp_path / "already-here.png"
    kept.write_bytes(PNG)

    monkeypatch.setattr(
        "backend.api.flask_app.analyze_image",
        lambda path, caption="", **kwargs: {
            "verdict": {"label": "unverified", "confidence": 0.1,
                        "summary": "stub", "counts": {}, "claims": []},
            "packet": {"input_type": "image"}, "claims": [],
            "stages": {}, "ms": 1.0,
        },
    )

    response = client.post("/api/check/image", json={"path": str(kept)})

    assert response.status_code == 200
    assert kept.exists()


# --- the shell's own behaviour ----------------------------------------------


def test_a_dead_stage_is_a_500_with_a_reason_not_a_crash(client, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("retriever is down")

    monkeypatch.setattr("backend.api.flask_app.analyze_text", explode)

    response = client.post("/api/check/text", json={"text": SCAM})

    assert response.status_code == 500
    assert "retriever is down" in response.get_json()["detail"]


def test_fields_can_arrive_as_json_or_as_a_form(client, stub_pipeline):
    """The page posts a form when a file is attached and JSON when not."""
    assert client.post("/api/check/text", json={"text": SCAM}).status_code == 200
    assert client.post("/api/check/text", data={"text": SCAM}).status_code == 200

    assert [call["args"][0] for call in stub_pipeline] == [SCAM, SCAM]


def test_the_graph_and_agentic_flags_reach_the_pipeline(client, stub_pipeline):
    client.post("/api/check/text?graph=true&agentic=1", json={"text": SCAM})

    kwargs = stub_pipeline[0]["kwargs"]

    assert kwargs["graph_json"] is True
    assert kwargs["agentic"] is True


def test_the_flags_default_to_off(client, stub_pipeline):
    client.post("/api/check/text", json={"text": SCAM})

    kwargs = stub_pipeline[0]["kwargs"]

    assert kwargs["graph_json"] is False
    assert kwargs["agentic"] is False


def test_health_reports_what_is_switched_on(client, monkeypatch):
    monkeypatch.setattr("backend.images.consistency.available", lambda: False)
    monkeypatch.setattr("backend.stance.rank.available", lambda: True)
    monkeypatch.setattr("backend.stance.nli.available", lambda: True)

    body = client.get("/api/health").get_json()

    assert body["status"] == "ok"
    assert body["api"] == "flask"
    assert set(body["retrievers"]) == {"factcheck", "seed_index", "web"}
    assert body["models"] == {"embedder": True, "nli": True, "clip": False}


def test_both_front_doors_shape_a_result_the_same_way():
    """
    The one thing that must never drift.

    Both apps call `shape.respond`, so this is really a guard against
    someone reintroducing a second copy of it.
    """
    from backend.api import flask_app, main, shape

    assert main.respond is shape.respond
    assert flask_app.respond is shape.respond
