import json
import re
from pathlib import Path

import pytest
from fakes import TEST_HOSTS, FakeSource
from fastapi.testclient import TestClient

from visionflow.api import create_app
from visionflow.jobs import JobRunner

WEB = Path(__file__).resolve().parents[1] / "web"


@pytest.fixture
def runner(tmp_path):
    r = JobRunner(tmp_path / "out", FakeSource())
    yield r
    r.shutdown()


@pytest.fixture
def client(runner, tmp_path):
    return TestClient(create_app(runner, tmp_path / "uploads", allowed_hosts=TEST_HOSTS))


def load_i18n() -> dict[str, dict[str, str]]:
    text = (WEB / "i18n.js").read_text(encoding="utf-8")
    prefix = "window.I18N = "
    start = text.index(prefix) + len(prefix)
    end = text.rindex(";")
    return json.loads(text[start:end])


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "/static/app.js" in r.text


def test_i18n_tr_and_en_have_same_keys():
    data = load_i18n()
    assert set(data) == {"tr", "en"}
    assert set(data["tr"]) == set(data["en"])
    assert all(isinstance(v, str) and v for lang in data.values() for v in lang.values())


def test_static_assets_served(client):
    for name, kind in (("app.js", "javascript"), ("i18n.js", "javascript"), ("style.css", "css")):
        r = client.get(f"/static/{name}")
        assert r.status_code == 200
        assert kind in r.headers["content-type"]


def test_every_data_i18n_key_used_in_html_exists():
    html = (WEB / "index.html").read_text(encoding="utf-8")
    keys = set(re.findall(r'data-i18n(?:-[a-z-]+)?="([^"]+)"', html))
    assert keys
    data = load_i18n()
    for lang in ("tr", "en"):
        assert keys <= set(data[lang]), keys - set(data[lang])


def test_api_still_reachable_next_to_ui(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_host_header_must_be_allowed(client):
    r = client.get("/api/health", headers={"Host": "evil.example"})
    assert r.status_code == 403
    assert "detail" in r.json()
    assert client.get("/", headers={"Host": "evil.example"}).status_code == 403
    for host in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000"):
        assert client.get("/api/health", headers={"Host": host}).status_code == 200


def test_custom_allowed_hosts(runner, tmp_path):
    app = create_app(runner, tmp_path / "u", allowed_hosts={"myhost"})
    c = TestClient(app)
    assert c.get("/api/health", headers={"Host": "myhost:80"}).status_code == 200
    assert c.get("/api/health", headers={"Host": "localhost"}).status_code == 403


def test_cross_origin_post_is_rejected(client):
    r = client.post(
        "/api/jobs",
        json={"videos": []},
        headers={"Host": "localhost:8000", "Origin": "http://evil.example"},
    )
    assert r.status_code == 403
    r = client.delete(
        "/api/jobs/" + "0" * 32,
        headers={"Host": "localhost:8000", "Origin": "http://localhost:9999"},
    )
    assert r.status_code == 403


def test_same_origin_post_is_allowed(client):
    r = client.post(
        "/api/jobs",
        json={"videos": []},
        headers={"Host": "localhost:8000", "Origin": "http://localhost:8000"},
    )
    assert r.status_code == 400  # reached the handler (bad_video_count), not blocked
    r = client.post("/api/jobs", json={"videos": []})  # no Origin header at all
    assert r.status_code == 400


def test_js_t_keys_exist_in_both_languages():
    js = (WEB / "app.js").read_text(encoding="utf-8")
    keys = set(re.findall(r'\bt\("([A-Za-z.]+[A-Za-z])"', js))
    data = load_i18n()
    for lang in ("tr", "en"):
        assert keys <= set(data[lang]), keys - set(data[lang])
    assert "result.retry" in keys
