from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.errors import StyleGuideError
from app.main import app
from app.services.project_service import reset_project_services
from app.services.style_guide_service import (
    StyleGuideService,
    reset_style_guide_service,
)
from app.utils.file_parser import parse_style_guide_bytes


@pytest.fixture(autouse=True)
def _reset_singleton():
    get_settings.cache_clear()
    reset_project_services()
    reset_style_guide_service()
    yield
    reset_style_guide_service()
    reset_project_services()
    get_settings.cache_clear()


def _write_project(root, name: str, style: str) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    (project_dir / "style.md").write_text(style, encoding="utf-8")
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": "ZH-EN",
                "source_lang": "zh",
                "target_lang": "en",
                "game": name,
                "style_guide": "style.md",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------- file_parser ----------


def test_parse_style_guide_bytes_basic():
    text = "# 风格指南\n保持武侠调性。\n禁止使用现代俚语。\n"
    parsed = parse_style_guide_bytes(text.encode("utf-8"), "guide.md")
    assert parsed.startswith("# 风格指南")
    assert "禁止使用现代俚语" in parsed
    # 整体两端 strip
    assert not parsed.endswith("\n")


def test_parse_style_guide_bytes_strips_bom():
    text = "保持武侠调性"
    raw = b"\xef\xbb\xbf" + text.encode("utf-8")
    assert parse_style_guide_bytes(raw, "guide.txt") == text


def test_parse_style_guide_bytes_normalizes_crlf():
    raw = "line1\r\nline2\rline3\n".encode("utf-8")
    out = parse_style_guide_bytes(raw, "guide.txt")
    assert out == "line1\nline2\nline3"


def test_parse_style_guide_bytes_rejects_unsupported_suffix():
    with pytest.raises(StyleGuideError):
        parse_style_guide_bytes(b"hello", "guide.docx")


def test_parse_style_guide_bytes_rejects_empty():
    with pytest.raises(StyleGuideError):
        parse_style_guide_bytes(b"   \n\n  ", "guide.md")


def test_parse_style_guide_bytes_rejects_non_utf8():
    raw = "保持调性".encode("gbk")
    with pytest.raises(StyleGuideError):
        parse_style_guide_bytes(raw, "guide.txt")


def test_parse_style_guide_bytes_rejects_oversize():
    big = ("a" * (1 * 1024 * 1024 + 1)).encode("utf-8")
    with pytest.raises(StyleGuideError):
        parse_style_guide_bytes(big, "guide.md")


# ---------- StyleGuideService ----------


def test_service_load_and_info():
    svc = StyleGuideService()
    assert svc.loaded is False
    svc.load("rule A\nrule B", filename="guide.md")
    assert svc.loaded is True
    info = svc.info()
    assert info["filename"] == "guide.md"
    assert info["char_count"] == len("rule A\nrule B")
    assert info["line_count"] == 2
    assert info["loaded_at"] is not None


def test_service_load_rejects_empty():
    svc = StyleGuideService()
    with pytest.raises(StyleGuideError):
        svc.load("   \n  ")


def test_service_overwrites_on_reload():
    svc = StyleGuideService()
    svc.load("first", filename="a.md")
    svc.load("second", filename="b.md")
    assert svc.get_rules() == "second"
    assert svc.filename == "b.md"


def test_service_clear():
    svc = StyleGuideService()
    svc.load("rule")
    svc.clear()
    assert svc.loaded is False
    assert svc.filename is None
    assert svc.get_rules() == ""


def test_build_system_prompt_without_rules_returns_base():
    svc = StyleGuideService()
    out = svc.build_system_prompt()
    # 未加载时仅返回基础角色描述，不出现 "## 风格指南" 段
    assert "## 风格指南" not in out
    assert "本地化译者" in out


def test_build_system_prompt_with_rules_appends_section():
    svc = StyleGuideService()
    svc.load("- 保持武侠调性\n- 禁止现代俚语")
    out = svc.build_system_prompt()
    assert "## 风格指南" in out
    assert "保持武侠调性" in out


def test_build_system_prompt_accepts_custom_base():
    svc = StyleGuideService()
    svc.load("- rule")
    out = svc.build_system_prompt(base="You are a translator.")
    assert out.startswith("You are a translator.")
    assert "## 风格指南" in out
    assert "- rule" in out


def test_load_from_dir_picks_md_first(tmp_path):
    (tmp_path / "z.txt").write_text("from txt", encoding="utf-8")
    (tmp_path / "a.md").write_text("from md", encoding="utf-8")
    svc = StyleGuideService()
    assert svc.load_from_dir(tmp_path) is True
    assert svc.get_rules() == "from md"
    assert svc.filename == "a.md"


def test_load_from_dir_missing_dir_returns_false(tmp_path):
    svc = StyleGuideService()
    assert svc.load_from_dir(tmp_path / "does-not-exist") is False
    assert svc.loaded is False


def test_load_from_dir_empty_dir_returns_false(tmp_path):
    svc = StyleGuideService()
    assert svc.load_from_dir(tmp_path) is False
    assert svc.loaded is False


# ---------- API ----------


def test_get_style_guide_when_unloaded():
    client = TestClient(app)
    resp = client.get("/api/v1/style-guide")
    assert resp.status_code == 200
    body = resp.json()
    assert body["loaded"] is False
    assert body["char_count"] == 0
    assert body["preview"] == ""
    assert body["rules"] is None


def test_upload_and_get_style_guide():
    client = TestClient(app)
    content = "# 风格指南\n- 保持武侠调性\n- 禁止使用现代俚语\n".encode("utf-8")
    files = {"file": ("guide.md", content, "text/markdown")}

    resp = client.post("/api/v1/style-guide/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["filename"] == "guide.md"
    assert body["line_count"] == 3
    assert body["char_count"] > 0

    resp = client.get("/api/v1/style-guide")
    assert resp.status_code == 200
    body = resp.json()
    assert body["loaded"] is True
    assert body["filename"] == "guide.md"
    assert "保持武侠调性" in body["preview"]
    assert body["rules"] is None  # 默认不返回完整正文

    resp = client.get("/api/v1/style-guide?full=true")
    assert resp.status_code == 200
    body = resp.json()
    assert body["rules"] is not None
    assert "禁止使用现代俚语" in body["rules"]


def test_upload_rejects_unsupported_format():
    client = TestClient(app)
    files = {"file": ("guide.docx", b"hello", "application/octet-stream")}
    resp = client.post("/api/v1/style-guide/upload", files=files)
    assert resp.status_code == 400


def test_upload_rejects_empty_body():
    client = TestClient(app)
    files = {"file": ("guide.md", b"   \n\n", "text/markdown")}
    resp = client.post("/api/v1/style-guide/upload", files=files)
    assert resp.status_code == 400


def test_project_id_gets_and_uploads_project_style_guide(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    _write_project(projects, "wwm/zh-en", "WWM style")
    monkeypatch.setenv("PROJECTS_DIR", str(projects))
    monkeypatch.setenv("DEFAULT_PROJECT", "wwm/zh-en")
    get_settings.cache_clear()
    reset_project_services()
    client = TestClient(app)

    resp = client.get("/api/v1/style-guide?project_id=wwm/zh-en&full=true")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["loaded"] is True
    assert body["rules"] == "WWM style"

    files = {"file": ("guide.md", b"NRC style", "text/markdown")}
    resp = client.post("/api/v1/style-guide/upload?project_id=wwm/zh-en", files=files)
    assert resp.status_code == 200, resp.text

    resp = client.get("/api/v1/style-guide?project_id=wwm/zh-en&full=true")
    assert resp.status_code == 200
    assert resp.json()["rules"] == "NRC style"

    resp = client.get("/api/v1/style-guide")
    assert resp.status_code == 200
    assert resp.json()["loaded"] is False
