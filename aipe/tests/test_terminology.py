from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app
from app.schemas.terminology import TermEntry
from app.services.project_service import reset_project_services
from app.services.terminology_service import (
    TerminologyService,
    reset_terminology_service,
)
from app.utils.file_parser import parse_terminology_bytes


@pytest.fixture(autouse=True)
def _reset_singleton():
    get_settings.cache_clear()
    reset_project_services()
    reset_terminology_service()
    yield
    reset_terminology_service()
    reset_project_services()
    get_settings.cache_clear()


def _make_xlsx(rows: list[tuple[str, str]]) -> bytes:
    df = pd.DataFrame(rows, columns=["中文", "英语"])
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _write_project(root, name: str, rows: list[tuple[str, str]]) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    terms = [{"source": source, "target": target} for source, target in rows]
    (project_dir / "terms.json").write_text(json.dumps(terms, ensure_ascii=False), encoding="utf-8")
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": "ZH-EN",
                "source_lang": "zh",
                "target_lang": "en",
                "game": name,
                "terminology": "terms.json",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


# ---------- file_parser ----------


def test_parse_terminology_bytes_basic():
    data = _make_xlsx([("燕云十六声", "Yanyun"), ("无名剑法", "Nameless Sword")])
    parsed = parse_terminology_bytes(data, "术语表.xlsx")
    assert parsed == [
        {"source": "燕云十六声", "target": "Yanyun"},
        {"source": "无名剑法", "target": "Nameless Sword"},
    ]


def test_parse_skips_blank_and_dup():
    data = _make_xlsx(
        [
            ("燕云", "Yanyun"),
            ("", "EmptySrc"),
            ("EmptyTgt", ""),
            ("燕云", "DupSourceShouldDrop"),
            ("无名剑法", "Nameless Sword"),
        ]
    )
    parsed = parse_terminology_bytes(data, "t.xlsx")
    assert [(p["source"], p["target"]) for p in parsed] == [
        ("燕云", "Yanyun"),
        ("无名剑法", "Nameless Sword"),
    ]


def test_parse_csv_with_english_headers():
    csv = b"source,target\nfoo,bar\nbaz,qux\n"
    parsed = parse_terminology_bytes(csv, "t.csv")
    assert parsed == [
        {"source": "foo", "target": "bar"},
        {"source": "baz", "target": "qux"},
    ]


# ---------- TerminologyService.find_matches ----------


def test_find_matches_long_term_priority():
    svc = TerminologyService()
    svc.load(
        [
            TermEntry(source="无名剑法", target="Nameless Sword"),
            TermEntry(source="无名", target="Nameless"),
        ]
    )
    text = "他施展无名剑法击退了无名敌人。"
    matches = svc.find_matches(text)

    sources = [m.source for m in matches]
    # 长术语优先匹配，「无名」单独出现时再走短术语
    assert "无名剑法" in sources
    assert "无名" in sources
    # 同一术语多次出现只返回一次
    assert sources.count("无名剑法") == 1
    assert sources.count("无名") == 1


def test_find_matches_no_hit_returns_empty():
    svc = TerminologyService()
    svc.load([TermEntry(source="燕云", target="Yanyun")])
    assert svc.find_matches("完全没有命中术语的句子。") == []


def test_find_matches_empty_text():
    svc = TerminologyService()
    svc.load([TermEntry(source="燕云", target="Yanyun")])
    assert svc.find_matches("") == []


def test_find_matches_unloaded_service():
    svc = TerminologyService()
    assert svc.find_matches("任意文本") == []


def test_find_matches_returns_full_entry_with_category():
    svc = TerminologyService()
    svc.load([TermEntry(source="燕云", target="Yanyun", category="地名")])
    matches = svc.find_matches("我从燕云来")
    assert len(matches) == 1
    assert matches[0].source == "燕云"
    assert matches[0].target == "Yanyun"
    assert matches[0].category == "地名"


# ---------- API ----------


def test_upload_and_list_terminology():
    client = TestClient(app)
    data = _make_xlsx([("燕云", "Yanyun"), ("无名剑法", "Nameless Sword")])
    files = {"file": ("术语表.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}

    resp = client.post("/api/v1/terminology/upload", files=files)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 2
    assert body["added"] == 2

    resp = client.get("/api/v1/terminology")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 2
    assert {item["source"] for item in body["items"]} == {"燕云", "无名剑法"}


def test_upload_rejects_unsupported_format():
    client = TestClient(app)
    files = {"file": ("t.txt", b"hello", "text/plain")}
    resp = client.post("/api/v1/terminology/upload", files=files)
    assert resp.status_code == 400


def test_project_id_lists_and_uploads_project_terminology(tmp_path, monkeypatch):
    projects = tmp_path / "projects"
    _write_project(projects, "wwm/zh-en", [("燕云", "Where Winds Meet")])
    monkeypatch.setenv("PROJECTS_DIR", str(projects))
    monkeypatch.setenv("DEFAULT_PROJECT", "wwm/zh-en")
    get_settings.cache_clear()
    reset_project_services()
    client = TestClient(app)

    resp = client.get("/api/v1/terminology?project_id=wwm/zh-en")
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["source"] == "燕云"

    files = {"file": ("术语表.xlsx", _make_xlsx([("洛克", "Roco")]), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")}
    resp = client.post("/api/v1/terminology/upload?project_id=wwm/zh-en", files=files)
    assert resp.status_code == 200, resp.text
    assert resp.json()["total"] == 1

    resp = client.get("/api/v1/terminology?project_id=wwm/zh-en")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 1
    assert body["items"][0]["source"] == "洛克"

    resp = client.get("/api/v1/terminology")
    assert resp.status_code == 200
    assert resp.json()["total"] == 0
