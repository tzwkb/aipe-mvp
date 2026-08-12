from fastapi.testclient import TestClient

from app.main import app
from app.schemas.capabilities import API_VERSION, CAPABILITY_SCHEMA_VERSION

client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "yanyun-ai-translate"
    assert body["version"] == API_VERSION
    assert body["capabilities_url"] == "/api/v1/capabilities"


def test_capabilities_contract() -> None:
    response = client.get("/api/v1/capabilities")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == CAPABILITY_SCHEMA_VERSION
    assert body["api_version"] == API_VERSION
    assert body["capabilities"]["scoped_rag"] == {
        "version": "1.0.0",
        "method": "POST",
        "path": "/api/v1/rag/search",
        "scopes": ["global", "project", "special"],
        "server_resolved_collection": True,
        "legacy_unscoped_collection": True,
    }
    assert body["capabilities"]["web_search"] == {
        "version": "1.0.0",
        "method": "POST",
        "path": "/api/v1/web-search",
        "providers": ["bocha"],
        "project_aware": True,
    }


def test_capabilities_openapi_contract_is_typed() -> None:
    response = app.openapi()["paths"]["/api/v1/capabilities"]["get"]["responses"]["200"]

    assert response["content"]["application/json"]["schema"]["$ref"].endswith(
        "/CapabilitiesResponse"
    )


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["service"] == "yanyun-ai-translate"
