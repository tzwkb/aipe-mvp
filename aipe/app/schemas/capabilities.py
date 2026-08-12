from typing import Literal

from pydantic import BaseModel

SERVICE_NAME = "yanyun-ai-translate"
API_VERSION = "1.1.0"
CAPABILITY_SCHEMA_VERSION = "1.0.0"


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: Literal["yanyun-ai-translate"] = SERVICE_NAME
    version: Literal["1.1.0"] = API_VERSION
    capabilities_url: Literal["/api/v1/capabilities"] = "/api/v1/capabilities"


class ScopedRAGCapability(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    method: Literal["POST"] = "POST"
    path: Literal["/api/v1/rag/search"] = "/api/v1/rag/search"
    scopes: list[Literal["global", "project", "special"]] = [
        "global",
        "project",
        "special",
    ]
    server_resolved_collection: Literal[True] = True
    legacy_unscoped_collection: Literal[True] = True


class WebSearchCapability(BaseModel):
    version: Literal["1.0.0"] = "1.0.0"
    method: Literal["POST"] = "POST"
    path: Literal["/api/v1/web-search"] = "/api/v1/web-search"
    providers: list[Literal["bocha"]] = ["bocha"]
    project_aware: Literal[True] = True


class CapabilitySet(BaseModel):
    scoped_rag: ScopedRAGCapability
    web_search: WebSearchCapability


class CapabilitiesResponse(BaseModel):
    schema_version: Literal["1.0.0"] = CAPABILITY_SCHEMA_VERSION
    service: Literal["yanyun-ai-translate"] = SERVICE_NAME
    api_version: Literal["1.1.0"] = API_VERSION
    capabilities: CapabilitySet


__all__ = [
    "API_VERSION",
    "CAPABILITY_SCHEMA_VERSION",
    "SERVICE_NAME",
    "CapabilitiesResponse",
    "CapabilitySet",
    "HealthResponse",
    "ScopedRAGCapability",
    "WebSearchCapability",
]
