from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.errors import StyleGuideError, TerminologyError
from app.services.style_guide_service import StyleGuideService
from app.services.terminology_service import TerminologyService
from app.utils.file_parser import parse_style_guide_bytes, parse_terminology_file

logger = logging.getLogger(__name__)


class ProjectProfileError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProjectAsset:
    asset_id: str
    kind: str
    path: Path | None
    required: bool
    availability: str
    authority: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    distribution: str = "internal_only"
    description: str = ""
    media_type: str | None = None
    content_schema: str | None = None


@dataclass(frozen=True)
class ProjectProfile:
    name: str
    language_pair: str
    source_lang: str
    target_lang: str
    game: str
    profile_dir: Path
    profile_contract_version: int = 1
    profile_status: str = "active"
    qdrant_collection: str | None = None
    reference_qdrant_collection: str | None = None
    reference_context: dict[str, Any] = field(default_factory=dict)
    web_search_prefix: str | None = None
    allow_web_search: bool = True
    background: str = ""
    style_guide_path: Path | None = None
    terminology_path: Path | None = None
    prompt_notes_path: Path | None = None
    checks_path: Path | None = None
    confirmed_rules_path: Path | None = None
    vision_system_prompt: str | None = None
    distribution: str = "internal_only"
    authority_policy: tuple[str, ...] = ()
    terminology_status: dict[str, Any] = field(default_factory=dict)
    assets: dict[str, ProjectAsset] = field(default_factory=dict)
    context_pipeline_mode: str = "off"
    capabilities: dict[str, dict[str, Any]] = field(default_factory=dict)
    module_context_views: dict[str, dict[str, Any]] = field(default_factory=dict)
    content_scope: dict[str, Any] = field(default_factory=dict)
    workflow: dict[str, Any] = field(default_factory=dict)
    gaps: tuple[dict[str, Any], ...] = ()


def _resolve_asset(profile_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(value)
    return p if p.is_absolute() else profile_dir / p


_ASSET_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ASSET_AVAILABILITY = {"included", "external", "missing"}
_CONTEXT_PIPELINE_MODES = {"off", "shadow", "enforce"}


def _mapping(value: Any, *, field_name: str, profile_path: Path) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ProjectProfileError(
            f"project profile {field_name} 必须是对象: {profile_path}"
        )
    return value


def _resolve_declared_asset_path(
    profile_dir: Path, value: str, profile_path: Path
) -> Path:
    raw_path = Path(value)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ProjectProfileError(
            f"project asset path 必须位于 profile 目录内: {profile_path}: {value!r}"
        )
    return profile_dir / raw_path


def _parse_project_assets(
    profile_dir: Path,
    raw_assets: Any,
    profile_path: Path,
) -> dict[str, ProjectAsset]:
    declarations = _mapping(raw_assets, field_name="assets", profile_path=profile_path)
    assets: dict[str, ProjectAsset] = {}
    for asset_id, raw in declarations.items():
        if not isinstance(asset_id, str) or not _ASSET_ID_RE.fullmatch(asset_id):
            raise ProjectProfileError(
                f"project profile asset id 非法: {profile_path}: {asset_id!r}"
            )
        if not isinstance(raw, dict):
            raise ProjectProfileError(
                f"project profile asset 声明必须是对象: {profile_path}: {asset_id}"
            )

        kind = str(raw.get("kind") or "").strip()
        if not kind:
            raise ProjectProfileError(
                f"project profile asset 缺少 kind: {profile_path}: {asset_id}"
            )
        required = raw.get("required", False)
        if not isinstance(required, bool):
            raise ProjectProfileError(
                f"project profile asset required 必须是布尔值: {profile_path}: {asset_id}"
            )
        availability = str(raw.get("availability") or "included").strip().lower()
        if availability not in _ASSET_AVAILABILITY:
            raise ProjectProfileError(
                f"project profile asset availability 非法: {profile_path}: {asset_id}={availability!r}"
            )

        raw_path = str(raw.get("path") or "").strip()
        resolved_path = (
            _resolve_declared_asset_path(profile_dir, raw_path, profile_path)
            if raw_path
            else None
        )
        if availability == "included":
            if resolved_path is None:
                raise ProjectProfileError(
                    f"included project asset 缺少 path: {profile_path}: {asset_id}"
                )
            if not resolved_path.exists():
                raise ProjectProfileError(
                    f"included project asset 不存在: {resolved_path}"
                )

        authority = _mapping(
            raw.get("authority"),
            field_name=f"assets.{asset_id}.authority",
            profile_path=profile_path,
        )
        provenance = _mapping(
            raw.get("provenance"),
            field_name=f"assets.{asset_id}.provenance",
            profile_path=profile_path,
        )
        assets[asset_id] = ProjectAsset(
            asset_id=asset_id,
            kind=kind,
            path=resolved_path,
            required=required,
            availability=availability,
            authority=authority,
            provenance=provenance,
            distribution=str(raw.get("distribution") or "internal_only").strip(),
            description=str(raw.get("description") or "").strip(),
            media_type=(
                str(raw["media_type"]).strip() if raw.get("media_type") else None
            ),
            content_schema=(
                str(raw["content_schema"]).strip()
                if raw.get("content_schema")
                else None
            ),
        )
    return assets


def _parse_context_contract(
    raw: dict[str, Any],
    *,
    profile_contract_version: int,
    assets: dict[str, ProjectAsset],
    profile_path: Path,
) -> tuple[str, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if profile_contract_version >= 2:
        for field_name in ("assets", "context_pipeline", "capabilities"):
            if field_name not in raw:
                raise ProjectProfileError(
                    f"profile contract v2 缺少 {field_name}: {profile_path}"
                )

    context_pipeline = _mapping(
        raw.get("context_pipeline"),
        field_name="context_pipeline",
        profile_path=profile_path,
    )
    mode = str(context_pipeline.get("mode") or "off").strip().lower()
    if mode not in _CONTEXT_PIPELINE_MODES:
        raise ProjectProfileError(
            f"project profile context_pipeline.mode 非法: {profile_path}: {mode!r}"
        )

    raw_capabilities = _mapping(
        raw.get("capabilities"),
        field_name="capabilities",
        profile_path=profile_path,
    )
    capabilities: dict[str, dict[str, Any]] = {}
    for capability_id, declaration in raw_capabilities.items():
        if not isinstance(capability_id, str) or not capability_id.strip():
            raise ProjectProfileError(
                f"capability id 非法: {profile_path}: {capability_id!r}"
            )
        if not isinstance(declaration, dict):
            raise ProjectProfileError(
                f"capability 声明必须是对象: {profile_path}: {capability_id}"
            )
        asset_id = declaration.get("asset")
        if asset_id is not None:
            asset_id = str(asset_id).strip()
            if asset_id not in assets:
                raise ProjectProfileError(
                    f"capability 引用了未声明 asset: {profile_path}: {capability_id} -> {asset_id}"
                )
            if (
                mode == "enforce"
                and declaration.get("required") is True
                and assets[asset_id].availability != "included"
            ):
                raise ProjectProfileError(
                    f"enforce capability 的必需 asset 未包含: {profile_path}: {capability_id} -> {asset_id}"
                )
        capabilities[capability_id] = declaration

    raw_views = _mapping(
        raw.get("module_context_views"),
        field_name="module_context_views",
        profile_path=profile_path,
    )
    module_context_views: dict[str, dict[str, Any]] = {}
    for module_id, view in raw_views.items():
        if (
            not isinstance(module_id, str)
            or not module_id.strip()
            or not isinstance(view, dict)
        ):
            raise ProjectProfileError(
                f"module_context_views 声明非法: {profile_path}: {module_id!r}"
            )
        refs = view.get("capabilities", [])
        if not isinstance(refs, list) or any(ref not in capabilities for ref in refs):
            raise ProjectProfileError(
                f"module_context_views 引用了未声明 capability: {profile_path}: {module_id}"
            )
        module_context_views[module_id] = view
    return mode, capabilities, module_context_views


def _safe_project_path(project_id: str) -> Path:
    p = Path(project_id)
    if p.is_absolute() or ".." in p.parts or not project_id.strip():
        raise ProjectProfileError(f"非法 project_id: {project_id!r}")
    return p


_LEGACY_PROJECT_ALIASES = {
    "wwm/en": "wwm/zh-en",
    "isekai/de": "isekai/en-de",
    "isekai/fr": "isekai/en-fr",
}


def _split_language_pair_codes(language_pair: str) -> tuple[str | None, str | None]:
    parts = [p for p in re.split(r"[-_>/→]+", language_pair.strip()) if p]
    if len(parts) < 2:
        return None, None
    return parts[0].lower(), parts[-1].lower()


def _normalize_lang_code(value: Any) -> str:
    return str(value or "").strip().lower()


class ProjectRegistry:
    def __init__(self, projects_dir: str | Path, default_project: str | None = None):
        self.projects_dir = Path(projects_dir)
        self.default_project = default_project
        self._cache: dict[str, ProjectProfile] = {}

    def get(self, project_id: str | None = None) -> ProjectProfile:
        effective_id = project_id or self.default_project
        if not effective_id:
            raise ProjectProfileError("未指定 project_id，且未配置 DEFAULT_PROJECT")

        canonical_id = self._resolve_project_id(effective_id)
        cached = self._cache.get(canonical_id)
        if cached is not None:
            return cached

        profile = self._load(canonical_id)
        self._cache[canonical_id] = profile
        return profile

    def _resolve_project_id(self, project_id: str) -> str:
        candidate = _safe_project_path(project_id).as_posix()
        if (self.projects_dir / candidate / "profile.json").exists():
            return candidate

        alias = _LEGACY_PROJECT_ALIASES.get(candidate)
        if (
            alias
            and (
                self.projects_dir / _safe_project_path(alias) / "profile.json"
            ).exists()
        ):
            return alias

        target_only = self._resolve_target_only_track(candidate)
        return target_only or candidate

    def _resolve_target_only_track(self, project_id: str) -> str | None:
        parts = Path(project_id).parts
        if len(parts) != 2:
            return None
        game, target = parts
        if "-" in target or "_" in target:
            return None

        game_dir = self.projects_dir / game
        if not game_dir.is_dir():
            return None

        matches: list[str] = []
        for profile_path in game_dir.glob("*/profile.json"):
            track = profile_path.parent.name
            if "-" not in track:
                continue
            _, track_target = _split_language_pair_codes(track)
            if track_target == target.lower():
                matches.append(f"{game}/{track}")

        if len(matches) > 1:
            raise ProjectProfileError(
                f"旧式 project_id={project_id!r} 只写目标语言，匹配到多个语言对: {', '.join(sorted(matches))}"
            )
        return matches[0] if matches else None

    def _load(self, project_id: str) -> ProjectProfile:
        profile_dir = self.projects_dir / _safe_project_path(project_id)
        profile_path = profile_dir / "profile.json"
        if not profile_path.exists():
            raise ProjectProfileError(f"project profile not found: {profile_path}")

        try:
            raw = json.loads(profile_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectProfileError(
                f"project profile 读取失败: {profile_path}: {exc}"
            ) from exc

        name = str(raw.get("name") or project_id).strip()
        if not name:
            raise ProjectProfileError(f"project profile 缺少 name: {profile_path}")

        source_from_pair, target_from_pair = _split_language_pair_codes(
            str(raw.get("language_pair") or "")
        )
        source_lang = _normalize_lang_code(raw.get("source_lang")) or (
            source_from_pair or ""
        )
        target_lang = _normalize_lang_code(raw.get("target_lang")) or (
            target_from_pair or ""
        )
        language_pair = str(raw.get("language_pair") or "").strip()
        if not language_pair and source_lang and target_lang:
            language_pair = f"{source_lang.upper()}-{target_lang.upper()}"

        allow_web_search = raw.get("allow_web_search", True)
        if not isinstance(allow_web_search, bool):
            raise ProjectProfileError(
                f"project profile allow_web_search 必须是布尔值: {profile_path}"
            )

        profile_contract_version = raw.get("profile_contract_version", 1)
        if (
            isinstance(profile_contract_version, bool)
            or not isinstance(profile_contract_version, int)
            or profile_contract_version not in {1, 2}
        ):
            raise ProjectProfileError(
                f"project profile profile_contract_version 仅支持 1 或 2: {profile_path}"
            )
        assets = _parse_project_assets(profile_dir, raw.get("assets"), profile_path)
        context_pipeline_mode, capabilities, module_context_views = (
            _parse_context_contract(
                raw,
                profile_contract_version=profile_contract_version,
                assets=assets,
                profile_path=profile_path,
            )
        )
        content_scope = _mapping(
            raw.get("content_scope"),
            field_name="content_scope",
            profile_path=profile_path,
        )
        workflow = _mapping(
            raw.get("workflow"),
            field_name="workflow",
            profile_path=profile_path,
        )
        raw_gaps = raw.get("gaps", [])
        if not isinstance(raw_gaps, list) or any(
            not isinstance(item, dict) for item in raw_gaps
        ):
            raise ProjectProfileError(
                f"project profile gaps 必须是对象数组: {profile_path}"
            )
        terminology_status = _mapping(
            raw.get("terminology_status"),
            field_name="terminology_status",
            profile_path=profile_path,
        )
        raw_authority_policy = raw.get("authority_policy", [])
        if not isinstance(raw_authority_policy, list) or any(
            not isinstance(item, str) or not item.strip()
            for item in raw_authority_policy
        ):
            raise ProjectProfileError(
                f"project profile authority_policy 必须是非空字符串数组: {profile_path}"
            )
        authority_policy = tuple(
            dict.fromkeys(item.strip() for item in raw_authority_policy)
        )
        reference_context = _mapping(
            raw.get("reference_context"),
            field_name="reference_context",
            profile_path=profile_path,
        )

        profile = ProjectProfile(
            name=name,
            language_pair=language_pair,
            source_lang=source_lang,
            target_lang=target_lang,
            game=str(raw.get("game") or "").strip(),
            profile_dir=profile_dir,
            profile_contract_version=profile_contract_version,
            profile_status=str(raw.get("profile_status") or "active").strip(),
            qdrant_collection=(
                str(raw["qdrant_collection"]).strip()
                if raw.get("qdrant_collection")
                else None
            ),
            reference_qdrant_collection=(
                str(raw["reference_qdrant_collection"]).strip()
                if raw.get("reference_qdrant_collection")
                else None
            ),
            reference_context=reference_context,
            web_search_prefix=(
                str(raw["web_search_prefix"]).strip()
                if raw.get("web_search_prefix")
                else None
            ),
            allow_web_search=allow_web_search,
            background=str(raw.get("background") or "").strip(),
            style_guide_path=_resolve_asset(profile_dir, raw.get("style_guide")),
            terminology_path=_resolve_asset(profile_dir, raw.get("terminology")),
            prompt_notes_path=_resolve_asset(profile_dir, raw.get("prompt_notes")),
            checks_path=_resolve_asset(profile_dir, raw.get("checks")),
            confirmed_rules_path=_resolve_asset(
                profile_dir, raw.get("confirmed_rules")
            ),
            vision_system_prompt=(
                str(raw["vision_system_prompt"]).strip()
                if raw.get("vision_system_prompt")
                else None
            ),
            distribution=str(raw.get("distribution") or "internal_only").strip(),
            authority_policy=authority_policy,
            terminology_status=terminology_status,
            assets=assets,
            context_pipeline_mode=context_pipeline_mode,
            capabilities=capabilities,
            module_context_views=module_context_views,
            content_scope=content_scope,
            workflow=workflow,
            gaps=tuple(raw_gaps),
        )
        logger.info("project profile loaded: %s", profile.name)
        return profile


class ProjectResourceManager:
    def __init__(self, registry: ProjectRegistry):
        self.registry = registry
        self._terminology: dict[str, TerminologyService] = {}
        self._style_guides: dict[str, StyleGuideService] = {}
        self._project_contexts: dict[str, Any] = {}

    def profile(self, project_id: str | None = None) -> ProjectProfile:
        return self.registry.get(project_id)

    def terminology(self, project_id: str | None = None) -> TerminologyService:
        profile = self.profile(project_id)
        cached = self._terminology.get(profile.name)
        if cached is not None:
            return cached

        svc = TerminologyService()
        if profile.terminology_path is not None:
            entries = _load_project_terminology(profile.terminology_path)
            svc.load(entries)
        self._terminology[profile.name] = svc
        return svc

    def replace_terminology(
        self, project_id: str | None, entries: list[dict[str, Any]]
    ) -> TerminologyService:
        profile = self.profile(project_id)
        svc = TerminologyService()
        svc.load(entries)
        path = _terminology_write_path(profile)
        _persist_project_terminology(
            path, [entry.model_dump() for entry in svc.entries]
        )
        if profile.terminology_path is None or path != profile.terminology_path:
            _set_profile_asset(profile.profile_dir, "terminology", path)
        self._terminology[profile.name] = svc
        return svc

    def style_guide(self, project_id: str | None = None) -> StyleGuideService:
        profile = self.profile(project_id)
        cached = self._style_guides.get(profile.name)
        if cached is not None:
            return cached

        svc = StyleGuideService()
        if profile.style_guide_path is not None:
            try:
                raw = profile.style_guide_path.read_bytes()
                text = parse_style_guide_bytes(raw, profile.style_guide_path.name)
                svc.load(text, filename=profile.style_guide_path.name)
            except (OSError, StyleGuideError) as exc:
                raise ProjectProfileError(
                    f"风格指南加载失败: {profile.style_guide_path}: {exc}"
                ) from exc
        self._style_guides[profile.name] = svc
        return svc

    def replace_style_guide(
        self, project_id: str | None, content: str, filename: str | None = None
    ) -> StyleGuideService:
        profile = self.profile(project_id)
        svc = StyleGuideService()
        svc.load(content, filename=filename)
        path = profile.style_guide_path or profile.profile_dir / "style_guide.md"
        _atomic_write_text(path, svc.get_rules())
        if profile.style_guide_path is None:
            _set_profile_asset(profile.profile_dir, "style_guide", path)
        self._style_guides[profile.name] = svc
        return svc

    def prompt_notes(self, project_id: str | None = None) -> str:
        profile = self.profile(project_id)
        if profile.prompt_notes_path is None or not profile.prompt_notes_path.exists():
            return ""
        return profile.prompt_notes_path.read_text(encoding="utf-8").strip()

    def project_context(self, project_id: str | None = None):
        from app.services.project_context_service import ProjectContextService

        profile = self.profile(project_id)
        cached = self._project_contexts.get(profile.name)
        if cached is not None:
            return cached
        service = ProjectContextService(profile)
        self._project_contexts[profile.name] = service
        return service


def _load_project_terminology(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectProfileError(f"术语 JSON 读取失败: {path}: {exc}") from exc
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
        raise ProjectProfileError(f"术语 JSON 必须是数组或包含 items 数组: {path}")

    try:
        return parse_terminology_file(path)
    except TerminologyError as exc:
        raise ProjectProfileError(f"术语表加载失败: {path}: {exc}") from exc


def _persist_project_terminology(path: Path, entries: list[dict[str, Any]]) -> None:
    suffix = path.suffix.lower()
    path.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".json":
        _atomic_write_text(
            path, json.dumps(entries, ensure_ascii=False, indent=2) + "\n"
        )
        return

    if suffix in {".csv", ".xlsx"}:
        try:
            import pandas as pd

            df = pd.DataFrame(entries)
            if suffix == ".csv":
                tmp = path.with_suffix(path.suffix + ".tmp")
                df.to_csv(tmp, index=False)
            else:
                tmp = path.with_name(path.name + ".tmp.xlsx")
                df.to_excel(tmp, index=False)
            tmp.replace(path)
            return
        except Exception as exc:
            raise ProjectProfileError(f"术语表写入失败: {path}: {exc}") from exc

    raise ProjectProfileError(f"不支持写入术语表格式: {suffix or '<无扩展名>'}")


def _terminology_write_path(profile: ProjectProfile) -> Path:
    path = profile.terminology_path
    if path is None:
        return profile.profile_dir / "terminology.json"
    if path.suffix.lower() in {".json", ".csv", ".xlsx"}:
        return path
    return profile.profile_dir / "terminology.json"


def _set_profile_asset(profile_dir: Path, key: str, path: Path) -> None:
    profile_path = profile_dir / "profile.json"
    try:
        raw = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectProfileError(f"profile 更新失败: {profile_path}: {exc}") from exc
    raw[key] = Path(
        os.path.relpath(path.resolve(), start=profile_dir.resolve())
    ).as_posix()
    _atomic_write_text(
        profile_path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


@lru_cache
def get_project_registry() -> ProjectRegistry:
    settings = get_settings()
    return ProjectRegistry(settings.projects_dir, settings.default_project)


@lru_cache
def get_project_resource_manager() -> ProjectResourceManager:
    return ProjectResourceManager(get_project_registry())


def reset_project_services() -> None:
    get_project_registry.cache_clear()
    get_project_resource_manager.cache_clear()


__all__ = [
    "ProjectAsset",
    "ProjectProfile",
    "ProjectProfileError",
    "ProjectRegistry",
    "ProjectResourceManager",
    "get_project_registry",
    "get_project_resource_manager",
    "reset_project_services",
]
