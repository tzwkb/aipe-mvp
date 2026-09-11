from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from app.services.project_service import ProjectAsset, ProjectProfile


class ProjectContextError(RuntimeError):
    pass


_SUPPORTED_RULE_CONDITION_KEYS = {
    "entity_ids",
    "speaker_entity_ids",
    "content_types",
    "scene_ids",
    "relationship_stages",
    "scene_tones",
    "contains_any",
}


class ProjectContextService:
    def __init__(self, profile: ProjectProfile):
        self.profile = profile
        self._loaded_asset_ids: set[str] = set()
        self._entities: list[dict[str, Any]] = []
        self._relations: list[dict[str, Any]] = []
        self._rules: list[dict[str, Any]] = []
        self._scenes: list[dict[str, Any]] = []
        self._entity_by_id: dict[str, dict[str, Any]] = {}
        self._authority_order = {
            issuer: index for index, issuer in enumerate(profile.authority_policy)
        }

    @property
    def mode(self) -> str:
        return self.profile.context_pipeline_mode

    def render(
        self,
        sources: list[str],
        *,
        content_type: str | None = None,
        speakers: list[str | None] | None = None,
        addressees: list[str | None] | None = None,
        dialog_id: str | None = None,
        scene_ids: list[str | None] | None = None,
        relationship_stages: list[str | None] | None = None,
        scene_tones: list[str | None] | None = None,
        context_notes: list[str | None] | None = None,
        module: str = "translation",
    ) -> str:
        if self.mode == "off":
            return ""
        self._ensure_loaded(module)
        if self.mode != "enforce":
            return ""

        source_values = [str(item or "").strip() for item in sources if str(item or "").strip()]
        speaker_values = [str(item or "").strip() for item in (speakers or []) if str(item or "").strip()]
        addressee_values = [str(item or "").strip() for item in (addressees or []) if str(item or "").strip()]
        scene_id_values = [str(item or "").strip() for item in (scene_ids or []) if str(item or "").strip()]
        relationship_stage_values = [
            str(item or "").strip() for item in (relationship_stages or []) if str(item or "").strip()
        ]
        scene_tone_values = [str(item or "").strip() for item in (scene_tones or []) if str(item or "").strip()]
        context_note_values = [
            str(item or "").strip() for item in (context_notes or []) if str(item or "").strip()
        ]
        all_values = [*source_values, *speaker_values, *addressee_values]
        direct_ids = self._match_entities(all_values)
        speaker_ids = self._match_entities(speaker_values)
        selected_ids = set(direct_ids)
        selected_scene = self._match_scene(source_values, dialog_id, scene_id_values)
        if selected_scene:
            selected_ids.update(selected_scene.get("participants") or [])

        effective_relationship_stages = list(relationship_stage_values)
        effective_scene_tones = list(scene_tone_values)
        if selected_scene:
            scene_stage = str(selected_scene.get("relationship_stage") or "").strip()
            scene_tone = str(selected_scene.get("scene_tone") or "").strip()
            if scene_stage:
                effective_relationship_stages.append(scene_stage)
            if scene_tone:
                effective_scene_tones.append(scene_tone)

        limits = self._limits_for(module)
        max_entities = _positive_int(limits.get("max_entities"), 12)
        max_facts = _positive_int(limits.get("max_facts_per_entity"), 3)
        max_relations = _positive_int(limits.get("max_relations"), 6)
        max_rules = _positive_int(limits.get("max_rules"), 6)
        max_chars = _positive_int(limits.get("max_chars"), 7000)

        ordered_ids = [
            entity["id"]
            for priority_ids in (direct_ids, selected_ids - direct_ids)
            for entity in self._entities
            if entity.get("id") in priority_ids
        ][:max_entities]
        selected_ids = set(ordered_ids)
        relations = self._select_relations(selected_ids, direct_ids, max_relations)
        related_ids = {
            endpoint
            for relation in relations
            for endpoint in (relation.get("from"), relation.get("to"))
            if isinstance(endpoint, str)
        }
        rule_entity_ids = selected_ids | related_ids
        rule_match_values = [
            *source_values,
            *speaker_values,
            *addressee_values,
            *scene_id_values,
            *effective_relationship_stages,
            *effective_scene_tones,
            *context_note_values,
        ]
        rules = self._select_rules(
            rule_match_values,
            rule_entity_ids,
            speaker_ids=speaker_ids,
            content_type=content_type,
            scene_id=(str(selected_scene.get("id")) if selected_scene else None),
            relationship_stages=effective_relationship_stages,
            scene_tones=effective_scene_tones,
            limit=max_rules,
        )

        explicit_context = {
            "addressee": _unique(addressee_values),
            "scene_id": _unique(scene_id_values),
            "relationship_stage": _unique(relationship_stage_values),
            "scene_tone": _unique(scene_tone_values),
            "context_note": _unique(context_note_values),
        }
        if (
            not ordered_ids
            and not relations
            and not selected_scene
            and not rules
            and not any(explicit_context.values())
        ):
            return ""

        lines = [
            "以下内容来自项目 profile 的可追溯上下文资产；source_backed/verified 可作为参考，candidate 仅用于提醒，不得覆盖原文或客户锁定术语。"
        ]
        if any(explicit_context.values()):
            lines.append("输入显式上下文：")
            for key, values in explicit_context.items():
                if values:
                    lines.append(f"- {key}：{' | '.join(values)}")
        if ordered_ids:
            lines.append("实体与事实：")
            for entity_id in ordered_ids:
                entity = self._entity_by_id[entity_id]
                label = _entity_label(entity)
                facts = sorted(
                    list(entity.get("facts") or []),
                    key=self._record_sort_key,
                )[:max_facts]
                if not facts:
                    lines.append(f"- {label}")
                    continue
                rendered = "；".join(
                    f"{fact.get('text', '')} [{_evidence_label(fact)}]"
                    for fact in facts
                    if str(fact.get("text") or "").strip()
                )
                lines.append(f"- {label}：{rendered}" if rendered else f"- {label}")

        if relations:
            lines.append("人物/组织关系：")
            for relation in relations:
                from_label = _entity_label(self._entity_by_id.get(str(relation.get("from")), {}))
                to_label = _entity_label(self._entity_by_id.get(str(relation.get("to")), {}))
                relation_type = str(relation.get("relation_type") or "related_to")
                attributes = relation.get("attributes") if isinstance(relation.get("attributes"), dict) else {}
                detail = str(attributes.get("evidence_text") or attributes.get("note") or "").strip()
                stage = str(attributes.get("relationship_stage") or "").strip()
                suffix = "；".join(part for part in (detail, f"阶段={stage}" if stage else "") if part)
                if suffix:
                    suffix = f"（{suffix}）"
                lines.append(
                    f"- {from_label} --{relation_type}--> {to_label}{suffix} [{_evidence_label(relation)}]"
                )

        if selected_scene:
            lines.append("场景上下文：")
            scene_bits = [
                str(selected_scene.get("title") or selected_scene.get("id") or "").strip(),
                f"关系阶段={selected_scene.get('relationship_stage')}"
                if selected_scene.get("relationship_stage")
                else "",
                f"语气={selected_scene.get('scene_tone')}" if selected_scene.get("scene_tone") else "",
            ]
            lines.append("- " + "；".join(bit for bit in scene_bits if bit))
            for fact in sorted(
                list(selected_scene.get("facts") or []),
                key=lambda item: self._record_sort_key(item) if isinstance(item, dict) else (0, 0),
            )[:max_facts]:
                if isinstance(fact, dict):
                    text = str(fact.get("text") or "").strip()
                    evidence = f" [{_evidence_label(fact)}]"
                else:
                    text = str(fact).strip()
                    evidence = ""
                if text:
                    lines.append(f"  - {text}{evidence}")

        if rules:
            lines.append("适用语境规则：")
            for rule in rules:
                status = str(rule.get("rule_status") or "draft")
                expectation = _format_expectation(rule.get("expect"))
                lines.append(
                    f"- {expectation} [status={status}; {_evidence_label(rule)}]"
                )

        rendered = "\n".join(lines)
        if len(rendered) > max_chars:
            rendered = rendered[: max_chars - 24].rstrip() + "\n[profile context truncated]"
        return rendered

    def _ensure_loaded(self, module: str) -> None:
        view = self.profile.module_context_views.get(module) or {}
        capability_ids = view.get("capabilities") if isinstance(view, dict) else None
        restrict_assets = isinstance(capability_ids, list)
        allowed_asset_ids = {
            str(self.profile.capabilities[capability_id]["asset"])
            for capability_id in (capability_ids or [])
            if capability_id in self.profile.capabilities
            and self.profile.capabilities[capability_id].get("asset")
        }
        for asset_id, asset in self.profile.assets.items():
            if restrict_assets and asset_id not in allowed_asset_ids:
                continue
            if asset_id in self._loaded_asset_ids:
                continue
            if asset.availability != "included" or asset.path is None:
                continue
            if asset.kind == "entity_registry":
                document = self._read_json(asset)
                self._load_entities(document, asset.path)
            elif asset.kind == "context_rules":
                document = self._read_json(asset)
                self._load_rules(document, asset.path)
            elif asset.kind == "scene_registry":
                document = self._read_json(asset)
                self._load_scenes(document, asset.path)
            self._loaded_asset_ids.add(asset_id)

    @staticmethod
    def _read_json(asset: ProjectAsset) -> dict[str, Any]:
        assert asset.path is not None
        try:
            value = json.loads(asset.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProjectContextError(f"project context asset 读取失败: {asset.path}: {exc}") from exc
        if not isinstance(value, dict):
            raise ProjectContextError(f"project context asset 顶层必须是对象: {asset.path}")
        return value

    def _load_entities(self, document: dict[str, Any], path: Path) -> None:
        if document.get("schema") not in {"aipe.entities", "lqe.entities"} or document.get("version") != 1:
            raise ProjectContextError(f"entity registry schema/version 不支持: {path}")
        entities = document.get("entities")
        relations = document.get("relations")
        if not isinstance(entities, list) or not isinstance(relations, list):
            raise ProjectContextError(f"entity registry 缺少 entities/relations 数组: {path}")
        seen: set[str] = set()
        for entity in entities:
            if not isinstance(entity, dict) or not str(entity.get("id") or "").strip():
                raise ProjectContextError(f"entity registry 含非法 entity: {path}")
            entity_id = str(entity["id"])
            if entity_id in seen:
                raise ProjectContextError(f"entity id 重复: {path}: {entity_id}")
            seen.add(entity_id)
        for relation in relations:
            if not isinstance(relation, dict):
                raise ProjectContextError(f"entity registry 含非法 relation: {path}")
            if relation.get("from") not in seen or relation.get("to") not in seen:
                raise ProjectContextError(
                    f"relation 引用了未知 entity: {path}: {relation.get('id')}"
                )
        self._entities.extend(entities)
        self._relations.extend(relations)
        self._entity_by_id.update({str(entity["id"]): entity for entity in entities})

    def _load_rules(self, document: dict[str, Any], path: Path) -> None:
        if document.get("schema") not in {"aipe.context-rules", "lqe.context-rules"} or document.get("version") != 1:
            raise ProjectContextError(f"context rules schema/version 不支持: {path}")
        rules = document.get("rules")
        if not isinstance(rules, list) or any(not isinstance(rule, dict) for rule in rules):
            raise ProjectContextError(f"context rules 缺少 rules 对象数组: {path}")
        for rule in rules:
            when = rule.get("when", {})
            if not isinstance(when, dict):
                raise ProjectContextError(
                    f"context rule when 必须是对象: {path}: {rule.get('id')}"
                )
            unsupported = sorted(set(when) - _SUPPORTED_RULE_CONDITION_KEYS)
            if unsupported:
                raise ProjectContextError(
                    f"context rule 含未知 when 条件: {path}: {rule.get('id')}: "
                    + ", ".join(unsupported)
                )
        self._rules.extend(rules)

    def _load_scenes(self, document: dict[str, Any], path: Path) -> None:
        if document.get("schema") != "aipe.scenes" or document.get("version") != 1:
            raise ProjectContextError(f"scene registry schema/version 不支持: {path}")
        scenes = document.get("scenes")
        if not isinstance(scenes, list) or any(not isinstance(scene, dict) for scene in scenes):
            raise ProjectContextError(f"scene registry 缺少 scenes 对象数组: {path}")
        self._scenes.extend(scenes)

    def _limits_for(self, module: str) -> dict[str, Any]:
        view = self.profile.module_context_views.get(module) or {}
        limits = view.get("limits")
        return limits if isinstance(limits, dict) else {}

    def _match_entities(self, values: list[str]) -> set[str]:
        matched: set[str] = set()
        for entity in self._entities:
            names = entity.get("names") if isinstance(entity.get("names"), dict) else {}
            aliases = [*(names.get("source") or []), *(names.get("target") or [])]
            if any(_contains_alias(str(alias), value) for alias in aliases for value in values):
                matched.add(str(entity["id"]))
        return matched

    def _match_scene(
        self,
        sources: list[str],
        dialog_id: str | None,
        scene_ids: list[str],
    ) -> dict[str, Any] | None:
        requested_ids = {str(dialog_id or "").strip(), *scene_ids}
        requested_ids.discard("")
        if requested_ids:
            for scene in self._scenes:
                aliases = {str(scene.get("id") or ""), *[str(x) for x in scene.get("aliases") or []]}
                if requested_ids.intersection(aliases):
                    return scene
        for scene in self._scenes:
            triggers = [str(item) for item in scene.get("triggers") or [] if str(item)]
            minimum = _positive_int(scene.get("trigger_min_matches"), 1)
            hits = sum(1 for trigger in triggers if any(trigger in source for source in sources))
            if triggers and hits >= minimum:
                return scene
        return None

    def _select_relations(
        self,
        selected_ids: set[str],
        direct_ids: set[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        if not selected_ids or limit <= 0:
            return []
        candidates = [
            relation
            for relation in self._relations
            if relation.get("from") in selected_ids or relation.get("to") in selected_ids
        ]
        candidates.sort(
            key=lambda relation: (
                0
                if relation.get("from") in direct_ids and relation.get("to") in direct_ids
                else 1
                if relation.get("from") in direct_ids or relation.get("to") in direct_ids
                else 2,
                *self._record_sort_key(relation),
                str(relation.get("id") or ""),
            )
        )
        return candidates[:limit]

    def _select_rules(
        self,
        match_values: list[str],
        entity_ids: set[str],
        *,
        speaker_ids: set[str],
        content_type: str | None,
        scene_id: str | None,
        relationship_stages: list[str],
        scene_tones: list[str],
        limit: int,
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for rule in sorted(
            self._rules,
            key=lambda item: (
                *self._record_sort_key(item),
                -int(item.get("priority") or 0),
                str(item.get("id") or ""),
            ),
        ):
            if str(rule.get("rule_status") or "draft") in {"deprecated", "rejected"}:
                continue
            target_lang = str(rule.get("target_lang") or "").strip().lower()
            if target_lang and target_lang != self.profile.target_lang.lower():
                continue
            when = rule.get("when") if isinstance(rule.get("when"), dict) else {}
            if not _condition_matches(
                when,
                sources=match_values,
                entity_ids=entity_ids,
                speaker_ids=speaker_ids,
                content_type=content_type,
                scene_id=scene_id,
                relationship_stages=relationship_stages,
                scene_tones=scene_tones,
            ):
                continue
            selected.append(rule)
            if len(selected) >= limit:
                break
        return selected

    def _record_sort_key(self, record: dict[str, Any]) -> tuple[int, int]:
        authority = record.get("authority") if isinstance(record.get("authority"), dict) else {}
        issuer = str(authority.get("issuer") or "").strip()
        authority_rank = self._authority_order.get(issuer, len(self._authority_order) + 1)
        status = str(
            record.get("verification_status")
            or record.get("rule_status")
            or "unrated"
        ).lower()
        status_rank = {
            "verified": 0,
            "confirmed": 0,
            "source_backed": 1,
            "accepted": 1,
            "draft": 2,
            "candidate": 3,
            "unverified": 4,
            "unrated": 5,
            "deprecated": 9,
            "rejected": 9,
        }.get(status, 5)
        return authority_rank, status_rank


def _condition_matches(
    when: dict[str, Any],
    *,
    sources: list[str],
    entity_ids: set[str],
    speaker_ids: set[str],
    content_type: str | None,
    scene_id: str | None,
    relationship_stages: list[str],
    scene_tones: list[str],
) -> bool:
    required_entities = {str(item) for item in when.get("entity_ids") or []}
    if required_entities and not required_entities.intersection(entity_ids):
        return False
    required_speakers = {str(item) for item in when.get("speaker_entity_ids") or []}
    if required_speakers and not required_speakers.intersection(speaker_ids):
        return False
    content_types = {str(item) for item in when.get("content_types") or []}
    if content_types and str(content_type or "") not in content_types:
        return False
    scene_ids = {str(item) for item in when.get("scene_ids") or []}
    if scene_ids and str(scene_id or "") not in scene_ids:
        return False
    required_stages = {str(item) for item in when.get("relationship_stages") or []}
    if required_stages and not required_stages.intersection(relationship_stages):
        return False
    required_tones = {str(item) for item in when.get("scene_tones") or []}
    if required_tones and not required_tones.intersection(scene_tones):
        return False
    contains_any = [str(item) for item in when.get("contains_any") or [] if str(item)]
    if contains_any and not any(token in source for token in contains_any for source in sources):
        return False
    return True


def _contains_alias(alias: str, value: str) -> bool:
    alias = alias.strip()
    value = value.strip()
    if not alias or not value:
        return False
    if re.search(r"[\u3400-\u9fff]", alias):
        return alias in value
    return re.search(rf"(?<![\w]){re.escape(alias)}(?![\w])", value, flags=re.IGNORECASE) is not None


def _entity_label(entity: dict[str, Any]) -> str:
    names = entity.get("names") if isinstance(entity.get("names"), dict) else {}
    targets = [str(item) for item in names.get("target") or [] if str(item)]
    sources = [str(item) for item in names.get("source") or [] if str(item)]
    label = " / ".join([*targets[:2], *sources[:1]])
    return label or str(entity.get("id") or "unknown-entity")


def _evidence_label(record: dict[str, Any]) -> str:
    status = str(record.get("verification_status") or record.get("rule_status") or "unrated")
    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    source_id = str(provenance.get("source_id") or provenance.get("source") or "unknown-source")
    coordinate = str(provenance.get("coordinate") or "").strip()
    return f"{status}; {source_id}{'@' + coordinate if coordinate else ''}"


def _format_expectation(value: Any) -> str:
    if not isinstance(value, dict):
        return str(value or "").strip()
    parts: list[str] = []
    for key, item in value.items():
        if isinstance(item, list):
            rendered = ", ".join(str(value) for value in item)
        elif isinstance(item, dict):
            rendered = ", ".join(f"{subkey}={subvalue}" for subkey, subvalue in item.items())
        else:
            rendered = str(item)
        parts.append(f"{key}={rendered}")
    return "；".join(parts)


def _positive_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


__all__ = ["ProjectContextError", "ProjectContextService"]
