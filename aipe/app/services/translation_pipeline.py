"""翻译 Pipeline 编排：术语命中 → RAG → Prompt → LLM → 输出。

术语命中结果与 RAG 检索结果一同作为 Prompt 的参考段，由 LLM 结合语境决定是否采用，
避免占位符强替换带来的"硬翻译"。

提供两条调用路径：
- ``translate_single``：单句无状态端到端翻译（默认路径）
- ``translate_group``：整组并排翻译（结构高度相似的句子用同一目标语句式，保证一致性）
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass

from app.errors import TranslationError
from app.schemas.rag import RAGSearchResult
from app.schemas.terminology import TermEntry
from app.schemas.translate import TranslationResult
from app.schemas.web_search import WebSearchResult
from app.services.llm_service import LLMService
from app.services.project_service import ProjectProfile, ProjectResourceManager
from app.services.rag_service import RAGDiagnostics, RAGService
from app.services.style_guide_service import ContentType, StyleGuideService
from app.services.terminology_service import TerminologyService
from app.services.vision_service import VisionService
from app.services.web_search_service import WebSearchService
from app.utils.cluster import common_template_repr

logger = logging.getLogger(__name__)


_BASE_SYSTEM = (
    "你是一名专业的游戏本地化译者，负责按项目语言方向翻译游戏文本。\n"
    "请遵守以下要求：\n\n"
    "## 翻译要求\n"
    "1. 术语参考段按文本功能使用：UI、任务、道具、技能、系统规则等功能性文本优先保持术语一致；剧情、口语、邮件、外观描述等叙事性文本应结合语境取舍，不要把术语表当作逐字替换表\n"
    "2. RAG 参考例句反映游戏内同类语境的真实译法，请结合相似度分数与上下文综合参考\n"
    "3. 遵守项目背景、风格指南和文本功能要求，保持目标语自然准确\n"
    "4. 按照下方输出格式要求输出，不要输出格式外的额外内容\n"
    "5. 外部网络参考段（如有）来自公开网络搜索，优先级最低；只用来理解专有名词的背景，与术语 / RAG 冲突时一律以术语 / RAG 为准\n\n"
    "## 富文本与变量标记规则\n"
    "待翻译文本中可能出现以下特殊标记，请按规则处理，确保输出中标记位置正确：\n\n"
    "1. **字体格式标签**（非打印字符）：`#G......#E`、`#C......#E`、`#Y......#E` 等用于定义二者之间的文字格式（如颜色）。\n"
    "   译文需要保留这些标签，并将对应的译文放在标签之间（即 `#G` 与 `#E` 之间）。\n\n"
    "2. **通用文本变量**（打印字符）：`{}` 为空占位符，翻译时将其视为实际文本内容的一部分处理，确保位置正确。\n\n"
    "3. **具名文本变量**（打印字符）：如 `{slot1_qishu_name}`、`{kungfu_main_name}` 等。\n"
    "   这些是真实文本的占位符，译文需考虑其实际含义，将其放在语法和语义都恰当的位置。\n\n"
    "4. **数值标量**（打印字符）：如 `{standard_value}`、`{qishu_standard_value}` 等。\n"
    "   这些是实际数值的占位符，译文需根据语境判断其在句子中的合理位置并正确放置。\n"
    "5. <x id=\"64\"/> 这样的没有实际意义的原文标签应该完全保留\n"
)


# 整组翻译输出解析正则：兼容 `1.` `1)` `1：` `1、` 等多种序号写法
_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[.\)）。、:：]\s*(.+?)\s*$")


_WEB_SECTION_HEADER = (
    "## 外部网络参考（可能不准 — 仅作背景理解，不要直接照抄措辞）\n"
    "以下内容来自公开网络搜索，可能含错误、口语化译法或与本游戏无关的同名条目。\n"
    "请把它当作\"了解专有名词的语境\"用，绝不能凌驾于术语参考和 RAG 参考之上。"
)


_LANGUAGE_NAMES = {
    "ZH": "Chinese",
    "ZHCN": "Chinese",
    "ZH-CN": "Chinese",
    "CN": "Chinese",
    "EN": "English",
    "ENUS": "English",
    "EN-US": "English",
    "DE": "German",
    "DEDE": "German",
    "DE-DE": "German",
    "FR": "French",
    "FRFR": "French",
    "FR-FR": "French",
    "JA": "Japanese",
    "JP": "Japanese",
    "RU": "Russian",
    "PT": "Portuguese",
    "ES": "Spanish",
}


_FUNCTIONAL_TERM_TYPES = {
    ContentType.UI,
    ContentType.QUEST,
    ContentType.QUEST_OBJECTIVE,
    ContentType.QUEST_DESCRIPTION,
    ContentType.ACHIEVEMENT,
    ContentType.SKILL,
    ContentType.HINT,
    ContentType.NOTIFICATION,
    ContentType.ITEM,
    ContentType.ENTITY,
    ContentType.PRESCRIPTION,
}

_CONTEXTUAL_TERM_TYPES = {
    ContentType.STORY,
    ContentType.SPEECH,
    ContentType.MARTIAL_RECORDS,
    ContentType.APPEARANCE_DESCRIPTION,
    ContentType.BESTIARY,
}

_TERM_SECTION_TITLE = (
    "## 术语参考（按文本功能使用；功能/UI/道具/任务类保持一致，口语/剧情/邮件类按语境取舍）"
)


@dataclass(frozen=True)
class _ProjectContext:
    profile: ProjectProfile | None
    terminology_svc: TerminologyService
    style_guide_svc: StyleGuideService
    rag_collection: str | None
    base_system: str
    web_search_prefix: str | None
    vision_system_prompt: str | None


class TranslationPipeline:
    """翻译流水线。``translate_single`` / ``translate_group`` 都是无状态端到端调用。"""

    def __init__(
        self,
        terminology_svc: TerminologyService,
        rag_svc: RAGService,
        style_guide_svc: StyleGuideService,
        llm_svc: LLMService,
        web_search_svc: WebSearchService | None = None,
        web_search_dense_threshold: float = 0.6,
        vision_svc: VisionService | None = None,
        project_resources: ProjectResourceManager | None = None,
    ):
        self.terminology_svc = terminology_svc
        self.rag_svc = rag_svc
        self.style_guide_svc = style_guide_svc
        self.llm_svc = llm_svc
        # 可选注入：未传或 disabled 时所有 Web 搜索逻辑短路跳过
        self.web_search_svc = web_search_svc
        self.default_web_search_dense_threshold = web_search_dense_threshold
        self.vision_svc = vision_svc
        self.project_resources = project_resources

    # ---------- 单句路径 ----------

    async def translate_single(
        self,
        text: str,
        *,
        enable_rag: bool = True,
        rag_threshold: float | None = None,
        rag_top_k: int | None = None,
        content_type: ContentType | None = None,
        rag_collection: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> TranslationResult:
        """端到端翻译一句。失败时返回 status=error，不抛异常（保证批量不被单句拖垮）。

        ``content_type`` 传入非空时直接复用，跳过 LLM 预分类。

        ``enable_web_search=True`` 且术语 0 命中 + RAG 弱召回时，调用 Web 搜索补充背景。
        """
        source = (text or "").strip()
        if not source:
            return TranslationResult(
                source=text or "",
                translation="",
                status="error",
                error_msg="empty input",
            )

        project_ctx = self._resolve_project(project_id, rag_collection)
        effective_collection = rag_collection or project_ctx.rag_collection

        if use_tm_exact_match:
            direct = await self._maybe_tm_exact_result(
                source,
                collection=effective_collection,
                content_type=content_type,
            )
            if direct is not None:
                return direct

        # Step 0: 文本类型分类（传入时跳过，否则 LLM 预分类）
        if content_type is None:
            content_type = await self._classify_text(source, style_guide_svc=project_ctx.style_guide_svc)
            logger.debug("文本分类: %r… → %s", source[:30], content_type.value)
        else:
            logger.debug("文本分类(复用传入): %r… → %s", source[:30], content_type.value)

        # Step 1: 术语命中匹配（仅作为 Prompt 参考段，不修改原文）
        term_matches = project_ctx.terminology_svc.find_matches(source)

        # Step 2: RAG 检索（异常降级为空参考）
        references: list[RAGSearchResult] = []
        diag: RAGDiagnostics = RAGDiagnostics(dense_top1=None, sparse_hits=0)
        if enable_rag:
            try:
                if self._web_search_active(enable_web_search):
                    references, diag = await self.rag_svc.search_with_diagnostics(
                        source,
                        threshold=rag_threshold,
                        top_k=rag_top_k,
                        collection=effective_collection,
                    )
                else:
                    references = await self.rag_svc.search(
                        source,
                        threshold=rag_threshold,
                        top_k=rag_top_k,
                        collection=effective_collection,
                    )
            except Exception as exc:
                logger.warning("RAG 检索失败，跳过参考: %s", exc)
                references = []
                diag = RAGDiagnostics(dense_top1=None, sparse_hits=0)

        # Step 2.5: Web 搜索兜底（仅在术语 0 命中 + RAG 弱召回 + 用户启用时触发）
        web_refs, web_triggered = await self._maybe_web_search(
            source,
            term_matches=term_matches,
            diag=diag,
            enable_web_search=enable_web_search,
            user_threshold=web_search_dense_threshold,
            enable_vision=enable_vision,
            web_search_prefix=project_ctx.web_search_prefix,
            vision_system_prompt=project_ctx.vision_system_prompt,
        )

        # Step 3: Prompt 组装
        messages = self._build_messages(
            source,
            term_matches,
            references,
            content_type,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
        )

        image_analysis = _extract_image_analysis(web_refs)

        # Step 4: LLM 翻译
        try:
            raw = await self.llm_svc.translate(messages)
            translation, translation_reason = _parse_single_llm_output(raw)
        except TranslationError as exc:
            return TranslationResult(
                source=source,
                translation=f"[ERROR: AI_FAIL] {source}",
                status="error",
                content_type=content_type.value,
                terminology_used=self._term_used_view(term_matches),
                rag_references=self._refs_view(references),
                web_references=self._web_view(web_refs),
                web_search_triggered=web_triggered if enable_web_search else None,
                image_analysis=image_analysis,
                error_msg=str(exc),
            )

        # Step 5: 输出
        return TranslationResult(
            source=source,
            translation=translation,
            translation_reason=translation_reason,
            status="success",
            content_type=content_type.value,
            terminology_used=self._term_used_view(term_matches),
            rag_references=self._refs_view(references),
            web_references=self._web_view(web_refs),
            web_search_triggered=web_triggered if enable_web_search else None,
            image_analysis=image_analysis,
        )

    # ---------- 整组路径 ----------

    async def translate_group(
        self,
        sources: list[str],
        *,
        enable_rag: bool = True,
        rag_threshold: float | None = None,
        rag_top_k: int | None = None,
        content_type: ContentType | None = None,
        rag_collection: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        """整组并排翻译。所有句子共享结构模板，要求 LLM 用同一目标语句式。

        输入 N 句、输出 N 句，顺序与输入对齐。失败（LLM 报错 / 解析失败）时
        自动 fallback 为逐句 ``translate_single``，确保对外语义与单句路径一致。

        ``content_type`` 传入非空时直接复用，跳过 LLM 预分类。

        ``enable_web_search=True`` 时只对**代表句**（template_repr 或首句）发一次 Web
        搜索，结果整组共享。
        """
        if not sources:
            return []
        if len(sources) == 1:
            return [
                await self.translate_single(
                    sources[0],
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    content_type=content_type,
                    rag_collection=rag_collection,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=use_tm_exact_match,
                )
            ]

        cleaned = [(s or "").strip() for s in sources]
        project_ctx = self._resolve_project(project_id, rag_collection)
        effective_collection = rag_collection or project_ctx.rag_collection
        if not all(cleaned):
            # 出现空串时回退单句（translate_single 会对空串返回 error 结果）
            return await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                content_type=content_type,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                use_tm_exact_match=use_tm_exact_match,
            )

        if use_tm_exact_match:
            direct_results = await self._collect_tm_exact_results(
                cleaned,
                collection=effective_collection,
                content_type=content_type,
            )
            if any(r is not None for r in direct_results):
                if all(r is not None for r in direct_results):
                    return [r for r in direct_results if r is not None]
                unmatched_indices = [i for i, r in enumerate(direct_results) if r is None]
                translated_unmatched = await self.translate_group(
                    [cleaned[i] for i in unmatched_indices],
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    content_type=content_type,
                    rag_collection=rag_collection,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=False,
                )
                return _merge_direct_results(direct_results, translated_unmatched)

        # Step 0: 分类（传入时跳过，否则对代表句预分类）
        if content_type is None:
            content_type = await self._classify_text(cleaned[0], style_guide_svc=project_ctx.style_guide_svc)
            logger.debug(
                "整组分类: size=%d 代表句=%r → %s",
                len(cleaned),
                cleaned[0][:30],
                content_type.value,
            )
        else:
            logger.debug(
                "整组分类(复用传入): size=%d 代表句=%r → %s",
                len(cleaned),
                cleaned[0][:30],
                content_type.value,
            )

        # Step 1: 术语命中（对每句独立扫描后并集去重，保持出现顺序）
        term_matches = self._collect_terms(cleaned, terminology_svc=project_ctx.terminology_svc)

        # Step 2: RAG（每句并行检索，按 score 合并去重）
        references: list[RAGSearchResult] = []
        if enable_rag:
            references = await self._collect_references(
                cleaned, threshold=rag_threshold, top_k=rag_top_k, collection=effective_collection
            )

        # Step 2.5: 整组 Web 搜索兜底（只用代表句触发，整组共享一次结果）
        template_hint = common_template_repr(cleaned)
        web_refs, web_triggered = await self._group_web_search(
            cleaned,
            template_hint=template_hint,
            term_matches=term_matches,
            enable_rag=enable_rag,
            rag_threshold=rag_threshold,
            rag_collection=effective_collection,
            enable_web_search=enable_web_search,
            user_threshold=web_search_dense_threshold,
            enable_vision=enable_vision,
            web_search_prefix=project_ctx.web_search_prefix,
            vision_system_prompt=project_ctx.vision_system_prompt,
        )

        # Step 3: 构建整组 prompt
        messages = self._build_group_messages(
            cleaned,
            term_matches,
            references,
            content_type,
            template_hint,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
        )

        # Step 4: LLM 调用
        try:
            raw = await self.llm_svc.translate(messages)
        except TranslationError as exc:
            logger.warning("整组翻译 LLM 失败，回退单句: %s", exc)
            return await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
            )

        # Step 5: 解析编号输出
        parsed = _parse_numbered_output(raw, expected=len(cleaned))
        if parsed is None:
            logger.warning(
                "整组翻译输出解析失败（期望 %d 行），回退单句。原始输出前 200 字: %r",
                len(cleaned),
                raw[:200],
            )
            return await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
            )

        term_view = self._term_used_view(term_matches)
        refs_view = self._refs_view(references)
        web_view = self._web_view(web_refs)
        web_triggered_view = web_triggered if enable_web_search else None
        image_analysis = _extract_image_analysis(web_refs)
        return [
            TranslationResult(
                source=src,
                translation=tr,
                translation_reason=reason,
                status="success",
                content_type=content_type.value,
                terminology_used=term_view,
                rag_references=refs_view,
                web_references=web_view,
                web_search_triggered=web_triggered_view,
                image_analysis=image_analysis,
            )
            for src, (tr, reason) in zip(cleaned, parsed)
        ]

    # ---------- 对话路径 ----------

    async def translate_dialog(
        self,
        sources: list[str],
        speakers: list[str | None],
        *,
        dialog_id: str | None = None,
        times: list[float | None] | None = None,
        enable_rag: bool = True,
        rag_threshold: float | None = None,
        rag_top_k: int | None = None,
        content_type: ContentType | None = None,
        rag_collection: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        """整段对话翻译。``sources`` 已按对话发生顺序排列，``speakers`` 与之等长。

        失败策略与 ``translate_group`` 不同：LLM 报错或输出对齐解析失败时，整段
        对话所有句子统一返回 ``status=error``，等待人工检查，**不回退单句**——因为
        丢失上下文的单句译文质量反而更差。

        ``len(sources) == 1`` 时直接走 ``translate_single`` 路径。
        """
        if not sources:
            return []
        if len(speakers) != len(sources):
            raise ValueError(
                f"speakers 长度({len(speakers)}) 与 sources 长度({len(sources)}) 不一致"
            )
        if times is not None and len(times) != len(sources):
            raise ValueError(
                f"times 长度({len(times)}) 与 sources 长度({len(sources)}) 不一致"
            )

        if len(sources) == 1:
            return [
                await self.translate_single(
                    sources[0],
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    content_type=content_type,
                    rag_collection=rag_collection,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=use_tm_exact_match,
                )
            ]

        cleaned = [(s or "").strip() for s in sources]
        project_ctx = self._resolve_project(project_id, rag_collection)
        effective_collection = rag_collection or project_ctx.rag_collection
        if not all(cleaned):
            # 出现空句时退化为单句路径，translate_single 会对空串返回 error
            return await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                content_type=content_type,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                use_tm_exact_match=use_tm_exact_match,
            )

        if use_tm_exact_match:
            direct_results = await self._collect_tm_exact_results(
                cleaned,
                collection=effective_collection,
                content_type=content_type,
            )
            if any(r is not None for r in direct_results):
                if all(r is not None for r in direct_results):
                    return [r for r in direct_results if r is not None]
                unmatched_indices = [i for i, r in enumerate(direct_results) if r is None]
                translated_unmatched = await self.translate_dialog(
                    [cleaned[i] for i in unmatched_indices],
                    [speakers[i] for i in unmatched_indices],
                    dialog_id=dialog_id,
                    times=[times[i] for i in unmatched_indices] if times is not None else None,
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    content_type=content_type,
                    rag_collection=rag_collection,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=False,
                )
                return _merge_direct_results(direct_results, translated_unmatched)

        # Step 0: 分类（传入时跳过；否则对代表句预分类）
        if content_type is None:
            content_type = await self._classify_text(cleaned[0], style_guide_svc=project_ctx.style_guide_svc)

        # Step 1: 术语命中并集
        term_matches = self._collect_terms(cleaned, terminology_svc=project_ctx.terminology_svc)

        # Step 2: RAG
        references: list[RAGSearchResult] = []
        if enable_rag:
            references = await self._collect_references(
                cleaned, threshold=rag_threshold, top_k=rag_top_k, collection=effective_collection
            )

        # Step 2.5: 对话 Web 搜索兜底（用首句作代表句，整段共享）
        web_refs, web_triggered = await self._group_web_search(
            cleaned,
            template_hint="",
            term_matches=term_matches,
            enable_rag=enable_rag,
            rag_threshold=rag_threshold,
            rag_collection=effective_collection,
            enable_web_search=enable_web_search,
            user_threshold=web_search_dense_threshold,
            enable_vision=enable_vision,
            web_search_prefix=project_ctx.web_search_prefix,
            vision_system_prompt=project_ctx.vision_system_prompt,
        )

        # Step 3: 对话 prompt（times 仅用于排序，不进 prompt）
        messages = self._build_dialog_messages(
            cleaned, speakers, term_matches, references, content_type, dialog_id,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
        )

        term_view = self._term_used_view(term_matches)
        refs_view = self._refs_view(references)
        web_view = self._web_view(web_refs)
        web_triggered_view = web_triggered if enable_web_search else None
        image_analysis = _extract_image_analysis(web_refs)

        # Step 4: LLM
        try:
            raw = await self.llm_svc.translate(messages)
        except TranslationError as exc:
            logger.warning(
                "对话翻译 LLM 失败 dialog_id=%s n=%d: %s", dialog_id, len(cleaned), exc
            )
            return self._dialog_error_results(
                cleaned,
                content_type,
                term_view,
                refs_view,
                reason="DIALOG_LLM_FAIL",
                error_msg=str(exc),
                web_view=web_view,
                web_triggered=web_triggered_view,
                image_analysis=image_analysis,
            )

        # Step 5: 解析编号输出
        # 对话路径不强求编号为 {1..N}：LLM 可能保留 time 风格的数字（如游戏内部的
        # 时间码）。只要求 N 行编号、按顺序映射到 N 个原文位置即可。
        parsed = _parse_numbered_output_positional(raw, expected=len(cleaned))
        if parsed is None:
            logger.warning(
                "对话翻译输出解析失败 dialog_id=%s 期望 %d 行，原始输出前 200 字: %r",
                dialog_id,
                len(cleaned),
                raw[:200],
            )
            return self._dialog_error_results(
                cleaned,
                content_type,
                term_view,
                refs_view,
                reason="DIALOG_PARSE_FAIL",
                error_msg=f"对话整段输出解析失败（期望 {len(cleaned)} 行）",
                web_view=web_view,
                web_triggered=web_triggered_view,
                image_analysis=image_analysis,
            )

        return [
            TranslationResult(
                source=src,
                translation=tr,
                translation_reason=reason,
                status="success",
                content_type=content_type.value,
                terminology_used=term_view,
                rag_references=refs_view,
                web_references=web_view,
                web_search_triggered=web_triggered_view,
                image_analysis=image_analysis,
            )
            for src, (tr, reason) in zip(cleaned, parsed)
        ]

    @staticmethod
    def _dialog_error_results(
        cleaned: list[str],
        content_type: ContentType,
        term_view: list[dict],
        refs_view: list[dict] | None,
        *,
        reason: str,
        error_msg: str,
        web_view: list[dict] | None = None,
        web_triggered: bool | None = None,
        image_analysis: str | None = None,
    ) -> list[TranslationResult]:
        return [
            TranslationResult(
                source=src,
                translation=f"[ERROR: {reason}] {src}",
                status="error",
                content_type=content_type.value,
                terminology_used=term_view,
                rag_references=refs_view,
                web_references=web_view,
                web_search_triggered=web_triggered,
                image_analysis=image_analysis,
                error_msg=error_msg,
            )
            for src in cleaned
        ]

    def _build_dialog_messages(
        self,
        sources: list[str],
        speakers: list[str | None],
        term_matches: list[TermEntry],
        references: list[RAGSearchResult],
        content_type: ContentType,
        dialog_id: str | None,
        *,
        web_refs: list[WebSearchResult] | None = None,
        style_guide_svc: StyleGuideService | None = None,
        base_system: str | None = None,
    ) -> list[dict[str, str]]:
        sg = style_guide_svc or self.style_guide_svc
        system = sg.build_system_prompt_for_type(base_system or _BASE_SYSTEM, content_type)

        header_bits = [f"以下是一段游戏内连续对话，共 {len(sources)} 句，按对话发生顺序列出。"]
        if dialog_id:
            header_bits.append(f"对话编号：{dialog_id}。")
        header_bits.extend(
            [
                "请结合上下文（说话人身份、上下句衔接、语气、信息流向）翻译每一句，",
                "确保整段译文在目标语中读起来像一段连贯、自然的对话。",
                "说话人姓名仅供你理解语境使用，**不要**在译文里输出说话人名字，",
                "也无需输出说话人姓名。",
            ]
        )

        parts: list[str] = ["## 对话翻译任务", "\n".join(header_bits)]

        if term_matches:
            parts.append(_TERM_SECTION_TITLE)
            parts.append(_format_term_section(term_matches))

        if references:
            parts.append(
                "## 参考翻译（从游戏语料库 RAG 检索到的相似句子，含相似度分数）"
            )
            parts.append(_format_refs_section(references))

        if web_refs:
            parts.append(_WEB_SECTION_HEADER)
            parts.append(_format_web_section(web_refs))

        parts.append(_format_feedback_rules(content_type))

        parts.append("## 对话内容")
        parts.append(_format_dialog_lines(sources, speakers))

        parts.append(
            "## 输出格式（严格遵守）\n"
            '逐行输出，每行格式为 `序号. {"translation": "目标语译文", "reason": "简述理由（中文，1句话）"}`，'
            "序号与对话编号严格对应，"
            f"共 {len(sources)} 行。不要重复输出说话人姓名，不要输出原文，"
            "不要加任何解释、标题或 Markdown。"
        )

        user = "\n\n".join(parts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ---------- helpers ----------

    def _resolve_project(self, project_id: str | None, rag_collection: str | None) -> _ProjectContext:
        if project_id and self.project_resources is not None:
            profile = self.project_resources.profile(project_id)
            prompt_notes = self.project_resources.prompt_notes(project_id)
            return _ProjectContext(
                profile=profile,
                terminology_svc=self.project_resources.terminology(project_id),
                style_guide_svc=self.project_resources.style_guide(project_id),
                rag_collection=rag_collection or profile.qdrant_collection,
                base_system=_base_system_for_project(profile, prompt_notes),
                web_search_prefix=profile.web_search_prefix,
                vision_system_prompt=profile.vision_system_prompt,
            )
        if project_id and self.project_resources is None:
            logger.warning("收到 project_id=%s，但未配置 ProjectResourceManager，使用旧全局状态", project_id)
        return _ProjectContext(
            profile=None,
            terminology_svc=self.terminology_svc,
            style_guide_svc=self.style_guide_svc,
            rag_collection=rag_collection,
            base_system=_BASE_SYSTEM,
            web_search_prefix=None,
            vision_system_prompt=None,
        )

    async def _classify_text(
        self,
        source: str,
        hint: str | None = None,
        *,
        style_guide_svc: StyleGuideService | None = None,
    ) -> ContentType:
        """对文本进行类型分类；失败时静默降级为 UNKNOWN。

        ``hint`` 传入时先尝试直接匹配为 ContentType，匹配成功则跳过 LLM 调用。
        """
        if hint:
            # 精确匹配
            for ct in ContentType:
                if ct.value == hint:
                    return ct
            # 宽松匹配（用户可能用了别名或大小写不一致）
            hint_norm = hint.strip()
            for ct in ContentType:
                if ct != ContentType.UNKNOWN and ct.value in hint_norm:
                    return ct

        sg = style_guide_svc or self.style_guide_svc
        if not sg.loaded:
            return ContentType.UNKNOWN
        messages = sg.build_classification_messages(source)
        raw = await self.llm_svc.classify(messages)
        if raw in {ct.value for ct in ContentType}:
            return ContentType(raw)
        # 宽松匹配：LLM 可能输出含有类型名的短语
        for ct in ContentType:
            if ct != ContentType.UNKNOWN and ct.value in raw:
                return ct
        logger.debug("分类结果 %r 不匹配已知类型，退化为 UNKNOWN", raw)
        return ContentType.UNKNOWN

    def _collect_terms(
        self,
        sources: list[str],
        *,
        terminology_svc: TerminologyService | None = None,
    ) -> list[TermEntry]:
        """对一组句子取术语命中并集，按首次出现顺序去重。"""
        seen: set[str] = set()
        out: list[TermEntry] = []
        terms = terminology_svc or self.terminology_svc
        for s in sources:
            for entry in terms.find_matches(s):
                if entry.source in seen:
                    continue
                seen.add(entry.source)
                out.append(entry)
        return out

    async def _collect_references(
        self,
        sources: list[str],
        *,
        threshold: float | None,
        top_k: int | None,
        collection: str | None = None,
    ) -> list[RAGSearchResult]:
        """对一组句子并行做 RAG 检索，按 (source,target) 去重并按 score 降序保留。

        单句检索异常不会拖垮整组，错误会被吞掉记日志。
        """
        try:
            results = await asyncio.gather(
                *(
                    self.rag_svc.search(s, threshold=threshold, top_k=top_k, collection=collection)
                    for s in sources
                ),
                return_exceptions=True,
            )
        except Exception as exc:
            logger.warning("整组 RAG 检索整体失败: %s", exc)
            return []

        seen: dict[tuple[str, str], RAGSearchResult] = {}
        for r in results:
            if isinstance(r, Exception):
                logger.debug("整组 RAG 单句失败已忽略: %s", r)
                continue
            for ref in r:
                key = (ref.source, ref.target)
                prev = seen.get(key)
                if prev is None or ref.score > prev.score:
                    seen[key] = ref

        merged = sorted(seen.values(), key=lambda r: r.score, reverse=True)
        per_query_k = top_k if top_k is not None else self.rag_svc.top_k
        cap = max(per_query_k, per_query_k * 2)
        return merged[:cap]

    async def _maybe_tm_exact_result(
        self,
        source: str,
        *,
        collection: str | None,
        content_type: ContentType | None,
    ) -> TranslationResult | None:
        try:
            matches = await self.rag_svc.find_exact_source_matches(
                source,
                collection=collection,
                top_k=1,
            )
        except Exception as exc:
            logger.warning("TM 精确匹配查询失败，降级为常规翻译: %s", exc)
            return None
        if not matches:
            return None

        match = matches[0]
        return TranslationResult(
            source=source,
            translation=match.target,
            translation_reason=(
                "TM_EXACT_MATCH: source 与 TM 语料完全一致，直接采用最高优先级 TM target，跳过 AI 翻译。"
            ),
            status="success",
            content_type=content_type.value if content_type is not None else None,
            rag_references=self._refs_view(matches),
            tm_exact_match_used=True,
            tm_exact_match_source=match.source,
            tm_exact_match_target=match.target,
            tm_exact_match_status=match.status,
            tm_exact_match_score=match.score,
        )

    async def _collect_tm_exact_results(
        self,
        sources: list[str],
        *,
        collection: str | None,
        content_type: ContentType | None,
    ) -> list[TranslationResult | None]:
        results = await asyncio.gather(
            *(
                self._maybe_tm_exact_result(
                    src,
                    collection=collection,
                    content_type=content_type,
                )
                for src in sources
            )
        )
        return list(results)

    async def _fallback_singles(
        self,
        sources: list[str],
        *,
        enable_rag: bool,
        rag_threshold: float | None,
        rag_top_k: int | None,
        content_type: ContentType | None = None,
        rag_collection: str | None = None,
        enable_web_search: bool = False,
        web_search_dense_threshold: float | None = None,
        enable_vision: bool = True,
        project_id: str | None = None,
        use_tm_exact_match: bool = False,
    ) -> list[TranslationResult]:
        out: list[TranslationResult] = []
        for src in sources:
            out.append(
                await self.translate_single(
                    src,
                    enable_rag=enable_rag,
                    rag_threshold=rag_threshold,
                    rag_top_k=rag_top_k,
                    content_type=content_type,
                    rag_collection=rag_collection,
                    enable_web_search=enable_web_search,
                    web_search_dense_threshold=web_search_dense_threshold,
                    enable_vision=enable_vision,
                    project_id=project_id,
                    use_tm_exact_match=use_tm_exact_match,
                )
            )
        return out

    # ---------- Web 搜索触发逻辑 ----------

    def _web_search_active(self, enable_web_search: bool) -> bool:
        """请求级开关 ∧ 注入了 web_search_svc ∧ 服务自身 enabled。任一缺失就跳过。"""
        return (
            enable_web_search
            and self.web_search_svc is not None
            and getattr(self.web_search_svc, "enabled", False)
        )

    @staticmethod
    def _should_trigger_web_search(
        *,
        term_count: int,
        dense_top1: float | None,
        sparse_hits: int,
        dense_threshold: float,
    ) -> bool:
        """术语未命中、sparse 未命中、且 dense top1 < 阈值时才触发。

        ``dense_top1 is None`` 视为 0（无召回，触发）。
        """
        if term_count > 0:
            return False
        if sparse_hits > 0:
            return False
        score = dense_top1 if dense_top1 is not None else 0.0
        return score < dense_threshold

    async def _maybe_web_search(
        self,
        query: str,
        *,
        term_matches: list[TermEntry],
        diag: RAGDiagnostics,
        enable_web_search: bool,
        user_threshold: float | None,
        enable_vision: bool = True,
        web_search_prefix: str | None = None,
        vision_system_prompt: str | None = None,
    ) -> tuple[list[WebSearchResult], bool]:
        """单句路径用：判定 + 调用 Web 搜索。返回 (refs, triggered)。

        永远不抛异常（WebSearchService 内部已静默降级）。
        """
        if not self._web_search_active(enable_web_search):
            return [], False
        thr = user_threshold if user_threshold is not None else self.default_web_search_dense_threshold
        if not self._should_trigger_web_search(
            term_count=len(term_matches),
            dense_top1=diag.dense_top1,
            sparse_hits=diag.sparse_hits,
            dense_threshold=thr,
        ):
            return [], False
        logger.info(
            "web_search trigger query=%r dense_top1=%s sparse_hits=%d",
            query[:30],
            diag.dense_top1,
            diag.sparse_hits,
        )
        try:
            refs = await self.web_search_svc.search(query, prefix=web_search_prefix)  # type: ignore[union-attr]
        except Exception as exc:  # WebSearchService 已经静默降级，这里只是双保险
            logger.warning("web search 异常，降级为空参考: %s", exc)
            refs = []

        # 对第一条结果的配图做多模态分析（仅在有图且有 vision_svc 且启用视觉分析时）
        if enable_vision and refs and self.vision_svc:
            first = refs[0]
            if first.image_url and first.image_analysis is None:
                analysis = await self.vision_svc.analyze_image(
                    first.image_url,
                    system_prompt=vision_system_prompt,
                )
                if analysis:
                    refs[0] = first.model_copy(update={"image_analysis": analysis})

        return refs, True

    async def _group_web_search(
        self,
        cleaned: list[str],
        *,
        template_hint: str,
        term_matches: list[TermEntry],
        enable_rag: bool,
        rag_threshold: float | None,
        rag_collection: str | None,
        enable_web_search: bool,
        user_threshold: float | None,
        enable_vision: bool = True,
        web_search_prefix: str | None = None,
        vision_system_prompt: str | None = None,
    ) -> tuple[list[WebSearchResult], bool]:
        """整组/对话路径用：只对代表句发一次诊断查询并触发 Web 搜索。

        代表句选择：``template_hint`` 非空时用 template_hint（组共有骨架），
        否则退到 ``cleaned[0]``。
        """
        if not self._web_search_active(enable_web_search):
            return [], False
        if not enable_rag:
            # RAG 关闭时没有诊断信息，认为是弱召回的扩展条件不成立 → 不触发
            return [], False

        rep_query = template_hint if template_hint else cleaned[0]
        try:
            _refs, diag = await self.rag_svc.search_with_diagnostics(
                rep_query,
                threshold=rag_threshold,
                top_k=1,
                collection=rag_collection,
            )
        except Exception as exc:
            logger.warning("整组 RAG 诊断查询失败，跳过 web search: %s", exc)
            return [], False

        return await self._maybe_web_search(
            rep_query,
            term_matches=term_matches,
            diag=diag,
            enable_web_search=enable_web_search,
            user_threshold=user_threshold,
            enable_vision=enable_vision,
            web_search_prefix=web_search_prefix,
            vision_system_prompt=vision_system_prompt,
        )

    def _build_messages(
        self,
        source: str,
        term_matches: list[TermEntry],
        references: list[RAGSearchResult],
        content_type: ContentType,
        *,
        web_refs: list[WebSearchResult] | None = None,
        style_guide_svc: StyleGuideService | None = None,
        base_system: str | None = None,
    ) -> list[dict[str, str]]:
        sg = style_guide_svc or self.style_guide_svc
        system = sg.build_system_prompt_for_type(base_system or _BASE_SYSTEM, content_type)

        parts: list[str] = []
        if term_matches:
            parts.append(_TERM_SECTION_TITLE)
            parts.append(_format_term_section(term_matches))

        if references:
            parts.append(
                "## 参考翻译（这是从游戏语料库中 RAG 检索到的相似句子，请根据 score 相似度分数进行参考）"
            )
            parts.append(_format_refs_section(references))

        if web_refs:
            parts.append(_WEB_SECTION_HEADER)
            parts.append(_format_web_section(web_refs))

        parts.append(_format_feedback_rules(content_type))

        parts.append("## 待翻译文本")
        parts.append(source)
        parts.append(
            "\n请以 JSON 格式输出（仅输出 JSON，不含任何其他文字或 markdown）：\n"
            '{"translation": "目标语译文", "reason": "用中文简述翻译理由，说明采用了哪些术语/RAG例句/网络资料，'
            '或关键语境判断（1-2句话）"}'
        )

        user = "\n\n".join(parts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _build_group_messages(
        self,
        sources: list[str],
        term_matches: list[TermEntry],
        references: list[RAGSearchResult],
        content_type: ContentType,
        template_hint: str,
        *,
        web_refs: list[WebSearchResult] | None = None,
        style_guide_svc: StyleGuideService | None = None,
        base_system: str | None = None,
    ) -> list[dict[str, str]]:
        sg = style_guide_svc or self.style_guide_svc
        system = sg.build_system_prompt_for_type(base_system or _BASE_SYSTEM, content_type)

        intro_lines = [
            f"以下 {len(sources)} 句源文本结构高度一致，请使用**完全一致的目标语句式**翻译。",
            "仅替换变量部分；句式骨架（前后缀、连接词、标点风格、大小写）必须 1:1 对应，"
            "确保整组译文模板严格统一。",
        ]
        if template_hint:
            intro_lines.append(f"识别出的共有结构提示：`{template_hint}`")

        parts: list[str] = ["## 整组翻译任务", "\n".join(intro_lines)]

        if term_matches:
            parts.append(_TERM_SECTION_TITLE)
            parts.append(_format_term_section(term_matches))

        if references:
            parts.append(
                "## 参考翻译（从游戏语料库 RAG 检索到的相似句子，含相似度分数）"
            )
            parts.append(_format_refs_section(references))

        if web_refs:
            parts.append(_WEB_SECTION_HEADER)
            parts.append(_format_web_section(web_refs))

        parts.append(_format_feedback_rules(content_type))

        parts.append("## 源文本")
        parts.append("\n".join(f"{i}. {s}" for i, s in enumerate(sources, 1)))

        parts.append(
            "## 输出格式（严格遵守）\n"
            '逐行输出，每行格式为 `序号. {"translation": "目标语译文", "reason": "简述理由（中文，1句话）"}`，'
            "序号与源文本编号严格对应，"
            f"共 {len(sources)} 行。不要输出任何额外解释、标题或 Markdown。"
        )

        user = "\n\n".join(parts)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    @staticmethod
    def _term_used_view(term_matches: list[TermEntry]) -> list[dict]:
        out: list[dict] = []
        for entry in term_matches:
            item = {"source": entry.source, "target": entry.target}
            if entry.category:
                item["category"] = entry.category
            out.append(item)
        return out

    @staticmethod
    def _refs_view(references: list[RAGSearchResult]) -> list[dict] | None:
        if not references:
            return None
        out: list[dict] = []
        for r in references:
            item = {"source": r.source, "target": r.target, "score": round(r.score, 4)}
            if r.status:
                item["status"] = r.status
            out.append(item)
        return out

    @staticmethod
    def _web_view(web_refs: list[WebSearchResult] | None) -> list[dict] | None:
        if not web_refs:
            return None
        return [r.model_dump() for r in web_refs]


def _merge_direct_results(
    direct_results: list[TranslationResult | None],
    translated_unmatched: list[TranslationResult],
) -> list[TranslationResult]:
    translated_iter = iter(translated_unmatched)
    merged: list[TranslationResult] = []
    for direct in direct_results:
        if direct is not None:
            merged.append(direct)
        else:
            merged.append(next(translated_iter))
    return merged


def _extract_translation_reason(text: str) -> tuple[str, str | None] | None:
    """Regex 兜底：提取含未转义引号的残缺 JSON 中的 translation 和 reason。

    translation 用 ``[^"]*`` 匹配（英文译文通常不含裸引号）；
    reason 用贪婪 ``.*`` 配合末尾 ``"\\s*}`` 锚点，正确吞掉内嵌引号。
    返回 None 表示未能提取 translation 字段。
    """
    trans_m = re.search(r'"translation"\s*:\s*"([^"]*)"', text)
    if not trans_m:
        return None
    reason_m = re.search(r'"reason"\s*:\s*"(.*)"[\s\n]*\}', text, re.DOTALL)
    reason = reason_m.group(1).strip() if reason_m else None
    return trans_m.group(1).strip(), reason or None


def _parse_single_llm_output(raw: str) -> tuple[str, str | None]:
    """从单句 LLM 输出中提取译文和理由。

    期望格式：``{"translation": "...", "reason": "..."}``。
    JSON 解析失败时尝试 regex 兜底；仍失败则将整体内容当作译文，理由返回 None。
    """
    text = raw.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "translation" in data:
            translation = str(data["translation"]).strip()
            reason = str(data.get("reason", "") or "").strip() or None
            return translation, reason
    except (json.JSONDecodeError, ValueError):
        pass
    result = _extract_translation_reason(text)
    if result is not None:
        return result
    return text, None


def _parse_line_text(text: str) -> tuple[str, str | None]:
    """从编号行的正文部分解析译文和理由。

    期望格式：``{"translation": "...", "reason": "..."}``。
    JSON 解析失败时尝试 regex 兜底；仍失败则将整体当译文，理由返回 None。
    """
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "translation" in data:
            translation = str(data["translation"]).strip()
            reason = str(data.get("reason", "") or "").strip() or None
            return translation, reason
    except (json.JSONDecodeError, ValueError):
        pass
    result = _extract_translation_reason(text)
    if result is not None:
        return result
    return text, None


def _parse_numbered_output_positional(
    raw: str, expected: int
) -> list[tuple[str, str | None]] | None:
    """按位置解析 LLM 编号输出，宽容编号数值本身。

    要求严格匹配 ``expected`` 行带数字前缀的输出，按出现顺序映射到原文位置；
    实际数字（``1`` / ``11000`` / 跳号等）不参与校验。仅当数字编号行数 ≠ expected
    时返回 None 触发上层 fallback。

    每行正文尝试解析为 ``{"translation": ..., "reason": ...}``，失败时整体当译文。
    """
    if not raw or expected <= 0:
        return None
    items: list[tuple[str, str | None]] = []
    for line in raw.splitlines():
        m = _NUMBERED_LINE_RE.match(line)
        if not m:
            continue
        text = m.group(2).strip()
        if not text:
            continue
        items.append(_parse_line_text(text))
    if len(items) != expected:
        return None
    return items


def _parse_numbered_output(raw: str, expected: int) -> list[tuple[str, str | None]] | None:
    """解析 LLM 整组输出为按序号排序的 (译文, 理由) 列表。

    要求严格匹配 ``expected`` 条、且序号集合恰好为 ``{1..expected}``；
    否则返回 None 触发上层 fallback。

    每行正文尝试解析为 ``{"translation": ..., "reason": ...}``，失败时整体当译文。
    """
    if not raw or expected <= 0:
        return None
    found: dict[int, tuple[str, str | None]] = {}
    for line in raw.splitlines():
        m = _NUMBERED_LINE_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        text = m.group(2).strip()
        if not text:
            continue
        # 同一序号重复时保留首条
        found.setdefault(idx, _parse_line_text(text))
    if len(found) != expected:
        return None
    if set(found.keys()) != set(range(1, expected + 1)):
        return None
    return [found[i] for i in range(1, expected + 1)]


def _format_feedback_rules(content_type: ContentType) -> str:
    """把 LQE 人工反馈沉淀为每次翻译都会看到的操作规则。"""
    lines = [
        "## AIPE 反馈优化规则",
        "- 标点与英文习惯：不要使用 em dash；中文省略号要改为英文 `...` 或重组短句；英文中使用 straight punctuation。",
        "- 语言自然度：不要机械贴中文语序；避免重复句式和重复用词；可在不改意义的前提下调整主谓宾、连接词和信息顺序。",
        "- 术语策略：术语/RAG 用来帮助判断专名和统一译法，不要把术语表当作逐字替换表；若术语在当前语境里只是普通含义、自称或泛称，应按英语自然表达处理。",
        "- 人称与称谓：自称、昵称、拟声化自指、玩家称呼要先判断说话关系；自称优先转为第一人称，面向玩家的称呼可转为 you/your。",
        "- 古典/文言感文本：先理解实词虚词和语义关系，再翻译；不要逐字对应，必要时在 reason 里说明不确定点。",
    ]

    if content_type in _FUNCTIONAL_TERM_TYPES:
        lines.extend(
            [
                "- 当前文本按功能性文本处理：术语一致性优先，任务/UI/道具/技能/规则说明需稳定复用术语和高分 RAG。",
                "- 功能性文本的译文应句式短、清楚、可执行；避免过度文学化或随意改写。",
            ]
        )
    elif content_type in _CONTEXTUAL_TERM_TYPES:
        lines.extend(
            [
                "- 当前文本按口语/剧情/邮件类文本处理：语气、说话人身份和上下文优先于机械术语套用。",
                "- 口语和 playful/casual 文本应自然、轻快，可使用 contraction 和常见英语口语表达；避免商务正式腔。",
            ]
        )
    else:
        lines.extend(
            [
                "- 当前文本类型不够明确：先判断它更像功能性说明还是叙事/口语，再决定术语强度。",
                "- 若是系统操作、任务、道具、奖励，偏严格；若是对白、邮件、叙述、背景文案，偏自然。",
            ]
        )
    return "\n".join(lines)


def _format_term_section(term_matches: list[TermEntry]) -> str:
    lines = []
    for entry in term_matches:
        cat = f"（{entry.category}）" if entry.category else ""
        lines.append(f'- 「{entry.source}」{cat} → "{entry.target}"')
    return "\n".join(lines)


def _format_dialog_lines(sources: list[str], speakers: list[str | None]) -> str:
    """格式化为 ``1. [说话人] 原文`` 形式；说话人缺失时直接 ``1. 原文``。

    刻意**不**把 ``time`` 写进 prompt：游戏里 ``time`` 字段可能是大整数（如 11000+），
    曾出现 LLM 把 ``(t=11000)`` 当成行号输出导致解析失败的 bug。``time`` 只用于排序。
    """
    lines: list[str] = []
    for i, src in enumerate(sources, 1):
        speaker = (speakers[i - 1] or "").strip() if speakers else ""
        if speaker:
            lines.append(f"{i}. [{speaker}] {src}")
        else:
            lines.append(f"{i}. {src}")
    return "\n".join(lines)


def _format_refs_section(references: list[RAGSearchResult]) -> str:
    lines = []
    for i, r in enumerate(references, 1):
        lines.append(
            f'{i}. [score={r.score:.3f}] 「{r.source}」 → "{r.target}"'
        )
    return "\n".join(lines)


def _format_web_section(web_refs: list[WebSearchResult]) -> str:
    """格式化网络参考段。image_analysis 优先于裸 image_url 注入 prompt。"""
    blocks: list[str] = []
    for i, r in enumerate(web_refs, 1):
        site = f" — {r.site_name}" if r.site_name else ""
        title_line = f"{i}. 「{r.title}」{site}"
        lines = [title_line, f"   摘要：{r.snippet}", f"   链接：{r.url}"]
        if r.image_analysis:
            lines.append(f"   配图视觉分析（多模态AI）：{r.image_analysis}")
        elif r.image_url:
            lines.append(f"   配图：{r.image_url}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _extract_image_analysis(web_refs: list[WebSearchResult] | None) -> str | None:
    """从 web_refs 中提取第一个非空的 image_analysis。"""
    if not web_refs:
        return None
    for r in web_refs:
        if r.image_analysis:
            return r.image_analysis
    return None


def _base_system_for_project(profile: ProjectProfile, prompt_notes: str = "") -> str:
    parts = [_BASE_SYSTEM.rstrip()]
    project_lines: list[str] = []
    if profile.game:
        project_lines.append(f"项目：{profile.game}")
    if profile.language_pair:
        project_lines.append(f"语言方向：{profile.language_pair}")
    source_lang = _language_name(profile.source_lang) if profile.source_lang else None
    target_lang = _language_name(profile.target_lang) if profile.target_lang else None
    if (not source_lang or not target_lang) and profile.language_pair:
        source_lang, target_lang = _language_pair_names(profile.language_pair)
    if source_lang and target_lang:
        project_lines.append(f"源语言：{source_lang}")
        project_lines.append(f"目标语言：{target_lang}")
        project_lines.append(f"请将源语言文本翻译为{target_lang}。")
    if profile.background:
        project_lines.append(f"背景与语气：{profile.background}")
    if project_lines:
        parts.append("## 项目背景\n" + "\n".join(project_lines))
    if prompt_notes:
        parts.append("## 项目补充提示\n" + prompt_notes)
    return "\n\n".join(parts)


def _language_pair_names(language_pair: str) -> tuple[str | None, str | None]:
    parts = [p for p in re.split(r"[-_>/→]+", language_pair.strip()) if p]
    if len(parts) < 2:
        return None, None
    return _language_name(parts[0]), _language_name(parts[-1])


def _language_name(code: str) -> str:
    normalized = re.sub(r"[^A-Za-z-]", "", code).upper()
    return _LANGUAGE_NAMES.get(normalized, code.strip())


__all__ = ["TranslationPipeline"]
