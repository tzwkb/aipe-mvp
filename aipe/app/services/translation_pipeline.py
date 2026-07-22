"""翻译 Pipeline 编排：术语命中 → RAG → Prompt → LLM → 输出。

术语命中结果与 RAG 检索结果一同作为 Prompt 的参考段，由 LLM 结合语境决定是否采用，
避免占位符强替换带来的"硬翻译"。

提供两条调用路径：
- ``translate_single``：单句无状态端到端翻译（默认路径）
- ``translate_group``：整组并排翻译（结构高度相似的句子用同一目标语句式，保证一致性）
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from app.errors import TranslationError
from app.schemas.rag import RAGSearchResult
from app.schemas.terminology import TermEntry
from app.schemas.translate import TranslationResult
from app.schemas.web_search import WebSearchResult
from app.services.llm_service import LLMService
from app.services.project_service import ProjectResourceManager
from app.services.rag_service import RAGDiagnostics, RAGService
from app.services.style_guide_service import ContentType, StyleGuideService
from app.services.terminology_service import TerminologyService
from app.services.translation_memory import TMExactMatchResolver, merge_exact_results
from app.services.translation_output import (
    extract_image_analysis,
    parse_numbered_output,
    parse_numbered_output_positional,
    parse_single_llm_output,
)
from app.services.translation_prompts import (
    BASE_SYSTEM,
    base_system_for_project,
    build_dialog_messages,
    build_group_messages,
    build_single_messages,
)
from app.services.vision_service import VisionService
from app.services.web_search_service import WebSearchService
from app.utils.cluster import common_template_repr

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _ProjectContext:
    terminology_svc: TerminologyService
    style_guide_svc: StyleGuideService
    rag_collection: str | None
    base_system: str
    target_lang: str
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
        self.tm_exact_match_resolver = TMExactMatchResolver(rag_svc)
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
            direct = (
                await self.tm_exact_match_resolver.resolve_many(
                    [source],
                    collection=effective_collection,
                    content_type=content_type,
                )
            )[0]
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
        messages = build_single_messages(
            source,
            term_matches,
            references,
            content_type,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
            target_lang=project_ctx.target_lang,
        )

        image_analysis = extract_image_analysis(web_refs)

        # Step 4: LLM 翻译
        try:
            raw = await self.llm_svc.translate(messages)
            translation, translation_reason = parse_single_llm_output(raw)
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

        direct_results: list[TranslationResult | None] = [None] * len(cleaned)
        if use_tm_exact_match:
            direct_results = await self.tm_exact_match_resolver.resolve_many(
                cleaned,
                collection=effective_collection,
                content_type=content_type,
            )
            if all(r is not None for r in direct_results):
                return [r for r in direct_results if r is not None]

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
        for direct in direct_results:
            if direct is not None and direct.content_type is None:
                direct.content_type = content_type.value

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
        messages = build_group_messages(
            cleaned,
            term_matches,
            references,
            content_type,
            template_hint,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
            target_lang=project_ctx.target_lang,
            locked_tm_results=direct_results,
        )

        # Step 4: LLM 调用
        try:
            raw = await self.llm_svc.translate(messages)
        except TranslationError as exc:
            logger.warning("整组翻译 LLM 失败，回退单句: %s", exc)
            fallback_results = await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                content_type=content_type,
            )
            return merge_exact_results(direct_results, fallback_results)

        # Step 5: 解析编号输出
        parsed = parse_numbered_output(raw, expected=len(cleaned))
        if parsed is None:
            logger.warning(
                "整组翻译输出解析失败（期望 %d 行），回退单句。原始输出前 200 字: %r",
                len(cleaned),
                raw[:200],
            )
            fallback_results = await self._fallback_singles(
                sources,
                enable_rag=enable_rag,
                rag_threshold=rag_threshold,
                rag_top_k=rag_top_k,
                rag_collection=rag_collection,
                enable_web_search=enable_web_search,
                web_search_dense_threshold=web_search_dense_threshold,
                enable_vision=enable_vision,
                project_id=project_id,
                content_type=content_type,
            )
            return merge_exact_results(direct_results, fallback_results)

        term_view = self._term_used_view(term_matches)
        refs_view = self._refs_view(references)
        web_view = self._web_view(web_refs)
        web_triggered_view = web_triggered if enable_web_search else None
        image_analysis = extract_image_analysis(web_refs)
        generated_results = [
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
        return merge_exact_results(direct_results, generated_results)

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

        direct_results: list[TranslationResult | None] = [None] * len(cleaned)
        if use_tm_exact_match:
            direct_results = await self.tm_exact_match_resolver.resolve_many(
                cleaned,
                collection=effective_collection,
                content_type=content_type,
            )
            if all(r is not None for r in direct_results):
                return [r for r in direct_results if r is not None]

        # Step 0: 分类（传入时跳过；否则对代表句预分类）
        if content_type is None:
            content_type = await self._classify_text(cleaned[0], style_guide_svc=project_ctx.style_guide_svc)
        for direct in direct_results:
            if direct is not None and direct.content_type is None:
                direct.content_type = content_type.value

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
        messages = build_dialog_messages(
            cleaned, speakers, term_matches, references, content_type, dialog_id,
            web_refs=web_refs,
            style_guide_svc=project_ctx.style_guide_svc,
            base_system=project_ctx.base_system,
            target_lang=project_ctx.target_lang,
            locked_tm_results=direct_results,
        )

        term_view = self._term_used_view(term_matches)
        refs_view = self._refs_view(references)
        web_view = self._web_view(web_refs)
        web_triggered_view = web_triggered if enable_web_search else None
        image_analysis = extract_image_analysis(web_refs)

        # Step 4: LLM
        try:
            raw = await self.llm_svc.translate(messages)
        except TranslationError as exc:
            logger.warning(
                "对话翻译 LLM 失败 dialog_id=%s n=%d: %s", dialog_id, len(cleaned), exc
            )
            error_results = self._dialog_error_results(
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
            return merge_exact_results(direct_results, error_results)

        # Step 5: 解析编号输出
        # 对话路径不强求编号为 {1..N}：LLM 可能保留 time 风格的数字（如游戏内部的
        # 时间码）。只要求 N 行编号、按顺序映射到 N 个原文位置即可。
        parsed = parse_numbered_output_positional(raw, expected=len(cleaned))
        if parsed is None:
            logger.warning(
                "对话翻译输出解析失败 dialog_id=%s 期望 %d 行，原始输出前 200 字: %r",
                dialog_id,
                len(cleaned),
                raw[:200],
            )
            error_results = self._dialog_error_results(
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
            return merge_exact_results(direct_results, error_results)

        generated_results = [
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
        return merge_exact_results(direct_results, generated_results)

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

    # ---------- helpers ----------

    def _resolve_project(self, project_id: str | None, rag_collection: str | None) -> _ProjectContext:
        if project_id and self.project_resources is not None:
            profile = self.project_resources.profile(project_id)
            prompt_notes = self.project_resources.prompt_notes(project_id)
            return _ProjectContext(
                terminology_svc=self.project_resources.terminology(project_id),
                style_guide_svc=self.project_resources.style_guide(project_id),
                rag_collection=rag_collection or profile.qdrant_collection,
                base_system=base_system_for_project(profile, prompt_notes),
                target_lang=profile.target_lang,
                web_search_prefix=profile.web_search_prefix,
                vision_system_prompt=profile.vision_system_prompt,
            )
        if project_id and self.project_resources is None:
            logger.warning("收到 project_id=%s，但未配置 ProjectResourceManager，使用旧全局状态", project_id)
        return _ProjectContext(
            terminology_svc=self.terminology_svc,
            style_guide_svc=self.style_guide_svc,
            rag_collection=rag_collection,
            base_system=BASE_SYSTEM,
            target_lang="en",
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


__all__ = ["TranslationPipeline"]
