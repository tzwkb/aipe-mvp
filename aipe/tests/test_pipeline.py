"""TranslationPipeline 单元测试。

LLM / Embedding 全部用 fake 实现，不依赖任何外部服务。
"""

from __future__ import annotations

import asyncio
import json

from app.schemas.rag import RAGSearchResult
from app.schemas.terminology import TermEntry
from app.schemas.web_search import WebSearchResult
from app.services.project_service import ProjectRegistry, ProjectResourceManager
from app.services.rag_service import RAGDiagnostics
from app.services.style_guide_service import ContentType, StyleGuideService
from app.services.terminology_service import TerminologyService
from app.services.translation_pipeline import TranslationPipeline


class FakeLLM:
    """记录 messages 的伪 LLM。``response_fn`` 决定返回内容。"""

    def __init__(self, response_fn=None, classify_fn=None):
        self.calls: list[list[dict]] = []
        self.classify_calls: list[list[dict]] = []
        self.response_fn = response_fn or (lambda msgs: "stubbed translation")
        self.classify_fn = classify_fn or (lambda msgs: "未知")

    async def translate(self, prompt, *, temperature=None, max_retries=None):
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        self.calls.append(msgs)
        out = self.response_fn(msgs)
        if asyncio.iscoroutine(out):
            return await out
        return out

    async def classify(self, messages):
        self.classify_calls.append(messages)
        out = self.classify_fn(messages)
        if asyncio.iscoroutine(out):
            return await out
        return out

    async def embed(self, text):
        return [0.0]

    async def embed_batch(self, texts):
        return [[0.0] for _ in texts]


class FakeRAG:
    def __init__(
        self,
        results: list[RAGSearchResult] | None = None,
        raise_exc: Exception | None = None,
        results_by_query: dict[str, list[RAGSearchResult]] | None = None,
        exact_results_by_query: dict[str, list[RAGSearchResult]] | None = None,
        top_k: int = 3,
        diagnostics: RAGDiagnostics | None = None,
        diagnostics_by_query: dict[str, RAGDiagnostics] | None = None,
    ):
        self._results = results or []
        self._raise = raise_exc
        self._by_query = results_by_query or {}
        self._exact_by_query = exact_results_by_query or {}
        self.top_k = top_k
        self.calls: list[tuple[str, float | None, int | None]] = []
        self.exact_calls: list[tuple[str, str | None, int | None]] = []
        self.diag_calls: list[str] = []
        self._diag_default = diagnostics or RAGDiagnostics(dense_top1=0.9, sparse_hits=1)
        self._diag_by_query = diagnostics_by_query or {}
        self.collection_calls: list[str | None] = []

    async def search(self, query, threshold=None, top_k=None, collection=None):
        self.calls.append((query, threshold, top_k))
        self.collection_calls.append(collection)
        if self._raise:
            raise self._raise
        if query in self._by_query:
            return list(self._by_query[query])
        return list(self._results)

    async def search_with_diagnostics(self, query, threshold=None, top_k=None, collection=None):
        self.calls.append((query, threshold, top_k))
        self.collection_calls.append(collection)
        self.diag_calls.append(query)
        if self._raise:
            raise self._raise
        results = list(self._by_query.get(query, self._results))
        diag = self._diag_by_query.get(query, self._diag_default)
        return results, diag

    async def find_exact_source_matches(self, source, collection=None, top_k=None):
        self.exact_calls.append((source, collection, top_k))
        self.collection_calls.append(collection)
        if self._raise:
            raise self._raise
        return list(self._exact_by_query.get(source, []))[: top_k or self.top_k]


class FakeWebSearch:
    """伪 WebSearchService：记录 search 调用并可配置返回结果或抛异常。"""

    def __init__(
        self,
        results: list[WebSearchResult] | None = None,
        raise_exc: Exception | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._results = results or []
        self._raise = raise_exc
        self.calls: list[str] = []

    async def search(self, query: str, *, prefix: str | None = None):
        self.calls.append(query)
        if self._raise:
            raise self._raise
        return list(self._results)


def _make_pipeline(
    *,
    terms: list[TermEntry] | None = None,
    style_rules: str = "保持武侠调性。",
    rag_results: list[RAGSearchResult] | None = None,
    llm_response_fn=None,
    rag_exc: Exception | None = None,
    diagnostics: RAGDiagnostics | None = None,
    web_search: "FakeWebSearch | None" = None,
    web_search_dense_threshold: float = 0.6,
):
    term_svc = TerminologyService()
    if terms:
        term_svc.load(terms)

    style_svc = StyleGuideService()
    if style_rules:
        style_svc.load(style_rules, filename="test.md")

    rag = FakeRAG(rag_results, raise_exc=rag_exc, diagnostics=diagnostics)
    llm = FakeLLM(response_fn=llm_response_fn)
    pipe = TranslationPipeline(
        term_svc,
        rag,
        style_svc,
        llm,
        web_search_svc=web_search,
        web_search_dense_threshold=web_search_dense_threshold,
    )
    return pipe, term_svc, rag, llm


def _write_project(
    root,
    name: str,
    *,
    language_pair: str = "ZH-EN",
    source_lang: str = "zh",
    target_lang: str = "en",
    term_source: str,
    term_target: str,
    style: str,
    collection: str,
) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    (project_dir / "terms.json").write_text(
        json.dumps([{"source": term_source, "target": term_target}], ensure_ascii=False),
        encoding="utf-8",
    )
    (project_dir / "style.md").write_text(style, encoding="utf-8")
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": language_pair,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "game": name,
                "style_guide": "style.md",
                "terminology": "terms.json",
                "qdrant_collection": collection,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def _write_min_project(
    root,
    name: str,
    *,
    language_pair: str,
    source_lang: str,
    target_lang: str,
    collection: str,
) -> None:
    project_dir = root / name
    project_dir.mkdir(parents=True)
    (project_dir / "profile.json").write_text(
        json.dumps(
            {
                "name": name,
                "language_pair": language_pair,
                "source_lang": source_lang,
                "target_lang": target_lang,
                "game": "Isekai",
                "qdrant_collection": collection,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_translate_single_happy_path():
    terms = [TermEntry(source="契丹", target="Khitan", category="势力")]
    refs = [RAGSearchResult(source="契丹来犯", target="The Khitan are attacking", score=0.92)]

    pipe, _, _, llm = _make_pipeline(
        terms=terms,
        rag_results=refs,
        llm_response_fn=lambda msgs: "The Khitan are attacking the gate.",
    )
    result = asyncio.run(pipe.translate_single("契丹来犯门口"))

    assert result.status == "success"
    assert result.translation == "The Khitan are attacking the gate."
    assert any(t["source"] == "契丹" for t in result.terminology_used)
    assert result.rag_references and result.rag_references[0]["source"] == "契丹来犯"

    # 原文应原样进入 prompt，不再带占位符；术语和 RAG 都作为参考段出现
    user_msg = llm.calls[0][1]["content"]
    assert "«TERM_" not in user_msg
    assert "契丹来犯门口" in user_msg
    assert "术语参考" in user_msg
    assert "Khitan" in user_msg
    assert "契丹来犯" in user_msg


def test_translate_single_uses_project_resources_and_collection(tmp_path):
    projects = tmp_path / "projects"
    _write_project(
        projects,
        "wwm/zh-en",
        term_source="燕云",
        term_target="Where Winds Meet",
        style="WWM style",
        collection="wwm_corpus",
    )
    project_resources = ProjectResourceManager(ProjectRegistry(projects_dir=projects, default_project="wwm/zh-en"))
    rag = FakeRAG()
    llm = FakeLLM(response_fn=lambda msgs: "From Where Winds Meet.")
    pipe = TranslationPipeline(
        TerminologyService(),
        rag,
        StyleGuideService(),
        llm,
        project_resources=project_resources,
    )

    result = asyncio.run(pipe.translate_single("燕云来客", project_id="wwm/zh-en"))

    assert result.status == "success"
    assert result.terminology_used == [{"source": "燕云", "target": "Where Winds Meet"}]
    assert rag.collection_calls == ["wwm_corpus"]
    system_msg = llm.calls[0][0]["content"]
    assert "WWM style" in system_msg


def test_translate_single_project_language_pair_overrides_hardcoded_english(tmp_path):
    projects = tmp_path / "projects"
    _write_min_project(
        projects,
        "isekai/en-de",
        language_pair="EN-DE",
        source_lang="en",
        target_lang="de",
        collection="isekai_de_corpus",
    )
    project_resources = ProjectResourceManager(ProjectRegistry(projects_dir=projects, default_project="isekai/en-de"))
    rag = FakeRAG()
    llm = FakeLLM(response_fn=lambda msgs: "Deutsche Übersetzung.")
    pipe = TranslationPipeline(
        TerminologyService(),
        rag,
        StyleGuideService(),
        llm,
        project_resources=project_resources,
    )

    result = asyncio.run(pipe.translate_single("Village Elder", project_id="isekai/en-de"))

    assert result.status == "success"
    system_msg = llm.calls[0][0]["content"]
    user_msg = llm.calls[0][1]["content"]
    assert "源语言：English" in system_msg
    assert "目标语言：German" in system_msg
    assert "翻译为英文" not in system_msg
    assert "英文译文" not in user_msg


def test_translate_single_no_term_match_skips_term_section():
    pipe, _, _, llm = _make_pipeline(
        terms=[TermEntry(source="契丹", target="Khitan")],
        llm_response_fn=lambda msgs: "ok",
    )
    asyncio.run(pipe.translate_single("一句完全无关的话。"))
    user_msg = llm.calls[0][1]["content"]
    assert "术语参考" not in user_msg


def test_translate_single_speech_prompt_softens_terms_and_self_reference():
    terms = [TermEntry(source="燕叽", target="Windtail")]
    pipe, _, _, llm = _make_pipeline(terms=terms, llm_response_fn=lambda msgs: "I got it.")

    asyncio.run(
        pipe.translate_single(
            "燕叽这就去看看！",
            content_type=ContentType.SPEECH,
        )
    )

    user_msg = llm.calls[0][1]["content"]
    assert "口语/剧情/邮件类文本" in user_msg
    assert "自称" in user_msg
    assert "第一人称" in user_msg
    assert "不要把术语表当作逐字替换表" in user_msg


def test_translate_single_functional_prompt_keeps_terms_strict_and_concise():
    terms = [TermEntry(source="燕云", target="Where Winds Meet")]
    pipe, _, _, llm = _make_pipeline(terms=terms, llm_response_fn=lambda msgs: "Task text.")

    asyncio.run(
        pipe.translate_single(
            "前往燕云完成任务",
            content_type=ContentType.QUEST_OBJECTIVE,
        )
    )

    user_msg = llm.calls[0][1]["content"]
    assert "功能性文本" in user_msg
    assert "术语一致性优先" in user_msg
    assert "句式短、清楚、可执行" in user_msg


def test_translate_single_prompt_includes_wwm_feedback_hard_style_rules():
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda msgs: "ok")

    asyncio.run(pipe.translate_single("这……这可怎么好……", content_type=ContentType.STORY))

    user_msg = llm.calls[0][1]["content"]
    assert "不要使用 em dash" in user_msg
    assert "中文省略号" in user_msg
    assert "straight punctuation" in user_msg


def test_translate_single_does_not_mutate_source_for_llm():
    """确认改造后不再做占位符替换：LLM 拿到的是原文。"""
    terms = [TermEntry(source="燕云", target="Yanyun")]
    pipe, _, _, llm = _make_pipeline(terms=terms, llm_response_fn=lambda msgs: "From Yanyun.")
    asyncio.run(pipe.translate_single("我从燕云来"))
    user_msg = llm.calls[0][1]["content"]
    assert "我从燕云来" in user_msg


def test_translate_single_llm_keeps_source_term_unchanged():
    """LLM 选择不采用术语译文时，pipeline 不再强制改写译文。"""
    terms = [TermEntry(source="契丹", target="Khitan")]
    pipe, *_ = _make_pipeline(
        terms=terms,
        llm_response_fn=lambda msgs: "The enemy is attacking.",
    )
    result = asyncio.run(pipe.translate_single("契丹来犯"))
    assert result.status == "success"
    # 不再有 [TERM_FIX_NEEDED] / 强制拼接
    assert result.translation == "The enemy is attacking."


def test_translate_single_empty_input_returns_error():
    pipe, *_ = _make_pipeline()
    result = asyncio.run(pipe.translate_single("   "))
    assert result.status == "error"
    assert result.error_msg == "empty input"


def test_translate_single_llm_fail_marks_error():
    from app.errors import TranslationError

    def boom(_):
        raise TranslationError("[ERROR: AI_FAIL] simulated")

    pipe, *_ = _make_pipeline(llm_response_fn=boom)
    result = asyncio.run(pipe.translate_single("hello"))
    assert result.status == "error"
    assert "[ERROR: AI_FAIL]" in result.translation


def test_translate_single_rag_disabled_skips_search():
    refs = [RAGSearchResult(source="X", target="Y", score=0.9)]
    pipe, _, rag, _ = _make_pipeline(rag_results=refs, llm_response_fn=lambda m: "ok")
    result = asyncio.run(pipe.translate_single("任意文本", enable_rag=False))
    assert result.status == "success"
    assert rag.calls == []
    assert result.rag_references is None


def test_translate_single_rag_failure_is_swallowed():
    pipe, *_ = _make_pipeline(
        rag_exc=RuntimeError("qdrant down"), llm_response_fn=lambda m: "ok"
    )
    result = asyncio.run(pipe.translate_single("任意文本"))
    assert result.status == "success"
    assert result.rag_references is None


def test_translate_single_tm_exact_match_can_skip_ai_translation():
    exact = RAGSearchResult(
        source="契丹来犯",
        target="The Khitan are attacking",
        score=1.0,
        status="Designer Reviewed",
    )
    rag = FakeRAG(exact_results_by_query={"契丹来犯": [exact]})
    llm = FakeLLM(response_fn=lambda m: "AI should not run")
    pipe = TranslationPipeline(TerminologyService(), rag, StyleGuideService(), llm)

    result = asyncio.run(
        pipe.translate_single(
            "契丹来犯",
            content_type=ContentType.QUEST_DESCRIPTION,
            use_tm_exact_match=True,
        )
    )

    assert result.status == "success"
    assert result.translation == "The Khitan are attacking"
    assert result.translation_reason and "TM_EXACT_MATCH" in result.translation_reason
    assert result.tm_exact_match_used is True
    assert result.tm_exact_match_source == "契丹来犯"
    assert result.tm_exact_match_target == "The Khitan are attacking"
    assert result.tm_exact_match_status == "Designer Reviewed"
    assert result.tm_exact_match_score == 1.0
    assert result.rag_references == [
        {
            "source": "契丹来犯",
            "target": "The Khitan are attacking",
            "score": 1.0,
            "status": "Designer Reviewed",
        }
    ]
    assert rag.exact_calls == [("契丹来犯", None, 1)]
    assert llm.calls == []
    assert llm.classify_calls == []


def test_translate_single_tm_exact_match_default_keeps_ai_flow():
    exact = RAGSearchResult(source="契丹来犯", target="The Khitan are attacking", score=1.0)
    rag = FakeRAG(exact_results_by_query={"契丹来犯": [exact]})
    llm = FakeLLM(response_fn=lambda m: "AI translation")
    pipe = TranslationPipeline(TerminologyService(), rag, StyleGuideService(), llm)

    result = asyncio.run(pipe.translate_single("契丹来犯", content_type=ContentType.QUEST_DESCRIPTION))

    assert result.translation == "AI translation"
    assert result.tm_exact_match_used is False
    assert rag.exact_calls == []
    assert len(llm.calls) == 1


def test_translate_group_tm_exact_match_only_translates_unmatched_sources():
    exact = RAGSearchResult(source="源1", target="TM target 1", score=1.0, status="Done")
    rag = FakeRAG(exact_results_by_query={"源1": [exact]})
    llm = FakeLLM(response_fn=lambda m: '{"translation": "AI target 2", "reason": "AI"}')
    pipe = TranslationPipeline(TerminologyService(), rag, StyleGuideService(), llm)

    results = asyncio.run(
        pipe.translate_group(
            ["源1", "源2"],
            content_type=ContentType.UI,
            use_tm_exact_match=True,
        )
    )

    assert [r.translation for r in results] == ["TM target 1", "AI target 2"]
    assert [r.tm_exact_match_used for r in results] == [True, False]
    assert len(llm.calls) == 1
    user_msg = llm.calls[0][1]["content"]
    assert "源2" in user_msg
    assert "源1" not in user_msg


# ---------- 整组翻译路径 ----------


def test_translate_group_happy_path_parses_numbered_output():
    sources = ["活动【柿】获取", "活动【燕】获取", "活动【聆】获取"]
    response = (
        "1. Obtain from Event: Persimmon\n"
        "2. Obtain from Event: Swallow\n"
        "3. Obtain from Event: Echo"
    )
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: response)

    results = asyncio.run(pipe.translate_group(sources))

    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    assert [r.translation for r in results] == [
        "Obtain from Event: Persimmon",
        "Obtain from Event: Swallow",
        "Obtain from Event: Echo",
    ]
    # 整组只发一次 LLM 调用（不算分类）
    assert len(llm.calls) == 1
    user_msg = llm.calls[0][1]["content"]
    assert "整组翻译任务" in user_msg
    assert "1. 活动【柿】获取" in user_msg
    assert "3. 活动【聆】获取" in user_msg
    # template hint 应该浮现
    assert "活动【" in user_msg and "】获取" in user_msg


def test_translate_group_includes_terminology_union():
    """整组路径应收集所有源文本的术语命中（去重并集）。"""
    terms = [
        TermEntry(source="柿业有成", target="Persimmon Success"),
        TermEntry(source="燕衔嘉礼", target="Swallow Brings Festive Rites"),
        TermEntry(source="活动", target="Event"),
    ]
    sources = ["活动【柿业有成】获取", "活动【燕衔嘉礼】获取"]
    response = "1. Obtained from Event: Persimmon Success\n2. Obtained from Event: Swallow Brings Festive Rites"
    pipe, _, _, llm = _make_pipeline(terms=terms, llm_response_fn=lambda m: response)

    results = asyncio.run(pipe.translate_group(sources))
    assert len(results) == 2

    user_msg = llm.calls[0][1]["content"]
    assert "术语参考" in user_msg
    # 三个术语全部出现（去重并集）
    assert "Persimmon Success" in user_msg
    assert "Swallow Brings Festive Rites" in user_msg
    assert "Event" in user_msg


def test_translate_group_includes_merged_rag_refs():
    """整组路径应对每句独立 RAG 检索后合并去重。"""
    refs_by_q = {
        "活动【柿】获取": [
            RAGSearchResult(source="活动【A】获取", target="Obtain from Event: A", score=0.9),
        ],
        "活动【燕】获取": [
            RAGSearchResult(source="活动【B】获取", target="Obtain from Event: B", score=0.85),
            RAGSearchResult(source="活动【A】获取", target="Obtain from Event: A", score=0.7),
        ],
    }
    rag = FakeRAG(results_by_query=refs_by_q, top_k=3)
    term_svc = TerminologyService()
    style_svc = StyleGuideService()
    style_svc.load("test", filename="t.md")
    llm = FakeLLM(response_fn=lambda m: "1. T1\n2. T2")
    pipe = TranslationPipeline(term_svc, rag, style_svc, llm)

    results = asyncio.run(pipe.translate_group(["活动【柿】获取", "活动【燕】获取"]))
    assert len(results) == 2

    # RAG 被分别检索（并行）
    queries = [c[0] for c in rag.calls]
    assert "活动【柿】获取" in queries and "活动【燕】获取" in queries

    user_msg = llm.calls[0][1]["content"]
    assert "参考翻译" in user_msg
    # A 出现在两次检索结果中，去重后只出现一次；保留较高 score=0.9
    assert user_msg.count("活动【A】获取") == 1
    assert "活动【B】获取" in user_msg

    # 结果对象上 rag_references 字段也合并了
    refs_view = results[0].rag_references
    assert refs_view is not None
    keys = {(r["source"], r["target"]) for r in refs_view}
    assert ("活动【A】获取", "Obtain from Event: A") in keys
    assert ("活动【B】获取", "Obtain from Event: B") in keys


def test_translate_group_parse_failure_falls_back_to_singles():
    """LLM 输出无法解析为 N 行编号 → 回退逐句 translate_single。"""
    responses = iter(
        [
            "胡乱输出 没有编号",  # 整组调用
            "Single one",  # fallback 第 1 句
            "Single two",  # fallback 第 2 句
        ]
    )

    def respond(_msgs):
        return next(responses)

    pipe, _, _, llm = _make_pipeline(llm_response_fn=respond)
    results = asyncio.run(pipe.translate_group(["源1", "源2"]))

    assert len(results) == 2
    assert [r.translation for r in results] == ["Single one", "Single two"]
    # 1 次整组失败 + 2 次单句
    assert len(llm.calls) == 3


def test_translate_group_llm_error_falls_back_to_singles():
    from app.errors import TranslationError

    state = {"group_call_done": False}

    def respond(_msgs):
        if not state["group_call_done"]:
            state["group_call_done"] = True
            raise TranslationError("[ERROR: AI_FAIL] simulated group failure")
        return "fallback ok"

    pipe, _, _, llm = _make_pipeline(llm_response_fn=respond)
    results = asyncio.run(pipe.translate_group(["a", "b"]))

    assert len(results) == 2
    assert all(r.translation == "fallback ok" and r.status == "success" for r in results)


def test_translate_group_empty_source_falls_back():
    pipe, *_ = _make_pipeline(llm_response_fn=lambda m: "irrelevant")
    results = asyncio.run(pipe.translate_group(["a", "   "]))
    assert results[0].status == "success"
    assert results[1].status == "error"
    assert results[1].error_msg == "empty input"


def test_translate_group_single_source_delegates_to_translate_single():
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: "Hello")
    results = asyncio.run(pipe.translate_group(["你好"]))
    assert len(results) == 1
    assert results[0].translation == "Hello"
    # 不会走整组 prompt（只调用一次普通 translate）
    user_msg = llm.calls[0][1]["content"]
    assert "整组翻译任务" not in user_msg


def test_translate_group_parser_numbered_variants():
    """`1)` `1：` `1、` 都应该被接受。"""
    sources = ["a", "b", "c"]
    response = "1) Alpha\n2： Beta\n3、 Gamma"
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: response)
    results = asyncio.run(pipe.translate_group(sources))
    assert [r.translation for r in results] == ["Alpha", "Beta", "Gamma"]


def test_translate_group_missing_one_line_triggers_fallback():
    """LLM 漏了一行 → 解析失败 → 回退。"""
    responses = iter(
        [
            "1. only-one-line",  # 整组：少了 2 行
            "fb-a",
            "fb-b",
            "fb-c",
        ]
    )
    pipe, *_ = _make_pipeline(llm_response_fn=lambda m: next(responses))
    results = asyncio.run(pipe.translate_group(["a", "b", "c"]))
    assert [r.translation for r in results] == ["fb-a", "fb-b", "fb-c"]


# ---------- 对话路径 ----------


def test_translate_dialog_happy_path_includes_speakers_in_prompt():
    sources = ["这是我的册子！", "可是出了什么事？", "当年她出宫时……"]
    speakers = ["若萍", "玩家", "若萍"]
    response = (
        "1. This is my booklet!\n"
        "2. Has something happened?\n"
        "3. Back when she left the palace..."
    )
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: response)

    results = asyncio.run(
        pipe.translate_dialog(
            sources,
            speakers,
            dialog_id="23600058",
            times=[None, 5.0, 15.0],
        )
    )

    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    assert [r.translation for r in results] == [
        "This is my booklet!",
        "Has something happened?",
        "Back when she left the palace...",
    ]
    user_msg = llm.calls[0][1]["content"]
    assert "对话翻译任务" in user_msg
    assert "23600058" in user_msg  # dialog_id 注入
    assert "[若萍]" in user_msg and "[玩家]" in user_msg
    assert "1. [若萍]" in user_msg  # 行格式
    # time 不应进 prompt，避免被 LLM 当作行号回写
    assert "t=" not in user_msg


def test_translate_dialog_parse_failure_marks_entire_segment_error():
    """LLM 输出无法对齐 → 整段标 error，不回退单句。"""
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: "胡乱输出 没有编号")

    results = asyncio.run(
        pipe.translate_dialog(["s1", "s2", "s3"], ["A", "B", "A"])
    )

    assert len(results) == 3
    assert all(r.status == "error" for r in results)
    assert all("DIALOG_PARSE_FAIL" in r.translation for r in results)
    # 仅一次 LLM 调用，不回退
    assert len(llm.calls) == 1


def test_translate_dialog_llm_error_marks_entire_segment_error():
    from app.errors import TranslationError

    def respond(_msgs):
        raise TranslationError("simulated llm fail")

    pipe, *_ = _make_pipeline(llm_response_fn=respond)
    results = asyncio.run(pipe.translate_dialog(["a", "b"], ["X", "Y"]))

    assert len(results) == 2
    assert all(r.status == "error" for r in results)
    assert all("DIALOG_LLM_FAIL" in r.translation for r in results)
    assert all("simulated llm fail" in (r.error_msg or "") for r in results)


def test_translate_dialog_single_line_delegates_to_single():
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: "Hello")
    results = asyncio.run(pipe.translate_dialog(["你好"], ["若萍"]))
    assert len(results) == 1
    assert results[0].translation == "Hello"
    user_msg = llm.calls[0][1]["content"]
    assert "对话翻译任务" not in user_msg


def test_translate_dialog_speaker_length_mismatch_raises():
    import pytest

    pipe, *_ = _make_pipeline(llm_response_fn=lambda m: "irrelevant")
    with pytest.raises(ValueError):
        asyncio.run(pipe.translate_dialog(["a", "b"], ["X"]))


def test_translate_dialog_accepts_non_1n_numbering():
    """回归：LLM 把游戏内部 time 数值当行号回写（如 11000-11005，跳过 11003），
    只要行数等于 N 就按位置映射。"""
    sources = [
        "只那么点人能出去……哎……",
        "这……这可怎么好……",
        "我的也给你吧",
        "不行，抽到谁就是谁，和平日一样",
        "别哭，留下的也未必就……",
    ]
    speakers = [None, None, None, None, None]
    response = (
        "11000. Only so few can leave...\n"
        "11001. This... what shall we do...\n"
        "11002. Take mine as well\n"
        "11004. No - whoever draws, draws. Same as always\n"
        "11005. Don't weep. Those who stay..."
    )
    pipe, _, _, llm = _make_pipeline(llm_response_fn=lambda m: response)

    results = asyncio.run(
        pipe.translate_dialog(
            sources, speakers, dialog_id="13001585", times=[11000.0, 11001.0, 11002.0, 11004.0, 11005.0]
        )
    )

    assert len(results) == 5
    assert all(r.status == "success" for r in results)
    assert [r.translation for r in results] == [
        "Only so few can leave...",
        "This... what shall we do...",
        "Take mine as well",
        "No - whoever draws, draws. Same as always",
        "Don't weep. Those who stay...",
    ]


def test_translate_dialog_prompt_does_not_include_time():
    """time 仅用于排序，不应注入 prompt，避免 LLM 当行号。"""
    pipe, _, _, llm = _make_pipeline(
        llm_response_fn=lambda m: "1. A\n2. B"
    )
    asyncio.run(
        pipe.translate_dialog(
            ["源1", "源2"], ["甲", "乙"], times=[11000.0, 11001.0]
        )
    )
    user_msg = llm.calls[0][1]["content"]
    assert "11000" not in user_msg
    assert "t=" not in user_msg


def test_translate_dialog_wrong_line_count_still_fails():
    """LLM 漏 1 行 → 行数对不上 → 整段标 error。"""
    pipe, *_ = _make_pipeline(llm_response_fn=lambda m: "1. only-one")
    results = asyncio.run(pipe.translate_dialog(["a", "b", "c"], ["X", "Y", "Z"]))
    assert all(r.status == "error" for r in results)
    assert all("DIALOG_PARSE_FAIL" in r.translation for r in results)


# ---------- Web 搜索兜底（弱召回 + 术语未命中） ----------


def _weak_recall_diag() -> RAGDiagnostics:
    """触发条件全真：dense_top1<0.6 且 sparse_hits=0。"""
    return RAGDiagnostics(dense_top1=0.4, sparse_hits=0)


def _web_results() -> list[WebSearchResult]:
    return [
        WebSearchResult(
            title="新外观「凌霄破」",
            snippet="2026 赛季限定外观，源自燕云十六声世界观。",
            url="https://example.com/lingxiao",
            site_name="example.com",
            image_url="https://example.com/img.png",
        ),
    ]


def test_web_search_triggers_when_all_conditions_met():
    """术语 0 命中 + dense_top1<0.6 + sparse_hits=0 + enable_web_search=True → 调用 Bocha。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, llm = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "translated",
    )
    result = asyncio.run(
        pipe.translate_single("XX-2026赛季限定外观·凌霄破", enable_web_search=True)
    )
    assert result.status == "success"
    assert len(web.calls) == 1
    user_msg = llm.calls[0][1]["content"]
    assert "外部网络参考" in user_msg
    assert "凌霄破" in user_msg
    assert "example.com/img.png" in user_msg  # 图片 URL 也注入
    assert result.web_search_triggered is True
    assert result.web_references and result.web_references[0]["title"] == "新外观「凌霄破」"


def test_web_search_skipped_when_term_matched():
    """术语命中 → 不触发 web search（即使其他条件满足）。"""
    terms = [TermEntry(source="凌霄破", target="Lingxiao Strike")]
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, llm = _make_pipeline(
        terms=terms,
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("XX·凌霄破", enable_web_search=True))
    assert web.calls == []
    user_msg = llm.calls[0][1]["content"]
    assert "外部网络参考" not in user_msg


def test_web_search_skipped_when_sparse_hits_present():
    """sparse 有命中 → 不触发 web search。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, _ = _make_pipeline(
        diagnostics=RAGDiagnostics(dense_top1=0.4, sparse_hits=2),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert web.calls == []


def test_web_search_skipped_when_dense_above_threshold():
    """dense_top1 ≥ 阈值 → 不触发 web search。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, _ = _make_pipeline(
        diagnostics=RAGDiagnostics(dense_top1=0.8, sparse_hits=0),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert web.calls == []


def test_web_search_skipped_when_request_flag_false():
    """请求级 enable_web_search=False → 即使三条件全真也不触发。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, _ = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    result = asyncio.run(pipe.translate_single("某条新词", enable_web_search=False))
    assert web.calls == []
    assert result.web_search_triggered is None
    assert result.web_references is None


def test_web_search_skipped_when_service_disabled():
    """web_search_svc.enabled=False → 不触发。"""
    web = FakeWebSearch(results=_web_results(), enabled=False)
    pipe, _, _, _ = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert web.calls == []


def test_web_search_skipped_when_service_not_injected():
    """web_search_svc=None → 不触发，行为等同未启用。"""
    pipe, _, _, _ = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=None,
        llm_response_fn=lambda m: "ok",
    )
    result = asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert result.status == "success"
    assert result.web_references is None


def test_web_search_failure_silently_degrades():
    """Web 搜索抛异常 → 翻译仍成功，web_references 为空。"""
    web = FakeWebSearch(raise_exc=RuntimeError("bocha exploded"))
    pipe, _, _, llm = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "translated",
    )
    result = asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert result.status == "success"
    assert result.web_search_triggered is True   # 试图触发了
    assert result.web_references is None          # 但没有结果（异常）
    user_msg = llm.calls[0][1]["content"]
    assert "外部网络参考" not in user_msg


def test_web_search_uses_search_with_diagnostics_path():
    """启用 web search 时 RAG 必须走 search_with_diagnostics 而不是普通 search。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, rag, _ = _make_pipeline(
        diagnostics=RAGDiagnostics(dense_top1=0.9, sparse_hits=2),  # 不触发但走 diagnostics
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    assert len(rag.diag_calls) == 1
    assert web.calls == []   # 条件不满足，没触发


def test_web_search_group_path_uses_single_representative_query():
    """整组路径只对代表句发一次 web 搜索，结果整组共享。"""
    web = FakeWebSearch(results=_web_results())
    response = "1. T1\n2. T2\n3. T3"
    pipe, _, _, llm = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: response,
    )
    results = asyncio.run(
        pipe.translate_group(["新词A", "新词B", "新词C"], enable_web_search=True)
    )
    assert len(results) == 3
    assert len(web.calls) == 1  # 一次代表句调用，整组共享
    user_msg = llm.calls[0][1]["content"]
    assert "外部网络参考" in user_msg
    # 整组每个结果都带 web_references
    assert all(r.web_search_triggered is True for r in results)
    assert all(r.web_references for r in results)


def test_web_search_dialog_path_uses_single_representative_query():
    """对话路径同样只调一次 web 搜索。"""
    web = FakeWebSearch(results=_web_results())
    pipe, _, _, llm = _make_pipeline(
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "1. A\n2. B\n3. C",
    )
    results = asyncio.run(
        pipe.translate_dialog(["新词1", "新词2", "新词3"], ["甲", "乙", "甲"], enable_web_search=True)
    )
    assert len(results) == 3
    assert len(web.calls) == 1
    user_msg = llm.calls[0][1]["content"]
    assert "外部网络参考" in user_msg
    assert all(r.web_search_triggered is True for r in results)


def test_web_search_prompt_section_ordering():
    """术语 / RAG / web 三段并存时，web 段必须排在术语和 RAG 之后。"""
    # 术语会让 _should_trigger_web_search 短路 → 用 dense_top1<0.6 + 无术语
    web = FakeWebSearch(results=_web_results())
    refs = [RAGSearchResult(source="某句", target="some line", score=0.55)]
    pipe, _, _, llm = _make_pipeline(
        rag_results=refs,
        diagnostics=_weak_recall_diag(),
        web_search=web,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(pipe.translate_single("某条新词", enable_web_search=True))
    user_msg = llm.calls[0][1]["content"]
    # RAG 段应该在 web 段之前
    assert user_msg.index("参考翻译") < user_msg.index("外部网络参考")
    # web 段在待翻译文本之前
    assert user_msg.index("外部网络参考") < user_msg.index("待翻译文本")


def test_web_search_custom_threshold_overrides_default():
    """请求级 web_search_dense_threshold 可覆盖默认值。"""
    web = FakeWebSearch(results=_web_results())
    # dense_top1=0.5；默认阈值 0.6 → 触发；但若请求设 0.4 → 不触发
    pipe, _, _, _ = _make_pipeline(
        diagnostics=RAGDiagnostics(dense_top1=0.5, sparse_hits=0),
        web_search=web,
        web_search_dense_threshold=0.6,
        llm_response_fn=lambda m: "ok",
    )
    asyncio.run(
        pipe.translate_single("某条新词", enable_web_search=True, web_search_dense_threshold=0.4)
    )
    assert web.calls == []   # 阈值放宽到 0.4，0.5 ≥ 0.4 → 不触发
