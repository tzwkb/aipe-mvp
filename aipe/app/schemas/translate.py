from pydantic import BaseModel, Field


class TranslateRequest(BaseModel):
    texts: list[str] = Field(..., description="待翻译文本列表")
    source_lang: str = Field("zh", description="源语言")
    target_lang: str = Field("en", description="目标语言")
    batch_size: int = Field(50, ge=1, le=500, description="每批次处理句数")
    enable_rag: bool = Field(True, description="是否启用 RAG")
    rag_threshold: float = Field(0.85, ge=0.0, le=1.0, description="RAG 相似度阈值")
    rag_top_k: int = Field(3, ge=1, le=10, description="RAG 返回 Top-K")
    rag_collection: str | None = Field(None, description="指定 RAG 检索的 Qdrant collection 名，不填则用默认值")
    project_id: str | None = Field(None, description="项目档案 ID，如 wwm/zh-en；不填使用默认项目或旧全局状态")
    task_id: str | None = Field(None, description="任务ID（断点续传用）")
    content_types: list[str | None] | None = Field(
        None, description="每句对应的文本类型（与 texts 一一对应），有则跳过 LLM 预分类"
    )
    enable_web_search: bool = Field(
        False,
        description=(
            "启用外部网络搜索补充（博查）。仅在术语 0 命中且 RAG 弱召回（dense top1 < 阈值"
            "且 sparse 无命中）时触发；需服务端 WEB_SEARCH_ENABLED=true 才生效。"
        ),
    )
    web_search_dense_threshold: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="本次请求触发 Web 搜索的 dense top1 阈值；不填用配置默认 0.6",
    )
    enable_vision: bool = Field(
        True,
        description=(
            "是否启用多模态模型（qwen-plus）对 Web 搜索配图进行分析，提供 image_analysis。"
            "仅在 enable_web_search=True 且触发了 Web 搜索时生效；关闭后 image_analysis 始终为 None。"
        ),
    )
    use_tm_exact_match: bool = Field(
        False,
        description=(
            "是否直接采用 TM 精确源文匹配结果。开启后，如果 RAG/TM collection 中存在 source 完全相同的条目，"
            "直接使用该 target 作为译文并跳过 LLM 翻译；未命中则照常翻译。"
        ),
    )


class TranslationResult(BaseModel):
    source: str = Field(..., description="原文")
    translation: str = Field(..., description="译文")
    translation_reason: str | None = Field(None, description="LLM 翻译理由（术语/RAG/网络参考的采用决策说明）")
    status: str = Field(..., description="success | error")
    content_type: str | None = Field(None, description="AI 预分类的文本类型")
    terminology_used: list[dict] = Field(default_factory=list, description="命中的术语列表")
    rag_references: list[dict] | None = Field(None, description="RAG 参考例句")
    web_references: list[dict] | None = Field(
        None,
        description="外部网络参考（博查搜索结果）；未触发或被屏蔽时为 None",
    )
    web_search_triggered: bool | None = Field(
        None,
        description="是否实际触发了 Web 搜索（含 cache 命中）。None=未启用 web search",
    )
    image_analysis: str | None = Field(
        None,
        description="多模态 LLM 对 Web 搜索配图的分析结果；未触发 web search 或无配图时为 None",
    )
    tm_exact_match_used: bool = Field(False, description="是否直接采用了 TM 精确源文匹配结果")
    tm_exact_match_source: str | None = Field(None, description="被直接采用的 TM 原文")
    tm_exact_match_target: str | None = Field(None, description="被直接采用的 TM 译文")
    tm_exact_match_status: str | None = Field(None, description="被直接采用的 TM 审校状态")
    tm_exact_match_score: float | None = Field(None, description="TM 精确匹配分数；精确源文命中固定为 1.0")
    error_msg: str | None = Field(None, description="错误信息")


class BatchTranslateResponse(BaseModel):
    task_id: str = Field(..., description="任务唯一ID")
    total: int = Field(..., description="总句数")
    completed: int = Field(..., description="已完成句数")
    results: list[TranslationResult] = Field(default_factory=list, description="翻译结果列表")
    progress_pct: float = Field(0.0, description="进度百分比")
    status: str = Field(..., description="running | completed | resumed")
