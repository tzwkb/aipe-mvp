from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM (OpenAI-compatible)
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    llm_temperature: float = 0.3
    llm_timeout: float = 60.0
    llm_max_retries: int = 5

    # Embedding (OpenAI-compatible)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int | None = None  # Matryoshka 截断维度，None 表示使用模型默认
    embedding_api_key: str = ""
    embedding_base_url: str | None = None
    embedding_timeout: float = 30.0

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "yanyun_corpus"

    # Project profiles
    projects_dir: str = "./data/projects"
    default_project: str | None = None

    # RAG (hybrid: dense + BM25 + RRF)
    # rag_threshold 仅作用在 dense 路召回（RRF 融合分与 cosine 不可比，故不在融合后过滤）。
    rag_threshold: float = 0.5
    rag_top_k: int = 3
    rag_dense_prefetch: int = 20    # dense 召回送入 RRF 的候选数
    rag_sparse_prefetch: int = 20   # sparse(BM25) 召回送入 RRF 的候选数

    # Batch
    batch_size: int = 50
    max_concurrent: int = 5
    batch_sleep: float = 0.5

    # 结构聚类（整组翻译路径）：把句式高度相似的句子合并成一次 LLM 调用，
    # 强制使用同一英文模板，保证一致性。
    # cluster_enabled=False 时退回旧的"逐句翻译"模式。
    cluster_enabled: bool = True
    cluster_pair_threshold: float = 0.4      # 成对前后缀相似度合并阈值（粗筛）
    cluster_min_coverage: float = 0.5        # 整组公共前后缀覆盖率门槛（细筛，防假阳性）
    cluster_max_group_size: int = 10         # 单次整组 LLM 调用的最大句数

    # 外部 Web 搜索（博查 / Bocha）—— 在 RAG 弱召回 + 术语未命中时调用，
    # 把公开网页摘要作为低优先级参考段注入 prompt。
    # 默认关闭（web_search_enabled=False）；接口层还有 enable_web_search 二级开关。
    web_search_enabled: bool = False
    bocha_api_key: str = ""
    bocha_endpoint: str = "https://api.bocha.cn/v1/web-search"
    bocha_count: int = 8
    bocha_summary: bool = True
    bocha_timeout: float = 8.0
    bocha_max_retries: int = 1
    web_search_dense_threshold: float = 0.6   # dense top1 < 该值才视为弱召回
    web_search_max_snippets: int = 3          # 注入 prompt 的 snippet 上限
    web_search_snippet_max_chars: int = 300   # 单条 snippet 截断长度
    web_search_cache_dir: str = "./data/cache/web_search"
    web_search_cache_enabled: bool = True

    # 多模态视觉分析（用于 Web 搜索配图，复用 llm_api_key / llm_base_url）
    vision_model: str = "qwen3.6-plus"

    # Paths
    data_dir: str = "./data"
    progress_dir: str = "./data/progress"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
