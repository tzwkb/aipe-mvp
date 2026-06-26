from pydantic import BaseModel, Field

# 优先级越小 = 质量越高；未知状态统一 99
PRIORITY_RANK: dict[str, int] = {
    "Designer Reviewed": 1,
    "CQA_Done": 2,
    "Done_LQA edited": 3,
    "Done": 4,
}

# 排序权重：adjusted_score = raw_score × weight
# 高质量语料在分数接近时自然排到前面，但不会完全压制高相似度的低质量条目
PRIORITY_WEIGHT: dict[str, float] = {
    "Designer Reviewed": 1.00,
    "CQA_Done": 0.97,
    "Done_LQA edited": 0.94,
    "Done": 0.90,
}
_DEFAULT_WEIGHT = 0.85


def status_to_rank(status: str | None) -> int:
    if not status:
        return 99
    return PRIORITY_RANK.get(status.strip(), 99)


def status_to_weight(status: str | None) -> float:
    if not status:
        return _DEFAULT_WEIGHT
    return PRIORITY_WEIGHT.get(status.strip(), _DEFAULT_WEIGHT)


class CorpusEntry(BaseModel):
    source: str = Field(..., description="中文原文")
    target: str = Field(..., description="英文译文")
    context: str | None = Field(None, description="上下文/场景说明")
    status: str | None = Field(None, description="审校状态，如 Designer Reviewed / CQA_Done 等")


class RAGSearchRequest(BaseModel):
    query: str = Field(..., description="检索原文")
    threshold: float = Field(0.85, ge=0.0, le=1.0)
    top_k: int = Field(3, ge=1, le=10)
    collection: str | None = Field(None, description="指定 Qdrant collection 名，不填则用默认值")


class RAGSearchResult(BaseModel):
    source: str = Field(..., description="参考原文")
    target: str = Field(..., description="参考译文")
    score: float = Field(..., description="相似度分数")
    status: str | None = Field(None, description="该条语料的审校状态")


class RAGSearchResponse(BaseModel):
    query: str
    total: int
    results: list[RAGSearchResult]


class CorpusUploadResponse(BaseModel):
    total: int = Field(..., description="本次上传总条数")
    indexed: int = Field(..., description="成功入库条数")
    message: str = Field("ok")
