from typing import Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_core import PydanticCustomError

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
    collection: str | None = Field(None, description="仅用于旧版兼容；与 scope 互斥")
    scope: Literal["global", "project", "special"] | None = Field(
        None,
        description="检索库作用域；不填时保留旧版 collection 语义",
    )
    project_id: str | None = Field(None, description="scope=project 时必填")

    @model_validator(mode="after")
    def validate_scope_contract(self) -> "RAGSearchRequest":
        if self.scope is not None and self.collection is not None:
            raise PydanticCustomError(
                "rag_scope_collection_conflict",
                "collection 仅适用于未传 scope 的旧版请求",
            )
        has_project_id = bool(self.project_id and self.project_id.strip())
        if self.scope == "project" and not has_project_id:
            raise PydanticCustomError(
                "rag_project_required",
                "scope=project 时必须传 project_id",
            )
        if self.scope != "project" and self.project_id is not None:
            raise PydanticCustomError(
                "rag_project_scope_conflict",
                "project_id 仅适用于 scope=project",
            )
        return self


class RAGSearchResult(BaseModel):
    source: str = Field(..., description="参考原文")
    target: str = Field(..., description="参考译文")
    score: float = Field(..., description="相似度分数")
    status: str | None = Field(None, description="该条语料的审校状态")


class RAGSearchResponse(BaseModel):
    query: str
    scope: Literal["global", "project", "special"] | None = None
    project_id: str | None = None
    total: int
    results: list[RAGSearchResult]


class CorpusUploadResponse(BaseModel):
    total: int = Field(..., description="本次上传总条数")
    indexed: int = Field(..., description="成功入库条数")
    message: str = Field("ok")
