from pydantic import BaseModel, Field


class TermEntry(BaseModel):
    source: str = Field(..., description="中文原文")
    target: str = Field(..., description="英文译文")
    category: str | None = Field(None, description="类型（角色名/地名/物品/技能...）")
    notes: str | None = Field(None, description="备注")


class TerminologyUploadResponse(BaseModel):
    total: int = Field(..., description="术语总数")
    added: int = Field(..., description="新增条数")
    updated: int = Field(..., description="更新条数")
    message: str = Field("ok", description="状态信息")


class TerminologyListResponse(BaseModel):
    total: int
    items: list[TermEntry]
