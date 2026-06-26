from pydantic import BaseModel, Field


class StyleGuideUploadResponse(BaseModel):
    filename: str = Field(..., description="原始文件名")
    char_count: int = Field(..., description="规则字符数（去除两端空白后）")
    line_count: int = Field(..., description="规则行数")
    message: str = Field("ok", description="状态信息")


class StyleGuideInfoResponse(BaseModel):
    loaded: bool = Field(..., description="是否已加载")
    filename: str | None = Field(None, description="当前已加载的文件名")
    char_count: int = Field(0, description="规则字符数")
    line_count: int = Field(0, description="规则行数")
    preview: str = Field("", description="规则前 500 字预览")
    rules: str | None = Field(None, description="完整规则正文（按需返回）")
