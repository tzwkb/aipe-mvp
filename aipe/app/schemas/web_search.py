"""Web 搜索结果 schema（博查 API 解析后的标准化形态）。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class WebSearchResult(BaseModel):
    """单条网络补充上下文。

    `image_url` 仅挂在第一条结果上（Bocha 返回的 images 顶图）。
    `image_analysis` 是多模态 LLM 对该图片的分析结果，由 VisionService 填充。
    """

    title: str = Field(..., description="网页标题，来自 webPages.value[].name")
    snippet: str = Field(..., description="摘要；summary=True 时为长摘要")
    url: str = Field(..., description="网页 URL")
    site_name: str | None = Field(None, description="站点名（如 央广网 / 搜狐网）")
    image_url: str | None = Field(
        None,
        description="顶部图片直链 contentUrl；通常只挂在第一条结果上",
    )
    image_analysis: str | None = Field(
        None,
        description="多模态 LLM（qwen3.6-plus）对配图的分析结果，用于向翻译 AI 注入视觉语境",
    )


__all__ = ["WebSearchResult"]
