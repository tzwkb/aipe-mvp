"""风格指南服务：加载 TXT/MD，注入 System Prompt。

设计参考技术设计文档 §5.4。风格指南作为 **System Prompt 级别规则** 注入每次 LLM 调用，
优先级高于术语表 / RAG 参考 / 待译文本。

支持来源：
- API 上传（``load`` 接收解析后的字符串）
- 启动期从 ``data/style_guide/`` 自动加载（``load_from_dir``）
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from app.errors import StyleGuideError

logger = logging.getLogger(__name__)


class ContentType(str, Enum):
    """风格指南中定义的 20 种游戏文本类型。"""
    UI = "UI文本"
    STORY = "剧情"
    LOCATION = "地区地点"
    CHARACTER = "角色/人物"
    FACTION = "势力"
    SKILL = "技能"
    APPEARANCE = "外观"
    APPEARANCE_DESCRIPTION = "外观描述"
    QUEST = "任务"
    QUEST_DESCRIPTION = "任务描述"
    QUEST_OBJECTIVE = "任务目标"
    ACHIEVEMENT = "成就"
    MARTIAL_RECORDS = "武林录"
    ENTITY = "entity相关"
    HINT = "提示文本"
    NOTIFICATION = "通知弹窗"
    ITEM = "道具"
    PRESCRIPTION = "药方"
    SPEECH = "话术"
    BESTIARY = "博物(图鉴)"
    UNKNOWN = "未知"


_TYPE_DESCRIPTIONS: dict[str, str] = {
    "UI文本": "界面按钮、菜单项、标签等极短文本（通常1-5个词）",
    "剧情": "故事叙述、角色对话、章节标题",
    "地区地点": "地名、区域、场景名称",
    "角色/人物": "NPC名字、玩家角色称谓",
    "势力": "帮派、门派、组织名称及介绍",
    "技能": "武功招式名称及效果描述",
    "外观": "服装、发型等外观配件名称",
    "外观描述": "外观/装扮道具的剧情化背景文案或题诗，讲述外观背后的来历、寄托或意境，带文学性与画面感（区别于简短的外观名称）",
    "任务": "任务名称、任务目标描述",
    "任务描述": "任务的剧情化引导/背景文案，带氛围感和叙事性，引导玩家进入任务情境（开场白、场景渲染、悬念铺陈等）",
    "任务目标": "任务追踪栏中的单步操作目标，高度功能化的玩家行动指令（如\"询问工匠为何叹气\"、\"去书生家看看\"），通常以动词开头、极简",
    "成就": "成就名称、达成条件说明",
    "武林录": "游戏百科、见闻录、知识条目内容",
    "entity相关": "可交互物体名称、操作按钮提示",
    "提示文本": "游戏内操作帮助提示",
    "通知弹窗": "系统弹窗、奖励通知消息",
    "道具": "道具、物品名称及描述",
    "药方": "丹药方剂名称及疗效描述",
    "话术": "市井NPC随机闲聊对话",
    "博物(图鉴)": "图鉴条目名称及收藏品描述",
}

# 匹配风格指南中的类型分节标题，如：#             **一、**     **UI文本**
_SECTION_SPLIT_RE = re.compile(
    r'(?m)^#+\s+\*\*[一二三四五六七八九十]+、\*\*\s+\*\*[^*\n]+\*\*'
)

_VALID_TYPE_VALUES: frozenset[str] = frozenset(ct.value for ct in ContentType)


_DEFAULT_SYSTEM_PROMPT = (
    "你是一名专业的游戏本地化译者，负责将中文游戏文本翻译为英文。\n"
    "你必须严格遵守以下规则。"
)

_SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


class StyleGuideService:
    """全局单例，启动时按需 ``load`` 整份风格指南。"""

    def __init__(self) -> None:
        self._rules: str = ""
        self._filename: str | None = None
        self._loaded_at: datetime | None = None
        self._general_rules: str = ""
        self._type_sections: dict[str, str] = {}

    # ---------- load ----------

    def load(self, content: str, filename: str | None = None) -> None:
        """加载已解析的风格指南正文。再次调用会整体覆盖（MVP 不支持增量）。"""
        text = content.strip()
        if not text:
            raise StyleGuideError("风格指南内容为空")
        self._rules = text
        self._filename = filename
        self._loaded_at = datetime.now(timezone.utc)
        self._general_rules, self._type_sections = self._parse_sections(text)
        logger.info(
            "风格指南加载完成: filename=%s chars=%d lines=%d",
            filename,
            len(text),
            text.count("\n") + 1,
        )

    def load_from_dir(self, dir_path: str | Path) -> bool:
        """启动期便利方法：从目录中挑选第一个 .md/.txt 文件加载。

        返回是否加载成功。目录不存在或无支持文件时返回 False，不抛异常，
        以保证服务可在没有风格指南的情况下启动。
        """
        d = Path(dir_path)
        if not d.exists() or not d.is_dir():
            return False
        # 优先 .md，其次 .markdown，再次 .txt；文件名升序，结果稳定。
        candidates: list[Path] = []
        for suffix in (".md", ".markdown", ".txt"):
            candidates.extend(sorted(d.glob(f"*{suffix}")))
        if not candidates:
            return False

        path = candidates[0]
        try:
            raw = path.read_bytes()
            text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n").strip()
        except (OSError, UnicodeDecodeError) as exc:
            logger.warning("启动期加载风格指南失败 %s: %s", path, exc)
            return False
        if not text:
            return False

        self.load(text, filename=path.name)
        return True

    def clear(self) -> None:
        self._rules = ""
        self._filename = None
        self._loaded_at = None
        self._general_rules = ""
        self._type_sections = {}

    # ---------- 查询 ----------

    @property
    def loaded(self) -> bool:
        return bool(self._rules)

    @property
    def filename(self) -> str | None:
        return self._filename

    @property
    def loaded_at(self) -> datetime | None:
        return self._loaded_at

    def get_rules(self) -> str:
        """返回风格指南正文（未加载时为空字符串）。"""
        return self._rules

    def info(self) -> dict[str, object]:
        """对外只读快照，用于上传响应 / 健康检查。"""
        return {
            "loaded": self.loaded,
            "filename": self._filename,
            "char_count": len(self._rules),
            "line_count": self._rules.count("\n") + 1 if self._rules else 0,
            "loaded_at": self._loaded_at.isoformat() if self._loaded_at else None,
        }

    # ---------- 类型分区解析 ----------

    @staticmethod
    def _parse_sections(text: str) -> tuple[str, dict[str, str]]:
        """将风格指南拆分为通用规则 + 各类型专属规则段。"""
        marker = "分类别规范细则"
        idx = text.find(marker)
        if idx == -1:
            return text, {}

        line_start = text.rfind("\n", 0, idx)
        general_rules = text[: line_start if line_start != -1 else idx].strip()
        type_block = text[idx:]

        sections: dict[str, str] = {}
        matches = list(_SECTION_SPLIT_RE.finditer(type_block))
        for i, m in enumerate(matches):
            # 从 header 行中提取类型名（末尾 **名称** 部分）
            header_line = type_block[m.start() : m.end()]
            name_match = re.search(r'\*\*([^*\n]+)\*\*\s*$', header_line)
            if not name_match:
                continue
            # 归一化：去除斜杠前后的多余空格（如 "角色 /人物" → "角色/人物"）
            name = re.sub(r'\s*/\s*', '/', name_match.group(1).strip())
            end = matches[i + 1].start() if i + 1 < len(matches) else len(type_block)
            sections[name] = type_block[m.start() : end].strip()

        logger.debug("风格指南解析完成: 通用规则 %d chars，类型分区 %s", len(general_rules), list(sections.keys()))
        return general_rules, sections

    def get_general_rules(self) -> str:
        return self._general_rules

    def get_type_rules(self, content_type: ContentType) -> str:
        return self._type_sections.get(content_type.value, "")

    def build_classification_messages(self, text: str) -> list[dict[str, str]]:
        """构建用于文本类型分类的 LLM messages。"""
        type_list = "\n".join(
            f"- {name}：{desc}" for name, desc in _TYPE_DESCRIPTIONS.items()
        )
        return [
            {
                "role": "system",
                "content": (
                    "你是一名游戏文本分类助手。"
                    "根据提供的游戏文本，从列表中选择最匹配的类型，仅输出该类型名称，不要输出任何解释。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"类型列表：\n{type_list}\n\n"
                    f"待分类文本：\n{text}\n\n"
                    "请输出类型名称："
                ),
            },
        ]

    # ---------- Prompt 注入 ----------

    def build_system_prompt(self, base: str | None = None) -> str:
        """组装 System Prompt：基础角色描述 + 完整风格指南（类型未知时的兜底）。"""
        head = (base or _DEFAULT_SYSTEM_PROMPT).rstrip()
        if not self._rules:
            return head
        return f"{head}\n\n## 风格指南\n{self._rules}"

    def build_system_prompt_for_type(
        self, base: str | None, content_type: ContentType
    ) -> str:
        """组装 System Prompt：通用规则 + 当前文本类型的专属规则段。

        比 ``build_system_prompt`` 更精准：LLM 只需关注与当前文本类型相关的规则，
        减少噪声。类型未知或解析失败时退化为完整风格指南注入。
        """
        head = (base or _DEFAULT_SYSTEM_PROMPT).rstrip()
        if not self._rules:
            return head

        if content_type == ContentType.UNKNOWN or not self._type_sections:
            return f"{head}\n\n## 风格指南\n{self._rules}"

        type_rules = self._type_sections.get(content_type.value, "")
        if not type_rules:
            return f"{head}\n\n## 风格指南\n{self._rules}"

        general = self._general_rules or self._rules
        return (
            f"{head}\n\n"
            f"## 风格指南（通用规则）\n{general}\n\n"
            f"## 当前文本类型：{content_type.value}\n{type_rules}"
        )


# ---------- 模块级单例 ----------

_singleton: StyleGuideService | None = None


def get_style_guide_service() -> StyleGuideService:
    global _singleton
    if _singleton is None:
        _singleton = StyleGuideService()
    return _singleton


def reset_style_guide_service() -> None:
    """测试用：清空单例状态。"""
    global _singleton
    _singleton = None


__all__ = [
    "ContentType",
    "StyleGuideService",
    "StyleGuideError",
    "get_style_guide_service",
    "reset_style_guide_service",
]
