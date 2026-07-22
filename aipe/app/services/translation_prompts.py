"""Pure prompt construction for translation flows."""

from __future__ import annotations

import re

from app.schemas.rag import RAGSearchResult
from app.schemas.terminology import TermEntry
from app.schemas.translate import TranslationResult
from app.schemas.web_search import WebSearchResult
from app.services.project_service import ProjectProfile
from app.services.style_guide_service import (
    ContentType,
    ContentTypeMode,
    StyleGuideService,
    content_type_mode,
)
from app.services.translation_memory import format_locked_tm_section


BASE_SYSTEM = (
    "你是一名专业的游戏本地化译者，负责按项目语言方向翻译游戏文本。\n"
    "请遵守以下要求：\n\n"
    "## 翻译要求\n"
    "1. 术语参考段按文本功能使用：UI、任务、道具、技能、系统规则等功能性文本优先保持术语一致；剧情、口语、邮件、外观描述等叙事性文本应结合语境取舍，不要把术语表当作逐字替换表\n"
    "2. RAG 参考例句反映游戏内同类语境的真实译法，请结合相似度分数与上下文综合参考\n"
    "3. 遵守项目背景、风格指南和文本功能要求，保持目标语自然准确\n"
    "4. 按照下方输出格式要求输出，不要输出格式外的额外内容\n"
    "5. 外部网络参考段（如有）来自公开网络搜索，优先级最低；只用来理解专有名词的背景，与术语 / RAG 冲突时一律以术语 / RAG 为准\n\n"
    "## 富文本与变量标记规则\n"
    "待翻译文本中可能出现以下特殊标记，请按规则处理，确保输出中标记位置正确：\n\n"
    "1. **字体格式标签**（非打印字符）：`#G......#E`、`#C......#E`、`#Y......#E` 等用于定义二者之间的文字格式（如颜色）。\n"
    "   译文需要保留这些标签，并将对应的译文放在标签之间（即 `#G` 与 `#E` 之间）。\n\n"
    "2. **通用文本变量**（打印字符）：`{}` 为空占位符，翻译时将其视为实际文本内容的一部分处理，确保位置正确。\n\n"
    "3. **具名文本变量**（打印字符）：如 `{slot1_qishu_name}`、`{kungfu_main_name}` 等。\n"
    "   这些是真实文本的占位符，译文需考虑其实际含义，将其放在语法和语义都恰当的位置。\n\n"
    "4. **数值标量**（打印字符）：如 `{standard_value}`、`{qishu_standard_value}` 等。\n"
    "   这些是实际数值的占位符，译文需根据语境判断其在句子中的合理位置并正确放置。\n"
    "5. <x id=\"64\"/> 这样的没有实际意义的原文标签应该完全保留\n"
)

_WEB_SECTION_HEADER = (
    "## 外部网络参考（可能不准 — 仅作背景理解，不要直接照抄措辞）\n"
    "以下内容来自公开网络搜索，可能含错误、口语化译法或与本游戏无关的同名条目。\n"
    "请把它当作\"了解专有名词的语境\"用，绝不能凌驾于术语参考和 RAG 参考之上。"
)

_TERM_SECTION_TITLE = (
    "## 术语参考（按文本功能使用；功能/UI/道具/任务类保持一致，口语/剧情/邮件类按语境取舍）"
)

_LANGUAGE_NAMES = {
    "ZH": "Chinese",
    "ZHCN": "Chinese",
    "ZH-CN": "Chinese",
    "CN": "Chinese",
    "EN": "English",
    "ENUS": "English",
    "EN-US": "English",
    "DE": "German",
    "DEDE": "German",
    "DE-DE": "German",
    "FR": "French",
    "FRFR": "French",
    "FR-FR": "French",
    "JA": "Japanese",
    "JP": "Japanese",
    "RU": "Russian",
    "PT": "Portuguese",
    "ES": "Spanish",
}


def build_single_messages(
    source: str,
    term_matches: list[TermEntry],
    references: list[RAGSearchResult],
    content_type: ContentType,
    *,
    style_guide_svc: StyleGuideService,
    base_system: str,
    target_lang: str,
    web_refs: list[WebSearchResult] | None = None,
) -> list[dict[str, str]]:
    system = style_guide_svc.build_system_prompt_for_type(base_system, content_type)
    parts = _reference_sections(term_matches, references, web_refs)
    parts.extend(
        [
            format_feedback_rules(content_type, target_lang),
            "## 待翻译文本",
            source,
            (
                "\n请以 JSON 格式输出（仅输出 JSON，不含任何其他文字或 markdown）：\n"
                '{"translation": "目标语译文", "reason": "用中文简述翻译理由，说明采用了哪些术语/RAG例句/网络资料，'
                '或关键语境判断（1-2句话）"}'
            ),
        ]
    )
    return _messages(system, parts)


def build_group_messages(
    sources: list[str],
    term_matches: list[TermEntry],
    references: list[RAGSearchResult],
    content_type: ContentType,
    template_hint: str,
    *,
    style_guide_svc: StyleGuideService,
    base_system: str,
    target_lang: str,
    web_refs: list[WebSearchResult] | None = None,
    locked_tm_results: list[TranslationResult | None] | None = None,
) -> list[dict[str, str]]:
    system = style_guide_svc.build_system_prompt_for_type(base_system, content_type)
    intro_lines = [
        f"以下 {len(sources)} 句源文本结构高度一致，请使用**完全一致的目标语句式**翻译。",
        "仅替换变量部分；句式骨架（前后缀、连接词、标点风格、大小写）必须 1:1 对应，"
        "确保整组译文模板严格统一。",
    ]
    if template_hint:
        intro_lines.append(f"识别出的共有结构提示：`{template_hint}`")

    parts = ["## 整组翻译任务", "\n".join(intro_lines)]
    parts.extend(_reference_sections(term_matches, references, web_refs))
    locked_section = format_locked_tm_section(locked_tm_results or [])
    if locked_section:
        parts.append(locked_section)
    parts.extend(
        [
            format_feedback_rules(content_type, target_lang),
            "## 源文本",
            "\n".join(f"{index}. {source}" for index, source in enumerate(sources, 1)),
            (
                "## 输出格式（严格遵守）\n"
                '逐行输出，每行格式为 `序号. {"translation": "目标语译文", "reason": "简述理由（中文，1句话）"}`，'
                "序号与源文本编号严格对应，"
                f"共 {len(sources)} 行。不要输出任何额外解释、标题或 Markdown。"
            ),
        ]
    )
    return _messages(system, parts)


def build_dialog_messages(
    sources: list[str],
    speakers: list[str | None],
    term_matches: list[TermEntry],
    references: list[RAGSearchResult],
    content_type: ContentType,
    dialog_id: str | None,
    *,
    style_guide_svc: StyleGuideService,
    base_system: str,
    target_lang: str,
    web_refs: list[WebSearchResult] | None = None,
    locked_tm_results: list[TranslationResult | None] | None = None,
) -> list[dict[str, str]]:
    system = style_guide_svc.build_system_prompt_for_type(base_system, content_type)
    header = [f"以下是一段游戏内连续对话，共 {len(sources)} 句，按对话发生顺序列出。"]
    if dialog_id:
        header.append(f"对话编号：{dialog_id}。")
    header.extend(
        [
            "请结合上下文（说话人身份、上下句衔接、语气、信息流向）翻译每一句，",
            "确保整段译文在目标语中读起来像一段连贯、自然的对话。",
            "说话人姓名仅供你理解语境使用，**不要**在译文里输出说话人名字，",
            "也无需输出说话人姓名。",
        ]
    )

    parts = ["## 对话翻译任务", "\n".join(header)]
    parts.extend(_reference_sections(term_matches, references, web_refs))
    locked_section = format_locked_tm_section(locked_tm_results or [])
    if locked_section:
        parts.append(locked_section)
    parts.extend(
        [
            format_feedback_rules(content_type, target_lang),
            "## 对话内容",
            _format_dialog_lines(sources, speakers),
            (
                "## 输出格式（严格遵守）\n"
                '逐行输出，每行格式为 `序号. {"translation": "目标语译文", "reason": "简述理由（中文，1句话）"}`，'
                "序号与对话编号严格对应，"
                f"共 {len(sources)} 行。不要重复输出说话人姓名，不要输出原文，"
                "不要加任何解释、标题或 Markdown。"
            ),
        ]
    )
    return _messages(system, parts)


def base_system_for_project(profile: ProjectProfile, prompt_notes: str = "") -> str:
    parts = [BASE_SYSTEM.rstrip()]
    project_lines: list[str] = []
    if profile.game:
        project_lines.append(f"项目：{profile.game}")
    if profile.language_pair:
        project_lines.append(f"语言方向：{profile.language_pair}")
    source_lang = _language_name(profile.source_lang) if profile.source_lang else None
    target_lang = _language_name(profile.target_lang) if profile.target_lang else None
    if (not source_lang or not target_lang) and profile.language_pair:
        source_lang, target_lang = _language_pair_names(profile.language_pair)
    if source_lang and target_lang:
        project_lines.extend(
            [
                f"源语言：{source_lang}",
                f"目标语言：{target_lang}",
                f"请将源语言文本翻译为{target_lang}。",
            ]
        )
    if profile.background:
        project_lines.append(f"背景与语气：{profile.background}")
    if project_lines:
        parts.append("## 项目背景\n" + "\n".join(project_lines))
    if prompt_notes:
        parts.append("## 项目补充提示\n" + prompt_notes)
    return "\n\n".join(parts)


def format_feedback_rules(content_type: ContentType, target_lang: str = "en") -> str:
    """Render persistent LQE feedback rules for the target language and text mode."""
    english_target = _is_english_target(target_lang)
    lines = ["## AIPE 反馈优化规则"]
    if english_target:
        lines.append(
            "- 标点与英文习惯：不要使用 em dash；中文省略号要改为英文 `...` 或重组短句；英文中使用 straight punctuation。"
        )
    lines.extend(
        [
            "- 语言自然度：不要机械贴源语言语序；避免重复句式和重复用词；可在不改意义的前提下调整主谓宾、连接词和信息顺序。",
            "- 术语策略：术语/RAG 用来帮助判断专名和统一译法，不要把术语表当作逐字替换表；若术语在当前语境里只是普通含义、自称或泛称，应按目标语言自然表达处理。",
            (
                "- 人称与称谓：自称、昵称、拟声化自指、玩家称呼要先判断说话关系；"
                "自称优先转为第一人称，面向玩家的称呼可转为 you/your。"
                if english_target
                else "- 人称与称谓：自称、昵称、拟声化自指、玩家称呼要先判断说话关系，并按目标语言的人称习惯处理。"
            ),
            "- 古典/文言感文本：先理解实词虚词和语义关系，再翻译；不要逐字对应，必要时在 reason 里说明不确定点。",
        ]
    )

    mode = content_type_mode(content_type)
    if mode == ContentTypeMode.FUNCTIONAL:
        lines.extend(
            [
                "- 当前文本按功能性文本处理：术语一致性优先，任务/UI/道具/技能/规则说明需稳定复用术语和高分 RAG。",
                "- 功能性文本的译文应句式短、清楚、可执行；避免过度文学化或随意改写。",
            ]
        )
    elif mode == ContentTypeMode.CONTEXTUAL:
        lines.extend(
            [
                "- 当前文本按口语/剧情/邮件类文本处理：语气、说话人身份和上下文优先于机械术语套用。",
                (
                    "- 口语和 playful/casual 文本应自然、轻快，可使用 contraction 和常见英语口语表达；避免商务正式腔。"
                    if english_target
                    else "- 口语和轻松文本应符合目标语言的自然表达与语域，避免不必要的正式腔。"
                ),
            ]
        )
    else:
        lines.extend(
            [
                "- 当前文本类型不够明确：先判断它更像功能性说明还是叙事/口语，再决定术语强度。",
                "- 若是系统操作、任务、道具、奖励，偏严格；若是对白、邮件、叙述、背景文案，偏自然。",
            ]
        )
    return "\n".join(lines)


def _reference_sections(
    term_matches: list[TermEntry],
    references: list[RAGSearchResult],
    web_refs: list[WebSearchResult] | None,
) -> list[str]:
    parts: list[str] = []
    if term_matches:
        parts.extend([_TERM_SECTION_TITLE, _format_term_section(term_matches)])
    if references:
        parts.extend(
            [
                "## 参考翻译（从游戏语料库 RAG 检索到的相似句子，含相似度分数）",
                _format_refs_section(references),
            ]
        )
    if web_refs:
        parts.extend([_WEB_SECTION_HEADER, _format_web_section(web_refs)])
    return parts


def _messages(system: str, parts: list[str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def _format_term_section(term_matches: list[TermEntry]) -> str:
    lines: list[str] = []
    for entry in term_matches:
        category = f"（{entry.category}）" if entry.category else ""
        lines.append(f'- 「{entry.source}」{category} → "{entry.target}"')
    return "\n".join(lines)


def _format_dialog_lines(sources: list[str], speakers: list[str | None]) -> str:
    lines: list[str] = []
    for index, source in enumerate(sources, 1):
        speaker = (speakers[index - 1] or "").strip()
        lines.append(f"{index}. [{speaker}] {source}" if speaker else f"{index}. {source}")
    return "\n".join(lines)


def _format_refs_section(references: list[RAGSearchResult]) -> str:
    return "\n".join(
        f'{index}. [score={ref.score:.3f}] 「{ref.source}」 → "{ref.target}"'
        for index, ref in enumerate(references, 1)
    )


def _format_web_section(web_refs: list[WebSearchResult]) -> str:
    blocks: list[str] = []
    for index, ref in enumerate(web_refs, 1):
        site = f" — {ref.site_name}" if ref.site_name else ""
        lines = [
            f"{index}. 「{ref.title}」{site}",
            f"   摘要：{ref.snippet}",
            f"   链接：{ref.url}",
        ]
        if ref.image_analysis:
            lines.append(f"   配图视觉分析（多模态AI）：{ref.image_analysis}")
        elif ref.image_url:
            lines.append(f"   配图：{ref.image_url}")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _is_english_target(target_lang: str) -> bool:
    normalized = (target_lang or "").strip().lower().replace("_", "-")
    return normalized in {"en", "english", "en-us", "en-gb"} or normalized.startswith("en-")


def _language_pair_names(language_pair: str) -> tuple[str | None, str | None]:
    parts = [part for part in re.split(r"[-_>/→]+", language_pair.strip()) if part]
    if len(parts) < 2:
        return None, None
    return _language_name(parts[0]), _language_name(parts[-1])


def _language_name(code: str) -> str:
    normalized = re.sub(r"[^A-Za-z-]", "", code).upper()
    return _LANGUAGE_NAMES.get(normalized, code.strip())


__all__ = [
    "BASE_SYSTEM",
    "base_system_for_project",
    "build_dialog_messages",
    "build_group_messages",
    "build_single_messages",
    "format_feedback_rules",
]
