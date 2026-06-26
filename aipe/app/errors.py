"""共享业务异常。

放在独立模块避免 ``app.main`` ↔ ``app.services`` 的循环导入。
"""


class TranslationError(Exception):
    """翻译流程业务异常（LLM 调用失败、术语校验失败等）。"""


class TerminologyError(Exception):
    """术语表加载 / 解析失败。"""


class StyleGuideError(Exception):
    """风格指南加载 / 解析失败。"""
