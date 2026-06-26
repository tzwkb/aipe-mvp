"""文本处理工具。"""


def chunk(items: list, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("size must be > 0")
    return [items[i : i + size] for i in range(0, len(items), size)]
