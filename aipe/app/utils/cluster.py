"""结构聚类：将句式高度相似的源文本聚成组，供整组翻译路径使用。

设计目标：
- 不依赖固定占位符 / 槽位标记，纯靠字符级相似度发现"同模板"句子
- 输入完全为中文 / 中英混排都适用；不引入新依赖（纯标准库）

算法：
1. 两两计算"公共前缀 + 公共后缀"对较长句长的占比作为成对相似度
   （比 n-gram Jaccard 更直接：模板句的"骨架"恰好集中在前后缀，
   变量部分长短不影响判定；不会被高占比变量稀释）
2. 成对相似度 ≥ ``pair_threshold`` 的句对走并查集合并
3. 对每个 cluster size ≥ 2 的组，整组再算"公共前后缀覆盖率"，
   ≥ ``min_coverage`` 才视为同模板组，否则拆回单句（防 union-find 链式假阳性）
4. 模板组按 ``max_group_size`` 切片输出（避免单次 LLM 调用 prompt 过长）

复杂度：O(N² · L)，L 为平均句长。早终止：长度差比例不足时直接跳过。
N≈几千 内毫秒级；更大规模再考虑 MinHash/LSH 预筛。

``char_ngrams`` / ``jaccard`` 作为通用工具仍对外暴露，便于其他组件复用。
"""

from __future__ import annotations

from dataclasses import dataclass


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """字符级 n-gram 集合。短于 n 的串原样返回单元素集合。"""
    text = (text or "").strip()
    if not text:
        return set()
    if len(text) < n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / (len(a) + len(b) - inter)


def common_prefix_len(strs: list[str]) -> int:
    if not strs:
        return 0
    n = 0
    for chars in zip(*strs):
        if len(set(chars)) == 1:
            n += 1
        else:
            break
    return n


def common_suffix_len(strs: list[str]) -> int:
    if not strs:
        return 0
    return common_prefix_len([s[::-1] for s in strs])


def affix_coverage_ratio(strs: list[str]) -> float:
    """min over members of (公共前缀长度 + 公共后缀长度) / 自身长度。

    若前后缀加起来超过最短成员长度，把后缀截短避免重叠双算。
    """
    if not strs:
        return 0.0
    cp = common_prefix_len(strs)
    cs = common_suffix_len(strs)
    min_len = min(len(s) for s in strs)
    if cp + cs > min_len:
        cs = max(0, min_len - cp)
    return min((cp + cs) / max(1, len(s)) for s in strs)


def common_template_repr(strs: list[str], placeholder: str = "…") -> str:
    """从一组同模板句子提取展示用的"前缀…后缀"形式，仅用于日志/prompt 提示。"""
    if not strs:
        return ""
    cp = common_prefix_len(strs)
    cs = common_suffix_len(strs)
    min_len = min(len(s) for s in strs)
    if cp + cs > min_len:
        cs = max(0, min_len - cp)
    first = strs[0]
    prefix = first[:cp]
    suffix = first[len(first) - cs :] if cs > 0 else ""
    if not prefix and not suffix:
        return ""
    return f"{prefix}{placeholder}{suffix}"


@dataclass(frozen=True)
class ClusterResult:
    """聚类结果。

    - ``groups``：每个元素为一组同模板句子的"原始索引列表"，组内顺序保留原顺序；
      已按 ``max_group_size`` 切片，可直接作为一次 LLM 整组调用的输入
    - ``singletons``：未进入任何模板组的句子的原始索引（保留原顺序）
    """

    groups: list[list[int]]
    singletons: list[int]


def _pairwise_affix_similarity(a: str, b: str) -> float:
    """成对相似度：(公共前缀长度 + 公共后缀长度) / max(|a|, |b|)。

    模板句 ``PREFIX + VAR + SUFFIX`` 的骨架集中在前后缀，
    用这个指标比 n-gram Jaccard 更稳定，且不受变量长度影响。
    """
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    cp = 0
    for x, y in zip(a, b):
        if x == y:
            cp += 1
        else:
            break
    cs = 0
    # 反向遍历比 reversed() 快，避免构造新字符串
    i, j = la - 1, lb - 1
    while i >= cp and j >= cp and a[i] == b[j]:
        cs += 1
        i -= 1
        j -= 1
    short_len = min(la, lb)
    if cp + cs > short_len:
        cs = max(0, short_len - cp)
    return (cp + cs) / max(la, lb)


def cluster_by_structure(
    texts: list[str],
    *,
    pair_threshold: float = 0.4,
    min_coverage: float = 0.5,
    max_group_size: int = 10,
) -> ClusterResult:
    """对输入文本做结构聚类。空输入返回空结果。

    参数：
    - ``pair_threshold``：成对前后缀相似度合并阈值（粗筛，建议 0.3~0.5）
    - ``min_coverage``：成组后的整体公共前后缀覆盖率门槛（细筛，防 union-find 链式假阳性）
    - ``max_group_size``：单个模板组的最大成员数；超出会切成多组
    """
    n = len(texts)
    if n == 0:
        return ClusterResult(groups=[], singletons=[])
    if n == 1:
        return ClusterResult(groups=[], singletons=[0])

    stripped = [(t or "").strip() for t in texts]

    parent = list(range(n))

    def find(x: int) -> int:
        # 路径压缩
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    lens = [len(s) for s in stripped]
    for i in range(n):
        if lens[i] == 0:
            continue
        for j in range(i + 1, n):
            if lens[j] == 0:
                continue
            # 长度差悬殊时成对相似度不可能达标，早终止节省 O(L)
            if min(lens[i], lens[j]) / max(lens[i], lens[j]) < pair_threshold:
                continue
            if _pairwise_affix_similarity(stripped[i], stripped[j]) >= pair_threshold:
                union(i, j)

    raw: dict[int, list[int]] = {}
    for i in range(n):
        raw.setdefault(find(i), []).append(i)

    groups: list[list[int]] = []
    singletons: list[int] = []
    # 按首个成员的原始索引排序，使输出确定可重现
    for members in sorted(raw.values(), key=lambda m: m[0]):
        if len(members) == 1:
            singletons.append(members[0])
            continue
        member_strs = [stripped[i] for i in members]
        if affix_coverage_ratio(member_strs) < min_coverage:
            # union-find 链式合并可能把不同模板拉到一起，拆回单句
            singletons.extend(members)
            continue
        # 切片，每组不超过 max_group_size
        for start in range(0, len(members), max_group_size):
            groups.append(members[start : start + max_group_size])

    singletons.sort()
    return ClusterResult(groups=groups, singletons=singletons)


__all__ = [
    "ClusterResult",
    "affix_coverage_ratio",
    "char_ngrams",
    "cluster_by_structure",
    "common_prefix_len",
    "common_suffix_len",
    "common_template_repr",
    "jaccard",
]
