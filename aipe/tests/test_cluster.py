"""结构聚类单测：n-gram Jaccard 聚类 + 公共前后缀覆盖率验证。"""

from __future__ import annotations

from app.utils.cluster import (
    affix_coverage_ratio,
    char_ngrams,
    cluster_by_structure,
    common_prefix_len,
    common_suffix_len,
    common_template_repr,
    jaccard,
)


def test_char_ngrams_basic():
    assert char_ngrams("abcd", n=3) == {"abc", "bcd"}
    # 短于 n 的串返回单元素集合
    assert char_ngrams("ab", n=3) == {"ab"}
    # 空 / 空白
    assert char_ngrams("") == set()
    assert char_ngrams("   ") == set()


def test_jaccard_edge_cases():
    assert jaccard(set(), set()) == 0.0
    assert jaccard({"a"}, set()) == 0.0
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"a", "c"}) == 1 / 3


def test_affix_coverage_event_pattern():
    """活动【X】获取 模式：前缀 3 + 后缀 3 / 10 = 60%"""
    strs = [
        "活动【柿业有成】获取",
        "活动【燕衔嘉礼】获取",
        "活动【聆音响岁】获取",
        "活动【踏岳寻珍】获取",
    ]
    assert common_prefix_len(strs) == 3  # "活动【"
    assert common_suffix_len(strs) == 3  # "】获取"
    assert abs(affix_coverage_ratio(strs) - 0.6) < 1e-9


def test_affix_coverage_no_slot_marker():
    """无固定槽位标记，仅靠前缀+后缀也能识别"""
    strs = [
        "完成天赋·首领中青挑战任务获得",
        "完成天赋·首领中打更人挑战任务获得",
        "完成天赋·首领中柏楚玉挑战任务获得",
        "完成天赋·首领中司南剑客挑战任务获得",
    ]
    assert common_prefix_len(strs) == 8  # "完成天赋·首领中"
    assert common_suffix_len(strs) == 6  # "挑战任务获得"
    assert affix_coverage_ratio(strs) > 0.7


def test_affix_coverage_unrelated_returns_zero():
    strs = ["我去打架了", "明天会下雨吗", "完全不同的句子"]
    assert affix_coverage_ratio(strs) == 0.0


def test_affix_overlap_does_not_double_count():
    """前后缀加起来超过最短串长度时，应避免重叠双算。"""
    strs = ["abc", "abc", "abc"]
    # 前缀=3, 后缀=3, 但句子只有 3 char。覆盖率应为 1.0 而不是 2.0。
    assert affix_coverage_ratio(strs) == 1.0


def test_cluster_event_pattern():
    texts = [
        "活动【柿业有成】获取",
        "活动【燕衔嘉礼】获取",
        "活动【聆音响岁】获取",
        "活动【踏岳寻珍】获取",
        "完全无关的另一句话",
    ]
    res = cluster_by_structure(texts)
    assert len(res.groups) == 1
    assert res.groups[0] == [0, 1, 2, 3]
    assert res.singletons == [4]


def test_cluster_boss_challenge_pattern():
    texts = [
        "完成天赋·首领中青挑战任务获得",
        "完成天赋·首领中打更人挑战任务获得",
        "完成天赋·首领中柏楚玉挑战任务获得",
        "完成天赋·首领中司南剑客挑战任务获得",
    ]
    res = cluster_by_structure(texts)
    assert len(res.groups) == 1
    assert res.groups[0] == [0, 1, 2, 3]
    assert res.singletons == []


def test_cluster_two_independent_groups():
    texts = [
        "活动【A】获取",
        "活动【B】获取",
        "完成天赋·首领中X挑战任务获得",
        "完成天赋·首领中Y挑战任务获得",
    ]
    res = cluster_by_structure(texts)
    assert len(res.groups) == 2
    # 按首成员索引排序，确定可重现
    assert res.groups[0] == [0, 1]
    assert res.groups[1] == [2, 3]


def test_cluster_max_group_size_splits():
    texts = [f"活动【item{i:02d}】获取" for i in range(12)]
    res = cluster_by_structure(texts, max_group_size=5)
    assert len(res.groups) == 3
    assert [len(g) for g in res.groups] == [5, 5, 2]
    # 切片保留原始顺序
    assert res.groups[0] == [0, 1, 2, 3, 4]
    assert res.groups[1] == [5, 6, 7, 8, 9]
    assert res.groups[2] == [10, 11]


def test_cluster_all_singletons_when_no_pattern():
    texts = ["第一种说法", "另外一种内容", "完全不同的东西"]
    res = cluster_by_structure(texts)
    assert res.groups == []
    assert sorted(res.singletons) == [0, 1, 2]


def test_cluster_empty_input():
    assert cluster_by_structure([]).groups == []
    assert cluster_by_structure([]).singletons == []


def test_cluster_single_input():
    res = cluster_by_structure(["只有这一句"])
    assert res.groups == []
    assert res.singletons == [0]


def test_cluster_min_coverage_rejects_false_positives():
    """min_coverage 提高时，前后缀骨架不够强的句对应被拦截。"""
    texts = [
        # 前缀公共 5 字 "前后都不同" + 后缀公共 3 字 "在那边"/"在这里" 不同
        "前后都不同我去打架了在那边",
        "前后都不同他去看书时在这里",
    ]
    # pair_threshold 较低能合并，但提高 min_coverage 后整组覆盖率不够
    res = cluster_by_structure(texts, pair_threshold=0.3, min_coverage=0.9)
    assert res.groups == []
    assert sorted(res.singletons) == [0, 1]


def test_common_template_repr_event_pattern():
    strs = ["活动【A】获取", "活动【B】获取"]
    assert common_template_repr(strs) == "活动【…】获取"


def test_common_template_repr_no_affix_returns_empty():
    assert common_template_repr(["甲乙丙", "戊己庚"]) == ""
