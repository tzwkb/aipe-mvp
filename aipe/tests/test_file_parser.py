"""文件解析工具测试：待译文本上传解析，含 content_type 列提取。"""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest

from app.utils.file_parser import parse_text_bytes


class TestParseTextBytes:
    """parse_text_bytes 返回 ``[{source, content_type}, ...]`` 的测试。"""

    def test_txt_no_content_type(self):
        data = "你好\n世界\n".encode("utf-8")
        result = parse_text_bytes(data, "test.txt")
        assert result == [
            {"source": "你好", "content_type": None},
            {"source": "世界", "content_type": None},
        ]

    def test_csv_with_content_type_column(self):
        df = pd.DataFrame(
            {
                "中文": ["少侠", "恭桶"],
                "文本类型": ["UI文本", "道具"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "test.csv")
        assert result == [
            {"source": "少侠", "content_type": "UI文本"},
            {"source": "恭桶", "content_type": "道具"},
        ]

    def test_csv_with_type_alias(self):
        """content_type 列支持别名：类型 / type / category 等。"""
        df = pd.DataFrame(
            {
                "source": ["a", "b"],
                "category": ["剧情", "UI文本"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "test.csv")
        assert result == [
            {"source": "a", "content_type": "剧情"},
            {"source": "b", "content_type": "UI文本"},
        ]

    def test_csv_no_content_type_column(self):
        """没有 content_type 列时，所有 content_type 为 None。"""
        df = pd.DataFrame({"source": ["a", "b"]})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "test.csv")
        assert result == [
            {"source": "a", "content_type": None},
            {"source": "b", "content_type": None},
        ]

    def test_xlsx_with_content_type(self):
        df = pd.DataFrame(
            {
                "中文": ["活动A", "活动B"],
                "类型": ["通知弹窗", "任务"],
            }
        )
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        result = parse_text_bytes(buf.getvalue(), "test.xlsx")
        assert result == [
            {"source": "活动A", "content_type": "通知弹窗"},
            {"source": "活动B", "content_type": "任务"},
        ]

    def test_json_string_array(self):
        data = b'["a", "b"]'
        result = parse_text_bytes(data, "test.json")
        assert result == [
            {"source": "a", "content_type": None},
            {"source": "b", "content_type": None},
        ]

    def test_json_object_array_with_content_type(self):
        data = '[{"source": "a", "content_type": "UI文本"}, {"source": "b", "type": "剧情"}]'.encode("utf-8")
        result = parse_text_bytes(data, "test.json")
        assert result == [
            {"source": "a", "content_type": "UI文本"},
            {"source": "b", "content_type": "剧情"},
        ]

    def test_json_object_array_no_type(self):
        data = b'[{"source": "a"}, {"src": "b"}]'
        result = parse_text_bytes(data, "test.json")
        assert result == [
            {"source": "a", "content_type": None},
            {"source": "b", "content_type": None},
        ]

    def test_empty_rows_skipped(self):
        df = pd.DataFrame(
            {
                "source": ["a", None, "", "b"],
                "类型": ["UI", None, "", "剧情"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "test.csv")
        assert result == [
            {"source": "a", "content_type": "UI"},
            {"source": "b", "content_type": "剧情"},
        ]


class TestParseDialogColumns:
    """xlsx/csv 同时携带 id/说话人/time 列时，应附加 dialog_id / speaker / time 字段。"""

    def test_xlsx_with_dialog_columns(self):
        df = pd.DataFrame(
            {
                "id": ["23600058", "23600058", "23600058"],
                "说话人": ["若萍", "玩家", "若萍"],
                "time": [None, 5.0, 11.0],
                "原文": ["这……这是我那本册子！", "栀婷可是出了什么事？", "当年她出宫时……"],
                "content_type": ["剧情", "剧情", "剧情"],
            }
        )
        buf = io.BytesIO()
        df.to_excel(buf, index=False)
        result = parse_text_bytes(buf.getvalue(), "对话.xlsx")
        assert len(result) == 3
        assert result[0] == {
            "source": "这……这是我那本册子！",
            "content_type": "剧情",
            "dialog_id": "23600058",
            "speaker": "若萍",
            "time": None,
        }
        assert result[1]["time"] == 5.0
        assert result[1]["speaker"] == "玩家"
        assert result[2]["time"] == 11.0

    def test_csv_without_dialog_columns_unchanged(self):
        """无 dialog 列时返回结构保持向后兼容，不带 dialog_id/speaker/time。"""
        df = pd.DataFrame({"source": ["a", "b"], "类型": ["UI", "剧情"]})
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "test.csv")
        for entry in result:
            assert "dialog_id" not in entry
            assert "speaker" not in entry
            assert "time" not in entry

    def test_content_type_does_not_steal_speaker_column(self):
        """`name` 同时在 speaker 别名和 (无) content_type 中：speaker 应取走它，
        content_type 不该误读说话人名字。"""
        df = pd.DataFrame(
            {
                "id": ["1", "1"],
                "name": ["甲", "乙"],
                "原文": ["你好", "再见"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "t.csv")
        assert result[0]["speaker"] == "甲"
        assert result[0]["content_type"] is None
        assert result[1]["speaker"] == "乙"

    def test_time_unparseable_becomes_none(self):
        df = pd.DataFrame(
            {
                "id": ["x", "x"],
                "说话人": ["A", "B"],
                "time": ["abc", "3.5"],
                "原文": ["s1", "s2"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "t.csv")
        assert result[0]["time"] is None
        assert result[1]["time"] == 3.5

    def test_json_object_array_with_dialog_fields(self):
        data = (
            '[{"source": "你好", "id": "x1", "speaker": "甲", "time": 1.0},'
            ' {"source": "再见", "id": "x1", "speaker": "乙", "time": 2.5}]'
        ).encode("utf-8")
        result = parse_text_bytes(data, "test.json")
        assert result == [
            {
                "source": "你好",
                "content_type": None,
                "dialog_id": "x1",
                "speaker": "甲",
                "time": 1.0,
            },
            {
                "source": "再见",
                "content_type": None,
                "dialog_id": "x1",
                "speaker": "乙",
                "time": 2.5,
            },
        ]

    def test_csv_preserves_expanded_dialog_context_columns(self):
        df = pd.DataFrame(
            {
                "id": ["scene-1", "scene-1"],
                "角色信息 无需本地化": ["奥黛丽", "阿尔杰"],
                "受话人": ["阿尔杰", "奥黛丽"],
                "场景ID": ["chapter34", "chapter34"],
                "关系阶段": ["founding_period", "founding_period"],
                "场景语气": ["guarded", "guarded"],
                "参考信息": ["交易开始", "交付配方"],
                "原文": ["我已经带来了。", "这是配方。"],
            }
        )
        buf = io.StringIO()
        df.to_csv(buf, index=False)

        result = parse_text_bytes(buf.getvalue().encode("utf-8"), "dialog.csv")

        assert result[0] == {
            "source": "我已经带来了。",
            "content_type": None,
            "dialog_id": "scene-1",
            "speaker": "奥黛丽",
            "addressee": "阿尔杰",
            "scene_id": "chapter34",
            "relationship_stage": "founding_period",
            "scene_tone": "guarded",
            "context_note": "交易开始",
        }

    def test_json_addressee_list_is_normalized_for_entity_matching(self):
        data = json.dumps(
            [
                {
                    "source": "诸位，请听我说。",
                    "id": "scene-1",
                    "speaker": "克莱恩",
                    "addressee_ids": ["奥黛丽", "阿尔杰"],
                    "scene_id": "chapter34",
                }
            ],
            ensure_ascii=False,
        ).encode("utf-8")

        result = parse_text_bytes(data, "dialog.json")

        assert result[0]["addressee"] == "奥黛丽 | 阿尔杰"
        assert result[0]["scene_id"] == "chapter34"


def test_project_workbook_layout_selects_sheet_ranges_and_dialog_metadata():
    cover = pd.DataFrame([["说明页"], ["不要作为待译文本"]])
    trial = pd.DataFrame([[None] * 4 for _ in range(10)])
    trial.iat[3, 0] = "灵界裂隙"
    trial.iat[3, 2] = "玩法"
    trial.iat[5, 0] = "奥黛丽"
    trial.iat[5, 1] = "下午好，愚者先生。"
    trial.iat[5, 2] = "塔罗会交易开始"
    trial.iat[6, 0] = "阿尔杰"
    trial.iat[6, 1] = "这是配方。"
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        cover.to_excel(writer, sheet_name="说明", header=False, index=False)
        trial.to_excel(writer, sheet_name="试译", header=False, index=False)

    content_scope = {
        "workbook_layout": {
            "sheet": "试译",
            "sections": [
                {
                    "id": "functional",
                    "row_start": 4,
                    "row_end": 4,
                    "source_column": "A",
                    "content_type_column": "C",
                    "context_note_column": "C",
                },
                {
                    "id": "dialogue",
                    "row_start": 6,
                    "row_end": 7,
                    "source_column": "B",
                    "speaker_column": "A",
                    "context_note_column": "C",
                    "content_type": "剧情",
                    "dialog_id": "scene.demo",
                    "scene_id": "scene.demo",
                    "relationship_stage": "early",
                    "scene_tone": "guarded",
                    "use_row_as_time": True,
                },
            ],
        }
    }

    result = parse_text_bytes(
        buf.getvalue(),
        "trial.xlsx",
        content_scope=content_scope,
    )

    assert result == [
        {
            "source": "灵界裂隙",
            "content_type": "玩法",
            "context_note": "玩法",
        },
        {
            "source": "下午好，愚者先生。",
            "content_type": "剧情",
            "dialog_id": "scene.demo",
            "speaker": "奥黛丽",
            "scene_id": "scene.demo",
            "relationship_stage": "early",
            "scene_tone": "guarded",
            "context_note": "塔罗会交易开始",
            "time": 6.0,
        },
        {
            "source": "这是配方。",
            "content_type": "剧情",
            "dialog_id": "scene.demo",
            "speaker": "阿尔杰",
            "scene_id": "scene.demo",
            "relationship_stage": "early",
            "scene_tone": "guarded",
            "context_note": None,
            "time": 7.0,
        },
    ]
