"""文件解析工具测试：待译文本上传解析，含 content_type 列提取。"""

from __future__ import annotations

import io

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
