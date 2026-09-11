import json
import unittest

from app.services.translation_output import parse_numbered_output, parse_single_llm_output


class TranslationOutputTests(unittest.TestCase):
    def test_fenced_json_keeps_quotes_tags_and_newlines(self):
        translation = 'Known as "#YDarkened Brow#E".\nOnly Lone Cloud can help.'
        payload = json.dumps({"translation": translation, "reason": "保留段落及标签"}, ensure_ascii=False)
        self.assertEqual(parse_single_llm_output("```json\n" + payload + "\n```"), (translation, "保留段落及标签"))

    def test_literal_newline_in_json_does_not_truncate_escaped_quote(self):
        raw = '{"translation": "First line.\nCalled \\"Shield\\" in combat.", "reason": "保留引号"}'
        self.assertEqual(parse_single_llm_output(raw), ('First line.\nCalled "Shield" in combat.', "保留引号"))

    def test_malformed_unescaped_quotes_keep_remainder(self):
        raw = '{"translation": "Use "Shield" now.\\nThen retreat.", "reason": "技能名为 "Shield""}'
        self.assertEqual(parse_single_llm_output(raw), ('Use "Shield" now.\nThen retreat.', '技能名为 "Shield"'))

    def test_numbered_rows_decode_escaped_content(self):
        raw = '1. {"translation": "Use \\"#YShield#E\\".", "reason": "技能"}\n2. Plain text'
        self.assertEqual(parse_numbered_output(raw, 2), [('Use "#YShield#E".', "技能"), ("Plain text", None)])

    def test_plain_text_is_preserved(self):
        self.assertEqual(parse_single_llm_output('Use {skill_name}.'), ('Use {skill_name}.', None))


if __name__ == "__main__":
    unittest.main()
