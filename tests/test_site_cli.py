"""
单元测试 - site_cli.py 核心函数
"""
import pytest
import sys
from pathlib import Path

# 添加 src 到路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from webvidgrab.site_cli import (
    _sanitize_filename_stem,
    _extract_page_title,
    _candidate_score,
    _strip_ansi,
    _output_template,
)


class TestSanitizeFilenameStem:
    """测试文件名清理函数"""

    def test_basic_clean(self):
        assert _sanitize_filename_stem("Hello World") == "Hello World"

    def test_remove_special_chars(self):
        assert _sanitize_filename_stem("test/file:name") == "test_file_name"

    def test_remove_invalid_chars(self):
        result = _sanitize_filename_stem('test<>?"*:|\\')
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result

    def test_trim_spaces(self):
        assert _sanitize_filename_stem("  trimmed  ") == "trimmed"

    def test_normalize_spaces(self):
        assert _sanitize_filename_stem("multiple   spaces") == "multiple spaces"

    def test_max_length(self):
        long_name = "a" * 300
        result = _sanitize_filename_stem(long_name)
        assert len(result) <= 180

    def test_empty_input(self):
        assert _sanitize_filename_stem("") == ""
        assert _sanitize_filename_stem("   ") == ""


class TestExtractPageTitle:
    """测试页面标题提取函数"""

    def test_og_title_meta(self):
        html = '<meta property="og:title" content="Test Video Title">'
        assert _extract_page_title(html) == "Test Video Title"

    def test_og_title_meta_reversed(self):
        html = '<meta content="Test Video Title" property="og:title">'
        assert _extract_page_title(html) == "Test Video Title"

    def test_title_tag(self):
        html = "<title>Page Title Here</title>"
        assert _extract_page_title(html) == "Page Title Here"

    def test_og_title_priority(self):
        html = """
        <title>Fallback Title</title>
        <meta property="og:title" content="OG Title">
        """
        assert _extract_page_title(html) == "OG Title"

    def test_no_title(self):
        assert _extract_page_title("<html><body>No title</body></html>") is None

    def test_sanitize_title(self):
        html = '<title>Invalid: File/Name</title>'
        result = _extract_page_title(html)
        assert "/" not in result
        assert ":" not in result


class TestCandidateScore:
    """测试候选 URL 评分函数"""

    def test_m3u8_highest(self):
        assert _candidate_score("https://example.com/video.m3u8") > 150

    def test_mpd_high(self):
        assert _candidate_score("https://example.com/video.mpd") > 150

    def test_mp4_medium(self):
        assert _candidate_score("https://example.com/video.mp4") >= 100

    def test_resolution_bonus(self):
        score_1080 = _candidate_score("https://example.com/1080p/video.m3u8")
        score_720 = _candidate_score("https://example.com/720p/video.m3u8")
        assert score_1080 > score_720

    def test_manifest_bonus(self):
        score_manifest = _candidate_score("https://example.com/manifest.m3u8")
        score_normal = _candidate_score("https://example.com/video.m3u8")
        assert score_manifest > score_normal

    def test_unknown_format(self):
        assert _candidate_score("https://example.com/page.html") == 0


class TestStripAnsi:
    """测试 ANSI 转义码清除函数"""

    def test_plain_text(self):
        assert _strip_ansi("Hello World") == "Hello World"

    def test_with_color_codes(self):
        text = "\033[31mRed Text\033[0m"
        assert _strip_ansi(text) == "Red Text"

    def test_multiple_codes(self):
        text = "\033[1;32mGreen Bold\033[0m and \033[34mBlue\033[0m"
        assert _strip_ansi(text) == "Green Bold and Blue"


class TestOutputTemplate:
    """测试输出模板函数"""

    def test_with_title(self):
        result = _output_template("My Video")
        assert "My Video" in result

    def test_without_title(self):
        # 当 title 为 None 时，模板仍然返回相同的格式（由 yt-dlp 处理）
        result = _output_template(None)
        # 验证返回的是有效的 yt-dlp 输出模板
        assert "%(title)s" in result or "%(id)s" in result

    def test_contains_date_placeholder(self):
        result = _output_template("Test")
        assert "%(ext)s" in result or "{ext}" in result or "download" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
