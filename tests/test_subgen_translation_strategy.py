"""
单元测试 - subgen 翻译分层回退策略
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from subgen import cli
from subgen.cli import SubtitleEntry, TranslationBackendError


def _entries() -> list[SubtitleEntry]:
    return [
        SubtitleEntry(index=1, start=0.0, end=1.0, text="hello there"),
        SubtitleEntry(index=2, start=1.0, end=2.0, text="general kenobi"),
    ]


class TestTranslationHttpFailureClassification:
    def test_classifies_input_inspection(self):
        body = '{"error":{"message":"Input data may contain inappropriate content","code":"data_inspection_failed"}}'
        assert cli._classify_translation_http_failure(400, body) == "input_inspection"

    def test_classifies_output_inspection(self):
        body = '{"error":{"message":"Output data may contain inappropriate content","code":"data_inspection_failed"}}'
        assert cli._classify_translation_http_failure(400, body) == "output_inspection"

    def test_classifies_other_http(self):
        assert cli._classify_translation_http_failure(500, "boom") == "http"


class TestMarkedTranslationParsing:
    def test_merges_continuation_lines(self):
        text = """
        [0001] 第一行
        续写部分
        [0002] 第二行
        """
        parsed = cli._parse_marked_translation_lines(text)
        assert parsed == {1: "第一行 续写部分", 2: "第二行"}


class TestLayeredFallbackStrategy:
    def test_falls_back_from_raw_to_b64(self, monkeypatch):
        calls: list[str] = []

        def fake_translate_chunk(entries, *, start, end, model_name, base_url, api_key, mode):
            calls.append(mode)
            if mode == "raw":
                raise TranslationBackendError("input_inspection", "blocked")
            if mode == "b64":
                return {1: "你好", 2: "将军"}
            raise AssertionError(f"unexpected mode: {mode}")

        monkeypatch.setattr(cli, "_translate_marked_chunk_mode", fake_translate_chunk)

        result = cli.translate_entries_layered_resilient(
            _entries(),
            model_name="qwen3-max",
            max_tokens=4000,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
            api_key="test-key",
        )

        assert result == ["你好", "将军"]
        assert calls == ["raw", "b64"]

    def test_uses_google_after_model_modes_exhausted(self, monkeypatch):
        calls: list[str] = []

        def fake_translate_chunk(entries, *, start, end, model_name, base_url, api_key, mode):
            calls.append(mode)
            raise TranslationBackendError("output_inspection", "blocked")

        def fake_google(entries, *, start, end):
            assert (start, end) == (0, 2)
            return {1: "谷歌译文一", 2: "谷歌译文二"}

        monkeypatch.setattr(cli, "_translate_marked_chunk_mode", fake_translate_chunk)
        monkeypatch.setattr(cli, "_google_translate_chunk", fake_google)

        result = cli.translate_entries_layered_resilient(
            _entries(),
            model_name="qwen3-max",
            max_tokens=4000,
            base_url="https://coding.dashscope.aliyuncs.com/v1",
            api_key="test-key",
        )

        assert result == ["谷歌译文一", "谷歌译文二"]
        assert calls == ["raw", "b64", "b64_soft"]

    def test_does_not_hide_hard_http_errors(self, monkeypatch):
        def fake_translate_chunk(entries, *, start, end, model_name, base_url, api_key, mode):
            raise TranslationBackendError("http", "bad gateway", status_code=502)

        monkeypatch.setattr(cli, "_translate_marked_chunk_mode", fake_translate_chunk)

        with pytest.raises(TranslationBackendError) as exc_info:
            cli.translate_entries_layered_resilient(
                _entries(),
                model_name="qwen3-max",
                max_tokens=4000,
                base_url="https://coding.dashscope.aliyuncs.com/v1",
                api_key="test-key",
            )

        assert exc_info.value.kind == "http"


class TestQwenRemoteDetection:
    def test_detects_dashscope_backend(self):
        assert cli._is_qwen_remote_backend(
            "qwen3-max-2026-01-23",
            "https://coding.dashscope.aliyuncs.com/v1",
        )

    def test_ignores_local_ollama_qwen(self):
        assert not cli._is_qwen_remote_backend(
            "qwen2.5:7b",
            "http://localhost:11434/v1",
        )

    def test_does_not_treat_minimax_as_qwen(self):
        assert not cli._is_qwen_remote_backend(
            "MiniMax/MiniMax-M2.7",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


class TestBailianRemoteDetection:
    def test_detects_minimax_on_bailian(self):
        assert cli._is_bailian_remote_backend(
            "MiniMax/MiniMax-M2.7",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

    def test_ignores_local_model_named_minimax(self):
        assert not cli._is_bailian_remote_backend(
            "minimax-local",
            "http://localhost:11434/v1",
        )


class TestTranslationSettings:
    def test_resolves_minimax_from_config_and_env(self, monkeypatch, tmp_path):
        cfg_path = tmp_path / "translation.toml"
        cfg_path.write_text(
            (
                "[minimax]\n"
                'base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"\n'
                'model = "MiniMax/MiniMax-M2.7"\n'
                'api_key = ""\n'
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("OPENCLAW_BAILIAN_API_KEY", "env-bailian-key")

        settings = cli.resolve_translation_settings(
            backend="minimax",
            config_path=cfg_path,
        )

        assert settings == {
            "backend": "minimax",
            "model": "MiniMax/MiniMax-M2.7",
            "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "api_key": "env-bailian-key",
        }
