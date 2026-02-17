from __future__ import annotations

import os
import tomllib
from pathlib import Path

try:
    from subgen.cli import (  # type: ignore
        BACKEND_DEFAULTS as BACKEND_DEFAULTS,
        DEFAULT_TRANSLATE_CONFIG_PATH as DEFAULT_TRANSLATE_CONFIG_PATH,
        resolve_translation_settings as resolve_translation_settings,
    )
except Exception:
    BACKEND_DEFAULTS = {
        "local": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "env_key": None,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-4.1-mini",
            "env_key": "OPENAI_API_KEY",
        },
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
    }
    DEFAULT_TRANSLATE_CONFIG_PATH = (
        Path(__file__).resolve().parents[2] / "config" / "translation.toml"
    )

    def resolve_translation_settings(
        backend: str,
        model_name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        config_path: Path | None = None,
    ) -> dict[str, str]:
        if backend not in BACKEND_DEFAULTS:
            raise RuntimeError(f"Unsupported backend: {backend}")

        defaults = BACKEND_DEFAULTS[backend]
        resolved_model = model_name or defaults["model"]
        resolved_base_url = base_url or defaults["base_url"]
        resolved_api_key = api_key

        if backend in {"openai", "deepseek"}:
            cfg_path = (config_path or DEFAULT_TRANSLATE_CONFIG_PATH).expanduser().resolve()
            if not cfg_path.exists():
                raise RuntimeError(
                    f"Translation config not found: {cfg_path}. Please create it first."
                )
            try:
                cfg_data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
            except Exception as exc:
                raise RuntimeError(f"Failed to parse translation config: {exc}") from exc

            section = cfg_data.get(backend, {})
            if isinstance(section, dict):
                resolved_model = model_name or str(section.get("model") or resolved_model)
                resolved_base_url = base_url or str(section.get("base_url") or resolved_base_url)
                resolved_api_key = api_key or str(section.get("api_key") or "")

            env_key = defaults.get("env_key")
            if not resolved_api_key and env_key:
                resolved_api_key = os.getenv(env_key) or ""
            if not resolved_api_key:
                raise RuntimeError(
                    f"{backend} backend requires api_key in config or environment variable."
                )
        else:
            resolved_api_key = api_key or "ollama"

        return {
            "backend": backend,
            "model": str(resolved_model),
            "base_url": str(resolved_base_url),
            "api_key": str(resolved_api_key),
        }

