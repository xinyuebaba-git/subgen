from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest
from typing import Any, Callable, Sequence

from tqdm import tqdm


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".m4v",
    ".webm",
    ".flv",
    ".wmv",
}

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

DEEPGRAM_ASR_MODELS = [
    "nova-3",
    "nova-2",
    "enhanced",
    "base",
]
_LAST_DEEPGRAM_DETECTED_LANGUAGE = ""
_LAST_DEEPGRAM_LANGUAGE_CONFIDENCE = 0.0

QWEN25_OLLAMA_TRANSLATION_OPTIONS: dict[str, Any] = {
    # Based on Qwen docs baseline (temperature/top_k/top_p/repeat_penalty) with
    # subtitle-translation-oriented constraints for stability and long context.
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "repeat_penalty": 1.05,
    "num_ctx": 32768,
    "num_predict": 2048,
}

DOLPHIN3_OLLAMA_TRANSLATION_OPTIONS: dict[str, Any] = {
    # More conservative decoding for subtitle translation stability.
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_ctx": 32768,
    "num_predict": 2048,
}
STRICT_RETRY_MAX_ATTEMPTS = 3

DEFAULT_TRANSLATE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "translation.toml"
)
DEFAULT_ASR_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "asr.toml"
)


@dataclass
class SubtitleEntry:
    index: int
    start: float
    end: float
    text: str


ProgressCallback = Callable[[str, float, float, str | None], None]


def _progress_desc(task: str, progress_label: str | None) -> str:
    if progress_label:
        return f"{task} [{progress_label}]"
    return task


def _make_progress_bar(
    *,
    task: str,
    total: float | int,
    unit: str,
    progress_label: str | None = None,
) -> tqdm:
    return tqdm(
        total=total,
        desc=_progress_desc(task, progress_label),
        unit=unit,
        disable=not sys.stderr.isatty(),
    )


def _emit_progress(
    callback: ProgressCallback | None,
    *,
    task: str,
    current: float,
    total: float,
    label: str | None,
) -> None:
    if callback is None:
        return
    callback(task, max(0.0, current), max(0.0, total), label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate source subtitles with local Whisper and optionally translate to Chinese."
        )
    )
    parser.add_argument("videos", nargs="+", help="One or more video file paths")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./subtitles"),
        help="Directory to write subtitle files",
    )
    parser.add_argument(
        "--source-language",
        default=None,
        help="Source language code for Whisper (default: auto detect)",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="ASR model name/path for whisper/faster-whisper",
    )
    parser.add_argument(
        "--asr-engine",
        choices=["whisper", "faster-whisper", "deepgram"],
        default="faster-whisper",
        help="ASR engine",
    )
    parser.add_argument(
        "--deepgram-api-key",
        default=None,
        help="Deepgram API key (or set DEEPGRAM_API_KEY)",
    )
    parser.add_argument(
        "--deepgram-model",
        default="nova-3",
        help="Deepgram ASR model name (e.g., nova-3)",
    )
    parser.add_argument(
        "--asr-config",
        type=Path,
        default=DEFAULT_ASR_CONFIG_PATH,
        help="Path to ASR provider config (for deepgram api_key/model)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="faster-whisper compute type (e.g., int8, float16)",
    )
    parser.add_argument(
        "--max-segment-duration",
        type=float,
        default=2.2,
        help="Max subtitle duration in seconds for tighter sync",
    )
    parser.add_argument(
        "--max-segment-chars",
        type=int,
        default=28,
        help="Max characters per subtitle line",
    )
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Only generate source subtitles and skip Chinese translation",
    )
    parser.add_argument(
        "--translate-model",
        default=None,
        help="Translation model name",
    )
    parser.add_argument(
        "--translate-backend",
        choices=["local", "openai", "deepseek"],
        default="local",
        help="Translation backend. local=Ollama, openai=OpenAI, deepseek=DeepSeek",
    )
    parser.add_argument(
        "--translate-base-url",
        default=None,
        help="Base URL override for translation backend",
    )
    parser.add_argument(
        "--translate-api-key",
        default=None,
        help="API key override for translation backend",
    )
    parser.add_argument(
        "--translate-config",
        type=Path,
        default=DEFAULT_TRANSLATE_CONFIG_PATH,
        help="Path to translation provider config (used by online backends)",
    )
    parser.add_argument(
        "--translation-max-tokens",
        type=int,
        default=4000,
        help="Approx max input tokens per translation request; subtitles are auto-split into multiple requests",
    )
    return parser.parse_args()


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
        "model": resolved_model,
        "base_url": resolved_base_url,
        "api_key": resolved_api_key,
    }


def _load_toml_config(path: Path) -> dict[str, Any]:
    p = path.expanduser().resolve()
    if not p.exists():
        return {}
    try:
        return tomllib.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def resolve_deepgram_settings(
    *,
    api_key: str | None,
    model_name: str | None,
    config_path: Path | None = None,
) -> tuple[str, str]:
    cfg = _load_toml_config(config_path or DEFAULT_ASR_CONFIG_PATH)
    sec = cfg.get("deepgram", {}) if isinstance(cfg, dict) else {}
    sec_key = str(sec.get("api_key") or "").strip() if isinstance(sec, dict) else ""
    sec_model = str(sec.get("model") or "").strip() if isinstance(sec, dict) else ""

    key = (api_key or "").strip() or (os.getenv("DEEPGRAM_API_KEY") or "").strip() or sec_key
    model = (model_name or "").strip() or sec_model or "nova-3"
    return key, model


def save_deepgram_settings(
    *,
    api_key: str,
    model_name: str,
    config_path: Path | None = None,
) -> Path:
    p = (config_path or DEFAULT_ASR_CONFIG_PATH).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    model = (model_name or "nova-3").strip() or "nova-3"
    key = (api_key or "").strip()
    body = (
        "[deepgram]\n"
        f'model = "{model}"\n'
        f'api_key = "{key}"\n'
    )
    p.write_text(body, encoding="utf-8")
    return p


def ensure_faster_whisper_installed() -> None:
    try:
        __import__("faster_whisper")
        return
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "faster-whisper"])
    __import__("faster_whisper")


def ensure_openai_whisper_installed() -> None:
    try:
        __import__("whisper")
        return
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai-whisper"])
    __import__("whisper")


def get_whisper_model_class():
    ensure_faster_whisper_installed()
    from faster_whisper import WhisperModel

    return WhisperModel


def get_available_whisper_models() -> list[str]:
    try:
        from faster_whisper.utils import available_models

        return list(available_models())
    except Exception:
        return [
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "distil-large-v2",
            "distil-large-v3",
            "turbo",
        ]


def ensure_faster_whisper_model_available(model_name_or_path: str) -> str:
    ensure_faster_whisper_installed()
    from faster_whisper.utils import available_models, download_model

    model_path = Path(model_name_or_path).expanduser()
    if model_path.exists():
        return str(model_path.resolve())

    model_name = model_name_or_path.strip()
    if model_name not in set(available_models()):
        return model_name_or_path

    try:
        local_path = download_model(model_name, local_files_only=True)
        return str(local_path)
    except Exception:
        local_path = download_model(model_name)
        return str(local_path)


def get_openai_whisper_available_models() -> list[str]:
    try:
        import whisper

        models = list(whisper.available_models())
        return sorted(models)
    except Exception:
        return [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "turbo",
        ]


def get_asr_model_choices(asr_engine: str) -> list[str]:
    if asr_engine == "whisper":
        return get_openai_whisper_available_models()
    if asr_engine == "faster-whisper":
        return get_available_whisper_models()
    if asr_engine == "deepgram":
        return list(DEEPGRAM_ASR_MODELS)
    return []


def load_asr_model(asr_engine: str, model_name: str):
    if asr_engine == "faster-whisper":
        model_ref = ensure_faster_whisper_model_available(model_name)
        WhisperModel = get_whisper_model_class()
        return WhisperModel(model_ref, device="auto", compute_type="int8")
    if asr_engine == "whisper":
        ensure_openai_whisper_installed()
        import whisper

        # whisper.load_model will download on first use if needed.
        try:
            return whisper.load_model(model_name)
        except (urlerror.URLError, OSError) as exc:
            raise RuntimeError(
                "Whisper 模型下载失败（网络不可用或 DNS 解析失败）。"
                "请先确保能联网后重试，或切换 ASR 引擎为 faster-whisper。"
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Whisper 模型加载失败: {exc}") from exc
    if asr_engine == "deepgram":
        return model_name.strip() or "nova-3"
    raise RuntimeError(f"Unsupported ASR engine: {asr_engine}")


def transcribe_with_deepgram(
    *,
    video_path: Path,
    source_language: str | None,
    max_segment_duration: float,
    max_segment_chars: int,
    api_key: str,
    model_name: str,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[SubtitleEntry]:
    global _LAST_DEEPGRAM_DETECTED_LANGUAGE, _LAST_DEEPGRAM_LANGUAGE_CONFIDENCE
    if not api_key.strip():
        raise RuntimeError("Deepgram API key 为空。请设置 --deepgram-api-key 或环境变量 DEEPGRAM_API_KEY")

    audio_bytes = video_path.read_bytes()
    params = {
        "model": model_name or "nova-3",
        "smart_format": "true",
        "punctuate": "true",
        "utterances": "true",
        "diarize": "false",
    }
    if source_language:
        params["language"] = source_language
    else:
        # Ask Deepgram to auto-detect source language when not explicitly set.
        params["detect_language"] = "true"
    url = "https://api.deepgram.com/v1/listen?" + urlparse.urlencode(params)
    req = urlrequest.Request(
        url,
        data=audio_bytes,
        headers={
            "Authorization": f"Token {api_key.strip()}",
            "Content-Type": "application/octet-stream",
            "Accept": "application/json",
        },
        method="POST",
    )

    _emit_progress(
        progress_callback,
        task="asr",
        current=0.0,
        total=1.0,
        label=progress_label,
    )
    try:
        with urlrequest.urlopen(req, timeout=600) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except urlerror.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        raise RuntimeError(f"Deepgram 请求失败: HTTP {exc.code} {body[:240]}") from exc
    except Exception as exc:
        raise RuntimeError(f"Deepgram 请求失败: {exc}") from exc

    channels = (
        payload.get("results", {}).get("channels", [])
        if isinstance(payload, dict)
        else []
    )
    alt = {}
    if channels and isinstance(channels[0], dict):
        alts = channels[0].get("alternatives", [])
        if alts and isinstance(alts[0], dict):
            alt = alts[0]
    transcript = str(alt.get("transcript") or "")
    detected_lang = str(alt.get("detected_language") or "")
    try:
        confidence = float(alt.get("language_confidence") or 0.0)
    except Exception:
        confidence = 0.0
    _LAST_DEEPGRAM_DETECTED_LANGUAGE = detected_lang
    _LAST_DEEPGRAM_LANGUAGE_CONFIDENCE = confidence
    words_raw = alt.get("words", []) if isinstance(alt, dict) else []
    words = []
    for w in words_raw if isinstance(words_raw, list) else []:
        if not isinstance(w, dict):
            continue
        words.append(
            SimpleNamespace(
                start=w.get("start"),
                end=w.get("end"),
                word=w.get("word"),
            )
        )
    duration = float(
        payload.get("metadata", {}).get("duration", 0.0)
        if isinstance(payload, dict)
        else 0.0
    )
    end_time = duration
    if end_time <= 0 and words:
        end_time = float(words[-1].end or 0.0)
    if end_time <= 0:
        end_time = 1.0

    splits = split_segment_words(
        words=words,
        text_fallback=transcript,
        start=0.0,
        end=end_time,
        max_duration=max_segment_duration,
        max_chars=max_segment_chars,
    )
    entries: list[SubtitleEntry] = []
    for s, e, t in splits:
        if e <= s:
            e = s + 0.35
        entries.append(
            SubtitleEntry(
                index=len(entries) + 1,
                start=round(s, 3),
                end=round(e, 3),
                text=t,
            )
        )

    _emit_progress(
        progress_callback,
        task="asr",
        current=1.0,
        total=1.0,
        label=progress_label,
    )
    return entries


def get_last_deepgram_detected_language() -> tuple[str, float]:
    return _LAST_DEEPGRAM_DETECTED_LANGUAGE, _LAST_DEEPGRAM_LANGUAGE_CONFIDENCE


def expand_videos(paths: Sequence[str]) -> list[Path]:
    expanded: list[Path] = []
    for p in paths:
        path = Path(p).expanduser().resolve()
        if path.is_dir():
            for child in sorted(path.iterdir()):
                if child.suffix.lower() in VIDEO_EXTENSIONS:
                    expanded.append(child)
        elif path.is_file():
            expanded.append(path)
    return expanded


def sec_to_srt_time(seconds: float) -> str:
    millis = max(0, int(round(seconds * 1000)))
    h = millis // 3_600_000
    m = (millis % 3_600_000) // 60_000
    s = (millis % 60_000) // 1000
    ms = millis % 1000
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def srt_time_to_sec(value: str) -> float:
    # Format: HH:MM:SS,mmm
    hhmmss, ms = value.strip().split(",")
    hh, mm, ss = hhmmss.split(":")
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000.0


def write_srt(entries: Sequence[SubtitleEntry], path: Path) -> None:
    lines: list[str] = []
    for idx, entry in enumerate(entries, start=1):
        lines.append(str(idx))
        lines.append(f"{sec_to_srt_time(entry.start)} --> {sec_to_srt_time(entry.end)}")
        lines.append(entry.text.strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def read_srt(path: Path) -> list[SubtitleEntry]:
    content = path.read_text(encoding="utf-8", errors="ignore")
    blocks = re.split(r"\n\s*\n", content.strip(), flags=re.M)
    entries: list[SubtitleEntry] = []
    for block in blocks:
        lines = [ln.rstrip("\n") for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        idx_line = lines[0].strip()
        time_line = lines[1].strip()
        text_lines = lines[2:] if idx_line.isdigit() else lines[1:]
        if "-->" not in time_line:
            continue
        start_raw, end_raw = [p.strip() for p in time_line.split("-->", 1)]
        try:
            start = srt_time_to_sec(start_raw)
            end = srt_time_to_sec(end_raw)
        except Exception:
            continue
        text = " ".join(t.strip() for t in text_lines).strip()
        if not text:
            continue
        entries.append(
            SubtitleEntry(
                index=len(entries) + 1,
                start=round(start, 3),
                end=round(end, 3),
                text=text,
            )
        )
    return entries


def _normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace(" ,", ",").replace(" .", ".")
    text = text.replace(" ?", "?").replace(" !", "!")
    return text


def _join_word_tokens(words: Sequence[str]) -> str:
    # Deepgram words are often plain tokens without leading spaces, while
    # whisper/faster-whisper tokens may already include spacing.
    if not words:
        return ""
    has_prefixed_space = any((w or "").startswith(" ") for w in words)
    if has_prefixed_space:
        return "".join(words)
    return " ".join((w or "").strip() for w in words if (w or "").strip())


def _split_text_prefer_punctuation(text: str, max_chars: int) -> list[str]:
    text = _normalize_text(text)
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    strong_punct = set("。.!?！？")
    soft_punct = set(",，")
    out: list[str] = []
    i = 0
    n = len(text)
    hard_limit = max(max_chars + 6, int(max_chars * 1.4))

    while i < n:
        j = min(n, i + max_chars)
        if j >= n:
            out.append(text[i:n].strip())
            break

        window = text[i:j]
        strong_idx = max((k for k, ch in enumerate(window) if ch in strong_punct), default=-1)
        if strong_idx >= 0:
            cut = i + strong_idx + 1
            out.append(text[i:cut].strip())
            i = cut
            continue

        # If not clearly over limit yet, keep waiting for punctuation.
        if j - i < hard_limit and j < n:
            k = min(n, i + hard_limit)
            ext = text[i:k]
            strong_ext = max((x for x, ch in enumerate(ext) if ch in strong_punct), default=-1)
            if strong_ext >= 0:
                cut = i + strong_ext + 1
                out.append(text[i:cut].strip())
                i = cut
                continue

            soft_ext = max((x for x, ch in enumerate(ext) if ch in soft_punct), default=-1)
            if soft_ext >= 0:
                cut = i + soft_ext + 1
                out.append(text[i:cut].strip())
                i = cut
                continue

            # hard split at extended limit as last resort.
            out.append(text[i:k].strip())
            i = k
            continue

        soft_idx = max((k for k, ch in enumerate(window) if ch in soft_punct), default=-1)
        if soft_idx >= 0:
            cut = i + soft_idx + 1
            out.append(text[i:cut].strip())
            i = cut
            continue

        # Last resort when no punctuation.
        out.append(text[i:j].strip())
        i = j

    return [seg for seg in out if seg]


def resegment_translated_entries(
    entries: Sequence[SubtitleEntry], max_duration: float, max_chars: int
) -> list[SubtitleEntry]:
    result: list[SubtitleEntry] = []
    for entry in entries:
        text = _normalize_text(entry.text)
        duration = max(0.3, entry.end - entry.start)
        duration_parts = max(1, int((duration + max_duration - 1e-9) // max_duration))
        target_chars = max_chars
        if duration_parts > 1 and text:
            target_chars = min(max_chars, max(8, int(len(text) / duration_parts) + 2))

        pieces = _split_text_prefer_punctuation(text, target_chars)
        if not pieces:
            continue

        total_len = sum(max(1, len(p)) for p in pieces)
        cur_start = entry.start
        for i, piece in enumerate(pieces):
            weight = max(1, len(piece)) / total_len
            if i == len(pieces) - 1:
                cur_end = entry.end
            else:
                cur_end = min(entry.end, cur_start + duration * weight)
            if cur_end <= cur_start:
                cur_end = min(entry.end, cur_start + 0.25)
            result.append(
                SubtitleEntry(
                    index=len(result) + 1,
                    start=round(cur_start, 3),
                    end=round(cur_end, 3),
                    text=piece,
                )
            )
            cur_start = cur_end
    return result


def split_segment_words(
    words: Sequence,
    text_fallback: str,
    start: float,
    end: float,
    max_duration: float,
    max_chars: int,
) -> list[tuple[float, float, str]]:
    if words:
        chunks: list[tuple[float, float, str]] = []
        cur_words: list[dict[str, float | str]] = []

        def chunk_text(items: Sequence[dict[str, float | str]]) -> str:
            return _normalize_text(_join_word_tokens([str(it["word"]) for it in items]))

        def flush_upto(idx: int) -> None:
            nonlocal cur_words
            if idx < 0 or idx >= len(cur_words):
                return
            part = cur_words[: idx + 1]
            c_start = float(part[0]["start"])
            c_end = float(part[-1]["end"])
            text = chunk_text(part)
            if text:
                chunks.append((c_start, c_end, text))
            cur_words = cur_words[idx + 1 :]

        def find_boundary(prefer_soft: bool = False) -> int | None:
            # Prefer sentence-end punctuation, then comma.
            strong = r"[.!?。！？]$"
            soft = r"[,，]$"
            pattern_order = [strong, soft] if prefer_soft else [strong]
            for pattern in pattern_order:
                for i in range(len(cur_words) - 1, -1, -1):
                    token = str(cur_words[i]["word"]).strip()
                    if re.search(pattern, token):
                        return i
            return None

        for w in words:
            w_start = float(w.start if w.start is not None else start)
            w_end = float(w.end if w.end is not None else max(w_start + 0.15, end))
            w_text = (w.word or "")
            if not w_text.strip():
                continue
            cur_words.append({"start": w_start, "end": w_end, "word": w_text})

            while cur_words:
                text_now = chunk_text(cur_words)
                duration_now = float(cur_words[-1]["end"]) - float(cur_words[0]["start"])
                over_limit = duration_now > max_duration or len(text_now) > max_chars
                hard_over = duration_now > max_duration * 1.6 or len(text_now) > int(
                    max_chars * 1.6
                )

                if not over_limit:
                    # Natural flush on sentence boundary to keep next subtitle starting cleanly.
                    if re.search(r"[.!?。！？]$", str(cur_words[-1]["word"]).strip()):
                        flush_upto(len(cur_words) - 1)
                    break

                # First preference: split at sentence punctuation.
                boundary = find_boundary(prefer_soft=False)
                # If still too long, allow comma split.
                if boundary is None and hard_over:
                    boundary = find_boundary(prefer_soft=True)
                # Last resort: force split before the newest word.
                if boundary is None and hard_over and len(cur_words) > 1:
                    boundary = len(cur_words) - 2

                if boundary is None:
                    # Keep accumulating and wait for punctuation, unless we're at a hard limit.
                    break

                flush_upto(boundary)

        if cur_words:
            flush_upto(len(cur_words) - 1)

        if chunks:
            return chunks

    text = _normalize_text(text_fallback)
    if not text:
        return []

    tokens = text.split(" ")
    if not tokens:
        return [(start, end, text)]

    total = len(tokens)
    duration = max(end - start, 0.4)
    chunks = []
    i = 0
    while i < total:
        j = i
        char_count = 0
        while j < total and char_count + len(tokens[j]) + 1 <= max_chars:
            char_count += len(tokens[j]) + 1
            j += 1
            if j - i >= 8:
                break
        if j == i:
            j += 1
        c_start = start + duration * (i / total)
        c_end = start + duration * (j / total)
        chunks.append((c_start, c_end, " ".join(tokens[i:j]).strip()))
        i = j

    return chunks


def transcribe_to_entries(
    model: Any,
    video_path: Path,
    source_language: str | None,
    max_segment_duration: float,
    max_segment_chars: int,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[SubtitleEntry]:
    segments, _info = model.transcribe(
        str(video_path),
        language=source_language,
        vad_filter=True,
        condition_on_previous_text=True,
        word_timestamps=True,
        beam_size=5,
        temperature=0.0,
        compression_ratio_threshold=2.4,
        no_speech_threshold=0.55,
    )

    total_duration = float(getattr(_info, "duration", 0.0) or 0.0)
    segment_iter = segments
    if total_duration <= 0:
        segment_iter = list(segments)
        total_duration = max(
            (float(getattr(segment, "end", 0.0) or 0.0) for segment in segment_iter),
            default=0.0,
        )

    progress = _make_progress_bar(
        task="ASR",
        total=total_duration,
        unit="s",
        progress_label=progress_label,
    )
    recognized_until = 0.0
    _emit_progress(
        progress_callback,
        task="asr",
        current=recognized_until,
        total=total_duration,
        label=progress_label,
    )

    entries: list[SubtitleEntry] = []
    try:
        for segment in segment_iter:
            seg_end = float(getattr(segment, "end", 0.0) or 0.0)
            if total_duration > 0:
                seg_end = min(seg_end, total_duration)
            if seg_end > recognized_until:
                progress.update(seg_end - recognized_until)
                recognized_until = seg_end
                _emit_progress(
                    progress_callback,
                    task="asr",
                    current=recognized_until,
                    total=total_duration,
                    label=progress_label,
                )

            splits = split_segment_words(
                words=list(segment.words or []),
                text_fallback=str(segment.text or ""),
                start=float(segment.start),
                end=float(segment.end),
                max_duration=max_segment_duration,
                max_chars=max_segment_chars,
            )
            for s, e, t in splits:
                if e <= s:
                    e = s + 0.35
                entries.append(
                    SubtitleEntry(
                        index=len(entries) + 1,
                        start=round(s, 3),
                        end=round(e, 3),
                        text=t,
                    )
                )
    finally:
        if total_duration > 0 and recognized_until < total_duration:
            progress.update(total_duration - recognized_until)
            recognized_until = total_duration
            _emit_progress(
                progress_callback,
                task="asr",
                current=recognized_until,
                total=total_duration,
                label=progress_label,
            )
        progress.close()

    return entries


def transcribe_with_asr_engine(
    asr_engine: str,
    model: Any,
    video_path: Path,
    source_language: str | None,
    max_segment_duration: float,
    max_segment_chars: int,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
    deepgram_api_key: str = "",
) -> list[SubtitleEntry]:
    if asr_engine == "faster-whisper":
        return transcribe_to_entries(
            model=model,
            video_path=video_path,
            source_language=source_language,
            max_segment_duration=max_segment_duration,
            max_segment_chars=max_segment_chars,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )

    if asr_engine == "whisper":
        result = model.transcribe(
            str(video_path),
            language=source_language,
            word_timestamps=True,
            condition_on_previous_text=True,
            temperature=0.0,
        )
        segments = result.get("segments", []) if isinstance(result, dict) else []
        total_duration = 0.0
        if isinstance(result, dict):
            total_duration = float(result.get("duration", 0.0) or 0.0)
        if total_duration <= 0:
            total_duration = max(
                (
                    float(segment.get("end", 0.0) or 0.0)
                    for segment in segments
                    if isinstance(segment, dict)
                ),
                default=0.0,
            )
        progress = _make_progress_bar(
            task="ASR",
            total=total_duration,
            unit="s",
            progress_label=progress_label,
        )
        recognized_until = 0.0
        _emit_progress(
            progress_callback,
            task="asr",
            current=recognized_until,
            total=total_duration,
            label=progress_label,
        )

        entries: list[SubtitleEntry] = []
        try:
            for segment in segments:
                if isinstance(segment, dict):
                    seg_end = float(segment.get("end", 0.0) or 0.0)
                else:
                    seg_end = 0.0
                if total_duration > 0:
                    seg_end = min(seg_end, total_duration)
                if seg_end > recognized_until:
                    progress.update(seg_end - recognized_until)
                    recognized_until = seg_end
                    _emit_progress(
                        progress_callback,
                        task="asr",
                        current=recognized_until,
                        total=total_duration,
                        label=progress_label,
                    )

                seg_words = []
                for w in segment.get("words", []) if isinstance(segment, dict) else []:
                    seg_words.append(
                        SimpleNamespace(
                            start=w.get("start"),
                            end=w.get("end"),
                            word=w.get("word"),
                        )
                    )
                splits = split_segment_words(
                    words=seg_words,
                    text_fallback=str(segment.get("text", "")),
                    start=float(segment.get("start", 0.0)),
                    end=float(segment.get("end", 0.0)),
                    max_duration=max_segment_duration,
                    max_chars=max_segment_chars,
                )
                for s, e, t in splits:
                    if e <= s:
                        e = s + 0.35
                    entries.append(
                        SubtitleEntry(
                            index=len(entries) + 1,
                            start=round(s, 3),
                            end=round(e, 3),
                            text=t,
                        )
                    )
        finally:
            if total_duration > 0 and recognized_until < total_duration:
                progress.update(total_duration - recognized_until)
                recognized_until = total_duration
                _emit_progress(
                    progress_callback,
                    task="asr",
                    current=recognized_until,
                    total=total_duration,
                    label=progress_label,
                )
            progress.close()

        return entries

    if asr_engine == "deepgram":
        model_name = model if isinstance(model, str) else "nova-3"
        return transcribe_with_deepgram(
            video_path=video_path,
            source_language=source_language,
            max_segment_duration=max_segment_duration,
            max_segment_chars=max_segment_chars,
            api_key=deepgram_api_key,
            model_name=model_name,
            progress_label=progress_label,
            progress_callback=progress_callback,
        )

    raise RuntimeError(f"Unsupported ASR engine: {asr_engine}")


def _extract_json(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fence:
        return fence.group(1)
    return text


def _estimate_tokens(text: str) -> int:
    # Lightweight approximation for mixed Chinese/English subtitle text.
    return max(1, len(text) // 3)


def _has_cjk(text: str) -> bool:
    return re.search(r"[\u4e00-\u9fff]", text) is not None


def _has_kana(text: str) -> bool:
    # Hiragana + Katakana.
    return re.search(r"[\u3040-\u30ff]", text) is not None


def _has_hangul(text: str) -> bool:
    return re.search(r"[\uac00-\ud7af]", text) is not None


def _has_latin(text: str) -> bool:
    return re.search(r"[A-Za-z]", text) is not None


def _norm_for_compare(text: str) -> str:
    t = _normalize_text(text).lower()
    # Keep letters and CJK/Kana/Hangul for rough similarity check.
    return re.sub(r"[^a-z0-9\u3040-\u30ff\u4e00-\u9fff\uac00-\ud7af]+", "", t)


def _norm_for_repeat_check(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"\s+", "", t)
    # Ignore punctuation for duplicate-line checks.
    return re.sub(r"[，。！？；：,.!?;:\"'“”‘’()（）【】\\[\\]{}<>-]+", "", t)


def _similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _looks_like_untranslated(source_text: str, translated_text: str) -> bool:
    source = (source_text or "").strip()
    out = (translated_text or "").strip()
    if not source:
        return False
    if not out:
        return True

    src_has_foreign = _has_latin(source) or _has_kana(source) or _has_hangul(source)
    if not src_has_foreign:
        # Source already looks Chinese-like; avoid unnecessary retries.
        return False

    # Any kana in final output strongly indicates Japanese was kept.
    if _has_kana(out):
        return True

    # Pure Latin output is not Chinese translation.
    if _has_latin(out) and not _has_cjk(out):
        return True

    s_norm = _norm_for_compare(source)
    o_norm = _norm_for_compare(out)
    if s_norm and o_norm:
        if s_norm == o_norm:
            return True
        ratio = difflib.SequenceMatcher(None, s_norm, o_norm).ratio()
        if ratio >= 0.92:
            return True

    return False


def _looks_like_repeated_neighbor(
    prev_source_text: str,
    prev_translated_text: str,
    source_text: str,
    translated_text: str,
) -> bool:
    prev_zh = _norm_for_repeat_check(prev_translated_text)
    cur_zh = _norm_for_repeat_check(translated_text)
    if not prev_zh or not cur_zh:
        return False
    if prev_zh != cur_zh:
        return False

    # If source lines are effectively the same, duplicate translation is acceptable.
    prev_src = _norm_for_repeat_check(prev_source_text)
    cur_src = _norm_for_repeat_check(source_text)
    return prev_src != cur_src


def _looks_like_carry_over_translation(
    prev_source_text: str,
    prev_translated_text: str,
    source_text: str,
    translated_text: str,
) -> bool:
    prev_zh = _norm_for_repeat_check(prev_translated_text)
    cur_zh = _norm_for_repeat_check(translated_text)
    if len(prev_zh) < 6 or len(cur_zh) < 6:
        return False

    zh_sim = _similarity(prev_zh, cur_zh)
    prev_src = _norm_for_compare(prev_source_text)
    cur_src = _norm_for_compare(source_text)
    src_sim = _similarity(prev_src, cur_src)
    return zh_sim >= 0.72 and src_sim <= 0.55


def _looks_overlong_vs_source(source_text: str, translated_text: str) -> bool:
    src = (source_text or "").strip()
    out = _norm_for_repeat_check(translated_text)
    if not src or not out:
        return False
    if _has_cjk(src):
        return False
    # For EN/JP->ZH subtitle translation, output being much longer than source
    # often indicates merged neighboring content.
    return len(out) > max(18, int(len(src) * 2.6))


def _needs_zh_translation(text: str) -> bool:
    # Source text needs translation when it contains common non-Chinese scripts.
    t = text or ""
    if _has_kana(t) or _has_hangul(t):
        return True
    return _has_latin(t) and not _has_cjk(t)


def _build_token_limited_batches(
    entries: Sequence[SubtitleEntry], max_tokens: int
) -> list[tuple[int, int]]:
    if max_tokens < 200:
        raise RuntimeError("translation_max_tokens is too small; must be >= 200")

    batches: list[tuple[int, int]] = []
    i = 0
    n = len(entries)
    while i < n:
        used = 120
        j = i
        while j < n:
            line_tokens = _estimate_tokens(entries[j].text) + 12
            if j > i and used + line_tokens > max_tokens:
                break
            used += line_tokens
            j += 1
            if used >= max_tokens:
                break
        if j == i:
            j += 1
        batches.append((i, j))
        i = j
    return batches


def _is_local_ollama_qwen25(model_name: str, base_url: str) -> bool:
    b = base_url.strip().lower()
    m = model_name.strip().lower()
    return ("localhost:11434" in b or "127.0.0.1:11434" in b) and m.startswith(
        "qwen2.5"
    )


def _is_local_ollama_dolphin(model_name: str, base_url: str) -> bool:
    b = base_url.strip().lower()
    m = model_name.strip().lower()
    if not ("localhost:11434" in b or "127.0.0.1:11434" in b):
        return False
    return "dolphin3" in m or "dolphin3-abliterated" in m


def _ollama_api_root(base_url: str) -> str:
    parsed = urlparse.urlparse(base_url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return "http://localhost:11434"


def _ollama_chat_once(
    api_root: str,
    model_name: str,
    messages: list[dict[str, str]],
    options: dict[str, Any],
    json_mode: bool = False,
) -> str:
    payload = {
        "model": model_name,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urlrequest.Request(
        f"{api_root}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlrequest.urlopen(req, timeout=180) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    obj = json.loads(body)
    msg = obj.get("message", {}) if isinstance(obj, dict) else {}
    return str(msg.get("content", "")).strip()


def _chat_completion_text(
    *,
    client,
    model_name: str,
    messages: list[dict[str, str]],
    base_url: str,
    use_ollama_native_params: bool,
) -> str:
    if use_ollama_native_params:
        api_root = _ollama_api_root(base_url)
        m = model_name.strip().lower()
        if m.startswith("qwen2.5"):
            options = dict(QWEN25_OLLAMA_TRANSLATION_OPTIONS)
        elif "dolphin3" in m or "dolphin3-abliterated" in m:
            options = dict(DOLPHIN3_OLLAMA_TRANSLATION_OPTIONS)
        else:
            options = dict(QWEN25_OLLAMA_TRANSLATION_OPTIONS)
        json_mode = any(
            "json" in str(m.get("content", "")).lower()
            for m in messages
            if isinstance(m, dict)
        )
        return _ollama_chat_once(
            api_root,
            model_name,
            messages,
            options,
            json_mode=json_mode,
        )

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=messages,
    )
    return (response.choices[0].message.content or "").strip()


def _translate_single_line(
    client,
    model_name: str,
    text: str,
    prev_ctx: Sequence[str],
    next_ctx: Sequence[str],
    base_url: str,
    use_ollama_native_params: bool,
    strict: bool = False,
) -> str:
    if strict:
        system_prompt = (
            "Translate the given subtitle text into Simplified Chinese ONLY. "
            "Only translate the current line. Do not include previous/next lines. "
            "Do not add inferred content. Do not explain."
        )
    else:
        system_prompt = (
            "You are an expert subtitle translator. Translate to Simplified Chinese. "
            "Use context only for term consistency. "
            "Translate ONLY the current subtitle line, without copying or summarizing "
            "previous/next lines. Return ONLY translated text."
        )
    payload = {
        "previous_context": list(prev_ctx),
        "text": text,
        "next_context": list(next_ctx),
    }
    return _chat_completion_text(
        client=client,
        model_name=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        base_url=base_url,
        use_ollama_native_params=use_ollama_native_params,
    )


def _retry_translate_if_needed(
    client,
    model_name: str,
    source_text: str,
    current_text: str,
    prev_ctx: Sequence[str],
    next_ctx: Sequence[str],
    base_url: str,
    use_ollama_native_params: bool,
) -> str:
    if not _needs_zh_translation(source_text):
        return current_text or source_text
    if current_text and not _looks_like_untranslated(source_text, current_text):
        return current_text

    candidate = current_text or source_text
    try:
        retry = _translate_single_line(
            client=client,
            model_name=model_name,
            text=source_text,
            prev_ctx=prev_ctx,
            next_ctx=next_ctx,
            base_url=base_url,
            use_ollama_native_params=use_ollama_native_params,
            strict=False,
        )
        if retry:
            candidate = retry
    except Exception:
        pass

    attempts = 0
    while attempts < STRICT_RETRY_MAX_ATTEMPTS and _looks_like_untranslated(
        source_text, candidate
    ):
        attempts += 1
        try:
            retry_strict = _translate_single_line(
                client=client,
                model_name=model_name,
                text=source_text,
                prev_ctx=prev_ctx,
                next_ctx=next_ctx,
                base_url=base_url,
                use_ollama_native_params=use_ollama_native_params,
                strict=True,
            )
            if retry_strict:
                candidate = retry_strict
        except Exception:
            continue

    return candidate or source_text


def translate_entries_contextual(
    entries: Sequence[SubtitleEntry],
    model_name: str,
    max_tokens: int,
    base_url: str,
    api_key: str,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    from openai import OpenAI

    use_ollama_native_params = _is_local_ollama_qwen25(
        model_name, base_url
    ) or _is_local_ollama_dolphin(model_name, base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)
    output = [e.text for e in entries]
    progress = _make_progress_bar(
        task="Translate",
        total=len(entries),
        unit="line",
        progress_label=progress_label,
    )
    translated_count = 0.0
    total_count = float(len(entries))
    _emit_progress(
        progress_callback,
        task="translate",
        current=translated_count,
        total=total_count,
        label=progress_label,
    )

    try:
        # For local Qwen2.5, translate line-by-line with context to reduce cross-line
        # merging/repetition and keep subtitle alignment with source entries.
        if use_ollama_native_params:
            for idx, entry in enumerate(entries):
                prev_ctx = [entries[i].text for i in range(max(0, idx - 3), idx)]
                next_ctx = [
                    entries[i].text for i in range(idx + 1, min(len(entries), idx + 4))
                ]
                translated = _translate_single_line(
                    client=client,
                    model_name=model_name,
                    text=entry.text,
                    prev_ctx=prev_ctx,
                    next_ctx=next_ctx,
                    base_url=base_url,
                    use_ollama_native_params=True,
                    strict=False,
                )
                translated = _retry_translate_if_needed(
                    client=client,
                    model_name=model_name,
                    source_text=entry.text,
                    current_text=translated,
                    prev_ctx=prev_ctx,
                    next_ctx=next_ctx,
                    base_url=base_url,
                    use_ollama_native_params=True,
                )

                # Guard against over-translation/line-merging from context.
                if _looks_overlong_vs_source(entry.text, translated):
                    for _ in range(STRICT_RETRY_MAX_ATTEMPTS):
                        retry_strict = _translate_single_line(
                            client=client,
                            model_name=model_name,
                            text=entry.text,
                            prev_ctx=[],
                            next_ctx=[],
                            base_url=base_url,
                            use_ollama_native_params=True,
                            strict=True,
                        )
                        if retry_strict:
                            translated = retry_strict
                        if not _looks_overlong_vs_source(entry.text, translated):
                            break

                # Guard against accidental duplicate carry-over from adjacent line.
                if idx > 0 and _looks_like_repeated_neighbor(
                    entries[idx - 1].text,
                    output[idx - 1],
                    entry.text,
                    translated,
                ):
                    for _ in range(STRICT_RETRY_MAX_ATTEMPTS):
                        retry_strict = _translate_single_line(
                            client=client,
                            model_name=model_name,
                            text=entry.text,
                            prev_ctx=[],
                            next_ctx=[],
                            base_url=base_url,
                            use_ollama_native_params=True,
                            strict=True,
                        )
                        if retry_strict:
                            translated = retry_strict
                        if not _looks_like_repeated_neighbor(
                            entries[idx - 1].text,
                            output[idx - 1],
                            entry.text,
                            translated,
                        ):
                            break

                # Guard against semantically similar carry-over even if text differs.
                if idx > 0 and _looks_like_carry_over_translation(
                    entries[idx - 1].text,
                    output[idx - 1],
                    entry.text,
                    translated,
                ):
                    for _ in range(STRICT_RETRY_MAX_ATTEMPTS):
                        retry_strict = _translate_single_line(
                            client=client,
                            model_name=model_name,
                            text=entry.text,
                            prev_ctx=[],
                            next_ctx=[],
                            base_url=base_url,
                            use_ollama_native_params=True,
                            strict=True,
                        )
                        if retry_strict:
                            translated = retry_strict
                        if not _looks_like_carry_over_translation(
                            entries[idx - 1].text,
                            output[idx - 1],
                            entry.text,
                            translated,
                        ):
                            break

                output[idx] = translated or entry.text
                progress.update(1)
                translated_count += 1.0
                _emit_progress(
                    progress_callback,
                    task="translate",
                    current=translated_count,
                    total=total_count,
                    label=progress_label,
                )
            return output

        batches = _build_token_limited_batches(entries, max_tokens=max_tokens)
        for batch_start, batch_end in batches:
            batch = entries[batch_start:batch_end]
            prev_ctx = entries[max(0, batch_start - 5) : batch_start]
            next_ctx = entries[batch_end : min(len(entries), batch_end + 5)]

            payload = {
                "previous_context": [e.text for e in prev_ctx],
                "current_batch": [{"id": i, "text": e.text} for i, e in enumerate(batch)],
                "next_context": [e.text for e in next_ctx],
            }

            system_prompt = (
                "You are an expert subtitle translator. Translate the current batch to Simplified Chinese "
                "using previous and next context to keep terms consistent and natural. "
                "Output STRICT JSON array only: [{\"id\":0,\"zh\":\"...\"}]. "
                "Keep each line concise for subtitles, no explanations."
            )

            text = _chat_completion_text(
                client=client,
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                base_url=base_url,
                use_ollama_native_params=use_ollama_native_params,
            )
            try:
                parsed = json.loads(_extract_json(text))
            except json.JSONDecodeError as exc:
                snippet = text[:180].replace("\n", " ")
                raise RuntimeError(
                    f"Translation model returned non-JSON output (model={model_name}). "
                    f"snippet={snippet}"
                ) from exc

            batch_map = {
                int(item["id"]): str(item["zh"]).strip()
                for item in parsed
                if isinstance(item, dict) and "id" in item and "zh" in item
            }

            for i, entry in enumerate(batch):
                translated = batch_map.get(i, "").strip()
                if not translated:
                    translated = entry.text

                prev_ctx = [e.text for e in batch[max(0, i - 2) : i]]
                next_ctx = [e.text for e in batch[i + 1 : i + 3]]
                translated = _retry_translate_if_needed(
                    client=client,
                    model_name=model_name,
                    source_text=entry.text,
                    current_text=translated,
                    prev_ctx=prev_ctx,
                    next_ctx=next_ctx,
                    base_url=base_url,
                    use_ollama_native_params=use_ollama_native_params,
                )

                output[batch_start + i] = translated or entry.text
                progress.update(1)
                translated_count += 1.0
                _emit_progress(
                    progress_callback,
                    task="translate",
                    current=translated_count,
                    total=total_count,
                    label=progress_label,
                )

        # Final global fallback: catch any remaining English-looking lines.
        for idx, entry in enumerate(entries):
            out = output[idx]
            if _needs_zh_translation(entry.text) and _looks_like_untranslated(entry.text, out):
                prev_ctx = [entries[i].text for i in range(max(0, idx - 2), idx)]
                next_ctx = [
                    entries[i].text for i in range(idx + 1, min(len(entries), idx + 3))
                ]
                output[idx] = _retry_translate_if_needed(
                    client=client,
                    model_name=model_name,
                    source_text=entry.text,
                    current_text=out,
                    prev_ctx=prev_ctx,
                    next_ctx=next_ctx,
                    base_url=base_url,
                    use_ollama_native_params=use_ollama_native_params,
                )

        return output
    finally:
        progress.close()


def _strip_redundant_prefix(prev_text: str, cur_text: str) -> str:
    prev = (prev_text or "").strip()
    cur = (cur_text or "").strip()
    if not prev or not cur:
        return cur
    if not cur.startswith(prev):
        return cur
    remain = cur[len(prev) :].lstrip(" ，。,.!?！？；;:：")
    return remain or cur


def redistribute_translated_by_source_timestamps(
    entries: Sequence[SubtitleEntry],
    translated_texts: Sequence[str],
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    max_tokens: int,
    progress_label: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> list[str]:
    if len(entries) != len(translated_texts):
        raise RuntimeError("Source subtitles and translated subtitles count mismatch.")
    if not entries:
        return []

    from openai import OpenAI

    use_ollama_native_params = _is_local_ollama_qwen25(
        model_name, base_url
    ) or _is_local_ollama_dolphin(model_name, base_url)
    client = OpenAI(api_key=api_key, base_url=base_url)
    output = [str(t or "").strip() for t in translated_texts]

    batches = _build_token_limited_batches(entries, max_tokens=max(600, max_tokens))
    progress = _make_progress_bar(
        task="Align",
        total=len(entries),
        unit="line",
        progress_label=progress_label,
    )
    aligned_count = 0.0
    total_count = float(len(entries))
    _emit_progress(
        progress_callback,
        task="align",
        current=aligned_count,
        total=total_count,
        label=progress_label,
    )

    try:
        for batch_start, batch_end in batches:
            batch = entries[batch_start:batch_end]
            payload = {
                "task": (
                    "Re-assign Chinese subtitles by source line boundaries. "
                    "Keep exactly one zh line for each source id. "
                    "Do not merge with neighbor lines, do not duplicate previous line."
                ),
                "previous_source_context": [
                    e.text for e in entries[max(0, batch_start - 4) : batch_start]
                ],
                "current_batch": [
                    {
                        "id": i,
                        "source": e.text,
                        "draft_zh": output[batch_start + i],
                    }
                    for i, e in enumerate(batch)
                ],
                "next_source_context": [
                    e.text for e in entries[batch_end : min(len(entries), batch_end + 4)]
                ],
            }
            system_prompt = (
                "You are a subtitle timing alignment editor. "
                "Rewrite ONLY Chinese text per source line id. "
                "Output STRICT JSON array only: [{\"id\":0,\"zh\":\"...\"}]. "
                "Must keep the same number of lines as input ids. "
                "Each line should be concise subtitle Chinese and aligned to that source line only."
            )
            text = _chat_completion_text(
                client=client,
                model_name=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                base_url=base_url,
                use_ollama_native_params=use_ollama_native_params,
            )
            try:
                parsed = json.loads(_extract_json(text))
            except json.JSONDecodeError:
                parsed = []
            batch_map = {
                int(item["id"]): str(item["zh"]).strip()
                for item in parsed
                if isinstance(item, dict) and "id" in item and "zh" in item
            }

            for i, entry in enumerate(batch):
                idx = batch_start + i
                candidate = batch_map.get(i, "").strip() or output[idx] or entry.text
                if idx > 0 and _looks_like_repeated_neighbor(
                    entries[idx - 1].text,
                    output[idx - 1],
                    entry.text,
                    candidate,
                ):
                    candidate = _strip_redundant_prefix(output[idx - 1], candidate)
                if idx > 0 and _looks_like_carry_over_translation(
                    entries[idx - 1].text,
                    output[idx - 1],
                    entry.text,
                    candidate,
                ):
                    candidate = _strip_redundant_prefix(output[idx - 1], candidate)
                output[idx] = (candidate or entry.text).strip()
                progress.update(1)
                aligned_count += 1.0
                _emit_progress(
                    progress_callback,
                    task="align",
                    current=aligned_count,
                    total=total_count,
                    label=progress_label,
                )
    finally:
        progress.close()

    return output


def main() -> None:
    args = parse_args()
    videos = expand_videos(args.videos)
    if not videos:
        raise SystemExit("No valid video files found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    deepgram_api_key = (args.deepgram_api_key or "").strip()
    if args.asr_engine == "deepgram":
        deepgram_api_key, resolved_deepgram_model = resolve_deepgram_settings(
            api_key=deepgram_api_key,
            model_name=args.deepgram_model,
            config_path=args.asr_config,
        )
        asr_model_name = resolved_deepgram_model
    else:
        asr_model_name = args.whisper_model
    model = load_asr_model(args.asr_engine, asr_model_name)

    do_translate = not args.no_translate
    translation_settings: dict[str, str] | None = None
    if do_translate:
        try:
            translation_settings = resolve_translation_settings(
                backend=args.translate_backend,
                model_name=args.translate_model,
                base_url=args.translate_base_url,
                api_key=args.translate_api_key,
                config_path=args.translate_config,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc

    multi_video_mode = len(videos) > 1
    for video in videos:
        print(f"\nProcessing: {video}")
        target_output_dir = video.parent if multi_video_mode else args.output_dir
        target_output_dir.mkdir(parents=True, exist_ok=True)
        if multi_video_mode:
            print(f"Output directory: {target_output_dir}")
        entries = transcribe_with_asr_engine(
            asr_engine=args.asr_engine,
            model=model,
            video_path=video,
            source_language=args.source_language,
            max_segment_duration=args.max_segment_duration,
            max_segment_chars=args.max_segment_chars,
            progress_label=video.name,
            deepgram_api_key=deepgram_api_key,
        )
        if args.asr_engine == "deepgram":
            lang, conf = get_last_deepgram_detected_language()
            if lang:
                print(f"Deepgram detected language: {lang} (confidence={conf:.3f})")

        stem = video.stem
        source_srt = target_output_dir / f"{stem}.source.srt"
        write_srt(entries, source_srt)
        print(f"Source subtitles: {source_srt}")

        if do_translate:
            zh_texts = translate_entries_contextual(
                entries,
                model_name=translation_settings["model"],
                max_tokens=args.translation_max_tokens,
                base_url=translation_settings["base_url"],
                api_key=translation_settings["api_key"],
                progress_label=video.name,
            )
            print("Running translation realignment by source timestamps...")
            zh_texts = redistribute_translated_by_source_timestamps(
                entries,
                zh_texts,
                model_name=translation_settings["model"],
                base_url=translation_settings["base_url"],
                api_key=translation_settings["api_key"],
                max_tokens=args.translation_max_tokens,
                progress_label=video.name,
            )
            zh_entries = [
                SubtitleEntry(
                    index=e.index,
                    start=e.start,
                    end=e.end,
                    text=zh_texts[idx],
                )
                for idx, e in enumerate(entries)
            ]
            zh_entries = resegment_translated_entries(
                zh_entries,
                max_duration=args.max_segment_duration,
                max_chars=args.max_segment_chars,
            )
            zh_srt = target_output_dir / f"{stem}.zh-CN.srt"
            write_srt(zh_entries, zh_srt)
            print(f"Chinese subtitles: {zh_srt}")


if __name__ == "__main__":
    main()
