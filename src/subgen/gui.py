from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import importlib
from pathlib import Path
from tkinter import font as tkfont
from tkinter import BOTH, END, LEFT, RIGHT, VERTICAL, W, Y, filedialog, messagebox, ttk
import tkinter as tk
from urllib import error as urlerror
from urllib import parse as urlparse
from urllib import request as urlrequest

from subgen.cli import (
    BACKEND_DEFAULTS,
    DEFAULT_ASR_CONFIG_PATH,
    DEFAULT_TRANSLATE_CONFIG_PATH,
    SubtitleEntry,
    ensure_faster_whisper_installed,
    ensure_openai_whisper_installed,
    get_asr_model_choices,
    get_last_deepgram_detected_language,
    load_asr_model,
    expand_videos,
    read_srt,
    resegment_translated_entries,
    translate_fulltext_then_llm_refill,
    resolve_translation_settings,
    resolve_deepgram_settings,
    save_deepgram_settings,
    transcribe_with_asr_engine,
    write_srt,
)

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

MODEL_PRESETS_DEFAULT = [
    "tiny",
    "base",
    "small",
    "medium",
    "large-v3",
    "turbo",
]

TOKEN_RECOMMENDATIONS = {
    "local": "推荐: 1200-2500（本地模型）",
    "openai": "推荐: 6000-10000（OpenAI）",
    "deepseek": "推荐: 6000-10000（DeepSeek）",
    "qwen": "推荐: 6000-10000（百炼Qwen）",
    "minimax": "推荐: 6000-10000（百炼MiniMax M2.7）",
}

LOCAL_TRANSLATION_MODELS = [
    "qwen2.5:7b",
    "dolphin3:8b",
    "dolphin3:latest",
    "nchapman/dolphin3.0-llama3:8b",
    "huihui_ai/dolphin3-abliterated",
]

UI_COLORS = {
    "window_bg": "#fff8f4",
    "hero_bg": "#fffdfb",
    "card_bg": "#ffffff",
    "card_border": "#f7dfd2",
    "shadow": "#fff0e8",
    "text": "#2c2320",
    "muted": "#a49791",
    "accent": "#ff6a45",
    "accent_active": "#ff8e4f",
    "accent_soft": "#fff2eb",
    "accent_alt": "#ff9c52",
    "success": "#43c59e",
    "danger": "#ff5c56",
    "input_bg": "#fffdfa",
    "log_bg": "#fffdfa",
}


def _parse_ollama_list_names(output: str) -> list[str]:
    lines = [ln.rstrip() for ln in (output or "").splitlines() if ln.strip()]
    if not lines:
        return []
    names: list[str] = []
    # Expected table format:
    # NAME   ID   SIZE   MODIFIED
    for i, line in enumerate(lines):
        if i == 0 and "NAME" in line.upper():
            continue
        name = line.split()[0].strip()
        if name and name.upper() != "NAME":
            names.append(name)
    return names


def _ensure_tkinter_tix_compat() -> None:
    # Python 3.13+ removed tkinter.tix in many builds. Some tkinterdnd2
    # releases still import "from tkinter import tix", so provide a minimal
    # compatibility alias to keep drag-drop available.
    try:
        import tkinter as _tk

        if hasattr(_tk, "tix"):
            return

        class _CompatTix:
            Tk = _tk.Tk

        _tk.tix = _CompatTix  # type: ignore[attr-defined]
    except Exception:
        return


def _check_tk_version_compat() -> str | None:
    try:
        import _tkinter

        tk_ver = str(getattr(_tkinter, "TK_VERSION", "") or "")
        if tk_ver and tk_ver.split(".", 1)[0].isdigit():
            major = int(tk_ver.split(".", 1)[0])
            if major >= 9:
                return (
                    f"当前 Tk 版本为 {tk_ver}，tkinterdnd2 暂不兼容 Tk 9。"
                    "请改用 Tk 8.6 的 Python 环境以启用拖拽。"
                )
    except Exception:
        return None
    return None


def _load_tkdnd() -> tuple[object | None, object | None, str | None]:
    tk_compat_err = _check_tk_version_compat()
    if tk_compat_err:
        return None, None, tk_compat_err
    _ensure_tkinter_tix_compat()
    try:
        from tkinterdnd2 import DND_FILES as _DND_FILES, TkinterDnD as _TkinterDnD
        return _DND_FILES, _TkinterDnD, None
    except Exception as exc_first:
        # Try universal package once for macOS/ARM compatibility.
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tkinterdnd2-universal"]
            )
            _ensure_tkinter_tix_compat()
            from tkinterdnd2 import DND_FILES as _DND_FILES, TkinterDnD as _TkinterDnD
            return _DND_FILES, _TkinterDnD, None
        except Exception as exc_second:
            msg = (
                f"拖拽扩展不可用：首次加载失败={exc_first}; "
                f"自动安装后仍失败={exc_second}"
            )
            return None, None, msg


def _repair_tkdnd_and_reload() -> tuple[object | None, object | None, str | None]:
    try:
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--force-reinstall",
                "tkinterdnd2-universal",
            ]
        )
        _ensure_tkinter_tix_compat()
        mod = importlib.import_module("tkinterdnd2.TkinterDnD")
        importlib.reload(mod)
        from tkinterdnd2 import DND_FILES as _DND_FILES, TkinterDnD as _TkinterDnD
        return _DND_FILES, _TkinterDnD, None
    except Exception as exc:
        return None, None, f"拖拽扩展修复失败: {exc}"

def _parse_drop_files(raw: str) -> list[Path]:
    items: list[str] = []
    token = ""
    in_brace = False
    for ch in raw:
        if ch == "{":
            in_brace = True
            token = ""
        elif ch == "}":
            in_brace = False
            if token:
                items.append(token)
            token = ""
        elif ch == " " and not in_brace:
            if token:
                items.append(token)
                token = ""
        else:
            token += ch
    if token:
        items.append(token)

    return [Path(x).expanduser().resolve() for x in items if x.strip()]


def _slugify_model_tag(text: str) -> str:
    t = (text or "").strip().lower()
    t = t.replace("https://", "").replace("http://", "")
    t = t.replace("/", "-").replace(":", "-").replace(".", "-")
    t = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in t)
    t = "-".join(part for part in t.split("-") if part)
    return t or "model"


class RoundButton(tk.Canvas):
    def __init__(
        self,
        master: tk.Widget,
        *,
        text: str,
        command=None,
        font_name: str = "Helvetica",
        font_size: int = 11,
        font_weight: str = "normal",
        min_width: int = 0,
        height: int = 38,
        radius: int = 16,
        padx: int = 18,
        palette: dict[str, str] | None = None,
        surface_bg: str = "#ffffff",
    ) -> None:
        super().__init__(
            master,
            highlightthickness=0,
            bd=0,
            bg=surface_bg,
            width=max(min_width, 40),
            height=height,
            cursor="hand2",
        )
        self._text = text
        self._command = command
        self._font = (font_name, font_size, font_weight)
        self._font_metrics = tkfont.Font(font=self._font)
        self._min_width = min_width
        self._height = height
        self._radius = radius
        self._padx = padx
        self._state = tk.NORMAL
        self._pressed = False
        self._hover = False
        self._palette = {
            "bg": "#ffffff",
            "fg": "#111111",
            "border": "#dddddd",
            "hover_bg": "#f7f7f7",
            "active_bg": "#efefef",
            "disabled_bg": "#f4f4f4",
            "disabled_fg": "#aaaaaa",
            "disabled_border": "#e5e5e5",
        }
        if palette:
            self._palette.update(palette)
        self.bind("<Configure>", lambda _e: self._draw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)
        self._draw()

    def _rounded_points(self, x1: float, y1: float, x2: float, y2: float, r: float) -> list[float]:
        r = max(0.0, min(r, (x2 - x1) / 2, (y2 - y1) / 2))
        return [
            x1 + r, y1, x1 + r, y1, x2 - r, y1, x2 - r, y1,
            x2, y1, x2, y1 + r, x2, y1 + r, x2, y2 - r,
            x2, y2 - r, x2, y2, x2 - r, y2, x2 - r, y2,
            x1 + r, y2, x1 + r, y2, x1, y2, x1, y2 - r,
            x1, y2 - r, x1, y1 + r, x1, y1 + r, x1, y1,
        ]

    def _current_colors(self) -> tuple[str, str, str]:
        if self._state == tk.DISABLED:
            return (
                self._palette["disabled_bg"],
                self._palette["disabled_fg"],
                self._palette["disabled_border"],
            )
        if self._pressed:
            return (
                self._palette.get("active_bg", self._palette["hover_bg"]),
                self._palette["fg"],
                self._palette["border"],
            )
        if self._hover:
            return (
                self._palette["hover_bg"],
                self._palette["fg"],
                self._palette["border"],
            )
        return self._palette["bg"], self._palette["fg"], self._palette["border"]

    def _draw(self) -> None:
        needed_width = max(
            self._min_width,
            self._font_metrics.measure(self._text) + self._padx * 2,
        )
        self.configure(width=needed_width, height=self._height)
        width = max(needed_width, self.winfo_width())
        height = max(self._height, self.winfo_height())
        self.delete("all")
        bg, fg, border = self._current_colors()
        self.create_polygon(
            self._rounded_points(1, 1, width - 1, height - 1, self._radius),
            smooth=True,
            splinesteps=32,
            fill=bg,
            outline=border,
            width=1,
        )
        self.create_text(
            width / 2,
            height / 2,
            text=self._text,
            fill=fg,
            font=self._font,
        )

    def _on_enter(self, _event=None) -> None:
        if self._state != tk.DISABLED:
            self._hover = True
            self._draw()

    def _on_leave(self, _event=None) -> None:
        self._hover = False
        self._pressed = False
        self._draw()

    def _on_press(self, _event=None) -> None:
        if self._state != tk.DISABLED:
            self._pressed = True
            self._draw()

    def _on_release(self, event=None) -> None:
        if self._state == tk.DISABLED:
            return
        inside = True
        if event is not None:
            inside = 0 <= event.x <= self.winfo_width() and 0 <= event.y <= self.winfo_height()
        was_pressed = self._pressed
        self._pressed = False
        self._draw()
        if was_pressed and inside and callable(self._command):
            self._command()

    def set_palette(self, **palette: str) -> None:
        self._palette.update(palette)
        self._draw()

    def configure(self, cnf=None, **kw):
        if cnf:
            kw.update(cnf)
        redraw = False
        if "text" in kw:
            self._text = str(kw.pop("text"))
            redraw = True
        if "command" in kw:
            self._command = kw.pop("command")
        if "state" in kw:
            self._state = kw.pop("state")
            super().configure(cursor=("arrow" if self._state == tk.DISABLED else "hand2"))
            redraw = True
        if "surface_bg" in kw:
            super().configure(bg=kw.pop("surface_bg"))
        result = super().configure(**kw)
        if redraw:
            self._draw()
        return result

    config = configure

    def cget(self, key):
        if key == "text":
            return self._text
        if key == "state":
            return self._state
        return super().cget(key)


class App:
    def __init__(self, root: tk.Tk, dnd_enabled: bool, dnd_reason: str | None = None):
        self.root = root
        self.root.title("SubGen - 自动字幕生成")
        self.root.geometry("1120x820")
        self.root.minsize(1024, 760)
        self.dnd_enabled = dnd_enabled
        self.colors = UI_COLORS.copy()
        self.style = ttk.Style(self.root)
        self._apply_visual_style()

        self.files: list[Path] = []
        self.log_queue: queue.Queue[object] = queue.Queue()
        self.running = False
        self.dnd_reason = dnd_reason or ""

        self.output_dir = tk.StringVar(value=str(Path("./subtitles").resolve()))
        self.asr_engine = tk.StringVar(value="deepgram")
        self.whisper_model = tk.StringVar(value="enhanced")
        _dg_key, _ = resolve_deepgram_settings(
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            model_name="",
            config_path=DEFAULT_ASR_CONFIG_PATH,
        )
        self.deepgram_api_key = tk.StringVar(value=_dg_key)
        self.max_duration = tk.StringVar(value="2.2")
        self.translate_enabled = tk.BooleanVar(value=True)
        self.translate_model = tk.StringVar(value=BACKEND_DEFAULTS["qwen"]["model"])
        self.translate_backend = tk.StringVar(value="qwen")
        self.ollama_status_text = tk.StringVar(value="检测中...")
        self.translate_max_tokens = tk.StringVar(value="2000")
        self.max_tokens_hint = tk.StringVar(value=TOKEN_RECOMMENDATIONS["qwen"])
        self.current_file_text = tk.StringVar(value="当前文件: -")
        self.asr_progress_var = tk.DoubleVar(value=0.0)
        self.asr_progress_text = tk.StringVar(value="ASR: 0.0%")
        self.translate_progress_var = tk.DoubleVar(value=0.0)
        self.translate_progress_text = tk.StringVar(value="翻译: 0.0%")
        self.progress_rings: dict[str, dict[str, object]] = {}
        self.tab_buttons: dict[str, tk.Button] = {}
        self.tab_pages: dict[str, ttk.Frame] = {}
        self.current_tab = "workbench"
        self.translate_backend.trace_add("write", self._on_translate_backend_change)
        self.asr_engine.trace_add("write", self._on_asr_engine_change)

        self._build_ui()
        self._poll_logs()

    def _pick_font_family(self, *candidates: str) -> str:
        available = {name.lower(): name for name in tkfont.families(self.root)}
        for candidate in candidates:
            match = available.get(candidate.lower())
            if match:
                return match
        return "Helvetica"

    def _apply_visual_style(self) -> None:
        self.root.configure(bg=self.colors["window_bg"])
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.ui_font = self._pick_font_family(
            "Avenir Next", "SF Pro Text", "PingFang SC", "Helvetica Neue"
        )
        self.display_font = self._pick_font_family(
            "Avenir Next", "SF Pro Display", "PingFang SC", "Helvetica Neue"
        )

        self.style.configure(
            ".",
            background=self.colors["window_bg"],
            foreground=self.colors["text"],
            font=(self.ui_font, 12),
        )
        self.style.configure("App.TFrame", background=self.colors["window_bg"])
        self.style.configure("Hero.TFrame", background=self.colors["hero_bg"])
        self.style.configure("Card.TFrame", background=self.colors["card_bg"])
        self.style.configure(
            "App.TNotebook",
            background=self.colors["window_bg"],
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )
        self.style.configure(
            "App.TNotebook.Tab",
            background="#e9e9ee",
            foreground=self.colors["muted"],
            padding=(20, 10),
            borderwidth=0,
            font=(self.ui_font, 12, "bold"),
        )
        self.style.map(
            "App.TNotebook.Tab",
            background=[("selected", self.colors["card_bg"]), ("active", "#f0f0f4")],
            foreground=[("selected", self.colors["text"]), ("active", self.colors["text"])],
        )
        self.style.configure(
            "Card.TLabelframe",
            background=self.colors["card_bg"],
            bordercolor=self.colors["card_border"],
            borderwidth=1,
            relief="solid",
        )
        self.style.configure(
            "Card.TLabelframe.Label",
            background=self.colors["card_bg"],
            foreground=self.colors["text"],
            font=(self.display_font, 13, "bold"),
        )
        self.style.configure(
            "Title.TLabel",
            background=self.colors["hero_bg"],
            foreground=self.colors["muted"],
            font=(self.ui_font, 11, "bold"),
        )
        self.style.configure(
            "HeroHeadline.TLabel",
            background=self.colors["hero_bg"],
            foreground=self.colors["text"],
            font=(self.display_font, 28, "bold"),
        )
        self.style.configure(
            "Subtitle.TLabel",
            background=self.colors["hero_bg"],
            foreground=self.colors["muted"],
            font=(self.ui_font, 12),
        )
        self.style.configure(
            "ShellMuted.TLabel",
            background=self.colors["window_bg"],
            foreground=self.colors["muted"],
            font=(self.ui_font, 12),
        )
        self.style.configure(
            "Body.TLabel",
            background=self.colors["card_bg"],
            foreground=self.colors["text"],
            font=(self.ui_font, 12),
        )
        self.style.configure(
            "Muted.TLabel",
            background=self.colors["card_bg"],
            foreground=self.colors["muted"],
            font=(self.ui_font, 11),
        )
        self.style.configure(
            "Pill.TLabel",
            background=self.colors["accent_soft"],
            foreground=self.colors["accent"],
            font=(self.ui_font, 11, "bold"),
            padding=(12, 6),
        )
        self.style.configure(
            "Primary.TButton",
            background=self.colors["accent"],
            foreground="#ffffff",
            borderwidth=0,
            relief="flat",
            focusthickness=0,
            padding=(18, 11),
            font=(self.ui_font, 12, "bold"),
        )
        self.style.map(
            "Primary.TButton",
            background=[("active", self.colors["accent_active"]), ("disabled", "#b9d7f8")],
            foreground=[("disabled", "#f7fbff")],
        )
        self.style.configure(
            "Secondary.TButton",
            background="#fff8f4",
            foreground=self.colors["text"],
            bordercolor="#ffd6c8",
            borderwidth=1,
            relief="solid",
            padding=(14, 9),
            font=(self.ui_font, 11),
        )
        self.style.map(
            "Secondary.TButton",
            background=[("active", self.colors["accent_soft"])],
            bordercolor=[("active", "#ffb89d")],
        )
        self.style.configure(
            "TCheckbutton",
            background=self.colors["card_bg"],
            foreground=self.colors["text"],
            font=(self.ui_font, 12),
        )
        self.style.map(
            "TCheckbutton",
            background=[("active", self.colors["card_bg"])],
            foreground=[("disabled", self.colors["muted"])],
        )
        self.style.configure(
            "TEntry",
            fieldbackground=self.colors["input_bg"],
            foreground=self.colors["text"],
            bordercolor=self.colors["card_border"],
            borderwidth=1,
            relief="solid",
            padding=8,
        )
        self.style.configure(
            "TCombobox",
            fieldbackground=self.colors["input_bg"],
            background=self.colors["input_bg"],
            foreground=self.colors["text"],
            bordercolor=self.colors["card_border"],
            arrowsize=14,
            padding=6,
        )
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", self.colors["input_bg"])],
            background=[("readonly", self.colors["input_bg"])],
            foreground=[("readonly", self.colors["text"])],
        )
        self.style.configure(
            "Flat.TCombobox",
            fieldbackground=self.colors["input_bg"],
            background=self.colors["input_bg"],
            foreground=self.colors["text"],
            borderwidth=0,
            relief="flat",
            arrowsize=14,
            padding=4,
        )
        self.style.map(
            "Flat.TCombobox",
            fieldbackground=[("readonly", self.colors["input_bg"])],
            background=[("readonly", self.colors["input_bg"])],
            foreground=[("readonly", self.colors["text"])],
        )
        self.style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e8e8ed",
            background=self.colors["accent"],
            bordercolor="#e8e8ed",
            lightcolor=self.colors["accent"],
            darkcolor=self.colors["accent"],
            thickness=9,
        )
        self.style.configure(
            "TScrollbar",
            background="#d6d6db",
            troughcolor="#f0f0f2",
            arrowcolor=self.colors["muted"],
            bordercolor="#f0f0f2",
        )

    def _style_surface_widget(self, widget: tk.Widget, *, background: str | None = None) -> None:
        bg = background or self.colors["card_bg"]
        widget.configure(bg=bg)

    def _style_text_widget(self, widget: tk.Widget, *, is_log: bool = False) -> None:
        background = self.colors["log_bg"] if is_log else self.colors["card_bg"]
        widget.configure(
            bg=background,
            fg=self.colors["text"],
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            font=(self.ui_font, 12),
            selectbackground=self.colors["accent"],
            selectforeground="#ffffff",
        )

    def _style_input_widget(self, widget: tk.Entry) -> None:
        widget.configure(
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground=self.colors["card_border"],
            highlightcolor=self.colors["accent"],
            bg=self.colors["input_bg"],
            fg=self.colors["text"],
            disabledbackground=self.colors["input_bg"],
            disabledforeground=self.colors["text"],
            insertbackground=self.colors["text"],
            font=(self.ui_font, 12),
        )

    def _style_flat_input_widget(self, widget: tk.Entry) -> None:
        widget.configure(
            relief="flat",
            borderwidth=0,
            highlightthickness=0,
            bg=self.colors["input_bg"],
            fg=self.colors["text"],
            disabledbackground=self.colors["input_bg"],
            disabledforeground=self.colors["text"],
            insertbackground=self.colors["text"],
            font=(self.ui_font, 12),
        )

    def _button_palette(self, variant: str) -> dict[str, str]:
        if variant == "primary":
            return {
                "bg": self.colors["accent"],
                "fg": "#ffffff",
                "border": self.colors["accent"],
                "hover_bg": self.colors["accent_active"],
                "active_bg": "#ff5d3a",
                "disabled_bg": "#ffd8cc",
                "disabled_fg": "#fff7f3",
                "disabled_border": "#ffd8cc",
            }
        if variant == "tab_active":
            return {
                "bg": self.colors["card_bg"],
                "fg": self.colors["accent"],
                "border": "#ffd8cc",
                "hover_bg": "#fffaf7",
                "active_bg": "#fff3ed",
                "disabled_bg": self.colors["card_bg"],
                "disabled_fg": self.colors["muted"],
                "disabled_border": "#ffd8cc",
            }
        if variant == "tab_idle":
            return {
                "bg": "#fff2eb",
                "fg": self.colors["muted"],
                "border": "#f4d8cb",
                "hover_bg": "#fff7f3",
                "active_bg": "#ffefe7",
                "disabled_bg": "#fff2eb",
                "disabled_fg": self.colors["muted"],
                "disabled_border": "#f4d8cb",
            }
        return {
            "bg": "#fff8f4",
            "fg": self.colors["text"],
            "border": "#ffd6c8",
            "hover_bg": "#fff1ea",
            "active_bg": "#ffe7dc",
            "disabled_bg": "#fff8f4",
            "disabled_fg": "#c0b3ad",
            "disabled_border": "#f3ddd3",
        }

    def _create_round_button(
        self,
        parent: tk.Widget,
        *,
        text: str,
        command,
        variant: str = "secondary",
        min_width: int = 0,
        height: int = 38,
        radius: int = 16,
        font_size: int = 11,
        font_weight: str = "normal",
        surface_bg: str | None = None,
    ) -> RoundButton:
        return RoundButton(
            parent,
            text=text,
            command=command,
            font_name=self.ui_font,
            font_size=font_size,
            font_weight=font_weight,
            min_width=min_width,
            height=height,
            radius=radius,
            palette=self._button_palette(variant),
            surface_bg=surface_bg or parent.cget("bg"),
        )

    def _create_input_shell(
        self,
        parent: tk.Widget,
        *,
        height: int = 42,
        radius: int = 14,
        surface_bg: str | None = None,
        min_width: int = 120,
        fill_color: str | None = None,
        border_color: str | None = None,
        inner_padx: int = 12,
        inner_pady: int = 7,
    ) -> tuple[tk.Canvas, tk.Frame]:
        host_bg = surface_bg or parent.cget("bg")
        shell_fill = fill_color or self.colors["input_bg"]
        shell_border = border_color or self.colors["card_border"]
        shell = tk.Canvas(
            parent,
            bg=host_bg,
            highlightthickness=0,
            bd=0,
            width=min_width,
            height=height,
        )
        inner = tk.Frame(shell, bg=shell_fill, bd=0, padx=inner_padx, pady=inner_pady)
        window_id = shell.create_window(0, 0, anchor="nw", window=inner)

        def redraw(_event=None) -> None:
            desired_width = max(min_width, inner.winfo_reqwidth() + inner_padx * 2)
            desired_height = max(height, inner.winfo_reqheight() + inner_pady * 2)
            shell.configure(width=max(shell.winfo_reqwidth(), desired_width), height=desired_height)
            width = max(shell.winfo_width(), desired_width)
            height_now = max(shell.winfo_height(), desired_height)
            shell.delete("decor")
            self._draw_rounded_rect(
                shell,
                1,
                1,
                width - 1,
                height_now - 1,
                radius=radius,
                fill=shell_fill,
                outline=shell_border,
                tags="decor",
            )
            shell.coords(window_id, inner_padx, inner_pady)
            shell.itemconfigure(
                window_id,
                width=max(20, width - inner_padx * 2),
                height=max(20, height_now - inner_pady * 2),
            )
            shell.tag_raise(window_id)

        shell.bind("<Configure>", redraw)
        inner.bind("<Configure>", lambda _e: shell.after_idle(redraw))
        shell.after_idle(redraw)
        return shell, inner

    def _rounded_rect_points(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> list[float]:
        r = max(0.0, min(radius, (x2 - x1) / 2, (y2 - y1) / 2))
        return [
            x1 + r,
            y1,
            x1 + r,
            y1,
            x2 - r,
            y1,
            x2 - r,
            y1,
            x2,
            y1,
            x2,
            y1 + r,
            x2,
            y1 + r,
            x2,
            y2 - r,
            x2,
            y2 - r,
            x2,
            y2,
            x2 - r,
            y2,
            x2 - r,
            y2,
            x1 + r,
            y2,
            x1 + r,
            y2,
            x1,
            y2,
            x1,
            y2 - r,
            x1,
            y2 - r,
            x1,
            y1 + r,
            x1,
            y1 + r,
            x1,
            y1,
        ]

    def _draw_rounded_rect(
        self,
        canvas: tk.Canvas,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        radius: float,
        fill: str,
        outline: str = "",
        tags: str | tuple[str, ...] = (),
    ) -> int:
        return canvas.create_polygon(
            self._rounded_rect_points(x1, y1, x2, y2, radius),
            smooth=True,
            splinesteps=32,
            fill=fill,
            outline=outline,
            tags=tags,
        )

    def _create_soft_card(
        self,
        parent: tk.Widget | ttk.Frame,
        *,
        padding: tuple[int, int] = (20, 18),
        background: str | None = None,
        corner_radius: int = 22,
    ) -> tuple[tk.Canvas, tk.Frame]:
        bg = background or self.colors["card_bg"]
        shell = tk.Canvas(
            parent,
            bg=self.colors["window_bg"],
            highlightthickness=0,
            bd=0,
            width=1,
            height=1,
        )
        body = tk.Frame(
            shell,
            bg=bg,
            bd=0,
            padx=padding[0],
            pady=padding[1],
        )
        window_id = shell.create_window(0, 0, anchor="nw", window=body)

        def redraw(_event=None) -> None:
            desired_width = body.winfo_reqwidth() + 40
            desired_height = body.winfo_reqheight() + 40
            shell.configure(
                width=max(shell.winfo_reqwidth(), desired_width),
                height=max(shell.winfo_reqheight(), desired_height),
            )
            width = max(shell.winfo_width(), 80)
            height = max(shell.winfo_height(), 80)
            shell.delete("decor")
            margin = 8
            shadow_offset = 6
            self._draw_rounded_rect(
                shell,
                margin + 2,
                margin + shadow_offset,
                width - margin - 2,
                height - margin + shadow_offset - 1,
                radius=corner_radius,
                fill=self.colors["shadow"],
                outline="",
                tags="decor",
            )
            self._draw_rounded_rect(
                shell,
                margin,
                margin,
                width - margin,
                height - margin,
                radius=corner_radius,
                fill=bg,
                outline=self.colors["card_border"],
                tags="decor",
            )
            shell.coords(window_id, margin + 12, margin + 12)
            shell.itemconfigure(
                window_id,
                width=max(20, width - (margin + 12) * 2),
                height=max(20, height - (margin + 12) * 2),
            )
            shell.tag_raise(window_id)

        shell.bind("<Configure>", redraw)
        body.bind("<Configure>", lambda _e: shell.after_idle(redraw))
        shell.after_idle(redraw)
        return shell, body

    def _create_card_block(
        self,
        parent: tk.Widget | ttk.Frame,
        *,
        title: str,
        subtitle: str | None = None,
        padding: tuple[int, int] = (20, 18),
    ) -> tuple[tk.Frame, tk.Frame]:
        shell, body = self._create_soft_card(parent, padding=padding)
        header = tk.Frame(body, bg=self.colors["card_bg"], bd=0)
        header.pack(fill="x")
        tk.Label(
            header,
            text=title,
            bg=self.colors["card_bg"],
            fg=self.colors["text"],
            font=(self.display_font, 18, "bold"),
        ).pack(anchor="w")
        if subtitle:
            tk.Label(
                header,
                text=subtitle,
                bg=self.colors["card_bg"],
                fg=self.colors["muted"],
                font=(self.ui_font, 11),
            ).pack(anchor="w", pady=(4, 0))
        content = tk.Frame(body, bg=self.colors["card_bg"], bd=0)
        content.pack(fill=BOTH, expand=True, pady=(14, 0))
        return shell, content

    def _create_progress_ring(
        self,
        parent: ttk.Frame,
        *,
        key: str,
        title: str,
        status_var: tk.StringVar,
        accent: str,
    ) -> tk.Frame:
        card, card_body = self._create_input_shell(
            parent,
            height=168,
            radius=18,
            surface_bg=self.colors["card_bg"],
            min_width=126,
            fill_color=self.colors["accent_soft"],
            border_color="#ffd9ca",
            inner_padx=12,
            inner_pady=12,
        )

        tk.Label(
            card_body,
            text=title,
            bg=self.colors["accent_soft"],
            fg=self.colors["text"],
            font=(self.display_font, 13, "bold"),
        ).pack(anchor="center")

        canvas = tk.Canvas(
            card_body,
            width=92,
            height=92,
            bg=self.colors["accent_soft"],
            highlightthickness=0,
            bd=0,
        )
        canvas.pack(pady=(8, 2))
        canvas.create_oval(16, 16, 76, 76, outline="#f2d9cf", width=7)
        arc_id = canvas.create_arc(
            16,
            16,
            76,
            76,
            start=90,
            extent=0,
            style="arc",
            outline=accent,
            width=7,
        )
        canvas.create_oval(
            27,
            27,
            65,
            65,
            fill=self.colors["accent_soft"],
            outline="",
        )
        text_id = canvas.create_text(
            46,
            42,
            text="0%",
            font=(self.display_font, 15, "bold"),
            fill=self.colors["text"],
        )
        subtitle_id = canvas.create_text(
            46,
            56,
            text="等待开始",
            font=(self.ui_font, 8),
            fill=self.colors["muted"],
        )
        status_label = tk.Label(
            card_body,
            text=status_var.get(),
            bg=self.colors["accent_soft"],
            fg=self.colors["muted"],
            font=(self.ui_font, 8),
            wraplength=104,
            justify="center",
        )
        status_label.pack(fill="x", pady=(4, 0))

        self.progress_rings[key] = {
            "canvas": canvas,
            "arc_id": arc_id,
            "text_id": text_id,
            "subtitle_id": subtitle_id,
            "status_widget": status_label,
            "status_var": status_var,
        }
        status_var.trace_add("write", lambda *_args, k=key: self._sync_progress_ring_status(k))
        self._sync_progress_ring_status(key)
        return card

    def _sync_progress_ring_status(self, key: str) -> None:
        ring = self.progress_rings.get(key)
        if not ring:
            return
        canvas = ring["canvas"]
        status_widget = ring.get("status_widget")
        status_var = ring["status_var"]
        if isinstance(status_widget, tk.Label) and isinstance(status_var, tk.StringVar):
            status_widget.configure(text=status_var.get())
        elif isinstance(canvas, tk.Canvas) and isinstance(status_var, tk.StringVar):
            canvas.itemconfigure(ring["status_id"], text=status_var.get())

    def _update_progress_ring(self, key: str, percent: float) -> None:
        ring = self.progress_rings.get(key)
        if not ring:
            return
        clamped = max(0.0, min(100.0, percent))
        canvas = ring["canvas"]
        arc_id = ring["arc_id"]
        text_id = ring["text_id"]
        if isinstance(canvas, tk.Canvas):
            canvas.itemconfigure(arc_id, extent=-(clamped * 3.6))
            canvas.itemconfigure(text_id, text=f"{clamped:.0f}%")
            canvas.itemconfigure(
                ring["subtitle_id"],
                text=("等待开始" if clamped <= 0 else "进行中" if clamped < 100 else "已完成"),
            )

    def _select_tab(self, tab_name: str) -> None:
        self.current_tab = tab_name
        page = self.tab_pages.get(tab_name)
        if page is not None:
            page.lift()
        for name, button in self.tab_buttons.items():
            is_active = name == tab_name
            button.set_palette(**self._button_palette("tab_active" if is_active else "tab_idle"))

    def _build_workbench_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1, minsize=260)

        toolbar = tk.Frame(parent, bg=self.colors["window_bg"], bd=0)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.columnconfigure(0, weight=1)
        ttk.Label(
            toolbar,
            textvariable=self.current_file_text,
            style="ShellMuted.TLabel",
        ).grid(row=0, column=0, sticky="w")

        files_shell, files_frame = self._create_card_block(
            parent,
            title="添加文件",
            padding=(16, 14),
        )
        files_shell.grid(row=1, column=0, sticky="ew")

        list_shell, list_container = self._create_input_shell(
            files_frame,
            height=112,
            radius=16,
            surface_bg=self.colors["card_bg"],
            min_width=400,
            fill_color=self.colors["card_bg"],
            border_color=self.colors["card_border"],
            inner_padx=10,
            inner_pady=10,
        )
        list_shell.pack(fill=BOTH, expand=True, pady=(6, 8))
        self.file_listbox = tk.Listbox(list_container, height=5, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        self._style_text_widget(self.file_listbox)
        self.file_listbox.configure(height=5, activestyle="none")
        scrollbar = ttk.Scrollbar(list_container, orient=VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        if self.dnd_enabled:
            self._bind_drop_targets([files_frame, list_container, self.file_listbox])

        if not self.dnd_enabled and self.dnd_reason:
            ttk.Label(files_frame, text=self.dnd_reason, style="Muted.TLabel").pack(anchor=W)

        btn_row = ttk.Frame(files_frame, style="Card.TFrame")
        btn_row.pack(fill="x")
        self._create_round_button(
            btn_row,
            text="添加文件",
            command=self._pick_files,
            variant="secondary",
            min_width=120,
            height=40,
            radius=18,
            surface_bg=self.colors["card_bg"],
        ).pack(side=LEFT)
        self._create_round_button(
            btn_row,
            text="添加文件夹",
            command=self._pick_folder,
            variant="secondary",
            min_width=120,
            height=40,
            radius=18,
            surface_bg=self.colors["card_bg"],
        ).pack(side=LEFT, padx=10)
        self._create_round_button(
            btn_row,
            text="移除选中",
            command=self._remove_selected,
            variant="secondary",
            min_width=120,
            height=40,
            radius=18,
            surface_bg=self.colors["card_bg"],
        ).pack(side=LEFT)
        self._create_round_button(
            btn_row,
            text="清空",
            command=self._clear_files,
            variant="secondary",
            min_width=100,
            height=40,
            radius=18,
            surface_bg=self.colors["card_bg"],
        ).pack(side=LEFT, padx=10)
        self.run_btn = self._create_round_button(
            btn_row,
            text="开始处理",
            command=self._start,
            variant="primary",
            min_width=124,
            height=40,
            radius=18,
            font_size=12,
            font_weight="bold",
            surface_bg=self.colors["card_bg"],
        )
        self.run_btn.pack(side=LEFT, padx=(8, 0))

        bottom_row = ttk.Frame(parent, style="App.TFrame")
        bottom_row.grid(row=2, column=0, sticky="nsew", pady=(10, 0))
        bottom_row.columnconfigure(0, weight=0)
        bottom_row.columnconfigure(1, weight=4)
        bottom_row.rowconfigure(0, weight=1)

        progress_shell, progress_frame = self._create_card_block(
            bottom_row,
            title="任务进度",
            padding=(12, 12),
        )
        progress_shell.grid(row=0, column=0, sticky="ns", padx=(0, 14))
        progress_shell.configure(width=296, height=184)
        progress_frame.columnconfigure(0, weight=1)
        progress_frame.columnconfigure(1, weight=1)
        ring_row = tk.Frame(progress_frame, bg=self.colors["card_bg"], bd=0)
        ring_row.pack(fill="both", expand=True, pady=(6, 0))
        asr_ring = self._create_progress_ring(
            ring_row,
            key="asr",
            title="语音识别",
            status_var=self.asr_progress_text,
            accent=self.colors["accent"],
        )
        asr_ring.pack(side=LEFT, fill="both", expand=True, padx=(0, 6))

        translate_ring = self._create_progress_ring(
            ring_row,
            key="translate",
            title="字幕翻译",
            status_var=self.translate_progress_text,
            accent=self.colors["accent_alt"],
        )
        translate_ring.pack(side=LEFT, fill="both", expand=True, padx=(6, 0))

        log_shell, log_frame = self._create_card_block(
            bottom_row,
            title="日志",
            padding=(16, 14),
        )
        log_shell.grid(row=0, column=1, sticky="nsew")
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_surface, log_inner = self._create_input_shell(
            log_frame,
            height=240,
            radius=16,
            surface_bg=self.colors["card_bg"],
            min_width=420,
            fill_color=self.colors["log_bg"],
            border_color=self.colors["card_border"],
            inner_padx=12,
            inner_pady=12,
        )
        log_surface.grid(row=0, column=0, sticky="nsew")
        self.log_text = tk.Text(log_inner, height=10, state=tk.DISABLED)
        self.log_text.pack(fill=BOTH, expand=True)
        self._style_text_widget(self.log_text, is_log=True)
        self.log_text.configure(padx=14, pady=14, spacing1=3, spacing3=3)

    def _build_settings_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

        tk.Label(
            parent,
            text="设置页收纳不需要频繁修改的参数，工作台只保留高频操作与反馈。",
            bg=self.colors["window_bg"],
            fg=self.colors["muted"],
            font=(self.ui_font, 12),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))

        asr_shell, asr_frame = self._create_card_block(
            parent,
            title="ASR 与输出",
            subtitle="控制识别引擎、输出目录与字幕切分参数。",
            padding=(18, 16),
        )
        asr_shell.grid(row=1, column=0, sticky="nsew", padx=(0, 9))
        trans_shell, trans_frame = self._create_card_block(
            parent,
            title="翻译与服务",
            subtitle="选择翻译后端、模型与在线服务配置。",
            padding=(18, 16),
        )
        trans_shell.grid(row=1, column=1, sticky="nsew", padx=(9, 0))

        ttk.Label(asr_frame, text="输出目录", style="Body.TLabel").grid(
            row=0, column=0, sticky=W, padx=(0, 8), pady=6
        )
        output_shell, output_inner = self._create_input_shell(
            asr_frame,
            min_width=300,
            surface_bg=self.colors["card_bg"],
        )
        output_shell.grid(row=0, column=1, sticky="ew", pady=4)
        self.output_dir_entry = tk.Entry(output_inner, state="disabled", width=42)
        self.output_dir_entry.pack(fill="x")
        self._style_flat_input_widget(self.output_dir_entry)
        self._render_output_dir(self.output_dir.get())
        self._create_round_button(
            asr_frame,
            text="浏览",
            command=self._pick_output_dir,
            variant="secondary",
            min_width=90,
            height=38,
            radius=16,
            surface_bg=self.colors["card_bg"],
        ).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(asr_frame, text="ASR 引擎", style="Body.TLabel").grid(
            row=1, column=0, sticky=W, padx=(0, 8), pady=6
        )
        asr_engine_shell, asr_engine_inner = self._create_input_shell(
            asr_frame,
            min_width=220,
            surface_bg=self.colors["card_bg"],
        )
        asr_engine_shell.grid(row=1, column=1, columnspan=2, sticky="w", pady=4)
        ttk.Combobox(
            asr_engine_inner,
            textvariable=self.asr_engine,
            values=["whisper", "faster-whisper", "deepgram"],
            state="readonly",
            width=20,
            style="Flat.TCombobox",
        ).pack(fill="x")

        ttk.Label(asr_frame, text="ASR 模型", style="Body.TLabel").grid(
            row=2, column=0, sticky=W, padx=(0, 8), pady=6
        )
        whisper_shell, whisper_inner = self._create_input_shell(
            asr_frame,
            min_width=220,
            surface_bg=self.colors["card_bg"],
        )
        whisper_shell.grid(row=2, column=1, columnspan=2, sticky="w", pady=4)
        self.whisper_model_combo = ttk.Combobox(
            whisper_inner,
            textvariable=self.whisper_model,
            values=MODEL_PRESETS_DEFAULT,
            state="normal",
            width=20,
            style="Flat.TCombobox",
        )
        self.whisper_model_combo.pack(fill="x")

        ttk.Label(asr_frame, text="Deepgram API Key", style="Body.TLabel").grid(
            row=3, column=0, sticky=W, padx=(0, 8), pady=6
        )
        api_shell, api_inner = self._create_input_shell(
            asr_frame,
            min_width=260,
            surface_bg=self.colors["card_bg"],
        )
        api_shell.grid(row=3, column=1, columnspan=2, sticky="w", pady=4)
        api_entry = tk.Entry(api_inner, textvariable=self.deepgram_api_key, show="*", width=28)
        api_entry.pack(fill="x")
        self._style_flat_input_widget(api_entry)

        ttk.Label(asr_frame, text="每条最大时长(秒)", style="Body.TLabel").grid(
            row=4, column=0, sticky=W, padx=(0, 8), pady=6
        )
        duration_shell, duration_inner = self._create_input_shell(
            asr_frame,
            min_width=120,
            surface_bg=self.colors["card_bg"],
        )
        duration_shell.grid(row=4, column=1, sticky="w", pady=4)
        duration_entry = tk.Entry(duration_inner, textvariable=self.max_duration, width=12)
        duration_entry.pack(fill="x")
        self._style_flat_input_widget(duration_entry)

        ttk.Checkbutton(trans_frame, text="翻译为简体中文", variable=self.translate_enabled).grid(
            row=0, column=0, columnspan=2, sticky=W, pady=(0, 8)
        )
        ttk.Label(trans_frame, text="翻译后端", style="Body.TLabel").grid(
            row=1, column=0, sticky=W, padx=(0, 8), pady=6
        )
        backend_shell, backend_inner = self._create_input_shell(
            trans_frame,
            min_width=220,
            surface_bg=self.colors["card_bg"],
        )
        backend_shell.grid(row=1, column=1, sticky="w", pady=4)
        self.translate_backend_combo = ttk.Combobox(
            backend_inner,
            textvariable=self.translate_backend,
            values=["local", "openai", "deepseek", "qwen", "minimax"],
            width=18,
            state="readonly",
            style="Flat.TCombobox",
        )
        self.translate_backend_combo.pack(fill="x")
        self.translate_backend_combo.bind("<<ComboboxSelected>>", self._on_translate_backend_change)

        ttk.Label(trans_frame, text="本地模型", style="Body.TLabel").grid(
            row=2, column=0, sticky=W, padx=(0, 8), pady=6
        )
        local_shell, local_inner = self._create_input_shell(
            trans_frame,
            min_width=280,
            surface_bg=self.colors["card_bg"],
        )
        local_shell.grid(row=2, column=1, sticky="w", pady=4)
        self.local_model_combo = ttk.Combobox(
            local_inner,
            textvariable=self.translate_model,
            values=LOCAL_TRANSLATION_MODELS,
            width=30,
            state="readonly",
            style="Flat.TCombobox",
        )
        self.local_model_combo.pack(fill="x")
        self.local_model_combo.bind("<<ComboboxSelected>>", self._on_local_model_change)

        ttk.Label(trans_frame, text="最大 Token 数", style="Body.TLabel").grid(
            row=3, column=0, sticky=W, padx=(0, 8), pady=6
        )
        token_shell, token_inner = self._create_input_shell(
            trans_frame,
            min_width=140,
            surface_bg=self.colors["card_bg"],
        )
        token_shell.grid(row=3, column=1, sticky="w", pady=4)
        token_entry = tk.Entry(token_inner, textvariable=self.translate_max_tokens, width=12)
        token_entry.pack(fill="x")
        self._style_flat_input_widget(token_entry)
        ttk.Label(
            trans_frame,
            textvariable=self.max_tokens_hint,
            style="Muted.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky=W, pady=(2, 4))
        ttk.Label(
            trans_frame,
            text=f"在线后端配置文件: {DEFAULT_TRANSLATE_CONFIG_PATH}",
            style="Muted.TLabel",
        ).grid(row=5, column=0, columnspan=2, sticky=W, pady=(0, 4))
        ttk.Label(trans_frame, text="翻译服务状态", style="Body.TLabel").grid(
            row=6, column=0, sticky=W, padx=(0, 8), pady=6
        )
        status_row = ttk.Frame(trans_frame, style="Card.TFrame")
        status_row.grid(row=6, column=1, sticky="ew", pady=4)
        self.ollama_status_label = tk.Label(
            status_row,
            textvariable=self.ollama_status_text,
            fg=self.colors["danger"],
            anchor="w",
            bg=self.colors["card_bg"],
            font=(self.ui_font, 12, "bold"),
        )
        self.ollama_status_label.pack(side=LEFT, fill=BOTH, expand=True)
        self._create_round_button(
            status_row,
            text="刷新检测",
            command=self._refresh_ollama_status_async,
            variant="secondary",
            min_width=108,
            height=38,
            radius=16,
            surface_bg=self.colors["card_bg"],
        ).pack(side=RIGHT)

        asr_frame.columnconfigure(1, weight=1)
        trans_frame.columnconfigure(1, weight=1)

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(18, 16, 18, 16))
        shell.pack(fill=BOTH, expand=True)

        hero_shell, hero = self._create_soft_card(
            shell,
            padding=(18, 14),
            background=self.colors["hero_bg"],
            corner_radius=26,
        )
        hero_shell.pack(fill="x", pady=(0, 10))

        header_row = tk.Frame(hero, bg=self.colors["hero_bg"], bd=0)
        header_row.pack(fill="x")
        title_group = tk.Frame(header_row, bg=self.colors["hero_bg"], bd=0)
        title_group.pack(side=LEFT, fill="x", expand=True)
        ttk.Label(hero, text="自动字幕生成", style="HeroHeadline.TLabel").pack(anchor=W)
        ttk.Label(
            hero,
            text="拖入视频后即可生成源字幕与中文字幕。",
            style="Subtitle.TLabel",
        ).pack(anchor=W, pady=(4, 0))

        top = ttk.Frame(shell, style="App.TFrame")
        top.pack(fill=BOTH, expand=True)

        tab_bar = tk.Frame(top, bg=self.colors["window_bg"], bd=0)
        tab_bar.pack(fill="x", pady=(0, 8))
        segmented_shell, segmented = self._create_soft_card(
            tab_bar,
            padding=(6, 6),
            background="#fff3ec",
            corner_radius=22,
        )
        segmented_shell.pack(anchor="w")
        for key, label in (("workbench", "工作台"), ("settings", "设置")):
            button = self._create_round_button(
                segmented,
                text=label,
                command=lambda value=key: self._select_tab(value),
                variant="tab_idle",
                min_width=94,
                height=36,
                radius=15,
                font_size=11,
                font_weight="bold",
                surface_bg="#fff3ec",
            )
            button.pack(side=LEFT, padx=3)
            self.tab_buttons[key] = button

        content_shell, content_host = self._create_soft_card(
            top,
            padding=(10, 12),
            background=self.colors["window_bg"],
            corner_radius=30,
        )
        content_shell.pack(fill=BOTH, expand=True)

        workbench_tab = ttk.Frame(content_host, style="App.TFrame", padding=(8, 18, 8, 8))
        settings_tab = ttk.Frame(content_host, style="App.TFrame", padding=(8, 18, 8, 8))
        for key, frame in (("workbench", workbench_tab), ("settings", settings_tab)):
            frame.place(relx=0, rely=0, relwidth=1, relheight=1)
            self.tab_pages[key] = frame

        self._build_workbench_tab(workbench_tab)
        self._build_settings_tab(settings_tab)
        self._select_tab("workbench")
        self._update_progress_ring("asr", 0.0)
        self._update_progress_ring("translate", 0.0)
        self._refresh_asr_model_choices()
        self._refresh_local_model_widget()
        self.root.after(200, self._refresh_ollama_status_async)
        self.root.after(350, self._load_local_models_async)
        if not self.dnd_enabled and self.dnd_reason:
            self._append_log_text(self.dnd_reason)

    def _add_labeled_entry(
        self,
        parent: ttk.Frame,
        label: str,
        var: tk.StringVar,
        row: int,
        browse=None,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            self._create_round_button(
                parent,
                text="浏览",
                command=browse,
                variant="secondary",
                min_width=90,
                height=38,
                radius=16,
                surface_bg=self.colors["card_bg"],
            ).grid(row=row, column=2, padx=(8, 0), pady=4)

    def _pick_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="选择视频文件",
            filetypes=[("Video Files", "*.mp4 *.mov *.mkv *.avi *.m4v *.webm *.flv *.wmv"), ("All Files", "*.*")],
        )
        self._add_paths([Path(p) for p in paths])

    def _pick_folder(self) -> None:
        folder = filedialog.askdirectory(title="选择包含视频的文件夹")
        if not folder:
            return
        videos = expand_videos([folder])
        self._add_paths(videos)

    def _pick_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="选择输出目录")
        if folder:
            selected = str(Path(folder).resolve())
            self.output_dir.set(selected)
            self._render_output_dir(selected)
            self._log(f"输出目录已选择: {selected}")

    def _bind_drop_targets(self, widgets: list[tk.Widget]) -> None:
        for widget in widgets:
            try:
                widget.drop_target_register("DND_Files")
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception as exc:
                self._log(f"拖拽绑定失败: {exc}")

    def _on_drop(self, event) -> None:
        dropped: list[Path] = []
        raw = str(getattr(event, "data", "") or "")
        if raw:
            try:
                dropped = [
                    Path(p).expanduser().resolve()
                    for p in self.root.tk.splitlist(raw)
                    if str(p).strip()
                ]
            except Exception:
                dropped = _parse_drop_files(raw)
        self._add_paths(dropped)

    def _add_paths(self, paths: list[Path]) -> None:
        items = expand_videos([str(p) for p in paths])
        existing = {str(p) for p in self.files}
        added = 0
        first_added_parent: Path | None = None
        for path in items:
            if path.suffix.lower() not in VIDEO_EXTENSIONS:
                continue
            k = str(path)
            if k in existing:
                continue
            self.files.append(path)
            self.file_listbox.insert(END, k)
            existing.add(k)
            if first_added_parent is None:
                first_added_parent = path.parent
            added += 1
        if added:
            self._log(f"已添加 {added} 个视频文件")
            if first_added_parent is not None:
                auto_out = str(first_added_parent.resolve())
                self.output_dir.set(auto_out)
                self._render_output_dir(auto_out)
                self._log(f"输出目录已自动切换到视频目录: {auto_out}")

    def _remove_selected(self) -> None:
        selected = list(self.file_listbox.curselection())
        if not selected:
            return
        for idx in reversed(selected):
            self.file_listbox.delete(idx)
            del self.files[idx]

    def _clear_files(self) -> None:
        self.file_listbox.delete(0, END)
        self.files.clear()

    def _start(self) -> None:
        if self.running:
            return
        if not self.files:
            messagebox.showwarning("提示", "请先添加至少一个视频文件。")
            return

        try:
            max_duration = float(self.max_duration.get().strip())
            max_tokens = int(self.translate_max_tokens.get().strip())
        except ValueError:
            messagebox.showerror("参数错误", "请检查时长/最大Token数字段格式。")
            return

        do_translate = self.translate_enabled.get()
        backend_raw = self.translate_backend_combo.get() if hasattr(self, "translate_backend_combo") else self.translate_backend.get()
        translate_backend = self._normalize_backend(backend_raw)
        self._log(f"翻译后端选择: raw='{backend_raw}' -> normalized='{translate_backend}'")
        translation_settings: dict[str, str] | None = None
        if do_translate:
            try:
                translation_settings = resolve_translation_settings(
                    backend=translate_backend,
                    model_name=self.translate_model.get().strip() if translate_backend == "local" else None,
                    config_path=DEFAULT_TRANSLATE_CONFIG_PATH,
                )
            except RuntimeError as exc:
                messagebox.showerror("翻译配置错误", str(exc))
                return

        if do_translate and translate_backend == "local":
            resolved_for_run = self._resolve_local_model_via_api(
                translation_settings["model"], translation_settings["base_url"]
            )
            if resolved_for_run and resolved_for_run != translation_settings["model"]:
                translation_settings["model"] = resolved_for_run
                self.translate_model.set(resolved_for_run)
                self._log(f"已自动匹配本地模型: {resolved_for_run}")
            ok, msg = self._check_ollama_status(
                backend=translate_backend,
                translate_enabled=do_translate,
                model_name=translation_settings["model"],
                base_url=translation_settings["base_url"],
            )
            self._set_ollama_status(ok, msg)
            if not ok:
                messagebox.showerror("Ollama 未就绪", msg)
                return

        out_dir = Path(self.output_dir.get().strip()).expanduser().resolve()
        self._render_output_dir(str(out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)

        whisper_model = self.whisper_model.get().strip()
        if not whisper_model:
            messagebox.showerror("参数错误", "ASR 模型不能为空。")
            return
        asr_engine = self.asr_engine.get().strip() or "deepgram"
        deepgram_api_key = self.deepgram_api_key.get().strip()
        if asr_engine == "deepgram" and not deepgram_api_key:
            deepgram_api_key, resolved_model = resolve_deepgram_settings(
                api_key=deepgram_api_key,
                model_name=whisper_model,
                config_path=DEFAULT_ASR_CONFIG_PATH,
            )
            if resolved_model:
                whisper_model = resolved_model
                self.whisper_model.set(resolved_model)
            if not deepgram_api_key:
                messagebox.showerror(
                    "参数错误",
                    "Deepgram API Key 不能为空（可在界面填写一次后自动保存到 config/asr.toml）。",
                )
                return
        if asr_engine == "deepgram":
            deepgram_api_key, resolved_model = resolve_deepgram_settings(
                api_key=deepgram_api_key,
                model_name=whisper_model,
                config_path=DEFAULT_ASR_CONFIG_PATH,
            )
            whisper_model = resolved_model
            self.whisper_model.set(resolved_model)
            self.deepgram_api_key.set(deepgram_api_key)
            cfg_path = save_deepgram_settings(
                api_key=deepgram_api_key,
                model_name=whisper_model,
                config_path=DEFAULT_ASR_CONFIG_PATH,
            )
            self._log(f"Deepgram 配置已保存: {cfg_path}")

        self.running = True
        self.run_btn.config(state=tk.DISABLED)

        worker = threading.Thread(
            target=self._run_pipeline,
            args=(
                list(self.files),
                out_dir,
                asr_engine,
                whisper_model,
                max_duration,
                do_translate,
                translate_backend,
                (translation_settings["model"] if do_translate else ""),
                (translation_settings["base_url"] if do_translate else ""),
                (translation_settings["api_key"] if do_translate else ""),
                max_tokens,
                deepgram_api_key,
            ),
            daemon=True,
        )
        worker.start()

    def _run_pipeline(
        self,
        files: list[Path],
        out_dir: Path,
        asr_engine: str,
        whisper_model: str,
        max_duration: float,
        do_translate: bool,
        translate_backend: str,
        translate_model: str,
        translate_base_url: str,
        translate_api_key: str,
        max_tokens: int,
        deepgram_api_key: str,
    ) -> None:
        try:
            if asr_engine == "faster-whisper":
                self._log("正在检查 Faster-Whisper 依赖...")
                ensure_faster_whisper_installed()
            elif asr_engine == "whisper":
                self._log("正在检查 Whisper 依赖...")
                ensure_openai_whisper_installed()
            elif asr_engine == "deepgram":
                self._log("使用 Deepgram 在线 ASR。")
            else:
                raise RuntimeError(f"暂不支持的ASR引擎: {asr_engine}")

            self._log(f"正在加载 ASR 模型 ({asr_engine}): {whisper_model}")
            model = load_asr_model(asr_engine, whisper_model)
            multi_video_mode = len(files) > 1

            for video in files:
                self._log(f"处理: {video}")
                self.log_queue.put(("__PROGRESS_RESET__", video.name, do_translate))
                target_out_dir = video.parent if multi_video_mode else out_dir
                target_out_dir.mkdir(parents=True, exist_ok=True)
                if multi_video_mode:
                    self._log(f"多文件模式输出目录: {target_out_dir}")

                def on_progress(
                    task: str,
                    current: float,
                    total: float,
                    label: str | None,
                ) -> None:
                    self.log_queue.put(("__PROGRESS__", task, current, total, label))

                source_path = target_out_dir / f"{video.stem}.source.srt"
                if source_path.exists():
                    entries = read_srt(source_path)
                    if entries:
                        self._log(f"检测到已有源字幕，跳过Whisper: {source_path}")
                        self.log_queue.put(("__PROGRESS__", "asr", 1.0, 1.0, video.name))
                    else:
                        self._log(f"已有源字幕为空或不可解析，重新生成: {source_path}")
                        entries = transcribe_with_asr_engine(
                            asr_engine=asr_engine,
                            model=model,
                            video_path=video,
                            source_language=None,
                            max_segment_duration=max_duration,
                            max_segment_chars=28,
                            progress_label=video.name,
                            progress_callback=on_progress,
                            deepgram_api_key=deepgram_api_key,
                        )
                        write_srt(entries, source_path)
                        self._log(f"已重新输出源字幕: {source_path}")
                else:
                    entries = transcribe_with_asr_engine(
                        asr_engine=asr_engine,
                        model=model,
                        video_path=video,
                        source_language=None,
                        max_segment_duration=max_duration,
                        max_segment_chars=28,
                        progress_label=video.name,
                        progress_callback=on_progress,
                        deepgram_api_key=deepgram_api_key,
                    )
                    write_srt(entries, source_path)
                    self._log(f"已输出源字幕: {source_path}")
                if asr_engine == "deepgram":
                    lang, conf = get_last_deepgram_detected_language()
                    if lang:
                        self._log(f"Deepgram 检测语言: {lang} (confidence={conf:.3f})")

                if do_translate:
                    self._log(f"翻译后端: {translate_backend} | base_url: {translate_base_url}")
                    if translate_backend == "local":
                        self._log(f"使用本地翻译模型: {translate_model}")
                        local_m = translate_model.strip().lower()
                        if local_m.startswith("qwen2.5") or "dolphin3" in local_m:
                            self._log("本地模型策略: 已启用逐行上下文翻译与重复行纠正")
                    elif translate_backend == "openai":
                        self._log(f"使用 OpenAI 翻译模型: {translate_model}")
                    elif translate_backend == "qwen":
                        self._log(f"使用 百炼Qwen 翻译模型: {translate_model}")
                    elif translate_backend == "minimax":
                        self._log(f"使用 百炼MiniMax 翻译模型: {translate_model}")
                    else:
                        self._log(f"使用 DeepSeek 翻译模型: {translate_model}")
                    self._log("正在执行：全文翻译 + 按时间戳语义回填...")
                    zh = translate_fulltext_then_llm_refill(
                        entries,
                        model_name=translate_model,
                        base_url=translate_base_url,
                        api_key=translate_api_key,
                        max_tokens=max_tokens,
                        progress_label=video.name,
                        progress_callback=on_progress,
                    )
                    zh_entries = [
                        SubtitleEntry(
                            index=entries[idx].index,
                            start=entries[idx].start,
                            end=entries[idx].end,
                            text=line,
                        )
                        for idx, line in enumerate(zh)
                    ]
                    zh_entries = resegment_translated_entries(
                        zh_entries,
                        max_duration=max_duration,
                        max_chars=28,
                    )
                    model_tag = _slugify_model_tag(f"{translate_backend}-{translate_model}")
                    zh_path = target_out_dir / f"{video.stem}.zh-CN.{model_tag}.srt"
                    write_srt(zh_entries, zh_path)
                    self._log(f"已输出中文字幕: {zh_path}")

            self._log("全部任务完成。")
        except Exception as exc:
            self._log(f"发生错误: {exc}")
        finally:
            self.log_queue.put("__DONE__")

    def _log(self, message: str) -> None:
        self.log_queue.put(message)

    def _on_asr_engine_change(self, *_args) -> None:
        self._refresh_asr_model_choices()

    def _refresh_asr_model_choices(self) -> None:
        engine = (self.asr_engine.get().strip() or "deepgram").lower()
        try:
            choices = get_asr_model_choices(engine)
        except Exception:
            # Fallback to common models if package is not ready yet.
            choices = MODEL_PRESETS_DEFAULT
        if not choices:
            choices = MODEL_PRESETS_DEFAULT
        self.whisper_model_combo["values"] = choices
        current = self.whisper_model.get().strip()
        if current not in choices:
            prefer = "medium" if "medium" in choices else choices[0]
            self.whisper_model.set(prefer)

    def _on_translate_backend_change(self, *_args) -> None:
        backend_raw = self.translate_backend_combo.get() if hasattr(self, "translate_backend_combo") else self.translate_backend.get()
        backend = self._normalize_backend(backend_raw)
        self.translate_backend.set(backend)
        self.max_tokens_hint.set(TOKEN_RECOMMENDATIONS.get(backend, TOKEN_RECOMMENDATIONS["local"]))
        if backend == "local":
            old_model = self.translate_model.get().strip()
            default_models = {v["model"] for v in BACKEND_DEFAULTS.values()}
            if not old_model or old_model in default_models:
                self.translate_model.set(BACKEND_DEFAULTS["local"]["model"])
        else:
            try:
                settings = resolve_translation_settings(
                    backend=backend,
                    config_path=DEFAULT_TRANSLATE_CONFIG_PATH,
                )
                self.translate_model.set(settings["model"])
                self._log(
                    f"{backend} 已从配置加载模型: {settings['model']}"
                )
            except RuntimeError as exc:
                self.translate_model.set(BACKEND_DEFAULTS[backend]["model"])
                self._log(f"{backend} 配置加载失败: {exc}")
        self._refresh_local_model_widget()
        self._refresh_ollama_status_async()

    def _refresh_local_model_widget(self) -> None:
        backend_raw = (
            self.translate_backend_combo.get()
            if hasattr(self, "translate_backend_combo")
            else self.translate_backend.get()
        )
        backend = self._normalize_backend(backend_raw)
        if backend == "local":
            self.local_model_combo.configure(state="readonly")
            model = self.translate_model.get().strip()
            values = list(self.local_model_combo.cget("values") or [])
            if not values:
                values = LOCAL_TRANSLATION_MODELS
                self.local_model_combo["values"] = values
            if model not in values:
                prefer = "qwen2.5:7b" if "qwen2.5:7b" in values else values[0]
                self.translate_model.set(prefer)
        else:
            self.local_model_combo.configure(state="disabled")

    def _on_local_model_change(self, _event=None) -> None:
        model_name = self.translate_model.get().strip()
        if model_name:
            self._log(f"本地翻译模型已切换: {model_name}")
        self._refresh_ollama_status_async()

    def _load_local_models_async(self) -> None:
        def worker() -> None:
            models: list[str] = []
            err: str | None = None
            try:
                out = subprocess.check_output(
                    ["ollama", "list"], stderr=subprocess.STDOUT, text=True, timeout=3.0
                )
                models = _parse_ollama_list_names(out)
            except Exception as exc:
                err = str(exc)
                # Fallback to API tags if CLI invocation fails in runtime environment.
                try:
                    resolved = self._resolve_local_model_via_api(
                        model_name=self.translate_model.get().strip(),
                        base_url=BACKEND_DEFAULTS["local"]["base_url"],
                    )
                    if resolved:
                        models = [resolved]
                except Exception:
                    pass
            self.log_queue.put(("__LOCAL_MODELS__", models, err))

        threading.Thread(target=worker, daemon=True).start()

    def _apply_local_models(self, models: list[str]) -> None:
        merged: list[str] = []
        seen: set[str] = set()
        for m in models + LOCAL_TRANSLATION_MODELS:
            name = (m or "").strip()
            if not name or name in seen:
                continue
            seen.add(name)
            merged.append(name)

        if not merged:
            return

        self.local_model_combo["values"] = merged
        current = self.translate_model.get().strip()
        if not current:
            self.translate_model.set(merged[0])
            return
        if current not in merged:
            fallback = "qwen2.5:7b" if "qwen2.5:7b" in merged else merged[0]
            self.translate_model.set(fallback)

    def _refresh_ollama_status_async(self) -> None:
        backend_raw = self.translate_backend_combo.get() if hasattr(self, "translate_backend_combo") else self.translate_backend.get()
        backend = self._normalize_backend(backend_raw)
        translate_enabled = bool(self.translate_enabled.get())
        model_name = self.translate_model.get().strip()
        base_url = BACKEND_DEFAULTS.get(backend, BACKEND_DEFAULTS["local"])["base_url"]
        self._append_log_text("正在检测翻译服务状态...")

        def worker() -> None:
            try:
                ok, msg = self._check_ollama_status(
                    backend=backend,
                    translate_enabled=translate_enabled,
                    model_name=model_name,
                    base_url=base_url,
                )
            except Exception as exc:
                ok, msg = False, f"状态检测异常: {exc}"
            self.log_queue.put(("__OLLAMA_STATUS__", ok, msg, backend))

        threading.Thread(target=worker, daemon=True).start()

    def _check_ollama_status(
        self,
        backend: str,
        translate_enabled: bool,
        model_name: str,
        base_url: str,
    ) -> tuple[bool, str]:
        if not translate_enabled:
            return True, "翻译已关闭"
        if backend != "local":
            if backend == "openai":
                return True, "当前使用 OpenAI 后端"
            if backend == "qwen":
                return True, "当前使用百炼 Qwen 后端"
            if backend == "minimax":
                return True, "当前使用百炼 MiniMax 后端"
            return True, "当前使用 DeepSeek 后端"

        base_url = base_url.rstrip("/")
        parsed = urlparse.urlparse(base_url)
        origin = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else "http://localhost:11434"
        models_url = f"{origin}/api/tags"

        try:
            req = urlrequest.Request(models_url, method="GET")
            with urlrequest.urlopen(req, timeout=1.5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            models = payload.get("models", []) if isinstance(payload, dict) else []
            names = [
                str(m.get("name", "")).strip() for m in models if isinstance(m, dict)
            ]

            if not model_name:
                return False, "翻译模型为空，请设置模型名"
            resolved = self._resolve_ollama_model_alias(model_name, names)
            if resolved:
                return True, f"可用: {resolved}" if resolved == model_name else f"可用: {resolved}（匹配到同系列模型）"
            return False, f"服务在线，但未找到模型 {model_name}（请先 ollama pull）"
        except (urlerror.URLError, TimeoutError, ValueError):
            return False, "无法连接 Ollama（请确认 ollama serve 已启动）"

    def _resolve_ollama_model_alias(
        self,
        wanted_model: str,
        installed_names: list[str],
    ) -> str | None:
        wanted = (wanted_model or "").strip()
        names = [n for n in installed_names if n]
        if not wanted or not names:
            return None
        if wanted in names:
            return wanted

        wanted_repo = wanted.split(":", 1)[0]
        family = [n for n in names if n.split(":", 1)[0] == wanted_repo]
        if not family and "dolphin3" in wanted.lower():
            family = [n for n in names if "dolphin3" in n.lower()]
        if not family:
            return None

        preferred_order = [f"{wanted_repo}:8b", f"{wanted_repo}:latest", wanted_repo]
        if "dolphin3" in wanted.lower():
            preferred_order = [
                "nchapman/dolphin3.0-llama3:8b",
                "dolphin3:8b",
                "dolphin3:latest",
                "huihui_ai/dolphin3-abliterated",
            ] + preferred_order
        for pref in preferred_order:
            if pref in family:
                return pref
        return sorted(family)[0]

    def _resolve_local_model_via_api(self, model_name: str, base_url: str) -> str | None:
        base_url = (base_url or "").rstrip("/")
        parsed = urlparse.urlparse(base_url)
        origin = (
            f"{parsed.scheme}://{parsed.netloc}"
            if parsed.scheme and parsed.netloc
            else "http://localhost:11434"
        )
        models_url = f"{origin}/api/tags"
        try:
            req = urlrequest.Request(models_url, method="GET")
            with urlrequest.urlopen(req, timeout=1.5) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            payload = json.loads(body)
            models = payload.get("models", []) if isinstance(payload, dict) else []
            names = [str(m.get("name", "")).strip() for m in models if isinstance(m, dict)]
            return self._resolve_ollama_model_alias(model_name, names)
        except Exception:
            return None

    def _set_ollama_status(self, ok: bool, text: str) -> None:
        self.ollama_status_text.set(text)
        self.ollama_status_label.config(fg=("#1f7a3a" if ok else "#d64545"))

    def _append_log_text(self, message: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(END, message + "\n")
        self.log_text.see(END)
        self.log_text.config(state=tk.DISABLED)

    def _format_progress_text(
        self,
        task: str,
        percent: float,
        current: float,
        total: float,
    ) -> str:
        if task == "asr":
            return f"ASR: {percent:.1f}% ({current:.1f}s/{total:.1f}s)"
        if task == "align":
            return f"校正: {percent:.1f}% ({int(current)}/{int(total)})"
        return f"翻译: {percent:.1f}% ({int(current)}/{int(total)})"

    def _reset_progress_ui(self, file_label: str, do_translate: bool) -> None:
        self.current_file_text.set(f"当前文件: {file_label}")
        self.asr_progress_var.set(0.0)
        self.asr_progress_text.set("ASR: 0.0%")
        self.translate_progress_var.set(0.0)
        self._update_progress_ring("asr", 0.0)
        self._update_progress_ring("translate", 0.0)
        if do_translate:
            self.translate_progress_text.set("翻译: 0.0%")
        else:
            self.translate_progress_text.set("翻译: 已跳过")

    def _apply_progress_update(
        self,
        task: str,
        current: float,
        total: float,
        label: str | None,
    ) -> None:
        if label:
            self.current_file_text.set(f"当前文件: {label}")
        if total <= 0:
            percent = 0.0
        else:
            percent = max(0.0, min(100.0, (current / total) * 100.0))
        text = self._format_progress_text(task, percent, current, total if total > 0 else 0.0)
        if task == "asr":
            self.asr_progress_var.set(percent)
            self.asr_progress_text.set(text)
            self._update_progress_ring("asr", percent)
        elif task in {"translate", "align"}:
            self.translate_progress_var.set(percent)
            self.translate_progress_text.set(text)
            self._update_progress_ring("translate", percent)

    def _render_output_dir(self, path: str) -> None:
        self.output_dir_entry.config(state="normal")
        self.output_dir_entry.delete(0, END)
        self.output_dir_entry.insert(0, path)
        self.output_dir_entry.xview_moveto(0)
        self.output_dir_entry.config(state="disabled")

    def _normalize_backend(self, value: str | None) -> str:
        raw = (value or "").strip().lower()
        normalized = raw.replace(" ", "").replace("_", "").replace("-", "")
        mapping = {
            "local": "local",
            "openai": "openai",
            "deepseek": "deepseek",
            "qwen": "qwen",
            "minimax": "minimax",
            "bailianminimax": "minimax",
        }
        return mapping.get(normalized, "local")

    def _poll_logs(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if isinstance(msg, tuple) and len(msg) == 4 and msg[0] == "__OLLAMA_STATUS__":
                    ok = bool(msg[1])
                    text = str(msg[2])
                    backend = str(msg[3])
                    self._set_ollama_status(ok, text)
                    self._append_log_text(f"状态检测[{backend}]: {text}")
                    continue
                if isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__LOCAL_MODELS__":
                    models = msg[1] if isinstance(msg[1], list) else []
                    err = str(msg[2]) if msg[2] else ""
                    if models:
                        self._apply_local_models(models)
                        self._append_log_text(
                            f"已从 ollama list 加载本地模型: {', '.join(models[:8])}"
                        )
                    elif err:
                        self._append_log_text(f"读取 ollama list 失败: {err}")
                    continue
                if isinstance(msg, tuple) and len(msg) == 3 and msg[0] == "__PROGRESS_RESET__":
                    file_label = str(msg[1])
                    do_translate = bool(msg[2])
                    self._reset_progress_ui(file_label, do_translate)
                    continue
                if isinstance(msg, tuple) and len(msg) == 5 and msg[0] == "__PROGRESS__":
                    task = str(msg[1]).strip().lower()
                    current = float(msg[2])
                    total = float(msg[3])
                    label = str(msg[4]) if msg[4] is not None else None
                    self._apply_progress_update(task, current, total, label)
                    continue
                if msg == "__DONE__":
                    self.running = False
                    self.run_btn.config(state=tk.NORMAL)
                    continue
                self._append_log_text(str(msg))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_logs)


def main() -> None:
    dnd_enabled = False
    dnd_reason: str | None = None
    _dnd_files, tkdnd_mod, load_err = _load_tkdnd()
    if load_err:
        dnd_reason = load_err

    if tkdnd_mod is None:
        root = tk.Tk()
    else:
        try:
            root = tkdnd_mod.Tk()
            dnd_enabled = True
        except Exception as exc:
            # One automatic repair attempt for tkdnd binary mismatch.
            _dnd_files2, tkdnd_mod2, repair_err = _repair_tkdnd_and_reload()
            if tkdnd_mod2 is not None:
                try:
                    root = tkdnd_mod2.Tk()
                    dnd_enabled = True
                    dnd_reason = "拖拽扩展已自动修复并启用"
                except Exception as exc2:
                    dnd_reason = f"拖拽扩展初始化失败: {exc}; 修复后仍失败: {exc2}"
                    root = tk.Tk()
            else:
                dnd_reason = f"拖拽扩展初始化失败: {exc}; {repair_err or ''}".strip()
                root = tk.Tk()
    App(root, dnd_enabled=dnd_enabled, dnd_reason=dnd_reason)
    root.mainloop()


if __name__ == "__main__":
    main()
