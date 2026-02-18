from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import importlib
from pathlib import Path
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
    redistribute_translated_by_source_timestamps,
    resolve_translation_settings,
    resolve_deepgram_settings,
    save_deepgram_settings,
    transcribe_with_asr_engine,
    translate_entries_contextual,
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
}

LOCAL_TRANSLATION_MODELS = [
    "qwen2.5:7b",
    "dolphin3:8b",
    "dolphin3:latest",
    "nchapman/dolphin3.0-llama3:8b",
    "huihui_ai/dolphin3-abliterated",
]


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


class App:
    def __init__(self, root: tk.Tk, dnd_enabled: bool, dnd_reason: str | None = None):
        self.root = root
        self.root.title("SubGen - 自动字幕生成")
        self.root.geometry("980x680")
        self.dnd_enabled = dnd_enabled

        self.files: list[Path] = []
        self.log_queue: queue.Queue[object] = queue.Queue()
        self.running = False
        self.dnd_reason = dnd_reason or ""

        self.output_dir = tk.StringVar(value=str(Path("./subtitles").resolve()))
        self.asr_engine = tk.StringVar(value="faster-whisper")
        self.whisper_model = tk.StringVar(value="medium")
        _dg_key, _ = resolve_deepgram_settings(
            api_key=os.getenv("DEEPGRAM_API_KEY", ""),
            model_name="",
            config_path=DEFAULT_ASR_CONFIG_PATH,
        )
        self.deepgram_api_key = tk.StringVar(value=_dg_key)
        self.max_duration = tk.StringVar(value="2.2")
        self.translate_enabled = tk.BooleanVar(value=True)
        self.translate_model = tk.StringVar(value=BACKEND_DEFAULTS["local"]["model"])
        self.translate_backend = tk.StringVar(value="local")
        self.ollama_status_text = tk.StringVar(value="检测中...")
        self.translate_max_tokens = tk.StringVar(value="2000")
        self.max_tokens_hint = tk.StringVar(value=TOKEN_RECOMMENDATIONS["local"])
        self.current_file_text = tk.StringVar(value="当前文件: -")
        self.asr_progress_var = tk.DoubleVar(value=0.0)
        self.asr_progress_text = tk.StringVar(value="ASR: 0.0%")
        self.translate_progress_var = tk.DoubleVar(value=0.0)
        self.translate_progress_text = tk.StringVar(value="翻译: 0.0%")
        self.translate_backend.trace_add("write", self._on_translate_backend_change)
        self.asr_engine.trace_add("write", self._on_asr_engine_change)

        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=BOTH, expand=True)

        files_frame = ttk.LabelFrame(top, text="视频文件", padding=10)
        files_frame.pack(fill=BOTH, expand=True)

        drop_text = "拖拽视频到此区域，或使用下方按钮添加"
        if not self.dnd_enabled:
            if self.dnd_reason:
                drop_text += f"（拖拽不可用：{self.dnd_reason}）"
            else:
                drop_text += "（当前环境未启用拖拽扩展，仅可按钮添加）"
        ttk.Label(files_frame, text=drop_text).pack(anchor=W)

        list_container = ttk.Frame(files_frame)
        list_container.pack(fill=BOTH, expand=True, pady=8)

        self.file_listbox = tk.Listbox(list_container, height=8, selectmode=tk.EXTENDED)
        self.file_listbox.pack(side=LEFT, fill=BOTH, expand=True)

        scrollbar = ttk.Scrollbar(list_container, orient=VERTICAL, command=self.file_listbox.yview)
        scrollbar.pack(side=RIGHT, fill=Y)
        self.file_listbox.config(yscrollcommand=scrollbar.set)

        if self.dnd_enabled:
            self._bind_drop_targets([files_frame, list_container, self.file_listbox])

        btn_row = ttk.Frame(files_frame)
        btn_row.pack(fill=BOTH)
        ttk.Button(btn_row, text="添加文件", command=self._pick_files).pack(side=LEFT)
        ttk.Button(btn_row, text="添加文件夹", command=self._pick_folder).pack(side=LEFT, padx=8)
        ttk.Button(btn_row, text="移除选中", command=self._remove_selected).pack(side=LEFT)
        ttk.Button(btn_row, text="清空", command=self._clear_files).pack(side=LEFT, padx=8)

        options = ttk.LabelFrame(top, text="参数", padding=10)
        options.pack(fill=BOTH, pady=10)

        ttk.Label(options, text="输出目录").grid(row=0, column=0, sticky=W, padx=(0, 8), pady=4)
        self.output_dir_entry = tk.Entry(
            options,
            state="disabled",
            relief="solid",
            borderwidth=1,
            bg="#000000",
            fg="#ff3b30",
            disabledforeground="#ff3b30",
            insertbackground="#ff3b30",
        )
        self.output_dir_entry.grid(row=0, column=1, sticky="ew", pady=4)
        self._render_output_dir(self.output_dir.get())
        ttk.Button(options, text="浏览", command=self._pick_output_dir).grid(
            row=0, column=2, padx=(8, 0), pady=4
        )

        ttk.Label(options, text="ASR 引擎").grid(row=1, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            options,
            textvariable=self.asr_engine,
            values=["whisper", "faster-whisper", "deepgram"],
            state="readonly",
        ).grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(options, text="ASR 模型").grid(row=2, column=0, sticky=W, padx=(0, 8), pady=4)
        self.whisper_model_combo = ttk.Combobox(
            options,
            textvariable=self.whisper_model,
            values=MODEL_PRESETS_DEFAULT,
            state="normal",
        )
        self.whisper_model_combo.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)

        ttk.Label(options, text="Deepgram API Key").grid(row=3, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(options, textvariable=self.deepgram_api_key, show="*").grid(
            row=3, column=1, columnspan=2, sticky="ew", pady=4
        )

        self._add_labeled_entry(options, "每条最大时长(秒)", self.max_duration, 4)

        trans_row = ttk.Frame(options)
        trans_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)
        ttk.Checkbutton(trans_row, text="翻译为简体中文", variable=self.translate_enabled).pack(side=LEFT)
        ttk.Label(trans_row, text="后端").pack(side=LEFT, padx=(14, 6))
        self.translate_backend_combo = ttk.Combobox(
            trans_row,
            textvariable=self.translate_backend,
            values=["local", "openai", "deepseek"],
            width=9,
            state="readonly",
        )
        self.translate_backend_combo.pack(side=LEFT)
        self.translate_backend_combo.bind("<<ComboboxSelected>>", self._on_translate_backend_change)
        ttk.Label(trans_row, text="本地模型").pack(side=LEFT, padx=(12, 6))
        self.local_model_combo = ttk.Combobox(
            trans_row,
            textvariable=self.translate_model,
            values=LOCAL_TRANSLATION_MODELS,
            width=28,
            state="readonly",
        )
        self.local_model_combo.pack(side=LEFT)
        self.local_model_combo.bind("<<ComboboxSelected>>", self._on_local_model_change)
        ttk.Label(trans_row, text="最大Token数").pack(side=LEFT, padx=(12, 6))
        ttk.Entry(trans_row, textvariable=self.translate_max_tokens, width=10).pack(side=LEFT)
        ttk.Label(
            trans_row,
            textvariable=self.max_tokens_hint,
            foreground="#7aa2f7",
        ).pack(side=LEFT, padx=(10, 0))
        ttk.Label(options, text=f"在线后端配置文件: {DEFAULT_TRANSLATE_CONFIG_PATH}").grid(
            row=6, column=0, columnspan=3, sticky=W, pady=(2, 4)
        )
        ttk.Label(options, text="Ollama 状态").grid(row=7, column=0, sticky=W, padx=(0, 8), pady=4)
        status_row = ttk.Frame(options)
        status_row.grid(row=7, column=1, columnspan=2, sticky="ew", pady=4)
        self.ollama_status_label = tk.Label(
            status_row,
            textvariable=self.ollama_status_text,
            fg="#d64545",
            anchor="w",
        )
        self.ollama_status_label.pack(side=LEFT, fill=BOTH, expand=True)
        ttk.Button(status_row, text="刷新检测", command=self._refresh_ollama_status_async).pack(side=RIGHT)

        options.columnconfigure(1, weight=1)

        run_row = ttk.Frame(top)
        run_row.pack(fill=BOTH)
        self.run_btn = ttk.Button(run_row, text="开始处理", command=self._start)
        self.run_btn.pack(side=LEFT)

        progress_frame = ttk.LabelFrame(top, text="任务进度", padding=10)
        progress_frame.pack(fill=BOTH, pady=(10, 0))
        ttk.Label(progress_frame, textvariable=self.current_file_text).grid(
            row=0, column=0, columnspan=2, sticky=W, pady=(0, 6)
        )
        ttk.Label(progress_frame, text="语音识别").grid(row=1, column=0, sticky=W, padx=(0, 8))
        self.asr_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.asr_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.asr_progress_bar.grid(row=1, column=1, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.asr_progress_text).grid(
            row=2, column=1, sticky="w", pady=(2, 6)
        )
        ttk.Label(progress_frame, text="字幕翻译").grid(row=3, column=0, sticky=W, padx=(0, 8))
        self.translate_progress_bar = ttk.Progressbar(
            progress_frame,
            variable=self.translate_progress_var,
            maximum=100.0,
            mode="determinate",
        )
        self.translate_progress_bar.grid(row=3, column=1, sticky="ew")
        ttk.Label(progress_frame, textvariable=self.translate_progress_text).grid(
            row=4, column=1, sticky="w", pady=(2, 0)
        )
        progress_frame.columnconfigure(1, weight=1)

        log_frame = ttk.LabelFrame(top, text="日志", padding=10)
        log_frame.pack(fill=BOTH, expand=True, pady=10)

        self.log_text = tk.Text(log_frame, height=12, state=tk.DISABLED)
        self.log_text.pack(fill=BOTH, expand=True)
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
            ttk.Button(parent, text="浏览", command=browse).grid(row=row, column=2, padx=(8, 0), pady=4)

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
        asr_engine = self.asr_engine.get().strip() or "faster-whisper"
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
                    else:
                        self._log(f"使用 DeepSeek 翻译模型: {translate_model}")
                    zh = translate_entries_contextual(
                        entries,
                        model_name=translate_model,
                        max_tokens=max_tokens,
                        base_url=translate_base_url,
                        api_key=translate_api_key,
                        progress_label=video.name,
                        progress_callback=on_progress,
                    )
                    self._log("正在进行译文二次校正（按原文时间戳对齐）...")
                    zh = redistribute_translated_by_source_timestamps(
                        entries,
                        zh,
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
        engine = (self.asr_engine.get().strip() or "faster-whisper").lower()
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
        elif task in {"translate", "align"}:
            self.translate_progress_var.set(percent)
            self.translate_progress_text.set(text)

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
