from __future__ import annotations

import json
import os
import queue
import sys
import webbrowser
import tkinter as tk
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, W, filedialog, messagebox, ttk
from typing import Callable

from webvidgrab.site_cli import ProbeResult, run_site_download
from webvidgrab.version import check_for_updates, get_version_info


class DarkOrangeColors:
    BACKGROUND = "#171717"
    CARD_BACKGROUND = "#242424"
    CARD_BACKGROUND_SOFT = "#2b2b2b"
    INPUT_BACKGROUND = "#202020"
    INPUT_BACKGROUND_SOFT = "#292929"
    PRIMARY = "#ff6b00"
    PRIMARY_HOVER = "#ffab40"
    PRIMARY_PRESSED = "#e65c00"
    TEXT_PRIMARY = "#f5f5f7"
    TEXT_SECONDARY = "#b3b3b8"
    TEXT_BODY = "#e8e8ed"
    TEXT_DISABLED = "#505050"
    TEXT_HIGHLIGHT = "#ff9f43"
    TEXT_PLACEHOLDER = "#707070"
    BORDER = "#3a3a3c"
    BORDER_LIGHT = "#4a4a4d"
    BORDER_FOCUS = "#ff8d3a"
    SEPARATOR = "#303033"
    SHADOW = "#101010"
    SUCCESS = "#4caf50"
    ERROR = "#f44336"
    LIST_SELECTION = "#ff6b00"


class DarkOrangeButton(tk.Canvas):
    def __init__(
        self,
        master,
        text: str,
        command: Callable | None = None,
        width: int = 120,
        height: int = 44,
        style: str = "primary",
        **kwargs,
    ):
        super().__init__(master, width=width, height=height, highlightthickness=0, **kwargs)
        self.text = text
        self.command = command
        self.style = style
        self.width = width
        self.height = height
        self.radius = 18
        self.state = "normal"
        self._draw_button()
        self._bind_events()

    def _mix_with_white(self, color: str, amount: float) -> str:
        color = color.lstrip("#")
        rgb = [int(color[i : i + 2], 16) for i in range(0, 6, 2)]
        mixed = [min(255, int(channel + (255 - channel) * amount)) for channel in rgb]
        return "#" + "".join(f"{value:02x}" for value in mixed)

    def _resolve_palette(self, hover: bool = False, pressed: bool = False) -> tuple[str, str, str]:
        if self.style == "primary":
            bg = DarkOrangeColors.PRIMARY_HOVER if hover else DarkOrangeColors.PRIMARY
            if pressed:
                bg = DarkOrangeColors.PRIMARY_PRESSED
            return bg, "#ffffff", self._mix_with_white(bg, 0.08)
        if self.style == "danger":
            bg = "#4a2320" if not hover else "#5a2a25"
            if pressed:
                bg = "#3a1b17"
            return bg, "#ffd7d2", "#7a3027"
        bg = DarkOrangeColors.CARD_BACKGROUND_SOFT if not hover else "#343437"
        if pressed:
            bg = "#262629"
        return bg, DarkOrangeColors.TEXT_HIGHLIGHT, DarkOrangeColors.BORDER_LIGHT

    def _rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw_button(self):
        self.delete("all")
        if self.state == "disabled":
            bg_color = "#404040"
            text_color = DarkOrangeColors.TEXT_DISABLED
            border_color = "#404040"
        elif self.state == "pressed":
            bg_color, text_color, border_color = self._resolve_palette(pressed=True)
        elif self.state == "hover":
            bg_color, text_color, border_color = self._resolve_palette(hover=True)
        else:
            bg_color, text_color, border_color = self._resolve_palette()

        self._rounded_rectangle(
            1, 4, self.width - 1, self.height,
            self.radius,
            fill=DarkOrangeColors.SHADOW,
            outline="",
        )
        self._rounded_rectangle(
            0, 0, self.width, self.height - 4,
            self.radius,
            fill=bg_color,
            outline=border_color,
            width=1,
        )
        self._rounded_rectangle(
            1, 1, self.width - 1, max(8, (self.height - 4) // 2),
            max(8, self.radius - 6),
            fill=self._mix_with_white(bg_color, 0.12),
            outline="",
        )
        self.create_text(
            self.width // 2,
            (self.height - 4) // 2,
            text=self.text,
            fill=text_color,
            font=("SF Pro Text", 14, "bold"),
        )

    def _bind_events(self):
        if self.state != "disabled":
            self.bind("<Enter>", self._on_enter)
            self.bind("<Leave>", self._on_leave)
            self.bind("<Button-1>", self._on_press)
            self.bind("<ButtonRelease-1>", self._on_release)

    def _on_enter(self, _event):
        self.state = "hover"
        self._draw_button()

    def _on_leave(self, _event):
        self.state = "normal"
        self._draw_button()

    def _on_press(self, _event):
        self.state = "pressed"
        self._draw_button()

    def _on_release(self, _event):
        self.state = "hover"
        self._draw_button()
        if self.command:
            self.command()

    def set_enabled(self, enabled: bool):
        self.state = "normal" if enabled else "disabled"
        self._draw_button()
        if not enabled:
            self.unbind("<Enter>")
            self.unbind("<Leave>")
            self.unbind("<Button-1>")
            self.unbind("<ButtonRelease-1>")
        else:
            self._bind_events()


class DarkOrangeCard(tk.Frame):
    def __init__(self, master, text: str = "", padding: int = 20, radius: int = 28, **kwargs):
        outer_bg = kwargs.pop("bg", DarkOrangeColors.BACKGROUND)
        super().__init__(master, bg=outer_bg, highlightthickness=0, borderwidth=0, **kwargs)
        self._radius = radius
        self._canvas = tk.Canvas(
            self,
            bg=outer_bg,
            highlightthickness=0,
            borderwidth=0,
            relief=tk.FLAT,
        )
        self._canvas.place(relx=0, rely=0, relwidth=1, relheight=1)
        self._canvas.bind("<Configure>", self._redraw)
        self.content = tk.Frame(self, bg=DarkOrangeColors.CARD_BACKGROUND)
        self.content.pack(fill=BOTH, expand=True, padx=padding, pady=padding)
        if text:
            self.title_label = tk.Label(
                self.content,
                text=text,
                font=("SF Pro Display", 16, "bold"),
                fg=DarkOrangeColors.TEXT_PRIMARY,
                bg=DarkOrangeColors.CARD_BACKGROUND,
                anchor="w",
            )
            self.title_label.pack(fill=tk.X, pady=(0, 14))
        else:
            self.title_label = None

    def _rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return self._canvas.create_polygon(points, smooth=True, **kwargs)

    def _redraw(self, _event=None):
        width = max(2, self._canvas.winfo_width())
        height = max(2, self._canvas.winfo_height())
        self._canvas.delete("card")
        self._rounded_rectangle(
            10, 12, width - 10, height - 2, self._radius,
            fill=DarkOrangeColors.SHADOW,
            outline="",
            tags="card",
        )
        self._rounded_rectangle(
            0, 0, width - 2, height - 12, self._radius,
            fill=DarkOrangeColors.CARD_BACKGROUND,
            outline=DarkOrangeColors.BORDER,
            width=1,
            tags="card",
        )
        self._rounded_rectangle(
            1, 1, width - 3, max(42, height // 3), self._radius - 6,
            fill=DarkOrangeColors.CARD_BACKGROUND_SOFT,
            outline="",
            tags="card",
        )
        self._canvas.lower("card")


class _DarkOrangeRoundedBox(tk.Frame):
    def __init__(
        self,
        master,
        *,
        height: int | None = None,
        radius: int = 16,
        outer_bg: str = DarkOrangeColors.CARD_BACKGROUND,
        fill_color: str = DarkOrangeColors.INPUT_BACKGROUND,
        padding_x: int = 14,
        padding_y: int = 11,
    ):
        super().__init__(master, bg=outer_bg)
        self._radius = radius
        self._fill_color = fill_color
        self._padding_x = padding_x
        self._padding_y = padding_y
        self._focused = False
        self._canvas = tk.Canvas(
            self,
            bg=outer_bg,
            highlightthickness=0,
            borderwidth=0,
            relief=tk.FLAT,
        )
        self._canvas.pack(fill=BOTH, expand=True)
        self._window_id: int | None = None
        self._canvas.bind("<Configure>", self._redraw)
        if height is not None:
            tk.Frame.configure(self, height=height)
            self.pack_propagate(False)

    def _rounded_rectangle(self, x1, y1, x2, y2, radius, **kwargs):
        points = [
            x1 + radius,
            y1,
            x2 - radius,
            y1,
            x2,
            y1,
            x2,
            y1 + radius,
            x2,
            y2 - radius,
            x2,
            y2,
            x2 - radius,
            y2,
            x1 + radius,
            y2,
            x1,
            y2,
            x1,
            y2 - radius,
            x1,
            y1 + radius,
            x1,
            y1,
        ]
        return self._canvas.create_polygon(points, smooth=True, **kwargs)

    def _attach_widget(self, widget: tk.Widget) -> None:
        self._window_id = self._canvas.create_window(
            self._padding_x,
            self._padding_y,
            anchor="nw",
            window=widget,
        )
        widget.bind("<FocusIn>", self._on_focus_in, add="+")
        widget.bind("<FocusOut>", self._on_focus_out, add="+")

    def _on_focus_in(self, _event) -> None:
        self._focused = True
        self._redraw()

    def _on_focus_out(self, _event) -> None:
        self._focused = False
        self._redraw()

    def _redraw(self, _event=None) -> None:
        width = max(2, self._canvas.winfo_width())
        height = max(2, self._canvas.winfo_height())
        self._canvas.delete("rounded_box")
        border_color = DarkOrangeColors.BORDER_FOCUS if self._focused else DarkOrangeColors.BORDER
        self._rounded_rectangle(
            1, 1, width - 1, height - 1, self._radius,
            fill=self._fill_color,
            outline=border_color,
            width=1,
            tags="rounded_box",
        )
        if self._window_id is not None:
            inner_width = max(1, width - self._padding_x * 2)
            inner_height = max(1, height - self._padding_y * 2)
            self._canvas.coords(self._window_id, self._padding_x, self._padding_y)
            self._canvas.itemconfigure(self._window_id, width=inner_width, height=inner_height)


class DarkOrangeEntry(_DarkOrangeRoundedBox):
    def __init__(self, master, **kwargs):
        super().__init__(master, height=52, radius=16, padding_x=14, padding_y=11)
        kwargs.setdefault("bg", DarkOrangeColors.INPUT_BACKGROUND)
        kwargs.setdefault("fg", DarkOrangeColors.TEXT_BODY)
        kwargs.setdefault("insertbackground", DarkOrangeColors.PRIMARY)
        kwargs.setdefault("font", ("SF Pro Text", 15))
        self.entry = tk.Entry(self._canvas, relief=tk.FLAT, borderwidth=0, highlightthickness=0, **kwargs)
        self._attach_widget(self.entry)

    def get(self, *args, **kwargs):
        return self.entry.get(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.entry.delete(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.entry.insert(*args, **kwargs)

    def bind(self, sequence=None, func=None, add=None):
        return self.entry.bind(sequence, func, add)


class DarkOrangeCombobox(_DarkOrangeRoundedBox):
    _style_name = "DarkOrange.TCombobox"
    _style_ready = False

    def __init__(self, master, **kwargs):
        super().__init__(master, height=52, radius=16, padding_x=10, padding_y=9)
        style = ttk.Style()
        if not DarkOrangeCombobox._style_ready:
            style.configure(
                self._style_name,
                foreground=DarkOrangeColors.TEXT_BODY,
                fieldbackground=DarkOrangeColors.INPUT_BACKGROUND,
                background=DarkOrangeColors.INPUT_BACKGROUND,
                arrowcolor=DarkOrangeColors.TEXT_BODY,
                borderwidth=0,
                padding=6,
                font=("SF Pro Text", 14),
            )
            style.map(
                self._style_name,
                fieldbackground=[("readonly", DarkOrangeColors.INPUT_BACKGROUND)],
                foreground=[("readonly", DarkOrangeColors.TEXT_BODY)],
            )
            DarkOrangeCombobox._style_ready = True
        self.combo = ttk.Combobox(self._canvas, style=self._style_name, **kwargs)
        self._attach_widget(self.combo)


class DarkOrangeText(_DarkOrangeRoundedBox):
    def __init__(self, master, **kwargs):
        text_height = kwargs.pop("height", None)
        fill_color = kwargs.get("bg", DarkOrangeColors.INPUT_BACKGROUND)
        fixed_height = max(96, int(text_height) * 26 + 20) if text_height is not None else None
        super().__init__(
            master,
            height=fixed_height,
            radius=18,
            padding_x=14,
            padding_y=12,
            fill_color=fill_color,
        )
        kwargs.setdefault("bg", DarkOrangeColors.INPUT_BACKGROUND)
        kwargs.setdefault("fg", DarkOrangeColors.TEXT_BODY)
        kwargs.setdefault("insertbackground", DarkOrangeColors.PRIMARY)
        kwargs.setdefault("font", ("SF Pro Text", 15))
        self.text = tk.Text(
            self._canvas,
            relief=tk.FLAT,
            borderwidth=0,
            highlightthickness=0,
            padx=0,
            pady=0,
            **kwargs,
        )
        self._attach_widget(self.text)
        self._placeholder_active = False

    def set_placeholder(self, placeholder: str):
        self.delete("1.0", END)
        self.insert("1.0", placeholder)
        self.text.configure(fg=DarkOrangeColors.TEXT_PLACEHOLDER)
        self._placeholder_active = True

    def delete_placeholder(self):
        if not self._placeholder_active:
            return
        content = self.get("1.0", END).strip()
        if content:
            self.delete("1.0", END)
            self.text.configure(fg=DarkOrangeColors.TEXT_BODY)
            self._placeholder_active = False

    def get(self, *args, **kwargs):
        return self.text.get(*args, **kwargs)

    def delete(self, *args, **kwargs):
        return self.text.delete(*args, **kwargs)

    def insert(self, *args, **kwargs):
        return self.text.insert(*args, **kwargs)

    def see(self, *args, **kwargs):
        return self.text.see(*args, **kwargs)

    def tag_configure(self, *args, **kwargs):
        return self.text.tag_configure(*args, **kwargs)

    def bind(self, sequence=None, func=None, add=None):
        return self.text.bind(sequence, func, add)

    def config(self, *args, **kwargs):
        return self.text.config(*args, **kwargs)


class DarkOrangeNotebook(ttk.Notebook):
    def __init__(self, master, **kwargs):
        super().__init__(master, style="DarkOrange.TNotebook", **kwargs)
        style = ttk.Style()
        style.configure(
            "DarkOrange.TNotebook",
            background=DarkOrangeColors.CARD_BACKGROUND,
            bordercolor=DarkOrangeColors.BORDER,
            tabmargins=[0, 0, 0, 0],
        )
        style.configure(
            "DarkOrange.TNotebook.Tab",
            background=DarkOrangeColors.INPUT_BACKGROUND,
            foreground=DarkOrangeColors.TEXT_SECONDARY,
            padding=[18, 10],
            font=("SF Pro Text", 13, "bold"),
            borderwidth=0,
        )
        style.map(
            "DarkOrange.TNotebook.Tab",
            background=[("selected", DarkOrangeColors.CARD_BACKGROUND_SOFT)],
            foreground=[("selected", DarkOrangeColors.TEXT_PRIMARY)],
        )


@dataclass
class DownloadTask:
    task_id: int
    url: str
    status: str = "pending"
    downloaded_fragments: int = 0
    total_fragments: int | None = None
    output_file: Path | None = None
    log_file: Path | None = None


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.version_info = get_version_info()
        self.root.title("PSiteDL - 视频下载工具")
        self.root.geometry("1200x850")
        self.root.minsize(1000, 700)
        self.root.configure(bg=DarkOrangeColors.BACKGROUND)

        self.output_dir = tk.StringVar(value=str((Path.home() / "Downloads").resolve()))
        self.browser = tk.StringVar(value="chrome")
        self.profile = tk.StringVar(value="Default")
        self.capture_seconds = tk.StringVar(value="30")
        self.use_runtime_capture = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="就绪")

        self.running = False
        self.log_panel_ratio = 0.40
        self._resize_job: str | None = None
        self.next_task_id = 1
        self.tasks: dict[int, DownloadTask] = {}
        self.pending_ids: list[int] = []
        self.active_futures: dict[Future, int] = {}
        self.completed_ids: list[int] = []
        self.executor: ThreadPoolExecutor | None = None
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.pending_state_path = self._project_root() / "logs" / "site_gui_pending_tasks.json"

        self._load_icon()
        self._build_ui()
        self._load_pending_state()
        self._create_menu()
        self._create_statusbar()
        self._bind_exit_handlers()
        self._poll_logs()

    def _project_root(self) -> Path:
        return Path(__file__).parent.parent.parent

    def _find_assets_dir(self) -> Path | None:
        project_root = self._project_root()
        candidates = [project_root / "assets", project_root / "PSiteDL" / "assets"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_icon(self) -> None:
        try:
            assets_dir = self._find_assets_dir()
            if assets_dir is None:
                return
            if os.name == "nt":
                icon_path = assets_dir / "psitedl_icon.ico"
                if icon_path.exists():
                    self.root.iconbitmap(str(icon_path))
                    return
            for icon_path in (assets_dir / "psitedl_icon.png", assets_dir / "icon-64.png", assets_dir / "icon-64.gif"):
                if not icon_path.exists():
                    continue
                try:
                    icon_img = tk.PhotoImage(file=str(icon_path))
                    self.root.iconphoto(True, icon_img)
                    self._icon_image = icon_img
                    return
                except Exception:
                    continue
        except Exception:
            pass

    def _bind_exit_handlers(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self._confirm_exit)
        for sequence in ("<Command-q>", "<Command-w>", "<Control-q>", "<Alt-F4>"):
            try:
                self.root.bind_all(sequence, self._on_exit_shortcut, add="+")
            except Exception:
                continue

    def _on_exit_shortcut(self, _event=None) -> str:
        self._confirm_exit()
        return "break"

    def _confirm_exit(self) -> None:
        active_count = len(self.active_futures)
        pending_count = len(self.pending_ids)
        completed_count = len(self.completed_ids)
        if self.running or active_count or pending_count:
            message = (
                "当前仍有下载任务。\n\n"
                f"正在下载：{active_count}\n"
                f"等待下载：{pending_count}\n"
                f"已完成：{completed_count}\n\n"
                "现在退出可能中断本次任务，确定要退出吗？"
            )
            title = "确认退出"
        else:
            message = "确定要退出 PSiteDL 吗？"
            title = "退出程序"
        if not messagebox.askokcancel(title, message, parent=self.root):
            return
        self.running = False
        self._save_pending_state()
        if self.executor is not None:
            self.executor.shutdown(wait=False, cancel_futures=True)
            self.executor = None
        self.root.quit()
        self.root.destroy()

    def _unfinished_task_ids(self) -> list[int]:
        active_ids = [tid for tid in self.active_futures.values() if tid in self.tasks]
        known_ids = self.pending_ids + active_ids
        seen: set[int] = set()
        unfinished_ids: list[int] = []
        for tid in known_ids:
            if tid in seen:
                continue
            task = self.tasks.get(tid)
            if task is None or task.status in {"done", "removed"}:
                continue
            seen.add(tid)
            unfinished_ids.append(tid)
        for tid, task in self.tasks.items():
            if tid in seen or task.status in {"done", "removed"}:
                continue
            unfinished_ids.append(tid)
        return unfinished_ids

    def _save_pending_state(self) -> None:
        unfinished_ids = self._unfinished_task_ids()
        if not unfinished_ids:
            try:
                self.pending_state_path.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                pass
            return
        try:
            self.pending_state_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "next_task_id": self.next_task_id,
                "tasks": [
                    {
                        "task_id": tid,
                        "url": self.tasks[tid].url,
                        "status": self.tasks[tid].status,
                    }
                    for tid in unfinished_ids
                ],
            }
            self.pending_state_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            self._append_log(f"[queue] 保存未完成任务失败：{exc}", "warning")

    def _load_pending_state(self) -> None:
        if not self.pending_state_path.exists():
            return
        try:
            payload = json.loads(self.pending_state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._append_log(f"[queue] 读取未完成任务失败：{exc}", "warning")
            return

        restored = 0
        max_task_id = self.next_task_id - 1
        for item in payload.get("tasks", []):
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            task_id = int(item.get("task_id") or self.next_task_id)
            if task_id in self.tasks:
                continue
            self.tasks[task_id] = DownloadTask(task_id=task_id, url=url, status="pending")
            self.pending_ids.append(task_id)
            restored += 1
            max_task_id = max(max_task_id, task_id)

        self.next_task_id = max(int(payload.get("next_task_id") or 0), max_task_id + 1, self.next_task_id)
        if restored:
            self._refresh_pending_list()
            self.status_text.set(f"已恢复 {restored} 个未完成任务")
            self._append_log(f"[queue] 已从本地恢复 {restored} 个未完成任务。", "success")
            self._save_pending_state()

    def _schedule_split_ratio(self, _event=None) -> None:
        if not hasattr(self, "main_paned"):
            return
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(60, self._apply_split_ratio)

    def _apply_split_ratio(self) -> None:
        self._resize_job = None
        try:
            total_width = self.main_paned.winfo_width()
            if total_width <= 1:
                return
            sash_x = int(total_width * (1.0 - self.log_panel_ratio))
            sash_x = max(620, min(total_width - 360, sash_x))
            self.main_paned.sash_pos(0, sash_x)
        except Exception:
            pass

    def _build_ui(self) -> None:
        main_frame = tk.Frame(self.root, bg=DarkOrangeColors.BACKGROUND)
        main_frame.pack(fill=BOTH, expand=True, padx=18, pady=(14, 10))
        self._build_header(main_frame)

        paned = tk.PanedWindow(
            main_frame,
            orient=tk.HORIZONTAL,
            bg=DarkOrangeColors.BACKGROUND,
            relief=tk.FLAT,
            borderwidth=0,
            sashwidth=10,
            sashrelief=tk.FLAT,
            showhandle=False,
        )
        paned.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.main_paned = paned
        left_frame = tk.Frame(paned, bg=DarkOrangeColors.BACKGROUND)
        right_frame = tk.Frame(paned, bg=DarkOrangeColors.BACKGROUND)
        paned.add(left_frame, minsize=620, width=740)
        paned.add(right_frame, minsize=360, width=500)
        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)
        self.root.bind("<Configure>", self._schedule_split_ratio, add="+")
        self.root.after_idle(self._apply_split_ratio)

    def _add_header_icon(self, parent) -> None:
        assets_dir = self._find_assets_dir()
        if assets_dir is None:
            return
        for icon_path in (assets_dir / "icon-64.png", assets_dir / "icon-64.gif"):
            if not icon_path.exists():
                continue
            try:
                self.header_icon = tk.PhotoImage(file=str(icon_path))
                tk.Label(parent, image=self.header_icon, bg=DarkOrangeColors.CARD_BACKGROUND).pack(side=tk.LEFT, padx=(0, 14))
                return
            except Exception:
                continue

    def _build_header(self, parent):
        hero_card = DarkOrangeCard(parent, padding=18, radius=28)
        hero_card.pack(fill=tk.X, pady=(0, 12))
        hero = hero_card.content
        title_row = tk.Frame(hero, bg=DarkOrangeColors.CARD_BACKGROUND)
        title_row.pack(fill=tk.X)
        self._add_header_icon(title_row)

        tk.Label(
            title_row,
            text="PSiteDL",
            font=("SF Pro Display", 28, "bold"),
            fg=DarkOrangeColors.TEXT_PRIMARY,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(side=tk.LEFT)

        tk.Label(
            title_row,
            text=f"v{self.version_info['version']}",
            font=("SF Pro Text", 10, "bold"),
            fg=DarkOrangeColors.TEXT_HIGHLIGHT,
            bg=DarkOrangeColors.INPUT_BACKGROUND_SOFT,
            padx=10,
            pady=5,
        ).pack(side=tk.LEFT, padx=(12, 0), pady=(4, 0))

        tk.Label(
            hero,
            text="网页视频切片探测与下载",
            font=("SF Pro Text", 12),
            fg=DarkOrangeColors.TEXT_SECONDARY,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
            justify=tk.LEFT,
        ).pack(anchor="w", pady=(8, 0))

    def _build_left_panel(self, parent):
        self._build_url_card(parent)
        self._build_settings_card(parent)
        self._build_controls_card(parent)
        self._build_tasks_card(parent)

    def _build_right_panel(self, parent):
        self._build_log_card(parent)

    def _build_url_card(self, parent):
        card = DarkOrangeCard(parent, padding=16)
        card.pack(fill=tk.X, pady=(0, 12))
        content = card.content
        tk.Label(
            content,
            text="待下载 URL",
            font=("SF Pro Display", 14, "bold"),
            fg=DarkOrangeColors.TEXT_HIGHLIGHT,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            content,
            text="每行一个播放页 URL，支持批量加入待下载队列。",
            font=("SF Pro Text", 11),
            fg=DarkOrangeColors.TEXT_SECONDARY,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 0))
        self.url_text = DarkOrangeText(content, height=3)
        self.url_text.pack(fill=tk.X, pady=(10, 0))
        self.url_text.set_placeholder("请输入视频播放页面 URL（每行一个）")
        self.url_text.bind("<FocusIn>", lambda e: self.url_text.delete_placeholder())

    def _create_setting_field(
        self,
        parent,
        label: str,
        values: list[str] | None,
        variable: tk.Variable,
        has_button: bool = False,
        button_text: str = "",
        button_command: Callable | None = None,
    ):
        frame = tk.Frame(parent, bg=DarkOrangeColors.CARD_BACKGROUND)
        tk.Label(
            frame,
            text=label,
            font=("SF Pro Text", 13, "bold"),
            fg=DarkOrangeColors.TEXT_HIGHLIGHT,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 6))
        if values:
            combo = DarkOrangeCombobox(frame, textvariable=variable, values=values, state="readonly")
            combo.pack(fill=tk.X)
        elif has_button:
            entry_frame = tk.Frame(frame, bg=DarkOrangeColors.CARD_BACKGROUND)
            entry_frame.pack(fill=tk.X)
            entry = DarkOrangeEntry(entry_frame, textvariable=variable)
            entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 8))
            DarkOrangeButton(
                entry_frame,
                text=button_text,
                command=button_command,
                width=82,
                height=46,
                style="secondary",
                bg=DarkOrangeColors.CARD_BACKGROUND,
            ).pack(side=tk.RIGHT)
        else:
            entry = DarkOrangeEntry(frame, textvariable=variable)
            entry.pack(fill=tk.X)
        return frame

    def _build_settings_card(self, parent):
        card = DarkOrangeCard(parent, text="设置", padding=16)
        card.pack(fill=tk.X, pady=(0, 12))
        content = card.content
        settings_grid = tk.Frame(content, bg=DarkOrangeColors.CARD_BACKGROUND)
        settings_grid.pack(fill=tk.X)
        settings_grid.grid_columnconfigure(0, weight=1, uniform="settings_col")
        settings_grid.grid_columnconfigure(1, weight=1, uniform="settings_col")

        self._create_setting_field(
            settings_grid,
            "浏览器",
            ["chrome", "chromium", "edge", "brave"],
            self.browser,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 12), pady=(0, 10))
        self._create_setting_field(
            settings_grid,
            "配置文件",
            None,
            self.profile,
        ).grid(row=0, column=1, sticky="ew", padx=(12, 0), pady=(0, 10))
        self._create_setting_field(
            settings_grid,
            "输出目录",
            None,
            self.output_dir,
            has_button=True,
            button_text="浏览",
            button_command=self._pick_output_dir,
        ).grid(row=1, column=0, sticky="ew", padx=(0, 12))
        self._create_setting_field(
            settings_grid,
            "运行时探测秒数",
            None,
            self.capture_seconds,
        ).grid(row=1, column=1, sticky="ew", padx=(12, 0))

        capture_check = tk.Checkbutton(
            content,
            text="启用运行时探测（后台静默抓取播放请求）",
            variable=self.use_runtime_capture,
            font=("SF Pro Text", 13),
            fg=DarkOrangeColors.TEXT_BODY,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            selectcolor=DarkOrangeColors.CARD_BACKGROUND,
            activebackground=DarkOrangeColors.CARD_BACKGROUND,
            activeforeground=DarkOrangeColors.TEXT_BODY,
            padx=0,
            pady=0,
        )
        capture_check.pack(anchor="w", pady=(10, 0))

    def _build_controls_card(self, parent):
        card = DarkOrangeCard(parent, padding=16)
        card.pack(fill=tk.X, pady=(0, 12))
        controls_frame = tk.Frame(card.content, bg=DarkOrangeColors.CARD_BACKGROUND)
        controls_frame.pack(fill=tk.X)
        self.add_btn = DarkOrangeButton(controls_frame, text="加入待下载", command=self._add_tasks, width=132, height=46, style="secondary", bg=DarkOrangeColors.CARD_BACKGROUND)
        self.add_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.start_btn = DarkOrangeButton(controls_frame, text="启动下载", command=self._start_queue, width=132, height=46, style="primary", bg=DarkOrangeColors.CARD_BACKGROUND)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 12))
        self.clear_pending_btn = DarkOrangeButton(controls_frame, text="清空待下载", command=self._clear_pending, width=132, height=46, style="danger", bg=DarkOrangeColors.CARD_BACKGROUND)
        self.clear_pending_btn.pack(side=tk.LEFT, padx=(0, 12))
        DarkOrangeButton(controls_frame, text="清空日志", command=self._clear_log, width=108, height=46, style="secondary", bg=DarkOrangeColors.CARD_BACKGROUND).pack(side=tk.LEFT, padx=(0, 18))
        tk.Label(
            controls_frame,
            textvariable=self.status_text,
            font=("SF Pro Text", 14),
            fg=DarkOrangeColors.TEXT_HIGHLIGHT,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _build_tasks_card(self, parent):
        card = DarkOrangeCard(parent, text="任务管理", padding=16)
        card.pack(fill=BOTH, expand=True, pady=(0, 12))
        task_tabs = DarkOrangeNotebook(card.content, padding=0)
        task_tabs.pack(fill=BOTH, expand=True, pady=(4, 0))

        pending_tab = tk.Frame(task_tabs, bg=DarkOrangeColors.CARD_BACKGROUND)
        active_tab = tk.Frame(task_tabs, bg=DarkOrangeColors.CARD_BACKGROUND)
        completed_tab = tk.Frame(task_tabs, bg=DarkOrangeColors.CARD_BACKGROUND)
        task_tabs.add(pending_tab, text="  待下载  ")
        task_tabs.add(active_tab, text="  下载中  ")
        task_tabs.add(completed_tab, text="  已完成  ")

        self.pending_list = tk.Listbox(
            pending_tab,
            font=("SF Pro Text", 14),
            fg=DarkOrangeColors.TEXT_BODY,
            bg=DarkOrangeColors.INPUT_BACKGROUND,
            selectbackground=DarkOrangeColors.LIST_SELECTION,
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            activestyle="none",
        )
        self.pending_list.pack(fill=BOTH, expand=True, padx=8, pady=8)

        style = ttk.Style()
        style.configure(
            "DarkOrange.Treeview",
            background=DarkOrangeColors.INPUT_BACKGROUND,
            foreground=DarkOrangeColors.TEXT_BODY,
            fieldbackground=DarkOrangeColors.INPUT_BACKGROUND,
            font=("SF Pro Text", 13),
        )
        style.configure(
            "DarkOrange.Treeview.Heading",
            background=DarkOrangeColors.CARD_BACKGROUND,
            foreground=DarkOrangeColors.TEXT_PRIMARY,
            font=("SF Pro Text", 13, "bold"),
        )
        self.active_tree = ttk.Treeview(
            active_tab,
            columns=("url", "progress", "status"),
            show="headings",
            height=12,
            style="DarkOrange.Treeview",
        )
        self.active_tree.heading("url", text="URL")
        self.active_tree.heading("progress", text="下载进度")
        self.active_tree.heading("status", text="状态")
        self.active_tree.column("url", width=560, anchor=W)
        self.active_tree.column("progress", width=180, anchor=W)
        self.active_tree.column("status", width=140, anchor=W)
        self.active_tree.pack(fill=BOTH, expand=True, padx=8, pady=8)

        self.completed_list = tk.Listbox(
            completed_tab,
            font=("SF Pro Text", 14),
            fg=DarkOrangeColors.TEXT_BODY,
            bg=DarkOrangeColors.INPUT_BACKGROUND,
            selectbackground=DarkOrangeColors.SUCCESS,
            selectforeground="#ffffff",
            highlightthickness=0,
            borderwidth=0,
            activestyle="none",
        )
        self.completed_list.pack(fill=BOTH, expand=True, padx=8, pady=8)

    def _build_log_card(self, parent):
        card = DarkOrangeCard(parent, text="运行日志", padding=16)
        card.pack(fill=BOTH, expand=True)
        tk.Label(
            card.content,
            text="保留关键下载日志，适合与左侧任务面板同时查看。",
            font=("SF Pro Text", 11),
            fg=DarkOrangeColors.TEXT_SECONDARY,
            bg=DarkOrangeColors.CARD_BACKGROUND,
            anchor="w",
        ).pack(fill=tk.X, pady=(0, 10))
        self.log_text = DarkOrangeText(
            card.content,
            font=("SF Mono", 12),
            fg="#ff9500",
            bg="#1e1e1e",
            state=tk.DISABLED,
        )
        self.log_text.pack(fill=BOTH, expand=True)
        self.log_text.tag_configure("info", foreground="#ff9500")
        self.log_text.tag_configure("warning", foreground="#ff9800")
        self.log_text.tag_configure("error", foreground="#f44336")
        self.log_text.tag_configure("success", foreground="#4caf50")

    def _create_menu(self) -> None:
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于 PSiteDL", command=self._show_about)
        help_menu.add_command(label="检查更新", command=self._check_updates)
        help_menu.add_separator()
        help_menu.add_command(label="打开项目目录", command=self._open_project_dir)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="更新到最新版本", command=self._update_to_latest)

    def _create_statusbar(self) -> None:
        version_text = f"PSiteDL {self.version_info['display']} | 分支：{self.version_info['branch'] or 'N/A'}"
        self.statusbar = tk.Label(
            self.root,
            text=version_text,
            fg=DarkOrangeColors.TEXT_SECONDARY,
            bg=DarkOrangeColors.INPUT_BACKGROUND,
            anchor=W,
            padx=8,
            pady=6,
        )
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _show_about(self) -> None:
        about_text = f"""PSiteDL - 网页视频下载工具

版本：{self.version_info['display']}
分支：{self.version_info['branch'] or 'N/A'}
Commit: {self.version_info['commit'] or 'N/A'}

功能:
• 网页视频流探测与下载
• 支持多浏览器 Cookie 自动捕获
• 并发下载支持
• 实时进度显示
"""
        messagebox.showinfo("关于 PSiteDL", about_text)

    def _check_updates(self) -> None:
        self.status_text.set("正在检查更新...")
        self.root.update()
        result = check_for_updates()
        if "error" in result:
            messagebox.showwarning("检查更新", f"检查失败：{result['error']}")
            self.status_text.set("就绪")
            return
        if result["has_update"]:
            changes_text = "\n".join(f"• {c}" for c in result["changes"][:10])
            msg = (
                f"发现新版本！\n\n当前版本：{result['current_version']}\n最新版本：{result['latest_version']}\n\n"
                f"更新内容:\n{changes_text}\n\n是否现在更新？"
            )
            if messagebox.askyesno("发现新版本", msg):
                self._update_to_latest()
        else:
            messagebox.showinfo("检查更新", f"已是最新版本：{result['current_version']}")
        self.status_text.set("就绪")

    def _update_to_latest(self) -> None:
        try:
            import subprocess

            project_root = self._project_root()
            self.status_text.set("正在更新...")
            self.root.update()
            result = subprocess.run(["git", "pull"], cwd=project_root, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=project_root,
                    capture_output=True,
                    timeout=30,
                )
                msg = f"更新成功！\n\n{result.stdout[:500]}\n\n需要重启 GUI 以应用更新。是否现在重启？"
                if messagebox.askyesno("更新完成", msg):
                    os.execv(sys.executable, [sys.executable, "-m", "webvidgrab.site_gui"])
            else:
                messagebox.showerror("更新失败", f"错误：{result.stderr}")
                self.status_text.set("就绪")
        except Exception as e:
            messagebox.showerror("更新失败", str(e))
            self.status_text.set("就绪")

    def _open_project_dir(self) -> None:
        try:
            webbrowser.open(f"file://{self._project_root()}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录：{e}")

    def _clear_log(self) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete("1.0", END)
        self.log_text.config(state=tk.DISABLED)

    def _pick_output_dir(self) -> None:
        p = filedialog.askdirectory(title="选择输出目录")
        if p:
            self.output_dir.set(str(Path(p).resolve()))

    def _add_tasks(self) -> None:
        raw = self.url_text.get("1.0", END)
        urls = [x.strip() for x in raw.splitlines() if x.strip()]
        if not urls:
            messagebox.showwarning("提示", "请填写至少一个 URL。")
            return
        existing = {t.url for t in self.tasks.values()}
        added = 0
        for url in urls:
            if url in existing:
                continue
            tid = self.next_task_id
            self.next_task_id += 1
            task = DownloadTask(task_id=tid, url=url)
            self.tasks[tid] = task
            self.pending_ids.append(tid)
            existing.add(url)
            added += 1
        self._refresh_pending_list()
        self._save_pending_state()
        self.url_text.delete("1.0", END)
        self._append_log(f"[queue] 新增任务 {added} 个。")

    def _clear_pending(self) -> None:
        if self.running:
            messagebox.showwarning("提示", "下载进行中，暂不允许清空待下载。")
            return
        for tid in self.pending_ids:
            self.tasks[tid].status = "removed"
        self.pending_ids = []
        self._refresh_pending_list()
        self._save_pending_state()
        self._append_log("[queue] 已清空待下载任务。")

    def _start_queue(self) -> None:
        if self.running:
            return
        if not self.pending_ids:
            messagebox.showwarning("提示", "待下载任务为空。")
            return
        try:
            seconds = int(self.capture_seconds.get().strip())
            if seconds < 10:
                raise ValueError()
        except ValueError:
            messagebox.showerror("参数错误", "运行时探测秒数建议 >= 10")
            return
        self.running = True
        self.status_text.set("队列下载中...")
        self.start_btn.set_enabled(False)
        self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="sitegrab")
        self._dispatch_jobs()

    def _dispatch_jobs(self) -> None:
        if not self.running or self.executor is None:
            return
        browser = self.browser.get().strip() or "chrome"
        profile = self.profile.get().strip() or "Default"
        use_runtime_capture = bool(self.use_runtime_capture.get())
        capture_seconds = int(self.capture_seconds.get().strip())
        out_dir = Path(self.output_dir.get().strip()).expanduser().resolve()
        while len(self.active_futures) < 3 and self.pending_ids:
            tid = self.pending_ids.pop(0)
            task = self.tasks[tid]
            task.status = "running"
            self._upsert_active_row(task)
            future = self.executor.submit(
                self._run_one_task,
                tid,
                out_dir,
                capture_seconds,
                browser,
                profile,
                use_runtime_capture,
            )
            self.active_futures[future] = tid
            future.add_done_callback(lambda fut: self.log_queue.put(("__TASK_DONE__", fut)))
        self._refresh_pending_list()
        self._save_pending_state()
        self._update_status_line()

    def _run_one_task(
        self,
        tid: int,
        out_dir: Path,
        seconds: int,
        browser: str,
        profile: str,
        use_runtime_capture: bool,
    ) -> ProbeResult:
        task = self.tasks[tid]

        def log_func(msg: str) -> None:
            self.log_queue.put(("__TASK_LOG__", (tid, msg)))

        def progress(downloaded: int, total: int | None) -> None:
            self.log_queue.put(("__PROGRESS__", (tid, downloaded, total)))

        return run_site_download(
            page_url=task.url,
            output_dir=out_dir,
            browser=browser,
            profile=profile,
            capture_seconds=max(10, int(seconds)),
            use_runtime_capture=use_runtime_capture,
            headless=True,
            log_func=log_func,
            progress_callback=progress,
            use_rich_progress=False,
        )

    def _handle_task_done(self, future: Future) -> None:
        tid = self.active_futures.pop(future, None)
        if tid is None:
            return
        task = self.tasks[tid]
        try:
            result = future.result()
        except Exception as exc:
            task.status = "failed"
            self._append_log(f"[task-{tid}] [error] {exc}", "error")
            self._remove_active_row(tid)
            self._update_status_line()
            self._dispatch_jobs()
            self._save_pending_state()
            self._finish_if_idle()
            return
        task.log_file = result.log_file
        task.output_file = result.output_file
        if result.ok and result.output_file is not None:
            task.status = "done"
            self.completed_ids.append(tid)
            self.completed_list.insert(END, f"#{tid} {result.output_file.name}")
            self._append_log(f"[task-{tid}] [saved] {result.output_file}", "success")
            self._append_log(f"[task-{tid}] [log] {result.log_file}")
        else:
            task.status = "failed"
            self._append_log(f"[task-{tid}] [failed] {result.log_file}", "warning")
        self._remove_active_row(tid)
        self._update_status_line()
        self._dispatch_jobs()
        self._save_pending_state()
        self._finish_if_idle()

    def _finish_if_idle(self) -> None:
        if self.running and not self.pending_ids and not self.active_futures:
            self.running = False
            self.status_text.set("全部任务完成")
            self.start_btn.set_enabled(True)
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=False)
                self.executor = None
            self._save_pending_state()

    def _progress_percent(self, task: DownloadTask) -> float | None:
        if task.status == "done":
            return 100.0
        if task.total_fragments and task.total_fragments > 0:
            return min(100.0, (task.downloaded_fragments / task.total_fragments) * 100.0)
        return None

    def _progress_text(self, task: DownloadTask) -> str:
        percent = self._progress_percent(task)
        if percent is not None:
            total_slots = 10
            filled_slots = min(total_slots, max(0, round((percent / 100.0) * total_slots)))
            bar = "█" * filled_slots + "░" * (total_slots - filled_slots)
            return f"{bar} {percent:5.1f}%"
        if task.downloaded_fragments > 0:
            return "□□□□□□□ 计算中"
        return "░░░░░░░░░░   0.0%"

    def _short_url(self, url: str, max_len: int = 80) -> str:
        if len(url) <= max_len:
            return url
        return url[: max_len - 3] + "..."

    def _status_text(self, task: DownloadTask) -> str:
        if task.status == "pending":
            return "等待中"
        if task.status == "running":
            return "下载中"
        if task.status == "done":
            return "已完成"
        if task.status == "failed":
            return "失败"
        if task.status == "removed":
            return "已移除"
        return task.status

    def _upsert_active_row(self, task: DownloadTask) -> None:
        iid = str(task.task_id)
        values = (self._short_url(task.url), self._progress_text(task), self._status_text(task))
        if self.active_tree.exists(iid):
            self.active_tree.item(iid, values=values)
        else:
            self.active_tree.insert("", END, iid=iid, values=values)

    def _remove_active_row(self, tid: int) -> None:
        iid = str(tid)
        if self.active_tree.exists(iid):
            self.active_tree.delete(iid)

    def _refresh_pending_list(self) -> None:
        self.pending_list.delete(0, END)
        for tid in self.pending_ids:
            task = self.tasks[tid]
            self.pending_list.insert(END, f"#{tid} {self._short_url(task.url)}")

    def _update_status_line(self) -> None:
        if not self.running:
            return
        self.status_text.set(
            f"下载中：正在{len(self.active_futures)} | 待下载{len(self.pending_ids)} | 已完成{len(self.completed_ids)}"
        )

    def _append_log(self, text: str, level: str = "info") -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(END, text + "\n", level)
        self.log_text.see(END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                tag, value = self.log_queue.get_nowait()
                if tag == "__TASK_LOG__":
                    tid, msg = value
                    self._append_log(f"[task-{tid}] {msg}")
                elif tag == "__PROGRESS__":
                    tid, downloaded, total = value
                    task = self.tasks.get(int(tid))
                    if task is not None:
                        task.downloaded_fragments = int(downloaded)
                        task.total_fragments = int(total) if total is not None else None
                        self._upsert_active_row(task)
                elif tag == "__TASK_DONE__":
                    self._handle_task_done(value)
                elif tag == "__ERROR__":
                    self._append_log(f"[error] {value}", "error")
                    messagebox.showerror("执行失败", str(value))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_logs)


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
