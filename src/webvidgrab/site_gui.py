from __future__ import annotations

import queue
import webbrowser
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, W, filedialog, messagebox, ttk
import tkinter as tk

from webvidgrab.site_cli import ProbeResult, run_site_download
from webvidgrab.version import get_version_info, check_for_updates


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
        self.root.title(f"PSiteDL v{self.version_info['version']}")
        self.root.geometry("1180x820")

        self.output_dir = tk.StringVar(value=str((Path.home() / "Downloads").resolve()))
        self.browser = tk.StringVar(value="chrome")
        self.profile = tk.StringVar(value="Default")
        self.capture_seconds = tk.StringVar(value="30")
        self.use_runtime_capture = tk.BooleanVar(value=True)
        self.status_text = tk.StringVar(value="就绪")

        self.running = False
        self.next_task_id = 1
        self.tasks: dict[int, DownloadTask] = {}
        self.pending_ids: list[int] = []
        self.active_futures: dict[Future, int] = {}
        self.completed_ids: list[int] = []
        self.executor: ThreadPoolExecutor | None = None
        self.log_queue: queue.Queue[tuple[str, object]] = queue.Queue()

        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        # 创建菜单栏
        self._create_menu()
        
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(top, text="任务输入", padding=10)
        form.pack(fill=BOTH)

        ttk.Label(form, text="网页播放URL（每行一个）").grid(row=0, column=0, sticky=W, padx=(0, 8), pady=4)
        self.url_text = tk.Text(form, height=3)
        self.url_text.grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(form, text="输出目录").grid(row=1, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.output_dir).grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="浏览", command=self._pick_output_dir).grid(row=1, column=3, sticky="ew", pady=4)

        ttk.Label(form, text="浏览器").grid(row=2, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.browser,
            values=["chrome", "chromium", "edge", "brave"],
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Profile").grid(row=2, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.profile).grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Label(form, text="运行时探测秒数").grid(row=3, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.capture_seconds).grid(row=3, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(
            form,
            text="启用运行时探测(会打开浏览器并抓播放请求)",
            variable=self.use_runtime_capture,
        ).grid(row=3, column=2, columnspan=2, sticky=W, pady=4)

        note = (
            "支持并发3线程下载；显示切片进度(已下载/总切片)；完成后自动进入“已完成任务”。"
        )
        ttk.Label(form, text=note, foreground="#666").grid(
            row=4, column=0, columnspan=4, sticky=W, pady=(4, 2)
        )

        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        ctrl = ttk.Frame(top)
        ctrl.pack(fill=BOTH, pady=(10, 0))
        self.add_btn = ttk.Button(ctrl, text="加入待下载", command=self._add_tasks)
        self.add_btn.pack(side=LEFT)
        self.start_btn = ttk.Button(ctrl, text="启动队列下载(3并发)", command=self._start_queue)
        self.start_btn.pack(side=LEFT, padx=(8, 0))
        self.clear_pending_btn = ttk.Button(ctrl, text="清空待下载", command=self._clear_pending)
        self.clear_pending_btn.pack(side=LEFT, padx=(8, 0))
        ttk.Button(ctrl, text="清空日志", command=self._clear_log).pack(side=LEFT, padx=(8, 0))
        ttk.Label(ctrl, textvariable=self.status_text).pack(side=LEFT, padx=(12, 0))

        split = ttk.Panedwindow(top, orient=tk.VERTICAL)
        split.pack(fill=BOTH, expand=True, pady=(10, 0))

        list_frame = ttk.Frame(split)
        split.add(list_frame, weight=3)
        log_frame = ttk.LabelFrame(split, text="运行日志", padding=10)
        split.add(log_frame, weight=2)

        task_tabs = ttk.Notebook(list_frame)
        task_tabs.pack(fill=BOTH, expand=True)

        pending_tab = ttk.Frame(task_tabs, padding=8)
        active_tab = ttk.Frame(task_tabs, padding=8)
        completed_tab = ttk.Frame(task_tabs, padding=8)
        task_tabs.add(pending_tab, text="待下载任务")
        task_tabs.add(active_tab, text="正在下载任务")
        task_tabs.add(completed_tab, text="已完成任务")

        self.pending_list = tk.Listbox(pending_tab, height=12)
        self.pending_list.pack(fill=BOTH, expand=True)

        self.active_tree = ttk.Treeview(
            active_tab,
            columns=("url", "progress", "status"),
            show="headings",
            height=12,
        )
        self.active_tree.heading("url", text="URL")
        self.active_tree.heading("progress", text="切片进度")
        self.active_tree.heading("status", text="状态")
        self.active_tree.column("url", width=680, anchor=W)
        self.active_tree.column("progress", width=140, anchor=W)
        self.active_tree.column("status", width=160, anchor=W)
        self.active_tree.pack(fill=BOTH, expand=True)

        self.completed_list = tk.Listbox(completed_tab, height=12)
        self.completed_list.pack(fill=BOTH, expand=True)

        self.log_text = tk.Text(log_frame, state=tk.DISABLED, height=14)
        self.log_text.pack(fill=BOTH, expand=True)

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
            messagebox.showwarning("提示", "请填写至少一个URL。")
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
        self._append_log("[queue] 已清空待下载任务。")

    def _start_queue(self) -> None:
        if self.running:
            return
        if not self.pending_ids:
            messagebox.showwarning("提示", "待下载任务为空。")
            return
        out = Path(self.output_dir.get().strip()).expanduser()
        if not str(out):
            messagebox.showerror("参数错误", "请填写输出目录")
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
        self.start_btn.config(state=tk.DISABLED)
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
            # 调试：打印到终端，确保被调用
            print(f"[GUI-LOG] task-{tid}: {msg}")
            self.log_queue.put(("__TASK_LOG__", (tid, msg)))

        def progress(downloaded: int, total: int | None) -> None:
            # 调试：打印到终端，确保被调用
            print(f"[GUI-PROGRESS] task-{tid}: {downloaded}/{total}")
            # 同时发送进度更新和日志消息
            self.log_queue.put(("__PROGRESS__", (tid, downloaded, total)))
            # 每次进度变化都记录日志（便于调试）
            if total:
                percent = (downloaded / total * 100) if total > 0 else 0
                self.log_queue.put(("__TASK_LOG__", (tid, f"📊 进度：{downloaded}/{total} ({percent:.1f}%)")))
            else:
                self.log_queue.put(("__TASK_LOG__", (tid, f"📊 进度：{downloaded}/?")))

        return run_site_download(
            page_url=task.url,
            output_dir=out_dir,
            browser=browser,
            profile=profile,
            capture_seconds=max(10, int(seconds)),
            use_runtime_capture=use_runtime_capture,
            log_func=log_func,
            progress_callback=progress,
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
            task.log_file = None
            self._append_log(f"[task-{tid}] [error] {exc}")
            self._remove_active_row(tid)
            self._update_status_line()
            self._dispatch_jobs()
            self._finish_if_idle()
            return

        task.log_file = result.log_file
        task.output_file = result.output_file
        if result.ok and result.output_file is not None:
            task.status = "done"
            self.completed_ids.append(tid)
            file_part = result.output_file.name
            self.completed_list.insert(END, f"#{tid} {file_part}")
            self._append_log(f"[task-{tid}] [saved] {result.output_file}")
            self._append_log(f"[task-{tid}] [log] {result.log_file}")
        else:
            task.status = "failed"
            self._append_log(f"[task-{tid}] [failed] {result.log_file}")

        self._remove_active_row(tid)
        self._update_status_line()
        self._dispatch_jobs()
        self._finish_if_idle()

    def _finish_if_idle(self) -> None:
        if self.running and not self.pending_ids and not self.active_futures:
            self.running = False
            self.status_text.set("全部任务完成")
            self.start_btn.config(state=tk.NORMAL)
            if self.executor is not None:
                self.executor.shutdown(wait=False, cancel_futures=False)
                self.executor = None

    def _progress_text(self, task: DownloadTask) -> str:
        if task.total_fragments is None:
            if task.downloaded_fragments > 0:
                return f"{task.downloaded_fragments}/?"
            return "-"
        return f"{task.downloaded_fragments}/{task.total_fragments}"

    def _short_url(self, url: str, max_len: int = 100) -> str:
        if len(url) <= max_len:
            return url
        return url[: max_len - 3] + "..."

    def _upsert_active_row(self, task: DownloadTask) -> None:
        iid = str(task.task_id)
        values = (self._short_url(task.url), self._progress_text(task), task.status)
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
            f"下载中: 正在{len(self.active_futures)} | 待下载{len(self.pending_ids)} | 已完成{len(self.completed_ids)}"
        )

    def _append_log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                tag, value = self.log_queue.get_nowait()
                # 调试：记录所有收到的消息
                self._append_log(f"[debug-queue] 收到消息：{tag}")
                
                if tag == "__TASK_LOG__":
                    tid, msg = value  # type: ignore[misc]
                    self._append_log(f"[task-{tid}] {msg}")
                elif tag == "__PROGRESS__":
                    tid, downloaded, total = value  # type: ignore[misc]
                    self._append_log(f"[debug-progress] 进度更新：{downloaded}/{total}")
                    task = self.tasks.get(int(tid))
                    if task is not None:
                        task.downloaded_fragments = int(downloaded)
                        task.total_fragments = int(total) if total is not None else None
                        self._upsert_active_row(task)
                elif tag == "__TASK_DONE__":
                    self._handle_task_done(value)  # type: ignore[arg-type]
                elif tag == "__ERROR__":
                    self._append_log(f"[error] {value}")
                    messagebox.showerror("执行失败", str(value))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_logs)

    def _create_menu(self) -> None:
        """创建菜单栏"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 帮助菜单
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="帮助", menu=help_menu)
        help_menu.add_command(label="关于 PSiteDL", command=self._show_about)
        help_menu.add_command(label="检查更新", command=self._check_updates)
        help_menu.add_separator()
        help_menu.add_command(label="打开项目目录", command=self._open_project_dir)
        
        # 工具菜单
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="工具", menu=tools_menu)
        tools_menu.add_command(label="更新到最新版本", command=self._update_to_latest)
        
        # 状态栏显示版本（完整信息）
        version_text = f"PSiteDL {self.version_info['display']} | 分支：{self.version_info['branch'] or 'N/A'}"
        self.statusbar = ttk.Label(
            self.root,
            text=version_text,
            relief=tk.SUNKEN,
            anchor=W,
            padding=(5, 2),
        )
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def _show_about(self) -> None:
        """显示关于对话框"""
        about_text = f"""PSiteDL - 网页视频下载工具

版本：{self.version_info['display']}
分支：{self.version_info['branch'] or 'N/A'}
Commit: {self.version_info['commit'] or 'N/A'}

功能:
• 网页视频流探测与下载
• 支持多浏览器 Cookie 自动捕获
• 并发下载支持
• 实时进度显示

© 2026"""
        messagebox.showinfo("关于 PSiteDL", about_text)

    def _check_updates(self) -> None:
        """检查更新"""
        self.status_text.set("正在检查更新...")
        self.root.update()
        
        result = check_for_updates()
        
        if "error" in result:
            messagebox.showwarning("检查更新", f"检查失败：{result['error']}")
            self.status_text.set("就绪")
            return
        
        if result["has_update"]:
            changes_text = "\n".join(f"• {c}" for c in result["changes"][:10])
            msg = f"""发现新版本！

当前版本：{result['current_version']}
最新版本：{result['latest_version']}

更新内容:
{changes_text}

是否现在更新？"""
            if messagebox.askyesno("发现新版本", msg):
                self._update_to_latest()
        else:
            messagebox.showinfo("检查更新", f"已是最新版本：{result['current_version']}")
        
        self.status_text.set("就绪")

    def _update_to_latest(self) -> None:
        """更新到最新版本"""
        try:
            from pathlib import Path
            import subprocess
            
            project_root = Path(__file__).parent.parent.parent
            
            # 拉取最新代码
            self.status_text.set("正在更新...")
            self.root.update()
            
            result = subprocess.run(
                ["git", "pull"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                # 重新安装依赖
                subprocess.run(
                    ["git", "submodule", "update", "--init", "--recursive"],
                    cwd=project_root,
                    capture_output=True,
                    timeout=30
                )
                
                msg = f"更新成功！\n\n{result.stdout[:500]}"
                msg += "\n\n需要重启 GUI 以应用更新。是否现在重启？"
                if messagebox.askyesno("更新完成", msg):
                    self.root.quit()
                    # 重启 GUI
                    import sys
                    import os
                    os.execv(sys.executable, [sys.executable, "-m", "webvidgrab.site_gui"])
            else:
                messagebox.showerror("更新失败", f"错误：{result.stderr}")
                self.status_text.set("就绪")
        except Exception as e:
            messagebox.showerror("更新失败", str(e))
            self.status_text.set("就绪")

    def _open_project_dir(self) -> None:
        """打开项目目录"""
        try:
            project_root = Path(__file__).parent.parent.parent
            webbrowser.open(f"file://{project_root}")
        except Exception as e:
            messagebox.showerror("错误", f"无法打开目录：{e}")


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
