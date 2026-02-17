from __future__ import annotations

import queue
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from tkinter import BOTH, END, LEFT, W, filedialog, messagebox, ttk
import tkinter as tk

from webvidgrab.site_cli import ProbeResult, run_site_download


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
        self.root.title("PSiteDL")
        self.root.geometry("1120x760")

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
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(top, text="任务输入", padding=10)
        form.pack(fill=BOTH)

        ttk.Label(form, text="网页播放URL（每行一个）").grid(row=0, column=0, sticky=W, padx=(0, 8), pady=4)
        self.url_text = tk.Text(form, height=5)
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
        ttk.Label(ctrl, textvariable=self.status_text).pack(side=LEFT, padx=(12, 0))

        list_frame = ttk.Frame(top)
        list_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        pending_frame = ttk.LabelFrame(list_frame, text="待下载任务", padding=8)
        pending_frame.pack(fill=BOTH, expand=False)
        self.pending_list = tk.Listbox(pending_frame, height=7)
        self.pending_list.pack(fill=BOTH, expand=True)

        active_frame = ttk.LabelFrame(list_frame, text="正在下载任务", padding=8)
        active_frame.pack(fill=BOTH, expand=False, pady=(8, 0))
        self.active_tree = ttk.Treeview(
            active_frame,
            columns=("url", "progress", "status"),
            show="headings",
            height=8,
        )
        self.active_tree.heading("url", text="URL")
        self.active_tree.heading("progress", text="切片进度")
        self.active_tree.heading("status", text="状态")
        self.active_tree.column("url", width=680, anchor=W)
        self.active_tree.column("progress", width=140, anchor=W)
        self.active_tree.column("status", width=160, anchor=W)
        self.active_tree.pack(fill=BOTH, expand=True)

        completed_frame = ttk.LabelFrame(list_frame, text="已完成任务", padding=8)
        completed_frame.pack(fill=BOTH, expand=False, pady=(8, 0))
        self.completed_list = tk.Listbox(completed_frame, height=7)
        self.completed_list.pack(fill=BOTH, expand=True)

        log_frame = ttk.LabelFrame(top, text="运行日志", padding=10)
        log_frame.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, state=tk.DISABLED)
        self.log_text.pack(fill=BOTH, expand=True)

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
                if tag == "__TASK_LOG__":
                    tid, msg = value  # type: ignore[misc]
                    self._append_log(f"[task-{tid}] {msg}")
                elif tag == "__PROGRESS__":
                    tid, downloaded, total = value  # type: ignore[misc]
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


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
