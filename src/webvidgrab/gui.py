from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import BOTH, END, LEFT, RIGHT, W, filedialog, messagebox, ttk

from webvidgrab.cli import (
    BROWSER_CHOICES,
    EJS_SOURCE_CHOICES,
    DownloadItem,
    build_session,
    build_strategy,
    download_with_yt_dlp,
    extract_youtube_tokens_from_har,
    extract_youtube_tokens_from_text,
    execute_downloads,
    extract_candidates,
    fetch_webpage,
    is_youtube_url,
    login_with_form,
    parse_key_values,
)
from webvidgrab.settings import (
    BACKEND_DEFAULTS,
    DEFAULT_TRANSLATE_CONFIG_PATH,
    resolve_translation_settings,
)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WebVidGrab - 网页视频下载")
        self.root.geometry("1100x760")

        self.url = tk.StringVar(value="https://m.youtube.com/watch?v=0puXA0G4Kg4&pp=0gcJCUABo7VqN5tD")
        self.output_dir = tk.StringVar(value=str(Path("./downloads").resolve()))
        self.backend = tk.StringVar(value="deepseek")
        self.model = tk.StringVar(value=str(BACKEND_DEFAULTS["deepseek"]["model"]))
        self.base_url = tk.StringVar(value=str(BACKEND_DEFAULTS["deepseek"]["base_url"]))
        self.api_key = tk.StringVar(value="")
        self.max_candidates = tk.StringVar(value="120")
        self.dry_run = tk.BooleanVar(value=False)
        self.login_url = tk.StringVar(value="")
        self.username = tk.StringVar(value="")
        self.password = tk.StringVar(value="")
        self.username_field = tk.StringVar(value="username")
        self.password_field = tk.StringVar(value="password")
        self.login_extra = tk.StringVar(value="")
        self.cookies_file = tk.StringVar(value="")
        self.yt_har_file = tk.StringVar(value="")
        self.cookies_from_browser = tk.StringVar(value="chrome")
        self.cookies_profile = tk.StringVar(value="Default")
        self.yt_ejs_source = tk.StringVar(value="github")
        self.yt_list_formats_on_fail = tk.BooleanVar(value=True)
        self.yt_extractor_args = tk.StringVar(value="youtube:player_client=web,web_safari")
        self.yt_extra_args = tk.StringVar(value="")
        self.yt_po_token = tk.StringVar(value="")
        self.yt_visitor_data = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="就绪")

        self.log_queue: queue.Queue[object] = queue.Queue()
        self.running = False
        self.strategy_items: list[DownloadItem] = []
        self.backend.trace_add("write", self._on_backend_change)

        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(top, text="任务参数", padding=10)
        form.pack(fill=BOTH)

        ttk.Label(form, text="网页 URL").grid(row=0, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.url).grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(form, text="保存目录").grid(row=1, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.output_dir).grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="浏览", command=self._pick_output_dir).grid(
            row=1, column=3, sticky="ew", padx=(8, 0), pady=4
        )

        ttk.Label(form, text="后端").grid(row=2, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.backend,
            values=["local", "openai", "deepseek"],
            state="readonly",
        ).grid(row=2, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="模型").grid(row=2, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.model).grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Label(form, text="Base URL").grid(row=3, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.base_url).grid(row=3, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(form, text="API Key(可选)").grid(row=4, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.api_key, show="*").grid(
            row=4, column=1, columnspan=3, sticky="ew", pady=4
        )

        ttk.Label(form, text="候选链接上限").grid(row=5, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.max_candidates, width=12).grid(row=5, column=1, sticky="w", pady=4)
        ttk.Checkbutton(form, text="仅生成策略，不下载", variable=self.dry_run).grid(
            row=5, column=2, columnspan=2, sticky=W, pady=4
        )

        ttk.Label(form, text="登录 URL(可选)").grid(row=6, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.login_url).grid(row=6, column=1, columnspan=3, sticky="ew", pady=4)
        ttk.Label(form, text="账号").grid(row=7, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.username).grid(row=7, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="密码").grid(row=7, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.password, show="*").grid(row=7, column=3, sticky="ew", pady=4)
        ttk.Label(form, text="账号字段名").grid(row=8, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.username_field).grid(row=8, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="密码字段名").grid(row=8, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.password_field).grid(row=8, column=3, sticky="ew", pady=4)
        ttk.Label(form, text="额外登录字段").grid(row=9, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.login_extra).grid(row=9, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(form, text="Cookies文件(可选)").grid(row=10, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.cookies_file).grid(row=10, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="浏览", command=self._pick_cookies_file).grid(
            row=10, column=3, sticky="ew", padx=(8, 0), pady=4
        )
        ttk.Label(form, text="YouTube HAR文件").grid(row=11, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_har_file).grid(row=11, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="浏览", command=self._pick_har_file).grid(
            row=11, column=3, sticky="ew", padx=(8, 0), pady=4
        )
        ttk.Label(form, text="浏览器会话").grid(row=12, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.cookies_from_browser,
            values=[""] + BROWSER_CHOICES,
            state="readonly",
        ).grid(row=12, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="浏览器Profile").grid(row=12, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.cookies_profile).grid(row=12, column=3, sticky="ew", pady=4)
        ttk.Label(form, text="YouTube EJS源").grid(row=13, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.yt_ejs_source,
            values=EJS_SOURCE_CHOICES,
            state="readonly",
        ).grid(row=13, column=1, sticky="ew", pady=4)
        ttk.Checkbutton(
            form,
            text="YouTube失败时自动列出格式",
            variable=self.yt_list_formats_on_fail,
        ).grid(row=13, column=2, columnspan=2, sticky=W, pady=4)
        ttk.Label(form, text="YouTube ExtractorArgs").grid(row=14, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_extractor_args).grid(
            row=14, column=1, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(form, text="YouTube PO Token").grid(row=15, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_po_token).grid(row=15, column=1, columnspan=3, sticky="ew", pady=4)
        ttk.Label(form, text="YouTube VisitorData").grid(row=16, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_visitor_data).grid(
            row=16, column=1, columnspan=3, sticky="ew", pady=4
        )
        ttk.Label(form, text="YouTube Extra Args").grid(row=17, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_extra_args).grid(
            row=17, column=1, columnspan=3, sticky="ew", pady=4
        )
        preset_row = ttk.Frame(form)
        preset_row.grid(row=18, column=0, columnspan=4, sticky="w", pady=(2, 4))
        ttk.Label(preset_row, text="参数预设").pack(side=LEFT)
        ttk.Button(
            preset_row,
            text="通用下载",
            command=lambda: self._set_yt_extra_preset('--format "bv*+ba/b" --merge-output-format mp4'),
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            preset_row,
            text="仅音频",
            command=lambda: self._set_yt_extra_preset('--extract-audio --audio-format mp3 --audio-quality 0'),
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            preset_row,
            text="排障模式",
            command=lambda: self._set_yt_extra_preset("-vU --list-formats"),
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            preset_row,
            text="清空",
            command=lambda: self._set_yt_extra_preset(""),
        ).pack(side=LEFT, padx=(8, 0))

        ttk.Label(form, text=f"配置文件: {DEFAULT_TRANSLATE_CONFIG_PATH}").grid(
            row=19, column=0, columnspan=4, sticky=W, pady=(2, 4)
        )

        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        run_row = ttk.Frame(top)
        run_row.pack(fill=BOTH, pady=(10, 0))
        self.run_btn = ttk.Button(run_row, text="开始分析与下载", command=self._start)
        self.run_btn.pack(side=LEFT)
        ttk.Label(run_row, textvariable=self.status_text).pack(side=LEFT, padx=(12, 0))

        strategy = ttk.LabelFrame(top, text="下载策略", padding=10)
        strategy.pack(fill=BOTH, expand=False, pady=(10, 0))

        cols = ("priority", "kind", "filename", "url")
        self.strategy_table = ttk.Treeview(strategy, columns=cols, show="headings", height=8)
        self.strategy_table.heading("priority", text="优先级")
        self.strategy_table.heading("kind", text="类型")
        self.strategy_table.heading("filename", text="文件名")
        self.strategy_table.heading("url", text="URL")
        self.strategy_table.column("priority", width=70, anchor="center")
        self.strategy_table.column("kind", width=100, anchor="center")
        self.strategy_table.column("filename", width=220)
        self.strategy_table.column("url", width=660)
        self.strategy_table.pack(fill=BOTH, expand=True)

        log_frame = ttk.LabelFrame(top, text="日志", padding=10)
        log_frame.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=14, state=tk.DISABLED)
        self.log_text.pack(fill=BOTH, expand=True)

    def _pick_output_dir(self) -> None:
        folder = filedialog.askdirectory(title="选择保存目录")
        if folder:
            self.output_dir.set(str(Path(folder).resolve()))

    def _pick_cookies_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择 cookies.txt 文件",
            filetypes=[("Text", "*.txt"), ("All Files", "*.*")],
        )
        if file_path:
            self.cookies_file.set(str(Path(file_path).resolve()))

    def _pick_har_file(self) -> None:
        file_path = filedialog.askopenfilename(
            title="选择 HAR 文件",
            filetypes=[("HAR", "*.har"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if file_path:
            self.yt_har_file.set(str(Path(file_path).resolve()))

    def _on_backend_change(self, *_: object) -> None:
        b = self.backend.get().strip() or "local"
        self.model.set(str(BACKEND_DEFAULTS[b]["model"]))
        self.base_url.set(str(BACKEND_DEFAULTS[b]["base_url"]))

    def _set_yt_extra_preset(self, value: str) -> None:
        self.yt_extra_args.set(value)
        if value:
            self._log(f"[youtube] 已应用参数预设: {value}")
        else:
            self._log("[youtube] 已清空额外参数")

    def _start(self) -> None:
        if self.running:
            return
        url = self.url.get().strip()
        if not url:
            messagebox.showwarning("提示", "请输入网页 URL。")
            return
        try:
            max_candidates = int(self.max_candidates.get().strip())
            if max_candidates < 10:
                raise ValueError()
        except ValueError:
            messagebox.showerror("参数错误", "候选链接上限必须是 >= 10 的整数。")
            return

        out_dir = Path(self.output_dir.get().strip()).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        backend = self.backend.get().strip() or "local"
        self.running = True
        self.run_btn.config(state=tk.DISABLED)
        self.status_text.set("运行中...")
        self._reset_strategy_table()

        worker = threading.Thread(
            target=self._run_pipeline,
            args=(
                url,
                out_dir,
                backend,
                self.model.get().strip(),
                self.base_url.get().strip(),
                self.api_key.get().strip(),
                max_candidates,
                self.dry_run.get(),
                self.login_url.get().strip(),
                self.username.get().strip(),
                self.password.get(),
                self.username_field.get().strip() or "username",
                self.password_field.get().strip() or "password",
                self.login_extra.get().strip(),
                self.cookies_file.get().strip(),
                self.yt_har_file.get().strip(),
                self.cookies_from_browser.get().strip(),
                self.cookies_profile.get().strip(),
                self.yt_ejs_source.get().strip() or "github",
                self.yt_list_formats_on_fail.get(),
                self.yt_extractor_args.get().strip(),
                self.yt_extra_args.get().strip(),
                self.yt_po_token.get().strip(),
                self.yt_visitor_data.get().strip(),
            ),
            daemon=True,
        )
        worker.start()

    def _run_pipeline(
        self,
        page_url: str,
        output_dir: Path,
        backend: str,
        model_name: str,
        base_url: str,
        api_key: str,
        max_candidates: int,
        dry_run: bool,
        login_url: str,
        username: str,
        password: str,
        username_field: str,
        password_field: str,
        login_extra_raw: str,
        cookies_file_raw: str,
        yt_har_file_raw: str,
        cookies_from_browser: str,
        cookies_profile: str,
        yt_ejs_source: str,
        yt_list_formats_on_fail: bool,
        yt_extractor_args: str,
        yt_extra_args: str,
        yt_po_token: str,
        yt_visitor_data: str,
    ) -> None:
        try:
            cookies_file = Path(cookies_file_raw).expanduser().resolve() if cookies_file_raw else None
            yt_har_file = Path(yt_har_file_raw).expanduser().resolve() if yt_har_file_raw else None
            opener, cookie_jar = build_session(cookies_file)
            login_extra = parse_key_values(
                [x.strip() for x in login_extra_raw.split(",") if x.strip()]
            )
            if login_url or username or password:
                if not (login_url and username and password):
                    raise RuntimeError("登录功能需要同时填写 登录URL/账号/密码。")
                self._log(f"[0/4] 登录中: {login_url}")
                login_with_form(
                    opener,
                    login_url=login_url,
                    username=username,
                    password=password,
                    username_field=username_field,
                    password_field=password_field,
                    extra_fields=login_extra,
                )

            if is_youtube_url(page_url) and not dry_run:
                po_token = yt_po_token or None
                visitor_data = yt_visitor_data or None
                if yt_har_file and (not po_token or not visitor_data):
                    har_po, har_visitor = extract_youtube_tokens_from_har(yt_har_file)
                    if not po_token and har_po:
                        po_token = har_po
                        self._log("[youtube] 已从HAR提取 PO Token。")
                    if not visitor_data and har_visitor:
                        visitor_data = har_visitor
                        self._log("[youtube] 已从HAR提取 VisitorData。")
                    if not po_token:
                        self._log("[youtube] HAR 未提取到 PO Token。")
                    if not visitor_data:
                        self._log("[youtube] HAR 未提取到 VisitorData。")
                if not po_token or not visitor_data:
                    try:
                        _, yt_html = fetch_webpage(page_url, opener=opener, referer=login_url or None)
                        html_po, html_visitor = extract_youtube_tokens_from_text(yt_html)
                        if not po_token and html_po:
                            po_token = html_po
                            self._log("[youtube] 已从页面源码提取 PO Token。")
                        if not visitor_data and html_visitor:
                            visitor_data = html_visitor
                            self._log("[youtube] 已从页面源码提取 VisitorData。")
                    except Exception:
                        pass

                self._log("[youtube] 检测到 YouTube 链接，使用 yt-dlp + 会话下载。")
                saved, yt_log_path = download_with_yt_dlp(
                    page_url,
                    output_dir,
                    cookies_from_browser=cookies_from_browser or None,
                    cookies_profile=cookies_profile or None,
                    cookies_file=cookies_file,
                    yt_ejs_source=yt_ejs_source,
                    yt_list_formats_on_fail=yt_list_formats_on_fail,
                    yt_extractor_args=yt_extractor_args or None,
                    yt_extra_args=yt_extra_args or None,
                    yt_po_token=po_token,
                    yt_visitor_data=visitor_data,
                )
                self._log(f"[youtube] yt-dlp完整日志: {yt_log_path}")
                if not saved:
                    raise RuntimeError("yt-dlp 执行后未发现输出文件。")
                for path in saved:
                    self._log(f"已保存: {path}")
                self.log_queue.put(("__DONE__",))
                return

            self._log(f"[1/4] 抓取网页: {page_url}")
            final_url, html = fetch_webpage(page_url, opener=opener, referer=login_url or None)
            self._log(f"最终 URL: {final_url}")

            self._log("[2/4] 提取候选链接...")
            candidates = extract_candidates(html, final_url, max_candidates)
            self._log(f"候选链接数量: {len(candidates)}")
            if not candidates:
                raise RuntimeError("网页源码中没有找到可疑候选链接。")

            self._log("[3/4] 调用大模型生成策略...")
            llm_settings = resolve_translation_settings(
                backend=backend,
                model_name=model_name or None,
                base_url=base_url or None,
                api_key=api_key or None,
                config_path=DEFAULT_TRANSLATE_CONFIG_PATH,
            )
            reasoning, downloads = build_strategy(
                llm_settings=llm_settings,
                page_url=page_url,
                final_url=final_url,
                html=html,
                candidates=candidates,
            )
            self._log(f"策略说明: {reasoning or 'N/A'}")
            self.strategy_items = downloads
            self.log_queue.put(("__STRATEGY__", downloads))

            if dry_run:
                self._log("Dry-run 模式，已跳过下载。")
                self.log_queue.put(("__DONE__",))
                return

            if not downloads:
                raise RuntimeError("大模型没有返回可下载目标。")

            self._log(f"[4/4] 下载到目录: {output_dir}")
            saved = execute_downloads(
                downloads,
                output_dir,
                opener=opener,
                cookie_jar=cookie_jar,
                referer=final_url,
            )
            if not saved:
                raise RuntimeError("下载执行完成，但没有成功保存文件。")
            for path in saved:
                self._log(f"已保存: {path}")
            self.log_queue.put(("__DONE__",))
        except Exception as exc:
            self.log_queue.put(("__ERROR__", str(exc)))

    def _reset_strategy_table(self) -> None:
        for item in self.strategy_table.get_children():
            self.strategy_table.delete(item)

    def _append_strategy(self, downloads: list[DownloadItem]) -> None:
        self._reset_strategy_table()
        for item in downloads:
            self.strategy_table.insert(
                "",
                END,
                values=(item.priority, item.kind, item.filename, item.url),
            )
        self._log(f"策略条目: {len(downloads)}")

    def _log(self, message: str) -> None:
        self.log_queue.put(("__LOG__", message))

    def _append_log_text(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                item = self.log_queue.get_nowait()
                if not isinstance(item, tuple):
                    continue
                tag = item[0]
                if tag == "__LOG__":
                    self._append_log_text(str(item[1]))
                elif tag == "__STRATEGY__":
                    self._append_strategy(item[1])
                elif tag == "__ERROR__":
                    self._append_log_text(f"错误: {item[1]}")
                    self.status_text.set("失败")
                    self.running = False
                    self.run_btn.config(state=tk.NORMAL)
                    messagebox.showerror("执行失败", str(item[1]))
                elif tag == "__DONE__":
                    self.status_text.set("完成")
                    self.running = False
                    self.run_btn.config(state=tk.NORMAL)
                    self._append_log_text("任务完成。")
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
