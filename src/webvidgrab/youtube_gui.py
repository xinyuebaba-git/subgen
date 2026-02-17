from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
import json
import re
import shlex
from urllib.parse import unquote
from urllib import parse as urlparse
from datetime import datetime
from pathlib import Path
from tkinter import BOTH, END, LEFT, W, filedialog, messagebox, ttk
import tkinter as tk

from webvidgrab.cli import (
    build_session,
    extract_youtube_tokens_from_har,
    extract_youtube_tokens_from_text,
    fetch_webpage,
)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YouTube 参数探测器")
        self.root.geometry("980x680")

        self.url = tk.StringVar(
            value="https://m.youtube.com/watch?v=0puXA0G4Kg4&pp=0gcJCUABo7VqN5tD"
        )
        self.browser = tk.StringVar(value="chrome")
        self.profile = tk.StringVar(value="Default")
        self.har_file = tk.StringVar(value="")
        self.capture_seconds = tk.StringVar(value="90")
        self.manual_po_token = tk.StringVar(value="")
        self.manual_visitor_data = tk.StringVar(value="")
        self.yt_extra_args = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="就绪")
        self.auto_po_token: str | None = None
        self.auto_visitor_data: str | None = None
        self.auto_cookies_file: Path | None = None

        self.log_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.running = False

        self._build_ui()
        self._poll_logs()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=12)
        top.pack(fill=BOTH, expand=True)

        form = ttk.LabelFrame(top, text="输入", padding=10)
        form.pack(fill=BOTH)

        ttk.Label(form, text="YouTube URL").grid(row=0, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.url).grid(row=0, column=1, columnspan=3, sticky="ew", pady=4)

        ttk.Label(form, text="浏览器").grid(row=1, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Combobox(
            form,
            textvariable=self.browser,
            values=["chrome", "chromium", "firefox", "safari", "edge", "brave"],
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Label(form, text="Profile").grid(row=1, column=2, sticky=W, padx=(12, 8), pady=4)
        ttk.Entry(form, textvariable=self.profile).grid(row=1, column=3, sticky="ew", pady=4)

        ttk.Label(form, text="HAR文件(可选)").grid(row=2, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.har_file).grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        ttk.Button(form, text="浏览", command=self._pick_har).grid(row=2, column=3, sticky="ew", pady=4)

        ttk.Label(form, text="手动 PO Token").grid(row=3, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.manual_po_token).grid(
            row=3, column=1, columnspan=3, sticky="ew", pady=4
        )

        ttk.Label(form, text="手动 Visitor Data").grid(row=4, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.manual_visitor_data).grid(
            row=4, column=1, columnspan=3, sticky="ew", pady=4
        )

        ttk.Label(form, text="附加 yt-dlp 参数").grid(row=5, column=0, sticky=W, padx=(0, 8), pady=4)
        ttk.Entry(form, textvariable=self.yt_extra_args).grid(
            row=5, column=1, columnspan=3, sticky="ew", pady=4
        )

        note = (
            "说明：Token优先级=手动输入 > 自动抓取 > HAR > 页面源码。"
            "可在“附加 yt-dlp 参数”中粘贴如 --format \"bv*+ba/b\" --extractor-retries 5。"
        )
        ttk.Label(form, text=note, foreground="#666").grid(
            row=6, column=0, columnspan=4, sticky=W, pady=(4, 2)
        )

        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        run_row = ttk.Frame(top)
        run_row.pack(fill=BOTH, pady=(10, 0))
        self.run_btn = ttk.Button(run_row, text="开始探测", command=self._start_probe)
        self.run_btn.pack(side=LEFT)
        ttk.Button(run_row, text="批量探测", command=self._start_batch_probe).pack(
            side=LEFT, padx=(8, 0)
        )
        ttk.Button(run_row, text="自动抓取Token", command=self._start_auto_capture).pack(
            side=LEFT, padx=(8, 0)
        )
        ttk.Label(run_row, text="抓取秒数").pack(side=LEFT, padx=(12, 6))
        ttk.Entry(run_row, textvariable=self.capture_seconds, width=6).pack(side=LEFT)
        ttk.Label(run_row, textvariable=self.status_text).pack(side=LEFT, padx=(12, 0))

        out = ttk.LabelFrame(top, text="结果与日志", padding=10)
        out.pack(fill=BOTH, expand=True, pady=(10, 0))

        batch_frame = ttk.LabelFrame(out, text="批量URL（每行一个，可选）", padding=8)
        batch_frame.pack(fill=BOTH, expand=False, pady=(0, 8))
        self.batch_urls_text = tk.Text(batch_frame, height=5)
        self.batch_urls_text.pack(fill=BOTH, expand=True)

        self.log_text = tk.Text(out, state=tk.DISABLED)
        self.log_text.pack(fill=BOTH, expand=True)

    def _pick_har(self) -> None:
        p = filedialog.askopenfilename(
            title="选择 HAR 文件",
            filetypes=[("HAR", "*.har"), ("JSON", "*.json"), ("All Files", "*.*")],
        )
        if p:
            self.har_file.set(str(Path(p).resolve()))

    def _start_probe(self) -> None:
        if self.running:
            return
        raw_url = self.url.get().strip()
        url = self._normalize_youtube_url(raw_url)
        if url != raw_url:
            self.url.set(url)
            self._log(f"[normalize-url] {raw_url} -> {url}")
        if not url:
            messagebox.showerror("参数错误", "请填写 YouTube URL")
            return
        self.running = True
        self.status_text.set("探测中...")
        self.run_btn.config(state=tk.DISABLED)
        worker = threading.Thread(
            target=self._run_probe,
            args=(url, self.browser.get().strip(), self.profile.get().strip(), self.har_file.get().strip()),
            daemon=True,
        )
        worker.start()

    def _start_auto_capture(self) -> None:
        if self.running:
            return
        raw_url = self.url.get().strip()
        url = self._normalize_youtube_url(raw_url)
        if url != raw_url:
            self.url.set(url)
            self._log(f"[normalize-url] {raw_url} -> {url}")
        if not url:
            messagebox.showerror("参数错误", "请填写 YouTube URL")
            return
        try:
            seconds = int(self.capture_seconds.get().strip())
            if seconds < 10:
                raise ValueError()
        except ValueError:
            messagebox.showerror("参数错误", "抓取秒数建议 >= 10")
            return
        self.running = True
        self.status_text.set("自动抓取中...")
        self.run_btn.config(state=tk.DISABLED)
        worker = threading.Thread(
            target=self._run_auto_capture,
            args=(url, seconds, self.browser.get().strip(), self.profile.get().strip()),
            daemon=True,
        )
        worker.start()

    def _start_batch_probe(self) -> None:
        if self.running:
            return
        raw = self.batch_urls_text.get("1.0", END)
        urls = [self._normalize_youtube_url(x.strip()) for x in raw.splitlines() if x.strip()]
        urls = [u for u in urls if u]
        if not urls:
            messagebox.showwarning("提示", "请在“批量URL”区域每行填写一个链接。")
            return
        self.running = True
        self.status_text.set("批量探测中...")
        self.run_btn.config(state=tk.DISABLED)
        worker = threading.Thread(
            target=self._run_batch_probe,
            args=(urls, self.browser.get().strip(), self.profile.get().strip(), self.har_file.get().strip()),
            daemon=True,
        )
        worker.start()

    def _normalize_youtube_url(self, raw: str) -> str:
        url = (raw or "").strip()
        if not url:
            return ""
        try:
            parsed = urlparse.urlparse(url)
        except Exception:
            return url
        host = (parsed.hostname or "").lower()
        if "youtube.com" not in host and "youtu.be" not in host:
            return url
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc
        if host == "m.youtube.com":
            netloc = netloc.replace("m.youtube.com", "www.youtube.com")
        q = urlparse.parse_qsl(parsed.query, keep_blank_values=True)
        q = [(k, v) for (k, v) in q if k.lower() != "pp"]
        query = urlparse.urlencode(q)
        rebuilt = parsed._replace(scheme=scheme, netloc=netloc, query=query)
        return urlparse.urlunparse(rebuilt)

    def _run_auto_capture(self, url: str, seconds: int, browser: str, profile: str) -> None:
        try:
            self._log("[auto-capture] 即将打开受控浏览器，请在弹出的页面播放视频。")
            po, visitor, cookies_file = self._capture_tokens_with_playwright(
                url, seconds, browser, profile
            )
            self.auto_po_token = po
            self.auto_visitor_data = visitor
            self.auto_cookies_file = cookies_file
            self._log(f"[auto-capture-po-token] {'yes' if po else 'no'}")
            self._log(f"[auto-capture-visitor-data] {'yes' if visitor else 'no'}")
            self._log(f"[auto-capture-cookies-file] {cookies_file}")
            self.log_queue.put(("__DONE__", "自动抓取完成"))
        except Exception as exc:
            self.log_queue.put(("__ERROR__", str(exc)))

    def _run_probe(self, url: str, browser: str, profile: str, har_path: str) -> None:
        try:
            self._log(f"[url] {url}")
            self._log(f"[browser] {browser}:{profile or '(default)'}")

            po_token: str | None = self.manual_po_token.get().strip() or self.auto_po_token
            visitor_data: str | None = (
                self.manual_visitor_data.get().strip() or self.auto_visitor_data
            )
            extra_args = self.yt_extra_args.get().strip()
            self._log(f"[manual-po-token] {'yes' if self.manual_po_token.get().strip() else 'no'}")
            self._log(
                f"[manual-visitor-data] {'yes' if self.manual_visitor_data.get().strip() else 'no'}"
            )
            self._log(f"[auto-po-token] {'yes' if po_token else 'no'}")
            self._log(f"[auto-visitor-data] {'yes' if visitor_data else 'no'}")
            self._log(f"[yt-extra-args] {extra_args if extra_args else '(none)'}")

            if har_path:
                hp = Path(har_path).expanduser().resolve()
                if hp.exists():
                    h_po, h_visitor = extract_youtube_tokens_from_har(hp)
                    if not po_token and h_po:
                        po_token = h_po
                    if not visitor_data and h_visitor:
                        visitor_data = h_visitor
                    self._log(f"[har] {hp}")
                    self._log(f"[har-po-token] {'yes' if h_po else 'no'}")
                    self._log(f"[har-visitor-data] {'yes' if h_visitor else 'no'}")
                else:
                    self._log(f"[har] file not found: {hp}")

            if not po_token or not visitor_data:
                opener, _ = build_session()
                _, html = fetch_webpage(url, opener=opener)
                html_po, html_visitor = extract_youtube_tokens_from_text(html)
                if not po_token and html_po:
                    po_token = html_po
                    self._log("[page-po-token] yes")
                if not visitor_data and html_visitor:
                    visitor_data = html_visitor
                    self._log("[page-visitor-data] yes")

            self._log(f"[final-po-token] {'yes' if po_token else 'no'}")
            self._log(f"[final-visitor-data] {'yes' if visitor_data else 'no'}")

            probe_ok, probe_msg, log_file = self._probe_formats(
                url=url,
                browser=browser,
                profile=profile,
                po_token=po_token,
                visitor_data=visitor_data,
                cookies_file=self.auto_cookies_file,
                extra_args=extra_args,
            )
            self._log(f"[probe-result] {'可下载格式可见' if probe_ok else '未发现可下载格式'}")
            self._log(probe_msg)
            self._log(f"[probe-log-file] {log_file}")
            self.log_queue.put(("__DONE__", "完成"))
        except Exception as exc:
            self.log_queue.put(("__ERROR__", str(exc)))

    def _run_batch_probe(self, urls: list[str], browser: str, profile: str, har_path: str) -> None:
        try:
            self._log(f"[batch-total] {len(urls)}")
            self._log(f"[browser] {browser}:{profile or '(default)'}")

            base_po = self.manual_po_token.get().strip() or self.auto_po_token
            base_visitor = self.manual_visitor_data.get().strip() or self.auto_visitor_data
            extra_args = self.yt_extra_args.get().strip()
            self._log(f"[manual-po-token] {'yes' if self.manual_po_token.get().strip() else 'no'}")
            self._log(
                f"[manual-visitor-data] {'yes' if self.manual_visitor_data.get().strip() else 'no'}"
            )
            self._log(f"[yt-extra-args] {extra_args if extra_args else '(none)'}")
            if har_path:
                hp = Path(har_path).expanduser().resolve()
                if hp.exists():
                    hpo, hvisitor = extract_youtube_tokens_from_har(hp)
                    base_po = base_po or hpo
                    base_visitor = base_visitor or hvisitor
                    self._log(f"[batch-har] {hp}")
                    self._log(f"[batch-har-po-token] {'yes' if hpo else 'no'}")
                    self._log(f"[batch-har-visitor-data] {'yes' if hvisitor else 'no'}")

            ok_count = 0
            fail_count = 0
            for idx, url in enumerate(urls, start=1):
                self._log(f"[batch-{idx}] {url}")
                po_token = base_po
                visitor_data = base_visitor
                if not po_token or not visitor_data:
                    try:
                        opener, _ = build_session()
                        _, html = fetch_webpage(url, opener=opener)
                        html_po, html_visitor = extract_youtube_tokens_from_text(html)
                        po_token = po_token or html_po
                        visitor_data = visitor_data or html_visitor
                    except Exception:
                        pass

                probe_ok, _probe_msg, probe_log = self._probe_formats(
                    url=url,
                    browser=browser,
                    profile=profile,
                    po_token=po_token,
                    visitor_data=visitor_data,
                    cookies_file=self.auto_cookies_file,
                    extra_args=extra_args,
                )
                if probe_ok:
                    ok_count += 1
                    self._log(f"[batch-{idx}-result] OK")
                else:
                    fail_count += 1
                    self._log(f"[batch-{idx}-result] FAIL")
                self._log(f"[batch-{idx}-probe-log] {probe_log}")

            self._log(f"[batch-summary] ok={ok_count} fail={fail_count}")
            self.log_queue.put(("__DONE__", "批量完成"))
        except Exception as exc:
            self.log_queue.put(("__ERROR__", str(exc)))

    def _probe_formats(
        self,
        *,
        url: str,
        browser: str,
        profile: str,
        po_token: str | None,
        visitor_data: str | None,
        cookies_file: Path | None,
        extra_args: str,
    ) -> tuple[bool, str, Path]:
        ytdlp = shutil.which("yt-dlp")
        if not ytdlp:
            raise RuntimeError("未找到 yt-dlp。请先安装后重试。")

        base_client_args = [
            "youtube:player_client=web,web_safari",
            "youtube:player_client=mweb,web,web_safari",
            "youtube:player_client=tv,web_safari,web",
            "youtube:player_client=tv_embedded,web_safari,web",
        ]
        client_args: list[str] = []
        for item in base_client_args:
            if item not in client_args:
                client_args.append(item)

        ejs_sources = ["github", "npm", "none"]
        attempt_outputs: list[str] = []
        last_stdout = ""
        last_stderr = ""
        last_cmd: list[str] = []
        has_video = False
        attempt_no = 0
        extra_tokens: list[str] = []
        if extra_args:
            try:
                extra_tokens = shlex.split(extra_args)
            except ValueError as exc:
                raise RuntimeError(f"附加 yt-dlp 参数格式错误: {exc}") from exc

        attempt_matrix: list[tuple[str, str]] = []
        for ext in client_args:
            attempt_matrix.append(("github", ext))
        for src in ejs_sources[1:]:
            attempt_matrix.append((src, client_args[0]))

        for src, player_arg in attempt_matrix[:8]:
            attempt_no += 1
            extractor_parts = [player_arg]
            if po_token:
                extractor_parts.append(
                    "youtube:po_token="
                    f"web.gvs+{po_token},web.player+{po_token},mweb.gvs+{po_token}"
                )
            if visitor_data:
                extractor_parts.append(f"youtube:visitor_data={visitor_data}")
            cmd = [ytdlp]
            if cookies_file and cookies_file.exists():
                cmd.extend(["--cookies", str(cookies_file)])
            else:
                cmd.extend(["--cookies-from-browser", f"{browser}:{profile}" if profile else browser])
            if src in {"github", "npm"}:
                cmd.extend(["--remote-components", f"ejs:{src}"])
            if extra_tokens:
                cmd.extend(extra_tokens)
            cmd.extend(["--extractor-args", ";".join(extractor_parts), "--list-formats", url])
            proc = subprocess.run(cmd, capture_output=True, text=True)
            attempt_outputs.append(
                f"[attempt-{attempt_no}] ejs={src} player={player_arg}\n"
                f"[attempt-{attempt_no}-cmd] {' '.join(cmd)}\n"
                f"[attempt-{attempt_no}-stdout]\n{proc.stdout or ''}\n"
                f"[attempt-{attempt_no}-stderr]\n{proc.stderr or ''}\n"
            )
            last_stdout = proc.stdout or ""
            last_stderr = proc.stderr or ""
            last_cmd = cmd

            in_table = False
            for line in last_stdout.splitlines():
                s = line.strip()
                if not s:
                    continue
                if s.startswith("ID  EXT"):
                    in_table = True
                    continue
                if not in_table or s.startswith("---"):
                    continue
                parts = s.split()
                if len(parts) < 2:
                    continue
                fmt_id = parts[0].lower()
                ext = parts[1].lower()
                if ext == "mhtml" or fmt_id in {"sb0", "sb1", "sb2"}:
                    continue
                has_video = True
                break
            if has_video:
                break

        log_file = (
            Path.cwd()
            / "logs"
            / "webvidgrab"
            / f"yt-probe-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("\n".join(attempt_outputs), encoding="utf-8")

        final_out = (last_stdout + "\n" + last_stderr).strip()
        diagnosis: list[str] = []
        if "n challenge solving failed" in (last_stderr or ""):
            diagnosis.append("检测到 YouTube n challenge 失败")
        if not po_token:
            diagnosis.append("未提供 po_token")
        if not visitor_data:
            diagnosis.append("未提供 visitor_data")
        hint = f"[diagnosis] {'; '.join(diagnosis)}\n" if diagnosis else ""
        msg = (
            hint + (final_out[-1600:] if final_out else f"No output. last cmd: {' '.join(last_cmd)}")
        )
        return has_video, msg, log_file

    def _capture_tokens_with_playwright(
        self,
        url: str,
        seconds: int,
        browser_name: str,
        profile: str,
    ) -> tuple[str | None, str | None, Path]:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:
            raise RuntimeError(
                "缺少 playwright。请先执行: python -m pip install playwright"
            ) from exc

        po_token: str | None = None
        visitor_data: str | None = None
        capture_snippets: list[str] = []

        def scan_text_blob(text: str) -> tuple[str | None, str | None]:
            if not text:
                return None, None
            text = text.replace("\\u003d", "=").replace("\\u0026", "&")
            po = None
            vd = None
            for pattern in [
                r'"poToken"\s*:\s*"([^"]+)"',
                r'"po_token"\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pattern, text)
                if m:
                    po = m.group(1).strip()
                    break
            for pattern in [
                r'"visitorData"\s*:\s*"([^"]+)"',
                r'"X-Goog-Visitor-Id"\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pattern, text)
                if m:
                    vd = m.group(1).strip()
                    break
            return po, vd

        profile_root: Path | None = None
        profile_arg = profile or "Default"
        if browser_name == "chrome":
            base = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
            d = base / profile_arg
            if d.exists() and base.exists():
                profile_root = base
                self._log(f"[auto-capture] 使用Chrome配置目录: {d}")
            else:
                self._log(f"[auto-capture] 未找到配置目录: {d}，改用临时浏览器上下文。")

        capture_notes: list[str] = []

        with sync_playwright() as p:
            if profile_root is not None:
                try:
                    context = p.chromium.launch_persistent_context(
                        user_data_dir=str(profile_root),
                        channel="chrome",
                        headless=False,
                        args=[f"--profile-directory={profile_arg}"],
                    )
                    pages = context.pages
                    page = pages[0] if pages else context.new_page()
                except Exception as exc:
                    capture_notes.append(f"persistent context failed: {exc}")
                    browser = p.chromium.launch(channel="chrome", headless=False)
                    context = browser.new_context()
                    page = context.new_page()
            else:
                browser = p.chromium.launch(channel="chrome", headless=False)
                context = browser.new_context()
                page = context.new_page()

            def is_target_url(u: str) -> bool:
                return (
                    "youtubei/v1/" in u
                    or "youtubei.googleapis.com" in u
                    or "player?key=" in u
                )

            def on_request(req) -> None:
                nonlocal po_token, visitor_data
                try:
                    if not is_target_url(req.url):
                        return
                    headers = req.headers or {}
                    if len(capture_snippets) < 80:
                        header_vd = headers.get("x-goog-visitor-id") or headers.get("X-Goog-Visitor-Id") or ""
                        post = req.post_data or ""
                        capture_snippets.append(
                            "[request] "
                            + self._sanitize_capture_snippet(
                                f"url={req.url}\nvisitor_header={header_vd}\npost={post[:1800]}"
                            )
                        )
                    if not visitor_data:
                        v = headers.get("x-goog-visitor-id") or headers.get("X-Goog-Visitor-Id")
                        if v:
                            visitor_data = v
                    post = req.post_data or ""
                    if post:
                        po, vd = scan_text_blob(post)
                        if not po_token and po:
                            po_token = po
                        if not visitor_data and vd:
                            visitor_data = vd
                        try:
                            payload = json.loads(post)
                            sid = payload.get("serviceIntegrityDimensions")
                            if isinstance(sid, dict):
                                po = sid.get("poToken")
                                if isinstance(po, str) and po.strip() and not po_token:
                                    po_token = po.strip()
                            ctx = payload.get("context")
                            if isinstance(ctx, dict):
                                client = ctx.get("client")
                                if isinstance(client, dict):
                                    vd = client.get("visitorData")
                                    if isinstance(vd, str) and vd.strip() and not visitor_data:
                                        visitor_data = vd.strip()
                        except Exception:
                            pass
                except Exception:
                    return

            def on_response(resp) -> None:
                nonlocal po_token, visitor_data
                try:
                    if not is_target_url(resp.url):
                        return
                    text = ""
                    try:
                        text = resp.text() or ""
                    except Exception:
                        text = ""
                    if not text:
                        return
                    if len(capture_snippets) < 80:
                        capture_snippets.append(
                            "[response] "
                            + self._sanitize_capture_snippet(
                                f"url={resp.url}\nbody={text[:1800]}"
                            )
                        )
                    text = unquote(text).replace("\\u003d", "=").replace("\\u0026", "&")
                    if not po_token:
                        m = re.search(r'"poToken"\s*:\s*"([^"]+)"', text)
                        if m:
                            po_token = m.group(1).strip()
                    if not visitor_data:
                        m = re.search(r'"visitorData"\s*:\s*"([^"]+)"', text)
                        if m:
                            visitor_data = m.group(1).strip()
                except Exception:
                    return

            context.on("request", on_request)
            context.on("response", on_response)
            page.goto(url, wait_until="domcontentloaded")
            try:
                page.bring_to_front()
            except Exception:
                pass
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass
            # Try to trigger playback so player API requests fire.
            for sel in [
                "button[aria-label*='Play']",
                "button[title*='Play']",
                ".ytp-large-play-button",
                ".ytp-play-button",
            ]:
                try:
                    page.locator(sel).first.click(timeout=1500)
                    break
                except Exception:
                    continue
            try:
                page.keyboard.press("k")
            except Exception:
                pass
            try:
                page.mouse.click(400, 300)
            except Exception:
                pass
            # Small auto-scroll to trigger additional network requests from player.
            try:
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(500)
                page.mouse.wheel(0, -800)
            except Exception:
                pass
            self._log(f"[auto-capture] 已打开页面，等待 {seconds}s 捕获请求...")
            deadline = time.time() + seconds
            last_action = 0.0
            action_step = 0
            while time.time() < deadline and (not po_token or not visitor_data):
                page.wait_for_timeout(500)
                now = time.time()
                if now - last_action >= 6:
                    try:
                        if action_step % 4 == 0:
                            page.keyboard.press("k")
                        elif action_step % 4 == 1:
                            page.keyboard.press("ArrowRight")
                        elif action_step % 4 == 2:
                            page.mouse.click(420, 320)
                        else:
                            page.mouse.wheel(0, 800)
                    except Exception:
                        pass
                    action_step += 1
                    last_action = now
            # Final in-page fallback scan: ytcfg/localStorage/sessionStorage.
            try:
                blob = page.evaluate(
                    """() => {
                        const out = {};
                        try {
                          if (window.ytcfg && typeof window.ytcfg.get === 'function') {
                            out.ytcfg_visitor = window.ytcfg.get('VISITOR_DATA') || '';
                            out.ytcfg_po = window.ytcfg.get('PO_TOKEN') || '';
                          }
                        } catch (_) {}
                        try {
                          const keys = [];
                          for (let i = 0; i < localStorage.length; i++) keys.push(localStorage.key(i));
                          out.local = keys.map(k => [k, localStorage.getItem(k)]);
                        } catch (_) {}
                        try {
                          const keys = [];
                          for (let i = 0; i < sessionStorage.length; i++) keys.push(sessionStorage.key(i));
                          out.session = keys.map(k => [k, sessionStorage.getItem(k)]);
                        } catch (_) {}
                        return JSON.stringify(out);
                    }"""
                )
                if isinstance(blob, str) and blob:
                    if not visitor_data:
                        m = re.search(r'"ytcfg_visitor"\s*:\s*"([^"]+)"', blob)
                        if m:
                            visitor_data = m.group(1).strip()
                    if not po_token:
                        m = re.search(r'"ytcfg_po"\s*:\s*"([^"]+)"', blob)
                        if m:
                            po_token = m.group(1).strip()
                    if not po_token:
                        m = re.search(r'poToken[^"]*"\s*:\s*"([^"]+)"', blob)
                        if m:
                            po_token = m.group(1).strip()
                    if not visitor_data:
                        m = re.search(r'visitorData[^"]*"\s*:\s*"([^"]+)"', blob)
                        if m:
                            visitor_data = m.group(1).strip()
            except Exception:
                pass
            cookies = context.cookies()
            cookies_file = self._write_netscape_cookies(cookies)
            if capture_notes or capture_snippets:
                notes_path = (
                    Path.cwd()
                    / "logs"
                    / "webvidgrab"
                    / f"auto-capture-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
                )
                notes_path.parent.mkdir(parents=True, exist_ok=True)
                lines: list[str] = []
                lines.append(f"[url] {url}")
                lines.append(f"[browser] {browser_name}:{profile_arg}")
                lines.append(f"[po-token-found] {'yes' if po_token else 'no'}")
                lines.append(f"[visitor-data-found] {'yes' if visitor_data else 'no'}")
                if capture_notes:
                    lines.append("[notes]")
                    lines.extend(capture_notes)
                if capture_snippets:
                    lines.append("[snippets]")
                    lines.extend(capture_snippets)
                notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                self._log(f"[auto-capture-detail-log] {notes_path}")
            context.close()

        return po_token, visitor_data, cookies_file

    def _write_netscape_cookies(self, cookies: list[dict]) -> Path:
        out = (
            Path.cwd()
            / "logs"
            / "webvidgrab"
            / f"auto-cookies-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Netscape HTTP Cookie File"]
        for c in cookies:
            domain = str(c.get("domain") or "")
            if not domain:
                continue
            include_sub = "TRUE" if domain.startswith(".") else "FALSE"
            path = str(c.get("path") or "/")
            secure = "TRUE" if c.get("secure") else "FALSE"
            expires = c.get("expires")
            # Netscape cookie format expects non-negative unix timestamp.
            if isinstance(expires, (int, float)):
                exp_i = int(expires)
                expires_s = str(exp_i if exp_i > 0 else 0)
            else:
                expires_s = "0"
            name = str(c.get("name") or "")
            value = str(c.get("value") or "")
            if not name:
                continue
            lines.append("\t".join([domain, include_sub, path, secure, expires_s, name, value]))
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def _sanitize_capture_snippet(self, text: str) -> str:
        s = text or ""
        patterns = [
            (r'("poToken"\s*:\s*")[^"]+(")', r"\1***\2"),
            (r'("visitorData"\s*:\s*")[^"]+(")', r"\1***\2"),
            (r'("authorization"\s*:\s*")[^"]+(")', r"\1***\2"),
            (r"(Cookie:\s*)[^\n]+", r"\1***"),
            (r"(SAPISID=)[^;\\s]+", r"\1***"),
            (r"(HSID=)[^;\\s]+", r"\1***"),
            (r"(SSID=)[^;\\s]+", r"\1***"),
            (r"(APISID=)[^;\\s]+", r"\1***"),
        ]
        for pat, rep in patterns:
            s = re.sub(pat, rep, s, flags=re.IGNORECASE)
        return s

    def _log(self, msg: str) -> None:
        self.log_queue.put(("__LOG__", msg))

    def _append_log(self, text: str) -> None:
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(END, text + "\n")
        self.log_text.see(END)
        self.log_text.config(state=tk.DISABLED)

    def _poll_logs(self) -> None:
        try:
            while True:
                tag, value = self.log_queue.get_nowait()
                if tag == "__LOG__":
                    self._append_log(value)
                elif tag == "__DONE__":
                    self.running = False
                    self.status_text.set(value)
                    self.run_btn.config(state=tk.NORMAL)
                elif tag == "__ERROR__":
                    self.running = False
                    self.status_text.set("失败")
                    self.run_btn.config(state=tk.NORMAL)
                    self._append_log(f"[error] {value}")
                    messagebox.showerror("执行失败", value)
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
