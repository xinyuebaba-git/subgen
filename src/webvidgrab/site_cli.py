from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from urllib import parse as urlparse
from urllib import request as urlrequest

DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


@dataclass
class ProbeResult:
    page_url: str
    final_candidate: str | None
    candidate_count: int
    output_file: Path | None
    log_file: Path
    ok: bool


def _fetch_text(url: str, referer: str | None = None) -> str:
    headers = {"User-Agent": DEFAULT_UA, "Accept": "*/*"}
    if referer:
        headers["Referer"] = referer
    req = urlrequest.Request(url, headers=headers)
    with urlrequest.urlopen(req, timeout=25) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="ignore")


def _sanitize_filename_stem(stem: str) -> str:
    s = re.sub(r'[\\/:*?"<>|]+', "_", (stem or "").strip())
    s = re.sub(r"\s+", " ", s).strip(" .")
    return s[:180]


def _extract_page_title(html: str) -> str | None:
    patterns = [
        r'<meta[^>]+property=["\']og:title["\'][^>]*content=["\']([^"\']+)["\']',
        r'<meta[^>]+content=["\']([^"\']+)["\'][^>]*property=["\']og:title["\']',
        r"<title[^>]*>(.*?)</title>",
    ]
    for p in patterns:
        m = re.search(p, html, flags=re.IGNORECASE | re.DOTALL)
        if not m:
            continue
        title = _sanitize_filename_stem(re.sub(r"\s+", " ", (m.group(1) or "").strip()))
        if title:
            return title
    return None


def _candidate_score(url: str) -> int:
    low = url.lower()
    score = 0
    if ".m3u8" in low:
        score += 200
    if ".mpd" in low:
        score += 180
    if ".mp4" in low or ".webm" in low:
        score += 100
    for k in ("2160", "1440", "1080", "720", "480", "360"):
        if k in low:
            score += int(k)
            break
    if any(k in low for k in ("master", "manifest", "playlist", "index")):
        score += 30
    return score


def _extract_candidates(html: str, base_url: str) -> list[str]:
    patterns = [
        r"""https?://[^\s"'<>\\]+""",
        r"""https?:\\/\\/[^\s"'<>\\]+""",
        r"""["']([^"'<>]+(?:\.m3u8|\.mpd|\.mp4|\.webm|\.m4s|\.ts)(?:\?[^"'<>]*)?)["']""",
    ]
    urls: list[str] = []
    for p in patterns:
        for raw in re.findall(p, html, flags=re.IGNORECASE):
            item = str(raw).strip().strip("\"'").replace("\\/", "/")
            if item.startswith("//"):
                item = "https:" + item
            item = urlparse.urljoin(base_url, item)
            low = item.lower()
            if not low.startswith(("http://", "https://")):
                continue
            if any(k in low for k in (".m3u8", ".mpd", ".mp4", ".webm", ".m4s", ".ts", "manifest", "playlist")):
                urls.append(item)
    dedup = list(dict.fromkeys(urls))
    dedup.sort(key=_candidate_score, reverse=True)
    return dedup


def _export_browser_cookies(
    browser: str,
    profile: str,
    log: Callable[[str], None],
) -> Path | None:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return None
    out = Path(tempfile.gettempdir()) / f"sitegrab-cookies-{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    browser_arg = f"{browser}:{profile}" if profile else browser
    cmd = [
        ytdlp,
        "--cookies-from-browser",
        browser_arg,
        "--cookies",
        str(out),
        "--simulate",
        "--skip-download",
        "https://cn.pornhub.com/",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0 and out.exists():
        log(f"[runtime-capture] 已导出浏览器cookie: {out}")
        return out
    return None


def _inject_cookies(context, url: str, cookie_file: Path) -> int:
    host = (urlparse.urlparse(url).hostname or "").lower()
    secure = urlparse.urlparse(url).scheme.lower() == "https"
    cookies: list[dict] = []
    for line in cookie_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        domain, _flag, path, secure_raw, _expires, name, value = parts
        d = domain.lstrip(".").lower()
        if not d or not name:
            continue
        if host and not (host == d or host.endswith("." + d)):
            continue
        if secure_raw.upper() == "TRUE" and not secure:
            continue
        cookies.append(
            {
                "name": name,
                "value": value,
                "domain": domain if domain.startswith(".") else d,
                "path": path or "/",
                "secure": secure_raw.upper() == "TRUE",
            }
        )
    if cookies:
        context.add_cookies(cookies)
    return len(cookies)


def _auto_confirm_age_gate(page, log: Callable[[str], None], quiet: bool = False) -> None:
    selectors = [
        "button:has-text('我已年满18岁')",
        "button:has-text('我已滿18歲')",
        "button:has-text('年满18')",
        "button:has-text('滿18')",
        "button:has-text('18+')",
        "button:has-text('I am 18')",
        "button:has-text('Over 18')",
        "button:has-text('Continue')",
        "button:has-text('Enter')",
        "button:has-text('Agree')",
        "a:has-text('18+')",
        "a:has-text('Continue')",
        "a:has-text('Enter')",
        "a:has-text('进入')",
        "a:has-text('進入')",
        "[role='button']:has-text('18+')",
        "[role='button']:has-text('Continue')",
        "input[type='submit'][value*='18']",
        "input[type='button'][value*='18']",
    ]
    hit = False
    frames = [page] + list(page.frames)
    for frame in frames:
        for sel in selectors:
            try:
                loc = frame.locator(sel).first
                if loc and loc.is_visible(timeout=250):
                    loc.click(timeout=1000)
                    hit = True
            except Exception:
                continue
        try:
            clicked = frame.evaluate(
                """() => {
                    const tokens = ['18+','年满18','滿18','over 18','i am 18','continue','enter','agree','进入','進入'];
                    const nodes = Array.from(document.querySelectorAll('button,a,[role="button"],div,span,input[type="button"],input[type="submit"]'));
                    const norm = s => (s || '').toLowerCase().replace(/\\s+/g, ' ').trim();
                    for (const el of nodes) {
                        const text = norm(el.innerText || el.textContent || el.value || '');
                        if (!text || !tokens.some(t => text.includes(t))) continue;
                        const st = window.getComputedStyle(el);
                        const visible = st && st.display !== 'none' && st.visibility !== 'hidden' && el.offsetParent !== null;
                        if (!visible) continue;
                        try { el.click(); return text.slice(0, 60); } catch (_) {}
                    }
                    return '';
                }"""
            )
            if isinstance(clicked, str) and clicked.strip():
                hit = True
                if not quiet:
                    log(f"[runtime-capture] 自动点击年龄确认: {clicked.strip()}")
        except Exception:
            pass
    if hit and not quiet:
        log("[runtime-capture] 已自动处理年龄确认弹层。")


def _capture_runtime_candidates(
    *,
    url: str,
    browser: str,
    profile: str,
    seconds: int,
    log: Callable[[str], None],
) -> list[str]:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        log("[runtime-capture] playwright 不可用，跳过运行时探测。")
        return []

    collected: list[str] = []
    cookies_file = _export_browser_cookies(browser, profile, log)

    def maybe_add(u: str) -> None:
        low = (u or "").lower()
        if low.startswith(("http://", "https://")) and any(
            x in low for x in (".m3u8", ".mpd", ".mp4", ".webm", ".m4s", ".ts", "manifest", "playlist")
        ):
            collected.append(u)

    try:
        with sync_playwright() as p:
            context = None
            page = None
            profile_arg = profile or "Default"
            if browser == "chrome":
                base = Path.home() / "Library" / "Application Support" / "Google" / "Chrome"
                if (base / profile_arg).exists():
                    try:
                        context = p.chromium.launch_persistent_context(
                            user_data_dir=str(base),
                            channel="chrome",
                            headless=False,
                            args=[f"--profile-directory={profile_arg}"],
                        )
                        page = context.pages[0] if context.pages else context.new_page()
                    except Exception as exc:
                        log(f"[runtime-capture] 复用Chrome会话失败: {exc}")
            if context is None or page is None:
                b = p.chromium.launch(channel="chrome", headless=False)
                context = b.new_context()
                page = context.new_page()
                log("[runtime-capture] 使用临时上下文。")

            if cookies_file and cookies_file.exists():
                injected = _inject_cookies(context, url, cookies_file)
                log(f"[runtime-capture] 已注入cookie: {injected}")

            context.on("request", lambda req: maybe_add(req.url))
            context.on("response", lambda resp: maybe_add(resp.url))
            page.goto(url, wait_until="domcontentloaded")
            _auto_confirm_age_gate(page, log)
            try:
                page.reload(wait_until="domcontentloaded")
            except Exception:
                pass
            _auto_confirm_age_gate(page, log)
            try:
                page.keyboard.press("k")
            except Exception:
                pass
            try:
                page.mouse.click(420, 320)
            except Exception:
                pass
            try:
                page.evaluate(
                    """() => {
                        for (const v of Array.from(document.querySelectorAll('video'))) {
                            try { v.muted = true; const p = v.play(); if (p?.catch) p.catch(() => {}); } catch (_) {}
                        }
                    }"""
                )
            except Exception:
                pass

            log(f"[runtime-capture] 已打开页面，等待 {seconds}s 捕获播放请求...")
            deadline = datetime.now().timestamp() + seconds
            while datetime.now().timestamp() < deadline:
                page.wait_for_timeout(500)
                _auto_confirm_age_gate(page, log, quiet=True)
            context.close()
    except Exception as exc:
        log(f"[runtime-capture] 运行时探测异常: {exc}")

    dedup = list(dict.fromkeys(collected))
    dedup.sort(key=_candidate_score, reverse=True)
    return dedup


def _probe_height(url: str, referer: str, log_lines: list[str]) -> int:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        return 0
    cmd = [ytdlp, "--add-header", f"Referer:{referer}", "--add-header", f"User-Agent:{DEFAULT_UA}", "--list-formats", url]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    log_lines.append(f"[probe-candidate] {url}")
    log_lines.append(f"[probe-candidate-exit] {proc.returncode}")
    log_lines.append(proc.stdout or "")
    log_lines.append(proc.stderr or "")
    best = 0
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"\b(\d{3,4})p\b", line.lower())
        if m:
            best = max(best, int(m.group(1)))
            continue
        m = re.search(r"\b(\d{2,4})x(\d{2,4})\b", line.lower())
        if m:
            best = max(best, int(m.group(2)))
    return best


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text or "")


def _output_template(preferred_title: str | None) -> str:
    return f"{preferred_title}.%(ext)s" if preferred_title else "%(title)s [%(id)s].%(ext)s"


def _rename_with_date_seq(path: Path) -> Path:
    if not path.exists():
        return path
    parent = path.parent
    stem = path.stem
    suffix = path.suffix
    date_tag = datetime.now().strftime("%Y%m%d")
    candidate = parent / f"{stem}_{date_tag}{suffix}"
    if not candidate.exists():
        path.rename(candidate)
        return candidate
    idx = 2
    while True:
        candidate = parent / f"{stem}_{date_tag}_{idx}{suffix}"
        if not candidate.exists():
            path.rename(candidate)
            return candidate
        idx += 1


def _download_with_ytdlp(
    *,
    target_url: str,
    page_url: str,
    output_dir: Path,
    preferred_title: str | None,
    log_lines: list[str],
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> Path | None:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError("未找到 yt-dlp。请先安装：python -m pip install -U yt-dlp")

    output_dir.mkdir(parents=True, exist_ok=True)
    existing_names = {p.name for p in output_dir.glob("*") if p.is_file()}
    cmd = [
        ytdlp,
        "--no-playlist",
        "--newline",
        "--progress",
        "--progress-template",
        "download:[site-progress] %(progress.fragment_index)s/%(progress.fragment_count)s",
        "--add-header",
        f"Referer:{page_url}",
        "--add-header",
        f"User-Agent:{DEFAULT_UA}",
        "--format",
        "bv*+ba/b",
        "--merge-output-format",
        "mp4",
        "--output",
        str(output_dir / _output_template(preferred_title)),
        "--print",
        "after_move:filepath",
        target_url,
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines: list[str] = []
    output_path: Path | None = None
    downloaded = 0
    total: int | None = None
    if progress_callback:
        progress_callback(0, None)

    if proc.stdout is not None:
        for raw in proc.stdout:
            line = (raw or "").rstrip("\n")
            lines.append(line)
            cleaned = _strip_ansi(line).lower()

            m_mark = re.search(r"\[site-progress\]\s*([0-9na]+)/([0-9na]+)", cleaned)
            if m_mark:
                a = m_mark.group(1)
                b = m_mark.group(2)
                new_downloaded = downloaded
                new_total = total
                if a.isdigit():
                    new_downloaded = int(a)
                if b.isdigit():
                    new_total = int(b)
                if new_downloaded != downloaded or new_total != total:
                    downloaded = new_downloaded
                    total = new_total
                    if progress_callback:
                        progress_callback(downloaded, total)

            m_total = re.search(r"total fragments:\s*(\d+)", cleaned) or re.search(
                r"fragments total:\s*(\d+)", cleaned
            )
            if m_total:
                new_total = int(m_total.group(1))
                if new_total != total:
                    total = new_total
                    if progress_callback:
                        progress_callback(downloaded, total)

            m_frag = (
                re.search(r"(?:frag|fragment)\s*[:#]?\s*(\d+)\s*/\s*(\d+)", cleaned)
                or re.search(r"\((?:frag|fragment)\s*(\d+)\s*/\s*(\d+)\)", cleaned)
                or re.search(r"downloaded\s+(\d+)\s+of\s+(\d+)\s+fragments", cleaned)
            )
            if m_frag:
                new_downloaded = int(m_frag.group(1))
                new_total = int(m_frag.group(2))
                if new_downloaded != downloaded or new_total != total:
                    downloaded = new_downloaded
                    total = new_total
                    if progress_callback:
                        progress_callback(downloaded, total)

            p = Path(line.strip()).expanduser()
            if p.exists() and p.is_file():
                output_path = p.resolve()

    code = proc.wait()
    log_lines.append(f"[download-cmd] {' '.join(cmd)}")
    log_lines.append(f"[download-exit] {code}")
    log_lines.append("[download-output]")
    log_lines.extend(lines)
    if code != 0:
        return None

    if output_path and output_path.exists():
        if output_path.name in existing_names:
            output_path = _rename_with_date_seq(output_path)
        return output_path

    newest = None
    newest_mtime = 0.0
    for p in output_dir.glob("*"):
        if p.is_file() and p.stat().st_mtime > newest_mtime:
            newest = p
            newest_mtime = p.stat().st_mtime
    if newest is None:
        return None
    resolved = newest.resolve()
    if resolved.name in existing_names:
        resolved = _rename_with_date_seq(resolved)
    return resolved


def run_site_download(
    *,
    page_url: str,
    output_dir: Path,
    browser: str = "chrome",
    profile: str = "Default",
    capture_seconds: int = 30,
    use_runtime_capture: bool = True,
    log_func: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int | None], None] | None = None,
) -> ProbeResult:
    log_lines: list[str] = []
    log_file = Path.cwd() / "logs" / "sitegrab" / f"sitegrab-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)

    def log(msg: str) -> None:
        log_lines.append(msg)
        if log_func:
            log_func(msg)

    output: Path | None = None
    download_target: str | None = None
    all_candidates: list[str] = []
    try:
        log(f"[url] {page_url}")
        log(f"[browser] {browser}:{profile or '(default)'}")
        html = _fetch_text(page_url, referer=page_url)
        page_title = _extract_page_title(html)
        log(f"[page-title] {page_title or '(none)'}")

        html_candidates = _extract_candidates(html, page_url)
        log(f"[html-candidates] {len(html_candidates)}")
        runtime_candidates: list[str] = []
        if use_runtime_capture:
            runtime_candidates = _capture_runtime_candidates(
                url=page_url,
                browser=browser,
                profile=profile,
                seconds=max(10, int(capture_seconds)),
                log=log,
            )
            log(f"[runtime-candidates] {len(runtime_candidates)}")

        all_candidates = list(dict.fromkeys(html_candidates + runtime_candidates))
        all_candidates.sort(key=_candidate_score, reverse=True)
        log(f"[all-candidates] {len(all_candidates)}")

        best_url = None
        best_score = -1
        for c in all_candidates[:20]:
            score = _candidate_score(c) + _probe_height(c, page_url, log_lines)
            if score > best_score:
                best_score = score
                best_url = c

        download_target = best_url or page_url
        log(f"[selected-url] {download_target}")
        output = _download_with_ytdlp(
            target_url=download_target,
            page_url=page_url,
            output_dir=output_dir,
            preferred_title=page_title,
            log_lines=log_lines,
            progress_callback=progress_callback,
        )
        if output is None and download_target != page_url:
            log("[fallback] 选中切片流失败，回退下载页面URL。")
            output = _download_with_ytdlp(
                target_url=page_url,
                page_url=page_url,
                output_dir=output_dir,
                preferred_title=page_title,
                log_lines=log_lines,
                progress_callback=progress_callback,
            )
    except Exception as exc:
        log(f"[fatal] {exc}")
    finally:
        log_file.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return ProbeResult(
        page_url=page_url,
        final_candidate=download_target,
        candidate_count=len(all_candidates),
        output_file=output,
        log_file=log_file,
        ok=output is not None,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe and download segmented video from webpage.")
    p.add_argument("url", help="Webpage playback URL")
    p.add_argument("--output-dir", type=Path, default=Path.home() / "Downloads")
    p.add_argument("--browser", default="chrome", choices=["chrome", "chromium", "edge", "brave"])
    p.add_argument("--profile", default="Default")
    p.add_argument("--capture-seconds", type=int, default=30)
    p.add_argument("--no-runtime-capture", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = run_site_download(
        page_url=args.url,
        output_dir=args.output_dir.expanduser().resolve(),
        browser=args.browser,
        profile=args.profile,
        capture_seconds=max(10, int(args.capture_seconds)),
        use_runtime_capture=not args.no_runtime_capture,
        log_func=print,
    )
    print(f"[log] {result.log_file}")
    if result.ok and result.output_file:
        print(f"[saved] {result.output_file}")
        return 0
    print("[error] 未下载到视频。请查看日志。")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
