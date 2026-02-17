from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from datetime import datetime
from urllib import error as urlerror
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import parse as urlparse
from urllib import request as urlrequest

from webvidgrab.settings import (
    DEFAULT_TRANSLATE_CONFIG_PATH,
    resolve_translation_settings,
)

VIDEO_EXTENSIONS = {
    ".mp4",
    ".m4v",
    ".mov",
    ".webm",
    ".mkv",
    ".avi",
    ".flv",
    ".wmv",
    ".ts",
    ".mpg",
    ".mpeg",
}

DEFAULT_CONFIG_PATH = DEFAULT_TRANSLATE_CONFIG_PATH
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_CHOICES = ["chrome", "chromium", "firefox", "safari", "edge", "brave"]
EJS_SOURCE_CHOICES = ["none", "github", "npm"]


@dataclass
class DownloadItem:
    url: str
    kind: str
    filename: str
    priority: int
    headers: dict[str, str]
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a webpage with LLM and download discovered video resources."
    )
    parser.add_argument("url", help="Web page URL to analyze")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory for downloaded video files",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "openai", "deepseek"],
        default="local",
        help="LLM backend",
    )
    parser.add_argument("--model", default=None, help="LLM model name")
    parser.add_argument("--base-url", default=None, help="LLM API base URL override")
    parser.add_argument("--api-key", default=None, help="LLM API key override")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Provider config path for online backends",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=120,
        help="Max extracted candidate URLs before LLM analysis",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print strategy without downloading",
    )
    parser.add_argument("--login-url", default=None, help="Login endpoint URL")
    parser.add_argument("--username", default=None, help="Login username")
    parser.add_argument("--password", default=None, help="Login password")
    parser.add_argument(
        "--username-field",
        default="username",
        help="Login form field for username",
    )
    parser.add_argument(
        "--password-field",
        default="password",
        help="Login form field for password",
    )
    parser.add_argument(
        "--login-extra",
        action="append",
        default=[],
        help="Extra login form field, format key=value; repeatable",
    )
    parser.add_argument(
        "--cookies-file",
        type=Path,
        default=None,
        help="Netscape cookies.txt file path",
    )
    parser.add_argument(
        "--cookies-from-browser",
        choices=BROWSER_CHOICES,
        default=None,
        help="Use browser session cookies via yt-dlp",
    )
    parser.add_argument(
        "--cookies-profile",
        default=None,
        help="Browser profile name/path used with --cookies-from-browser",
    )
    parser.add_argument(
        "--yt-ejs-source",
        choices=EJS_SOURCE_CHOICES,
        default="github",
        help="yt-dlp EJS remote components source for YouTube",
    )
    parser.add_argument(
        "--yt-list-formats-on-fail",
        action="store_true",
        help="List available YouTube formats automatically when yt-dlp fails",
    )
    parser.add_argument(
        "--yt-extractor-args",
        default="youtube:player_client=web,web_safari",
        help="yt-dlp extractor args for YouTube",
    )
    parser.add_argument(
        "--yt-log-file",
        type=Path,
        default=None,
        help="Path to save full yt-dlp stdout/stderr log",
    )
    parser.add_argument(
        "--yt-extra-args",
        default=None,
        help="Raw extra yt-dlp arguments string, e.g. \"--format bv*+ba/b --merge-output-format mp4\"",
    )
    parser.add_argument(
        "--yt-po-token",
        default=None,
        help="YouTube PO token (optional, advanced)",
    )
    parser.add_argument(
        "--yt-visitor-data",
        default=None,
        help="YouTube visitor_data (optional, advanced)",
    )
    parser.add_argument(
        "--yt-har-file",
        type=Path,
        default=None,
        help="HAR file path to auto-extract YouTube PO token and visitor_data",
    )
    return parser.parse_args()


def build_session(
    cookies_file: Path | None = None,
) -> tuple[urlrequest.OpenerDirector, http.cookiejar.CookieJar]:
    jar = http.cookiejar.CookieJar()
    if cookies_file:
        load_netscape_cookies(jar, cookies_file)
    opener = urlrequest.build_opener(urlrequest.HTTPCookieProcessor(jar))
    opener.addheaders = [
        ("User-Agent", DEFAULT_USER_AGENT),
        ("Accept", "*/*"),
        ("Accept-Language", "zh-CN,zh;q=0.9,en;q=0.8"),
    ]
    return opener, jar


def load_netscape_cookies(jar: http.cookiejar.CookieJar, cookies_file: Path) -> None:
    path = cookies_file.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"cookies file not found: {path}")
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 7:
            continue
        domain, flag, cookie_path, secure_raw, expires_raw, name, value = parts
        secure = secure_raw.upper() == "TRUE"
        domain_specified = domain.startswith(".")
        expires = None
        try:
            expires = int(expires_raw)
        except Exception:
            expires = None
        ck = http.cookiejar.Cookie(
            version=0,
            name=name,
            value=value,
            port=None,
            port_specified=False,
            domain=domain.lstrip("."),
            domain_specified=domain_specified,
            domain_initial_dot=domain.startswith("."),
            path=cookie_path or "/",
            path_specified=True,
            secure=secure,
            expires=expires,
            discard=expires is None,
            comment=None,
            comment_url=None,
            rest={},
            rfc2109=False,
        )
        jar.set_cookie(ck)


def parse_key_values(raw_items: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in raw_items:
        item = (raw or "").strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid key=value: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid key=value: {item}")
        result[key] = value
    return result


def login_with_form(
    opener: urlrequest.OpenerDirector,
    *,
    login_url: str,
    username: str,
    password: str,
    username_field: str,
    password_field: str,
    extra_fields: dict[str, str] | None,
) -> None:
    payload = {
        username_field: username,
        password_field: password,
    }
    if extra_fields:
        payload.update(extra_fields)
    body = urlparse.urlencode(payload).encode("utf-8")
    req = urlrequest.Request(
        login_url,
        data=body,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": login_url,
        },
    )
    with opener.open(req, timeout=30):
        return


def fetch_webpage(
    url: str,
    *,
    opener: urlrequest.OpenerDirector | None = None,
    referer: str | None = None,
) -> tuple[str, str]:
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    if referer:
        headers["Referer"] = referer
    req = urlrequest.Request(url, headers=headers)
    opener_to_use = opener or urlrequest.build_opener()
    with opener_to_use.open(req, timeout=25) as resp:
        final_url = str(resp.geturl())
        content = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return final_url, content.decode(charset, errors="ignore")


def _normalize_candidate(raw: str, base_url: str) -> str:
    cleaned = raw.strip().strip("\"'").replace("\\/", "/")
    if not cleaned:
        return ""
    if cleaned.startswith("//"):
        cleaned = "https:" + cleaned
    joined = urlparse.urljoin(base_url, cleaned)
    parsed = urlparse.urlparse(joined)
    if parsed.scheme not in {"http", "https"}:
        return ""
    return joined


def extract_candidates(html: str, base_url: str, limit: int) -> list[str]:
    patterns = [
        r"""(?:src|href|data-src|data-url|content)\s*=\s*["']([^"'<>]+)["']""",
        r"""["'](https?://[^"'<> ]+)["']""",
        r"""["']([^"'<> ]+\.(?:m3u8|mp4|webm|m4v|mov|mkv)(?:\?[^"'<> ]*)?)["']""",
        r"""(https?:\\/\\/[^"'<> ]+)""",
    ]
    found: list[str] = []
    for pattern in patterns:
        for item in re.findall(pattern, html, flags=re.IGNORECASE):
            normalized = _normalize_candidate(item, base_url)
            if normalized:
                found.append(normalized)
    deduped = list(dict.fromkeys(found))
    scored = sorted(deduped, key=_candidate_score, reverse=True)
    return scored[: max(10, limit)]


def _candidate_score(url: str) -> int:
    score = 0
    low = url.lower()
    if ".m3u8" in low:
        score += 120
    for ext in VIDEO_EXTENSIONS:
        if ext in low:
            score += 100
            break
    if any(k in low for k in ("video", "stream", "playlist", "play", "media")):
        score += 20
    if "blob:" in low or "data:" in low:
        score -= 80
    return score


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("LLM response does not contain a JSON object.")
    return text[start : end + 1]


def build_strategy(
    *,
    llm_settings: dict[str, str],
    page_url: str,
    final_url: str,
    html: str,
    candidates: list[str],
) -> tuple[str, list[DownloadItem]]:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "openai package is required for LLM analysis. Install dependencies first."
        ) from exc

    snippet = html[:15000]
    prompt = (
        "你是视频下载策略规划器。你会从网页源码和候选链接中识别可下载的视频资源。"
        "必须只返回 JSON。格式如下："
        '{'
        '"reasoning":"...",'
        '"downloads":[{"url":"...","kind":"direct|m3u8","filename":"...","priority":1,"headers":{},"note":"..."}]'
        "}"
        "规则：downloads 里只保留最有价值的候选，priority 越小越优先，最多 5 条。"
    )
    payload = {
        "page_url": page_url,
        "final_url": final_url,
        "candidate_urls": candidates[:80],
        "html_snippet": snippet,
    }
    client = OpenAI(api_key=llm_settings["api_key"], base_url=llm_settings["base_url"])
    response = client.chat.completions.create(
        model=llm_settings["model"],
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    text = response.choices[0].message.content or "{}"
    parsed = json.loads(_extract_json_block(text))
    reasoning = str(parsed.get("reasoning") or "").strip()
    downloads_raw = parsed.get("downloads")
    downloads: list[DownloadItem] = []
    if isinstance(downloads_raw, list):
        for idx, item in enumerate(downloads_raw):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            kind = str(item.get("kind") or "").strip().lower()
            if kind not in {"direct", "m3u8"}:
                kind = "m3u8" if ".m3u8" in url.lower() else "direct"
            filename = str(item.get("filename") or "").strip()
            filename = filename or infer_filename(url, kind, idx)
            priority = int(item.get("priority") or (idx + 1))
            headers = item.get("headers") if isinstance(item.get("headers"), dict) else {}
            normalized_headers = {
                str(k): str(v) for k, v in headers.items() if str(k).strip() and str(v).strip()
            }
            note = str(item.get("note") or "").strip()
            downloads.append(
                DownloadItem(
                    url=url,
                    kind=kind,
                    filename=sanitize_filename(filename),
                    priority=priority,
                    headers=normalized_headers,
                    note=note,
                )
            )
    downloads.sort(key=lambda x: x.priority)
    if downloads:
        return reasoning, downloads
    return reasoning, build_fallback_strategy(candidates)


def build_fallback_strategy(candidates: list[str]) -> list[DownloadItem]:
    items: list[DownloadItem] = []
    for idx, url in enumerate(candidates[:30]):
        lower = url.lower()
        if ".m3u8" not in lower and not any(ext in lower for ext in VIDEO_EXTENSIONS):
            continue
        kind = "m3u8" if ".m3u8" in lower else "direct"
        items.append(
            DownloadItem(
                url=url,
                kind=kind,
                filename=infer_filename(url, kind, idx),
                priority=idx + 1,
                headers={},
                note="fallback strategy",
            )
        )
        if len(items) >= 5:
            break
    return items


def sanitize_filename(name: str) -> str:
    clean = re.sub(r"[^\w.\-]+", "_", name.strip(), flags=re.ASCII).strip("._")
    return clean or "video.mp4"


def infer_filename(url: str, kind: str, index: int) -> str:
    path = urlparse.urlparse(url).path
    raw = Path(path).name or f"video_{index + 1}"
    raw = sanitize_filename(raw)
    if "." not in raw:
        raw += ".mp4" if kind == "direct" else ".mp4"
    if raw.lower().endswith(".m3u8"):
        raw = raw[:-5] + ".mp4"
    return raw


def cookie_header_for_url(jar: http.cookiejar.CookieJar, url: str) -> str:
    parsed = urlparse.urlparse(url)
    host = parsed.hostname or ""
    path = parsed.path or "/"
    secure = parsed.scheme == "https"
    pairs: list[str] = []
    for cookie in jar:
        if cookie.secure and not secure:
            continue
        domain = (cookie.domain or "").lstrip(".")
        if domain and host and not host.endswith(domain):
            continue
        c_path = cookie.path or "/"
        if not path.startswith(c_path):
            continue
        pairs.append(f"{cookie.name}={cookie.value}")
    return "; ".join(pairs)


def download_direct(
    url: str,
    out_path: Path,
    headers: dict[str, str],
    *,
    opener: urlrequest.OpenerDirector | None = None,
    referer: str | None = None,
) -> None:
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if referer:
        req_headers["Referer"] = referer
    req_headers.update(headers)
    req = urlrequest.Request(url, headers=req_headers)
    opener_to_use = opener or urlrequest.build_opener()
    with opener_to_use.open(req, timeout=45) as resp:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as f:
            while True:
                chunk = resp.read(1024 * 512)
                if not chunk:
                    break
                f.write(chunk)


def download_m3u8(
    url: str,
    out_path: Path,
    headers: dict[str, str],
    *,
    cookie_jar: http.cookiejar.CookieJar | None = None,
    referer: str | None = None,
) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        raise RuntimeError("ffmpeg not found. Install ffmpeg to download m3u8 streams.")
    cmd = [ffmpeg_bin, "-y"]
    req_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if referer:
        req_headers["Referer"] = referer
    req_headers.update(headers)
    if cookie_jar:
        cookie_header = cookie_header_for_url(cookie_jar, url)
        if cookie_header:
            req_headers["Cookie"] = cookie_header

    if req_headers:
        header_lines = "".join(f"{k}: {v}\r\n" for k, v in req_headers.items())
        cmd.extend(["-headers", header_lines])
    cmd.extend(["-i", url, "-c", "copy", str(out_path)])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip()
        raise RuntimeError(f"ffmpeg failed: {err[-500:]}")


def execute_downloads(
    items: list[DownloadItem],
    output_dir: Path,
    *,
    opener: urlrequest.OpenerDirector | None = None,
    cookie_jar: http.cookiejar.CookieJar | None = None,
    referer: str | None = None,
) -> list[Path]:
    saved: list[Path] = []
    for item in items:
        out_path = output_dir / item.filename
        print(f"[download] {item.kind} -> {out_path.name}")
        if item.kind == "m3u8":
            download_m3u8(
                item.url,
                out_path,
                item.headers,
                cookie_jar=cookie_jar,
                referer=referer,
            )
        else:
            download_direct(
                item.url,
                out_path,
                item.headers,
                opener=opener,
                referer=referer,
            )
        if out_path.exists() and out_path.stat().st_size > 0:
            saved.append(out_path)
    return saved


def is_youtube_url(url: str) -> bool:
    host = (urlparse.urlparse(url).hostname or "").lower()
    return "youtube.com" in host or "youtu.be" in host


def download_with_yt_dlp(
    page_url: str,
    output_dir: Path,
    *,
    cookies_from_browser: str | None = None,
    cookies_profile: str | None = None,
    cookies_file: Path | None = None,
    yt_ejs_source: str = "github",
    yt_list_formats_on_fail: bool = False,
    yt_extractor_args: str | None = None,
    yt_log_file: Path | None = None,
    yt_extra_args: str | None = None,
    yt_po_token: str | None = None,
    yt_visitor_data: str | None = None,
) -> tuple[list[Path], Path]:
    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError(
            "yt-dlp not found. Install it first, e.g. `python -m pip install -U yt-dlp`."
        )

    run_env = os.environ.copy()
    path_items = [
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    merged_path = run_env.get("PATH", "")
    for p in path_items:
        if p not in merged_path.split(":"):
            merged_path = f"{p}:{merged_path}" if merged_path else p
    run_env["PATH"] = merged_path

    resolved_log_file = (
        yt_log_file.expanduser().resolve()
        if yt_log_file
        else (Path.cwd() / "logs" / "webvidgrab" / f"yt-dlp-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log")
    )
    resolved_log_file.parent.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    log_lines.append(f"[time] {datetime.now().isoformat(timespec='seconds')}")
    log_lines.append(f"[url] {page_url}")
    node_bin = shutil.which("node", path=run_env.get("PATH"))
    deno_bin = shutil.which("deno", path=run_env.get("PATH"))
    log_lines.append(f"[runtime-node] {node_bin or 'not-found'}")
    log_lines.append(f"[runtime-deno] {deno_bin or 'not-found'}")
    log_lines.append(f"[input-po-token] {'yes' if (yt_po_token and yt_po_token.strip()) else 'no'}")
    log_lines.append(
        f"[input-visitor-data] {'yes' if (yt_visitor_data and yt_visitor_data.strip()) else 'no'}"
    )
    if node_bin:
        node_ver = subprocess.run([node_bin, "--version"], capture_output=True, text=True, env=run_env)
        log_lines.append(f"[runtime-node-version] {(node_ver.stdout or node_ver.stderr or '').strip()}")
    if deno_bin:
        deno_ver = subprocess.run([deno_bin, "--version"], capture_output=True, text=True, env=run_env)
        log_lines.append(f"[runtime-deno-version] {(deno_ver.stdout or deno_ver.stderr or '').strip()}")

    extra_tokens: list[str] = []
    if yt_extra_args and yt_extra_args.strip():
        try:
            extra_tokens = shlex.split(yt_extra_args.strip())
        except ValueError as exc:
            raise RuntimeError(f"Invalid --yt-extra-args: {exc}") from exc

    def merged_extractor_args(base_args: str | None) -> str | None:
        items: list[str] = []
        if base_args and base_args.strip():
            items.append(base_args.strip())
        if yt_po_token and yt_po_token.strip():
            token = yt_po_token.strip()
            items.append(
                "youtube:po_token="
                f"web.gvs+{token},web.player+{token},mweb.gvs+{token}"
            )
        if yt_visitor_data and yt_visitor_data.strip():
            items.append(f"youtube:visitor_data={yt_visitor_data.strip()}")
        if not items:
            return None
        return ";".join(items)

    def build_cmd(
        *,
        ejs_source: str | None,
        extractor_args: str | None,
        list_formats: bool = False,
    ) -> list[str]:
        cmd = [ytdlp]
        if not list_formats:
            cmd.extend(
                [
                    "--no-playlist",
                    "-o",
                    str(output_dir / "%(title)s [%(id)s].%(ext)s"),
                    "--print",
                    "after_move:filepath",
                ]
            )
        if cookies_from_browser:
            if cookies_profile:
                cmd.extend(["--cookies-from-browser", f"{cookies_from_browser}:{cookies_profile}"])
            else:
                cmd.extend(["--cookies-from-browser", cookies_from_browser])
        if cookies_file:
            cmd.extend(["--cookies", str(cookies_file.expanduser().resolve())])
        if ejs_source in {"github", "npm"}:
            cmd.extend(["--remote-components", f"ejs:{ejs_source}"])
        merged_args = merged_extractor_args(extractor_args)
        if merged_args:
            cmd.extend(["--extractor-args", merged_args])
        if extra_tokens:
            cmd.extend(extra_tokens)
        if list_formats:
            cmd.extend(["--list-formats", page_url])
        else:
            cmd.append(page_url)
        return cmd

    preferred_args = (yt_extractor_args or "").strip()
    extractor_candidates = [
        preferred_args,
        "youtube:player_client=web_safari,web",
        "youtube:player_client=mweb,web_safari,web",
        "youtube:player_client=tv,web_safari,web",
    ]
    extractor_candidates = [x for i, x in enumerate(extractor_candidates) if x and x not in extractor_candidates[:i]]

    source_candidates = [yt_ejs_source]
    if yt_ejs_source == "github":
        source_candidates.append("npm")
    elif yt_ejs_source == "npm":
        source_candidates.append("github")
    source_candidates.append("none")
    source_candidates = [x for i, x in enumerate(source_candidates) if x and x not in source_candidates[:i]]

    attempt_matrix: list[tuple[str, str]] = []
    attempt_matrix.append((source_candidates[0], extractor_candidates[0]))
    for ext in extractor_candidates[1:]:
        attempt_matrix.append((source_candidates[0], ext))
    for src in source_candidates[1:]:
        attempt_matrix.append((src, extractor_candidates[0]))
    attempt_matrix = attempt_matrix[:7]

    last_err = ""
    last_source = source_candidates[0]
    last_args = extractor_candidates[0]

    for i, (src, ext_args) in enumerate(attempt_matrix, start=1):
        cmd = build_cmd(ejs_source=src, extractor_args=ext_args, list_formats=False)
        log_lines.append(f"[attempt-{i}] ejs={src} extractor_args={ext_args}")
        log_lines.append(f"[attempt-{i}-cmd] {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
        log_lines.append(f"[attempt-{i}-exit] {proc.returncode}")
        log_lines.append(f"[attempt-{i}-stdout]")
        log_lines.append(proc.stdout or "")
        log_lines.append(f"[attempt-{i}-stderr]")
        log_lines.append(proc.stderr or "")
        last_err = (proc.stderr or "").strip() or (proc.stdout or "").strip()
        last_source = src
        last_args = ext_args

        if proc.returncode == 0:
            saved: list[Path] = []
            for line in (proc.stdout or "").splitlines():
                p = Path(line.strip()).expanduser()
                if p.exists() and p.is_file():
                    saved.append(p.resolve())
            if not saved:
                for child in sorted(output_dir.glob("*")):
                    if child.is_file():
                        saved.append(child.resolve())
            resolved_log_file.write_text("\n".join(log_lines), encoding="utf-8")
            return saved, resolved_log_file

        if "n challenge solving failed" not in (proc.stderr or ""):
            break

    extra = ""
    if yt_list_formats_on_fail:
        list_cmd = build_cmd(ejs_source=last_source, extractor_args=last_args, list_formats=True)
        log_lines.append(f"[list-formats-cmd] {' '.join(list_cmd)}")
        list_proc = subprocess.run(list_cmd, capture_output=True, text=True, env=run_env)
        log_lines.append(f"[list-formats-exit] {list_proc.returncode}")
        log_lines.append("[list-formats-stdout]")
        log_lines.append(list_proc.stdout or "")
        log_lines.append("[list-formats-stderr]")
        log_lines.append(list_proc.stderr or "")
        extra = (list_proc.stdout or list_proc.stderr or "").strip()
        if extra:
            extra = "\n\n[yt-dlp --list-formats]\n" + extra[-1200:]

    resolved_log_file.write_text("\n".join(log_lines), encoding="utf-8")
    raise RuntimeError(
        f"yt-dlp failed after {len(attempt_matrix)} attempts: {last_err[-800:]}{extra}\nFull log saved: {resolved_log_file}"
    )


def extract_youtube_tokens_from_har(har_file: Path) -> tuple[str | None, str | None]:
    path = har_file.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"HAR file not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception as exc:
        raise RuntimeError(f"Failed to parse HAR: {exc}") from exc

    log = raw.get("log")
    entries = log.get("entries") if isinstance(log, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("Invalid HAR structure: missing log.entries")

    po_token: str | None = None
    visitor_data: str | None = None

    def scan_text_blob(text: str) -> None:
        nonlocal po_token, visitor_data
        if not text:
            return
        text = text.replace("\\u003d", "=").replace("\\u0026", "&")
        if po_token is None:
            for pattern in [
                r'"poToken"\s*:\s*"([^"]+)"',
                r'"po_token"\s*:\s*"([^"]+)"',
                r'"gvsToken"\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pattern, text)
                if m:
                    po_token = m.group(1).strip()
                    break
        if visitor_data is None:
            for pattern in [
                r'"visitorData"\s*:\s*"([^"]+)"',
                r'"VISITOR_DATA"\s*:\s*"([^"]+)"',
                r'"X-Goog-Visitor-Id"\s*:\s*"([^"]+)"',
            ]:
                m = re.search(pattern, text)
                if m:
                    visitor_data = m.group(1).strip()
                    break

    def scan_json_obj(obj: Any) -> None:
        nonlocal po_token, visitor_data
        if isinstance(obj, dict):
            for k, v in obj.items():
                kl = str(k).lower()
                if po_token is None and kl.endswith("potoken") and isinstance(v, str) and v.strip():
                    po_token = v.strip()
                if (
                    visitor_data is None
                    and ("visitordata" in kl or kl == "x-goog-visitor-id")
                    and isinstance(v, str)
                    and v.strip()
                ):
                    visitor_data = v.strip()
                scan_json_obj(v)
        elif isinstance(obj, list):
            for x in obj:
                scan_json_obj(x)

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        request = entry.get("request")
        if not isinstance(request, dict):
            continue
        url = str(request.get("url") or "")
        if "youtube" not in url and "googlevideo" not in url and "youtubei" not in url:
            continue

        headers = request.get("headers")
        if isinstance(headers, list):
            for item in headers:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "").strip().lower()
                value = str(item.get("value") or "").strip()
                if not value:
                    continue
                if visitor_data is None and name == "x-goog-visitor-id":
                    visitor_data = value

        post_data = request.get("postData")
        if isinstance(post_data, dict):
            text = post_data.get("text")
            if isinstance(text, str) and text.strip():
                try:
                    payload = json.loads(text)
                    scan_json_obj(payload)
                except Exception:
                    scan_text_blob(text)

        response = entry.get("response")
        if isinstance(response, dict):
            content = response.get("content")
            if isinstance(content, dict):
                text = content.get("text")
                if isinstance(text, str) and text.strip():
                    scan_text_blob(text)
                    if po_token is None or visitor_data is None:
                        try:
                            scan_json_obj(json.loads(text))
                        except Exception:
                            pass
        if po_token and visitor_data:
            break

    return po_token, visitor_data


def extract_youtube_tokens_from_text(text: str) -> tuple[str | None, str | None]:
    po_token: str | None = None
    visitor_data: str | None = None
    if not text:
        return None, None
    text = text.replace("\\u003d", "=").replace("\\u0026", "&")
    for pattern in [
        r'"poToken"\s*:\s*"([^"]+)"',
        r'"po_token"\s*:\s*"([^"]+)"',
        r'"gvsToken"\s*:\s*"([^"]+)"',
    ]:
        m = re.search(pattern, text)
        if m:
            po_token = m.group(1).strip()
            break
    for pattern in [
        r'"visitorData"\s*:\s*"([^"]+)"',
        r'"VISITOR_DATA"\s*:\s*"([^"]+)"',
        r'"X-Goog-Visitor-Id"\s*:\s*"([^"]+)"',
    ]:
        m = re.search(pattern, text)
        if m:
            visitor_data = m.group(1).strip()
            break
    return po_token, visitor_data


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        opener, cookie_jar = build_session(args.cookies_file)
        login_extra = parse_key_values(args.login_extra)
        if args.login_url or args.username or args.password:
            if not (args.login_url and args.username and args.password):
                raise RuntimeError(
                    "Login requires --login-url, --username and --password together."
                )
            print(f"[0/4] Logging in via: {args.login_url}")
            login_with_form(
                opener,
                login_url=args.login_url,
                username=args.username,
                password=args.password,
                username_field=args.username_field,
                password_field=args.password_field,
                extra_fields=login_extra,
            )

        if is_youtube_url(args.url) and not args.dry_run:
            yt_po_token = args.yt_po_token
            yt_visitor_data = args.yt_visitor_data
            if args.yt_har_file and (not yt_po_token or not yt_visitor_data):
                har_po_token, har_visitor = extract_youtube_tokens_from_har(args.yt_har_file)
                if not yt_po_token and har_po_token:
                    yt_po_token = har_po_token
                    print("[youtube] Loaded PO token from HAR.")
                if not yt_visitor_data and har_visitor:
                    yt_visitor_data = har_visitor
                    print("[youtube] Loaded visitor_data from HAR.")
                if not yt_po_token:
                    print("[youtube] HAR did not provide PO token.")
                if not yt_visitor_data:
                    print("[youtube] HAR did not provide visitor_data.")
            if not yt_po_token or not yt_visitor_data:
                try:
                    _, yt_html = fetch_webpage(args.url, opener=opener, referer=args.login_url)
                    html_po, html_visitor = extract_youtube_tokens_from_text(yt_html)
                    if not yt_po_token and html_po:
                        yt_po_token = html_po
                        print("[youtube] Loaded PO token from page source.")
                    if not yt_visitor_data and html_visitor:
                        yt_visitor_data = html_visitor
                        print("[youtube] Loaded visitor_data from page source.")
                except Exception:
                    pass

            print("[youtube] Detected YouTube URL, using yt-dlp downloader with browser session.")
            saved_files, yt_log_path = download_with_yt_dlp(
                args.url,
                output_dir,
                cookies_from_browser=args.cookies_from_browser,
                cookies_profile=args.cookies_profile,
                cookies_file=args.cookies_file,
                yt_ejs_source=args.yt_ejs_source,
                yt_list_formats_on_fail=args.yt_list_formats_on_fail,
                yt_extractor_args=args.yt_extractor_args,
                yt_log_file=args.yt_log_file,
                yt_extra_args=args.yt_extra_args,
                yt_po_token=yt_po_token,
                yt_visitor_data=yt_visitor_data,
            )
            print(f"[youtube] yt-dlp full log: {yt_log_path}")
            if not saved_files:
                print("No file saved.")
                return 4
            print("\nSaved files:")
            for p in saved_files:
                print(f"- {p}")
            return 0

        llm_settings = resolve_translation_settings(
            backend=args.backend,
            model_name=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            config_path=args.config,
        )
        print(f"[1/4] Fetching page: {args.url}")
        final_url, html = fetch_webpage(args.url, opener=opener, referer=args.login_url)
        print(f"[2/4] Extracting candidates from page source")
        candidates = extract_candidates(html, final_url, args.max_candidates)
        if not candidates:
            print("No candidates discovered in html.")
            return 2

        print(f"[3/4] Asking LLM to build download strategy")
        reasoning, downloads = build_strategy(
            llm_settings=llm_settings,
            page_url=args.url,
            final_url=final_url,
            html=html,
            candidates=candidates,
        )
        print(f"\nStrategy reasoning: {reasoning or 'N/A'}")
        print("Planned downloads:")
        for i, item in enumerate(downloads, start=1):
            print(f"{i}. [{item.kind}] {item.url} -> {item.filename}")

        if args.dry_run:
            return 0
        if not downloads:
            print("No downloadable targets from strategy.")
            return 3

        print(f"\n[4/4] Downloading files to: {output_dir}")
        saved = execute_downloads(
            downloads,
            output_dir,
            opener=opener,
            cookie_jar=cookie_jar,
            referer=final_url,
        )
        if not saved:
            print("No file saved.")
            return 4
        print("\nSaved files:")
        for p in saved:
            print(f"- {p}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except urlerror.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}", file=sys.stderr)
        if exc.code in {401, 403}:
            print(
                "可能原因：目标站点需要登录态/会话 Cookie，或反爬校验触发。",
                file=sys.stderr,
            )
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
