from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_LOG_LINES: list[str] = []
_LOG_PATH: Path | None = None


def _emit(msg: str) -> None:
    print(msg)
    _LOG_LINES.append(msg)


def _write_log_file() -> None:
    global _LOG_PATH
    if _LOG_PATH is None:
        return
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _LOG_PATH.write_text("\n".join(_LOG_LINES) + "\n", encoding="utf-8")


def run(cmd: list[str], check: bool = True) -> None:
    _emit("+ " + " ".join(cmd))
    subprocess.run(cmd, check=check)


def run_with_fallback(
    offline_cmd: list[str],
    online_cmd: list[str],
    mode: str,
    step_name: str,
) -> None:
    if mode == "online":
        run(online_cmd)
        return

    try:
        run(offline_cmd)
    except subprocess.CalledProcessError:
        if mode == "offline":
            raise
        _emit(f"[WARN] Offline {step_name} failed, fallback to online install.")
        run(online_cmd)


def parse_requires_python(req: str) -> tuple[int, int]:
    if not req.startswith(">="):
        return (3, 10)
    ver = req[2:].strip()
    parts = ver.split(".")
    major = int(parts[0]) if parts and parts[0].isdigit() else 3
    minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
    return (major, minor)


def ensure_python_version(required: str) -> None:
    min_major, min_minor = parse_requires_python(required)
    cur = sys.version_info
    if (cur.major, cur.minor) < (min_major, min_minor):
        raise SystemExit(
            f"Python {required} required, current={cur.major}.{cur.minor}.{cur.micro}"
        )
    _emit(f"[OK] Python version: {cur.major}.{cur.minor}.{cur.micro} (required {required})")


def ensure_venv(venv_dir: Path) -> Path:
    python_bin = venv_dir / "bin" / "python"
    if python_bin.exists():
        return python_bin
    run([sys.executable, "-m", "venv", str(venv_dir)])
    return python_bin


def install_python_stack(
    python_bin: Path,
    wheels_dir: Path,
    mode: str,
    deps: list[str],
    project_dir: Path,
) -> None:
    pip_online = [str(python_bin), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"]
    pip_offline = [
        str(python_bin),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        "--upgrade",
        "pip",
        "setuptools",
        "wheel",
    ]
    run_with_fallback(pip_offline, pip_online, mode=mode, step_name="pip bootstrap")

    dep_online = [str(python_bin), "-m", "pip", "install", *deps]
    dep_offline = [
        str(python_bin),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        *deps,
    ]
    run_with_fallback(dep_offline, dep_online, mode=mode, step_name="dependency install")

    proj_online = [str(python_bin), "-m", "pip", "install", "-e", str(project_dir)]
    proj_offline = [
        str(python_bin),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        "-e",
        str(project_dir),
    ]
    run_with_fallback(proj_offline, proj_online, mode=mode, step_name="project install")


def detect_ffmpeg(auto_install: bool) -> bool:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        _emit(f"[OK] ffmpeg found: {ffmpeg_bin}")
        return True
    _emit("[WARN] ffmpeg not found.")
    if not auto_install:
        return False

    install_cmds: list[list[str]] = []
    if shutil.which("brew"):
        install_cmds.append(["brew", "install", "ffmpeg"])
    is_root = hasattr(os, "geteuid") and os.geteuid() == 0
    if shutil.which("apt-get"):
        if is_root:
            install_cmds.append(["apt-get", "update"])
            install_cmds.append(["apt-get", "install", "-y", "ffmpeg"])
        elif shutil.which("sudo"):
            install_cmds.append(["sudo", "apt-get", "update"])
            install_cmds.append(["sudo", "apt-get", "install", "-y", "ffmpeg"])
    if shutil.which("dnf"):
        if is_root:
            install_cmds.append(["dnf", "install", "-y", "ffmpeg"])
        elif shutil.which("sudo"):
            install_cmds.append(["sudo", "dnf", "install", "-y", "ffmpeg"])

    for cmd in install_cmds:
        try:
            run(cmd)
            if shutil.which("ffmpeg"):
                _emit("[OK] ffmpeg installed.")
                return True
        except Exception:
            continue
    _emit("[WARN] ffmpeg auto-install failed; install it manually for m3u8 merge.")
    return False


def ensure_python_module(python_bin: Path, module_name: str, pip_name: str, mode: str, wheels_dir: Path) -> bool:
    check = subprocess.run(
        [str(python_bin), "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"]
    )
    if check.returncode == 0:
        _emit(f"[OK] python module ready: {module_name}")
        return True

    _emit(f"[WARN] missing python module: {module_name}, trying auto-install...")
    online_cmd = [str(python_bin), "-m", "pip", "install", "-U", pip_name]
    offline_cmd = [
        str(python_bin),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        "-U",
        pip_name,
    ]
    try:
        run_with_fallback(offline_cmd, online_cmd, mode=mode, step_name=f"install {pip_name}")
    except Exception as exc:
        _emit(f"[WARN] auto-install failed for {pip_name}: {exc}")
        return False

    check2 = subprocess.run(
        [str(python_bin), "-c", f"import importlib.util,sys;sys.exit(0 if importlib.util.find_spec('{module_name}') else 1)"]
    )
    if check2.returncode == 0:
        _emit(f"[OK] auto-installed module: {module_name}")
        return True
    _emit(f"[WARN] module still missing after install: {module_name}")
    return False


def ensure_runtime_templates(project_dir: Path) -> None:
    env_file = project_dir / ".env.psitedl"
    if not env_file.exists():
        env_file.write_text(
            (
                "# Optional runtime env for PSiteDL launchers\n"
                "export HTTP_PROXY=\"\"\n"
                "export HTTPS_PROXY=\"\"\n"
            ),
            encoding="utf-8",
        )
        _emit(f"[OK] created env template: {env_file}")


def check_runtime_requirements(python_bin: Path, project_dir: Path, mode: str, wheels_dir: Path) -> None:
    _emit("\n[CHECK] runtime requirements")
    ensure_runtime_templates(project_dir)
    ensure_python_module(python_bin, "yt_dlp", "yt-dlp", mode=mode, wheels_dir=wheels_dir)
    ensure_python_module(python_bin, "playwright", "playwright", mode=mode, wheels_dir=wheels_dir)
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        _emit(f"[OK] ffprobe found: {ffprobe}")
    else:
        _emit("[WARN] ffprobe not found. ffmpeg package usually provides it.")


def write_launchers(bundle_dir: Path, venv_dir: Path) -> None:
    cli = bundle_dir / "run_psitedl.sh"
    gui = bundle_dir / "run_psitedl_gui.sh"

    cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'VENV="{venv_dir}"\n'
        f'PROJ="{bundle_dir / "project"}"\n'
        'if [ -f "$PROJ/.env.psitedl" ]; then\n'
        "  # shellcheck disable=SC1090\n"
        '  source "$PROJ/.env.psitedl"\n'
        "fi\n"
        "if [ ! -x \"$VENV/bin/psitedl\" ]; then\n"
        "  echo 'psitedl command not found in venv, please rerun deploy script.'\n"
        "  exit 1\n"
        "fi\n"
        'exec "$VENV/bin/psitedl" "$@"\n',
        encoding="utf-8",
    )
    gui.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'VENV="{venv_dir}"\n'
        f'PROJ="{bundle_dir / "project"}"\n'
        'if [ -f "$PROJ/.env.psitedl" ]; then\n'
        "  # shellcheck disable=SC1090\n"
        '  source "$PROJ/.env.psitedl"\n'
        "fi\n"
        "if [ ! -x \"$VENV/bin/psitedl-gui\" ]; then\n"
        "  echo 'psitedl-gui command not found in venv, please rerun deploy script.'\n"
        "  exit 1\n"
        "fi\n"
        'exec "$VENV/bin/psitedl-gui"\n',
        encoding="utf-8",
    )
    cli.chmod(0o755)
    gui.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a portable PSiteDL bundle on a target machine."
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=Path.cwd(),
        help="Bundle directory path (contains bundle_manifest.json)",
    )
    parser.add_argument(
        "--venv-dir",
        type=Path,
        default=None,
        help="Virtualenv path (default: <bundle-dir>/.venv)",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "offline", "online"],
        default="auto",
        help="Install mode: auto=offline first then online fallback",
    )
    parser.add_argument(
        "--no-auto-ffmpeg",
        action="store_true",
        help="Disable best-effort ffmpeg auto install",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show actions only")
    return parser.parse_args()


def main() -> None:
    global _LOG_PATH
    args = parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    _LOG_PATH = bundle_dir / "deploy-logs" / f"deploy-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    manifest_path = bundle_dir / "bundle_manifest.json"
    wheels_dir = bundle_dir / "wheels"
    project_dir = bundle_dir / "project"

    if not manifest_path.exists():
        _emit(f"[ERROR] manifest not found: {manifest_path}")
        _write_log_file()
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not project_dir.exists():
        _emit(f"[ERROR] project dir not found: {project_dir}")
        _write_log_file()
        raise SystemExit(f"project dir not found: {project_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_python = manifest.get("project", {}).get("requires_python") or ">=3.10"
    ensure_python_version(required_python)

    try:
        detect_ffmpeg(auto_install=not args.no_auto_ffmpeg)

        if args.dry_run:
            _emit("[DRY-RUN] Skip venv/dependency installation.")
            return

        venv_dir = (
            args.venv_dir.expanduser().resolve()
            if args.venv_dir
            else (bundle_dir / ".venv").resolve()
        )
        python_bin = ensure_venv(venv_dir)

        deps = [str(d) for d in manifest.get("dependencies", [])]
        if not deps:
            _emit("[WARN] empty dependency list in manifest")

        install_mode = args.mode
        if install_mode != "online" and not wheels_dir.exists():
            if install_mode == "offline":
                raise SystemExit(f"offline mode selected but wheels dir missing: {wheels_dir}")
            _emit("[WARN] wheels dir missing, switch to online mode")
            install_mode = "online"

        install_python_stack(
            python_bin=python_bin,
            wheels_dir=wheels_dir,
            mode=install_mode,
            deps=deps,
            project_dir=project_dir,
        )

        check_runtime_requirements(
            python_bin=python_bin,
            project_dir=project_dir,
            mode=install_mode,
            wheels_dir=wheels_dir,
        )
        write_launchers(bundle_dir, venv_dir)

        _emit("\nDeploy completed.")
        _emit(f"- venv: {venv_dir}")
        _emit(f"- CLI launcher: {bundle_dir / 'run_psitedl.sh'}")
        _emit(f"- GUI launcher: {bundle_dir / 'run_psitedl_gui.sh'}")
    finally:
        _write_log_file()
        if _LOG_PATH is not None:
            print(f"[LOG] {_LOG_PATH}")


if __name__ == "__main__":
    main()

