from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], check: bool = True) -> None:
    print("+", " ".join(cmd))
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
        print(f"[WARN] Offline {step_name} failed, fallback to online install.")
        run(online_cmd)


def parse_requires_python(req: str) -> tuple[int, int]:
    # Supports ">=3.10" format.
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


def restore_model_cache(bundle_dir: Path, manifest: dict, dry_run: bool = False) -> None:
    asr = manifest.get("asr", {})
    model_cache = asr.get("model_cache")
    if not model_cache:
        print("[INFO] No model cache in bundle manifest, skip restoring model cache.")
        return

    cache_kind = model_cache.get("cache_kind")
    rel = model_cache.get("cache_rel_path")
    if not rel:
        print("[WARN] Invalid model cache metadata, skip model restore.")
        return

    src = bundle_dir / "model-cache" / rel
    if not src.exists():
        print(f"[WARN] model-cache path missing: {src}")
        return

    if cache_kind == "huggingface_repo":
        target = Path.home() / ".cache" / "huggingface" / "hub" / src.name
        if dry_run:
            print(f"[DRY-RUN] copytree {src} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target, dirs_exist_ok=True)
        print(f"[OK] restored faster-whisper cache: {target}")
        return

    if cache_kind == "whisper_pt":
        target = Path.home() / ".cache" / "whisper" / src.name
        if dry_run:
            print(f"[DRY-RUN] copy2 {src} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        print(f"[OK] restored openai-whisper cache: {target}")
        return

    if cache_kind == "local_path":
        target = Path.home() / ".cache" / "subgen" / "models" / src.name
        if dry_run:
            print(f"[DRY-RUN] copytree {src} -> {target}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src, target, dirs_exist_ok=True)
        print(f"[OK] restored local ASR model path: {target}")
        return

    print(f"[WARN] unsupported cache kind: {cache_kind}")


def detect_ffmpeg(auto_install: bool) -> None:
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        print(f"[OK] ffmpeg found: {ffmpeg_bin}")
        return

    print("[WARN] ffmpeg not found.")
    if not auto_install:
        return

    # Best-effort auto install; continue even if failed.
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
                print("[OK] ffmpeg installed.")
                return
        except Exception:
            continue

    print("[WARN] ffmpeg auto-install failed; install it manually if ASR decode fails.")


def write_launchers(bundle_dir: Path, venv_dir: Path, manifest: dict) -> None:
    asr = manifest.get("asr", {})
    engine = str(asr.get("engine") or "faster-whisper")
    model = str(asr.get("model") or "medium")

    cli = bundle_dir / "run_subgen.sh"
    gui = bundle_dir / "run_subgen_gui.sh"

    cli.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'VENV="{venv_dir}"\n'
        "if [ ! -x \"$VENV/bin/subgen\" ]; then\n"
        "  echo 'subgen command not found in venv, please rerun deploy script.'\n"
        "  exit 1\n"
        "fi\n"
        f'exec "$VENV/bin/subgen" --asr-engine {engine} --whisper-model "{model}" "$@"\n',
        encoding="utf-8",
    )
    gui.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f'VENV="{venv_dir}"\n'
        "if [ ! -x \"$VENV/bin/subgen-gui\" ]; then\n"
        "  echo 'subgen-gui command not found in venv, please rerun deploy script.'\n"
        "  exit 1\n"
        "fi\n"
        "exec \"$VENV/bin/subgen-gui\"\n",
        encoding="utf-8",
    )
    cli.chmod(0o755)
    gui.chmod(0o755)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a portable SubGen bundle on a target machine."
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
        "--skip-model-restore",
        action="store_true",
        help="Skip restoring bundled ASR model cache",
    )
    parser.add_argument(
        "--no-auto-ffmpeg",
        action="store_true",
        help="Disable best-effort ffmpeg auto install",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show actions only")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bundle_dir = args.bundle_dir.expanduser().resolve()
    manifest_path = bundle_dir / "bundle_manifest.json"
    wheels_dir = bundle_dir / "wheels"
    project_dir = bundle_dir / "project"

    if not manifest_path.exists():
        raise SystemExit(f"manifest not found: {manifest_path}")
    if not project_dir.exists():
        raise SystemExit(f"project dir not found: {project_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_python = (
        manifest.get("project", {}).get("requires_python")
        or ">=3.10"
    )
    ensure_python_version(required_python)

    detect_ffmpeg(auto_install=not args.no_auto_ffmpeg)

    if args.dry_run:
        print("[DRY-RUN] Skip venv/dependency installation.")
        if not args.skip_model_restore:
            restore_model_cache(bundle_dir, manifest, dry_run=True)
        return

    venv_dir = (
        args.venv_dir.expanduser().resolve()
        if args.venv_dir
        else (bundle_dir / ".venv").resolve()
    )
    python_bin = ensure_venv(venv_dir)

    deps = [str(d) for d in manifest.get("dependencies", [])]
    if not deps:
        print("[WARN] empty dependency list in manifest")

    install_mode = args.mode
    if install_mode != "online" and not wheels_dir.exists():
        if install_mode == "offline":
            raise SystemExit(f"offline mode selected but wheels dir missing: {wheels_dir}")
        print("[WARN] wheels dir missing, switch to online mode")
        install_mode = "online"

    install_python_stack(
        python_bin=python_bin,
        wheels_dir=wheels_dir,
        mode=install_mode,
        deps=deps,
        project_dir=project_dir,
    )

    if not args.skip_model_restore:
        restore_model_cache(bundle_dir, manifest)

    write_launchers(bundle_dir, venv_dir, manifest)

    print("\nDeploy completed.")
    print(f"- venv: {venv_dir}")
    print(f"- CLI launcher: {bundle_dir / 'run_subgen.sh'}")
    print(f"- GUI launcher: {bundle_dir / 'run_subgen_gui.sh'}")


if __name__ == "__main__":
    main()
