from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = REPO_ROOT / "dist"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def load_pyproject_dependencies() -> list[str]:
    pyproject = REPO_ROOT / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    deps = project.get("dependencies", [])
    return [str(d) for d in deps]


def copy_project_tree(bundle_project_dir: Path) -> None:
    bundle_project_dir.mkdir(parents=True, exist_ok=True)
    for rel in ["README.md", "pyproject.toml", "config", "src"]:
        src = REPO_ROOT / rel
        dst = bundle_project_dir / rel
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        elif src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def export_wheels(wheels_dir: Path, deps: list[str]) -> None:
    wheels_dir.mkdir(parents=True, exist_ok=True)
    if not deps:
        return
    run([sys.executable, "-m", "pip", "download", "-d", str(wheels_dir), *deps])


def reuse_previous_wheels(wheels_dir: Path) -> tuple[Path, int] | None:
    bundles = sorted(
        [p for p in DIST_DIR.glob("psitedl-bundle-*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for b in bundles:
        src_wheels = b / "wheels"
        if not src_wheels.exists():
            continue
        if src_wheels.resolve() == wheels_dir.resolve():
            continue
        files = sorted(src_wheels.glob("*.whl"))
        if not files:
            continue
        wheels_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for f in files:
            dst = wheels_dir / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                copied += 1
        return (src_wheels, copied)
    return None


def create_archive(bundle_dir: Path, output_tgz: Path) -> None:
    output_tgz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tgz, "w:gz") as tf:
        tf.add(bundle_dir, arcname=bundle_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable PSiteDL deployment bundle with dependencies."
    )
    parser.add_argument(
        "--bundle-name",
        default=None,
        help="Bundle directory name (default auto-generated)",
    )
    parser.add_argument(
        "--skip-wheels",
        action="store_true",
        help="Skip pip wheel download step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (REPO_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"pyproject.toml not found under: {REPO_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bundle_name = args.bundle_name or f"psitedl-bundle-{timestamp}"

    bundle_dir = DIST_DIR / bundle_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_project_dir = bundle_dir / "project"
    wheels_dir = bundle_dir / "wheels"

    copy_project_tree(bundle_project_dir)

    deps = load_pyproject_dependencies()
    # PSiteDL runtime extras.
    deps.extend(["yt-dlp", "playwright", "pip", "setuptools", "wheel"])

    seen = set()
    normalized_deps: list[str] = []
    for dep in deps:
        key = dep.strip().lower()
        if key and key not in seen:
            normalized_deps.append(dep)
            seen.add(key)

    if not args.skip_wheels:
        try:
            export_wheels(wheels_dir, normalized_deps)
        except subprocess.CalledProcessError:
            reused = reuse_previous_wheels(wheels_dir)
            if reused is None:
                raise RuntimeError(
                    "Failed to download wheels and no reusable local wheels were found in dist/."
                ) from None
            src_wheels, copied = reused
            print(
                "[WARN] wheel download failed; reused local wheels from "
                f"{src_wheels} (copied {copied} files)."
            )

    deploy_script_src = REPO_ROOT / "scripts" / "deploy_psitedl_bundle.py"
    if not deploy_script_src.exists():
        raise SystemExit(
            "Missing deploy script: scripts/deploy_psitedl_bundle.py. Build aborted."
        )
    shutil.copy2(deploy_script_src, bundle_dir / "deploy_psitedl_bundle.py")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "project": {
            "name": "PSiteDL",
            "requires_python": ">=3.10",
        },
        "dependencies": normalized_deps,
        "runtime_tools": ["ffmpeg", "yt-dlp", "playwright"],
        "entrypoints": {"cli": "psitedl", "gui": "psitedl-gui"},
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    output_tgz = DIST_DIR / f"{bundle_name}.tar.gz"
    create_archive(bundle_dir, output_tgz)

    print("\nBundle created:")
    print(f"- Directory: {bundle_dir}")
    print(f"- Archive:   {output_tgz}")
    print("\nTarget machine deploy command:")
    print(f"python3 {bundle_name}/deploy_psitedl_bundle.py --bundle-dir {bundle_name}")


if __name__ == "__main__":
    main()

