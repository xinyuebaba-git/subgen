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


def run_capture(cmd: list[str]) -> str:
    print("+", " ".join(cmd))
    completed = subprocess.run(
        cmd,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def slugify(text: str) -> str:
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    s = "".join(out)
    while "--" in s:
        s = s.replace("--", "-")
    return s.strip("-") or "model"


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
        [p for p in DIST_DIR.glob("subgen-bundle-*") if p.is_dir()],
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


def find_python_with_module(module_name: str) -> Path | None:
    candidates = [
        REPO_ROOT / ".venv311" / "bin" / "python",
        REPO_ROOT / ".venv" / "bin" / "python",
        Path(sys.executable),
    ]
    seen: set[str] = set()
    for py in candidates:
        py = py.expanduser().resolve()
        if str(py) in seen or not py.exists():
            continue
        seen.add(str(py))
        check_cmd = [
            str(py),
            "-c",
            (
                "import importlib.util,sys;"
                "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
            ),
            module_name,
        ]
        rc = subprocess.run(check_cmd).returncode
        if rc == 0:
            return py
    return None


def find_local_faster_whisper_repo(model_name: str) -> Path | None:
    hub_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not hub_dir.exists():
        return None

    norm = model_name.strip().replace("/", "-").replace(":", "-")
    candidates = [
        hub_dir / f"models--Systran--faster-whisper-{norm}",
        hub_dir / f"models--systran--faster-whisper-{norm}",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    # Fallback fuzzy match for unusual model aliases.
    needle = f"faster-whisper-{norm}".lower()
    for p in hub_dir.glob("models--*"):
        name = p.name.lower()
        if needle in name and p.is_dir():
            return p.resolve()
    return None


def ensure_helper_python_with_module(
    module_name: str,
    pip_name: str,
    bundle_dir: Path,
    wheels_dir: Path | None,
) -> Path:
    helper_dir = bundle_dir / ".model-helper-venv"
    helper_python = helper_dir / "bin" / "python"
    if not helper_python.exists():
        run([sys.executable, "-m", "venv", str(helper_dir)])

    offline_cmd = [
        str(helper_python),
        "-m",
        "pip",
        "install",
        "--no-index",
        "--find-links",
        str(wheels_dir),
        pip_name,
    ]
    online_cmd = [str(helper_python), "-m", "pip", "install", pip_name]

    if wheels_dir is not None and wheels_dir.exists():
        try:
            run(offline_cmd)
        except Exception:
            run(online_cmd)
    else:
        run(online_cmd)

    check = [
        str(helper_python),
        "-c",
        (
            "import importlib.util,sys;"
            "sys.exit(0 if importlib.util.find_spec(sys.argv[1]) else 1)"
        ),
        module_name,
    ]
    if subprocess.run(check).returncode != 0:
        raise RuntimeError(
            f"Failed to prepare helper runtime with module: {module_name}"
        )
    return helper_python


def collect_faster_whisper_model(
    model_name: str,
    model_cache_dir: Path,
    bundle_dir: Path,
    wheels_dir: Path | None,
    model_offline_only: bool = False,
) -> dict[str, str]:
    model_candidate = Path(model_name).expanduser().resolve()
    if model_candidate.exists():
        model_path = model_candidate
    else:
        repo = find_local_faster_whisper_repo(model_name)
        if repo is not None:
            model_path = repo
        else:
            if model_offline_only:
                raise RuntimeError(
                    f"Offline-only mode enabled and local faster-whisper model is missing: {model_name}. "
                    "Please pre-download it on this machine first."
                )
            py = find_python_with_module("faster_whisper")
            if py is None:
                py = ensure_helper_python_with_module(
                    module_name="faster_whisper",
                    pip_name="faster-whisper",
                    bundle_dir=bundle_dir,
                    wheels_dir=wheels_dir,
                )
            download_cmd = [
                str(py),
                "-c",
                (
                    "import json,sys;"
                    "from faster_whisper.utils import available_models,download_model;"
                    "name=sys.argv[1];"
                    "known=set(available_models());"
                    "assert name in known, f'Unknown faster-whisper model: {name}';"
                    "path='';"
                    "try:\n"
                    " path=download_model(name, local_files_only=True)\n"
                    "except Exception:\n"
                    " path=download_model(name)\n"
                    "print(json.dumps({'model_path':str(path)}))"
                ),
                model_name,
            ]
            out = run_capture(download_cmd)
            payload = json.loads(out.splitlines()[-1])
            model_path = Path(str(payload["model_path"])).expanduser().resolve()

    repo_root = None
    for p in [model_path, *model_path.parents]:
        if p.name.startswith("models--"):
            repo_root = p
            break

    if repo_root is None:
        if not model_path.exists():
            raise RuntimeError(f"faster-whisper model not found: {model_path}")
        dst = model_cache_dir / "faster-whisper-local" / slugify(model_name)
        shutil.copytree(model_path, dst, dirs_exist_ok=True)
        return {
            "engine": "faster-whisper",
            "model_name": model_name,
            "cache_kind": "local_path",
            "cache_rel_path": str(dst.relative_to(model_cache_dir)),
        }

    dst = model_cache_dir / "huggingface" / "hub" / repo_root.name
    shutil.copytree(repo_root, dst, dirs_exist_ok=True)
    return {
        "engine": "faster-whisper",
        "model_name": model_name,
        "cache_kind": "huggingface_repo",
        "cache_rel_path": str(dst.relative_to(model_cache_dir)),
    }


def collect_openai_whisper_model(
    model_name: str,
    model_cache_dir: Path,
    bundle_dir: Path,
    wheels_dir: Path | None,
    model_offline_only: bool = False,
) -> dict[str, str]:
    src = Path.home() / ".cache" / "whisper" / f"{model_name}.pt"
    if not src.exists():
        if model_offline_only:
            raise RuntimeError(
                f"Offline-only mode enabled and local whisper model cache is missing: {src}"
            )
        py = find_python_with_module("whisper")
        if py is None:
            py = ensure_helper_python_with_module(
                module_name="whisper",
                pip_name="openai-whisper",
                bundle_dir=bundle_dir,
                wheels_dir=wheels_dir,
            )
        run(
            [str(py), "-c", "import whisper,sys; whisper.load_model(sys.argv[1])", model_name]
        )

    if not src.exists():
        raise RuntimeError(
            f"openai-whisper model cache not found after download: {src}"
        )

    dst = model_cache_dir / "whisper" / src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return {
        "engine": "whisper",
        "model_name": model_name,
        "cache_kind": "whisper_pt",
        "cache_rel_path": str(dst.relative_to(model_cache_dir)),
    }


def create_archive(bundle_dir: Path, output_tgz: Path) -> None:
    output_tgz.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output_tgz, "w:gz") as tf:
        tf.add(bundle_dir, arcname=bundle_dir.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a portable SubGen deployment bundle with dependencies and ASR model cache."
        )
    )
    parser.add_argument(
        "--asr-engine",
        choices=["faster-whisper", "whisper"],
        default="faster-whisper",
        help="ASR engine whose model cache should be packed",
    )
    parser.add_argument(
        "--asr-model",
        default="large-v3",
        help="ASR model name to pre-download and include in the bundle",
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
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip ASR model cache export",
    )
    parser.add_argument(
        "--model-offline-only",
        action="store_true",
        help=(
            "Only use already cached local ASR model; "
            "fail immediately if missing (no online download retries)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not (REPO_ROOT / "pyproject.toml").exists():
        raise SystemExit(f"pyproject.toml not found under: {REPO_ROOT}")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    model_tag = slugify(args.asr_model)
    bundle_name = args.bundle_name or f"subgen-bundle-{args.asr_engine}-{model_tag}-{timestamp}"

    bundle_dir = DIST_DIR / bundle_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_project_dir = bundle_dir / "project"
    wheels_dir = bundle_dir / "wheels"
    model_cache_dir = bundle_dir / "model-cache"

    copy_project_tree(bundle_project_dir)

    deps = load_pyproject_dependencies()
    if args.asr_engine == "whisper":
        deps.append("openai-whisper")
    deps.extend(["pip", "setuptools", "wheel"])

    # Deduplicate while preserving order.
    seen = set()
    normalized_deps = []
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

    model_info: dict[str, str] | None = None
    if not args.skip_model:
        if args.asr_engine == "faster-whisper":
            model_info = collect_faster_whisper_model(
                args.asr_model,
                model_cache_dir,
                bundle_dir=bundle_dir,
                wheels_dir=(wheels_dir if wheels_dir.exists() else None),
                model_offline_only=args.model_offline_only,
            )
        else:
            model_info = collect_openai_whisper_model(
                args.asr_model,
                model_cache_dir,
                bundle_dir=bundle_dir,
                wheels_dir=(wheels_dir if wheels_dir.exists() else None),
                model_offline_only=args.model_offline_only,
            )

    deploy_script_src = REPO_ROOT / "scripts" / "deploy_subgen_bundle.py"
    if not deploy_script_src.exists():
        raise SystemExit(
            "Missing deploy script: scripts/deploy_subgen_bundle.py. Build aborted."
        )
    shutil.copy2(deploy_script_src, bundle_dir / "deploy_subgen_bundle.py")

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "host": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "project": {
            "name": "subgen",
            "requires_python": ">=3.10",
        },
        "dependencies": normalized_deps,
        "asr": {
            "engine": args.asr_engine,
            "model": args.asr_model,
            "model_cache": model_info,
        },
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
    print(f"python3 {bundle_name}/deploy_subgen_bundle.py --bundle-dir {bundle_name}")


if __name__ == "__main__":
    main()
