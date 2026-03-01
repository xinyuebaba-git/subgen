"""
PSiteDL 版本管理模块
"""
import subprocess
import json
from pathlib import Path
from typing import Optional

# 尝试从 pyproject.toml 读取版本
def get_version_from_pyproject() -> str:
    """从 pyproject.toml 读取版本号"""
    try:
        pyproject = Path(__file__).parent.parent.parent / "pyproject.toml"
        content = pyproject.read_text(encoding="utf-8")
        for line in content.splitlines():
            if line.startswith("version = "):
                return line.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "0.1.0"


def get_git_version() -> Optional[str]:
    """从 git tag 获取版本信息"""
    try:
        # 尝试获取最近的 tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_branch() -> Optional[str]:
    """获取当前 git 分支"""
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_commit() -> Optional[str]:
    """获取当前 git commit hash"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_version_info() -> dict:
    """获取完整版本信息"""
    version = get_version_from_pyproject()
    git_version = get_git_version()
    branch = get_git_branch()
    commit = get_git_commit()
    
    return {
        "version": version,
        "git_version": git_version,
        "branch": branch,
        "commit": commit,
        "display": f"{version} ({git_version})" if git_version else version,
    }


def check_for_updates() -> dict:
    """
    检查更新
    返回：{"has_update": bool, "current_version": str, "latest_version": str, "changes": list}
    """
    try:
        project_root = Path(__file__).parent.parent.parent
        result = subprocess.run(
            ["git", "fetch", "--tags"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # 获取最新 tag
        result = subprocess.run(
            ["git", "describe", "--tags", "--abbrev=0"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return {"has_update": False, "error": "无法获取最新版本"}
        
        latest_version = result.stdout.strip()
        current_version = get_git_version() or get_version_from_pyproject()
        
        # 比较版本
        has_update = (latest_version != current_version)
        
        # 获取更新日志
        changes = []
        if has_update:
            result = subprocess.run(
                ["git", "log", "--oneline", f"{current_version}..{latest_version}"],
                cwd=project_root,
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                changes = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        
        return {
            "has_update": has_update,
            "current_version": current_version,
            "latest_version": latest_version,
            "changes": changes,
        }
    except Exception as e:
        return {
            "has_update": False,
            "current_version": get_version_from_pyproject(),
            "latest_version": "未知",
            "error": str(e),
        }


if __name__ == "__main__":
    import json
    info = get_version_info()
    print("版本信息:")
    print(json.dumps(info, indent=2, ensure_ascii=False))
    
    print("\n检查更新:")
    update_info = check_for_updates()
    print(json.dumps(update_info, indent=2, ensure_ascii=False))
