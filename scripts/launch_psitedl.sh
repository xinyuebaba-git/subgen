#!/bin/bash
set -e

# PSiteDL 一键启动脚本（带自动更新检查）
PROJECT_DIR='/Users/yr001/Documents/New project'
cd "$PROJECT_DIR"

echo "========================================"
echo "  PSiteDL - 网页视频下载工具"
echo "========================================"

# 获取当前版本
CURRENT_VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "unknown")
echo "当前版本：$CURRENT_VERSION"

# 检查更新（后台静默检查）
echo "检查更新中..."
git fetch --tags --quiet 2>/dev/null || true
LATEST_VERSION=$(git describe --tags --abbrev=0 2>/dev/null || echo "")

if [ -n "$LATEST_VERSION" ] && [ "$CURRENT_VERSION" != "$LATEST_VERSION" ]; then
    echo ""
    echo "⚠️  发现新版本：$LATEST_VERSION"
    echo "   当前版本：$CURRENT_VERSION"
    echo ""
    read -p "是否现在更新？(y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "正在更新..."
        git pull
        git submodule update --init --recursive
        
        # 重新安装依赖
        echo "更新依赖..."
        if [ -x '.venv311/bin/python' ]; then
            PY='.venv311/bin/python'
        elif [ -x '.venv/bin/python' ]; then
            PY='.venv/bin/python'
        else
            PY='python3'
        fi
        
        "$PY" -m pip install -e . -q
        echo "✅ 更新完成！"
        echo ""
        
        # 更新版本号
        CURRENT_VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo "unknown")
        echo "新版本：$CURRENT_VERSION"
    else
        echo "跳过更新，启动旧版本..."
    fi
else
    echo "✅ 已是最新版本"
fi

echo ""
echo "正在启动 PSiteDL GUI..."
echo "========================================"

# 选择 Python
if [ -x '.venv311/bin/python' ]; then
  PY='.venv311/bin/python'
elif [ -x '.venv/bin/python' ]; then
  PY='.venv/bin/python'
else
  PY='python3'
fi

# 确保依赖已安装
"$PY" -m pip install -e . -q 2>/dev/null || true
"$PY" -m pip install -U yt-dlp playwright -q 2>/dev/null || true

# 启动 GUI
exec "$PY" -m webvidgrab.site_gui
