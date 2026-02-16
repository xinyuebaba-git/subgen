# subgen (macOS)

一个用于自动字幕生成的命令行工具：

1. 指定一个或多个视频文件。
2. 调用本地 ASR（`whisper` 或 `faster-whisper`）生成源字幕（`.source.srt`）。
3. 调用翻译大模型，结合前后文翻译为简体中文（`.zh-CN.srt`）。

## 特性

- 本地语音识别（Whisper）
- 支持 ASR 引擎：`whisper`、`faster-whisper`（首次使用可自动安装依赖）
- 默认本地大模型翻译（Ollama）
- 批量视频处理
- 更短时间轴切分（默认 `2.2s`）以提升字幕同步
- 翻译时注入上下文窗口，提升术语一致性和语句自然度

## 环境

- macOS
- Python 3.10+
- 建议 `ffmpeg` 可用（Whisper 解码依赖）

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## 跨机器完整部署（含环境自检 + 依赖补齐 + ASR 模型迁移）

当目标机器环境不一致时，推荐使用仓库内置的部署包流程：

### 1) 在当前机器打包

示例（打包 `faster-whisper` 的 `large-v3` 本地模型）：

```bash
python3 scripts/build_subgen_bundle.py \
  --asr-engine faster-whisper \
  --asr-model large-v3
```

打包结果：

- `dist/subgen-bundle-.../`：部署目录（含源码、wheels、模型缓存、部署脚本）
- `dist/subgen-bundle-....tar.gz`：可直接拷贝到目标机器的压缩包

### 2) 在目标机器部署

解压后进入部署目录执行：

```bash
python3 deploy_subgen_bundle.py --bundle-dir .
```

部署脚本会自动执行：

- Python 版本检测（要求 `>=3.10`）
- `ffmpeg` 检测（默认尝试自动安装，失败则告警）
- 虚拟环境创建（`./.venv`）
- 依赖安装：优先离线 `wheels`，失败自动回退在线安装
- 恢复 ASR 模型缓存（`faster-whisper` 或 `whisper`）
- 生成启动脚本：`run_subgen.sh`、`run_subgen_gui.sh`

### 3) 运行

```bash
./run_subgen.sh /path/to/video.mp4
./run_subgen_gui.sh
```

可选参数：

- 完全离线安装：`python3 deploy_subgen_bundle.py --bundle-dir . --mode offline`
- 强制在线安装：`python3 deploy_subgen_bundle.py --bundle-dir . --mode online`
- 禁止自动安装 ffmpeg：`python3 deploy_subgen_bundle.py --bundle-dir . --no-auto-ffmpeg`
- 打包时模型仅离线缓存（缺失立即失败）：`python3 scripts/build_subgen_bundle.py --asr-engine faster-whisper --asr-model medium --model-offline-only`

## ASR 引擎与模型

- GUI 可选 ASR：`whisper` / `faster-whisper`
- 切换引擎后，模型下拉会按该引擎支持列表更新（如 `small`、`medium`、`large`）
- 首次使用某引擎时，若本机未安装对应包会自动安装；首次加载模型会自动下载权重

常见模型示例：
- `whisper`: `tiny`, `base`, `small`, `medium`, `large`, `turbo`
- `faster-whisper`: `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3`, `turbo`

官方说明：
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)

## 本地翻译模型准备（推荐）

默认翻译后端是本地 Ollama（无需云端 API Key）：

```bash
# 安装 Ollama 后
ollama pull qwen2.5:7b
# 可选：Dolphin 3 8B
ollama pull dolphin3:8b
# 可选：Dolphin 3 Abliterated（社区模型）
ollama pull huihui_ai/dolphin3-abliterated
```

确保 Ollama 服务可访问（默认 `http://localhost:11434`）。

当本地翻译模型为 `qwen2.5:*` 且使用 Ollama（`localhost:11434`）时，程序会自动应用 Qwen2.5 推荐采样参数：

- `temperature=0.7`
- `top_k=20`
- `top_p=0.8`
- `repeat_penalty=1.05`
- `num_ctx=32768`

同时在需要 JSON 返回的批量翻译请求中启用 JSON 模式，降低解析失败概率。

## 在线翻译配置文件

当翻译后端选择 `openai` 或 `deepseek` 时，程序会读取：

- `config/translation.toml`

示例结构：

```toml
[openai]
base_url = "https://api.openai.com/v1"
model = "gpt-4.1-mini"
api_key = "..."

[deepseek]
base_url = "https://api.deepseek.com/v1"
model = "deepseek-chat"
api_key = "..."
```

## 用法

### 1) 仅生成源字幕

```bash
subgen /path/video1.mp4 /path/video2.mov --no-translate
```

### 2) 生成源字幕 + 中文字幕（本地翻译）

```bash
subgen /path/video1.mp4 /path/video2.mov \
  --output-dir ./subtitles \
  --whisper-model medium \
  --translate-backend local \
  --translate-model qwen2.5:7b
```

输出示例：

- `subtitles/video1.source.srt`
- `subtitles/video1.zh-CN.srt`
- 当切换翻译模型重复翻译时，若 `video1.source.srt` 已存在会直接复用并跳过 Whisper。
- 中文字幕会附带后端与模型标识，示例：`video1.zh-CN.openai-gpt-4-1-mini.srt`

### 3) 可选：切换到 OpenAI 翻译后端

```bash
export OPENAI_API_KEY="your_api_key"
subgen /path/video.mp4 \
  --translate-backend openai \
  --translate-model gpt-4.1-mini
```

### 4) 可选：切换到 DeepSeek 翻译后端

```bash
export DEEPSEEK_API_KEY="your_api_key"
subgen /path/video.mp4 \
  --translate-backend deepseek \
  --translate-model deepseek-chat
```

## GUI（拖拽 + 可视化）

安装依赖后可直接启动图形界面：

```bash
subgen-gui
```

界面支持：

- 拖拽视频文件到列表区域
- 点击“添加文件/添加文件夹”导入视频
- 输出目录选择后会在界面中显示当前目录
- ASR 引擎下拉选择（`whisper` / `faster-whisper`）
- Whisper 模型下拉会随 ASR 引擎自动变化
- 若本机没有所选 Whisper 模型，首次运行会自动下载
- 翻译后端可选 `local/openai/deepseek`，默认 `local`
- 本地翻译模型下拉可选：`qwen2.5:7b`、`dolphin3:8b`、`huihui_ai/dolphin3-abliterated`
- 本地翻译默认连接 `http://localhost:11434/v1`
- OpenAI 默认连接 `https://api.openai.com/v1`
- DeepSeek 默认连接 `https://api.deepseek.com/v1`
- 在线后端（OpenAI/DeepSeek）会自动读取 `config/translation.toml`
- 启动后会显示 `Ollama 状态` 红绿提示，可手动“刷新检测”
- 点击“开始处理”前会再次检测 Ollama，未就绪会阻止启动并提示原因
- 一键开始处理并在日志区域查看进度

## 关键参数建议

- `--max-segment-duration`：默认 `2.2` 秒。要更紧同步可降到 `1.6~2.0`。
- `--max-segment-chars`：默认 `28`。短句更易读、更跟画面。
- `--source-language`：默认自动识别，可手动指定如 `en`、`ja`。
- `--translation-max-tokens`：默认 `4000`。超过上限会自动拆分为多次翻译请求。

## 示例：追求更紧同步

```bash
subgen /path/video.mp4 \
  --max-segment-duration 1.8 \
  --max-segment-chars 24
```

## 说明

- `whisper` / `faster-whisper` 首次运行会下载模型（如果传的是模型名，如 `medium`）。
- 也可以传本地模型目录给 `--whisper-model`（`faster-whisper`）。
- GUI 中可选择 `ASR 引擎`，若未安装会在首次使用自动安装后继续运行。
