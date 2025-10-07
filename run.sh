#!/bin/bash

# IdeaSearch Streamlit App 启动脚本
# 功能：自动检查环境、安装依赖并启动应用

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

# 打印标题
clear
echo -e "${CYAN}"
echo "╔════════════════════════════════════════════╗"
echo "║   🚀 IdeaSearch Streamlit App Launcher     ║"
echo "║   基于大语言模型的智能符号回归系统         ║"
echo "║   AI-Powered Symbolic Regression App       ║"
echo "╚════════════════════════════════════════════╝"
echo -e "${NC}"

# 解析命令行参数
PORT="${PORT:-8501}"
HOST="${HOST:-localhost}"
AUTO_OPEN="${AUTO_OPEN:-true}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --no-browser)
            AUTO_OPEN="false"
            shift
            ;;
        --help|-h)
            echo -e "${CYAN}用法${NC}: $0 [选项]"
            echo ""
            echo -e "${CYAN}选项${NC}:"
            echo "  --port PORT      设置端口号 (默认: 8501)"
            echo "  --host HOST      设置主机地址 (默认: localhost)"
            echo "  --no-browser     不自动打开浏览器"
            echo "  --help, -h       显示此帮助信息"
            echo ""
            echo -e "${CYAN}示例${NC}:"
            echo "  $0                     # 使用默认配置启动"
            echo "  $0 --port 8080         # 在端口 8080 启动"
            echo "  $0 --host 0.0.0.0      # 允许外部访问"
            echo "  $0 --no-browser        # 不自动打开浏览器"
            echo ""
            echo -e "${CYAN}环境变量${NC}:"
            echo "  PORT=8080 $0           # 通过环境变量设置端口"
            echo "  HOST=0.0.0.0 $0        # 通过环境变量设置主机"
            exit 0
            ;;
        *)
            print_error "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# 检查项目目录
print_info "检查项目环境..."
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml 文件不存在，请确保在项目根目录运行此脚本"
    exit 1
fi

if [ ! -f "app.py" ]; then
    print_error "app.py 文件不存在"
    exit 1
fi

print_success "项目文件检查通过 ✓"

# 检查 Python 版本
print_info "检查 Python 版本..."
if ! command -v python3 &> /dev/null; then
    print_error "未找到 Python 3，请安装 Python 3.10 或更高版本"
    print_info "安装指南："
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.10"
    echo "  macOS: brew install python@3.10"
    echo "  Windows: https://www.python.org/downloads/"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
REQUIRED_VERSION="3.10"

# 版本比较函数
version_ge() {
    [ "$(echo -e "$1\n$2" | sort -V | tail -n1)" = "$1" ]
}

if ! version_ge "$PYTHON_VERSION" "$REQUIRED_VERSION"; then
    print_error "Python 版本过低: $PYTHON_VERSION (需要 >= $REQUIRED_VERSION)"
    exit 1
fi

print_success "Python 版本: $PYTHON_VERSION ✓"

# 检查 uv 包管理器
print_info "检查 uv 包管理器..."
if ! command -v uv &> /dev/null; then
    print_warning "未找到 uv 包管理器，正在自动安装..."
    
    # 检测操作系统
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        if command -v curl &> /dev/null; then
            curl -LsSf https://astral.sh/uv/install.sh | sh
        elif command -v wget &> /dev/null; then
            wget -qO- https://astral.sh/uv/install.sh | sh
        else
            print_error "需要 curl 或 wget 来安装 uv"
            exit 1
        fi
    else
        print_error "请手动安装 uv: https://github.com/astral-sh/uv"
        echo "  Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\""
        exit 1
    fi
    
    # 重新加载环境变量
    export PATH="$HOME/.cargo/bin:$PATH"
    source ~/.bashrc 2>/dev/null || source ~/.zshrc 2>/dev/null || true
    
    if ! command -v uv &> /dev/null; then
        print_error "uv 自动安装失败"
        print_info "请手动安装 uv："
        echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "  然后重新启动终端"
        exit 1
    fi
    print_success "uv 安装成功 ✓"
else
    UV_VERSION=$(uv --version | cut -d' ' -f2)
    print_success "uv 版本: $UV_VERSION ✓"
fi

# 检查和创建虚拟环境
print_info "准备 Python 虚拟环境..."
if [ ! -d ".venv" ]; then
    print_info "创建虚拟环境..."
    uv venv --python python3
fi
print_success "虚拟环境就绪 ✓"

# 同步项目依赖
print_info "同步项目依赖..."
if ! uv sync --frozen; then
    print_warning "frozen 同步失败，尝试常规同步..."
    if ! uv sync; then
        print_error "依赖安装失败"
        print_info "尝试解决方案："
        echo "  1. 删除 .venv 目录后重试"
        echo "  2. 手动运行: uv sync --reinstall"
        echo "  3. 检查网络连接和代理设置"
        exit 1
    fi
fi
print_success "依赖安装完成 ✓"

# 创建必要目录
print_info "准备应用目录..."
mkdir -p logs
mkdir -p data
print_success "应用目录就绪 ✓"

# 检查 API 密钥文件
print_info "检查 API 密钥配置..."
if [ ! -f "api_keys.json" ]; then
    if [ -f "api_keys.json.example" ]; then
        print_warning "未找到 api_keys.json，从示例文件创建"
        cp api_keys.json.example api_keys.json
        print_warning "⚠️  请编辑 api_keys.json 添加您的 API 密钥"
        print_info "支持的模型: Gemini_2.5_Flash, GPT_4o_Mini, Qwen_Plus, Deepseek_V3"
    else
        print_warning "未找到 api_keys.json 文件"
        print_info "请创建 api_keys.json 文件并添加 API 密钥"
        print_info "参考 README.md 中的配置示例"
    fi
else
    # 简单验证 JSON 格式
    if python3 -c "import json; json.load(open('api_keys.json'))" 2>/dev/null; then
        print_success "API 密钥文件格式正确 ✓"
    else
        print_error "api_keys.json 格式错误，请检查 JSON 语法"
        exit 1
    fi
fi

# 启动应用前的最终检查
print_info "启动前检查..."
sleep 1

echo ""
print_header "════════════════════════════════════════"
print_header "🚀 启动 IdeaSearch Streamlit 应用"
print_header "════════════════════════════════════════"
echo ""

print_info "配置信息："
echo "  📍 主机地址: ${HOST}"
echo "  🔌 端口号: ${PORT}"
echo "  🌐 访问地址: http://${HOST}:${PORT}"
echo "  🖥️  自动打开浏览器: $( [[ "$AUTO_OPEN" == "true" ]] && echo "是" || echo "否" )"
echo ""

print_info "⏳ 正在启动应用，请稍候..."
print_info "💡 提示: 使用 Ctrl+C 停止应用"
echo ""

# 构建 streamlit 命令参数
STREAMLIT_ARGS=(
    "--server.port" "$PORT"
    "--server.address" "$HOST"
    "--server.headless" "true"
    "--browser.gatherUsageStats" "false"
    "--theme.base" "light"
    "--server.maxUploadSize" "200"
    "--server.enableWebsocketCompression" "true"
)

# 根据浏览器设置添加参数
if [[ "$AUTO_OPEN" == "false" ]]; then
    STREAMLIT_ARGS+=("--server.allowRunOnSave" "true")
fi

# 启动 Streamlit 应用
exec uv run streamlit run app.py "${STREAMLIT_ARGS[@]}"
