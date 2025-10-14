# 🚀 IdeaSearch Streamlit App

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io)

一个基于 IdeaSearch 框架的智能符号回归 Web 应用，使用大语言模型自动发现数学表达式。只需绘制曲线，AI 即可为您找到最佳拟合公式

## ✨ 主要特性

- 🎨 **交互式绘图画布** - 直观绘制目标曲线，支持多种绘制模式
- 🤖 **多模型支持** - 集成 GPT、Gemini、Qwen、DeepSeek 等主流 LLM
- 🧠 **Fuzzy 模式** - 使用自然语言理论描述辅助拟合
- 📊 **实时可视化** - 动态展示拟合进度和结果对比
- 🏝️ **岛屿进化算法** - 并行探索多个解空间，提高拟合质量
- 📈 **Pareto 前沿分析** - 平衡表达式复杂度与拟合精度
- 📁 **多数据格式支持** - 支持画布绘制和 NPZ 文件上传

## 📋 前置要求

- Python 3.10 或更高版本
- [uv](https://github.com/astral-sh/uv) 包管理器（推荐，快速且可靠）
- 有效的 LLM API 密钥（支持 Gemini、OpenAI、Qwen、DeepSeek 等）
- IdeaSearch-framework 和 IdeaSearch-fit 依赖库（自动安装）

## 🎬 快速开始

### 1. 环境准备

```bash
# 克隆仓库
git clone <repository-url>
cd ideasearch-streamlit-app

# 安装 uv 包管理器（如未安装）
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 安装依赖

```bash
uv sync
```

### 3. 配置 API 密钥

```bash
# 复制示例配置文件
cp api_keys.json.example api_keys.json

# 编辑配置文件，添加您的 API 密钥
```

API 密钥格式：
```json
{
  "Gemini_2.5_Flash": [{
    "api_key": "your-api-key-here",
    "base_url": "https://generativelanguage.googleapis.com/v1beta",
    "model": "gemini-2.0-flash-exp"
  }],
  "GPT_4o_Mini": [{
    "api_key": "your-openai-api-key",
    "base_url": "https://api.openai.com/v1",
    "model": "gpt-4o-mini"
  }]
}
```

### 4. 启动应用

```bash
# 使用启动脚本（推荐）
./run.sh

# 或手动启动
uv run streamlit run app.py --server.port 8501
```

应用将自动在浏览器中打开：`http://localhost:8501`

## 📖 使用指南

### 三种拟合模式

应用提供三个标签页，对应不同的使用场景：

1. **🎨 绘制曲线拟合** - 交互式画布绘制
2. **📁 上传文件拟合** - NPZ 数据文件上传
3. **⚙️ 高级配置拟合** - 完整的物理单位验证和变量配置

### 基本工作流程

1. **数据准备**
   - **画布模式**: 在左侧画布绘制目标曲线，支持自由绘制、直线、点三种模式
   - **文件模式**: 上传包含 'x', 'y' 和可选 'error' 键的 NPZ 文件
   - 支持数据平滑和插值处理，提高拟合质量

2. **参数配置**（侧边栏）
   - **模型选择**: 推荐使用 Gemini 2.5 Flash（速度与质量平衡）
   - **函数配置**: 选择可用的数学函数（sin, cos, exp, log, sqrt 等）
   - **拟合参数**: 
     - 岛屿数量：并行搜索种群数（推荐：3-8）
     - 循环次数：进化代数（推荐：3-10）
     - 单元交互数：每代的 LLM 调用次数（推荐：5-10）
     - 目标分数：达到后提前停止（默认：80.0）
   - **Fuzzy 模式**: 启用自然语言理论描述辅助

3. **执行拟合**
   - 点击"开始拟合"按钮
   - 实时监控拟合进度和 API 调用情况
   - 动态查看当前最佳表达式和分数
   - 可随时手动停止拟合过程

4. **结果分析**
   - **拟合对比图**: 原始数据 vs 拟合曲线可视化
   - **分数历史**: 展示拟合质量随迭代的改进过程
   - **Pareto 前沿**: 分析复杂度与精度的权衡关系
   - **API 调用日志**: 详细的模型调用记录和性能统计

### 高级功能

- **Fuzzy 模式**: 先生成自然语言理论描述，再转换为数学表达式
- **单位验证**: 在高级模式中配置物理量单位，确保表达式量纲正确
- **多维数据**: 支持多特征输入的复杂函数拟合
- **实时更新**: 每个 epoch 后立即更新结果，无需等待完整循环

## 🏗️ 项目结构

```
ideasearch-streamlit-app/
├── app.py                      # 主应用入口，包含三个拟合模式的标签页
├── src/
│   ├── components/             # UI 组件模块
│   │   ├── canvas.py          # 交互式绘图画布组件
│   │   ├── config.py          # 侧边栏配置界面组件
│   │   └── results.py         # 结果可视化和日志展示组件
│   ├── core/                   # 核心业务逻辑
│   │   └── fitting.py         # FittingEngine 拟合引擎类
│   └── config/                 # 配置文件
│       └── default.yaml       # 默认参数配置
├── logs/                       # 拟合日志目录（自动生成）
│   ├── fit_YYYYMMDD_HHMMSS/   # 每次拟合的详细日志
│   └── db_YYYYMMDD_HHMMSS/    # IdeaSearcher 数据库文件
├── api_keys.json              # API 密钥配置（需手动创建）
├── api_keys.json.example      # API 密钥配置示例
├── pyproject.toml             # uv 项目依赖配置
├── run.sh                     # 启动脚本（支持参数配置）
├── ARCHITECTURE.md            # 完整的项目架构文档
└── README.md                  # 本文件
```

## ⚙️ 配置说明

### 关键参数调优指南

| 参数 | 推荐值 | 说明 | 调优建议 |
|------|--------|------|----------|
| **岛屿数量** | 3-8 | 并行进化种群数 | 增加提高多样性，但消耗更多 API |
| **循环次数** | 3-10 | 进化代数 | 更多循环通常得到更好结果 |
| **单元交互数** | 5-10 | 每循环的 LLM 调用次数 | 平衡探索深度与成本 |
| **目标分数** | 80.0 | 自动停止阈值 | 根据精度要求调整（0-100） |
| **采样温度** | 10-30 | 生成随机性控制 | 高温度增加创造性，低温度更稳定 |
| **启用变异** | false | 表达式变异操作 | 开启可增加搜索多样性，但可能降低稳定性 |
| **启用交叉** | false | 表达式交叉操作 | 开启可结合不同表达式特征 |
| **优化方法** | L-BFGS-B | 参数优化算法 | L-BFGS-B适合平滑函数，differential-evolution适合复杂函数 |
| **优化试验次数** | 5 | 每个表达式的参数优化次数 | 减小可显著提高拟合速度，增加可能提高精度 |

### API 密钥配置格式

```json
{
  "ModelName": [{
    "api_key": "your-api-key",
    "base_url": "https://api.provider.com/v1",
    "model": "model-identifier"
  }]
}
```

**支持的模型名称**（精确匹配）：
- **Gemini 系列**: `Gemini_2.5_Flash`, `Gemini_2.5_Pro`, `Gemini_Pro`
- **OpenAI 系列**: `GPT_4o`, `GPT_4o_Mini`, `GPT_4_Turbo`
- **国产模型**: `Qwen_Plus`, `Qwen_Max`, `Qwen3`, `Doubao`
- **开源模型**: `Deepseek_V3`, `Grok_4`

## 🔧 依赖项

核心依赖：
- `streamlit` - Web 应用框架
- `numpy`, `pandas` - 数据处理
- `plotly` - 交互式可视化
- `scipy` - 科学计算
- `streamlit-drawable-canvas` - 绘图画布组件

集成模块：
- `IdeaSearch-framework` - 核心优化引擎
- `IdeaSearch-fit` - 符号回归适配器

## 🔧 故障排查

### 环境问题

**Q: Python 版本不兼容？**
```bash
# 检查版本
python3 --version  # 需要 >= 3.10

# 升级 Python（Ubuntu/Debian）
sudo apt update && sudo apt install python3.10
```

**Q: uv 安装失败？**
```bash
# 手动安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 或使用 pip 安装
pip install uv
```

### API 相关问题

**Q: API 调用失败？**
1. 检查 `api_keys.json` 格式（JSON 语法正确性）
2. 确认 API 密钥有效且有余额
3. 验证网络连接和代理设置
4. 检查模型名称是否与配置文件键名完全匹配

**Q: 某个模型无响应？**
- 尝试切换到其他模型（如 Gemini 2.5 Flash）
- 检查该 API 服务商的状态页面
- 降低并发请求数（减少采样器/评估器数量）

### 拟合效果问题

**Q: 拟合结果不理想？**
1. **增加搜索强度**：提高岛屿数量（3→8）和循环次数（5→10）
2. **启用 Fuzzy 模式**：使用自然语言理论描述辅助
3. **尝试不同模型**：GPT-4o 通常比 Mini 版本效果更好
4. **优化数据质量**：确保画布曲线清晰、数据点分布均匀
5. **调整函数库**：根据预期函数类型选择合适的基础函数

**Q: 日志在哪里查看？**
- 自动保存在 `logs/` 目录下
- 每次拟合创建带时间戳的子目录
- 包含完整的拟合过程、API 调用和错误信息
- 可在终端查看实时输出日志

### 性能优化建议

- **快速测试**：岛屿数=3，循环数=3，交互数=5
- **高质量拟合**：岛屿数=8，循环数=10，交互数=10
- **成本控制**：使用 Gemini 2.5 Flash 或 GPT-4o Mini
- **复杂表达式**：启用 Fuzzy 模式，使用 GPT-4o

## 📚 相关文档

- [ARCHITECTURE.md](ARCHITECTURE.md) - 完整的项目架构设计文档
- [IdeaSearch 框架]() - 核心算法框架文档
- [IdeaSearch-fit]() - 符号回归适配器文档

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库并创建特性分支
2. 遵循现有代码风格和注释规范
3. 添加必要的测试和文档
4. 提交 Pull Request 并描述变更内容

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 📞 支持与反馈

遇到问题或有改进建议？
- 📋 查看 [现有 Issues](../../issues)
- 🆕 [创建新 Issue](../../issues/new) 并提供详细信息
- 📧 联系开发团队

---

**🎯 开始探索 AI 驱动的符号回归吧！**

*基于大语言模型的智能数学表达式发现系统*

hhhhhhhh
gggggggggg
