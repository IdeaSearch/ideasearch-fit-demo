# IdeaSearch Streamlit App - 项目概览

## 📋 项目简介

这是一个基于 IdeaSearch 框架的智能符号回归 Web 应用，使用大语言模型进行数学表达式拟合。用户可以通过绘制曲线，让 AI 自动发现背后的数学规律。

## 🏗️ 项目架构

```
ideasearch-streamlit-app/
├── app.py                      # 主应用入口
├── src/                        # 源代码目录
│   ├── components/             # UI 组件
│   │   ├── canvas.py          # 绘图画布组件
│   │   ├── config.py          # 配置侧边栏组件
│   │   └── results.py         # 结果展示组件
│   ├── core/                   # 核心逻辑
│   │   └── fitting.py         # 拟合引擎
│   ├── config/                 # 配置文件
│   │   └── default.yaml       # 默认配置
│   └── utils/                  # 工具函数（预留）
├── data/                       # 数据存储目录
├── results/                    # 结果输出目录
├── api_keys.json              # API 密钥配置
├── pyproject.toml             # 项目依赖配置
├── README.md                  # 项目说明
├── USAGE.md                   # 使用指南
└── QUICKSTART.md              # 快速开始
```

## 🔧 核心模块

### 1. 画布组件 (`src/components/canvas.py`)

**功能**：
- 提供交互式绘图界面
- 支持自由绘制、直线、点三种模式
- 提取画布数据并转换为数值数组
- 数据平滑和插值处理

**关键函数**：
- `render_drawing_canvas()`: 渲染画布 UI
- `extract_curve_from_canvas()`: 从画布提取曲线数据
- `smooth_curve()`: 曲线平滑
- `interpolate_curve()`: 曲线插值
- `process_canvas_data()`: 完整数据处理流程

### 2. 配置组件 (`src/components/config.py`)

**功能**：
- 侧边栏配置界面
- API 密钥管理
- 模型选择
- 函数和参数配置

**关键函数**：
- `render_sidebar_config()`: 渲染侧边栏
- `load_available_models()`: 加载可用模型列表
- `load_default_config()`: 加载默认配置

### 3. 结果展示组件 (`src/components/results.py`)

**功能**：
- 实时进度监控
- API 调用日志展示
- 拟合结果可视化
- Pareto 前沿分析
- 历史分数曲线

**关键函数**：
- `render_progress_section()`: 进度条和指标
- `render_api_calls_log()`: API 调用记录表
- `render_fitting_comparison()`: 拟合对比图
- `render_pareto_frontier()`: Pareto 前沿图
- `render_score_history()`: 历史分数曲线

### 4. 拟合引擎 (`src/core/fitting.py`)

**功能**：
- 封装 IdeaSearch 和 IdeaSearchFitter
- 管理拟合流程
- 状态跟踪和更新
- 异步执行支持

**核心类**：`FittingEngine`

**关键方法**：
- `initialize_fitter()`: 初始化 IdeaSearchFitter
- `initialize_searcher()`: 初始化 IdeaSearcher
- `run_fitting()`: 执行拟合主循环
- `get_state()`: 获取当前状态
- `get_pareto_frontier()`: 获取 Pareto 前沿
- `evaluate_expression()`: 评估数学表达式

### 5. 主应用 (`app.py`)

**功能**：
- 整合所有组件
- 管理应用状态
- 处理用户交互
- 异步拟合控制

**核心流程**：
1. 初始化 session state
2. 渲染 UI 组件
3. 处理用户输入
4. 启动/停止拟合
5. 实时更新结果

## 🔄 数据流

```
用户绘制曲线
    ↓
提取画布数据 (canvas.py)
    ↓
数据处理（平滑/插值）
    ↓
创建 FittingEngine (fitting.py)
    ↓
初始化 IdeaSearchFitter
    ↓
初始化 IdeaSearcher
    ↓
执行拟合循环
    ↓
LLM 生成表达式
    ↓
评估表达式质量
    ↓
更新最佳结果
    ↓
展示结果 (results.py)
```

## 🎯 关键技术

### 1. Streamlit 状态管理

使用 `st.session_state` 管理应用状态：
- `fitting_engine`: 拟合引擎实例
- `is_fitting`: 拟合状态标志
- `fitting_state`: 拟合详细状态
- `original_data`: 原始数据
- `pareto_data`: Pareto 前沿数据

### 2. 异步执行

使用 Python `threading` 模块实现异步拟合：
- 主线程负责 UI 更新
- 子线程执行拟合计算
- 回调函数更新状态

### 3. IdeaSearch 集成

**IdeaSearchFitter 配置**：
```python
IdeaSearchFitter(
    result_path=...,           # 结果存储路径
    data={"x": x, "y": y},     # 数据
    functions=[...],            # 可用函数
    constant_whitelist=[...],   # 常量白名单
    generate_fuzzy=True,        # Fuzzy 模式
    metric_mapping="logarithm", # 度量映射
)
```

**IdeaSearcher 配置**：
```python
ideasearcher.set_database_path(...)
ideasearcher.bind_helper(fitter)
ideasearcher.set_models([...])
ideasearcher.set_samplers_num(...)
ideasearcher.set_evaluators_num(...)
# ... 其他参数
```

## 📊 配置说明

### API 密钥格式 (`api_keys.json`)

```json
{
  "ModelName": [
    {
      "api_key": "your-api-key",
      "base_url": "https://api.provider.com/v1",
      "model": "model-identifier"
    }
  ]
}
```

### 默认配置 (`src/config/default.yaml`)

包含以下配置节：
- `app`: 应用基本配置
- `canvas`: 画布配置
- `ideasearch`: IdeaSearch 参数
- `fitter`: IdeaSearchFitter 参数
- `data`: 数据处理参数
- `ui`: UI 显示参数

## 🚀 启动流程

1. **环境准备**
   ```bash
   uv sync  # 安装依赖
   ```

2. **配置 API**
   ```bash
   cp api_keys.json.example api_keys.json
   # 编辑 api_keys.json
   ```

3. **启动应用**
   ```bash
   ./run.sh  # 或 uv run streamlit run app.py
   ```

4. **使用应用**
   - 绘制曲线
   - 配置参数
   - 开始拟合
   - 查看结果

## 🛠️ 扩展开发

### 添加新组件

1. 在 `src/components/` 创建新文件
2. 实现组件函数
3. 在 `__init__.py` 中导出
4. 在 `app.py` 中引入使用

### 添加新功能

1. 在 `src/core/` 或 `src/utils/` 添加实现
2. 更新配置文件（如需要）
3. 在主应用中集成
4. 更新文档

### 自定义配置

修改 `src/config/default.yaml` 调整默认参数

## 📝 代码规范

- **Python 版本**: 3.10+
- **代码风格**: PEP 8
- **类型提示**: 推荐使用 typing
- **文档字符串**: Google 风格
- **模块化**: 功能拆分为独立组件
- **错误处理**: 使用 try-except 捕获异常

## 🔍 调试技巧

### 1. 查看日志

在终端查看 Streamlit 输出日志

### 2. 调试模式

在 `default.yaml` 中设置：
```yaml
ui:
  show_debug_info: true
```

### 3. 检查状态

使用 Streamlit 的调试工具：
```python
st.write(st.session_state)  # 查看所有状态
```

### 4. 错误追踪

在 `fitting.py` 中启用详细错误输出：
```python
import traceback
traceback.print_exc()
```

## 🎯 性能优化

1. **减少重绘**：使用 `st.cache_data` 缓存数据
2. **异步执行**：拟合过程在后台线程运行
3. **按需加载**：只在需要时加载大型模块
4. **配置优化**：根据需求调整岛屿数和循环数

## 📚 相关文档

- [README.md](README.md) - 项目介绍
- [USAGE.md](USAGE.md) - 详细使用指南
- [QUICKSTART.md](QUICKSTART.md) - 快速开始
- [IdeaSearch 文档](../IdeaSearch-framework/README.md) - 框架文档

## 🤝 贡献指南

1. Fork 本仓库
2. 创建特性分支
3. 提交更改
4. 发起 Pull Request

## 📄 许可证

本项目是 IdeaSearch 生态系统的一部分，遵循相同的许可证。

---

**Happy Coding! 🎉**
