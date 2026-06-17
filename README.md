# 多模态虚假新闻检测系统

这是一个面向新闻文本和配图联合核验的本地应用。系统包含 React 前端、FastAPI 后端和 Python AI 推理核心，支持 URL 内容抓取、中文文本处理、多模态预测、SHAP 模态贡献度、文本归因、图像热力图、案例归档、批量处理和效能分析。

## 项目结构

```text
backend_api/          FastAPI 接口服务
ai_engine_core/       模型推理、解释分析、案例库和 Streamlit 页面
frontend_web/         React + Vite 前端
python_requirements.txt
启动平台.bat
```

## 环境要求

- Python 3.10 或以上
- Node.js 18 或以上
- Windows 10/11，macOS 或 Linux 均可运行核心服务
- 如需更快的图像推理，建议使用支持 CUDA 的 NVIDIA 显卡

## 安装依赖

安装 Python 依赖：

```bash
pip install -r python_requirements.txt
```

安装前端依赖：

```bash
cd frontend_web
npm install
```

## 启动方式

启动后端接口：

```bash
uvicorn backend_api.main:app --host 127.0.0.1 --port 8000
```

启动前端页面：

```bash
cd frontend_web
npm run dev
```

默认前端地址为 `http://127.0.0.1:5173`，后端接口地址为 `http://127.0.0.1:8000`。

Windows 环境下也可以尝试双击 `启动平台.bat` 一键启动。

## 数据和模型

仓库中保留了 `ai_engine_core/models/` 下的小型模型权重文件。`sample_dataset/` 数据集文件体积较大，默认不纳入 Git 仓库。若需要使用样本库或效能分析，请自行将数据文件放回 `sample_dataset/` 目录。

## 运行说明

- 检测页面支持手动输入文本、URL 抓取、本地图片上传和候选图片选择。
- 中文输入会尝试进入翻译流程，系统会保留原文和模型实际使用文本。
- 检测完成后，可以查看预测结论、P(Real)、置信度、模态贡献度、文本归因和图像热力图。
- 保存为案例后，系统会写入本地 `ai_engine_core/outputs/`，该目录是运行产物，不纳入版本管理。

## 注意事项

本项目用于内容审核辅助和实验演示。模型输出不应直接作为最终事实判定，建议结合人工复核、原始来源和其他证据一起使用。
