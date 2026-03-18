**PaperResearchAgent** 是一个基于 LLM 的科研辅助工具，用于自动化论文搜索、解析、分析和报告生成。它支持知识库管理、PDF 文件上传与处理，以及前端交互式报告展示。

---

## 功能

- **论文搜索与下载**  
  - 支持 arXiv 等平台的论文搜索
  - PDF 文件自动下载与解析

- **论文解析与分析**  
  - 自动提取论文核心方法、数据集、评估指标、实验结果等信息
  - 可生成结构化分析报告

- **知识库管理**  
  - 支持增删改查知识库和文档
  - 支持上传 PDF、JSONL 文件

- **前端展示**  
  - Vue 3 前端界面
  - 查询数据库、查看分析结果

## 核心节点（Core Nodes）

PaperResearchAgent 的核心设计是一个 **论文搜索 → 阅读 → 分析 → 报告生成** 的流水线，每个阶段用“节点”管理状态和处理逻辑。

---

### 1️⃣ Search Node（搜索节点）

- **职责**：查询论文、获取元信息（title、abstract、authors、PDF 链接等）  
- **输入**：用户输入的查询关键词或条件  
- **输出**：论文列表（包含基础信息）  
- **关键组件**：
  - arXiv / Google Scholar / 自定义知识库查询
  - 分数排序、引用排序、筛选最新论文  
- **状态标记**：`ExecutionState.SEARCHING`

---

### 2️⃣ Reading Node（阅读节点）

- **职责**：下载论文 PDF 并解析，提取结构化内容  
- **输入**：搜索节点输出的论文列表  
- **输出**：`ExtractedPaperData`（包含方法、数据集、指标、实验结果、贡献等）  
- **关键组件**：
  - PDF 下载 Worker
  - AutoGen 解析 Agent
  - 异步任务调度（支持多论文并行处理）  
- **状态标记**：`ExecutionState.READING` / `ExecutionState.PARSING`

---

### 3️⃣ Analyse Node（分析节点）

- **职责**：对阅读节点提取的数据进行汇总、分析，生成报告  
- **输入**：多个论文解析结果  
- **输出**：综合分析报告，可能包含：
  - 主要方法与原理  
  - 数据集使用情况  
  - 实验结果总结  
  - 创新点与局限  
- **关键组件**：
  - AutoGen 分析 Agent
  - 异步队列 `state_queue` → 与前端推送分析进度  
- **状态标记**：`ExecutionState.ANALYZING`

---

### 🔹 节点关系图（逻辑流）

```text
[Search Node] --> [Reading Node] --> [Analyse Node]


## 快速开始（Quick Start）

### 1️⃣ 环境准备

1. 克隆仓库：
```bash
git clone https://github.com/Xxxie-v/PaperResearchAgent.git
cd PaperResearchAgent

2.环境配置：
pip install -r requirements.txt
复制 .env.example 为 .env 并填写您的API密钥
修改 models.yaml 中的参数

3.运行系统：
poetry run python main.py
python download_worker.py //下载调度
python analyse_worker.py //分析调度
python feeder_main.py //总体调度
python worker_main.py //总控
npm run dev //前端




