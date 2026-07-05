# SynapseRAG-MCP

[English](../README.md) | [中文](README.zh-CN.md) | [Dansk](README.da.md) | [Deutsch](README.de.md)

SynapseRAG-MCP 是一个模块化的检索增强生成（RAG）系统：从文档切分、向量化、稠密 + 稀疏混合检索（RRF 融合）、重排序到答案生成，流水线的每个环节都是可替换的独立组件。内置 MCP server，让智能体直接查询你的知识库；附带 Streamlit 可观测仪表盘，覆盖导入、追踪、评测与实时配置。既可以完全本地运行，也可以公开部署——按会话隔离工作区，数据一小时自动过期。

**[在线体验 →](https://synapserag-mcp-server.streamlit.app)**

![SynapseRAG MCP 仪表盘截图](screenshot.jpg)

## 安装

```bash
pip install -r requirements.txt
```

## 启动仪表盘

本地模式——完整仪表盘，读取 `config/settings.yaml`，无掩码：

```bash
python scripts/start_dashboard.py
```

## 导入文件

```bash
python scripts/ingest.py --path pdfs/
python scripts/ingest.py --path pdfs/example.pdf
python scripts/ingest.py --path pdfs/ --collection my_docs
python scripts/ingest.py --path pdfs/ --force
```

## 查询

```bash
python scripts/query.py --query "your question"
python scripts/query.py --query "your question" --top-k 5 --collection my_docs
python scripts/query.py --query "your question" --verbose
```

## 评测

```bash
python scripts/evaluate.py
```

## 运行测试

```bash
pytest
pytest -m unit
```

## 部署（Streamlit Community Cloud）

公开模式提供相同的仪表盘，并增加 Start（登录）与 Session 页面：按浏览器会话隔离工作区、数据一小时 TTL、对访客掩码本地路径与密钥（管理员可见未掩码数据）。

```bash
streamlit run streamlit_app.py
```

1. 推送本仓库到 GitHub，在 share.streamlit.io 创建应用，入口为 `streamlit_app.py`。
2. 在应用的 Settings → Secrets 中配置：

```toml
ADMIN_PASSWORD = "change-me-to-a-long-random-string"
OWNER_LLM_PROVIDER = "openai"
OWNER_LLM_MODEL = "example-model-name"
OWNER_LLM_BASE_URL = "https://api.example-llm.com/v1"
OWNER_LLM_API_KEY = "sk-example-fake-key-000000"

# 可选：为 admin 会话配置真实 embedding 模型（guest 始终使用免费的本地哈希）
OWNER_EMBEDDING_PROVIDER = "openai"
OWNER_EMBEDDING_MODEL = "BAAI/bge-m3"
OWNER_EMBEDDING_BASE_URL = "https://api.siliconflow.cn/v1"
OWNER_EMBEDDING_API_KEY = "sk-example-fake-key-000000"
OWNER_EMBEDDING_DIMENSIONS = "1024"
```

3. 在你的 LLM 服务商后台设置消费上限——内置的 owner 配额（30 次/小时、200 次/天）在进程重启后重置。
4. 访客开启会话后，可在 Settings 页配置自己的 LLM、embedding 模型与 rerank 选项（仅保存在内存中）；不配置也可直接使用免费的本地哈希 embedding 进行检索。

`.streamlit/secrets.toml` 已被 git 忽略，切勿提交真实凭据。

## 开源协议

[GNU AGPL-3.0](../LICENSE) © 2026 Theo Zou
