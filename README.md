# TCM Neo4j Agent

面向生产的中医知识图谱智能体服务，融合大模型推理、知识图谱与结构化数据库检索能力，专注“文本解析 + 证候推荐”的闭环能力。  
强调可扩展、低延迟与稳定性，适合作为中医 AI 业务的后端核心组件。

## 技术栈与能力亮点
- **FastAPI**：高并发、高性能的 API 服务框架
- **LangChain + LangGraph**：多步推理流程编排（解析、筛选、重排）
- **Neo4j**：图谱节点与关系检索
- **MySQL**：基础元数据与证候库支撑
- **OpenAI-compatible LLM**：可替换模型，支持私有部署与兼容 API

关键能力：
- **结构化解析**：从文本中抽取方剂、证型、症状等实体
- **证候推荐**：两段式候选筛选 + 重排，提升准确度
- **低延迟策略**：缓存元数据、工具与候选集，减少重复数据库调用
- **可扩展**：通过数据库配置动态生成工具与标签映射

## 启动方式
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8183 --log-level debug
```

## 安装依赖
```bash
pip install -r requirement.txt
```

## 环境变量（示例）
建议在 `.env` 中配置：
```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DB=sci_manage_system

OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=your_key
OPENAI_MODEL=deepseek-chat
```

可选优化参数（缓存与策略）：
```
REL_TYPES_CACHE_TTL_SECONDS=300
NODE_TYPES_CACHE_TTL_SECONDS=300
TOOLS_CACHE_TTL_SECONDS=300
NODE_LIST_CACHE_TTL_SECONDS=300
SYNDROME_CANDIDATE_CACHE_TTL_SECONDS=300
AGENT_CACHE_TTL_SECONDS=300
ENABLE_EXTRA_RELATIONS=true
```

## 接口概览
- `POST /nlp/parse` 文本解析
- `POST /tcm/syndrome/recommend`
- `POST /tcm/symptom/recommend`
 证候推荐

## 接口示例
`POST /nlp/parse`
```json
{
  "text": "患者头晕乏力，方用补中益气汤，考虑气虚证。",
  "limit": 120
}
```

`POST /tcm/syndrome/recommend`
```json
{
  "text": "头晕乏力，纳差，气短"
}
```

`POST /tcm/symptom/recommend`
```json
{
  "text": "ͷ�η������ɲ����",
  "candidateLimit": 2000,
  "maxReturn": 50
}
```
