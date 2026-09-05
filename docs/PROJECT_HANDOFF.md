# GraphTutor 项目交接文档

> 更新时间：2026-08-31
> 当前来源机器路径：`/Users/a2022-710/Desktop/cc/ LangGraph`（目录名前有一个空格，仅代表本机路径）
> 当前分支：`master`
> 当前远端：`origin=git@gitee.com:git_bai/work.git`，`github=git@github.com:Bai1623/GraphTutor-LangGraph-MultiAgent-System.git`
> 当前业务代码基线：`3e5b2d9 feat: validate golden evaluation schemas`

本文档用于换电脑后快速恢复开发环境，并说明项目当前进度、关键模块、验证命令和后续注意事项。换电脑当天只想按步骤核对时，先看 [`NEW_MACHINE_CHECKLIST.md`](NEW_MACHINE_CHECKLIST.md)。

新电脑接手时以 Gitee `origin/master` 的最新提交为唯一准线，不要照搬来源机器的绝对路径。当前来源机器没有 `.env` 和向量索引；密钥必须从密码管理器或旧开发环境单独迁移，不能通过 Git 传递。

## 1. 当前状态

项目名：GraphTutor，基于 LangGraph 的多智能体 RAG 高考辅导系统。

当前代码主线已完成 v0.3.0 级别功能：

- 后端 FastAPI + LangGraph 19 节点多智能体工作流。
- 学术问答分支：Supervisor 路由、并行 RAG + Web 检索、答案生成、幻觉评估、查询改写重试。
- 学习规划分支：政策搜索、情报收集、计划起草、双审查员、共识检查、HIL 人工审批、反馈路由、计划微调。
- 情绪支持分支：独立情绪支持 Agent。
- 前端 Next.js 16 + React 19 + React Flow，支持聊天、SSE 流式展示、DAG 节点状态、计划审阅、自然语言反馈和 Markdown 下载。
- RAG 管线：ChromaDB / NumPy fallback、BM25、BGE reranker、中文试卷章节感知分块。
- 文档解析和 OCR：上传 PDF/DOCX/图片后走可选 MCP 或本地 fallback，并将大结果落盘为 artifact。
- 上下文压缩：RAG/Web/政策/文档解析结果落盘，state 只保留 preview 和 artifact 引用；会话超预算后压缩为结构化摘要。
- 生产基础设施：登录、限流、quota、上传安全、OpenTelemetry、health/readiness/metrics、Dockerfile、docker-compose、CI。

最近完成的工程化推进：

- `BASE-01`：用 `.python-version=3.11`、`.nvmrc=20` 固定本地运行时主版本。
- `BASE-02`：完成新电脑后端、前端和探针运行基线，记录在 [`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md)。
- `BASE-03`：增加 `scripts/project_doctor.py`，可在不输出密钥内容的前提下检查运行时、锁文件、知识数据、密钥状态和向量索引。
- `BASE-04`：增加 `scripts/run_baseline.py`，一键执行环境检查、后端测试/覆盖率、Ruff、Mypy、前端 lint/typecheck/build，并生成忽略提交的本地报告。
- `EVAL-01`：为 routing、RAG、hallucination、planning、quality gate、context compression 六类 Golden Dataset 增加统一 Pydantic schema 校验，并接入两个评测入口。缺字段、错误类型、非法阈值和重复 case ID 会在评测启动前失败。
- `EVAL-02`：为六类 Golden Dataset 增加数据集名称、语义化版本、来源、更新时间和研究目标；评测 JSON/Markdown 报告同步记录完整元数据，并补充版本约定文档和回归测试。

下一项按既定顺序推进 `EVAL-03`：增加 Golden Dataset 数据分布和覆盖率报告，按 subject、difficulty、query type 等维度暴露样本空白。每次只做一个小推进，完成验证后再提交；是否推送以当次用户指令为准。

最近关键提交：

```text
3e5b2d9 feat: validate golden evaluation schemas
51284fe feat: add reproducible baseline report
09ad072 feat: add local environment doctor
798d493 docs: remove baseline markdown whitespace
f5f8c23 docs: record new machine runtime baseline
2c6b6d2 chore: pin local runtime versions
686938e docs: add new machine setup checklist
552b9cd docs: add project handoff guide
```

写入本文档前，`master` 与 `origin/master` 同步，无未提交业务代码改动。换电脑时以远端 `origin/master` 最新提交为准。

## 2. 换电脑恢复步骤

### 2.1 克隆代码

已经在 Gitee 配置 SSH key 时，优先使用当前实际推送协议：

```bash
ssh -T git@gitee.com
git clone git@gitee.com:git_bai/work.git
cd work
git status --short --branch
```

新电脑暂未配置 Gitee SSH key 时，可以先用 HTTPS：

```bash
git clone https://gitee.com/git_bai/work.git
cd work
```

Gitee 克隆完成后，如需保留 GitHub 镜像远端：

```bash
git remote add github git@github.com:Bai1623/GraphTutor-LangGraph-MultiAgent-System.git
git fetch github
git remote -v
```

日常默认推送目标是 `origin`（Gitee）；不要在没有明确要求时把未同步的提交单独推到 GitHub。

### 2.2 准备后端环境

要求：

- Python 3.11+
- uv
- 可选 PostgreSQL
- DeepSeek API Key
- SiliconFlow API Key

安装和同步：

```bash
python -m pip install --upgrade pip
python -m pip install uv
python -m uv sync --extra dev --locked
```

如果新电脑默认 `python` 不是 3.11+，指定解释器：

```bash
python -m uv sync --extra dev --locked --python 3.11
```

### 2.3 准备环境变量

不要把旧电脑的 `.env` 提交到 Git。换机时手动复制或重新创建：

```bash
cp .env.example .env
```

最低必填：

```bash
DEEPSEEK_API_KEY=...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

SILICONFLOW_API_KEY=...

FALLBACK_MODEL=Qwen/Qwen2.5-7B-Instruct
FALLBACK_API_KEY=...
FALLBACK_BASE_URL=https://api.siliconflow.cn/v1

AUTH_USERNAME=admin
AUTH_PASSWORD=123456
AUTH_SECRET=replace_with_a_long_random_string
AUTH_COOKIE_SECURE=false

ALLOWED_ORIGINS=http://localhost:3000
CHROMA_PERSIST_DIR=chroma_store/
```

可选项：

- `POLICY_MCP_URL` / `POLICY_MCP_COMMAND`：官方政策或招生数据 MCP。
- `DOCUMENT_MCP_URL` / `DOCUMENT_MCP_COMMAND`：PDF/DOCX/图片解析 MCP。
- `OCR_API_KEY`、`OCR_BASE_URL`、`OCR_MODEL`：图片 OCR。
- `DB_URI`：PostgreSQL LangGraph checkpointer；不设置时系统会降级为无状态运行。
- `UPLOAD_AV_COMMAND`：生产环境接入杀毒扫描命令，例如 `clamscan --no-summary`。

### 2.4 构建知识库索引

Git 中只跟踪原始知识文件，`chroma_store/` 不跟踪。换电脑后需要重建索引：

```bash
python -m uv run python scripts/build_index.py
```

当前跟踪的知识库包括：

- `data/chinese/`：2024、2025 高考语文试卷文本。
- `data/math/`：2025 高考数学知识点文本。
- `data/english/`：2025 高考英语知识点文本。

本地运行产物不需要迁移，除非你特别想保留历史调试数据：

- `chroma_store/`
- `data/context_artifacts/`
- `data/audit/`
- `data/quota/`
- `logs/`
- `artifacts/`
- `.pytest_cache/`
- `.venv/`

### 2.5 启动后端

本地开发推荐：

```bash
python -m uv run python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

健康检查：

```bash
curl http://127.0.0.1:8002/healthz
curl http://127.0.0.1:8002/readyz
curl http://127.0.0.1:8002/metrics
```

说明：

- `/healthz` 是存活探针，只要进程可响应即可。
- `/readyz` 会检查 graph、settings、prompts、checkpointer 状态。
- `/metrics` 返回进程内指标快照。

### 2.6 准备前端环境

要求：

- Node.js 20 推荐，至少 Node.js 18+
- npm

项目策略是前端统一使用 npm，不使用 pnpm：

```bash
cd frontend
npm ci
echo "NEXT_PUBLIC_API_URL=http://localhost:8002" > .env
npm run dev
```

浏览器访问：

```text
http://localhost:3000
```

注意：

- 只保留 `frontend/package-lock.json`。
- 不要提交 `frontend/pnpm-lock.yaml`。
- `frontend/next-env.d.ts` 可能会被 `next dev` / `next build` 自动改动；如果只是 `.next/dev/types` 与 `.next/types` 的路径切换，不属于业务改动。

### 2.7 Docker Compose

如果新电脑安装了 Docker：

```bash
docker compose up -d
docker compose --profile observability up -d
```

Compose 会启动：

- `postgres`
- `app`
- 可选 `jaeger`

当前 Dockerfile 已有容器级 `HEALTHCHECK`；`docker-compose.yml` 也给 `app` 配了 `/healthz` 健康检查。

## 3. 关键源码地图

后端入口：

- `app.py`：FastAPI app、登录中间件、限流、SSE、上传、任务队列、健康检查。
- `src/schemas.py`：请求和响应 Pydantic schema。

LangGraph：

- `src/graph/builder.py`：19 节点 StateGraph 构建。
- `src/graph/state.py`：`TutorState` 和 `context_reducer`。
- `src/graph/supervisor.py`：意图分类，输出 `academic` / `planning` / `emotional` / `unknown`。
- `src/graph/academic.py`：学术问答分支，RAG/Web 并行、答案生成、幻觉评估、改写重试。
- `src/graph/planner.py`：政策搜索和学习规划情报收集。
- `src/graph/plan_adversarial.py`：计划起草、双审查、共识、HIL、反馈路由、微调。
- `src/graph/emotional.py`：情绪支持。
- `src/graph/llm.py`：节点级 LLM 工厂、流式输出、fallback。

RAG：

- `src/rag/loader.py`：加载 PDF/MD/TXT。
- `src/rag/indexer.py`：构建向量索引。
- `src/rag/retriever.py`：向量检索 + BM25 + reranker。
- `src/rag/section_splitter.py`：中文试卷章节感知分块。
- `src/rag/simple_store.py`：无 onnxruntime 时的 NumPy fallback。
- `src/rag/reranker.py`：SiliconFlow BGE reranker。

上下文和记忆：

- `src/memory/artifacts.py`：完整大结果落盘。
- `src/memory/compressor.py`：多轮对话压缩。
- `src/memory/context_builder.py`：构建记忆上下文。
- `src/memory/long_term.py`：长期记忆 JSON store。
- `src/memory/extractor.py`：长期记忆提取。
- `docs/context_compression.md`：上下文压缩设计说明。

工具：

- `src/tools/search_tool.py`：DuckDuckGo search wrapper。
- `src/tools/policy_search.py`：官方政策 MCP / Web fallback。
- `src/tools/document_question_parser.py`：PDF/DOCX/图片题目解析。
- `src/tools/ocr_tool.py`：OCR query 构建。

安全和运行：

- `src/auth.py`：签名 cookie 登录。
- `src/middleware/rate_limit.py`：Token bucket 限流。
- `src/quota.py`：每日用户 quota。
- `src/security/upload_security.py`：上传安全校验、静态恶意内容拦截、可选 AV。
- `src/task_queue.py`：长任务后台队列。

前端：

- `frontend/app/page.tsx`：主页面、SSE 消费、HIL resume、反馈流转。
- `frontend/components/chat-area.tsx`：聊天区域和上传文件入口。
- `frontend/components/plan-review.tsx`：计划草稿审阅、反馈、下载。
- `frontend/components/right-panel.tsx`：节点轨迹、React Flow DAG、日志、Token 用量。
- `frontend/components/left-sidebar.tsx`：会话历史。

配置：

- `config/settings.yaml`：温度、超时、重试次数、RAG 参数、memory 参数。
- `config/prompts/*.xml`：提示词模板。
- `.env.example`：环境变量模板。
- `frontend/.env.example`：前端环境变量模板。

评测与工程基线：

- `src/evaluation/golden_dataset.py`：六类 Golden Dataset 的严格 schema、阈值和 case ID 校验。
- `scripts/run_eval.py`：routing、RAG、hallucination、planning、quality gate 统一评测入口。
- `scripts/run_compression_harness.py`：上下文压缩离线/在线评测入口。
- `scripts/project_doctor.py`：新电脑环境自检，不输出密钥值。
- `scripts/run_baseline.py`：可复现的全量离线工程基线。

测试：

- `tests/test_integration.py` 是手动 live integration harness，不是普通 pytest 单测。
- `tests/conftest.py` 已配置默认 pytest 忽略 `test_integration.py`，并隔离 rate limiter 全局状态。

## 4. 当前 Graph 拓扑

核心分支：

```text
supervisor
├── academic_router
│   ├── rag_retrieve
│   └── web_search
│       ↓ fan-in
│   generate_answer
│   ↓
│   evaluate_hallucination
│   ├── end
│   └── rewrite_query → academic_router
├── search_policy
│   ↓
│   gather_intel
│   ↓
│   drafter
│   ├── reviewer_academic
│   └── reviewer_emotional
│       ↓ fan-in
│   consensus_check
│   ├── plan_output
│   │   ├── confirm → end
│   │   └── feedback → feedback_router
│   │       ├── tweak → plan_tweak → plan_output
│   │       └── rewrite → drafter
│   └── adv_rewrite → drafter
├── emotional_response
└── handle_unknown
```

Graph 节点数：19。

## 5. 重要接口

认证：

- `POST /auth/login`
- `GET /auth/me`
- `POST /auth/logout`

对话：

- `POST /stream`：开始一次图执行，返回 SSE。
- `POST /resume`：恢复 HIL 中断，支持确认计划或自然语言反馈。

文档：

- `POST /documents/parse`：上传 PDF/DOCX/图片并解析题目。
- `POST /ocr`：OCR 图片识别。
- `GET /tasks/{task_id}`：查询后台任务状态。

反馈：

- `POST /feedback`：对回答点赞/点踩，写入 JSONL。

运行状态：

- `GET /healthz`
- `GET /readyz`
- `GET /metrics`
- `GET /quota/me`

SSE 事件：

- `thread_id`
- `node_event`
- `token`
- `text`
- `usage`
- `interrupt`
- `done`
- `error`

## 6. 验证命令

换机后优先运行一键基线：

```bash
python -m uv run python scripts/run_baseline.py
```

该命令依次执行环境 doctor、后端测试与覆盖率、Ruff、Mypy、前端 lint/typecheck/build。报告写入被 Git 忽略的 `artifacts/baseline/`。

后端单测：

```bash
OTEL_TRACING_ENABLED=false python -m uv run pytest
```

CI 风格后端验证：

```bash
OTEL_TRACING_ENABLED=false python -m uv run pytest tests/ --ignore=tests/test_integration.py --cov --cov-report=term-missing -v --tb=short
python -m uv run ruff check .
python -m uv run mypy
```

前端验证：

```bash
cd frontend
npm run lint
npm run typecheck
npm run build
```

安全专项：

```bash
OTEL_TRACING_ENABLED=false python -m uv run pytest tests/test_security.py tests/test_upload_security.py tests/test_quota.py -v --tb=short
```

压缩 harness：

```bash
python -m uv run python scripts/run_compression_harness.py --output artifacts/eval
```

评测 harness：

```bash
python -m uv run python scripts/run_eval.py --suite quality_gate --output artifacts/eval/
python -m uv run python scripts/run_eval.py --suite rag --output artifacts/eval/
python -m uv run python scripts/run_eval.py --suite routing --output artifacts/eval/
python -m uv run python scripts/run_eval.py --suite hallucination --output artifacts/eval/
python -m uv run python scripts/run_eval.py --suite planning --output artifacts/eval/
```

手动 live integration：

```bash
python -m tests.test_integration --quick
```

这个脚本需要真实 `.env`、真实 API key、知识库索引，不能当普通 mock 单测跑。

最近一次本地验证记录：

```text
基线日期：2026-08-31
业务代码基线：3e5b2d9
完整基线：7/7 steps passed
后端：532 passed, 1 skipped, 2 warnings
覆盖率：78.93%（门槛 70%）
Ruff：通过
Mypy：54 source files 无错误
前端 lint/typecheck/build：通过
Golden Dataset：6/6 schema 校验通过
```

基线 warning 仍包括 `langchain-community` sunset、Starlette/httpx deprecation、用户目录额外 `package-lock.json` 导致的 Next workspace root 推断提示。这些不是本轮失败，但应在后续依赖治理中处理。

## 7. CI 当前规则

`.github/workflows/ci.yml` 包含四个 job：

- `unit-tests`：Python 3.11 / 3.12 / 3.13 矩阵，uv locked sync，ruff，mypy，pytest coverage。
- `security-audit`：安全、上传、quota 测试。
- `frontend-build`：npm lockfile 策略、npm ci、lint、typecheck、build。
- `docker-build`：Docker Buildx 构建镜像，不 push。

注意：CI 仍显式使用 `--ignore=tests/test_integration.py`。本地 `tests/conftest.py` 也做了忽略，防止直接 `pytest` 时误收集手动 live harness。

## 8. 换机需要迁移和不需要迁移的东西

必须迁移或重建：

- Git 仓库代码。
- `.env` 中的 API key 和认证配置，建议通过密码管理器复制，不要提交。
- 前端 `.env` 或 `.env.local`，至少配置 `NEXT_PUBLIC_API_URL`。
- 重新运行 `scripts/build_index.py` 生成 `chroma_store/`。

可以不迁移：

- `.venv/`
- `frontend/node_modules/`
- `frontend/.next/`
- `.pytest_cache/`
- `.coverage`
- `logs/`
- `artifacts/`
- `data/context_artifacts/`
- `data/audit/`
- `data/quota/`

按需迁移：

- 如果你想保留旧电脑上真实对话产生的大结果引用，复制 `data/context_artifacts/`。
- 如果你想保留上传审计记录，复制 `data/audit/`。
- 如果你想保留每日 quota 计数，复制 `data/quota/`。
- 如果想避免重新构建索引，可以复制 `chroma_store/`，但通常重建更干净。

## 9. 常见坑

### Python 版本

项目要求 Python 3.11+。macOS 系统自带 Python 可能是 3.9，不要直接用它跑项目。

推荐：

```bash
python -m uv sync --extra dev --locked --python 3.11
```

### uv 命令

README 中有些命令写成：

```bash
python -m uv run python ...
```

如果新电脑装了独立 `uv`，也可以写成：

```bash
uv run python ...
```

以实际可用命令为准。

### 测试 warning

目前普通测试仍会看到外部依赖 warning：

- `langchain-community` sunset warning。
- `fastapi.testclient` / Starlette 关于 `httpx` 的 deprecation warning。

这些是依赖栈 warning，不是当前业务失败。

### `test_integration.py`

这是手动脚本，需要真实 API 和 graph，不要纳入普通单测。直接跑普通 `pytest` 时已经被忽略。

### rate limiter

生产代码中 rate limiter 是进程级单例。测试里通过 autouse fixture 重置，避免 TestClient 请求跨用例污染。不要把这个 fixture 删除，否则认证测试可能因为前面用例消耗令牌而得到 429。

### Next 自动生成文件

`frontend/next-env.d.ts` 由 Next 自动维护。`npm run dev` 和 `npm run build` 可能让它在以下两种 import 之间切换：

```ts
import "./.next/dev/types/routes.d.ts";
import "./.next/types/routes.d.ts";
```

如果只是这个变化，通常不要作为业务改动提交。

### 运行时数据

`.gitignore` 已忽略：

- `chroma_store/`
- `data/context_artifacts/`
- `data/audit/`
- `data/quota/`
- `logs/`
- `artifacts/`

换机后看不到这些目录是正常的。

## 10. 后续推进顺序

下一台电脑首先完成环境闭环：

1. 克隆 Gitee `master`，运行 `scripts/project_doctor.py`。
2. 从安全渠道恢复 `.env`，构建 `chroma_store/`。
3. 运行 `scripts/run_baseline.py`，确认离线基线没有因换机变化。
4. 实测后端、前端、登录、学术问答、规划 HIL 和上传解析。
5. 如有 Docker，补做 `docker compose up -d` 的本机验证。

环境闭环后按综合型科研工程路线继续：

1. `EVAL-03`：数据分布和覆盖率报告，按 subject、difficulty、query type 等维度暴露空白。
2. `EVAL-04`：离线回归门禁，把不依赖密钥的 schema/静态评测接入 CI。
3. `EXP-01`：实验配置快照和随机种子，保证消融实验可复现。
4. `EXP-02`：统一实验结果目录和对比报告，支持 baseline/variant 横向比较。
5. `OBS-01`：完善 token、延迟、重试、fallback、工具轮次等成本指标的实验聚合。
6. `ENG-01`：处理外部依赖 deprecation warning 和 Next workspace root 警告。
7. `E2E-01`：为 `/stream`、`/resume`、上传解析增加真实浏览器路径回归。

其他已知工程建议：

- 完善 `docs/context_compression.md` 中的 artifact 恢复工具，支持按 `artifact_id`、题号、页码恢复完整内容。
- 公网部署前修改 `AUTH_PASSWORD`、生成强 `AUTH_SECRET`，并设置 `AUTH_COOKIE_SECURE=true`。
- 多实例部署时将 rate limiter 和 quota 迁移到 Redis 或 PostgreSQL。
- 给 Docker Compose 增加 `.env` 缺失时的友好提示。
- 增加按节点的读时投影 `build_node_context(state, node_name)`，减少无关上下文。

暂不建议：

- 不要把 `data/context_artifacts/`、`chroma_store/` 直接纳入 Git。
- 不要把 `.env`、API key、真实用户上传文件提交。
- 不要同时混用 npm、pnpm、yarn 管理前端依赖。

## 11. 换机后最短自检流程

```bash
# 1. 克隆并进入仓库
git clone git@gitee.com:git_bai/work.git
cd work
git status --short --branch

# 2. 后端依赖
python -m pip install uv
python -m uv sync --extra dev --locked --python 3.11

# 3. 无密钥环境检查
python -m uv run python scripts/project_doctor.py

# 4. 环境变量
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY / SILICONFLOW_API_KEY / AUTH_SECRET 等

# 5. 构建索引
python -m uv run python scripts/build_index.py

# 6. 完整离线基线
python -m uv run python scripts/run_baseline.py

# 7. 启动后端
python -m uv run python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

另一个终端：

```bash
cd frontend
nvm use
npm ci
echo "NEXT_PUBLIC_API_URL=http://localhost:8002" > .env
npm run dev
```

打开：

```text
http://localhost:3000
```

登录默认账号密码：

```text
admin / 123456
```

本地开发可以使用默认密码；公网部署必须修改。
