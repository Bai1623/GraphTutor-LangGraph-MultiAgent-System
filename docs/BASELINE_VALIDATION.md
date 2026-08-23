# 本地运行基线验证

> 验证日期：2026-08-23
>
> 基线提交：`2c6b6d2`
>
> 环境：macOS x86_64、Python 3.11.16、Node.js 20.20.2、npm 10.8.2

本文记录新电脑首次恢复后的可复现运行基线。密钥、`.env`、索引文件和其他运行时数据不写入 Git。

## 验证结果

| 检查项 | 结果 | 证据 |
|---|---|---|
| Python 锁定依赖 | 通过 | `uv sync --extra dev --locked --python 3.11` |
| 前端锁定依赖 | 通过 | `npm ci` |
| 后端冷启动 | 通过 | 首次启动约 65 秒后监听 `127.0.0.1:8002` |
| 后端存活探针 | 通过 | `GET /healthz` 返回 200 和 `{"status":"ok"}` |
| 后端就绪探针 | 通过 | `GET /readyz` 返回 200，graph/settings/prompts/checkpointer 均为 `true` |
| 后端指标接口 | 通过 | `GET /metrics` 返回 200 和进程指标快照 |
| 前端生产构建 | 通过 | Next.js 16.1.6 完成静态页面构建 |
| 前端生产启动 | 通过 | 首页返回 200，页面标题为“高考辅导 AI 助手” |
| 知识文件解析 | 通过 | 数学 12、语文 62、英语 14，共 88 个 chunk |
| 向量索引构建 | 阻塞 | 本机未配置 `SILICONFLOW_API_KEY`，无法调用 BGE-M3 embedding |
| 在线问答与规划 HIL | 阻塞 | 本机未配置 DeepSeek/SiliconFlow API 密钥 |

## 可复现命令

后端无状态启动和探针：

```bash
AUTH_SECRET=<local-secret> OTEL_TRACING_ENABLED=false \
  uv run uvicorn app:app --host 127.0.0.1 --port 8002

curl http://127.0.0.1:8002/healthz
curl http://127.0.0.1:8002/readyz
curl http://127.0.0.1:8002/metrics
```

前端生产构建和启动：

```bash
cd frontend
nvm use
NEXT_PUBLIC_API_URL=http://localhost:8002 npm run build
npm run start -- --hostname 127.0.0.1 --port 3000
```

知识文件本地解析不依赖 API；完整索引构建需要先在 `.env` 配置 `SILICONFLOW_API_KEY`：

```bash
uv run python scripts/build_index.py
```

## 已识别的工程缺口

1. `/readyz` 当前只检查 graph、settings、prompts 和 checkpointer，不检查 LLM 密钥与向量索引是否可用。因此本机缺少密钥和索引时仍会返回 ready。
2. 首次后端冷启动约 65 秒，需要后续拆分导入耗时并建立启动耗时基线。
3. 当前电脑的用户目录存在额外 `package-lock.json`，Next.js 会提示 workspace root 推断警告；仓库自身仍只使用 `frontend/package-lock.json`。
4. `output: "standalone"` 配置下执行 `next start` 会产生提示，Docker/standalone 路径应单独验证。

## 完成剩余基线的条件

从密码管理器或旧电脑恢复 `.env` 后，依次执行索引构建、学术问答、规划 HIL 和上传解析验证。任何真实密钥都不得写入本文档或提交到 Git。
