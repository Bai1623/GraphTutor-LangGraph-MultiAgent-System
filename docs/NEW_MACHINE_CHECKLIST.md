# 新电脑开发恢复清单

这份清单用于换电脑当天快速核对。完整背景和模块说明见 [`PROJECT_HANDOFF.md`](PROJECT_HANDOFF.md)。

最近一次新电脑实测结果见 [`BASELINE_VALIDATION.md`](BASELINE_VALIDATION.md)。

## 1. 拉取代码

```bash
git clone https://gitee.com/git_bai/work.git
cd work
git status --short --branch
```

确认分支显示为：

```text
## master...origin/master
```

## 2. 准备本地密钥

从旧电脑或密码管理器恢复 `.env`，不要提交到 Git。

最低需要确认：

- `DEEPSEEK_API_KEY`
- `SILICONFLOW_API_KEY`
- `FALLBACK_API_KEY`
- `AUTH_SECRET`
- `AUTH_USERNAME`
- `AUTH_PASSWORD`
- `ALLOWED_ORIGINS=http://localhost:3000`
- `CHROMA_PERSIST_DIR=chroma_store/`

如果没有旧 `.env`：

```bash
cp .env.example .env
```

然后手动补齐真实值。

## 3. 后端环境

```bash
python -m pip install uv
python -m uv sync --extra dev --locked --python 3.11
python -m uv run python scripts/build_index.py
```

启动后端：

```bash
python -m uv run python -m uvicorn app:app --host 127.0.0.1 --port 8002
```

另开终端检查：

```bash
curl http://127.0.0.1:8002/healthz
curl http://127.0.0.1:8002/readyz
```

## 4. 前端环境

```bash
cd frontend
nvm use
npm ci
echo "NEXT_PUBLIC_API_URL=http://localhost:8002" > .env
npm run dev
```

访问：

```text
http://localhost:3000
```

## 5. 最小功能自检

- 登录默认账号：`admin / 123456`，公网部署前必须修改。
- 发一条学术问答，确认有流式回复和节点轨迹。
- 发一条规划类问题，确认进入计划审阅。
- 在计划审阅里提交一条自然语言反馈，确认可以继续生成计划。
- 上传一个小 PDF 或图片，确认解析任务能完成。

## 6. 不需要从旧电脑复制

- `.venv/`
- `frontend/node_modules/`
- `frontend/.next/`
- `.pytest_cache/`
- `.coverage`
- `logs/`
- `artifacts/`

## 7. 按需复制

- `chroma_store/`：通常重建即可，复制可节省索引时间。
- `data/context_artifacts/`：只在需要旧对话的大结果引用时复制。
- `data/audit/`：只在需要旧上传审计记录时复制。
- `data/quota/`：只在需要保留旧 quota 计数时复制。
