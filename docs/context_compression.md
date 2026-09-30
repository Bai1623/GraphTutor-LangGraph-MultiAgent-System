# 上下文压缩机制说明书

本文档记录本项目的上下文压缩设计。后续每次修改上下文、记忆、RAG、工具调用、文档解析或 prompt 裁剪逻辑时，都应同步补充本文档。

## 目标

高考导师系统的上下文压力主要来自多轮对话、RAG 检索、Web/政策搜索、文档解析、OCR、Agent 工具循环。压缩机制的目标不是简单摘要，而是按信息类型分流：

- 可恢复的大结果落盘，只把预览和引用放入 state/prompt。
- 对话历史超过预算后再结构化摘要。
- 长期稳定事实进入长期记忆。
- 不可恢复的用户约束、当前任务、关键结论优先保留。

## 当前层级

### 1. 大结果落盘

新增 `src/memory/artifacts.py`，提供 `ContextArtifactStore`。

完整 payload 写入：

```text
data/context_artifacts/<YYYY-MM-DD>/<artifact_id>.json
```

运行产物由 `.gitignore` 忽略，不进入版本库。

state/prompt 中只保留：

```json
{
  "artifact_id": "ctx_xxx",
  "preview": "短预览文本",
  "artifact_ref": {
    "artifact_id": "ctx_xxx",
    "kind": "rag_retrieval_doc",
    "path": "...",
    "preview": "...",
    "stats": {"chars": 1234},
    "created_at": "..."
  },
  "recoverable": true
}
```

### 2. RAG/Web context 瘦身

`src/graph/academic.py` 中：

- `rag_retrieve` 会把每条 RAG 文档完整内容落盘为 `rag_retrieval_doc`。
- `web_search` 会把每条 Web 搜索结果落盘为 `web_search_result`。
- `state["context"]` 中的 `content` 改为 preview，同时保留 `artifact_id` 和 `artifact_ref`。
- `_format_retrieved` / `_format_search` 会把 artifact id 放入提示词，方便后续追踪。

### 3. 政策搜索结果瘦身

`src/graph/planner.py` 中：

- 官方 MCP 政策结果落盘为 `official_policy_result`。
- DuckDuckGo fallback 政策结果落盘为 `policy_web_fallback_result`。
- `state["search_results"]` 只保留结构化字段、preview、artifact 引用。

`src/tools/policy_search.py` 的 `format_policy_results` 会在 prompt 中带上 artifact id。

### 4. 文档解析 artifact

`src/tools/document_question_parser.py` 中：

- PDF/DOCX/图片解析后的完整 `questions + recognized_text` 落盘为 `document_parse`。
- `/documents/parse` 响应新增 `artifact_id`、`preview`、`artifacts`。
- `recognized_text` 返回预览，避免大文档通过接口和前端再次进入模型上下文。
- `query` 只包含题目结构预览、识别内容预览、用户补充问题、artifact id。

这解决了“上传 PDF 后整份内容直接作为用户输入给 AI”的问题：AI 先看到可读预览和用户问题，完整原文留在 artifact 中等待后续按需恢复。

### 5. 会话摘要

已有 `src/memory/compressor.py`：

- 估算消息 token。
- 超过 `memory.soft_limit_tokens` 后触发压缩。
- 保留最近 `memory.recent_turns` 轮完整对话。
- 更早对话合并成 `SessionEpisode` 结构化摘要。

当前摘要结构保留：

- `task`
- `gaokao_state`
- `artifact_refs`
- `current_documents`
- `current_questions`
- `student_state`
- `constraints`
- `decisions`
- `knowledge_progress`
- `open_loops`

`gaokao_state` 专门保存高考场景中的年级、省份、选科/方向、目标分、目标院校、科目目标、薄弱点、近期成绩和学习偏好。

`artifact_refs`、`current_documents`、`current_questions` 用来保证压缩后仍能找回上传 PDF、OCR、政策搜索、RAG 文档等完整内容。摘要 prompt 明确要求不得改写 `artifact_id`。

### 6. 长期记忆

已有 `src/memory/long_term.py` 和 `src/memory/extractor.py`：

- 对话结束后提取可跨会话复用的学生画像、学习进展、短期 episode。
- 通过 JSON store 持久化。
- 下一轮按 query/intent/subject 召回相关事实。

## 编码门禁

`tests/test_encoding.py` 已将 `src/memory/` 纳入 mojibake 扫描，防止压缩 prompt、摘要标签、长期记忆提示词再次乱码。

当前扫描异常字符包括：

- `锛`
- `鈥`
- `骞`
- `妫`
- `绱`
- `�`

## Compression Harness

新增 `src/memory/compression_harness.py` 和 `scripts/run_compression_harness.py`，用于离线评估压缩质量。

默认 golden suite：

```text
eval/golden/compression.yaml
```

运行：

```bash
python -m uv run python scripts/run_compression_harness.py --output artifacts/eval
```

默认模式是 `offline_static_episode`，不调用真实 LLM。每个 case 提供 `expected_episode`，harness 用它构造压缩后的上下文，然后计算：

- `token_reduction`：压缩前后 token 降幅。
- `constraint_retention`：用户硬约束保留率。
- `answer_consistency`：答案关键项在压缩前后是否仍可见。
- `artifact_recoverability`：摘要中的 artifact id 是否存在且能从 artifact store 恢复。

如果后续要评估真实摘要器，可使用：

```bash
python -m uv run python scripts/run_compression_harness.py --use-llm --output artifacts/eval
```

当前 golden case 覆盖：

- 上传试卷后继续追问知识点。
- 志愿规划中保留官方政策 artifact 和预算/公办约束。

## 按节点读取上下文

新增 `build_node_context(state, node_name)` 读时投影层，并接入 Supervisor、情绪支持、计划起草和学术回答节点。每个节点通过字段白名单只读取完成职责所需的摘要：

- Supervisor 读取任务、学生基本画像、薄弱点和待处理事项，不读取文档正文引用。
- 情绪支持读取学生状态、学习偏好和约束，不读取题目与 artifact。
- 计划起草读取目标、成绩、约束、结论和学习进展，不读取文档 artifact。
- 学术回答读取题目、知识进展及可恢复 artifact，不读取无关情绪状态。

长期记忆仍按召回结果注入。未注册节点默认使用完整压缩记忆；历史自由文本摘要无法可靠拆分时也完整保留，避免兼容升级造成信息丢失。

## Agent 工具结果瘦身

Academic Agent 的 `search_knowledge_base` 和 `search_web` 工具结果超过 `academic.tool_result_compaction_threshold`（默认 1400 字符）时，会写入 `agent_tool_result` artifact。后续 `ToolMessage` 只保留预览、artifact id 和完整长度；模型需要细节时可继续调用 `recover_context_artifact`。

短结果直接内联，避免无意义的磁盘写入；`recover_context_artifact` 的结果本身已经有最大恢复长度限制，因此不再次压缩，保证按需恢复真正能提供细节。

## 后续改造建议

1. 已增加 artifact 恢复工具：`recover_context_artifact` 会校验 `artifact_id`，并可按题号或页码恢复有界内容；Academic Agent 可在预览不足时自主调用。
2. 已增加按节点的读时投影：`build_node_context(state, node_name)`，四个提示词消费节点已使用字段白名单。
3. 已对 Agent 工具循环的大结果接入 `ContextArtifactStore`，短结果内联，恢复结果保留既有边界。
4. 扩展 compression harness 的 live LLM 模式，引入真实回答对比或 judge 评分。
