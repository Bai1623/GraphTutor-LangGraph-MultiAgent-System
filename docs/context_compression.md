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
- `student_state`
- `constraints`
- `decisions`
- `knowledge_progress`
- `open_loops`

后续建议把 `artifact_refs`、`current_documents`、`current_questions` 纳入 `SessionEpisode`。

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

## 后续改造建议

1. 增加 artifact 恢复工具：按 `artifact_id`、题号、页码恢复完整内容。
2. 把 `SessionEpisode` 扩展为高考场景 schema，显式保存 `artifact_refs`。
3. 增加按节点的读时投影：`build_node_context(state, node_name)`。
4. 给压缩机制增加 harness：token 降幅、约束保留率、artifact 可恢复率、回答一致性。
5. 对 Agent 工具循环结果也接入 `ContextArtifactStore`，避免 ToolMessage 撑爆上下文。
