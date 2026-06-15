# GraphTutor 三层记忆压缩方案

## 1. 目标

当前项目已经具备：

- LangGraph Checkpointer 保存会话消息
- 最近 8 轮完整保留
- 旧消息 LLM 自由文本摘要
- 每个 `thread_id` 最多 20 条长期事实

下一步不应继续堆叠更多“层”，而应提高三件事：

1. 压缩是否真的减少 Checkpointer 中的消息；
2. 压缩后是否保留高考辅导任务所需的约束和证据；
3. 长期记忆是否能按用户、意图、时效正确召回，而不是全量注入。

建议采用三层渐进式架构：

```text
L1 Working Context  ->  L2 Session Episode  ->  L3 User Memory
原文工作集              结构化会话摘要           跨会话画像/事件
低成本、高频             LLM、低频触发            异步提取与按需召回
```

## 2. 当前实现的主要问题

### 2.1 消息可能没有真正删除

`TutorState.messages` 使用 `add_messages` reducer。压缩节点返回一个更短的
`messages` 列表时，Reducer 会按消息 ID 合并，而不是把旧列表整体替换。

因此当前实现可能只新增一条摘要消息，旧消息仍保留在 Checkpointer 中。正确做法是：

```python
[
    RemoveMessage(id=REMOVE_ALL_MESSAGES),
    SystemMessage(content=...),
    *recent_messages,
]
```

或者将“持久化原始日志”和“每轮装配给模型的上下文”拆成两个字段，后者使用覆盖语义。

### 2.2 固定轮数不是可靠预算

8 轮数学短问答可能只有 2,000 tokens，一轮长作文批改可能超过 8,000 tokens。
压缩触发应基于 token 预算，而不是消息数量或字符数。

### 2.3 压缩前已经丢信息

当前每条溢出消息只取 200 字，全部溢出内容再截到 2,000 字。长题干、作文原文、
学习计划和模型给出的关键步骤可能在进入摘要模型前就被截断。

### 2.4 摘要是自由文本，无法稳定更新

当前 400 字摘要混合了用户画像、当前任务、结论、情绪和计划约束。
多次“摘要的摘要”后容易出现：

- 数字漂移，例如目标分数、每天可用时间被改写；
- 当前情绪覆盖长期偏好；
- 已完成任务和待办事项混淆；
- 数学结论失去题号、条件或证据来源。

### 2.5 长期记忆并非真正跨会话

API 使用 `thread_id` 作为长期记忆 Key。新建会话会生成新的 `thread_id`，因此用户换会话后
无法复用历史画像。需要分离：

- `user_id`: 跨会话身份；
- `thread_id`: 单次会话或任务；
- `memory_id`: 单条记忆。

### 2.6 长期事实缺少更新、冲突和时效

当前只做字符串精确去重和 FIFO 淘汰。以下内容会同时存在：

- “每天可学习 3 小时”
- “现在每天只能学习 1 小时”
- “目标 600 分”
- “目标调整为 630 分”

高考政策、考试日期、模考成绩和情绪状态还具有不同的有效期，不能统一永久保存。

## 3. 推荐的三层压缩

## L1: Working Context

目标：不调用 LLM，快速保持本轮任务连续性。

保存内容：

- 当前用户消息，完整保留；
- 最近 3-6 个完整对话回合；
- 当前未完成任务所需的题干、作文、计划草案；
- 最近一次工具结果的精简版本；
- L2 摘要和本轮相关的 L3 记忆。

触发策略：

```text
soft_limit = 模型上下文的 60%
hard_limit = 模型上下文的 78%
reserved_output = 20%
```

达到 `soft_limit` 时执行确定性清理：

1. 删除重复的 RAG/Web 结果；
2. 工具结果只保留 top-N、标题、关键片段和引用；
3. 删除已经被后续回答覆盖的中间 Agent 输出；
4. 保留最近回合和所有 pinned 消息；
5. 仍超预算时进入 L2。

高考场景的 pinned 内容：

- 当前题目原文、选项、作文原文；
- 用户明确给出的分数、时间、目标院校；
- 当前学习计划的硬约束；
- HIL 尚未确认的计划草案和最新反馈；
- 安全相关的情绪风险信号。

不要把 pinned 内容只放进摘要。可将长原文存到外部对象存储或数据库，
上下文中保存 `artifact_id + 标题 + 摘要`，需要时再读取。

## L2: Session Episode

目标：把已经离开工作窗口的多个回合压缩为可更新、可验证的“会话事件”。

不要生成一段 400 字自由文本，改为结构化数据：

```json
{
  "task": {
    "intent": "academic|planning|emotional",
    "subject": "math",
    "topic": "导数单调性",
    "status": "active|blocked|completed"
  },
  "student_state": {
    "current_difficulty": ["含参讨论不会分类"],
    "emotion": "焦虑但可继续学习"
  },
  "constraints": [
    {"key": "daily_minutes", "value": 120, "source_message_id": "m12"}
  ],
  "decisions": [
    {"text": "先补导数分类讨论，再做压轴题", "source_message_id": "m18"}
  ],
  "knowledge_progress": [
    {
      "knowledge_point": "导数判断单调性",
      "mastery": 0.45,
      "evidence": "两次把定义域遗漏",
      "source_message_ids": ["m8", "m16"]
    }
  ],
  "open_loops": [
    {"text": "明天检查错题第 3 题", "due_at": "2026-06-13"}
  ],
  "artifacts": [
    {"id": "essay_01", "type": "essay", "summary": "议论文初稿"}
  ]
}
```

压缩规则：

1. 数字、日期、题号、公式、用户原话约束优先复制，不允许润色；
2. 每个结论带 `source_message_id`，支持回查；
3. 新摘要与旧摘要按字段合并，不做整段重写；
4. `open_loops` 完成后转为 `completed`，下次压缩可删除；
5. 最近一次压缩后只处理新增消息，避免重复摘要；
6. 每 3-5 次增量压缩做一次重建，防止摘要漂移。

建议触发：

- L1 清理后仍超过 `soft_limit`；
- 未压缩旧消息超过 4,000 tokens；
- 规划/HIL 分支完成一个阶段；
- 当前任务从 academic 切换到 planning 等新意图。

## L3: User Memory

目标：跨会话保存稳定画像、学习进展和重要事件，并按当前请求召回。

建议拆为三种记忆类型，但仍属于同一层：

```text
profile     稳定画像：年级、选科、偏好、长期目标
progress    学习进展：知识点掌握度、典型错误、近期模考
episode     重要事件：制定过的计划、承诺的复盘、明显情绪变化
```

单条记忆建议字段：

```json
{
  "memory_id": "mem_xxx",
  "user_id": "user_xxx",
  "type": "profile|progress|episode",
  "subject": "math",
  "topic": "derivative",
  "content": "含参导数分类讨论薄弱",
  "value": {"mastery": 0.45},
  "confidence": 0.88,
  "importance": 0.80,
  "created_at": "2026-06-12T13:00:00+08:00",
  "last_confirmed_at": "2026-06-12T13:00:00+08:00",
  "valid_until": null,
  "source_thread_id": "thread_xxx",
  "source_message_ids": ["m8", "m16"],
  "status": "active|superseded|expired"
}
```

写入策略：

- 只从用户明确陈述、反复行为证据或已确认计划中写入；
- 助手自行推测的内容不能直接成为高置信度事实；
- 同一 `type + subject + topic` 先做 upsert，不直接 append；
- 新事实与旧事实冲突时，将旧记录标记为 `superseded`；
- 情绪状态短 TTL，学习目标中 TTL，年级/选科长 TTL；
- 低置信度记忆在回答中使用前应向用户确认。

召回策略：

不要每轮全量注入 20 条事实。先根据当前查询生成 memory query，再打分：

```text
score =
  0.40 * semantic_similarity
  + 0.25 * importance
  + 0.20 * recency
  + 0.15 * confidence
```

按意图过滤：

- academic: 召回当前学科的弱项、掌握度、典型错误；
- planning: 召回目标、可用时间、选科、近期进展、计划偏好；
- emotional: 召回近期压力事件、沟通偏好，但避免重复强化过期负面标签；
- supervisor: 原则上不需要完整记忆，只需极少量路由相关信息。

建议注入上限为 5-8 条、600-1,000 tokens。

## 4. 上下文装配顺序

每个业务节点不再直接读取全部 `state["messages"]`，统一通过 Context Builder：

```text
System Prompt
-> 当前任务指令
-> L3 相关长期记忆
-> L2 结构化会话摘要
-> 当前 pinned artifacts
-> L1 最近完整回合
-> 当前用户消息
-> 本节点需要的 RAG/Web 上下文
```

不同节点使用不同预算。Supervisor 只看当前消息和少量摘要；Academic 节点重点保留题目与
知识进展；Planner 节点重点保留硬约束、计划决策与用户反馈；Emotional 节点重点保留近期
情绪轨迹和沟通偏好。

## 5. 对当前代码的改造位置

### 第一阶段：先修正确性

1. `src/graph/builder.py`
   - 使用 `RemoveMessage(id=REMOVE_ALL_MESSAGES)` 真正替换旧消息；
   - 压缩节点从固定 16 条改为 token 预算；
   - 给压缩节点增加 before/after token、耗时、压缩率埋点。

2. `src/memory/compressor.py`
   - 删除字符截断；
   - 使用 token counter；
   - 输出 Pydantic 结构化 `SessionEpisode`；
   - 添加 source message anchors；
   - 压缩失败时采用确定性 trim，而不是无限保留全量历史。

3. 测试
   - 验证 Checkpointer 中旧消息确实被删除；
   - 验证最后一条用户消息、当前题干和 pinned 内容不丢；
   - 验证压缩连续执行 10 次后关键约束仍一致。

### 第二阶段：长期记忆升级

1. 将 `thread_id` 和 `user_id` 分离；
2. 把 JSON 字符串列表改为结构化记录；
3. 增加 upsert、冲突替换、TTL 和删除接口；
4. 增加 `retrieve_memories(user_id, query, intent, subject)`；
5. 记忆抽取改为异步任务，不阻塞用户响应。

### 第三阶段：按节点装配上下文

新增 `src/memory/context_builder.py`，让各节点声明：

```python
ContextPolicy(
    recent_turns=4,
    memory_types={"progress"},
    memory_top_k=6,
    include_session_episode=True,
    include_artifacts=True,
    token_budget=12_000,
)
```

消除各节点自行拼接全部历史的重复逻辑。

## 6. 推荐配置

```yaml
memory:
  context_window_tokens: 32000
  reserved_output_tokens: 5000
  soft_limit_ratio: 0.60
  hard_limit_ratio: 0.78
  recent_turns_min: 3
  recent_turns_max: 6
  compact_min_overflow_tokens: 4000
  episode_max_tokens: 1200
  long_term_top_k: 6
  long_term_max_tokens: 800
  consolidation_every_compactions: 4
```

不要把这些数字写死。不同模型上下文和不同业务节点应允许覆盖。

## 7. 评估方案

构造至少 30 组、每组 30-50 轮的长会话，覆盖：

- 连续数学题讲解；
- 长作文多轮批改；
- 学习计划反复修改；
- 学术问题与情绪支持交替；
- 用户修改目标分数、时间和科目弱项；
- 跨会话继续昨天的计划。

指标：

```text
compression_ratio       压缩前后输入 tokens 比
constraint_retention    数字/日期/目标/约束保留率
task_continuity         压缩后继续完成任务的成功率
memory_precision@k      召回记忆中真正相关的比例
stale_memory_rate       使用过期或被替代记忆的比例
anchor_recovery_rate    能否根据 anchor 找回原始证据
latency_p95             压缩与召回增加的延迟
cost_per_30_turns       每 30 轮总 token 和费用
```

建议项目目标：

- 上下文 tokens 下降 35%-55%；
- 关键约束保留率 >= 98%；
- 记忆 Precision@5 >= 90%；
- stale memory rate < 2%；
- 30 轮任务连续完成率不低于无压缩基线 2 个百分点。

## 8. 最适合项目展示的技术点

可以将该工作点描述为：

> 面向高考辅导长会话设计三层渐进式记忆压缩：基于 token budget 的工作集裁剪、
> 带证据锚点的结构化会话摘要，以及具备冲突更新、TTL 和意图过滤的跨会话记忆召回。
> 针对题目原文、学习计划硬约束和情绪风险信息设置 pinned 保留策略，并通过长期会话
> 压测评估压缩率、约束保留率、过期记忆率与任务连续完成率。

这比“固定保留 8 轮 + 400 字摘要 + 20 条事实”更能体现工程价值，同时仍保持三层结构，
复杂度明显低于完整的五层压缩流水线。
