"""记忆系统——短期记忆 + 长期记忆

短期记忆（Short-term）：
  LangGraph 的 messages 字段（add_messages reducer）天然提供短期记忆。
  每个节点的 LLM 调用都会获得完整对话历史，实现"当前会话内"的上下文感知。

长期记忆（Long-term）：
  JSON 文件持久化用户关键信息（学科偏好、弱项、学习风格等）。
  跨会话保持，下次对话时自动注入系统提示词。
  提取逻辑：用 Supervisor 小模型从对话中抽取用户事实，零额外成本。
"""

from src.memory.long_term import MemoryStore, get_memory_store

__all__ = ["MemoryStore", "get_memory_store"]
