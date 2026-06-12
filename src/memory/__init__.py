"""记忆系统——三层压缩架构

Layer 1 - 窗口消息 (Window):
  最近 N 轮完整保留，作为"工作记忆"

Layer 2 - 会话摘要 (Session Summary):
  超出窗口的旧消息压缩为摘要，注入消息列表头部

Layer 3 - 长期事实 (Long-term Facts):
  跨会话持久化的离散事实，通过 MemoryStore 管理
"""

from src.memory.compressor import compress_conversation
from src.memory.long_term import MemoryStore, get_memory_store

__all__ = ["MemoryStore", "get_memory_store", "compress_conversation"]

