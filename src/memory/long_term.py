"""长期记忆模块——基于 JSON 文件的简单持久化用户记忆

设计思路（最小可行方案）：
1. 每个用户（通过 thread_id 标识）拥有一个事实列表
2. 每次对话结束后，用小模型从对话中提取关键事实
3. 下次对话开始时，将用户事实注入系统提示词
4. 事实以 JSON 文件存储，简单可靠、无需数据库

存储结构：
{
  "thread_abc123": {
    "facts": [
      "用户是2025届高三学生，选考物化生",
      "用户数学较弱，特别是导数部分",
      "用户每天可用学习时间约4小时",
      "用户偏好刷题为主的复习方式"
    ],
    "last_updated": "2026-06-04T10:30:00"
  }
}

面试追问点：
- 为什么选 JSON 文件而不是数据库？
  答：长期记忆数据量小（每个用户几条~几十条事实），不需要索引查询。
  JSON 文件零依赖、零配置，适合 demo 和小规模部署。
- 为什么不嵌入向量数据库做语义检索？
  答：用户事实数量少，全量注入提示词即可（几十条约 500 tokens）。
  语义检索的价值在事实超过百条时才体现。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

logger = logging.getLogger(__name__)

# 存储目录：项目根 / data / memory /
_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "memory"
_STORE_FILE = _STORE_DIR / "user_memories.json"
_MAX_FACTS_PER_USER = 20  # 每人最多保留 20 条事实


class MemoryStore:
    """JSON 文件持久化的长期记忆存储器。

    线程安全（使用 Lock），支持多用户隔离。
    自动去重：相同语义的事实不会重复添加。
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._data: dict = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # 文件读写
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """懒加载：首次访问时从磁盘读取，之后使用内存缓存。"""
        if self._loaded:
            return

        _STORE_DIR.mkdir(parents=True, exist_ok=True)

        if _STORE_FILE.exists():
            try:
                with open(_STORE_FILE, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                logger.info("Loaded %d user memories from %s", len(self._data), _STORE_FILE)
            except (json.JSONDecodeError, IOError):
                logger.warning("Failed to load memory file, starting fresh")
                self._data = {}
        else:
            self._data = {}

        self._loaded = True

    def _save(self) -> None:
        """将内存中的数据写回磁盘。

        使用临时文件 + 原子重命名，确保写入过程中不会损坏已有数据。
        """
        _STORE_DIR.mkdir(parents=True, exist_ok=True)
        tmp_file = _STORE_FILE.with_suffix(".json.tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
            tmp_file.replace(_STORE_FILE)  # 原子替换
        except Exception:
            logger.warning("Failed to save memories", exc_info=True)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def add_fact(self, user_id: str, fact: str) -> bool:
        """为用户添加一条事实。

        - 自动去重：如果事实文本完全相同，不添加
        - 自动裁剪：超过 _MAX_FACTS_PER_USER 条时移除最旧的

        Returns:
            True 表示添加成功（新增），False 表示重复（未添加）
        """
        with self._lock:
            self._ensure_loaded()

            user_entry = self._data.setdefault(user_id, {"facts": [], "last_updated": ""})
            facts: list = user_entry["facts"]

            # 去重
            if fact in facts:
                return False

            # 添加
            facts.append(fact)

            # 裁剪
            while len(facts) > _MAX_FACTS_PER_USER:
                facts.pop(0)

            user_entry["last_updated"] = datetime.now().isoformat()
            self._save()
            return True

    def get_facts(self, user_id: str) -> list[str]:
        """获取指定用户的所有事实列表。

        Returns:
            事实字符串列表，不存在时返回空列表
        """
        with self._lock:
            self._ensure_loaded()
            entry = self._data.get(user_id, {})
            return list(entry.get("facts", []))

    def clear_facts(self, user_id: str) -> None:
        """清空指定用户的所有记忆（用于测试或隐私重置）。"""
        with self._lock:
            self._ensure_loaded()
            if user_id in self._data:
                del self._data[user_id]
                self._save()

    def summarize_for_prompt(self, user_id: str) -> str:
        """将用户事实格式化为可注入 LLM 系统提示词的文本。

        Returns:
            格式化后的记忆文本，无记忆时返回空字符串。
            示例："[关于该用户的记忆]\n- 用户是高三学生\n- 数学较弱\n"
        """
        facts = self.get_facts(user_id)
        if not facts:
            return ""
        lines = ["[关于该用户的记忆]", *(f"- {f}" for f in facts)]
        return "\n".join(lines)


# ============================================================================
# 全局单例（进程内共享）
# ============================================================================

_memory_store: Optional[MemoryStore] = None


def get_memory_store() -> MemoryStore:
    """获取 MemoryStore 全局单例。

    使用懒初始化，只在首次调用时创建，之后复用。
    """
    global _memory_store
    if _memory_store is None:
        _memory_store = MemoryStore()
    return _memory_store
