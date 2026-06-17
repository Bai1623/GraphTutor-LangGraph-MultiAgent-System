"""工具层——封装 RAG 检索和 Web 搜索为 LangChain Tool 格式

LangGraph 的节点可以直接调用 Python 函数，不需要非得用 LangChain Tool。
但将功能封装为 @tool 的好处是：
1. 与 LangChain 生态兼容（未来可用于 LangSmith、LangServe 等）
2. 清晰定义函数的输入输出 schema
3. 可以为 future 的 Agent 递归调用（LLM 自主选择工具）做准备

当前架构中，工具是在图节点中硬编码调用的（如 academic.py 中直接调用
rag_retrieve 函数），而非让 LLM 自主选择。这种"预编排"方式
更适合确定性流程（先检索再回答），比让 LLM 自己决定更可靠。
"""

__all__ = [
    "rag_retrieve",
    "get_search_tool",
    "search",
    "search_knowledge_base",
    "search_web",
]


def __getattr__(name: str):
    if name == "rag_retrieve":
        from src.tools.rag_tool import rag_retrieve

        return rag_retrieve
    if name in {"get_search_tool", "search"}:
        from src.tools.search_tool import get_search_tool, search

        return {"get_search_tool": get_search_tool, "search": search}[name]
    if name in {"search_knowledge_base", "search_web"}:
        from src.tools.agent_tools import search_knowledge_base, search_web

        return {
            "search_knowledge_base": search_knowledge_base,
            "search_web": search_web,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
