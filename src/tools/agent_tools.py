"""Agent 工具定义 —— 为 Function Calling 准备的两把"自主查询"工具

与当前图节点的本质区别：
- 图节点（rag_retrieve / web_search）：代码预定义调用时机，LLM 被动接受结果
- Function Calling 工具：LLM 自主决定要不要调、什么时候调、用什么参数调

这两个工具通过 @tool 装饰器暴露 name / description / args_schema，
当被 bind_tools() 绑定到 LLM 时，LLM 会在推理过程中"意识到"自己
可以调用这些工具来补充信息，从而从"被动接收上下文"升级为"主动按需查询"。

面试追问点：
- 为什么不用 Function Calling 替代图节点而是并存？
  答：图节点保证基线（并行检索结果一定在上下文里），FC 是增量补充。
  这样即使 LLM 一次工具都不调，也至少有一轮检索结果可用。
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.rag.retriever import retrieve
from src.tools.search_tool import search as web_search_fn


@tool
def search_knowledge_base(query: str) -> str:
    """搜索本地高考知识库，获取历年真题、课程大纲、知识点讲解、答题技巧等。

    当需要查找具体的学科知识、真题原文、解题方法时使用此工具。
    查询应使用关键词或简短描述，效果最佳。

    参数:
        query: 检索查询文本，如"导数单调性判断方法"或"2024新课标Ⅰ卷文言文"
    """
    result = retrieve(query=query)
    docs = result.get("docs", [])
    if not docs:
        return "知识库中未找到相关内容。"
    parts = []
    for i, d in enumerate(docs[:3], 1):
        parts.append(
            f"[{i}] 来源：{d.get('source', '未知')}"
            f"（相关度：{d.get('score', d.get('rerank_score', 'N/A'))}）\n"
            f"{d.get('content', '')}"
        )
    return "\n\n".join(parts)


@tool
def search_web(query: str) -> str:
    """搜索互联网获取最新高考政策、分数线、考试动态等时效性信息。

    当需要查询政策变化、录取数据、新闻动态或知识库未覆盖的内容时使用。
    查询应使用关键词或简短描述。

    参数:
        query: 搜索查询文本，如"2026年高考改革最新政策"
    """
    results = web_search_fn(query)
    if not results:
        return "网络搜索未找到相关内容。"
    parts = []
    for i, r in enumerate(results[:3], 1):
        parts.append(
            f"[{i}] {r.get('title', '无标题')}\n"
            f"链接：{r.get('url', '')}\n"
            f"{r.get('content', '')}"
        )
    return "\n\n".join(parts)
