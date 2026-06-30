"""DuckDuckGo 联网搜索工具

提供懒加载单例的搜索工具和统一的 search() 函数。
将 DuckDuckGo 的原始输出（snippet/title/link）映射为统一的
{content, title, url} 格式，下游消费者不依赖特定的搜索服务商。

搜索策略：
- max_results=3：减少无效信息，聚焦最相关的 3 条
- lazy singleton：避免跨节点重复创建搜索实例
- 异常静默处理：任何失败返回空列表，不阻塞主流程

面试追问点：
- 为什么用 DuckDuckGo 而不是百度搜索？
  答：DuckDuckGo 不需要申请 API Key，零配置。
  对于高考政策搜索，百度可能需要 API 额度限制。
  DuckDuckGo 适合 demo 和小规模使用，生产环境建议替换为 SerpAPI 或百度。
- 返回空列表后下游怎么处理？
  academic.py 的 generate_answer 节点会收到空列表，
  但仍然会基于 RAG 结果生成回答，只是没有网络补充信息。
"""

from __future__ import annotations

from langchain_community.tools import DuckDuckGoSearchResults

_search_tool: DuckDuckGoSearchResults | None = None


def get_search_tool() -> DuckDuckGoSearchResults:
    """懒加载单例——避免因多次实例化导致的 API 限流。

    DuckDuckGoSearchResults 实例在进程生命周期内复用，
    避免了每次搜索都重新建立 HTTP 连接的开销。
    """
    global _search_tool
    if _search_tool is None:
        _search_tool = DuckDuckGoSearchResults(
            num_results=3,           # 只取前 3 条，避免信息过载
            output_format="list",    # 返回列表格式，方便处理
        )
    return _search_tool


def search(query: str) -> list[dict]:
    """执行网络搜索并返回统一格式的结果。

    将 DuckDuckGo 的原始输出字段（snippet, title, link）
    映射为标准字段（content, title, url），使得：
    1. 下游消费者不需要关心具体搜索服务商
    2. 切换搜索服务（如换到 SerpAPI/百度）时改这里一个函数即可
    3. 所有调用方通过统一的接口使用搜索结果

    参数：
        query: 搜索查询文本

    返回：
        [{"content": "...", "title": "...", "url": "..."}, ...]
        搜索失败时返回空列表 []
    """
    tool = get_search_tool()
    try:
        results = tool.invoke(query)
    except Exception:
        # —— 容错：搜索失败时返回空结果 ——
        return []

    # DuckDuckGo 有时返回字符串（较少见），兜底处理
    if isinstance(results, str):
        return [{"content": results, "title": "", "url": ""}]

    # 字段映射：DuckDuckGo → 统一格式
    return [
        {
            "content": r.get("snippet", r.get("content", "")),
            "title": r.get("title", ""),
            "url": r.get("link", r.get("url", "")),
        }
        for r in results
    ]
