"""统一 LLM 工厂 + 跨厂商容灾降级（面试核心考点）

这个模块是系统的"电源适配器"——所有节点通过 get_node_llm() 获取 LLM 实例，
不需要关心 API Key、base_url、temperature 等配置细节。

设计要点：
1. 配置驱动: 每个节点可以从 settings.yaml 覆盖 model/temperature/api_key_env
   例如 supervision 节点配置了不同的 model（Qwen2.5-7B）和 base_url（SiliconFlow）
2. 统一 streaming: 所有 LLM 实例默认开启 streaming=True，确保前端能收到逐字流式输出
3. 跨厂商容灾: invoke_with_fallback() 捕获可恢复异常，自动切换到备用模型
4. OpenAI 兼容接口: 无论 DeepSeek 还是 SiliconFlow，都通过 ChatOpenAI 统一调用

面试追问点：
- 为什么用 ChatOpenAI 而不是 DeepSeek 的 SDK？
  答：两者都兼容 OpenAI API 格式，ChatOpenAI 是 LangChain 的抽象层，
  切换厂商只需改 base_url 和 api_key，不改业务代码
- 容灾覆盖了哪些异常？为什么？
  超时 (APITimeoutError)、连接断开 (APIConnectionError)、服务端错误 (InternalServerError)
  限流 (RateLimitError) — 这些都是"可重试"的异常。认证错误 (401) 不重试。
"""

from __future__ import annotations

import logging
import os

from langchain_openai import ChatOpenAI

from src.config import get_setting

logger = logging.getLogger(__name__)


# ============================================================================
# 可重试异常集合
# ============================================================================
# 定义了触发 fallback 的异常类型清单。
# 优先尝试导入 openai 模块以获取更精确的异常类型；
# 如果 openai 未安装则回退到基础异常。

_FALLBACK_ERRORS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)

try:
    import openai

    _FALLBACK_ERRORS = (
        TimeoutError,
        ConnectionError,
        openai.APITimeoutError,       # API 超时（如 DeepSeek 响应慢）
        openai.APIConnectionError,    # 网络连接失败（如 DNS、TLS 问题）
        openai.InternalServerError,   # 服务端 5xx 错误
        openai.RateLimitError,        # API 限流（Rate Limit）
    )
except ImportError:
    pass


# ============================================================================
# LLM 工厂函数
# ============================================================================

def get_node_llm(node_name: str, **overrides) -> ChatOpenAI:
    """根据节点名称获取配置好的 LLM 实例。

    每个节点（如 "supervisor"、"planner"、"academic"、"emotional"）
    可以在 settings.yaml 中有独立的模型配置，实现"按需分配"。

    配置优先级（从高到低）：
    1. 调用时传入的 **overrides（显式覆盖）
    2. settings.yaml 中该节点的配置（如 supervisor.model）
    3. 环境变量 DEEPSEEK_*（全局默认）
    4. 代码中的硬编码默认值

    面试追问：为什么 override 放在最后 update？
    因为 **overrides 是调用方传入的"最高优先级"配置，
    需要覆盖 defaults 中的所有值（包括 streaming=True）。
    例如测试时可以 override streaming=False 来获得同步响应。
    """
    model = get_setting(f"{node_name}.model", os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    api_key_env = get_setting(f"{node_name}.api_key_env", "DEEPSEEK_API_KEY")
    base_url = get_setting(f"{node_name}.base_url", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    temperature = get_setting(f"{node_name}.temperature", 0.7)

    defaults = dict(
        model=model,
        api_key=os.getenv(api_key_env),          # 动态读取环境变量（不硬编码）
        base_url=base_url,
        temperature=temperature,
        streaming=True,                            # 全局开启流式输出
    )
    defaults.update(overrides)                     # 调用方覆盖优先级最高
    return ChatOpenAI(**defaults)


def get_primary_llm(**overrides) -> ChatOpenAI:
    """获取主力模型实例（默认 DeepSeek）。

    用于不需要节点级配置的场景（如直接调用 LLM 的脚本），
    直接从环境变量读取配置。
    """
    defaults = dict(
        model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0.7,
        streaming=True,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


def get_fallback_llm(**overrides) -> ChatOpenAI:
    """获取容灾模型实例（默认也指向 DeepSeek，可配置为 SiliconFlow Qwen）。

    容灾模型的默认值指向主力模型的配置，这意味着即使没有配置单独的 fallback，
    "重试一次"本身也有价值（可能只是瞬时的网络抖动）。

    推荐配置：
    - FALLBACK_MODEL=Qwen/Qwen2.5-7B-Instruct
    - FALLBACK_API_KEY=SILICONFLOW_API_KEY
    - FALLBACK_BASE_URL=https://api.siliconflow.cn/v1
    这样主力用 DeepSeek（便宜、快），容灾用 SiliconFlow Qwen（不同厂商，避免共同故障）
    """
    defaults = dict(
        model=os.getenv("FALLBACK_MODEL", os.getenv("DEEPSEEK_MODEL", "deepseek-chat")),
        api_key=os.getenv("FALLBACK_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or "not-configured",
        base_url=os.getenv("FALLBACK_BASE_URL", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")),
        temperature=0.7,
        streaming=True,
    )
    defaults.update(overrides)
    return ChatOpenAI(**defaults)


# ============================================================================
# 容灾调用函数
# ============================================================================

def invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """尝试主力模型，失败时自动切换到容灾模型。

    这是系统稳定性的关键函数。所有图节点都通过这个函数调用 LLM，
    而不是直接调用 llm.invoke()。

    工作流程：
    1. 尝试 primary.invoke(messages)
    2. 成功 → 记录 fallback_used=False，返回响应
    3. 失败（且异常在 _FALLBACK_ERRORS 中）→
       a. 没有配置 fallback → 直接抛出异常
       b. 有 fallback → 记录容灾事件到 OTel span，调用 fallback.invoke()

    面试追问：为什么不用 LangChain 的 fallback 机制？
    答：需要精确控制容灾事件的 OTel 埋点（记录 fallback_model 和失败原因），
    以及支持 structured_output 实例的容灾（with_structured_output 返回的
    不是简单的 ChatModel，LangChain 的 fallback 不保证工作）。
    """
    try:
        response = primary.invoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        return fallback.invoke(messages)


async def async_invoke_with_fallback(primary, messages, *, fallback=None, span=None):
    """invoke_with_fallback 的异步版本。

    使用 ainvoke() 替代 invoke()，不阻塞事件循环。
    适用于 FastAPI SSE 端点或异步图节点（所有 @traced_node 节点都使用此函数）。

    与同步版本完全相同的容灾逻辑，只是调用方式不同。
    """
    try:
        response = await primary.ainvoke(messages)
        if span is not None:
            span.set_attribute("llm.fallback_used", False)
        return response
    except _FALLBACK_ERRORS as exc:
        if fallback is None:
            raise

        logger.warning(
            "Primary LLM failed (%s: %s), falling back",
            type(exc).__name__,
            exc,
        )

        if span is not None:
            span.set_attribute("llm.fallback_used", True)
            span.set_attribute(
                "llm.fallback_model",
                getattr(fallback, "model_name", "unknown"),
            )
            span.add_event(
                "llm.fallback_triggered",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )

        return await fallback.ainvoke(messages)
