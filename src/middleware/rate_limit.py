"""Token Bucket 限流中间件——防止API滥用和恶意刷量

算法：Token Bucket（令牌桶）
- 每个请求消耗一个令牌，令牌不足返回 429
- 令牌以固定速率补充（refill_rate 个/秒）
- 桶有容量上限（burst_size），支持短时突发流量

为什么选 Token Bucket 而不是固定窗口？
- 固定窗口有边界效应：用户可能在前一秒用完配额，下一秒又有配额
- Token Bucket 平滑限流：令牌均匀补充，无尖锐边界
- 支持短时突发：burst_size > 1 允许微小的流量尖峰

面试追问点：
- 内存存储的局限：重启丢失所有计数器，多实例部署需要 Redis
- 当前实现适合单机 demo，生产环境用 Redis + 滑动窗口更可靠
- 为什么不直接限 IP 而是用 X-Forwarded-For？因为可能部署在反向代理后面

配置（环境变量）：
- RATE_LIMIT_ENABLED: 是否启用限流（默认 true）
- RATE_LIMIT_REQUESTS: 每秒补充令牌数（默认 5）
- RATE_LIMIT_BURST: 桶容量上限（默认 10）
"""

from __future__ import annotations

import logging
import os
import time
from threading import Lock
from typing import Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_RATE = 5     # 每秒补充 5 个令牌
DEFAULT_BURST = 10   # 桶最多容纳 10 个令牌（允许 2 秒突发）


class TokenBucket:
    """单个 IP 的令牌桶。

    线程安全，使用 Lock 保护并发访问。
    """

    def __init__(self, rate: float, burst: int) -> None:
        self._rate = rate
        self._burst = burst
        self._tokens = float(burst)  # 初始满桶
        self._last_refill = time.monotonic()
        self._lock = Lock()

    def consume(self, tokens: int = 1) -> bool:
        """尝试消费令牌。

        Returns:
            True 表示消费成功，False 表示令牌不足
        """
        with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            return False

    def _refill(self) -> None:
        """根据时间流逝补充令牌。"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._burst, self._tokens + elapsed * self._rate)
        self._last_refill = now

    @property
    def available(self) -> float:
        """当前可用令牌数（用于调试）。"""
        with self._lock:
            self._refill()
            return self._tokens


class TokenBucketRateLimiter:
    """基于内存的 IP 级别 Token Bucket 限流器。

    自动清理超过 5 分钟未活动的桶，防止内存泄漏。
    """

    def __init__(
        self,
        rate: float = DEFAULT_RATE,
        burst: int = DEFAULT_BURST,
        cleanup_interval: int = 300,
    ) -> None:
        self._rate = rate
        self._burst = burst
        self._cleanup_interval = cleanup_interval
        self._buckets: dict[str, tuple[TokenBucket, float]] = {}  # ip → (bucket, last_access)
        self._lock = Lock()
        self._last_cleanup = time.monotonic()

    def is_allowed(self, client_ip: str) -> bool:
        """检查指定 IP 的请求是否被允许。

        Returns:
            True 表示允许，False 表示被限流
        """
        self._maybe_cleanup()

        with self._lock:
            entry = self._buckets.get(client_ip)
            if entry is None:
                bucket = TokenBucket(self._rate, self._burst)
            else:
                bucket = entry[0]

            allowed = bucket.consume(1)
            self._buckets[client_ip] = (bucket, time.monotonic())

            if not allowed:
                logger.warning("Rate limit triggered for IP: %s", client_ip)

            return allowed

    def _maybe_cleanup(self) -> None:
        """定期清理长时间未活动的桶（每 cleanup_interval 秒一次）。"""
        now = time.monotonic()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        with self._lock:
            stale = [
                ip
                for ip, (_, last_access) in self._buckets.items()
                if now - last_access > self._cleanup_interval
            ]
            for ip in stale:
                del self._buckets[ip]
            if stale:
                logger.debug("Cleaned up %d stale rate-limit buckets", len(stale))
            self._last_cleanup = now

    @property
    def active_ips(self) -> int:
        """当前活跃 IP 数（调试用）。"""
        with self._lock:
            return len(self._buckets)


# ============================================================================
# 全局单例
# ============================================================================

_limiter: Optional[TokenBucketRateLimiter] = None


def get_rate_limiter() -> TokenBucketRateLimiter:
    """获取全局限流器单例。"""
    global _limiter
    if _limiter is None:
        rate = float(os.getenv("RATE_LIMIT_REQUESTS", str(DEFAULT_RATE)))
        burst = int(os.getenv("RATE_LIMIT_BURST", str(DEFAULT_BURST)))
        _limiter = TokenBucketRateLimiter(rate=rate, burst=burst)
        logger.info("Rate limiter initialized: rate=%s/s burst=%s", rate, burst)
    return _limiter


# ============================================================================
# FastAPI 中间件工厂
# ============================================================================

def _get_client_ip(request: Request) -> str:
    """获取客户端真实 IP。

    优先从 X-Forwarded-For 读取（用于反向代理部署），
    回退到 request.client.host（直连）。
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def create_rate_limit_middleware() -> type[BaseHTTPMiddleware]:
    """创建 FastAPI 限流中间件类（工厂函数）。

    返回一个中间件类，可以直接通过 app.add_middleware() 注册。
    使用工厂函数是为了从环境变量读取配置。
    """

    class RateLimitMiddleware(BaseHTTPMiddleware):
        """Token Bucket 限流中间件。

        在处理请求前检查令牌桶。被限流的请求返回 429 状态码，
        并附带 Retry-After 和 X-RateLimit 头信息。
        """

        async def dispatch(self, request: Request, call_next: Callable):
            # 健康检查等路径跳过限流
            path = request.url.path
            if path in ("/health", "/ping", "/docs", "/openapi.json", "/redoc"):
                return await call_next(request)

            limiter = get_rate_limiter()
            client_ip = _get_client_ip(request)

            if not limiter.is_allowed(client_ip):
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "请求过于频繁，请稍后重试",
                        "detail": f"当前限制为每秒 {limiter._rate} 次请求",
                    },
                    headers={
                        "Retry-After": "1",
                        "X-RateLimit-Limit": str(limiter._rate),
                    },
                )

            return await call_next(request)

    return RateLimitMiddleware
