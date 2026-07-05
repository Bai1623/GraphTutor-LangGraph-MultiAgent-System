"""Official policy/admission search tool with optional MCP backend.

The planner should prefer structured official data over generic web search.
This module keeps the integration optional: configure a policy/admission MCP
server through environment variables, otherwise callers can fall back to the
existing web search path.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any

DEFAULT_POLICY_MCP_TOOL = "policy_search"
DEFAULT_PROTOCOL_VERSION = "2025-06-18"


@dataclass(frozen=True)
class PolicySearchResult:
    source: str
    url: str
    published_at: str | None
    province: str | None
    topic: str | None
    content: str
    confidence: str = "official"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PolicySearchError(RuntimeError):
    """Raised when the configured official policy backend cannot be used."""


def search_official_policy(
    query: str,
    *,
    province: str | None = None,
    topic: str | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search official policy/admission data through a configured MCP tool.

    Environment variables:
        POLICY_MCP_URL: Streamable HTTP MCP endpoint.
        POLICY_MCP_COMMAND: stdio MCP server command.
        POLICY_MCP_TOOL: MCP tool name, default ``policy_search``.
        POLICY_MCP_TIMEOUT_SECONDS: per-call timeout, default 10.

    Returns a list of normalized dictionaries with stable keys:
    source/url/published_at/province/topic/content/confidence.
    """
    if not query.strip():
        return []

    tool_name = os.getenv("POLICY_MCP_TOOL", DEFAULT_POLICY_MCP_TOOL)
    timeout_s = _timeout_seconds()
    arguments = {
        "query": query,
        "province": province,
        "topic": topic,
        "limit": limit,
    }

    endpoint = os.getenv("POLICY_MCP_URL")
    command = os.getenv("POLICY_MCP_COMMAND")
    if endpoint:
        payload = _call_http_mcp(endpoint, tool_name, arguments, timeout_s)
    elif command:
        payload = _call_stdio_mcp(command, tool_name, arguments, timeout_s)
    else:
        return []

    return _normalize_policy_results(payload, default_province=province, default_topic=topic)[:limit]


def format_policy_results(results: list[dict[str, Any]]) -> str:
    """Format structured policy results for planner prompts."""
    if not results:
        return ""
    parts = []
    for i, result in enumerate(results, 1):
        source = result.get("source") or "unknown"
        published = result.get("published_at") or "unknown-date"
        province = result.get("province") or "national"
        topic = result.get("topic") or "policy"
        url = result.get("url") or ""
        content = result.get("content") or ""
        confidence = result.get("confidence") or "official"
        artifact_id = result.get("artifact_id") or ""
        artifact_line = f"\nArtifact: {artifact_id}" if artifact_id else ""
        parts.append(
            f"[{i}] {source} | {province} | {topic} | {published} | {confidence}\n"
            f"URL: {url}\n"
            f"{content}{artifact_line}"
        )
    return "\n\n".join(parts)


def _timeout_seconds() -> float:
    raw = os.getenv("POLICY_MCP_TIMEOUT_SECONDS", "10")
    try:
        return max(1.0, min(float(raw), 60.0))
    except ValueError:
        return 10.0


def _json_rpc_request(method: str, request_id: int | str, params: dict[str, Any]) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }


def _initialize_request(request_id: int | str) -> dict[str, Any]:
    return _json_rpc_request(
        "initialize",
        request_id,
        {
            "protocolVersion": DEFAULT_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "gaokao-tutor", "version": "0.3.0"},
        },
    )


def _initialized_notification() -> dict[str, Any]:
    return {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}


def _tools_call_request(
    request_id: int | str,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    return _json_rpc_request(
        "tools/call",
        request_id,
        {
            "name": tool_name,
            "arguments": {k: v for k, v in arguments.items() if v is not None},
        },
    )


def _call_stdio_mcp(
    command: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> Any:
    args = shlex.split(command, posix=os.name != "nt")
    if not args:
        raise PolicySearchError("POLICY_MCP_COMMAND is empty")

    messages = "\n".join(
        json.dumps(message, ensure_ascii=False)
        for message in (
            _initialize_request(1),
            _initialized_notification(),
            _tools_call_request(2, tool_name, arguments),
        )
    ) + "\n"
    try:
        completed = subprocess.run(
            args,
            input=messages,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PolicySearchError("MCP stdio request timed out") from exc

    if completed.returncode not in (0, None) and not completed.stdout:
        raise PolicySearchError(completed.stderr[:300] or "MCP stdio server failed")

    response = _find_json_rpc_response(completed.stdout, 2)
    if "error" in response:
        raise PolicySearchError(str(response["error"]))
    return response.get("result")


def _find_json_rpc_response(output: str, request_id: int | str) -> dict[str, Any]:
    for line in output.splitlines():
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(message, dict) and message.get("id") == request_id:
            return message
    raise PolicySearchError("MCP stdio response did not include tools/call result")


def _call_http_mcp(
    endpoint: str,
    tool_name: str,
    arguments: dict[str, Any],
    timeout_s: float,
) -> Any:
    session_id: str | None = None
    init_response, headers = _post_mcp(endpoint, _initialize_request(1), timeout_s, session_id)
    session_id = headers.get("mcp-session-id") or headers.get("Mcp-Session-Id")
    if isinstance(init_response, dict) and init_response.get("error"):
        raise PolicySearchError(str(init_response["error"]))

    _post_mcp(endpoint, _initialized_notification(), timeout_s, session_id, expect_body=False)
    response, _ = _post_mcp(endpoint, _tools_call_request(2, tool_name, arguments), timeout_s, session_id)
    if isinstance(response, dict) and response.get("error"):
        raise PolicySearchError(str(response["error"]))
    return response.get("result") if isinstance(response, dict) else response


def _post_mcp(
    endpoint: str,
    message: dict[str, Any],
    timeout_s: float,
    session_id: str | None,
    *,
    expect_body: bool = True,
) -> tuple[Any, dict[str, str]]:
    body = json.dumps(message, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "MCP-Protocol-Version": DEFAULT_PROTOCOL_VERSION,
    }
    if session_id:
        headers["Mcp-Session-Id"] = session_id
    request = urllib.request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            response_headers = dict(response.headers.items())
            if not expect_body or response.status == 202:
                return None, response_headers
            raw = response.read().decode("utf-8")
            content_type = response.headers.get("Content-Type", "")
            if "text/event-stream" in content_type:
                return _parse_sse_json_rpc(raw), response_headers
            return json.loads(raw) if raw else None, response_headers
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise PolicySearchError(f"MCP HTTP request failed: {exc}") from exc


def _parse_sse_json_rpc(raw: str) -> Any:
    for line in raw.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if not payload:
            continue
        message = json.loads(payload)
        if isinstance(message, dict) and "id" in message:
            return message
    return None


def _normalize_policy_results(
    payload: Any,
    *,
    default_province: str | None,
    default_topic: str | None,
) -> list[dict[str, Any]]:
    raw_results = _extract_results_payload(payload)
    results: list[dict[str, Any]] = []
    for item in raw_results:
        if isinstance(item, PolicySearchResult):
            results.append(item.to_dict())
            continue
        if not isinstance(item, dict):
            continue
        content = str(item.get("content") or item.get("text") or item.get("snippet") or "").strip()
        if not content:
            continue
        result = PolicySearchResult(
            source=str(item.get("source") or item.get("title") or "official policy source"),
            url=str(item.get("url") or item.get("link") or ""),
            published_at=item.get("published_at") or item.get("date"),
            province=item.get("province") or default_province,
            topic=item.get("topic") or default_topic,
            content=content,
            confidence=str(item.get("confidence") or "official"),
        )
        results.append(result.to_dict())
    return results


def _extract_results_payload(payload: Any) -> list[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        structured_content = payload.get("structuredContent")
        if isinstance(structured_content, list):
            return structured_content
        results = payload.get("results")
        if isinstance(results, list):
            return results
        content = payload.get("content")
        if isinstance(content, list):
            extracted: list[Any] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = str(part.get("text", "")).strip()
                    if not text:
                        continue
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = {"content": text}
                    extracted.extend(_extract_results_payload(parsed))
                elif isinstance(part, dict):
                    extracted.append(part)
            return extracted
        return [payload]
    if isinstance(payload, str):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            return [{"content": payload}]
        return _extract_results_payload(parsed)
    return []
