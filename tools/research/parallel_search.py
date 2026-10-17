"""Anonymous Parallel Search MCP tools for the research task dispatcher."""
import json
from typing import List, Type
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from langchain.tools import BaseTool
from langchain.pydantic_v1 import BaseModel, Field

from .common.model_schemas import ContentItem, ResearchToolOutput

MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "research-agent (+https://github.com/z33pX/research-agent)"
PROTOCOL_VERSION = "2025-03-26"


class ParallelMCPError(RuntimeError):
    """A protocol or remote tool error; never an empty success."""


def _response(response, request_id):
    response.raise_for_status()
    if response.headers.get("content-type", "").startswith("text/event-stream"):
        messages = []
        for event in response.text.replace("\r\n", "\n").split("\n\n"):
            data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
            if data:
                messages.append(json.loads(data))
    else:
        messages = [response.json()]
    for message in messages:
        if message.get("id") == request_id:
            if "error" in message:
                raise ParallelMCPError(f"Parallel MCP: {message['error']}")
            if "result" in message:
                return message["result"]
    raise ParallelMCPError("Parallel MCP returned no matching JSON-RPC result")


def _payload(method, params, request_id):
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def _initialize_params():
    return {"protocolVersion": PROTOCOL_VERSION, "capabilities": {},
            "clientInfo": {"name": "research-agent", "version": "1.0"}}


def _accept_session(client, response, result):
    version = result.get("protocolVersion")
    if version not in {"2024-11-05", "2025-03-26", "2025-06-18"}:
        raise ParallelMCPError("Parallel MCP negotiated an unsupported protocol version")
    client.headers["MCP-Protocol-Version"] = version
    if response.headers.get("Mcp-Session-Id"):
        client.headers["Mcp-Session-Id"] = response.headers["Mcp-Session-Id"]


def _tool_output(result):
    if result.get("isError"):
        detail = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        raise ParallelMCPError(f"Parallel MCP tool failed: {detail}")
    data = result.get("structuredContent")
    if data is None:
        text = "\n".join(c.get("text", "") for c in result.get("content", []) if c.get("type") == "text")
        try:
            data = json.loads(text)
        except ValueError as error:
            raise ParallelMCPError("Parallel MCP returned invalid tool content") from error
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ParallelMCPError("Parallel MCP returned no results array")
    if data.get("errors") and not data["results"]:
        raise ParallelMCPError(f"Parallel fetch failed: {data['errors']}")
    items = []
    for hit in data["results"]:
        url = hit["url"]
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ParallelMCPError("Parallel MCP returned an invalid source URL")
        excerpts = "\n\n".join(hit.get("excerpts", []))
        items.append(ContentItem(url=url, title=hit.get("title") or url,
                                 snippet=excerpts, content=hit.get("full_content") or excerpts,
                                 source="parallel"))
    return ResearchToolOutput(content=items, summary="")


class SearchInput(BaseModel):
    query: str = Field(..., min_length=1, description="The research question to answer.")


class FetchInput(SearchInput):
    urls: List[str] = Field(..., min_items=1, max_items=20, description="Specific HTTP(S) pages to read.")


class _ParallelTool(BaseTool):
    timeout: float = Field(default=60.0, gt=0)
    session_id: str = Field(default_factory=lambda: str(uuid4()))

    def _arguments(self, query, urls=None):
        if urls is None:
            return "web_search", {"objective": query, "search_queries": [query], "session_id": self.session_id}
        for url in urls:
            parsed = urlsplit(url)
            if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
                raise ValueError("Fetch URLs must be HTTP(S) URLs without credentials")
        return "web_fetch", {"urls": urls, "objective": query[:200], "session_id": self.session_id}

    def _run(self, query, urls=None):
        name, arguments = self._arguments(query, urls)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/event-stream"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(MCP_URL, json=_payload("initialize", _initialize_params(), 1))
            _accept_session(client, response, _response(response, 1))
            client.post(MCP_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"}).raise_for_status()
            tools = _response(client.post(MCP_URL, json=_payload("tools/list", {}, 2)), 2)
            if name not in {tool["name"] for tool in tools.get("tools", [])}:
                raise ParallelMCPError(f"Parallel MCP does not advertise {name}")
            result = _response(client.post(MCP_URL, json=_payload("tools/call", {"name": name, "arguments": arguments}, 3)), 3)
        return _tool_output(result)

    async def _arun(self, query, urls=None):
        name, arguments = self._arguments(query, urls)
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, text/event-stream"}
        # Real async I/O lets cancellation close the in-flight HTTP request.
        async with httpx.AsyncClient(timeout=self.timeout, headers=headers) as client:
            response = await client.post(MCP_URL, json=_payload("initialize", _initialize_params(), 1))
            _accept_session(client, response, _response(response, 1))
            response = await client.post(MCP_URL, json={"jsonrpc": "2.0", "method": "notifications/initialized"})
            response.raise_for_status()
            tools = _response(await client.post(MCP_URL, json=_payload("tools/list", {}, 2)), 2)
            if name not in {tool["name"] for tool in tools.get("tools", [])}:
                raise ParallelMCPError(f"Parallel MCP does not advertise {name}")
            result = _response(await client.post(MCP_URL, json=_payload("tools/call", {"name": name, "arguments": arguments}, 3)), 3)
        return _tool_output(result)


class ParallelSearch(_ParallelTool):
    name: str = "parallel-search"
    description: str = "Search the web for research evidence and cited excerpts. Use this for general factual questions."
    args_schema: Type[BaseModel] = SearchInput


class ParallelFetch(_ParallelTool):
    name: str = "parallel-fetch"
    description: str = "Read specific web pages when search excerpts are insufficient or the user supplied a URL."
    args_schema: Type[BaseModel] = FetchInput
