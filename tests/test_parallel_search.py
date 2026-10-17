"""Deterministic caller/dispatch tests; model and external services are scripted."""
import asyncio
import importlib
import json
import os
import runpy
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
from tools import ParallelSearch, ParallelFetch, create_default_tools
from tools import research
from tools.research.parallel_search import MCP_URL, USER_AGENT, ParallelMCPError

REAL_CLIENT = httpx.Client
REAL_ASYNC_CLIENT = httpx.AsyncClient
EXCERPT = "Research evidence with a source URL. " * 20
HIT = {"url": "https://example.org/research?q=a%20b", "title": "Research", "excerpts": [EXCERPT]}


class Server:
    def __init__(self, tool_result=None, sse=False, status=200, rpc_error=None):
        self.requests = []
        self.tool_result = tool_result if tool_result is not None else {"structuredContent": {"results": [HIT]}}
        self.sse = sse
        self.status = status
        self.rpc_error = rpc_error

    def __call__(self, request):
        self.requests.append(request)
        payload = json.loads(request.content)
        method = payload["method"]
        if method == "notifications/initialized":
            return httpx.Response(202)
        if method == "initialize":
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"protocolVersion": "2025-03-26"}},
                                  headers={"Mcp-Session-Id": "server-session"})
        if method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            if self.status != 200:
                return httpx.Response(self.status, json={"error": "Service unavailable"})
            result = self.tool_result
        response = {"jsonrpc": "2.0", "id": payload["id"], "result": result}
        if self.rpc_error:
            response = {"jsonrpc": "2.0", "id": payload["id"], "error": self.rpc_error}
        if self.sse:
            return httpx.Response(200, text="event: message\r\ndata: " + json.dumps(response) + "\r\n\r\n",
                                  headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=response)


def clients(server):
    stack = ExitStack()
    stack.enter_context(patch("httpx.Client", side_effect=lambda **kw: REAL_CLIENT(transport=httpx.MockTransport(server), **kw)))
    stack.enter_context(patch("httpx.AsyncClient", side_effect=lambda **kw: REAL_ASYNC_CLIENT(transport=httpx.MockTransport(server), **kw)))
    return stack


def default_tools(provider=None):
    # Isolate old Eezo-registered specialty tools; exercise the real selector.
    with patch.dict(research.__dict__, {
        "SimilarWebSearch": MagicMock(return_value=SimpleNamespace(name="traffic")),
        "ExaCompanySearch": MagicMock(return_value=SimpleNamespace(name="company")),
        "NewsSearch": MagicMock(return_value=SimpleNamespace(name="news")),
        "YouComSearch": MagicMock(return_value=SimpleNamespace(name="you-com-search")),
    }):
        return create_default_tools(provider)


class CallerTests(unittest.TestCase):
    def test_fresh_default_search_and_fetch_requests(self):
        with patch.dict(os.environ, {}, clear=True):
            search, fetch, *_ = default_tools()
            server = Server(sse=True)
            with clients(server):
                output = search.invoke({"query": "research question"})
                fetched = fetch.invoke({"query": "read evidence", "urls": [HIT["url"]]})
        self.assertEqual(output.content[0].content, EXCERPT)
        self.assertEqual(output.content[0].url, HIT["url"])
        self.assertEqual(fetched.content[0].source, "parallel")
        calls = []
        for req in server.requests:
            self.assertEqual(str(req.url), MCP_URL)
            self.assertEqual(req.headers["user-agent"], USER_AGENT)
            self.assertNotIn("authorization", req.headers)
            self.assertEqual(req.extensions["timeout"]["read"], 60)
            payload = json.loads(req.content)
            if payload["method"] != "initialize":
                self.assertEqual(req.headers["Mcp-Session-Id"], "server-session")
                self.assertEqual(req.headers["MCP-Protocol-Version"], "2025-03-26")
            if payload["method"] == "tools/call":
                calls.append(payload["params"])
        self.assertEqual(calls[0]["name"], "web_search")
        self.assertEqual(calls[0]["arguments"]["search_queries"], ["research question"])
        self.assertEqual(calls[1]["name"], "web_fetch")
        self.assertEqual(calls[1]["arguments"]["urls"], [HIT["url"]])
        self.assertEqual(calls[0]["arguments"]["session_id"], calls[1]["arguments"]["session_id"])

    def test_existing_credentials_overrides_and_disable(self):
        with patch.dict(os.environ, {"YOUCOM_API_KEY": "saved-key"}, clear=True):
            self.assertEqual(default_tools()[0].name, "you-com-search")
            self.assertIsInstance(default_tools("parallel")[0], ParallelSearch)
        with patch.dict(os.environ, {"RESEARCH_WEB_SEARCH": "youcom"}, clear=True):
            self.assertEqual(default_tools()[0].name, "you-com-search")
            self.assertIsInstance(default_tools("parallel")[0], ParallelSearch)
        with patch.dict(os.environ, {"RESEARCH_WEB_SEARCH": "none"}, clear=True), clients(Server()):
            self.assertEqual([t.name for t in default_tools()], ["traffic", "company", "news"])
        with patch.dict(os.environ, {"YOUCOM_API_KEY": "   "}, clear=True):
            self.assertIsInstance(default_tools()[0], ParallelSearch)
        with self.assertRaises(ValueError):
            default_tools("invalid")

    def test_empty_results_text_fallback_and_full_fetch(self):
        for data, expected in [({"results": []}, 0), ({"results": [dict(HIT, title=None, full_content="Full page")]}, 1)]:
            with clients(Server({"content": [{"type": "text", "text": json.dumps(data)}]})):
                result = ParallelFetch().invoke({"query": "read", "urls": [HIT["url"]]})
            self.assertEqual(len(result.content), expected)
            if expected:
                self.assertEqual(result.content[0].title, HIT["url"])
                self.assertEqual(result.content[0].content, "Full page")

    def test_errors_surface_without_fallback(self):
        cases = [(Server(status=429), httpx.HTTPStatusError),
                 (Server(rpc_error={"code": -32603, "message": "failed"}), ParallelMCPError),
                 (Server({"isError": True, "content": [{"type": "text", "text": "quota exceeded"}]}), ParallelMCPError),
                 (Server({"structuredContent": {"results": [], "errors": [{"url": HIT["url"], "error_type": "failed"}]}}), ParallelMCPError),
                 (Server({"content": [{"type": "text", "text": "not JSON"}]}), ParallelMCPError),
                 (Server({"structuredContent": {"wrong": []}}), ParallelMCPError),
                 (Server({"structuredContent": {"results": [dict(HIT, url="file:///tmp/data")]}}), ParallelMCPError)]
        for server, error in cases:
            with self.subTest(error=error, result=server.tool_result), clients(server):
                with self.assertRaises(error):
                    ParallelSearch().invoke({"query": "question"})
            self.assertTrue(all(str(r.url) == MCP_URL for r in server.requests))
            self.assertLessEqual(len(server.requests), 4)

    def test_timeout_propagates_and_input_urls_are_validated(self):
        seen = []
        def timeout(request):
            seen.append(request)
            raise httpx.ReadTimeout("timed out", request=request)
        with clients(timeout), self.assertRaises(httpx.ReadTimeout):
            ParallelSearch(timeout=0.5).invoke({"query": "question"})
        self.assertEqual(seen[0].extensions["timeout"]["read"], 0.5)
        server = Server()
        with clients(server):
            for url in ["file:///etc/passwd", "https://user:password@example.org", "https:///missing-host"]:
                with self.assertRaises(ValueError):
                    ParallelFetch().invoke({"query": "read", "urls": [url]})
        self.assertEqual(server.requests, [])

    def test_async_search_and_fetch(self):
        server = Server()
        async def run():
            a = await ParallelSearch().ainvoke({"query": "question"})
            b = await ParallelFetch().ainvoke({"query": "read", "urls": [HIT["url"]]})
            return a, b
        with clients(server):
            a, b = asyncio.run(run())
        self.assertEqual(a.content[0].content, b.content[0].content)
        self.assertEqual(len(server.requests), 8)
        self.assertTrue(all(r.headers["user-agent"] == USER_AGENT for r in server.requests))

    def test_async_cancellation_closes_client(self):
        async def run():
            started = asyncio.Event()
            closed = asyncio.Event()
            async def wait(request):
                started.set()
                await asyncio.Event().wait()
            class ClosingTransport(httpx.MockTransport):
                async def aclose(self):
                    closed.set()
            with patch("httpx.AsyncClient", side_effect=lambda **kw: REAL_ASYNC_CLIENT(transport=ClosingTransport(wait), **kw)):
                task = asyncio.create_task(ParallelSearch().ainvoke({"query": "question"}))
                await asyncio.wait_for(started.wait(), 2)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertTrue(closed.is_set())
        asyncio.run(run())

    def test_async_errors_and_timeouts_propagate(self):
        async def run_error():
            with clients(Server(status=503)):
                with self.assertRaises(httpx.HTTPStatusError):
                    await ParallelFetch().ainvoke({"query": "read", "urls": [HIT["url"]]})
        async def run_timeout():
            async def timeout(request):
                self.assertEqual(request.extensions["timeout"]["read"], 0.5)
                raise httpx.ReadTimeout("timed out", request=request)
            with clients(timeout):
                with self.assertRaises(httpx.ReadTimeout):
                    await ParallelSearch(timeout=0.5).ainvoke({"query": "question"})
        asyncio.run(run_error())
        asyncio.run(run_timeout())


class DispatcherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Only external service boundaries are mocked. LangChain invocation,
        # ResearchTask dispatch/result consumption, and ContentDB are real.
        with patch("eezo.Eezo"), patch("langfuse.Langfuse"), patch("groq.Groq"), patch("openai.Client"):
            cls.module = importlib.import_module("research_agent.research_task")
            cls.agent_module = importlib.import_module("research_agent.research_agent")

    def collect(self, tools, selected="parallel-search", args=None):
        module = self.module
        response = SimpleNamespace(additional_kwargs={"tool_calls": [{"function": {"name": selected, "arguments": json.dumps(args or {})}}]})
        model = MagicMock()
        model.bind_tools.return_value.invoke.return_value = response
        trace = SimpleNamespace(id="trace")
        message = MagicMock()
        task = module.ResearchTask("1", "Research topic", [], trace, None)
        db = module.ContentDB()
        with patch.object(module, "ChatOpenAI", return_value=model), patch.object(module, "l"), patch.object(module, "WebBaseLoader") as loader:
            loader.return_value.aload.return_value = []
            result = task.collect_content(db, message, tools, "Research topic")
        return result, db, model, message

    def test_scripted_model_dispatch_and_database_consumption(self):
        server = Server()
        with patch.dict(os.environ, {}, clear=True), clients(server):
            tools = default_tools()
            result, db, model, message = self.collect(tools)
        self.assertEqual(model.bind_tools.call_args.args[0], tools)
        self.assertEqual(result[0].url, HIT["url"])
        self.assertEqual(db.get_doc_by_url(HIT["url"]).content, EXCERPT)
        self.assertTrue(result[0].id)
        self.assertTrue(message.notify.called)
        call = json.loads(server.requests[-1].content)
        self.assertEqual(call["params"]["arguments"]["objective"], "Research topic")

    def test_scripted_model_fetch_dispatch(self):
        server = Server()
        with patch.dict(os.environ, {}, clear=True), clients(server):
            result, db, *_ = self.collect(default_tools(), "parallel-fetch", {"urls": [HIT["url"]]})
        self.assertEqual(db.get_doc_by_url(HIT["url"]).source, "parallel")
        self.assertEqual(json.loads(server.requests[-1].content)["params"]["name"], "web_fetch")

    def test_explicit_allowlist_and_provider_failure(self):
        server = Server(status=500)
        with clients(server):
            with self.assertRaises(StopIteration):
                self.collect([])  # An ungranted model tool call cannot execute.
            self.assertEqual(server.requests, [])
            with self.assertRaises(httpx.HTTPStatusError):
                self.collect([ParallelSearch()])
        with patch.object(self.agent_module, "Langfuse"):
            explicit = []
            agent = self.agent_module.ResearchAgent(explicit)
        self.assertIs(agent.tools, explicit)

    def test_app_entrypoint_uses_selector(self):
        from langchain.pydantic_v1 import BaseModel
        class Input(BaseModel):
            query: str
        client = MagicMock()
        client.get_agent.side_effect = lambda name: SimpleNamespace(
            agent_id=name, description="Existing research tool", input_model=Input)
        client.on.side_effect = lambda name: lambda handler: handler
        with patch.dict(os.environ, {}, clear=True), patch("eezo.Eezo", return_value=client), \
                patch("langfuse.Langfuse"), patch("groq.Groq"), patch("openai.Client"), \
                patch("dotenv.load_dotenv"):
            app = runpy.run_path("app.py")
        self.assertEqual([t.name for t in app["research_agent"].tools], [
            "parallel-search", "parallel-fetch", "similar-web-search", "exa-company-search", "news-search"])
        client.connect.assert_called_once()


if __name__ == "__main__":
    unittest.main()
