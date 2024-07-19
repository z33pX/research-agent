from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool

from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langfuse import Langfuse
from prompts import Prompt
from typing import Type
from eezo.agent import Agent
from eezo import Eezo

import requests
import os

l = Langfuse()
e = Eezo()

summarize_search_results = Prompt("summarize-search-results")
agent: Agent = e.get_agent("you-com-search")
if agent is None:
    agent: Agent = e.create_agent(
        agent_id="you-com-search",
        description="Invoke when the user asks a general question. It works like Google Search. Don't use this to search for companies.",
    )


class YouComSearch(BaseTool):
    name: str = agent.agent_id
    description: str = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    include_summary: bool = False

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def you_com_search(self, query):
        headers = {"X-API-Key": os.environ["YOUCOM_API_KEY"]}
        params = {"query": query}
        return requests.get(
            f"https://api.ydc-index.io/rag?query={query}",
            params=params,
            headers=headers,
        ).json()

    def _run(self, **kwargs) -> ResearchToolOutput:
        result = self.you_com_search(kwargs["query"])

        content = []
        for hit in result["hits"]:
            if type(hit["ai_snippets"]) == str:
                hit["ai_snippets"] = [hit["ai_snippets"]]
            content.append(
                ContentItem(
                    url=hit["url"],
                    title=hit["title"],
                    snippet=hit["snippet"],
                    content="\n".join(hit["ai_snippets"]),
                )
            )

        summary = ""
        if self.include_summary:
            formatted_content = "\n\n".join([f"### {item}" for item in content])

            system_prompt = summarize_search_results.compile(
                search_results_str=formatted_content, user_prompt=kwargs["query"]
            )

            summary = langfuse_model_wrapper(
                name="SummarizeSearchResults",
                system_prompt=system_prompt,
                prompt=summarize_search_results,
                user_prompt=kwargs["query"],
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )
        return ResearchToolOutput(content=content, summary=summary)
