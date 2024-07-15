from .common.model_schemas import ContentItem, ResearchToolOutput
from .base_tool import ResearchTool
from langchain.tools import BaseTool

from langchain_community.document_loaders import WebBaseLoader
from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from langfuse import Langfuse
from typing import Type, List
from prompts import Prompt
from eezo import Eezo

import logging
import requests
import os

l = Langfuse()
e = Eezo()

summarize_search_results = Prompt("summarize-search-results")
agent = e.get_agent("exa_company_search")
if agent is None:
    agent = e.create_agent(
        agent_id="exa_company_search",
        description="Invoke when the user wants to search one or multiple companies. This tool only finds companies that might fit the user request and returns only company urls and landingpage summaries. No other data. This tool cannot compare companies or find similar companies.",
    )


class ExaCompanySearch(BaseTool):
    name: str = agent.name
    description: str = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    include_summary: bool = False

    def __init__(self, include_summary: bool = False):
        super().__init__()
        self.include_summary = include_summary

    def scrape_pages(self, urls: List[str]):
        # https://python.langchain.com/docs/integrations/document_loaders/web_base/
        loader = WebBaseLoader(
            urls,
            proxies={
                # https://docs.zyte.com/zyte-api/usage/proxy-mode.html#zyte-api-proxy-mode
                scheme: "http://{os.getenv('ZYTE_API_KEY')}:@api.zyte.com:8011"
                for scheme in ("http", "https")
            },
        )
        loader.requests_per_second = 5
        try:
            docs = loader.aload()
            for doc in docs:
                while "\n\n" in doc.page_content:
                    doc.page_content = doc.page_content.replace("\n\n", "\n")
                while "  " in doc.page_content:
                    doc.page_content = doc.page_content.replace("  ", " ")
        except Exception as error:
            logging.error(f"Error scraping additional content: {error}")
            docs = []
        # span.end()
        return docs

    def _run(self, **kwargs) -> ResearchToolOutput:
        # https://docs.exa.ai/reference/search

        url = "https://api.exa.ai/search"

        payload = {
            "category": "company",
            "query": kwargs["query"],
            "contents": {"text": {"includeHtmlTags": False}},
            "numResults": 3,
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "x-api-key": os.getenv("EXA_API_KEY"),
        }

        response = requests.post(url, json=payload, headers=headers)

        urls = [result["url"] for result in response.json()["results"]]
        webpages = self.scrape_pages(urls)

        content = []
        for result in response.json()["results"]:
            webpage = next(
                (
                    doc.page_content
                    for doc in webpages
                    if doc.metadata.get("source") == result["url"]
                ),
                "",
            )
            title = result.get("title", "") + " - " + result.get("publishedDate", "")
            content.append(
                ContentItem(
                    url=result["url"],
                    title=title,
                    snippet=result.get("text", ""),
                    content=webpage,
                    source="Exa AI",
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
