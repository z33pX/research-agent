from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool

from utils.langfuse_json_model_wrapper import langfuse_json_model_wrapper
from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel
from langfuse import Langfuse
from typing import Type, List
from prompts import Prompt
from eezo import Eezo

import logging
import os

l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_EXA_COMPANY_SEARCH"])
select_content = Prompt("research-agent-select-content")
summarize_search_results = Prompt("summarize-search-results")


class NewsSearch(BaseTool):
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
        return docs

    def decide_what_to_use(
        self, content: List[dict], research_topic: str
    ) -> List[dict]:
        formatted_snippets = ""
        for i, doc in enumerate(content):
            formatted_snippets += f"{i}: {doc['title']}: {doc['snippet']}\n"

        system_prompt = select_content.compile(
            research_topic=research_topic, formatted_snippets=formatted_snippets
        )

        class ModelResponse(BaseModel):
            snippet_indeces: List[int]

        response: ModelResponse = langfuse_json_model_wrapper(
            name="SelectContent",
            system_prompt=system_prompt,
            user_prompt="Pick the snippets you want to include in the summary.",
            prompt=select_content,
            base_model=ModelResponse,
        )

        indices = [i for i in response.snippet_indeces if i < len(content)]
        return [content[i] for i in indices]

    def _run(self, **kwargs) -> ResearchToolOutput:
        # https://python.langchain.com/docs/integrations/tools/google_serper/
        google_serper = GoogleSerperAPIWrapper(type="news", k=10)

        response = google_serper.results(query=kwargs["query"])

        news_results = response["news"]
        # exampel = {
        #     "title": "Alarum’s subsidiary NetNut launched New SERP Scraper API Product",
        #     "link": "https://www.martechcube.com/alarums-subsidiary-netnut-launched-new-serp-scraper-api-product/",
        #     "snippet": "SERP Scraper APIs allow businesses to obtain SERP data from search engines automatically. The SERP Scraper API delivers real-time...",
        #     "date": "8 hours ago",
        #     "source": "MarTech Cube",
        #     "imageUrl": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTlcKMXYBagHZJnKI4BxO9M-wmLodmLdpQjf9AKW9YmVJGam0vna-yZ2kb7Hg&s",
        #     "position": 1
        # }

        # Add x if it doesn't exist
        for news in news_results:
            if "snippet" not in news:
                news["snippet"] = ""
            if "date" not in news:
                news["date"] = ""
            if "source" not in news:
                news["source"] = ""
            if "title" not in news:
                news["title"] = ""
            if "link" not in news:
                news["link"] = ""
            if "imageUrl" not in news:
                news["imageUrl"] = ""

        selected_results = self.decide_what_to_use(
            content=news_results, research_topic=kwargs["query"]
        )

        webpage_urls = [result["link"] for result in selected_results]
        webpages = self.scrape_pages(webpage_urls)

        content = []
        for news in news_results:
            webpage = next(
                (
                    doc.page_content
                    for doc in webpages
                    if doc.metadata.get("source") == news["link"]
                ),
                "",
            )
            title = news.get("title", "") + " - " + news.get("date", "")
            content.append(
                ContentItem(
                    url=news["link"],
                    title=title,
                    snippet=news.get("text", ""),
                    content=webpage,
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
                user_prompt=f"Summarize and group the search results based on this: '{kwargs['query']}'. Include links, dates, and snippets from the search results.",
                model="llama3-70b-8192",
                host="groq",
                temperature=0.7,
            )

        return ResearchToolOutput(content=content, summary=summary)
