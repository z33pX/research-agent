from .common.model_schemas import ContentItem, ResearchToolOutput
from langchain.tools import BaseTool

from utils.langfuse_model_wrapper import langfuse_model_wrapper
from langchain.pydantic_v1 import BaseModel
from eezo.interface.message import Message
from bs4 import BeautifulSoup
from langfuse import Langfuse
from typing import Type
from eezo import Eezo

import requests
import os

l = Langfuse()
e = Eezo()

agent = e.get_agent(os.environ["TOOL_SIMILAR_WEB_SEARCH"])
generate_paragraph = l.get_prompt("summarize-text-into-three-paragraphs")
summarize_similarweb = l.get_prompt("summarize-similarweb-search-result")


class SimilarWebSearch(BaseTool):
    name: str = agent.name
    description: str = agent.description
    args_schema: Type[BaseModel] = agent.input_model
    user_prompt: str | None = None
    chat_message: Message | None
    include_summary: bool = False

    def __init__(
        self,
        include_summary: bool = False,
        user_prompt: str = "",
        chat_message: Message | None = None,
    ):
        super().__init__()
        self.include_summary = include_summary
        self.chat_message = chat_message
        self.user_prompt = user_prompt

    def brave_search(self, query, count):
        """
        Perform a search query using Brave Search API.

        Args:
        - query: The search query string.

        Returns:
        - The search results as a JSON object.
        """
        # Define the endpoint URL for the Brave Web Search API
        url = "https://api.search.brave.com/res/v1/web/search"

        # Prepare headers with the API key and accept headers
        headers = {
            "X-Subscription-Token": os.getenv("BRAVE_SEARCH_API_KEY"),
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
        }

        # Prepare query parameters
        params = {"q": query, "count": count}

        # Make the GET request to the Brave Search API
        response = requests.get(url, headers=headers, params=params)

        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "error": "Failed to fetch search results",
                "status_code": response.status_code,
            }

    def _run(self, **kwargs) -> ResearchToolOutput:
        entity_name = kwargs.get("entity_name")
        instructions = kwargs.get("instructions")

        search_results = self.brave_search(entity_name + " website", count=1)
        result = search_results["web"]["results"][0]
        domain = result["url"].split("/")[2]

        if self.chat_message:
            self.chat_message.add(
                "text", text=f"Searching on SimilarWeb for [{entity_name}]({domain})..."
            )
            self.chat_message.notify()

        url = f"https://www.similarweb.com/website/{domain}/#overview"
        response = requests.post(
            "https://api.zyte.com/v1/extract",
            auth=(os.getenv("ZYTE_API_KEY"), ""),
            json={"url": url, "browserHtml": True},
        )

        if self.chat_message:
            self.chat_message.add("text", text="Generating a report...")
            self.chat_message.notify()

        text = ""
        if response.status_code == 200:
            soup = BeautifulSoup(response.json().get("browserHtml", ""), "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            snippet = langfuse_model_wrapper(
                name="GenerateParagraph",
                system_prompt=generate_paragraph.compile(text=text),
                prompt=generate_paragraph,
                user_prompt=f"Generate a snippet based on the given text:\n{text}",
                model="gpt-3.5-turbo-1106",
                temperature=0.7,
            )

            content = [
                ContentItem(
                    url=url,
                    title=f"SimilarWeb data for {entity_name}",
                    snippet=snippet,
                    content=text,
                    source="SimilarWeb",
                )
            ]
        else:
            content = []

        summary = ""
        if self.include_summary and len(content) > 0:
            system_prompt = summarize_similarweb.compile(
                text=text, instructions=instructions, user_prompt=self.user_prompt
            )
            summary = langfuse_model_wrapper(
                name="SimilarWebSearchSummary",
                system_prompt=system_prompt,
                user_prompt="Generate a detailed report based on the given text.",
                prompt=summarize_similarweb,
                temperature=0.7,
            )
        return ResearchToolOutput(content=content, summary=summary)
