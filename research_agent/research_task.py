from utils.langfuse_json_model_wrapper import langfuse_json_model_wrapper
from utils.langfuse_model_wrapper import langfuse_model_wrapper
from .db import ContentDB

from tools.research.common.model_schemas import ContentItem
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langfuse.client import StatefulTraceClient
from typing import List, Dict, Any, Optional
from eezo.interface.message import Message
from eezo.interface import Context
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel
from pinecone import Pinecone
from langfuse import Langfuse


import logging
import openai
import uuid
import json
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
oc = openai.Client()
l = Langfuse()


select_content = l.get_prompt("research-agent-select-content")
extract_notes = l.get_prompt("research-agent-extract-notes-from-webpages")
assessing_information_sufficiency = l.get_prompt(
    "research-agent-assessing-information-sufficiency"
)


class TaskResult(BaseModel):
    result: Optional[str] = ""
    content_used: Optional[List[str]] = []
    content_urls: Optional[List[str]] = []
    research_topic: Optional[str] = ""
    id: str
    error: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "result": self.result,
            "content_used": self.content_used,
            "content_urls": self.content_urls,
            "research_topic": self.research_topic,
            "id": self.id,
            "error": self.error,
        }


class ResearchTask:
    def __init__(
        self,
        id: str,
        research_topic: str,
        dependencies: List[str],
        trace: StatefulTraceClient,
        eezo_context: Context,
    ):
        self.id = id
        self.research_topic = research_topic
        self.dependencies = dependencies
        self.trace = trace
        self.eezo_context = eezo_context

    def decide_what_to_use(
        self,
        db: ContentDB,
        m: Message,
        content_ids: List[str],
        research_topic: str,
    ) -> List[str]:
        """
        This function will decide what content to use for generating the summary.

        Args:
            db (ContentDB): The database object to interact with the content database.
            m (Message): The message object to send notifications.
            content_ids (List[str]): The content ids to decide what to use.
            research_topic (str): The research topic for which to decide what to use.

        Returns:
            List[str]: The content ids to use for generating the summary.
        """
        span = l.span(
            trace_id=self.trace.id,
            name="decide_what_to_use",
            input={"content_ids": content_ids, "research_topic": research_topic},
        )
        # 1. Get all snippets for each content_id.
        content_objs: List[ContentItem | None] = [
            db.get_doc_by_id(content_id) for content_id in content_ids
        ]
        content_objs: List[ContentItem] = [
            content for content in content_objs if content
        ]

        # Prepare the prompt.
        formatted_snippets = ""
        for i, content in enumerate(content_objs):
            formatted_snippets += f"{i}: {content}\n"

        system_prompt = select_content.compile(
            research_topic=research_topic, formatted_snippets=formatted_snippets
        )

        class Response(BaseModel):
            snippet_indeces: List[int]

        # Ask the LLM which content seems most relevant.
        response: Response = langfuse_json_model_wrapper(
            trace=self.trace,
            observation_id=span.id,
            name="SelectContent",
            system_prompt=system_prompt,
            user_prompt="Pick the snippets you want to include in the summary.",
            prompt=select_content,
            base_model=Response,
        )

        # Parse the response to get the chosen content_ids.
        choosen_ids = [i for i in response.snippet_indeces if i < len(content_ids)]

        logging.info("------" * 10)
        logging.info(f"Chosen snippets for question '{research_topic}'")
        results = []
        m.add("text", text="**Decided to use:**\n\n")
        for idx in choosen_ids:
            snippet = content_objs[idx].snippet[:150]
            logging.info(f"- {snippet} ...")
            m.add(
                "text", text=f"- [{content_objs[idx].title}]({content_objs[idx].url})"
            )
            results.append({"content_id": content_ids[idx], "snippet": snippet})
        logging.info("------" * 10)

        content_ids_to_use = [content_ids[i] for i in choosen_ids]
        span.end(output={"results": results})
        return content_ids_to_use

    def check_if_more_info_needed(
        self,
        db: ContentDB,
        m: Message,
        research_topic: str,
        content_ids: List[str],
    ) -> List[str]:
        """
        This function will check if more information is needed to generate the summary.

        Args:
            db (ContentDB): The database object to interact with the content database.
            m (Message): The message object to send notifications.
            research_topic (str): The research topic for which to check if more information is needed.
            content_ids (List[str]): The content ids to check if more information is needed.

        Returns:
            List[str]: The additional questions that need to be answered.
        """
        m.add("text", text="Checking if more information is needed...\n\n")
        m.notify()

        span = l.span(
            trace_id=self.trace.id,
            name="check_if_more_info_needed",
            input={"research_topic": research_topic, "content_ids": content_ids},
        )
        # 1. Get the content snippets for the given content_ids.
        content_objs: List[ContentItem] = [
            db.get_doc_by_id(content_id) for content_id in content_ids
        ]
        content_snippets = [content.snippet for content in content_objs if content]

        # 2. Prepare the prompt.
        formatted_content = "Available data:\n" + "\n".join(content_snippets)
        system_prompt = assessing_information_sufficiency.compile(
            research_topic=research_topic, formatted_content=formatted_content
        )

        class Response(BaseModel):
            more_info_needed: bool
            research_topics: List[str]

        # 3. Ask the LLM if more information is needed.
        response: Response = langfuse_json_model_wrapper(
            trace=self.trace,
            observation_id=span.id,
            name="AssessInformationSufficiency",
            system_prompt=system_prompt,
            user_prompt="Is the given content enough to generate the summary for the research topic?",
            prompt=assessing_information_sufficiency,
            base_model=Response,
        )

        # 4. Log the response.
        try:
            logging.info("------" * 10)
            logging.info(f"Additional questions next to '{research_topic}':")
            span = l.span(trace_id=self.trace.id, name="EezoMessage")
            m.add("text", text=f"**Expanding on question** {research_topic}\n\n")
            results = []
            for question in response.research_topics:
                logging.info(f" - {question[:150]} ...")
                m.add("text", text=f"**-** {question}")
            m.notify()
            logging.info("------" * 10)
            results.append(
                {
                    "additional_questions": response.research_topics,
                }
            )
            span.end(output={"results": results})
        except Exception as error:
            logging.error(f"Error in check_if_more_info_needed: {error}")

        if len(response.research_topics) == 0:
            logging.info(
                f"{self.id} - Content is sufficient for '{self.research_topic}'"
            )
            m.add("text", text="This content is sufficient for the summary:\n\n")
            for content_id in content_ids:
                content: ContentItem = db.get_doc_by_id(content_id)
                if content:
                    m.add("text", text=f"- [{content.title}]({content.url})")
                else:
                    logging.error(f"{self.id} - Content not found for id {content_id}")
            m.notify()

        return response.research_topics

    def collect_content(
        self,
        db: ContentDB,
        m: Message,
        tools: List[BaseTool],
        research_topic: str,
    ) -> List[ContentItem]:
        """
        This function will collect content for the given research_topic using the provided tools.

        Args:
            db (ContentDB): The database object to interact with the content database.
            m (Message): The message object to send notifications.
            tools (List[BaseTool]): The tools to use for collecting content.
            research_topic (str): The research topic for which to collect content.

        Returns:
            List[ContentItem]: The content items collected for the research topic.
        """
        span = l.span(
            trace_id=self.trace.id,
            name="collect_content",
            input={"research_topic": research_topic},
        )

        existing_content: List[ContentItem] = []
        results: List[ContentItem] = []

        # 1. Execute a tool agent to select tools to execute that can help in collecting content.
        tool_span = l.span(
            trace_id=self.trace.id,
            parent_observation_id=span.id,
            name="ToolAgent",
            input={
                "research_topic": research_topic,
                "tools": [tool.name for tool in tools],
            },
        )

        for attempt in range(1, 4):  # Attempt 3 times, counting starts from 1
            model_with_tools = ChatOpenAI(model="gpt-3.5-turbo").bind_tools(tools)
            openai_result = model_with_tools.invoke(
                [HumanMessage(content=research_topic)],
                config={"callbacks": [tool_span.get_langchain_handler()]},
            )

            if "tool_calls" in openai_result.additional_kwargs:
                break
            logging.info(f"No tools to execute. Attempt {attempt} failed.")
        else:
            logging.error("Failed to execute tools after 3 attempts. Exiting.")
            return existing_content
        tool_span.end(
            output={"tool_calls": openai_result.additional_kwargs["tool_calls"]}
        )

        # 2. Execute the tools.
        tool_execution_span = l.span(
            trace_id=self.trace.id,
            parent_observation_id=span.id,
            name="ToolsExecution",
            input={
                "tools_to_be_called": openai_result.additional_kwargs["tool_calls"],
            },
        )

        for tool_call in openai_result.additional_kwargs["tool_calls"]:
            tool = next(t for t in tools if t.name == tool_call["function"]["name"])
            payload = json.loads(tool_call["function"]["arguments"])
            payload["query"] = research_topic  # Add this as a default argument.
            results.extend(
                tool.invoke(
                    payload,
                    config={"callbacks": [tool_execution_span.get_langchain_handler()]},
                ).content
            )
        tool_execution_span.end(
            output={"results": [content.dict() for content in results]}
        )

        # 3. Check if urls are already in the content to prevent scraping them again
        # except if the content is less than 500 characters. If so, we scrape the content.
        content_result = []
        for content in results:
            content_obj: ContentItem = db.get_doc_by_url(content.url)
            if content_obj and len(content_obj.content) >= 500:
                logging.info(
                    f"Content already exists for {content.url} and is > 500 characters."
                )
                existing_content.append(content_obj)
            else:
                content_result.append(content)
        results = content_result

        # 4. Content < 500? If yes, we scrape the provided URL to enrich the content.
        urls_to_scrape = [
            {"index": i, "url": content.url}
            for i, content in enumerate(results)
            if len(content.content) < 500
        ]
        payload = [url["url"] for url in urls_to_scrape]

        scraping_span = l.span(
            trace_id=self.trace.id,
            parent_observation_id=span.id,
            name="ScrapeContent",
            input={"payload": payload, "urls": [url["url"] for url in urls_to_scrape]},
            metadata={"proxy": "zyte", "method": "WebBaseLoader"},
        )
        # https://python.langchain.com/docs/integrations/document_loaders/web_base/
        logging.info(
            f"Scraping content from {len(urls_to_scrape)} URLs to enrich the content ."
        )
        loader = WebBaseLoader(
            payload,
            proxies={
                # https://docs.zyte.com/zyte-api/usage/proxy-mode.html#zyte-api-proxy-mode
                scheme: "http://{os.getenv('ZYTE_API_KEY')}:@api.zyte.com:8011"
                for scheme in ("http", "https")
            },
        )
        loader.requests_per_second = 5
        try:
            docs = loader.aload()
        except Exception as error:
            logging.error(f"Error scraping additional content: {error}")
            docs = []
        scraping_span.end(output={"docs": [doc.page_content for doc in docs]})

        # [Document(page_content=" ... ", lookup_str='', metadata={'source': 'https://www.espn.com/'}, lookup_index=0)]
        for i, doc in enumerate(docs):
            logging.info(
                f"Scraped content from {urls_to_scrape[i]['url']} successfully."
            )
            results[urls_to_scrape[i]["index"]].content = doc.page_content

        for content in results:
            content.id = str(uuid.uuid4())

        # Add existing_content to results
        results.extend(existing_content)

        span.end(output={"results": [content.dict() for content in results]})

        if len(results) > 0:
            m.add("text", text=f"**Found new content** for {self.research_topic}:\n\n")
            for content in results:
                db.upsert_doc(content)
                # logging.info(f"{self.id} - - {content.snippet}")
                m.add("text", text=f"- [{content.title}]({content.url})")
            m.notify()
        else:
            logging.error(f"{self.id} - No content found for '{self.research_topic}'")
            m.add(
                "text",
                text=f"No content found for this research topic {self.research_topic}",
            )
            m.notify()

        return results

    def execute(
        self,
        db: ContentDB,
        state: Dict[str, TaskResult],
        tools: List[BaseTool],
    ) -> TaskResult:
        logging.info(f"Executing task {self.id}:")
        relevant_state = {dep: state[dep] for dep in self.dependencies}

        # Get the content used by previous tasks.
        # length of content_ids will be zero if this is a root task.
        content_ids = [item.content_used for item in relevant_state.values()]
        content_ids = [item for sublist in content_ids for item in sublist]

        m = self.eezo_context.new_message()
        m.add("text", text=f"**Researching {self.id}** - {self.research_topic}\n\n")
        m.notify()

        if content_ids:
            # Do we need more information besides the given content?
            research_topics = self.check_if_more_info_needed(
                db, m, self.research_topic, content_ids
            )
            if len(research_topics) > 0:
                # Yes, we need more information.
                for research_topic in research_topics:
                    results = self.collect_content(db, m, tools, research_topic)
                    content_ids.extend([content.id for content in results])
        else:
            # We definitely need more information.
            results = self.collect_content(db, m, tools, self.research_topic)
            content_ids.extend([content.id for content in results])

        span = l.span(trace_id=self.trace.id, name="EezoMessage")
        span.end()

        # Select what information to use for the summary.
        content_ids = self.decide_what_to_use(db, m, content_ids, self.research_topic)

        # Process the content to generate the summary.
        content_docs: List[ContentItem] = [
            db.get_doc_by_id(content_id) for content_id in content_ids
        ]
        formatted_webpages = ""
        for i, content in enumerate(content_docs):
            if content:
                formatted_webpages += f"Webpage {i + 1}:\nTitle: {content.title}\nUrl: {content.url}\nContent: {content.content}\n\n"

        logging.info(
            f"{self.id} - Generating notes for topic '{self.research_topic}'..."
        )
        system_prompt = extract_notes.compile(
            research_topic=self.research_topic, formatted_webpages=formatted_webpages
        )
        notes = langfuse_model_wrapper(
            trace=self.trace,
            name="ConvertWebpagesToNotes",
            system_prompt=system_prompt,
            prompt=extract_notes,
            user_prompt="Generate 20 to 30 bullet point notes based on the content provided.",
            temperature=0.5,
        )

        content_urls = []
        for content_id in content_ids:
            content: ContentItem = db.get_doc_by_id(content_id)
            if content:
                content_urls.append(content.url)

        results = TaskResult(
            result=notes,
            content_used=content_ids,
            content_urls=content_urls,
            research_topic=self.research_topic,
            id=self.id,
            error="",
        )

        span = l.span(
            trace_id=self.trace.id,
            name="Embedding",
            metadata={"database": "pinecone"},
            input={"notes": notes},
        )
        index = pc.Index("research-agent")
        data = oc.embeddings.create(input=notes, model="text-embedding-3-small")
        span.end()

        span = l.span(
            trace_id=self.trace.id,
            name="Upsert",
            metadata={"database": "pinecone"},
            input={"text": notes},
        )
        index.upsert([(str(uuid.uuid4()), data.data[0].embedding, {"text": notes})])
        span.end()

        return results
