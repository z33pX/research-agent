from utils.langfuse_json_model_wrapper import langfuse_json_model_wrapper
from utils.langfuse_model_wrapper import langfuse_model_wrapper
from .db import ContentDB

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.messages import HumanMessage
from langfuse.client import StatefulTraceClient
from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from langfuse import Langfuse
from pydantic import BaseModel

from eezo.interface import Interface
from prompts import Prompt
from eezo import Eezo

import logging
import openai
import uuid
import json
import os

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
oc = openai.Client()
l = Langfuse()


select_content = Prompt("research-agent-select-content")
extract_notes = Prompt("research-agent-extract-notes-from-webpages")
assessing_information_sufficiency = Prompt(
    "research-agent-assessing-information-sufficiency"
)


class TaskResult(BaseModel):
    result: Optional[str] = ""
    content_used: Optional[List[str]] = []
    content_urls: Optional[List[str]] = []
    research_topic: Optional[str] = ""
    id: str
    error: str

    def to_dict(self):
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
        eezo_interface: Interface,
    ):
        self.id = id
        self.research_topic = research_topic
        self.dependencies = dependencies
        self.trace = trace
        self.eezo_interface = eezo_interface

    def decide_what_to_use(
        self,
        db: ContentDB,
        content_ids: List[str],
        research_topic: str,
    ) -> List[str]:
        """
        This function will ask the LLM to select the content that seems most relevant
        """
        span = l.span(
            trace_id=self.trace.id,
            name="decide_what_to_use",
            input={"content_ids": content_ids, "research_topic": research_topic},
        )
        # 1. Get all snippets for each content_id.
        content_objs = [db.get_doc_by_id(content_id) for content_id in content_ids]
        content_objs = [content for content in content_objs if content]

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
        logging.info(f"Chosen snippet for question '{research_topic}'")
        results = []
        m = self.eezo_interface.new_message()
        m.add("text", text=f"Decided to focus on:\n")
        for idx in choosen_ids:
            snippet = content_objs[idx].snippet[:150]
            logging.info(f"- {snippet} ...")
            results.append({"content_id": content_ids[idx], "snippet": snippet})
            m.add("text", text=f"- {content_objs[idx].title}")
        m.notify()
        logging.info("------" * 10)

        content_ids_to_use = [content_ids[i] for i in choosen_ids]
        span.end(output={"results": results})
        return content_ids_to_use

    def check_if_more_info_needed(
        self,
        db: ContentDB,
        research_topic: str,
        content_ids: List[str],
    ):
        """
        This function will check if the given content_ids are enough to
        generate the summary for the research_topic. If not, it will return new
        topics for which more information is needed.
        """
        span = l.span(
            trace_id=self.trace.id,
            name="check_if_more_info_needed",
            input={"research_topic": research_topic, "content_ids": content_ids},
        )
        # 1. Get the content snippets for the given content_ids.
        content_objs = [db.get_doc_by_id(content_id) for content_id in content_ids]
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
            m = self.eezo_interface.new_message()
            m.add("text", text=f"Expanding on question **{research_topic}**\n\n")
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

        return response.research_topics

    def collect_content(
        self,
        db: ContentDB,
        tools: List[BaseTool],
        research_topic: str,
    ) -> List[Dict[str, Any]]:
        """
        Collects new information based on the research topics, adds it to the content, and returns the indices of the new content .
        """
        span = l.span(
            trace_id=self.trace.id,
            name="collect_content",
            input={"research_topic": research_topic},
        )

        new_ids = []
        results = []

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
            return new_ids
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
            content_obj = db.get_doc_by_url(content.url)
            if content_obj and len(content_obj.content) >= 500:
                logging.info(
                    f"Content already exists for {content.url} and is > 500 characters."
                )
                new_ids.append(content_obj.id)
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

        span.end(output={"results": [content.dict() for content in results]})

        return results

    def execute(
        self,
        db: ContentDB,
        state: Dict[str, TaskResult],
        tools: List[BaseTool],
    ) -> TaskResult:
        relevant_state = {dep: state[dep] for dep in self.dependencies}
        logging.info(f"Executing task {self.id}")

        # Get the content used by previous tasks.
        # length of content_ids will be zero if this is a root task.
        content_ids = [item.content_used for item in relevant_state.values()]
        content_ids = [item for sublist in content_ids for item in sublist]

        m = self.eezo_interface.new_message()
        m.add("text", text=f"Researching **{self.research_topic}**...\n\n")
        m.notify()
        if content_ids:
            # Do we need more information besides the given content?
            research_topics = self.check_if_more_info_needed(
                db, self.research_topic, content_ids
            )
            if len(research_topics) > 0:
                # Yes, we need more information.
                for research_topic in research_topics:
                    results = self.collect_content(db, tools, research_topic)

                    m.add(
                        "text", text=f"Found new content for **{research_topic}**\n\n"
                    )
                    for content in results:
                        db.upsert_doc(content)
                        logging.info(f"- {content.snippet}")
                        m.add("text", text=f"- [{content.title}]({content.url})")

                    content_ids.extend([content.id for content in results])
            else:
                logging.info(f"Content is sufficient for '{self.research_topic}'")
        else:
            # We definitely need more information.
            results = self.collect_content(db, tools, self.research_topic)

            m.add("text", text=f"Found new content for **{self.research_topic}**\n\n")
            for content in results:
                db.upsert_doc(content)
                logging.info(f"- {content.snippet}")
                m.add("text", text=f"- [{content.title}]({content.url})")

            content_ids.extend([content.id for content in results])

        span = l.span(trace_id=self.trace.id, name="EezoMessage")
        m.notify()
        span.end()

        # Select what information to use for the summary.
        content_ids = self.decide_what_to_use(db, content_ids, self.research_topic)

        # Process the content to generate the summary.
        content_docs = [db.get_doc_by_id(content_id) for content_id in content_ids]
        formatted_webpages = ""
        for i, content in enumerate(content_docs):
            if content:
                formatted_webpages += f"Webpage {i + 1}:\nTitle: {content.title}\nUrl: {content.url}\nContent: {content.content}\n\n"

        system_prompt = extract_notes.compile(
            research_topic=self.research_topic, formatted_webpages=formatted_webpages
        )

        logging.info(f"Generating notes for topic '{self.research_topic}'...")
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
            content = db.get_doc_by_id(content_id)
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
