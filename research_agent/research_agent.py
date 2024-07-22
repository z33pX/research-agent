from utils.langfuse_json_model_wrapper import langfuse_json_model_wrapper
from utils.langfuse_model_wrapper import langfuse_model_wrapper
from .research_task_scheduler import TaskScheduler
from .research_task import ResearchTask, TaskResult
from langfuse.client import StatefulTraceClient
from eezo.interface import Context
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from langfuse import Langfuse
from datetime import datetime
from typing import List, Dict, Any
from prompts import Prompt
from eezo import Eezo

import json


l = Langfuse()
e = Eezo()

generate_outline = Prompt("research-agent-generate-outline")
outline_to_dag = Prompt("research-agent-outline-to-dag-conversion")
research_section_summarizer = Prompt("research-section-summarizer")


class Question(BaseModel):
    """
    Represents an individual research question.

    Attributes:
        id (str): A unique identifier for each question, reflecting its position and dependency structure.
        text (str): The text of the question.
        dependencies (List[str]): A list of IDs that this question depends on. An empty array indicates no dependencies.
    """

    id: str = Field(
        ...,
        description="A unique identifier for each question, reflecting its position and dependency structure.",
    )
    text: str = Field(..., description="The text of the question.")
    dependencies: List[str] = Field(
        default_factory=list,
        description="A list of IDs that this question depends on. An empty array indicates no dependencies.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts the Question to a dictionary.

        Returns:
            dict: The Question as a dictionary.
        """
        return {
            "id": self.id,
            "text": self.text,
            "dependencies": self.dependencies,
        }


class ResearchOutline(BaseModel):
    """
    Represents a research outline consisting of a list of questions.

    Attributes:
        questions (List[Question]): A list of main questions and subquestions.
    """

    questions: List[Question] = Field(
        ...,
        description="A list of main questions and subquestions.",
        min_items=1,
    )

    def to_dict(self) -> Dict["str", Any]:
        """
        Converts the ResearchOutline to a dictionary.

        Returns:
            dict: The ResearchOutline as a dictionary.
        """
        return {"questions": [question.to_dict() for question in self.questions]}


class ResearchAgent:
    """
    Orchestrates the research process using various tools and prompts.

    Attributes:
        tools (List[BaseTool]): A list of tools available for the research tasks.
    """

    def __init__(self, tools: List[BaseTool]):
        """
        Initializes the ResearchAgent with a list of tools and an instance of the Langfuse client.

        Args:
            tools (List[BaseTool]): A list of tools available for the research tasks.
        """
        self.tools = tools
        self.langfuse = Langfuse()

    def invoke(self, eezo_context: Context, **kwargs) -> None:
        """
        Executes the research process.

        Args:
            eezo_context (Context): The eezo_context to communicate with.
            **kwargs: Additional keyword arguments, including the user's query.
        """

        trace: StatefulTraceClient = self._start_trace()
        self._send_message(eezo_context, trace, "Generating outline...")

        # Genreate oultine
        outline: str = self._generate_outline(trace, kwargs["query"])
        self._send_message(eezo_context, trace, "Generating outline... done.", outline)

        # Convert outline to DAG
        research_outline: ResearchOutline = self._convert_outline_to_dag(trace, outline)
        self._send_message(eezo_context, trace, "Planning tasks... done.")

        # Plan and execute tasks
        results: List[TaskResult] = self._plan_and_execute(
            research_outline, trace, eezo_context
        )

        # Generate final report
        final_report = self._generate_final_report(results, trace)
        self._send_message(
            eezo_context, trace, "Generating final report...", final_report
        )

        # Save final report to json file
        self._save_final_report(
            outline, kwargs["query"], research_outline, results, final_report
        )

    def _start_trace(self) -> StatefulTraceClient:
        """
        Starts a new Langfuse trace for the research process.

        Returns:
            StatefulTraceClient: The trace client instance.
        """
        return self.langfuse.trace(name="ResearchAgent")

    def _generate_outline(self, trace, query: str) -> str:
        """
        Generates the research outline using the provided query.

        Args:
            trace (StatefulTraceClient): The trace client instance.
            query (str): The user's query.

        Returns:
            str: The generated research outline.
        """
        system_prompt = generate_outline.compile(user_prompt=query)
        return langfuse_model_wrapper(
            name="GenerateOutline",
            trace=trace,
            system_prompt=system_prompt,
            prompt=generate_outline,
            user_prompt=query,
        )

    def _convert_outline_to_dag(self, trace, outline: str) -> ResearchOutline:
        """
        Converts the research outline into a directed acyclic graph (DAG).

        Args:
            trace (StatefulTraceClient): The trace client instance.
            outline (str): The research outline.

        Returns:
            ResearchOutline: The converted research outline as a DAG.
        """
        system_prompt = outline_to_dag.compile(output_schema="", outline=outline)
        return langfuse_json_model_wrapper(
            name="ConvertOutlineToDAG",
            trace=trace,
            system_prompt=system_prompt,
            user_prompt="Parse the outline into the json schema",
            prompt=outline_to_dag,
            base_model=ResearchOutline,
        )

    def _plan_and_execute(
        self, research_outline: ResearchOutline, trace, eezo_context: Context
    ) -> List[TaskResult]:
        """
        Executes the research tasks based on the DAG.

        Args:
            research_outline (ResearchOutline): The research outline as a DAG.
            trace (StatefulTraceClient): The trace client instance.
            eezo_context (Context): The eezo_context to communicate with.

        Returns:
            List[TaskResult]: The results of the research tasks.
        """
        task_list = []
        for question in research_outline.questions:
            task_list.append(
                ResearchTask(
                    id=question.id,
                    research_topic=question.text,
                    dependencies=question.dependencies,
                    trace=trace,
                    eezo_context=eezo_context,
                )
            )

        scheduler = TaskScheduler(task_list, self.tools)
        scheduler.execute()
        return scheduler.get_results()

    def _generate_final_report(self, results: List[TaskResult], trace) -> str:
        """
        Generates the final report from the research results.

        Args:
            results (List[TaskResult]): The results of the research tasks.
            trace (StatefulTraceClient): The trace client instance.

        Returns:
            str: The final report.
        """
        final_report = ""
        for task_result in results:
            if task_result.error is not "":
                continue
            if len(task_result.content_used) == 0:
                final_report += f"{task_result.id} {task_result.research_topic}\nNo content found.\n\n"
            else:
                system_prompt = research_section_summarizer.compile(
                    research_topic=task_result.research_topic,
                    section_notes=task_result.result,
                )
                section_summary = langfuse_model_wrapper(
                    name="GenerateSectionSummary",
                    trace=trace,
                    system_prompt=system_prompt,
                    prompt=research_section_summarizer,
                    user_prompt="Generate a summary of the section",
                    model="llama3-70b-8192",
                    host="groq",
                )
                final_report += f"**{task_result.id} {task_result.research_topic}**\n{section_summary}\n\n"
        return final_report

    def _save_final_report(
        self,
        outline: str,
        query: str,
        research_outline: ResearchOutline,
        results: List[TaskResult],
        final_report: str,
    ) -> None:
        """
        Saves the final research report to a JSON file.

        Args:
            outline (str): The research outline.
            query (str): The user's query.
            research_outline (ResearchOutline): The research outline as a DAG.
            results (list): The results of the research tasks.
            final_report (str): The final report.
        """
        human_readable_timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        with open(f"research_{human_readable_timestamp}.json", "w") as f:
            json.dump(
                {
                    "outline": outline,
                    "query": query,
                    "dag": json.loads(research_outline.model_dump_json()),
                    "results": [result.to_dict() for result in results],
                    "final_report": final_report,
                },
                f,
                indent=4,
            )

    def _send_message(
        self, eezo_context: Context, trace, text: str, content: str = ""
    ) -> None:
        """
        Sends a message to the Eezo eezo_context, optionally including additional content.

        Args:
            eezo_context (Context): The eezo_context to communicate with.
            trace (StatefulTraceClient): The trace client instance.
            text (str): The text message to send.
            content (str): Additional content to include in the message.
        """
        if eezo_context:
            span = self.langfuse.span(trace_id=trace.id, name="EezoMessage")
            m = eezo_context.new_message()
            c = m.add("text", text=text)
            if content:
                m.replace(c.id, "text", text=content)
                span.end(
                    output={"text": content},
                )
            else:
                span.end(
                    output={"text": text},
                )
            m.notify()
