from .research_task import ResearchTask, TaskResult
from .db import ContentDB

from langchain.tools import BaseTool
from collections import defaultdict
from typing import List, Dict, Any

import concurrent.futures
import os


class TaskScheduler:
    def __init__(
        self,
        tasks: List[ResearchTask],
        tools: List[BaseTool],
    ):
        """
        Initializes a TaskScheduler to manage and execute a list of ResearchTasks using specified tools.

        Args:
            tasks (List[ResearchTask]): A list of ResearchTask instances to be scheduled and executed.
            tools (List[BaseTool]): A list of BaseTool instances to be used by the ResearchTasks during execution.

        The constructor also initializes a content database and sets up the dependencies between tasks based on the
        ResearchTask definitions.
        """
        self.tasks: List[ResearchTask] = tasks
        self.state: Dict[str, TaskResult] = (
            {}
        )  # State to keep results and intermediate data
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.db: ContentDB = ContentDB(
            current_folder + "/db/content.db"
        )  # Initializes the content database
        self.tools: List[BaseTool] = tools  # Tools used across tasks
        self.dependents = defaultdict(
            list
        )  # Mapping from task ID to dependent task IDs
        self.in_degree = defaultdict(int)  # Count of unsatisfied dependencies per task
        self.setup_dependencies()  # Setup task dependencies
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=4
        )  # Executor for managing threads

    def setup_dependencies(self):
        """
        Initializes the dependency relations between tasks.
        This method populates the in_degree and dependents mappings based on task dependencies.
        """
        for task in self.tasks:
            print(task)
            self.in_degree[task.id] = len(task.dependencies)
            for dep in task.dependencies:
                self.dependents[dep].append(task.id)

    def execute_task(self, task: ResearchTask) -> TaskResult:
        """
        Executes a given ResearchTask.

        Args:
            task (ResearchTask): The task to be executed.

        Returns:
            TaskResult: The result of the task execution.
        """
        try:
            result: TaskResult = task.execute(self.db, self.state, self.tools)
            return result
        except Exception as e:
            return TaskResult(id=task.id, error=str(e))

    def execute(self):
        """
        Executes all tasks in the scheduler respecting the dependencies.
        This method manages the scheduling and execution of tasks, ensuring that all dependencies are resolved
        before task execution.
        """
        futures = {}  # Dictionary to keep track of futures for each task
        for task in [t for t in self.tasks if self.in_degree[t.id] == 0]:
            future = self.executor.submit(self.execute_task, task)
            futures[task.id] = future

        while futures:
            done, _ = concurrent.futures.wait(
                futures.values(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                result = future.result()
                self.state.update({result.id: result})
                task_id = result.id

                # Remove completed task from futures
                del futures[task_id]

                for dependent_id in self.dependents[task.id]:
                    self.in_degree[dependent_id] -= 1
                    if self.in_degree[dependent_id] == 0:
                        task = next(t for t in self.tasks if t.id == dependent_id)
                        future = self.executor.submit(self.execute_task, task)
                        futures[task.id] = future

        self.executor.shutdown()

    def get_results(self) -> List[TaskResult]:
        """
        Retrieves the final state containing the results of all executed tasks.

        Returns:
            List[TaskResult]: A list of TaskResult instances containing the results of all executed tasks.
        """
        return list(self.state.values())
