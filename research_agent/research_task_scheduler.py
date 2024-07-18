# Import necessary modules and classes
from .research_task import ResearchTask, TaskResult
from .db import ContentDB

from langchain.tools import BaseTool
from collections import defaultdict
from typing import List, Dict
import concurrent.futures
import threading
import logging
import os


class TaskScheduler:
    """
    Schedules and manages the execution of a set of research tasks, taking into account their dependencies.

    Attributes:
        tasks (List[ResearchTask]): List of research tasks to be scheduled.
        tools (List[BaseTool]): List of tools to be used in tasks.
        state (Dict[str, TaskResult]): Stores the state/results of tasks.
        db (ContentDB): Database for storing task content.
        dependents (defaultdict): Tracks task dependents.
        in_degree (defaultdict): Tracks task dependencies count.
        task_map (Dict): Maps task IDs to task objects for fast lookup.
        executor (ThreadPoolExecutor): Thread pool executor for concurrent task execution.
        lock (threading.Lock): Lock for thread-safe operations.
    """

    def __init__(
        self,
        tasks: List[ResearchTask],  # List of research tasks to be scheduled
        tools: List[BaseTool],  # List of tools to be used in tasks
    ):
        """
        Initializes the TaskScheduler with a list of tasks and tools.

        Args:
            tasks (List[ResearchTask]): The tasks to be executed.
            tools (List[BaseTool]): The tools available for task execution.
        """
        self.tasks: List[ResearchTask] = tasks
        self.state: Dict[str, TaskResult] = {}
        current_folder = os.path.dirname(os.path.abspath(__file__))
        self.db: ContentDB = ContentDB(current_folder + "/db/content.db")
        self.tools: List[BaseTool] = tools
        self.dependents = defaultdict(list)
        self.in_degree = defaultdict(int)
        self.task_map = {task.id: task for task in tasks}
        self.setup_dependencies()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.lock = threading.Lock()

    def setup_dependencies(self) -> None:
        """
        Sets up the task dependencies by populating the in_degree and dependents attributes.
        """
        for task in self.tasks:
            self.in_degree[task.id] = len(task.dependencies)
            for dep in task.dependencies:
                self.dependents[dep].append(task.id)

    def execute_task(self, task: ResearchTask) -> TaskResult:
        """
        Executes a single task and returns the result.

        Args:
            task (ResearchTask): The task to be executed.

        Returns:
            TaskResult: The result of the executed task.
        """
        try:
            result: TaskResult = task.execute(self.db, self.state, self.tools)
            return result
        except Exception as e:
            logging.error(f"Error executing task {task.id}: {str(e)}")
            return TaskResult(id=task.id, error=str(e))

    def execute(self) -> None:
        """
        Executes all tasks in the scheduler, respecting their dependencies.
        """
        futures = {}
        # Find tasks with no dependencies and submit them for execution
        for task in [t for t in self.tasks if self.in_degree[t.id] == 0]:
            logging.info(f"Executing task {task.id}")
            future = self.executor.submit(self.execute_task, task)
            futures[task.id] = future

        while futures:
            # Wait for the first task to complete
            done, _ = concurrent.futures.wait(
                futures.values(), return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in done:
                result = future.result()
                task_id = result.id

                with self.lock:
                    self.state[task_id] = result

                    # Check dependent tasks and update their in-degree
                    for dependent_id in self.dependents[task_id]:
                        self.in_degree[dependent_id] -= 1
                        if self.in_degree[dependent_id] == 0:
                            dependent_task = self.task_map[dependent_id]
                            logging.info(
                                f"Executing dependent task {dependent_task.id}"
                            )
                            future = self.executor.submit(
                                self.execute_task, dependent_task
                            )
                            futures[dependent_task.id] = future

                del futures[task_id]

        logging.info("All tasks executed.")
        self.executor.shutdown()

    def get_results(self) -> List[TaskResult]:
        """
        Retrieves the results of all executed tasks.

        Returns:
            List[TaskResult]: A list of task results.
        """
        return list(self.state.values())
