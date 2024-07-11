from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from .common.model_schemas import ResearchToolOutput


class ResearchTool(BaseTool):
    """
    Base class for all tools, providing a common structure and functionality.
    Inherits from langchain.tools.BaseTool.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]
    include_summary: bool = Field(default=False)

    def _run(self, **kwargs):
        """
        The main logic of the tool. Must be implemented by subclasses.

        Args:
            kwargs: Arbitrary keyword arguments specific to the tool's functionality.

        Returns:
            The result of the tool's execution.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def invoke(self, **kwargs) -> ResearchToolOutput:
        # Extend the description with the description of ResearchToolOutput
        """
        Invokes the tool

        Args:
            kwargs: Arbitrary keyword arguments specific to the tool's functionality.

        Returns:
            ResearchToolOutput: The result of the tool's execution.
        """
        if input in kwargs:
            kwargs = kwargs["input"]
        return super().invoke(input=kwargs)
