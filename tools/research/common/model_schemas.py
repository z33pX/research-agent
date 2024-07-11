from pydantic import BaseModel
from typing import List, Optional


class ContentItem(BaseModel):
    """
    Represents a single content item.

    Attributes:
        url (str): The URL of the content item.
        title (str): The title of the content item.
        snippet (str): A short snippet or description of the content item.
        content (str): The full content of the item.
        source (str): The source of the content item.
    """

    url: str
    title: str
    snippet: str
    content: str
    source: Optional[str] = ""
    id: Optional[str] = ""

    def __str__(self):
        return f"{self.title}\n{self.url}\n{self.snippet}"

    def to_dict(self):
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "content": self.content,
            "source": self.source,
            "id": self.id,
        }


class ResearchToolOutput(BaseModel):
    """
    Represents the output of a research tool.

    Attributes:
        content (List[ContentItem]): A list of content items generated or processed by the tool.
        summary (str): A summary of the content items.
    """

    content: List[ContentItem]
    summary: str
