from tools.research.common.model_schemas import ContentItem
from typing import Optional

import threading
import logging
import sqlite3
import os


class ContentDB:
    def __init__(self, db_path: str = ":memory:"):
        """
        Initializes the ContentDB instance, setting up an SQLite database.

        Args:
            db_path (str): The file path to the SQLite database. Defaults to an in-memory database.
                           This allows for persistent data storage when a file path is provided.

        This constructor also ensures the database contains a 'content' table, which is created if it doesn't exist.
        """
        self.lock = threading.Lock()  # Ensures that database operations are thread-safe

        if db_path != ":memory:":
            # Ensures the directory for the database file exists
            db_dir = os.path.dirname(db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)

        # Allow multi-threaded access to the database by setting check_same_thread to False
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        with self.lock:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS content (
                    id TEXT PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    snippet TEXT,
                    content TEXT,
                    source TEXT
                )
                """
            )
            self.conn.commit()

    def get_doc_by_id(self, id: str) -> Optional[ContentItem]:
        """
        Retrieves a document by its unique ID.

        Args:
            id (str): The unique identifier for the document.

        Returns:
            Optional[ContentItem]: A ContentItem instance if found, else None.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, url, title, snippet, content, source FROM content WHERE id = ?",
                (id,),
            )
            row = cursor.fetchone()
            return (
                ContentItem(
                    **dict(
                        zip(["id", "url", "title", "snippet", "content", "source"], row)
                    )
                )
                if row
                else None
            )

    def get_doc_by_url(self, url: str) -> Optional[ContentItem]:
        """
        Retrieves a document by its URL.

        Args:
            url (str): The URL associated with the document.

        Returns:
            Optional[ContentItem]: A ContentItem instance if found, else None.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT id, url, title, snippet, content, source FROM content WHERE url = ?",
                (url,),
            )
            row = cursor.fetchone()
            return (
                ContentItem(
                    **dict(
                        zip(["id", "url", "title", "snippet", "content", "source"], row)
                    )
                )
                if row
                else None
            )

    def upsert_doc(self, doc: ContentItem):
        """
        Inserts a new document or updates an existing one based on the URL conflict.

        Args:
            doc (ContentItem): A ContentItem instance containing the document data.
        """
        with self.lock:
            cursor = self.conn.cursor()
            try:
                cursor.execute(
                    """
                    INSERT INTO content (id, url, title, snippet, content, source)
                    VALUES (:id, :url, :title, :snippet, :content, :source)
                    ON CONFLICT(url) DO UPDATE SET
                    id=excluded.id,
                    title=excluded.title,
                    snippet=excluded.snippet,
                    content=excluded.content,
                    source=excluded.source
                    """,
                    doc.to_dict(),
                )
                self.conn.commit()
                logging.info(f"Document inserted/updated successfully: {doc.id}")
            except sqlite3.IntegrityError as e:
                logging.error(f"Error inserting/updating document: {e}")
                raise

    def delete_doc(self, id: str):
        """
        Deletes a document by its ID.

        Args:
            id (str): The unique identifier for the document to be deleted.
        """
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM content WHERE id = ?", (id,))
            self.conn.commit()

    def generate_snippet(self, text: str) -> str:
        """
        Generates a text snippet from the provided text.

        Args:
            text (str): The text from which to generate the snippet.

        Returns:
            str: A string representing the snippet, truncated to 150 characters plus an ellipsis.
        """
        return text[:150] + "..."  # Simplistic snippet generation for demo purposes
