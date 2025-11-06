"""RAG state defination for langgraph"""

from typing import List
from pydantic import BaseModel
from langchain_core.documents import Document

class RAGState(BaseModel):

    """RAG State Model"""
    question: str = ""
    documents: List[Document] = []
    answer: str = ""