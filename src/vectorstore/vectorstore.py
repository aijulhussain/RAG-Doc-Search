"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class VectorStore:
    """Manages vector store for application"""

    def __init__(self):
        self.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = None
        self.retriever =  None
    

    def create_retriever(self, documents: List[Document]):
        """ Create a vector store from documents.
        Args:
            documents: List of documents to be embedded and stored.
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        self.retriever = self.vectorstore.as_retriever()
    
    def get_retriver(self):
        """ Get the retriever for querying the vector store.
        
        Returns:
            The retriever object.
        """
        if self.retriever is None:
            raise ValueError("Retriever has not been created. Call create_retriever() first.")
        return self.retriever