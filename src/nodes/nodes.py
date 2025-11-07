"""Langgraph nodes for RAG"""

from src.state.rag_state import RAGState


class RAGNodes:
    """Contain node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        """Initialize RAG nodes with retriever and LLM.
        Args: 
            retriever: The document retriever object.
            llm: language model object."""
        self.retriever = retriever
        self.llm = llm  

    
    def retriever_docs(self, state: RAGState) -> RAGState:
        """Node to retrieve documents based on query in state.
        Args:
            state: The current RAGState object.
        
            Returns: Updated RAGState with retrieved documents."""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question, 
            retrieved_docs=docs
        )

    def generate_answer(self, state: RAGState) -> RAGState:
        """Node to generate answer based on retrieved documents in state.
        Args:
            state: The current RAGState object.
        
            Returns: Updated RAGState with generated answer."""
        context = "\n".join([doc.page_content for doc in state.retrieved_docs])
        prompt = f"""Use the following context to answer the question:
        Context: {context}
        Question: {state.question}
        """
        response=self.llm.invoke(prompt)
        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=response.content
        )







