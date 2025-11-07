"""Graph builder module."""

from langgraph.graph import StateGraph, START, END
from src.state.rag_state import RAGState
from src.nodes.nodes import RAGNodes


class GraphBuilder:
    """Build and manges the langgraph workflow."""

    def __init__(self, retriever, llm):
        """Initialize the graph builder with retriever and LLM.
        Args: 
            retriever: The document retriever object.
            llm: language model object."""
        self.nodes= RAGNodes()
        self.graph = None
    

    def build(self):
        """Build the state graph for RAG workflow.
        Returns: compiled graph object."""

        builder = StateGraph(RAGState)

        builder.add_node("retriever", self.nodes.retriever_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        builder.set_entry_point("retriever")

        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        self.graph = builder.compile()

        return self.graph
    
    def run(self, question: str)->dict:
        """Run the graph with the given question.
        Args:
            question: The input question string.
        
            Returns: The final answer dictionary."""
        
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)