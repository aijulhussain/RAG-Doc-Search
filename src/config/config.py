"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

class Config:
    """Configuration class for Agentic RAG system"""

    # === API Keys ===
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # === Model Configuration ===
    MODEL_NAME = "llama-3.3-70b-versatile"

    # === Document Processing ===
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # === Default URLs ===
    DEFAULT_URLS = [
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
        "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
    ]

    @classmethod
    def get_llm(cls):
        """Initialize and return the Groq chat model"""
        if not cls.GROQ_API_KEY:
            raise ValueError("❌ GROQ_API_KEY not found in environment. Please set it in your .env file.")
        
        return ChatGroq(
            api_key=cls.GROQ_API_KEY,
            model=cls.MODEL_NAME,
            temperature=0.7,
            max_tokens=None
        )
