"""
🤖 LLM Client Abstraction
Unified interface for multiple LLM providers
"""

import os
from typing import Optional, Dict, Any
from langchain_openai import ChatOpenAI

class LLMClient:
    """Unified LLM client that works with any provider through environment variables."""
    
    def __init__(self):
        self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("No API key found. Set via --api-key flag.")
    
    def get_chat_client(self):
        """Get a LangChain chat client configured for the current provider."""
        # Use OpenAI-compatible interface (works for OpenRouter, OpenAI, etc.)
        return ChatOpenAI(
            model_name=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the configured LLM."""
        from langchain.schema import HumanMessage, SystemMessage
        
        client = self.get_chat_client()
        
        messages = []
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
        messages.append(HumanMessage(content=prompt))
        
        response = client.invoke(messages)
        return response.content
    
    def generate_score(self, prompt: str, system_prompt: Optional[str] = None, score_range: tuple = (0.0, 1.0)) -> float:
        """Generate a numerical score using the configured LLM."""
        scoring_prompt = f"""
        {system_prompt or "You are an expert evaluator."}
        
        {prompt}
        
        Respond with ONLY a numerical score between {score_range[0]} and {score_range[1]}.
        """
        
        try:
            response = self.generate_response(scoring_prompt)
            score = float(response.strip())
            return max(score_range[0], min(score_range[1], score))
        except (ValueError, TypeError):
            # Fallback to middle of range
            return (score_range[0] + score_range[1]) / 2 