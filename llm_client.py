"""
🤖 LLM Client Abstraction
Unified interface for multiple LLM providers using LangChain universal support
"""

import os
from typing import Optional, Dict, Any
from langchain.schema import HumanMessage, SystemMessage

class LLMClient:
    """Unified LLM client that works with any provider through LangChain universal support."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        if model_config:
            self.model = model_config.get("llm_model", os.getenv("RAGCLI_LLM_MODEL", "gpt-4"))
            self.api_key = model_config.get("api_key", os.getenv("OPENAI_API_KEY"))
            self.api_base = model_config.get("api_base", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        else:
            self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("No API key found. Set via --api-key flag.")
        
        # Configure environment variables for LangChain universal support
        self._setup_environment()
    
    def _setup_environment(self):
        """Setup environment variables for LangChain universal support."""
        model_lower = self.model.lower()
        
        if "claude" in model_lower or "anthropic" in model_lower:
            # Anthropic
            os.environ["ANTHROPIC_API_KEY"] = self.api_key
        elif "gemini" in model_lower or "google" in model_lower:
            # Google Gemini
            os.environ["GOOGLE_API_KEY"] = self.api_key
        elif "openrouter.ai" in str(self.api_base):
            # OpenRouter
            os.environ["OPENAI_API_KEY"] = self.api_key
            os.environ["OPENAI_API_BASE"] = self.api_base
        else:
            # OpenAI or other OpenAI-compatible
            os.environ["OPENAI_API_KEY"] = self.api_key
            if self.api_base:
                os.environ["OPENAI_API_BASE"] = self.api_base
    
    def get_chat_client(self):
        """Get a LangChain chat client using universal support."""
        # Dynamically import and create the appropriate client based on model
        model_lower = self.model.lower()
        
        if "claude" in model_lower or "anthropic" in model_lower:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(
                model=self.model,
                anthropic_api_key=self.api_key,
                temperature=0
            )
        elif "gemini" in model_lower or "google" in model_lower:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0
            )
        else:
            # Default to OpenAI-compatible (OpenAI, OpenRouter, etc.)
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                temperature=0
            )
    
    def generate_response(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response using the configured LLM."""
        client = self.get_chat_client()
        
        # For universal LLM support, we need to handle different client types
        try:
            # Try chat interface first
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            messages.append(HumanMessage(content=prompt))
            
            response = client.invoke(messages)
            return response.content
        except AttributeError:
            # Fallback to direct text generation
            full_prompt = f"{system_prompt + '\n\n' if system_prompt else ''}{prompt}"
            return client.invoke(full_prompt)
    
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