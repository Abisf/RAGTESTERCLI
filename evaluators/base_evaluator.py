"""
Base evaluator class for RAGTESTERCLI

Defines the interface that all evaluators must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class BaseEvaluator(ABC):
    """Abstract base class for all RAG evaluators."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        """
        Initialize evaluator with optional model configuration.
        
        Args:
            model_config: Optional configuration for LLM models and API keys
        """
        self.model_config = model_config or {}
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any]) -> float:
        """
        Evaluate a single RAG instance.
        
        Args:
            data: Dictionary containing 'question', 'context', and 'answer'
        
        Returns:
            float: Evaluation score (typically 0.0 to 1.0)
        """
        pass
    
    def batch_evaluate(self, data_list: List[Dict[str, Any]]) -> List[float]:
        """
        Evaluate multiple RAG instances.
        
        Default implementation calls evaluate() for each item.
        Subclasses can override for more efficient batch processing.
        
        Args:
            data_list: List of RAG instance dictionaries
        
        Returns:
            List[float]: List of evaluation scores
        """
        return [self.evaluate(item) for item in data_list]
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for the specified provider from model_config or environment."""
        import os
        
        # First check if API key is provided in model_config
        if self.model_config and 'api_key' in self.model_config:
            return self.model_config['api_key']
        
        # Environment variable mapping for different providers
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'google': 'GOOGLE_API_KEY',
            'gemini': 'GEMINI_API_KEY',  # LiteLLM uses GEMINI_API_KEY
            'mistral': 'MISTRAL_API_KEY',
            'together': 'TOGETHER_API_KEY',
            'huggingface': 'HUGGINGFACE_API_KEY'
        }
        
        env_key = env_key_map.get(provider.lower())
        if env_key:
            return os.getenv(env_key)
        return None
    
    def get_model_name(self, model_type: str, default: str = "gpt-4") -> str:
        """
        Get model name from configuration with fallback to default.
        
        Args:
            model_type: Type of model (e.g., 'extractor_name', 'checker_name', 'llm_model')
            default: Default model name
        
        Returns:
            str: Model name to use
        """
        if self.model_config and model_type in self.model_config:
            return self.model_config[model_type]
        return default
    
    def get_provider_from_model(self, model_name: str) -> str:
        """Detect LLM provider from model name."""
        model_lower = model_name.lower()
        
        if model_lower.startswith(('gpt-', 'text-', 'davinci', 'curie', 'babbage', 'ada')):
            return 'openai'
        elif model_lower.startswith(('claude-', 'claude')):
            return 'anthropic'
        elif model_lower.startswith(('gemini-', 'palm-', 'bison')):
            return 'gemini'  # Use 'gemini' for LiteLLM compatibility
        elif model_lower.startswith(('mistral-', 'mixtral-')):
            return 'mistral'
        elif model_lower.startswith(('llama-', 'meta-llama')):
            return 'together'
        else:
            return 'openai'  # Default fallback 