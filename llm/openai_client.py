"""OpenAI client wrapper for RAGTESTERCLI."""

import json
from typing import Dict, Any, Optional, List
from openai import OpenAI
from config import config


class OpenAIClient:
    """Wrapper for OpenAI API interactions."""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.llm_config = config.get_llm_config()
        self.api_key = api_key or self.llm_config['api_key']
        self.model = model or self.llm_config['model']
        
        # Initialize client as None, create it lazily when needed
        self.client = None
    
    def _ensure_client(self):
        """Ensure the OpenAI client is initialized."""
        if self.client is None:
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable or configure in YAML file.")
            self.client = OpenAI(api_key=self.api_key)
    
    def generate_completion(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate completion using OpenAI API."""
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            self._ensure_client()
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.llm_config.get('temperature', 0.0),
                max_tokens=max_tokens or self.llm_config.get('max_tokens', 1000)
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {str(e)}")
    
    def generate_score(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        score_range: tuple = (0, 1)
    ) -> float:
        """Generate a numerical score using OpenAI API."""
        
        score_instruction = f"""
Please provide your response as a JSON object with the following format:
{{
    "score": <float between {score_range[0]} and {score_range[1]}>,
    "reasoning": "<brief explanation>"
}}
"""
        
        full_prompt = prompt + "\n\n" + score_instruction
        response = self.generate_completion(full_prompt, system_prompt)
        
        try:
            result = json.loads(response)
            score = float(result.get('score', 0))
            
            # Ensure score is within range
            score = max(score_range[0], min(score_range[1], score))
            return score
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            # Fallback: try to extract number from response
            import re
            numbers = re.findall(r'\d+\.?\d*', response)
            if numbers:
                score = float(numbers[0])
                return max(score_range[0], min(score_range[1], score))
            
            raise RuntimeError(f"Could not parse score from LLM response: {str(e)}")


# Global client instance (lazily initialized)
llm_client = OpenAIClient() 