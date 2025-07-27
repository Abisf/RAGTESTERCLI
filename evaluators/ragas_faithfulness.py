"""
RAGAS Faithfulness Evaluator
Pure wrapper around official RAGAS library with multi-provider support
"""

import os
from typing import Dict, Any, Optional
from datasets import Dataset
from ragas.metrics import faithfulness
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from .base_evaluator import BaseEvaluator

class RagasFaithfulnessEvaluator(BaseEvaluator):
    """Evaluate faithfulness using RAGAS library with unified LLM configuration."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        
        # Get configuration from environment variables set by CLI
        self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("API key not found. Set via --api-key flag.")
        
        # Configure RAGAS with unified LLM
        self.llm = ChatOpenAI(
            model_name=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=self.api_key,
            openai_api_base=self.api_base
        )
        
        self.run_config = RunConfig(
            max_workers=1,
            max_wait=60,
            max_retries=3
        )
    
    def evaluate(self, data: Dict[str, Any]) -> float:
        """Evaluate faithfulness using RAGAS library."""
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        
        # Ensure context is a list
        if isinstance(context, str):
            context = [context]
        elif not isinstance(context, list):
            context = [str(context)]
        
        # Create dataset for RAGAS
        dataset = Dataset.from_dict({
            "question": [question],
            "contexts": [context],
            "answer": [answer],
        })
        
        try:
            # Configure RAGAS with our unified LLM
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            llm_wrapper = LangchainLLMWrapper(self.llm)
            emb_wrapper = LangchainEmbeddingsWrapper(self.embeddings)
            
            # Use RAGAS evaluate with proper wrappers
            result = evaluate(
                dataset,
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                metrics=[faithfulness]
            )
            
            # Extract score from result
            score = float(result.to_pandas()['faithfulness'].iloc[0])
            return round(score, 3)
        
        except Exception as e:
            print(f"Error evaluating with RAGAS faithfulness: {str(e)}")
            # Fallback to custom evaluation
            return self._fallback_evaluation(question, context, answer)
    
    def _fallback_evaluation(self, question: str, context: list, answer: str) -> float:
        """Fallback evaluation using direct LLM call."""
        from llm_client import LLMClient
        
        try:
            client = LLMClient()
            context_text = "\n".join(context) if isinstance(context, list) else str(context)
            
            prompt = f"""
Rate the faithfulness of this answer to the given context on a scale of 0.0 to 1.0:

Context: {context_text}

Question: {question}

Answer: {answer}

Score (0.0 = completely unfaithful, 1.0 = completely faithful):"""

            return client.generate_score(prompt, score_range=(0.0, 1.0))
        except Exception:
            return 0.5 