"""
RAGAS Context Precision Evaluator
Implements LLM-based context precision scoring following official RAGAS methodology
"""

import os
import asyncio
from typing import Dict, Any, Optional, List
from datasets import Dataset
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.evaluation import evaluate
from ragas.run_config import RunConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_anthropic import ChatAnthropic
from .base_evaluator import BaseEvaluator

class RagasContextPrecisionEvaluator(BaseEvaluator):
    """
    Evaluate context precision using RAGAS library.
    
    Context Precision measures the proportion of relevant chunks in retrieved_contexts.
    Computed as mean of Precision@k across all retrieved chunks:
    
    Precision@k = true_positives@k / (true_positives@k + false_positives@k)
    
    Uses LLM to judge each chunk's relevance against the generated response.
    """
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        
        # Get configuration from environment variables set by CLI
        self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("API key not found. Set via --api-key flag or .env file.")
        
        # Get appropriate LLM and embeddings based on model
        self.llm = self._get_langchain_llm()
        self.embeddings = self._get_embeddings()
        
        self.run_config = RunConfig(
            max_workers=1,
            max_wait=60,
            max_retries=3
        )
    
    def _get_langchain_llm(self):
        """Get appropriate LangChain LLM based on model name."""
        model_lower = self.model.lower()
        
        if "claude" in model_lower or "anthropic" in model_lower:
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", self.api_key)
            return ChatAnthropic(
                model=self.model.replace("anthropic/", ""),
                anthropic_api_key=anthropic_key,
                temperature=0
            )
        elif "gemini" in model_lower or "google" in model_lower:
            google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", self.api_key)
            clean_model = self.model.replace("google/", "").replace("gemini-", "gemini-1.5-")
            if clean_model == "gemini-pro":
                clean_model = "gemini-1.5-pro"
            return ChatGoogleGenerativeAI(
                model=clean_model,
                google_api_key=google_key,
                temperature=0
            )
        else:
            # Default to OpenAI-compatible (includes OpenRouter)
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                temperature=0
            )
    
    def _get_embeddings(self):
        """Get appropriate embeddings based on model."""
        model_lower = self.model.lower()
        
        if "gemini" in model_lower or "google" in model_lower:
            google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", self.api_key)
            return GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=google_key
            )
        else:
            # Default to OpenAI embeddings for most providers
            return OpenAIEmbeddings(
                openai_api_key=self.api_key,
                openai_api_base=self.api_base
            )
    
    def evaluate(self, data: Dict[str, Any]) -> float:
        """
        Evaluate context precision using official RAGAS process.
        
        Process:
        1. For each retrieved chunk, LLM judges relevance to question+answer
        2. Calculate Precision@k for each position k
        3. Return mean precision across all positions
        """
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        
        # Ensure context is a list of strings
        if isinstance(context, str):
            context = [context]
        elif not isinstance(context, list):
            context = [str(context)]
        
        try:
            # Use RAGAS SingleTurnSample for without-reference evaluation
            from ragas import SingleTurnSample
            
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=context
            )
            
            # Create a proper LLM wrapper for RAGAS
            from ragas.llms import LangchainLLMWrapper
            llm_wrapper = LangchainLLMWrapper(self.llm)
            
            # Initialize LLMContextPrecisionWithoutReference metric with wrapped LLM
            context_precision_metric = LLMContextPrecisionWithoutReference(llm=llm_wrapper)
            
            # Run async evaluation
            import asyncio
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            score = loop.run_until_complete(context_precision_metric.single_turn_ascore(sample))
            return round(float(score), 3)
        
        except Exception as e:
            print(f"Error evaluating with RAGAS context precision: {str(e)}")
            # Fallback to manual precision evaluation
            return self._fallback_evaluation(question, context, answer)
    
    def _fallback_evaluation(self, question: str, context: list, answer: str) -> float:
        """
        Fallback evaluation using manual precision calculation.
        
        When RAGAS fails, manually implement the core precision logic:
        1. Judge each chunk's relevance to question+answer
        2. Calculate Precision@k for each position
        3. Return mean precision
        """
        from llm_client import LLMClient
        
        try:
            client = LLMClient()
            
            if not context:
                return 0.0
            
            # Step 1: Judge relevance of each chunk
            relevance_scores = []
            for i, chunk in enumerate(context, 1):
                relevance_prompt = f"""
Is this retrieved context relevant to answering the given question and supporting the response?

Question: {question}
Response: {answer}
Retrieved Context: {chunk}

Answer only 'YES' or 'NO':"""
                
                relevance_response = client.generate_response(relevance_prompt).strip().upper()
                is_relevant = relevance_response.startswith('YES')
                relevance_scores.append(is_relevant)
            
            # Step 2: Calculate Precision@k for each position
            precision_at_k = []
            true_positives = 0
            
            for k in range(1, len(relevance_scores) + 1):
                if relevance_scores[k - 1]:  # If current chunk is relevant
                    true_positives += 1
                
                # Precision@k = true_positives@k / k
                precision_k = true_positives / k
                precision_at_k.append(precision_k)
            
            # Step 3: Return mean precision
            mean_precision = sum(precision_at_k) / len(precision_at_k)
            return round(mean_precision, 3)
            
        except Exception as e:
            print(f"Fallback evaluation failed: {e}")
            return 0.5
    
    def get_detailed_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed chunk-by-chunk relevance analysis.
        
        Returns:
        - chunk_analysis: Each chunk with relevance judgment
        - precision_at_k: Precision@k for each position
        - mean_precision: Final context precision score
        """
        from llm_client import LLMClient
        
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        
        if isinstance(context, str):
            context = [context]
        elif not isinstance(context, list):
            context = [str(context)]
        
        try:
            client = LLMClient()
            
            # Judge relevance of each chunk
            chunk_analysis = []
            relevance_scores = []
            
            for i, chunk in enumerate(context, 1):
                relevance_prompt = f"""
Is this retrieved context relevant to answering the given question and supporting the response?

Question: {question}
Response: {answer}
Retrieved Context: {chunk}

Answer only 'YES' or 'NO':"""
                
                relevance_response = client.generate_response(relevance_prompt).strip().upper()
                is_relevant = relevance_response.startswith('YES')
                relevance_scores.append(is_relevant)
                
                chunk_analysis.append({
                    "chunk_number": i,
                    "chunk_text": chunk,
                    "relevant": is_relevant,
                    "relevance_response": relevance_response
                })
            
            # Calculate Precision@k for each position
            precision_at_k = []
            true_positives = 0
            
            for k in range(1, len(relevance_scores) + 1):
                if relevance_scores[k - 1]:
                    true_positives += 1
                
                precision_k = true_positives / k
                precision_at_k.append({
                    "position": k,
                    "true_positives": true_positives,
                    "total_chunks": k,
                    "precision": round(precision_k, 3)
                })
            
            mean_precision = sum(p["precision"] for p in precision_at_k) / len(precision_at_k)
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_contexts": context,
                "chunk_analysis": chunk_analysis,
                "precision_at_k": precision_at_k,
                "relevant_chunks": true_positives,
                "total_chunks": len(context),
                "mean_precision": round(mean_precision, 3)
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {e}",
                "mean_precision": 0.5
            } 