"""
RAGChecker Hallucination Evaluator
Pure wrapper around official RAGChecker library with multi-provider support
"""

import os
from typing import Dict, Any, Optional
from ragchecker import RAGChecker, RAGResults
from ragchecker.metrics import hallucination
from .base_evaluator import BaseEvaluator

class RAGCheckerHallucinationEvaluator(BaseEvaluator):
    """Evaluate hallucination using RAGChecker library with unified LLM configuration."""
    
    def __init__(self, model_config: Optional[Dict[str, Any]] = None):
        super().__init__(model_config)
        
        # Get configuration from environment variables set by CLI
        self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        
        if not self.api_key:
            raise ValueError("API key not found. Set via --api-key flag or .env file.")
        
        # Configure RAGChecker with unified model
        # For OpenRouter/custom bases, we need to handle the model format
        extractor_model = self._format_model_for_ragchecker(self.model)
        checker_model = self._format_model_for_ragchecker(self.model)
        
        # Initialize RAGChecker
        ragchecker_config = {
            "extractor_name": extractor_model,
            "checker_name": checker_model
        }
        
        # Set environment variables for RAGChecker to pick up
        if self.api_base and self.api_base != "https://api.openai.com/v1":
            # For custom API bases (OpenRouter, etc.)
            os.environ["OPENAI_BASE_URL"] = self.api_base
        
        self.checker = RAGChecker(**ragchecker_config)
    
    def _format_model_for_ragchecker(self, model_name: str) -> str:
        """Format model name for RAGChecker/LiteLLM compatibility."""
        # Check if we're using OpenRouter
        if self.api_base and "openrouter.ai" in self.api_base:
            # For OpenRouter, use the model name directly or with openrouter/ prefix
            if not model_name.startswith('openrouter/'):
                return f'openrouter/{model_name}'
            return model_name
        
        # For Gemini direct API
        if model_name.startswith('gemini'):
            gemini_model_map = {
                'gemini-pro': 'gemini/gemini-1.5-pro',
                'gemini-flash': 'gemini/gemini-1.5-flash',
                'gemini-1.5-pro': 'gemini/gemini-1.5-pro',
                'gemini-1.5-flash': 'gemini/gemini-1.5-flash'
            }
            return gemini_model_map.get(model_name, f'gemini/{model_name}')
        
        # For other providers, return as-is
        return model_name
    
    def evaluate(self, data: Dict[str, Any]) -> float:
        """Evaluate hallucination using RAGChecker library."""
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        
        # Ensure context is a list
        if isinstance(context, str):
            context = [context]
        
        # Build RAGChecker input format
        rag_data = {
            "results": [
                {
                    "query_id": "1",
                    "query": question,
                    "gt_answer": "",
                    "response": answer,
                    "retrieved_context": [
                        {"doc_id": f"doc_{i}", "text": ctx} 
                        for i, ctx in enumerate(context)
                    ]
                }
            ]
        }
        
        try:
            # Use RAGChecker pure wrapper
            rag_results = RAGResults.from_dict(rag_data)
            self.checker.evaluate(rag_results, [hallucination])
            
            # Extract score from raw output
            raw_output = rag_results.to_dict()
            print("✓ Using official RAGChecker hallucination implementation")
            return self._extract_hallucination_score(raw_output)
        
        except Exception as e:
            print(f"Error evaluating with RAGChecker hallucination: {str(e)}")
            print("→ Using fallback (returning 0.0)")
            return 0.0
    
    def _extract_hallucination_score(self, result: Dict[str, Any]) -> float:
        """Extract hallucination score from RAGChecker output."""
        try:
            # Navigate to the hallucination score in RAGChecker output
            score = result["metrics"]["generator_metrics"]["hallucination"]
            return float(score)
        except (KeyError, IndexError, TypeError) as e:
            print(f"Score extraction failed: {e}")
            return 0.0
    
    def get_raw_output(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Get raw RAGChecker output for analysis."""
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        
        if isinstance(context, str):
            context = [context]
        
        rag_data = {
            "results": [
                {
                    "query_id": "1",
                    "query": question,
                    "gt_answer": "",
                    "response": answer,
                    "retrieved_context": [
                        {"doc_id": f"doc_{i}", "text": ctx} 
                        for i, ctx in enumerate(context)
                    ]
                }
            ]
        }
        
        rag_results = RAGResults.from_dict(rag_data)
        self.checker.evaluate(rag_results, [hallucination])
        return rag_results.to_dict()
    
    def get_detailed_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed hallucination analysis showing specific issues found.
        
        Returns:
        - hallucination_score: Overall hallucination score (0-100)
        - analysis_breakdown: Detailed breakdown of what was detected
        - response_segments: Analysis of different parts of the response
        """
        try:
            raw_output = self.get_raw_output(data)
            
            question = data.get('question', '')
            context = data.get('context', [])
            answer = data.get('answer', '')
            
            if isinstance(context, str):
                context = [context]
            
            # Extract the detailed analysis from RAGChecker output
            hallucination_score = self._extract_hallucination_score(raw_output)
            
            # Get detailed breakdown from RAGChecker results
            detailed_breakdown = self._extract_detailed_breakdown(raw_output)
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_contexts": context,
                "hallucination_score": hallucination_score,
                "hallucination_percentage": hallucination_score,
                "analysis_breakdown": detailed_breakdown,
                "interpretation": self._interpret_score(hallucination_score),
                "raw_ragchecker_output": raw_output
            }
            
        except Exception as e:
            return {
                "error": f"Detailed analysis failed: {e}",
                "hallucination_score": 0.0
            }
    
    def _extract_detailed_breakdown(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed breakdown from RAGChecker output."""
        try:
            # Navigate to detailed metrics in RAGChecker output
            results = raw_output.get("results", [])
            if not results:
                return {"error": "No results found in RAGChecker output"}
            
            result = results[0]  # First (and typically only) result
            
            # Extract various components analyzed by RAGChecker
            breakdown = {
                "query_id": result.get("query_id", "1"),
                "response_length": len(result.get("response", "")),
                "context_count": len(result.get("retrieved_context", [])),
                "metrics_analyzed": []
            }
            
            # Add metrics that were computed
            if "metrics" in raw_output:
                metrics = raw_output["metrics"]
                if "generator_metrics" in metrics:
                    gen_metrics = metrics["generator_metrics"]
                    breakdown["generator_metrics"] = {
                        "hallucination": gen_metrics.get("hallucination", 0.0)
                    }
                    breakdown["metrics_analyzed"].append("hallucination_detection")
            
            return breakdown
            
        except Exception as e:
            return {"error": f"Could not extract breakdown: {e}"}
    
    def _interpret_score(self, score: float) -> str:
        """Provide human-readable interpretation of hallucination score."""
        if score == 0.0:
            return "No hallucinations detected - response appears fully grounded in context"
        elif score <= 25.0:
            return "Low hallucination risk - response is mostly grounded with minor unsupported claims"
        elif score <= 50.0:
            return "Moderate hallucination risk - response contains some unsupported information"
        elif score <= 75.0:
            return "High hallucination risk - response contains significant unsupported claims"
        else:
            return "Very high hallucination risk - response is largely unsupported by context"
