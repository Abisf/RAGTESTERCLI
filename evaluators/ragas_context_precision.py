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
        
        # Get configuration from model_config (passed from CLI) or fallback to environment
        if model_config:
            self.model = model_config.get("llm_model", os.getenv("RAGCLI_LLM_MODEL", "gpt-4"))
            self.api_key = model_config.get("api_key", os.getenv("OPENAI_API_KEY"))
            self.api_base = model_config.get("api_base", os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"))
        else:
            # Fallback to environment variables if no model_config provided
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
        
        # Check if we're using OpenRouter (which provides OpenAI-compatible API for all models)
        if self.api_base and "openrouter.ai" in self.api_base:
            # For OpenRouter, always use ChatOpenAI regardless of model type
            return ChatOpenAI(
                model_name=self.model,
                openai_api_key=self.api_key,
                openai_api_base=self.api_base,
                temperature=0
            )
        elif "claude" in model_lower or "anthropic" in model_lower:
            # Direct Anthropic API (not via OpenRouter)
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", self.api_key)
            return ChatAnthropic(
                model=self.model.replace("anthropic/", ""),
                anthropic_api_key=anthropic_key,
                temperature=0
            )
        elif "gemini" in model_lower or "google" in model_lower:
            # Direct Google API (not via OpenRouter)
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
            # Default to OpenAI-compatible (includes direct OpenAI)
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
            print("✓ Using official RAGAS context precision implementation")
            return round(float(score), 3)
        
        except Exception as e:
            print(f"Error evaluating with RAGAS context precision: {str(e)}")
            print("→ Using fallback manual implementation")
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
            # Pass model_config to LLMClient so it uses the correct API key and base
            client = LLMClient(self.model_config)
            
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
            
            # Step 3: Calculate weighted context precision (RAGAS formula)
            # Context Precision = Σ(Precision@k × v_k) / Σ(v_k)
            # where v_k = 1 if item at rank k is relevant, 0 otherwise
            weighted_sum = 0.0
            total_relevant = 0
            
            for k in range(len(relevance_scores)):
                if relevance_scores[k]:  # v_k = 1 if relevant
                    weighted_sum += precision_at_k[k]  # Precision@(k+1) × 1
                    total_relevant += 1
            
            # If no relevant chunks, return 0
            if total_relevant == 0:
                context_precision = 0.0
            else:
                context_precision = weighted_sum / total_relevant
            
            # For compatibility, return the RAGAS context precision (but could return both)
            return round(context_precision, 3)
            
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
            # Pass model_config to LLMClient so it uses the correct API key and base
            client = LLMClient(self.model_config)
            
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
            
            # Calculate weighted context precision (RAGAS formula)
            # Context Precision = Σ(Precision@k × v_k) / Σ(v_k)
            weighted_sum = 0.0
            total_relevant = 0
            
            for i, chunk_data in enumerate(chunk_analysis):
                if chunk_data["relevant"]:  # v_k = 1 if relevant
                    weighted_sum += precision_at_k[i]["precision"]  # Precision@(k+1) × 1
                    total_relevant += 1
            
            # If no relevant chunks, return 0
            if total_relevant == 0:
                context_precision = 0.0
            else:
                context_precision = weighted_sum / total_relevant
            
            # Generate enhanced diagnostic insights
            diagnostic_insights = self._generate_precision_insights(chunk_analysis, context_precision)
            
            # Generate retrieval quality analysis
            retrieval_analysis = self._analyze_retrieval_quality(chunk_analysis, precision_at_k)
            
            # Calculate simple (unranked) context precision for noise analysis
            simple_context_precision = true_positives / len(context) if context else 0.0
            
            return {
                "question": question,
                "answer": answer,
                "retrieved_contexts": context,
                "chunk_analysis": chunk_analysis,
                "precision_at_k": precision_at_k,
                "relevant_chunks": true_positives,
                "total_chunks": len(context),
                
                # RAGAS Context Precision (Average Precision - ranking quality)
                "ragas_context_precision": round(context_precision, 3),
                
                # Simple Context Precision (relevant/total - noise analysis)  
                "simple_context_precision": round(simple_context_precision, 3),
                
                # Enhanced diagnostic information
                "retrieval_quality_analysis": retrieval_analysis,
                "diagnostic_insights": diagnostic_insights,
                "summary_statistics": {
                    "ragas_context_precision": context_precision,
                    "simple_context_precision": simple_context_precision,
                    "irrelevant_chunks": len(context) - true_positives,
                    "noise_rate": (len(context) - true_positives) / len(context) if context else 0.0,
                    "early_precision": precision_at_k[0]["precision"] if precision_at_k else 0.0,
                    "late_precision": precision_at_k[-1]["precision"] if precision_at_k else 0.0,
                    "calculation_methods": {
                        "ragas": "average_precision_weighted_by_relevance",
                        "simple": "relevant_chunks_divided_by_total"
                    }
                },
                
                # Summary verdict for casual users
                "summary_verdict": self._generate_summary_verdict(chunk_analysis, context_precision, simple_context_precision),
                
                # Structured action recommendations
                "action_recommendations": self._generate_action_recommendations(chunk_analysis, context_precision, simple_context_precision)
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {e}",
                "mean_precision": 0.5
            }
    
    def _generate_precision_insights(self, chunk_analysis: list, mean_precision: float) -> Dict[str, Any]:
        """Generate actionable diagnostic insights for context precision."""
        insights = {
            "primary_issues": [],
            "recommendations": [],
            "severity": "low",
            "specific_problems": []
        }
        
        if not chunk_analysis:
            return insights
        
        irrelevant_chunks = [c for c in chunk_analysis if not c["relevant"]]
        
        # Analyze irrelevant chunks for patterns
        if len(irrelevant_chunks) > 0:
            insights["specific_problems"] = [
                {
                    "chunk_number": chunk["chunk_number"],
                    "chunk_text": chunk["chunk_text"][:100] + "..." if len(chunk["chunk_text"]) > 100 else chunk["chunk_text"],
                    "issue": "Irrelevant to question/answer"
                }
                for chunk in irrelevant_chunks
            ]
        
        # Generate recommendations based on precision score
        if mean_precision < 0.3:
            insights["primary_issues"].append("Critical precision failure - majority of chunks irrelevant")
            insights["recommendations"].extend([
                "Review retrieval similarity thresholds - too permissive",
                "Improve query formulation for more targeted retrieval",
                "Consider semantic reranking to filter irrelevant chunks",
                "Evaluate embedding model quality for domain"
            ])
            insights["severity"] = "critical"
            
        elif mean_precision < 0.5:
            insights["primary_issues"].append("Low precision - significant noise in retrieval")
            insights["recommendations"].extend([
                "Tighten similarity thresholds for retrieval",
                "Add reranking stage to filter irrelevant content",
                "Review query expansion strategies"
            ])
            insights["severity"] = "high"
            
        elif mean_precision < 0.7:
            insights["primary_issues"].append("Moderate precision issues - some irrelevant chunks")
            insights["recommendations"].extend([
                "Fine-tune retrieval parameters",
                "Consider hybrid retrieval strategies (keyword + semantic)"
            ])
            insights["severity"] = "medium"
        
        # Analyze position-based patterns (early precision analysis)
        if len(chunk_analysis) > 2:
            # Check early precision (top-3 chunks)
            early_relevant = sum(1 for c in chunk_analysis[:3] if c["relevant"])
            early_irrelevant = sum(1 for c in chunk_analysis[:3] if not c["relevant"])
            
            # Good early precision means relevant chunks appear in top positions
            if chunk_analysis[0]["relevant"]:  # First chunk is relevant
                if early_irrelevant >= 2:
                    insights["primary_issues"].append("Good early precision but noisy tail")
                    insights["recommendations"].append("Keep top results - prune or rerank the rest to suppress noise")
            else:  # First chunk is not relevant
                insights["primary_issues"].append("Poor early precision - top chunks are irrelevant") 
                insights["recommendations"].append("Prioritize fixing ranking algorithm - top results should be most relevant")
        
        return insights
    
    def _analyze_retrieval_quality(self, chunk_analysis: list, precision_at_k: list) -> Dict[str, Any]:
        """Analyze retrieval quality patterns and trends."""
        if not chunk_analysis or not precision_at_k:
            return {"error": "Insufficient data for retrieval quality analysis"}
        
        try:
            quality_analysis = {
                "precision_trend": "unknown",
                "ranking_quality": "unknown",
                "position_analysis": [],
                "noise_pattern": "unknown"
            }
            
            # Analyze precision trend
            if len(precision_at_k) >= 3:
                early_precision = precision_at_k[0]["precision"]
                mid_precision = precision_at_k[len(precision_at_k)//2]["precision"]
                late_precision = precision_at_k[-1]["precision"]
                
                if late_precision > mid_precision > early_precision:
                    quality_analysis["precision_trend"] = "improving"
                elif early_precision > mid_precision > late_precision:
                    quality_analysis["precision_trend"] = "degrading"
                else:
                    quality_analysis["precision_trend"] = "stable"
            
            # Analyze ranking quality
            relevant_positions = [i for i, chunk in enumerate(chunk_analysis) if chunk["relevant"]]
            if relevant_positions:
                avg_relevant_position = sum(relevant_positions) / len(relevant_positions)
                total_positions = len(chunk_analysis)
                
                if avg_relevant_position < total_positions * 0.3:
                    quality_analysis["ranking_quality"] = "good_early_ranking"
                elif avg_relevant_position > total_positions * 0.7:
                    quality_analysis["ranking_quality"] = "poor_early_ranking"
                else:
                    quality_analysis["ranking_quality"] = "moderate_ranking"
            
            # Position-by-position analysis
            for i, (chunk, precision) in enumerate(zip(chunk_analysis, precision_at_k)):
                quality_analysis["position_analysis"].append({
                    "position": i + 1,
                    "relevant": chunk["relevant"],
                    "precision_at_position": precision["precision"],
                    "impact": "positive" if chunk["relevant"] else "negative"
                })
            
            # Analyze noise patterns
            irrelevant_count = sum(1 for chunk in chunk_analysis if not chunk["relevant"])
            if irrelevant_count == 0:
                quality_analysis["noise_pattern"] = "no_noise"
            elif irrelevant_count < len(chunk_analysis) * 0.3:
                quality_analysis["noise_pattern"] = "low_noise"
            elif irrelevant_count < len(chunk_analysis) * 0.7:
                quality_analysis["noise_pattern"] = "moderate_noise"
            else:
                quality_analysis["noise_pattern"] = "high_noise"
            
            return quality_analysis
            
        except Exception as e:
            return {"error": f"Retrieval quality analysis failed: {e}"}
    
    def _generate_summary_verdict(self, chunk_analysis: list, ragas_precision: float, simple_precision: float) -> str:
        """Generate a concise summary verdict for casual users."""
        if not chunk_analysis:
            return "❓ No chunks to analyze"
        
        # Identify relevant chunks
        relevant_chunks = [i+1 for i, chunk in enumerate(chunk_analysis) if chunk["relevant"]]
        irrelevant_chunks = [i+1 for i, chunk in enumerate(chunk_analysis) if not chunk["relevant"]]
        
        # Generate verdict based on pattern
        if ragas_precision >= 0.8 and simple_precision >= 0.8:
            return "✅ Excellent retrieval: good ranking AND low noise"
        elif ragas_precision >= 0.8 and simple_precision < 0.5:
            if len(relevant_chunks) > 0 and relevant_chunks[0] == 1:
                if len(irrelevant_chunks) > 2:
                    chunk_range = f"{irrelevant_chunks[0]}–{irrelevant_chunks[-1]}" if len(irrelevant_chunks) > 1 else str(irrelevant_chunks[0])
                    return f"✅ Good ranking early; ⚠️ noisy tail — prune chunks {chunk_range}"
                else:
                    return "✅ Good ranking with minor noise"
            else:
                return "⚠️ Good overall ranking but some noise present"
        elif ragas_precision < 0.5 and simple_precision >= 0.8:
            return "⚠️ Clean content but poor ranking — relevant chunks appear late"
        else:
            return "❌ Poor retrieval: both ranking and noise issues need fixing"
    
    def _generate_action_recommendations(self, chunk_analysis: list, ragas_precision: float, simple_precision: float) -> dict:
        """Generate structured action recommendations."""
        if not chunk_analysis:
            return {"action": "none", "reason": "no_chunks_to_analyze"}
        
        recommendations = {
            "action": "none",
            "target_ranks": [],
            "keep_ranks": [],
            "reason": "",
            "confidence": "low"
        }
        
        # Identify relevant and irrelevant chunk positions
        relevant_positions = [i+1 for i, chunk in enumerate(chunk_analysis) if chunk["relevant"]]
        irrelevant_positions = [i+1 for i, chunk in enumerate(chunk_analysis) if not chunk["relevant"]]
        
        # Generate recommendations based on patterns
        if ragas_precision >= 0.8 and simple_precision < 0.5:
            # Good ranking but noisy - recommend pruning
            if len(relevant_positions) > 0 and relevant_positions[0] <= 2:  # Early relevant chunks
                recommendations.update({
                    "action": "prune",
                    "target_ranks": irrelevant_positions,
                    "keep_ranks": relevant_positions,
                    "reason": "high_noise_with_good_early_ranking",
                    "confidence": "high"
                })
            else:
                recommendations.update({
                    "action": "rerank_and_prune",
                    "target_ranks": irrelevant_positions,
                    "reason": "noise_with_mixed_ranking",
                    "confidence": "medium"
                })
        
        elif ragas_precision < 0.5 and simple_precision >= 0.8:
            # Poor ranking but clean content - recommend reranking
            recommendations.update({
                "action": "rerank",
                "target_ranks": list(range(1, len(chunk_analysis) + 1)),
                "reason": "poor_ranking_with_clean_content",
                "confidence": "high"
            })
        
        elif ragas_precision < 0.5 and simple_precision < 0.5:
            # Both poor - recommend full overhaul
            recommendations.update({
                "action": "overhaul_retrieval",
                "target_ranks": irrelevant_positions,
                "reason": "poor_ranking_and_high_noise",
                "confidence": "high"
            })
        
        else:
            # Good performance overall
            recommendations.update({
                "action": "maintain",
                "reason": "good_performance",
                "confidence": "high"
            })
        
        return recommendations 