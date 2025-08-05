"""
RAGChecker Hallucination Evaluator
Pure wrapper around official RAGChecker library with multi-provider support

RAGChecker Hallucination Formula:
    hallucination = np.mean(unfaithful & ~answer2response)
    
Where:
    - unfaithful = ~np.max(retrieved2response, axis=1) 
      (claims NOT supported by ANY retrieved chunk)
    - ~answer2response 
      (claims that are incorrect according to ground truth)
    - unfaithful & ~answer2response 
      (claims that are BOTH incorrect AND unsupported by chunks)
    - np.mean(...) 
      (average over ALL response claims, not just incorrect ones)

Result: Percentage of ALL response claims that are hallucinated
        (incorrect relative to ground truth AND unsupported by any retrieved chunk)
"""

import os
from typing import Dict, Any, Optional
from ragchecker import RAGChecker, RAGResults
from ragchecker.metrics import hallucination
from .base_evaluator import BaseEvaluator

class RAGCheckerHallucinationEvaluator(BaseEvaluator):
    """
    Evaluate hallucination using RAGChecker library with unified LLM configuration.
    
    RAGChecker measures hallucination as the fraction of response claims that are:
    1. Incorrect according to ground truth AND 
    2. Unsupported by any retrieved chunk
    
    This captures pure inventions - claims that are both factually wrong 
    and have no basis in the provided context.
    """
    
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
        """
        Evaluate hallucination using RAGChecker library.
        
        Returns:
            float: Hallucination score as percentage (0-100)
                   Computed by official RAGChecker using:
                   np.mean(unfaithful & ~answer2response) * 100
                   
                   Represents the percentage of ALL response claims that are:
                   - Incorrect according to ground truth AND
                   - Unsupported by any retrieved chunk
                   
                   Note: This is a pure wrapper - we call RAGChecker.evaluate() 
                   and extract their computed score.
        """
        question = data.get('question', '')
        context = data.get('context', [])
        answer = data.get('answer', '')
        ground_truth = data.get('ground_truth', '')  # Optional ground truth
        
        # Ensure context is a list
        if isinstance(context, str):
            context = [context]
        
        # Build RAGChecker input format
        rag_data = {
            "results": [
                {
                    "query_id": "1",
                    "query": question,
                    "gt_answer": ground_truth,  # Use actual ground truth if provided
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
        ground_truth = data.get('ground_truth', '')  # Optional ground truth
        
        if isinstance(context, str):
            context = [context]
        
        rag_data = {
            "results": [
                {
                    "query_id": "1",
                    "query": question,
                    "gt_answer": ground_truth,  # Use actual ground truth if provided
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
        Get detailed hallucination analysis with claim-level diagnostics.
        
        Returns comprehensive analysis including:
        - Step-by-step claim verification process
        - Formula explanations and calculations
        - Context chunk relevance analysis
        - Detailed diagnostic insights
        - Summary verdict and action recommendations
        """
        try:
            raw_output = self.get_raw_output(data)
            
            question = data.get('question', '')
            context = data.get('context', [])
            answer = data.get('answer', '')
            ground_truth = data.get('ground_truth', '')
            
            if isinstance(context, str):
                context = [context]
            
            # Extract the detailed analysis from RAGChecker output
            hallucination_score = self._extract_hallucination_score(raw_output)
            
            # Get comprehensive claim-level analysis
            claim_analysis = self._extract_claim_level_analysis(raw_output)
            
            # Get context relevance analysis
            context_analysis = self._extract_context_analysis(raw_output, context)
            
            # Calculate additional metrics
            total_claims = claim_analysis.get("total_claims", 0)
            hallucinated_claims = claim_analysis.get("summary", {}).get("hallucinated_claims", 0)
            correct_grounded_claims = claim_analysis.get("summary", {}).get("correct_grounded_claims", 0)
            context_supported_but_incorrect = claim_analysis.get("summary", {}).get("context_supported_but_incorrect", 0)
            missing_context_evidence = claim_analysis.get("summary", {}).get("missing_context_evidence", 0)
            
            # Calculate additional rates
            hallucination_rate = hallucinated_claims / total_claims if total_claims > 0 else 0.0
            grounding_rate = correct_grounded_claims / total_claims if total_claims > 0 else 0.0
            context_support_rate = (correct_grounded_claims + context_supported_but_incorrect) / total_claims if total_claims > 0 else 0.0
            
            # Get diagnostic insights
            diagnostic_insights = self._generate_diagnostic_insights(claim_analysis, context_analysis, hallucination_score)
            
            # Generate summary verdict
            summary_verdict = self._generate_summary_verdict(hallucination_score, hallucinated_claims, total_claims, bool(ground_truth))
            
            # Generate action recommendations
            action_recommendations = self._generate_action_recommendations(claim_analysis, context_analysis, hallucination_score, bool(ground_truth))
            
            # Generate formula explanation
            formula_explanation = self._generate_formula_explanation(hallucination_score, hallucinated_claims, total_claims, bool(ground_truth))
            
            return {
                "question": question,
                "answer": answer,
                "ground_truth": ground_truth,
                "retrieved_contexts": context,
                "hallucination_score": hallucination_score,
                "hallucination_percentage": hallucination_score,
                "evaluation_mode": "full" if ground_truth else "context_only",
                "interpretation": self._interpret_score(hallucination_score, bool(ground_truth)),
                
                # Formula and calculation details
                "formula_explanation": formula_explanation,
                
                # Detailed claim-level analysis
                "claim_analysis": claim_analysis,
                "context_analysis": context_analysis,
                "diagnostic_insights": diagnostic_insights,
                
                # Summary and recommendations
                "summary_verdict": summary_verdict,
                "action_recommendations": action_recommendations,
                
                # Additional metrics
                "summary_statistics": {
                    "total_claims": total_claims,
                    "hallucinated_claims": hallucinated_claims,
                    "correct_grounded_claims": correct_grounded_claims,
                    "context_supported_but_incorrect": context_supported_but_incorrect,
                    "missing_context_evidence": missing_context_evidence,
                    "hallucination_rate": hallucination_rate,
                    "grounding_rate": grounding_rate,
                    "context_support_rate": context_support_rate,
                    "claim_density": total_claims / len(answer.split()) if answer else 0.0,
                    "average_claim_length": sum(len(claim.get("claim_text", "").split()) for claim in claim_analysis.get("claims", [])) / total_claims if total_claims > 0 else 0.0
                },
                
                # Legacy fields for backward compatibility
                "analysis_breakdown": self._extract_detailed_breakdown(raw_output),
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
    
    def _extract_claim_level_analysis(self, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """Extract claim-level entailment analysis from RAGChecker output."""
        try:
            results = raw_output.get("results", [])
            if not results:
                return {"error": "No results found", "claims": []}
            
            result = results[0]
            response_claims = result.get("response_claims", [])
            retrieved2response = result.get("retrieved2response", [])
            answer2response = result.get("answer2response", [])
            
            claims_analysis = []
            
            for i, claim in enumerate(response_claims):
                # Handle both list format (tokens) and string format
                if isinstance(claim, list):
                    claim_text = ' '.join(claim)
                else:
                    claim_text = str(claim)
                    
                claim_data = {
                    "claim_number": i + 1,
                    "claim_text": claim_text,
                    "context_support": {
                        "supported": False,
                        "best_chunk_index": None,
                        "entailment_scores": [],
                        "explanation": ""
                    },
                    "ground_truth_support": {
                        "supported": False,
                        "entailment_score": 0.0,
                        "explanation": ""
                    },
                    "classification": "unknown"
                }
                
                # Analyze context support
                if i < len(retrieved2response):
                    context_scores = retrieved2response[i] if isinstance(retrieved2response[i], list) else [retrieved2response[i]]
                    claim_data["context_support"]["entailment_scores"] = context_scores
                    
                    # Find best supporting chunk - handle RAGChecker string format
                    if context_scores:
                        def score_to_float(score):
                            if score in ['True', 'Entailment']:
                                return 1.0
                            elif score in ['False', 'Contradiction']:
                                return 0.0
                            elif score in ['Neutral']:
                                return 0.5
                            elif isinstance(score, (int, float)):
                                return float(score)
                            else:
                                return 0.0
                        
                        numeric_scores = [score_to_float(s) for s in context_scores]
                        max_score_idx = max(range(len(numeric_scores)), key=lambda x: numeric_scores[x])
                        best_score = numeric_scores[max_score_idx]
                        best_score_label = context_scores[max_score_idx]
                        
                        if best_score > 0.7:  # Strong support
                            claim_data["context_support"]["supported"] = True
                            claim_data["context_support"]["best_chunk_index"] = max_score_idx
                            claim_data["context_support"]["explanation"] = f"Supported by chunk {max_score_idx + 1} ({best_score_label})"
                        elif best_score > 0.3:  # Weak support
                            claim_data["context_support"]["explanation"] = f"Weak support from chunk {max_score_idx + 1} ({best_score_label})"
                        else:
                            claim_data["context_support"]["explanation"] = f"No sufficient context support ({best_score_label})"
                
                # Analyze ground truth support
                if i < len(answer2response):
                    gt_score = answer2response[i]
                    
                    # Convert RAGChecker string format to numeric
                    if gt_score in ['True', 'Entailment']:
                        gt_numeric = 1.0
                        supported = True
                    elif gt_score in ['False', 'Contradiction']:
                        gt_numeric = 0.0
                        supported = False
                    elif gt_score in ['Neutral']:
                        gt_numeric = 0.5
                        supported = False  # Neutral is not strong support
                    elif isinstance(gt_score, (int, float)):
                        gt_numeric = float(gt_score)
                        supported = gt_numeric > 0.7
                    else:
                        gt_numeric = 0.0
                        supported = False
                    
                    claim_data["ground_truth_support"]["supported"] = supported
                    claim_data["ground_truth_support"]["entailment_score"] = gt_numeric
                    claim_data["ground_truth_support"]["explanation"] = f"Ground truth: {gt_score}" + (" (supported)" if supported else " (not supported)")
                
                # Classify the claim
                context_supported = claim_data["context_support"]["supported"]
                gt_supported = claim_data["ground_truth_support"]["supported"]
                
                if context_supported and gt_supported:
                    claim_data["classification"] = "correct_grounded"
                elif context_supported and not gt_supported:
                    claim_data["classification"] = "context_supported_but_incorrect"
                elif not context_supported and gt_supported:
                    claim_data["classification"] = "missing_context_evidence"
                else:
                    claim_data["classification"] = "hallucination"
                
                claims_analysis.append(claim_data)
            
            return {
                "total_claims": len(response_claims),
                "claims": claims_analysis,
                "summary": self._summarize_claim_analysis(claims_analysis)
            }
            
        except Exception as e:
            return {"error": f"Could not extract claim analysis: {e}", "claims": []}
    
    def _extract_context_analysis(self, raw_output: Dict[str, Any], contexts: list) -> Dict[str, Any]:
        """Analyze context chunk relevance and usage."""
        try:
            results = raw_output.get("results", [])
            if not results:
                return {"error": "No results found", "chunks": []}
            
            result = results[0]
            retrieved2response = result.get("retrieved2response", [])
            
            chunk_analysis = []
            
            for chunk_idx, chunk_text in enumerate(contexts):
                chunk_data = {
                    "chunk_number": chunk_idx + 1,
                    "chunk_text": chunk_text,
                    "supports_claims": [],
                    "relevance_score": 0.0,
                    "usage_analysis": "unused"
                }
                
                # Check which claims this chunk supports
                supporting_claims = 0
                total_support_score = 0.0
                
                for claim_idx, claim_entailments in enumerate(retrieved2response):
                    if isinstance(claim_entailments, list) and chunk_idx < len(claim_entailments):
                        entailment = claim_entailments[chunk_idx]
                        
                        # Convert RAGChecker string format to numeric
                        if entailment in ['True', 'Entailment']:
                            score = 1.0
                        elif entailment in ['False', 'Contradiction']:
                            score = 0.0
                        elif entailment in ['Neutral']:
                            score = 0.5
                        elif isinstance(entailment, (int, float)):
                            score = float(entailment)
                        else:
                            score = 0.0
                        
                        if score > 0.5:
                            chunk_data["supports_claims"].append({
                                "claim_number": claim_idx + 1,
                                "entailment_score": score
                            })
                            supporting_claims += 1
                            total_support_score += score
                
                # Calculate relevance and usage
                if supporting_claims > 0:
                    chunk_data["relevance_score"] = total_support_score / len(retrieved2response) if retrieved2response else 0.0
                    chunk_data["usage_analysis"] = f"supports_{supporting_claims}_claims"
                else:
                    chunk_data["usage_analysis"] = "irrelevant_noise"
                
                chunk_analysis.append(chunk_data)
            
            return {
                "total_chunks": len(contexts),
                "chunks": chunk_analysis,
                "summary": self._summarize_context_analysis(chunk_analysis)
            }
            
        except Exception as e:
            return {"error": f"Could not extract context analysis: {e}", "chunks": []}
    
    def _summarize_claim_analysis(self, claims: list) -> Dict[str, Any]:
        """Generate summary statistics for claim analysis."""
        if not claims:
            return {}
        
        total = len(claims)
        correct_grounded = sum(1 for c in claims if c["classification"] == "correct_grounded")
        hallucinations = sum(1 for c in claims if c["classification"] == "hallucination")
        missing_context = sum(1 for c in claims if c["classification"] == "missing_context_evidence")
        context_but_wrong = sum(1 for c in claims if c["classification"] == "context_supported_but_incorrect")
        
        return {
            "total_claims": total,
            "correct_grounded_claims": correct_grounded,
            "hallucinated_claims": hallucinations,
            "missing_context_evidence": missing_context,
            "context_supported_but_incorrect": context_but_wrong,
            "hallucination_rate": hallucinations / total if total > 0 else 0.0,
            "grounding_rate": correct_grounded / total if total > 0 else 0.0
        }
    
    def _summarize_context_analysis(self, chunks: list) -> Dict[str, Any]:
        """Generate summary statistics for context analysis."""
        if not chunks:
            return {}
        
        total = len(chunks)
        relevant_chunks = sum(1 for c in chunks if c["usage_analysis"] != "irrelevant_noise")
        unused_chunks = sum(1 for c in chunks if c["usage_analysis"] == "unused")
        noise_chunks = sum(1 for c in chunks if c["usage_analysis"] == "irrelevant_noise")
        
        avg_relevance = sum(c["relevance_score"] for c in chunks) / total if total > 0 else 0.0
        
        return {
            "total_chunks": total,
            "relevant_chunks": relevant_chunks,
            "unused_chunks": unused_chunks,
            "noise_chunks": noise_chunks,
            "context_precision": relevant_chunks / total if total > 0 else 0.0,
            "average_relevance_score": avg_relevance
        }
    
    def _generate_diagnostic_insights(self, claim_analysis: Dict[str, Any], context_analysis: Dict[str, Any], hallucination_score: float) -> Dict[str, Any]:
        """Generate actionable diagnostic insights."""
        insights = {
            "primary_issues": [],
            "recommendations": [],
            "severity": "low"
        }
        
        # Analyze claim patterns
        if "summary" in claim_analysis:
            summary = claim_analysis["summary"]
            
            if summary.get("hallucination_rate", 0) > 0.3:
                insights["primary_issues"].append("High hallucination rate detected")
                insights["recommendations"].append("Review prompt engineering to emphasize grounding in context")
                insights["severity"] = "high"
            
            if summary.get("missing_context_evidence", 0) > 0:
                insights["primary_issues"].append("Claims missing context evidence")
                insights["recommendations"].append("Improve retrieval recall - expand query or increase chunk count")
        
        # Analyze context patterns
        if "summary" in context_analysis:
            ctx_summary = context_analysis["summary"]
            
            if ctx_summary.get("context_precision", 0) < 0.5:
                insights["primary_issues"].append("Low context precision - too much irrelevant content")
                insights["recommendations"].append("Improve retrieval precision - refine similarity thresholds or reranking")
            
            if ctx_summary.get("noise_chunks", 0) > 2:
                insights["primary_issues"].append("Significant context noise detected")
                insights["recommendations"].append("Consider more aggressive filtering or better semantic matching")
        
        # Set severity based on overall score
        if hallucination_score > 75:
            insights["severity"] = "critical"
        elif hallucination_score > 50:
            insights["severity"] = "high"
        elif hallucination_score > 25:
            insights["severity"] = "medium"
        
        return insights
    
    def _interpret_score(self, score: float, has_ground_truth: bool = False) -> str:
        """
        Provide human-readable interpretation of hallucination score.
        
        Score represents the percentage of ALL response claims that are both:
        - Incorrect according to ground truth AND
        - Unsupported by any retrieved chunk (pure inventions)
        """
        base_context = "context" if not has_ground_truth else "context and ground truth"
        
        if score == 0.0:
            return f"No pure hallucinations detected - all claims are either correct or supported by {base_context}"
        elif score <= 25.0:
            return f"Low hallucination risk - {score:.1f}% of claims are pure inventions (incorrect + unsupported by {base_context})"
        elif score <= 50.0:
            return f"Moderate hallucination risk - {score:.1f}% of claims are pure inventions (incorrect + unsupported by {base_context})"
        elif score <= 75.0:
            return f"High hallucination risk - {score:.1f}% of claims are pure inventions (incorrect + unsupported by {base_context})"
        else:
            return f"Very high hallucination risk - {score:.1f}% of claims are pure inventions (incorrect + unsupported by {base_context})"
    
    def _generate_formula_explanation(self, hallucination_score: float, hallucinated_claims: int, total_claims: int, has_ground_truth: bool) -> Dict[str, Any]:
        """Generate detailed formula explanation for RAGChecker hallucination."""
        if has_ground_truth:
            formula = "np.mean(unfaithful & ~answer2response) * 100"
            calculation = f"Hallucinated Claims / Total Claims = {hallucinated_claims} / {total_claims} = {hallucination_score:.1f}%"
            interpretation = f"{hallucination_score:.1f}% of ALL response claims are pure hallucinations (incorrect + unsupported by context)"
            explanation = "Pure hallucinations = claims that are BOTH incorrect according to ground truth AND unsupported by any retrieved chunk"
        else:
            formula = "np.mean(unfaithful) * 100"
            calculation = f"Unsupported Claims / Total Claims = {hallucinated_claims} / {total_claims} = {hallucination_score:.1f}%"
            interpretation = f"{hallucination_score:.1f}% of ALL response claims are unsupported by context"
            explanation = "Context-only evaluation = claims that are NOT supported by any retrieved chunk (regardless of correctness)"
        
        return {
            "ragchecker_formula": formula,
            "calculation": calculation,
            "interpretation": interpretation,
            "explanation": explanation,
            "evaluation_mode": "full" if has_ground_truth else "context_only"
        }
    
    def _generate_summary_verdict(self, hallucination_score: float, hallucinated_claims: int, total_claims: int, has_ground_truth: bool) -> str:
        """Generate a concise, human-readable summary verdict."""
        mode_text = "with ground truth" if has_ground_truth else "context-only"
        
        if hallucination_score == 0.0:
            return f"✅ No hallucinations detected ({hallucination_score:.1f}%) - {hallucinated_claims}/{total_claims} claims are pure inventions ({mode_text})"
        elif hallucination_score <= 25.0:
            return f"✅ Low hallucination risk ({hallucination_score:.1f}%) - {hallucinated_claims}/{total_claims} claims are pure inventions ({mode_text})"
        elif hallucination_score <= 50.0:
            return f"⚠️  Moderate hallucination risk ({hallucination_score:.1f}%) - {hallucinated_claims}/{total_claims} claims are pure inventions ({mode_text})"
        elif hallucination_score <= 75.0:
            return f"❌ High hallucination risk ({hallucination_score:.1f}%) - {hallucinated_claims}/{total_claims} claims are pure inventions ({mode_text})"
        else:
            return f"🚨 Critical hallucination risk ({hallucination_score:.1f}%) - {hallucinated_claims}/{total_claims} claims are pure inventions ({mode_text})"
    
    def _generate_action_recommendations(self, claim_analysis: Dict[str, Any], context_analysis: Dict[str, Any], hallucination_score: float, has_ground_truth: bool) -> Dict[str, Any]:
        """Generate structured action recommendations based on analysis."""
        recommendations = {
            "action": "none",
            "reason": "no_issues_detected",
            "confidence": "low",
            "specific_actions": []
        }
        
        # Analyze claim patterns
        claim_summary = claim_analysis.get("summary", {})
        hallucinated_claims = claim_summary.get("hallucinated_claims", 0)
        correct_grounded_claims = claim_summary.get("correct_grounded_claims", 0)
        context_supported_but_incorrect = claim_summary.get("context_supported_but_incorrect", 0)
        missing_context_evidence = claim_summary.get("missing_context_evidence", 0)
        
        # Analyze context patterns
        context_summary = context_analysis.get("summary", {})
        context_precision = context_summary.get("context_precision", 0.0)
        noise_chunks = context_summary.get("noise_chunks", 0)
        
        if hallucination_score > 75:
            recommendations["action"] = "critical_fix"
            recommendations["reason"] = "critical_hallucination_risk"
            recommendations["confidence"] = "high"
            recommendations["specific_actions"].append("Immediate review of generation constraints")
            recommendations["specific_actions"].append("Add strict fact-checking to generation pipeline")
            recommendations["specific_actions"].append("Consider using more conservative generation parameters")
        
        elif hallucination_score > 50:
            recommendations["action"] = "improve_generation"
            recommendations["reason"] = "high_hallucination_risk"
            recommendations["confidence"] = "high"
            recommendations["specific_actions"].append("Review prompt engineering to emphasize grounding")
            recommendations["specific_actions"].append("Add fact-checking constraints to generation")
            recommendations["specific_actions"].append("Consider using more conservative temperature settings")
        
        elif hallucination_score > 25:
            recommendations["action"] = "optimize_retrieval"
            recommendations["reason"] = "moderate_hallucination_risk"
            recommendations["confidence"] = "medium"
            recommendations["specific_actions"].append("Improve retrieval to provide more relevant context")
            recommendations["specific_actions"].append("Expand query formulation for better context matching")
            recommendations["specific_actions"].append("Consider increasing the number of retrieved chunks")
        
        elif context_precision < 0.5:
            recommendations["action"] = "improve_retrieval"
            recommendations["reason"] = "low_context_precision"
            recommendations["confidence"] = "medium"
            recommendations["specific_actions"].append("Improve retrieval precision through better similarity matching")
            recommendations["specific_actions"].append("Add reranking to filter irrelevant chunks")
            recommendations["specific_actions"].append("Review embedding model quality for domain")
        
        elif hallucination_score <= 10:
            recommendations["action"] = "maintain"
            recommendations["reason"] = "good_performance"
            recommendations["confidence"] = "high"
            recommendations["specific_actions"].append("Current approach is working well")
            recommendations["specific_actions"].append("Monitor for consistency across different queries")
            recommendations["specific_actions"].append("Consider fine-tuning for domain-specific improvements")
        
        return recommendations
