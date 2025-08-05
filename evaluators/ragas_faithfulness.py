"""
RAGAS Faithfulness Evaluator
Implements claim-based faithfulness scoring following official RAGAS methodology
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
    """
    Evaluate faithfulness using RAGAS library with claim-based scoring.
    
    Follows the official RAGAS process:
    1. Extract discrete claims from the generated answer
    2. Verify each claim against the retrieved context
    3. Compute ratio: supported_claims / total_claims
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
            from langchain_anthropic import ChatAnthropic
            anthropic_key = os.getenv("ANTHROPIC_API_KEY", self.api_key)
            return ChatAnthropic(
                model=self.model.replace("anthropic/", ""),
                anthropic_api_key=anthropic_key,
                temperature=0
            )
        elif "gemini" in model_lower or "google" in model_lower:
            # Direct Google API (not via OpenRouter)
            from langchain_google_genai import ChatGoogleGenerativeAI
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
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
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
        Evaluate faithfulness using official RAGAS claim-based process.
        
        Process:
        1. Break generated answer into discrete claims
        2. Verify each claim against retrieved context  
        3. Return ratio: supported_claims / total_claims
        
        Uses official RAGAS library which implements this methodology internally.
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
            # Create dataset for RAGAS evaluation
            dataset = Dataset.from_dict({
                "question": [question],
                "contexts": [context],
                "answer": [answer],
            })
            
            # Configure RAGAS with our LLM
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            
            llm_wrapper = LangchainLLMWrapper(self.llm)
            emb_wrapper = LangchainEmbeddingsWrapper(self.embeddings)
            
            # Use RAGAS evaluate with proper wrappers
            result = evaluate(
                dataset,
                llm=llm_wrapper,
                embeddings=emb_wrapper,
                metrics=[faithfulness],
                run_config=self.run_config
            )
            
            # Extract score from result
            score = float(result.to_pandas()['faithfulness'].iloc[0])
            print("✓ Using official RAGAS faithfulness implementation")
            return round(score, 3)
        
        except Exception as e:
            print(f"Error evaluating with RAGAS faithfulness: {str(e)}")
            print("→ Using fallback manual implementation")
            # Fallback to manual claim-based evaluation
            return self._fallback_evaluation(question, context, answer)
    
    def _fallback_evaluation(self, question: str, context: list, answer: str) -> float:
        """
        Fallback evaluation using simplified claim-based approach.
        
        When RAGAS fails, manually implement the core faithfulness logic:
        1. Extract claims from answer
        2. Check each claim against context
        3. Return supported_claims / total_claims
        """
        from llm_client import LLMClient
        
        try:
            client = LLMClient()
            context_text = "\n".join(context) if isinstance(context, list) else str(context)
            
            # Step 1: Extract claims from answer
            claims_prompt = f"""
Break the following answer into individual factual claims. List each claim on a new line:

Answer: {answer}

Claims:"""
            
            claims_response = client.generate_response(claims_prompt)
            # Filter out instruction text and only keep actual claims
            raw_lines = [line.strip() for line in claims_response.split('\n') if line.strip()]
            claims = []
            for line in raw_lines:
                # Skip common instruction/header lines
                if any(phrase in line.lower() for phrase in [
                    'here are the individual factual claims',
                    'individual factual claims from',
                    'factual claims:',
                    'claims:'
                ]):
                    continue
                # Skip lines that look like numbered headers (just numbers and periods)
                if line.strip().endswith(':') and len(line.strip()) < 20:
                    continue
                
                # Clean up the claim text - remove numbering
                clean_claim = line.strip()
                # Remove common numbering patterns: "1.", "1)", "(1)", "- ", "• "
                import re
                clean_claim = re.sub(r'^\s*[\d]+[\.\)]\s*', '', clean_claim)  # "1. " or "1) "
                clean_claim = re.sub(r'^\s*\([\d]+\)\s*', '', clean_claim)   # "(1) "
                clean_claim = re.sub(r'^\s*[-•]\s*', '', clean_claim)        # "- " or "• "
                clean_claim = clean_claim.strip()
                
                if clean_claim:  # Only add non-empty claims
                    claims.append(clean_claim)
            
            if not claims:
                return 0.0
            
            # Step 2: Verify each claim against context
            supported_claims = 0
            for claim in claims:
                verification_prompt = f"""
Can the following claim be supported by the given context? Answer only 'YES' or 'NO'.

Context: {context_text}

Claim: {claim}

Answer:"""
                
                verification_response = client.generate_response(verification_prompt).strip().upper()
                if verification_response.startswith('YES'):
                    supported_claims += 1
            
            # Step 3: Calculate faithfulness score
            faithfulness_score = supported_claims / len(claims)
            return round(faithfulness_score, 3)
            
        except Exception as e:
            print(f"Fallback evaluation failed: {e}")
            return 0.5
    
    def get_detailed_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get detailed claim-by-claim analysis for transparency.
        
        Returns comprehensive analysis including:
        - Step-by-step claim verification process
        - Formula explanations and calculations
        - Context utilization analysis
        - Detailed diagnostic insights
        - Summary verdict and action recommendations
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
            context_text = "\n".join(context)
            
            # Extract claims
            claims_prompt = f"""
Break the following answer into individual factual claims. List each claim on a new line:

Answer: {answer}

Claims:"""
            
            claims_response = client.generate_response(claims_prompt)
            # Filter out instruction text and only keep actual claims
            raw_lines = [line.strip() for line in claims_response.split('\n') if line.strip()]
            claims = []
            for line in raw_lines:
                # Skip common instruction/header lines
                if any(phrase in line.lower() for phrase in [
                    'here are the individual factual claims',
                    'individual factual claims from',
                    'factual claims:',
                    'claims:'
                ]):
                    continue
                # Skip lines that look like numbered headers (just numbers and periods)
                if line.strip().endswith(':') and len(line.strip()) < 20:
                    continue
                
                # Clean up the claim text - remove numbering
                clean_claim = line.strip()
                # Remove common numbering patterns: "1.", "1)", "(1)", "- ", "• "
                import re
                clean_claim = re.sub(r'^\s*[\d]+[\.\)]\s*', '', clean_claim)  # "1. " or "1) "
                clean_claim = re.sub(r'^\s*\([\d]+\)\s*', '', clean_claim)   # "(1) "
                clean_claim = re.sub(r'^\s*[-•]\s*', '', clean_claim)        # "- " or "• "
                clean_claim = clean_claim.strip()
                
                if clean_claim:  # Only add non-empty claims
                    claims.append(clean_claim)
            
            # Verify each claim with detailed process
            claim_analysis = []
            supported_claims = 0
            verification_steps = []
            
            for i, claim in enumerate(claims, 1):
                verification_prompt = f"""
Can the following claim be supported by the given context? Answer only 'YES' or 'NO'.

Context: {context_text}

Claim: {claim}

Answer:"""
                
                verification_response = client.generate_response(verification_prompt).strip().upper()
                is_supported = verification_response.startswith('YES')
                
                if is_supported:
                    supported_claims += 1
                
                # Create detailed verification step
                verification_step = {
                    "step_number": i,
                    "claim": claim,
                    "verification_prompt": verification_prompt,
                    "verification_response": verification_response,
                    "is_supported": is_supported,
                    "support_status": "SUPPORTED" if is_supported else "NOT SUPPORTED"
                }
                verification_steps.append(verification_step)
                
                claim_analysis.append({
                    "claim_number": i,
                    "claim_text": claim,
                    "supported": is_supported,
                    "verification_response": verification_response,
                    "verification_step": verification_step
                })
            
            # Calculate faithfulness score with detailed formula
            total_claims = len(claims)
            faithfulness_score = supported_claims / total_claims if total_claims > 0 else 0.0
            unsupported_claims = total_claims - supported_claims
            
            # Calculate additional metrics
            support_rate = supported_claims / total_claims if total_claims > 0 else 0.0
            unsupported_rate = unsupported_claims / total_claims if total_claims > 0 else 0.0
            
            # Generate enhanced diagnostic insights
            diagnostic_insights = self._generate_faithfulness_insights(claim_analysis, faithfulness_score)
            
            # Generate context usage analysis
            context_usage = self._analyze_context_usage(claims, context, client)
            
            # Generate summary verdict
            summary_verdict = self._generate_summary_verdict(faithfulness_score, supported_claims, total_claims)
            
            # Generate action recommendations
            action_recommendations = self._generate_action_recommendations(claim_analysis, context_usage, faithfulness_score)
            
            return {
                "question": question,
                "answer": answer,
                "context": context,
                "total_claims": total_claims,
                "supported_claims": supported_claims,
                "unsupported_claims": unsupported_claims,
                "faithfulness_score": faithfulness_score,
                "support_rate": support_rate,
                "unsupported_rate": unsupported_rate,
                
                # Detailed verification process
                "verification_steps": verification_steps,
                "claim_analysis": claim_analysis,
                
                # Formula and calculation details
                "formula_explanation": {
                    "faithfulness_formula": "Supported Claims / Total Claims",
                    "calculation": f"{supported_claims} / {total_claims} = {faithfulness_score:.3f}",
                    "interpretation": f"{faithfulness_score:.1%} of claims are supported by context"
                },
                
                # Context usage analysis
                "context_usage_analysis": context_usage,
                
                # Diagnostic insights
                "diagnostic_insights": diagnostic_insights,
                
                # Summary and recommendations
                "summary_verdict": summary_verdict,
                "action_recommendations": action_recommendations,
                
                # Additional metrics
                "summary_statistics": {
                    "support_rate": support_rate,
                    "unsupported_rate": unsupported_rate,
                    "claim_density": total_claims / len(answer.split()) if answer else 0.0,
                    "average_claim_length": sum(len(claim.split()) for claim in claims) / total_claims if total_claims > 0 else 0.0
                }
            }
            
        except Exception as e:
            return {
                "error": f"Detailed analysis failed: {e}",
                "faithfulness_score": 0.0
            }
    
    def _generate_faithfulness_insights(self, claim_analysis: list, faithfulness_score: float) -> Dict[str, Any]:
        """Generate actionable diagnostic insights for faithfulness."""
        insights = {
            "primary_issues": [],
            "recommendations": [],
            "severity": "low",
            "specific_problems": []
        }
        
        if not claim_analysis:
            return insights
        
        unsupported_claims = [c for c in claim_analysis if not c["supported"]]
        
        # Analyze unsupported claims for patterns
        if len(unsupported_claims) > 0:
            insights["specific_problems"] = [
                {
                    "claim_number": claim["claim_number"],
                    "claim_text": claim["claim_text"],
                    "issue": "Not supported by retrieved context"
                }
                for claim in unsupported_claims
            ]
        
        # Generate recommendations based on faithfulness score
        if faithfulness_score < 0.3:
            insights["primary_issues"].append("Critical faithfulness failure - majority of claims unsupported")
            insights["recommendations"].extend([
                "Review prompt to emphasize grounding in provided context",
                "Check if retrieval is providing sufficient relevant information",
                "Consider post-processing to filter ungrounded claims"
            ])
            insights["severity"] = "critical"
            
        elif faithfulness_score < 0.5:
            insights["primary_issues"].append("Low faithfulness - significant unsupported claims")
            insights["recommendations"].extend([
                "Improve prompt engineering for better context adherence",
                "Evaluate retrieval quality and coverage"
            ])
            insights["severity"] = "high"
            
        elif faithfulness_score < 0.7:
            insights["primary_issues"].append("Moderate faithfulness issues")
            insights["recommendations"].append("Fine-tune generation to better utilize provided context")
            insights["severity"] = "medium"
        
        return insights
    
    def _analyze_context_usage(self, claims: list, context: list, client) -> Dict[str, Any]:
        """Analyze how well context chunks are utilized by checking which claims they support."""
        try:
            chunk_analysis = []
            
            for chunk_idx, chunk_text in enumerate(context):
                chunk_data = {
                    "chunk_number": chunk_idx + 1,
                    "chunk_text": chunk_text,
                    "supporting_claims": [],
                    "usage_status": "unused",
                    "relevance_score": 0.0
                }
                
                # Check which claims this chunk supports
                supporting_claims = 0
                for claim_idx, claim in enumerate(claims):
                    verification_prompt = f"""
Can the following claim be supported by this specific context chunk? Answer only 'YES' or 'NO'.

Context Chunk: {chunk_text}

Claim: {claim}

Answer:"""
                    
                    verification_response = client.generate_response(verification_prompt).strip().upper()
                    is_supported = verification_response.startswith('YES')
                    
                    if is_supported:
                        chunk_data["supporting_claims"].append({
                            "claim_number": claim_idx + 1,
                            "claim_text": claim
                        })
                        supporting_claims += 1
                
                # Determine usage status
                if supporting_claims > 0:
                    chunk_data["usage_status"] = f"supports_{supporting_claims}_claims"
                    chunk_data["relevance_score"] = supporting_claims / len(claims) if claims else 0.0
                else:
                    chunk_data["usage_status"] = "unused_or_irrelevant"
                
                chunk_analysis.append(chunk_data)
            
            return {
                "total_chunks": len(context),
                "chunks": chunk_analysis,
                "summary": self._summarize_context_usage(chunk_analysis)
            }
            
        except Exception as e:
            return {
                "error": f"Context usage analysis failed: {e}",
                "chunks": []
            }
    
    def _summarize_context_usage(self, chunks: list) -> Dict[str, Any]:
        """Generate summary statistics for context usage analysis."""
        if not chunks:
            return {}
        
        total_chunks = len(chunks)
        used_chunks = sum(1 for c in chunks if c["usage_status"] != "unused_or_irrelevant")
        unused_chunks = total_chunks - used_chunks
        
        total_claims_supported = sum(len(c["supporting_claims"]) for c in chunks)
        avg_relevance = sum(c["relevance_score"] for c in chunks) / total_chunks if total_chunks > 0 else 0.0
        
        return {
            "total_chunks": total_chunks,
            "used_chunks": used_chunks,
            "unused_chunks": unused_chunks,
            "context_utilization_rate": used_chunks / total_chunks if total_chunks > 0 else 0.0,
            "total_claims_supported": total_claims_supported,
            "average_relevance_score": avg_relevance
        }
    
    def _generate_summary_verdict(self, faithfulness_score: float, supported_claims: int, total_claims: int) -> str:
        """Generate a concise, human-readable summary verdict."""
        if faithfulness_score >= 0.9:
            return f"✅ Excellent faithfulness ({faithfulness_score:.1%}) - {supported_claims}/{total_claims} claims supported"
        elif faithfulness_score >= 0.7:
            return f"✅ Good faithfulness ({faithfulness_score:.1%}) - {supported_claims}/{total_claims} claims supported"
        elif faithfulness_score >= 0.5:
            return f"⚠️  Moderate faithfulness ({faithfulness_score:.1%}) - {supported_claims}/{total_claims} claims supported"
        elif faithfulness_score >= 0.3:
            return f"❌ Poor faithfulness ({faithfulness_score:.1%}) - {supported_claims}/{total_claims} claims supported"
        else:
            return f"❌ Very poor faithfulness ({faithfulness_score:.1%}) - {supported_claims}/{total_claims} claims supported"
    
    def _generate_action_recommendations(self, claim_analysis: list, context_usage: Dict[str, Any], faithfulness_score: float) -> Dict[str, Any]:
        """Generate structured action recommendations based on analysis."""
        recommendations = {
            "action": "none",
            "reason": "no_issues_detected",
            "confidence": "low",
            "specific_actions": []
        }
        
        # Analyze claim patterns
        unsupported_claims = [c for c in claim_analysis if not c["supported"]]
        supported_claims = [c for c in claim_analysis if c["supported"]]
        
        # Analyze context usage
        context_summary = context_usage.get("summary", {})
        utilization_rate = context_summary.get("context_utilization_rate", 0.0)
        
        if faithfulness_score < 0.5:
            if len(unsupported_claims) > len(supported_claims):
                recommendations["action"] = "improve_generation"
                recommendations["reason"] = "high_unsupported_claims"
                recommendations["confidence"] = "high"
                recommendations["specific_actions"].append("Review prompt engineering to emphasize grounding")
                recommendations["specific_actions"].append("Add fact-checking constraints to generation")
            else:
                recommendations["action"] = "improve_retrieval"
                recommendations["reason"] = "insufficient_context_support"
                recommendations["confidence"] = "medium"
                recommendations["specific_actions"].append("Expand retrieval to include more relevant context")
                recommendations["specific_actions"].append("Improve query formulation for better context matching")
        
        elif utilization_rate < 0.5:
            recommendations["action"] = "optimize_context"
            recommendations["reason"] = "low_context_utilization"
            recommendations["confidence"] = "medium"
            recommendations["specific_actions"].append("Improve context relevance through better retrieval")
            recommendations["specific_actions"].append("Consider context reranking to prioritize useful chunks")
        
        elif faithfulness_score >= 0.8:
            recommendations["action"] = "maintain"
            recommendations["reason"] = "good_faithfulness"
            recommendations["confidence"] = "high"
            recommendations["specific_actions"].append("Current approach is working well")
            recommendations["specific_actions"].append("Monitor for consistency across different queries")
        
        return recommendations 