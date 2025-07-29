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
        
        # Get configuration from environment variables set by CLI
        self.model = os.getenv("RAGCLI_LLM_MODEL", "gpt-4")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
        
        if not self.api_key:
            raise ValueError("API key not found. Set via --api-key flag or .env file.")
        
        # Configure RAGAS LLM for claim extraction and verification
        self.llm = ChatOpenAI(
            model_name=self.model,
            openai_api_key=self.api_key,
            openai_api_base=self.api_base,
            temperature=0  # Deterministic for evaluation
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
        
        Returns:
        - extracted_claims: List of claims found in the answer
        - claim_verification: Each claim with support status
        - faithfulness_score: Final calculated score
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
            
            # Verify each claim
            claim_analysis = []
            supported_claims = 0
            
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
                
                claim_analysis.append({
                    "claim_number": i,
                    "claim_text": claim,
                    "supported": is_supported,
                    "verification_response": verification_response
                })
            
            faithfulness_score = supported_claims / len(claims) if claims else 0.0
            
            return {
                "question": question,
                "answer": answer,
                "context": context,
                "extracted_claims": claims,
                "claim_analysis": claim_analysis,
                "supported_claims": supported_claims,
                "total_claims": len(claims),
                "faithfulness_score": round(faithfulness_score, 3)
            }
            
        except Exception as e:
            return {
                "error": f"Analysis failed: {e}",
                "faithfulness_score": 0.5
            } 