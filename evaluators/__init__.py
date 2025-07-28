"""
Evaluator registry for RAGTESTERCLI

Manages available evaluators and provides factory function.
"""

from typing import Dict, Type, Optional, Any
from .base_evaluator import BaseEvaluator
from .ragas_faithfulness import RagasFaithfulnessEvaluator
from .ragas_context_precision import RagasContextPrecisionEvaluator
from .ragchecker_hallucination import RAGCheckerHallucinationEvaluator

# Registry of available evaluators
EVALUATORS: Dict[str, Type[BaseEvaluator]] = {
    "faithfulness_ragas": RagasFaithfulnessEvaluator,
    "context_precision_ragas": RagasContextPrecisionEvaluator,
    "hallucination_ragchecker": RAGCheckerHallucinationEvaluator,
}

# Descriptions for each evaluator
EVALUATOR_DESCRIPTIONS: Dict[str, str] = {
    "faithfulness_ragas": "RAGAS faithfulness metric - measures how faithful the answer is to the given context",
    "context_precision_ragas": "RAGAS context precision metric - measures proportion of relevant chunks in retrieved contexts",
    "hallucination_ragchecker": "RAGChecker hallucination metric - detects hallucinations in generated responses",
}

def get_evaluator(metric_name: str, model_config: Optional[Dict[str, Any]] = None) -> BaseEvaluator:
    """
    Get evaluator instance for the specified metric.
    
    Args:
        metric_name: Name of the evaluation metric
        model_config: Optional model configuration for flexible LLM support
    
    Returns:
        BaseEvaluator: Instance of the requested evaluator
    
    Raises:
        ValueError: If metric_name is not supported
    """
    if metric_name not in EVALUATORS:
        available = ", ".join(EVALUATORS.keys())
        raise ValueError(f"Unknown metric: {metric_name}. Available metrics: {available}")
    
    evaluator_class = EVALUATORS[metric_name]
    
    # Pass model configuration if evaluator supports it
    try:
        return evaluator_class(model_config=model_config)
    except TypeError:
        # Fallback for evaluators that don't accept model_config yet
        return evaluator_class()

def get_available_metrics() -> Dict[str, str]:
    """Get dictionary of available metrics and their descriptions."""
    return EVALUATOR_DESCRIPTIONS.copy() 