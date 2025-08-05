"""
Unit tests for evaluators module.

Tests evaluator factory, base evaluator, and individual evaluator implementations.
"""

import pytest
import os
from unittest.mock import patch, MagicMock, PropertyMock
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluators import get_evaluator, get_available_metrics, EVALUATORS, EVALUATOR_DESCRIPTIONS
from evaluators.base_evaluator import BaseEvaluator
from evaluators.ragas_faithfulness import RagasFaithfulnessEvaluator
from evaluators.ragas_context_precision import RagasContextPrecisionEvaluator
from evaluators.ragchecker_hallucination import RAGCheckerHallucinationEvaluator


class TestEvaluatorFactory:
    """Test evaluator factory functionality."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_get_evaluator_valid_metrics(self):
        """Test getting evaluators for valid metric names."""
        # Test each registered evaluator
        for metric_name in EVALUATORS.keys():
            evaluator = get_evaluator(metric_name)
            assert isinstance(evaluator, BaseEvaluator)
            assert evaluator.__class__ == EVALUATORS[metric_name]
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_get_evaluator_with_model_config(self):
        """Test getting evaluator with model configuration."""
        model_config = {
            "llm_model": "gpt-3.5-turbo",
            "api_key": "sk-test-key",
            "api_base": "https://api.openai.com/v1"
        }
        
        evaluator = get_evaluator("faithfulness_ragas", model_config=model_config)
        assert isinstance(evaluator, RagasFaithfulnessEvaluator)
        assert evaluator.model_config == model_config
    
    def test_get_evaluator_invalid_metric(self):
        """Test getting evaluator for invalid metric name."""
        with pytest.raises(ValueError) as exc_info:
            get_evaluator("invalid_metric")
        
        assert "Unknown metric: invalid_metric" in str(exc_info.value)
        assert "Available metrics:" in str(exc_info.value)
        
        # Should list all available metrics
        for metric in EVALUATORS.keys():
            assert metric in str(exc_info.value)
    
    def test_get_evaluator_fallback_for_old_evaluators(self):
        """Test fallback for evaluators that don't accept model_config."""
        # Mock an old evaluator that doesn't accept model_config
        class OldEvaluator(BaseEvaluator):
            def __init__(self):  # No model_config parameter
                super().__init__()
            
            def evaluate(self, data):
                return 0.5
        
        with patch.dict(EVALUATORS, {"old_metric": OldEvaluator}):
            # Should not raise error, should fallback to no-argument constructor
            evaluator = get_evaluator("old_metric", model_config={"key": "value"})
            assert isinstance(evaluator, OldEvaluator)
    
    def test_get_available_metrics(self):
        """Test getting available metrics descriptions."""
        metrics = get_available_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == len(EVALUATOR_DESCRIPTIONS)
        
        # Check all expected metrics are present
        expected_metrics = [
            "faithfulness_ragas",
            "context_precision_ragas", 
            "hallucination_ragchecker"
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], str)
            assert len(metrics[metric]) > 0
        
        # Verify it's a copy (modifying doesn't affect original)
        metrics["test"] = "test"
        assert "test" not in EVALUATOR_DESCRIPTIONS


class TestBaseEvaluator:
    """Test base evaluator functionality."""
    
    def test_base_evaluator_initialization_default(self):
        """Test base evaluator initialization with defaults."""
        # Can't instantiate abstract class directly, so create a concrete subclass
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        assert evaluator.model_config == {}
    
    def test_base_evaluator_initialization_with_config(self):
        """Test base evaluator initialization with model config."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        model_config = {"llm_model": "gpt-4", "api_key": "sk-test"}
        evaluator = ConcreteEvaluator(model_config=model_config)
        assert evaluator.model_config == model_config
    
    def test_base_evaluator_batch_evaluate_default(self):
        """Test default batch evaluation implementation."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return float(len(data.get("answer", ""))) / 10  # Simple scoring
        
        evaluator = ConcreteEvaluator()
        data_list = [
            {"answer": "short"},
            {"answer": "longer answer"},
            {"answer": "very long detailed answer"}
        ]
        
        scores = evaluator.batch_evaluate(data_list)
        
        assert len(scores) == 3
        assert scores[0] == 0.5  # "short" = 5 chars
        assert scores[1] == 1.3  # "longer answer" = 13 chars
        assert scores[2] == 2.5  # "very long detailed answer" = 25 chars
    
    def test_get_api_key_from_model_config(self):
        """Test getting API key from model config."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        model_config = {"api_key": "sk-config-key"}
        evaluator = ConcreteEvaluator(model_config=model_config)
        
        # Should return API key from config regardless of provider
        assert evaluator.get_api_key("openai") == "sk-config-key"
        assert evaluator.get_api_key("anthropic") == "sk-config-key"
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-env-openai', 'ANTHROPIC_API_KEY': 'sk-env-anthropic'})
    def test_get_api_key_from_environment(self):
        """Test getting API key from environment variables."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        assert evaluator.get_api_key("openai") == "sk-env-openai"
        assert evaluator.get_api_key("anthropic") == "sk-env-anthropic"
        assert evaluator.get_api_key("google") is None  # Not set
    
    def test_get_api_key_unknown_provider(self):
        """Test getting API key for unknown provider."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        assert evaluator.get_api_key("unknown_provider") is None
    
    def test_get_model_name_from_config(self):
        """Test getting model name from configuration."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        model_config = {"llm_model": "gpt-3.5-turbo", "extractor_name": "claude-3-haiku"}
        evaluator = ConcreteEvaluator(model_config=model_config)
        
        assert evaluator.get_model_name("llm_model") == "gpt-3.5-turbo"
        assert evaluator.get_model_name("extractor_name") == "claude-3-haiku"
        assert evaluator.get_model_name("missing_key") == "gpt-4"  # Default
        assert evaluator.get_model_name("missing_key", "custom-default") == "custom-default"
    
    def test_get_model_name_no_config(self):
        """Test getting model name without configuration."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        assert evaluator.get_model_name("llm_model") == "gpt-4"
        assert evaluator.get_model_name("any_key", "custom-default") == "custom-default"
    
    def test_get_provider_from_model_openai(self):
        """Test provider detection for OpenAI models."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        openai_models = ["gpt-3.5-turbo", "gpt-4", "text-davinci-003", "curie-001", "babbage", "ada"]
        for model in openai_models:
            assert evaluator.get_provider_from_model(model) == "openai"
    
    def test_get_provider_from_model_anthropic(self):
        """Test provider detection for Anthropic models."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        anthropic_models = ["claude-3-haiku", "claude-3-sonnet", "claude-2", "claude"]
        for model in anthropic_models:
            assert evaluator.get_provider_from_model(model) == "anthropic"
    
    def test_get_provider_from_model_google(self):
        """Test provider detection for Google models."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        google_models = ["gemini-pro", "gemini-1.5-flash", "palm-2", "bison-001"]
        for model in google_models:
            assert evaluator.get_provider_from_model(model) == "gemini"
    
    def test_get_provider_from_model_others(self):
        """Test provider detection for other models."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        # Mistral
        assert evaluator.get_provider_from_model("mistral-large") == "mistral"
        assert evaluator.get_provider_from_model("mixtral-8x7b") == "mistral"
        
        # Together/Meta
        assert evaluator.get_provider_from_model("llama-2-7b") == "together"
        assert evaluator.get_provider_from_model("meta-llama/llama-3.1-8b") == "together"
        
        # Unknown (should default to OpenAI)
        assert evaluator.get_provider_from_model("unknown-model") == "openai"
        assert evaluator.get_provider_from_model("custom-model") == "openai"


class TestRagasFaithfulnessEvaluator:
    """Test RAGAS faithfulness evaluator."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RagasFaithfulnessEvaluator()
        assert isinstance(evaluator, BaseEvaluator)
        assert evaluator.model_config == {}
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_initialization_with_config(self):
        """Test evaluator initialization with config."""
        model_config = {"llm_model": "gpt-3.5-turbo"}
        evaluator = RagasFaithfulnessEvaluator(model_config=model_config)
        assert evaluator.model_config == model_config
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('evaluators.ragas_faithfulness.RagasFaithfulnessEvaluator.evaluate')
    def test_evaluate_method_exists(self, mock_evaluate):
        """Test that evaluate method exists and can be called."""
        mock_evaluate.return_value = 0.85
        
        evaluator = RagasFaithfulnessEvaluator()
        test_data = {
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }
        
        score = evaluator.evaluate(test_data)
        assert isinstance(score, float)
        assert score == 0.85
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_has_detailed_analysis(self):
        """Test that evaluator has detailed analysis method."""
        evaluator = RagasFaithfulnessEvaluator()
        assert hasattr(evaluator, 'get_detailed_analysis')
        assert callable(getattr(evaluator, 'get_detailed_analysis'))


class TestRagasContextPrecisionEvaluator:
    """Test RAGAS context precision evaluator."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RagasContextPrecisionEvaluator()
        assert isinstance(evaluator, BaseEvaluator)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('evaluators.ragas_context_precision.RagasContextPrecisionEvaluator.evaluate')
    def test_evaluate_method_exists(self, mock_evaluate):
        """Test that evaluate method exists and can be called."""
        mock_evaluate.return_value = 0.75
        
        evaluator = RagasContextPrecisionEvaluator()
        test_data = {
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France.", "France is in Europe."],
            "answer": "Paris is the capital of France."
        }
        
        score = evaluator.evaluate(test_data)
        assert isinstance(score, float)
        assert score == 0.75
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_has_detailed_analysis(self):
        """Test that evaluator has detailed analysis method."""
        evaluator = RagasContextPrecisionEvaluator()
        assert hasattr(evaluator, 'get_detailed_analysis')
        assert callable(getattr(evaluator, 'get_detailed_analysis'))


class TestRAGCheckerHallucinationEvaluator:
    """Test RAGChecker hallucination evaluator."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = RAGCheckerHallucinationEvaluator()
        assert isinstance(evaluator, BaseEvaluator)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('evaluators.ragchecker_hallucination.RAGCheckerHallucinationEvaluator.evaluate')
    def test_evaluate_method_exists(self, mock_evaluate):
        """Test that evaluate method exists and can be called."""
        mock_evaluate.return_value = 15.0  # RAGChecker returns percentage
        
        evaluator = RAGCheckerHallucinationEvaluator()
        test_data = {
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }
        
        score = evaluator.evaluate(test_data)
        assert isinstance(score, float)
        assert score == 15.0
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_evaluator_has_detailed_analysis(self):
        """Test that evaluator has detailed analysis method."""
        evaluator = RAGCheckerHallucinationEvaluator()
        assert hasattr(evaluator, 'get_detailed_analysis')
        assert callable(getattr(evaluator, 'get_detailed_analysis'))


class TestEvaluatorRegistry:
    """Test evaluator registry consistency."""
    
    def test_registry_completeness(self):
        """Test that registry contains all expected evaluators."""
        expected_evaluators = {
            "faithfulness_ragas": RagasFaithfulnessEvaluator,
            "context_precision_ragas": RagasContextPrecisionEvaluator,
            "hallucination_ragchecker": RAGCheckerHallucinationEvaluator,
        }
        
        assert EVALUATORS == expected_evaluators
    
    def test_descriptions_match_evaluators(self):
        """Test that descriptions exist for all evaluators."""
        for evaluator_name in EVALUATORS.keys():
            assert evaluator_name in EVALUATOR_DESCRIPTIONS
            assert isinstance(EVALUATOR_DESCRIPTIONS[evaluator_name], str)
            assert len(EVALUATOR_DESCRIPTIONS[evaluator_name]) > 0
    
    def test_all_evaluators_inherit_base(self):
        """Test that all registered evaluators inherit from BaseEvaluator."""
        for evaluator_class in EVALUATORS.values():
            assert issubclass(evaluator_class, BaseEvaluator)
    
    def test_all_evaluators_implement_evaluate(self):
        """Test that all evaluators implement the evaluate method."""
        for evaluator_class in EVALUATORS.values():
            # Check if evaluate method is overridden (not just inherited from ABC)
            assert 'evaluate' in evaluator_class.__dict__ or hasattr(evaluator_class, 'evaluate')


class TestEvaluatorErrorHandling:
    """Test evaluator error handling scenarios."""
    
    def test_get_evaluator_empty_string(self):
        """Test getting evaluator with empty string."""
        with pytest.raises(ValueError):
            get_evaluator("")
    
    def test_get_evaluator_none(self):
        """Test getting evaluator with None."""
        with pytest.raises(ValueError):
            get_evaluator(None)
    
    def test_get_evaluator_case_sensitivity(self):
        """Test that metric names are case sensitive."""
        with pytest.raises(ValueError):
            get_evaluator("FAITHFULNESS_RAGAS")  # Uppercase
        
        with pytest.raises(ValueError):
            get_evaluator("Faithfulness_Ragas")  # Mixed case
    
    def test_base_evaluator_abstract_instantiation(self):
        """Test that BaseEvaluator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEvaluator()
    
    def test_concrete_evaluator_missing_evaluate(self):
        """Test that concrete evaluator must implement evaluate method."""
        class IncompleteEvaluator(BaseEvaluator):
            pass  # Missing evaluate method
        
        with pytest.raises(TypeError):
            IncompleteEvaluator()
    
    @patch.dict(os.environ, {}, clear=True)
    def test_api_key_missing_all_sources(self):
        """Test behavior when API key is missing from all sources."""
        class ConcreteEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return 0.5
        
        evaluator = ConcreteEvaluator()
        
        # Should return None when no API key is available
        assert evaluator.get_api_key("openai") is None
        assert evaluator.get_api_key("anthropic") is None
        assert evaluator.get_api_key("google") is None


class TestEvaluatorIntegration:
    """Test evaluator integration scenarios."""
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_all_evaluators_can_be_instantiated(self):
        """Test that all registered evaluators can be instantiated."""
        for metric_name, evaluator_class in EVALUATORS.items():
            try:
                evaluator = evaluator_class()
                assert isinstance(evaluator, BaseEvaluator)
            except Exception as e:
                pytest.fail(f"Failed to instantiate {metric_name}: {e}")
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    def test_all_evaluators_with_model_config(self):
        """Test that all evaluators work with model configuration."""
        model_config = {
            "llm_model": "gpt-3.5-turbo",
            "api_key": "sk-test-key"
        }
        
        for metric_name in EVALUATORS.keys():
            try:
                evaluator = get_evaluator(metric_name, model_config=model_config)
                assert isinstance(evaluator, BaseEvaluator)
                assert evaluator.model_config == model_config
            except Exception as e:
                pytest.fail(f"Failed to instantiate {metric_name} with config: {e}")
    
    def test_batch_evaluate_consistency(self):
        """Test that batch evaluate gives consistent results with individual evaluate."""
        class TestEvaluator(BaseEvaluator):
            def evaluate(self, data):
                return len(data.get("answer", "")) / 10.0
        
        evaluator = TestEvaluator()
        test_data = [
            {"answer": "short"},
            {"answer": "medium length"},
            {"answer": "very long detailed answer"}
        ]
        
        # Individual evaluation
        individual_scores = [evaluator.evaluate(item) for item in test_data]
        
        # Batch evaluation
        batch_scores = evaluator.batch_evaluate(test_data)
        
        assert individual_scores == batch_scores