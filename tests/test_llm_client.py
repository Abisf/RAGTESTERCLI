"""
Unit tests for LLM client modules.

Tests OpenAI client wrapper, API integrations, error handling, and edge cases.
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm.openai_client import OpenAIClient, llm_client


class TestOpenAIClientInitialization:
    """Test OpenAI client initialization."""
    
    def test_client_initialization_default(self):
        """Test client initialization with default config."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
            client = OpenAIClient()
            assert client.api_key == 'sk-test-key'
            assert client.model == 'gpt-4'  # Default from config
            assert client.client is None  # Lazy initialization
    
    def test_client_initialization_with_params(self):
        """Test client initialization with explicit parameters."""
        client = OpenAIClient(api_key='sk-custom-key', model='gpt-3.5-turbo')
        assert client.api_key == 'sk-custom-key'
        assert client.model == 'gpt-3.5-turbo'
        assert client.client is None
    
    def test_client_initialization_missing_api_key(self):
        """Test client initialization with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIClient()
            assert client.api_key is None
    
    def test_ensure_client_success(self):
        """Test successful client initialization."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'}):
            client = OpenAIClient()
            client._ensure_client()
            assert client.client is not None
    
    def test_ensure_client_missing_api_key(self):
        """Test client initialization with missing API key."""
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIClient()
            with pytest.raises(ValueError) as exc_info:
                client._ensure_client()
            assert "OpenAI API key not found" in str(exc_info.value)


class TestOpenAIClientGenerateCompletion:
    """Test completion generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_success(self, mock_openai_class):
        """Test successful completion generation."""
        # Mock OpenAI client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a test response."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion("Test prompt")
        
        assert result == "This is a test response."
        mock_client.chat.completions.create.assert_called_once()
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_with_system_prompt(self, mock_openai_class):
        """Test completion generation with system prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "System-guided response."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion(
            "User prompt",
            system_prompt="You are a helpful assistant."
        )
        
        assert result == "System-guided response."
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2
        assert messages[0]['role'] == 'system'
        assert messages[0]['content'] == "You are a helpful assistant."
        assert messages[1]['role'] == 'user'
        assert messages[1]['content'] == "User prompt"
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_with_parameters(self, mock_openai_class):
        """Test completion generation with custom parameters."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Custom response."
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion(
            "Test prompt",
            temperature=0.5,
            max_tokens=500
        )
        
        assert result == "Custom response."
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['max_tokens'] == 500
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_api_error(self, mock_openai_class):
        """Test completion generation with API error."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API error")
        mock_openai_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            self.client.generate_completion("Test prompt")
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_empty_response(self, mock_openai_class):
        """Test completion generation with empty response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = ""
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion("Test prompt")
        
        assert result == ""
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_completion_whitespace_trimming(self, mock_openai_class):
        """Test that response whitespace is trimmed."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "  Response with whitespace  "
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion("Test prompt")
        
        assert result == "Response with whitespace"


class TestOpenAIClientGenerateScore:
    """Test score generation functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_success(self, mock_openai_class):
        """Test successful score generation."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 0.85, "reasoning": "Good response"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_score("Rate this response from 0 to 1")
        
        assert result == 0.85
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 1
        assert "JSON object" in messages[0]['content']
        assert "score" in messages[0]['content']
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_custom_range(self, mock_openai_class):
        """Test score generation with custom range."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 75, "reasoning": "Good"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_score("Rate from 0 to 100", score_range=(0, 100))
        
        assert result == 75.0
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_out_of_range_clamping(self, mock_openai_class):
        """Test that scores are clamped to the specified range."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 1.5, "reasoning": "Too high"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_score("Rate this", score_range=(0, 1))
        
        assert result == 1.0  # Clamped to max
    
        # Test lower bound clamping
        mock_response.choices[0].message.content = '{"score": -0.5, "reasoning": "Too low"}'
        result = self.client.generate_score("Rate this", score_range=(0, 1))
        assert result == 0.0  # Clamped to min
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_invalid_json_fallback(self, mock_openai_class):
        """Test score generation with invalid JSON fallback."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "The score is 0.75 based on the analysis"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_score("Rate this response")
        
        assert result == 0.75  # Extracted from text
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_no_numbers_fallback(self, mock_openai_class):
        """Test score generation with no numbers in response."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This response is good"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            self.client.generate_score("Rate this response")
        
        assert "Could not parse score" in str(exc_info.value)
    
    @patch('llm.openai_client.OpenAI')
    def test_generate_score_missing_score_key(self, mock_openai_class):
        """Test score generation with missing score key in JSON."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"reasoning": "Good response"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_score("Rate this response")
        
        assert result == 0.0  # Default value


class TestOpenAIClientErrorHandling:
    """Test error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
    
    @patch('llm.openai_client.OpenAI')
    def test_client_network_error(self, mock_openai_class):
        """Test handling of network errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = ConnectionError("Network error")
        mock_openai_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            self.client.generate_completion("Test prompt")
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @patch('llm.openai_client.OpenAI')
    def test_client_rate_limit_error(self, mock_openai_class):
        """Test handling of rate limit errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
        mock_openai_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            self.client.generate_completion("Test prompt")
        
        assert "OpenAI API error" in str(exc_info.value)
    
    @patch('llm.openai_client.OpenAI')
    def test_client_authentication_error(self, mock_openai_class):
        """Test handling of authentication errors."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("Invalid API key")
        mock_openai_class.return_value = mock_client
        
        with pytest.raises(RuntimeError) as exc_info:
            self.client.generate_completion("Test prompt")
        
        assert "OpenAI API error" in str(exc_info.value)
    
    def test_client_missing_api_key_error(self):
        """Test error when API key is missing."""
        # Create client with no API key and no environment variable
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIClient(api_key=None)
            
            with pytest.raises(ValueError) as exc_info:
                client._ensure_client()
            
            assert "OpenAI API key not found" in str(exc_info.value)


class TestOpenAIClientIntegration:
    """Test integration scenarios."""
    
    def test_global_client_instance(self):
        """Test that global client instance works."""
        # Test that global instance can be accessed
        assert llm_client is not None
        assert isinstance(llm_client, OpenAIClient)
    
    @patch('llm.openai_client.OpenAI')
    def test_end_to_end_completion_flow(self, mock_openai_class):
        """Test complete end-to-end completion flow."""
        # Mock successful API call
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Comprehensive test response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
        
        # Test completion generation
        result = client.generate_completion(
            "Generate a comprehensive test response",
            system_prompt="You are a helpful test assistant.",
            temperature=0.7,
            max_tokens=200
        )
        
        assert result == "Comprehensive test response"
        
        # Verify API call parameters
        call_args = mock_client.chat.completions.create.call_args
        assert call_args[1]['model'] == 'gpt-3.5-turbo'
        assert call_args[1]['temperature'] == 0.7
        assert call_args[1]['max_tokens'] == 200
    
    @patch('llm.openai_client.OpenAI')
    def test_end_to_end_score_generation_flow(self, mock_openai_class):
        """Test complete end-to-end score generation flow."""
        # Mock successful API call with JSON response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 0.92, "reasoning": "Excellent response"}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
        
        # Test score generation
        score = client.generate_score(
            "Rate the quality of this response from 0 to 1",
            system_prompt="You are a quality assessment expert.",
            score_range=(0, 1)
        )
        
        assert score == 0.92
        
        # Verify API call included score instructions
        call_args = mock_client.chat.completions.create.call_args
        messages = call_args[1]['messages']
        assert len(messages) == 2
        assert "JSON object" in messages[1]['content']
        assert "score" in messages[1]['content']


class TestOpenAIClientEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.client = OpenAIClient(api_key='sk-test-key', model='gpt-3.5-turbo')
    
    @patch('llm.openai_client.OpenAI')
    def test_empty_prompt(self, mock_openai_class):
        """Test handling of empty prompt."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response to empty prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion("")
        
        assert result == "Response to empty prompt"
    
    @patch('llm.openai_client.OpenAI')
    def test_very_long_prompt(self, mock_openai_class):
        """Test handling of very long prompt."""
        long_prompt = "A" * 10000  # Very long prompt
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response to long prompt"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion(long_prompt)
        
        assert result == "Response to long prompt"
    
    @patch('llm.openai_client.OpenAI')
    def test_special_characters_in_prompt(self, mock_openai_class):
        """Test handling of special characters in prompt."""
        special_prompt = "Test with special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response with special chars"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion(special_prompt)
        
        assert result == "Response with special chars"
    
    @patch('llm.openai_client.OpenAI')
    def test_unicode_characters_in_prompt(self, mock_openai_class):
        """Test handling of unicode characters in prompt."""
        unicode_prompt = "Test with unicode: éñüßαβγδεζηθικλμνξοπρστυφχψω"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Unicode response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion(unicode_prompt)
        
        assert result == "Unicode response"
    
    @patch('llm.openai_client.OpenAI')
    def test_score_range_edge_cases(self, mock_openai_class):
        """Test score range edge cases."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = '{"score": 1.0}'
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        # Test negative range - should not raise ValueError as the method doesn't validate this
        # The method will work with negative ranges
        result = self.client.generate_score("Test", score_range=(-1, 1))
        assert result == 1.0  # Mock returns 1.0, clamped to range
        
        # Test invalid range (min > max) - should not raise ValueError
        # The method will work with invalid ranges
        result = self.client.generate_score("Test", score_range=(1, 0))
        assert result == 1.0  # Mock returns 1.0, clamped to range
    
    @patch('llm.openai_client.OpenAI')
    def test_zero_tokens_request(self, mock_openai_class):
        """Test request with zero max_tokens."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Response"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client
        
        result = self.client.generate_completion("Test", max_tokens=0)
        
        assert result == "Response"
        call_args = mock_client.chat.completions.create.call_args
        # The client uses default max_tokens from config when 0 is passed
        assert call_args[1]['max_tokens'] == 1000  # Default from config


class TestOpenAIClientConfiguration:
    """Test configuration handling."""
    
    def test_config_integration(self):
        """Test integration with config module."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-config-key'}):
            client = OpenAIClient()
            assert client.api_key == 'sk-config-key'
    
    def test_model_override(self):
        """Test model override functionality."""
        client = OpenAIClient(model='gpt-3.5-turbo')
        assert client.model == 'gpt-3.5-turbo'
    
    def test_api_key_override(self):
        """Test API key override functionality."""
        client = OpenAIClient(api_key='sk-override-key')
        assert client.api_key == 'sk-override-key'
    
    def test_config_priority(self):
        """Test configuration priority (explicit > config > env)."""
        # Test that explicit parameters override config
        client = OpenAIClient(api_key='sk-explicit', model='gpt-3.5-turbo')
        assert client.api_key == 'sk-explicit'
        assert client.model == 'gpt-3.5-turbo' 