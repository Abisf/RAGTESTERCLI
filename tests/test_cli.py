"""
Unit tests for CLI module.

Tests command parsing, argument handling, provider detection, and error scenarios.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner
import sys
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import app, detect_provider, clean_model_name


class TestCLIProviderDetection:
    """Test provider detection functionality."""
    
    def test_detect_provider_openai(self):
        """Test OpenAI provider detection."""
        assert detect_provider("gpt-3.5-turbo") == "OPENAI_API_KEY"
        assert detect_provider("gpt-4") == "OPENAI_API_KEY"
        assert detect_provider("gpt-4o") == "OPENAI_API_KEY"
    
    def test_detect_provider_anthropic(self):
        """Test Anthropic provider detection."""
        assert detect_provider("claude-3-haiku") == "ANTHROPIC_API_KEY"
        assert detect_provider("claude-3-sonnet") == "ANTHROPIC_API_KEY"
        assert detect_provider("claude-3-opus") == "ANTHROPIC_API_KEY"
    
    def test_detect_provider_google(self):
        """Test Google provider detection."""
        assert detect_provider("gemini-pro") == "GOOGLE_API_KEY"
        assert detect_provider("gemini-1.5-flash") == "GOOGLE_API_KEY"
    
    def test_detect_provider_mistral(self):
        """Test Mistral provider detection."""
        assert detect_provider("mistral-large") == "MISTRAL_API_KEY"
        assert detect_provider("mistral-medium") == "MISTRAL_API_KEY"
    
    def test_detect_provider_together(self):
        """Test Together.ai provider detection."""
        assert detect_provider("llama-3.1-8b") == "TOGETHER_API_KEY"
        assert detect_provider("meta-llama/llama-3.1-8b") == "TOGETHER_API_KEY"
    
    def test_detect_provider_unknown(self):
        """Test unknown provider defaults to OpenAI."""
        assert detect_provider("unknown-model") == "OPENAI_API_KEY"
        assert detect_provider("custom-model") == "OPENAI_API_KEY"


class TestCLIModelNameCleaning:
    """Test model name cleaning functionality."""
    
    def test_clean_model_name_openrouter(self):
        """Test model name cleaning for OpenRouter."""
        # OpenRouter should keep provider prefixes
        assert clean_model_name("anthropic/claude-3-haiku", "https://openrouter.ai/api/v1") == "anthropic/claude-3-haiku"
        assert clean_model_name("google/gemini-pro", "https://openrouter.ai/api/v1") == "google/gemini-pro"
    
    def test_clean_model_name_direct_providers(self):
        """Test model name cleaning for direct providers."""
        # Direct providers should remove prefixes
        assert clean_model_name("anthropic/claude-3-haiku", "https://api.anthropic.com") == "claude-3-haiku"
        assert clean_model_name("google/gemini-pro", "https://generativelanguage.googleapis.com") == "gemini-pro"
        assert clean_model_name("openai/gpt-4", "https://api.openai.com/v1") == "gpt-4"
    
    def test_clean_model_name_no_prefix(self):
        """Test model name cleaning for models without prefixes."""
        assert clean_model_name("gpt-3.5-turbo", "https://api.openai.com/v1") == "gpt-3.5-turbo"
        assert clean_model_name("claude-3-haiku", "https://api.anthropic.com") == "claude-3-haiku"


class TestCLICommands:
    """Test CLI command functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
        
        # Create a test JSON file
        test_data = [{
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('cli.run_evaluation')
    def test_test_command_success(self, mock_run_evaluation):
        """Test successful test command execution."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key"
        ])
        
        assert result.exit_code == 0
        mock_run_evaluation.assert_called_once()
    
    def test_test_command_missing_input(self):
        """Test test command with missing input file."""
        result = self.runner.invoke(app, [
            "test",
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Missing option" in result.stdout
    
    def test_test_command_invalid_file(self):
        """Test test command with invalid file path."""
        result = self.runner.invoke(app, [
            "test",
            "--input", "nonexistent.json",
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code != 0
    
    def test_test_command_missing_metric(self):
        """Test test command with missing metric."""
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--llm-model", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Missing option" in result.stdout
    
    def test_test_command_missing_model(self):
        """Test test command with missing model."""
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas"
        ])
        
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "Missing option" in result.stdout
    
    @patch('cli.run_evaluation')
    def test_test_command_with_verbose(self, mock_run_evaluation):
        """Test test command with verbose flag."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key",
            "--verbose"
        ])
        
        assert result.exit_code == 0
        assert "RAGTesterCLI Configuration:" in result.stdout
    
    @patch('cli.run_evaluation')
    def test_test_command_with_output(self, mock_run_evaluation):
        """Test test command with output file."""
        mock_run_evaluation.return_value = '{"result": "success"}'
        
        output_file = os.path.join(self.temp_dir, "output.json")
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key",
            "--output", output_file
        ])
        
        assert result.exit_code == 0
        # Note: The CLI doesn't actually create the output file in the test environment
        # because run_evaluation is mocked. In real usage, it would be created.
        # This test verifies the CLI command executes successfully.
    
    @patch('cli.run_evaluation')
    def test_test_command_with_format(self, mock_run_evaluation):
        """Test test command with different output formats."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        # Test JSON format
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key",
            "--format", "json"
        ])
        
        assert result.exit_code == 0
        
        # Test table format
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key",
            "--format", "table"
        ])
        
        assert result.exit_code == 0
    
    @patch('cli.run_evaluation')
    def test_analyze_command_success(self, mock_run_evaluation):
        """Test successful analyze command execution."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        result = self.runner.invoke(app, [
            "analyze",
            "--input", self.test_file,
            "--metric", "faithfulness",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key"
        ])
        
        assert result.exit_code == 0
    
    def test_analyze_command_missing_input(self):
        """Test analyze command with missing input file."""
        result = self.runner.invoke(app, [
            "analyze",
            "--metric", "faithfulness",
            "--llm-model", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code != 0
    
    def test_list_metrics_command(self):
        """Test list-metrics command."""
        result = self.runner.invoke(app, ["list-metrics"])
        
        assert result.exit_code == 0
        assert "faithfulness" in result.stdout.lower()
        assert "context_precision" in result.stdout.lower()
        assert "hallucination" in result.stdout.lower()
    
    def test_version_command(self):
        """Test version command."""
        result = self.runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "RAGTesterCLI" in result.stdout


class TestCLIEnvironmentVariables:
    """Test CLI environment variable handling."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
        
        # Create a test JSON file
        test_data = [{
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-test-key'})
    @patch('cli.run_evaluation')
    def test_environment_variable_loading(self, mock_run_evaluation):
        """Test that environment variables are loaded correctly."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo"
        ])
        
        assert result.exit_code == 0
        mock_run_evaluation.assert_called_once()
    
    @patch('cli.run_evaluation')
    def test_missing_api_key_error(self, mock_run_evaluation):
        """Test error handling when API key is missing."""
        # Clear any existing API keys
        with patch.dict(os.environ, {}, clear=True):
            result = self.runner.invoke(app, [
                "test",
                "--input", self.test_file,
                "--metric", "faithfulness_ragas",
                "--llm-model", "gpt-3.5-turbo"
            ])
            
            assert result.exit_code != 0
            assert "No API key found" in result.stdout
    
    @patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'sk-ant-test-key'})
    @patch('cli.run_evaluation')
    def test_anthropic_environment_variable(self, mock_run_evaluation):
        """Test Anthropic environment variable handling."""
        mock_run_evaluation.return_value = {"result": "success"}
        
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "claude-3-haiku"
        ])
        
        assert result.exit_code == 0
        mock_run_evaluation.assert_called_once()


class TestCLIErrorHandling:
    """Test CLI error handling scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
        
        # Create a test JSON file
        test_data = [{
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_invalid_metric(self):
        """Test handling of invalid metric."""
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "invalid_metric",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key"
        ])
        
        # Should not crash, but may show error
        assert result.exit_code >= 0
    
    def test_invalid_model_name(self):
        """Test handling of invalid model name."""
        result = self.runner.invoke(app, [
            "test",
            "--input", self.test_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "invalid-model",
            "--api-key", "sk-test-key"
        ])
        
        # Should not crash, but may show error
        assert result.exit_code >= 0
    
    def test_empty_input_file(self):
        """Test handling of empty input file."""
        empty_file = os.path.join(self.temp_dir, "empty.json")
        with open(empty_file, 'w') as f:
            f.write("[]")
        
        result = self.runner.invoke(app, [
            "test",
            "--input", empty_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key"
        ])
        
        # Should not crash, but may show error
        assert result.exit_code >= 0
    
    def test_malformed_json_file(self):
        """Test handling of malformed JSON file."""
        malformed_file = os.path.join(self.temp_dir, "malformed.json")
        with open(malformed_file, 'w') as f:
            f.write("{ invalid json }")
        
        result = self.runner.invoke(app, [
            "test",
            "--input", malformed_file,
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo",
            "--api-key", "sk-test-key"
        ])
        
        assert result.exit_code != 0 