"""
Unit tests for runner module.

Tests input loading, evaluation running, output formatting, and error handling.
"""

import pytest
import os
import tempfile
import json
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import load_input_data, run_evaluation, format_output, save_output


class TestLoadInputData:
    """Test input data loading functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_load_valid_json_list(self):
        """Test loading valid JSON list."""
        test_data = [
            {
                "question": "What is the capital of France?",
                "context": ["Paris is the capital of France."],
                "answer": "Paris is the capital of France."
            },
            {
                "question": "What is 2+2?",
                "context": ["Basic arithmetic."],
                "answer": "4"
            }
        ]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = load_input_data(self.test_file)
        
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["question"] == "What is the capital of France?"
        assert result[1]["answer"] == "4"
    
    def test_load_valid_json_dict(self):
        """Test loading valid JSON dict (converted to list)."""
        test_data = {
            "question": "What is the capital of France?",
            "context": ["Paris is the capital of France."],
            "answer": "Paris is the capital of France."
        }
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = load_input_data(self.test_file)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["question"] == "What is the capital of France?"
    
    def test_load_empty_json_list(self):
        """Test loading empty JSON list."""
        test_data = []
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        result = load_input_data(self.test_file)
        
        assert isinstance(result, list)
        assert len(result) == 0
    
    def test_load_missing_file(self):
        """Test loading missing file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            load_input_data("nonexistent.json")
        
        assert "Input file not found" in str(exc_info.value)
    
    def test_load_invalid_json(self):
        """Test loading invalid JSON."""
        with open(self.test_file, 'w') as f:
            f.write("{ invalid json }")
        
        with pytest.raises(ValueError) as exc_info:
            load_input_data(self.test_file)
        
        assert "Invalid JSON in input file" in str(exc_info.value)
    
    def test_load_empty_file(self):
        """Test loading empty file."""
        with open(self.test_file, 'w') as f:
            f.write("")
        
        with pytest.raises(ValueError) as exc_info:
            load_input_data(self.test_file)
        
        assert "Invalid JSON in input file" in str(exc_info.value)


class TestRunEvaluation:
    """Test evaluation running functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
        self.output_file = os.path.join(self.temp_dir, "output.json")
        
        # Create test data
        test_data = [
            {
                "question": "What is the capital of France?",
                "context": ["Paris is the capital of France."],
                "answer": "Paris is the capital of France."
            }
        ]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_success(self, mock_get_evaluator):
        """Test successful evaluation run."""
        # Mock evaluator
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = 0.85
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="faithfulness_ragas",
            verbose=False
        )
        
        # Verify result is JSON string
        assert isinstance(result, str)
        result_data = json.loads(result)
        
        assert result_data["metric"] == "faithfulness_ragas"
        assert result_data["summary"]["total_items"] == 1
        assert result_data["summary"]["average_score"] == 0.85
        assert result_data["scores"] == [0.85]
        
        mock_get_evaluator.assert_called_once_with("faithfulness_ragas", model_config=None)
        mock_evaluator.evaluate.assert_called_once()
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_with_model_config(self, mock_get_evaluator):
        """Test evaluation run with model configuration."""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = 0.75
        mock_get_evaluator.return_value = mock_evaluator
        
        model_config = {
            "llm_model": "gpt-4",
            "api_key": "sk-test-key",
            "api_base": "https://api.openai.com/v1"
        }
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="context_precision_ragas",
            model_config=model_config
        )
        
        result_data = json.loads(result)
        assert result_data["metric"] == "context_precision_ragas"
        assert result_data["scores"] == [0.75]
        
        mock_get_evaluator.assert_called_once_with("context_precision_ragas", model_config=model_config)
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_with_output_file(self, mock_get_evaluator):
        """Test evaluation run with output file."""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = 0.90
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="hallucination_ragchecker",
            output_file=self.output_file,
            verbose=True
        )
        
        # Verify output file was created
        assert os.path.exists(self.output_file)
        
        # Verify file content matches result
        with open(self.output_file, 'r') as f:
            file_content = f.read()
        
        assert file_content == result
        
        result_data = json.loads(result)
        assert result_data["scores"] == [0.90]
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_error_handling(self, mock_get_evaluator):
        """Test evaluation run with evaluation errors."""
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = Exception("Evaluation failed")
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="faithfulness_ragas",
            verbose=True
        )
        
        result_data = json.loads(result)
        
        # Should default to 0.0 on error
        assert result_data["scores"] == [0.0]
        assert result_data["summary"]["average_score"] == 0.0
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_multiple_items(self, mock_get_evaluator):
        """Test evaluation run with multiple items."""
        # Create test data with multiple items
        test_data = [
            {"question": "Q1", "context": ["C1"], "answer": "A1"},
            {"question": "Q2", "context": ["C2"], "answer": "A2"},
            {"question": "Q3", "context": ["C3"], "answer": "A3"}
        ]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = [0.8, 0.6, 0.9]
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="context_precision_ragas"
        )
        
        result_data = json.loads(result)
        
        assert result_data["summary"]["total_items"] == 3
        assert result_data["scores"] == [0.8, 0.6, 0.9]
        assert result_data["summary"]["average_score"] == 0.767  # (0.8 + 0.6 + 0.9) / 3
        assert result_data["summary"]["min_score"] == 0.6
        assert result_data["summary"]["max_score"] == 0.9
        
        assert mock_evaluator.evaluate.call_count == 3
    
    @patch('runner.get_evaluator')
    def test_run_evaluation_empty_data(self, mock_get_evaluator):
        """Test evaluation run with empty data."""
        # Create empty test data
        test_data = []
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        mock_evaluator = MagicMock()
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="faithfulness_ragas"
        )
        
        result_data = json.loads(result)
        
        assert result_data["summary"]["total_items"] == 0
        assert result_data["scores"] == []
        assert result_data["summary"]["average_score"] == 0.0
        assert result_data["summary"]["min_score"] == 0.0
        assert result_data["summary"]["max_score"] == 0.0
        
        mock_evaluator.evaluate.assert_not_called()


class TestFormatOutput:
    """Test output formatting functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_result = {
            "metric": "faithfulness_ragas",
            "summary": {
                "total_items": 2,
                "average_score": 0.75,
                "min_score": 0.5,
                "max_score": 1.0,
                "execution_time_seconds": 5.2
            },
            "scores": [0.5, 1.0]
        }
    
    def test_format_output_json(self):
        """Test JSON output formatting."""
        result = format_output(self.test_result, "json")
        
        assert isinstance(result, str)
        parsed = json.loads(result)
        assert parsed == self.test_result
        
        # Verify it's pretty-printed (indented)
        assert "\n" in result
        assert "  " in result
    
    def test_format_output_csv(self):
        """Test CSV output formatting."""
        result = format_output(self.test_result, "csv")
        
        assert isinstance(result, str)
        lines = result.strip().split('\n')
        
        # Should have header + 2 data rows
        assert len(lines) == 3
        assert lines[0] == "item_index,score"
        assert lines[1] == "0,0.5"
        assert lines[2] == "1,1.0"
    
    def test_format_output_table(self):
        """Test table output formatting."""
        result = format_output(self.test_result, "table")
        
        assert isinstance(result, str)
        assert "Metric" in result
        assert "faithfulness_ragas" in result
        assert "Total Items" in result
        assert "2" in result
        assert "Average Score" in result
        assert "0.75" in result
        assert "Individual Scores:" in result
    
    def test_format_output_invalid_format(self):
        """Test invalid output format."""
        with pytest.raises(ValueError) as exc_info:
            format_output(self.test_result, "invalid_format")
        
        assert "Unsupported output format" in str(exc_info.value)
    
    def test_format_output_empty_scores(self):
        """Test formatting with empty scores."""
        empty_result = {
            "metric": "test_metric",
            "summary": {
                "total_items": 0,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "execution_time_seconds": 0.1
            },
            "scores": []
        }
        
        # Test JSON format
        json_result = format_output(empty_result, "json")
        parsed = json.loads(json_result)
        assert parsed["scores"] == []
        
        # Test CSV format
        csv_result = format_output(empty_result, "csv")
        lines = csv_result.strip().split('\n')
        assert len(lines) == 1  # Only header
        assert lines[0] == "item_index,score"
        
        # Test table format
        table_result = format_output(empty_result, "table")
        assert "Total Items" in table_result
        assert "0" in table_result


class TestSaveOutput:
    """Test output saving functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.output_file = os.path.join(self.temp_dir, "output.txt")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_save_output_json(self):
        """Test saving JSON output."""
        test_output = '{"test": "data"}'
        
        save_output(test_output, self.output_file, "json")
        
        assert os.path.exists(self.output_file)
        
        with open(self.output_file, 'r') as f:
            content = f.read()
        
        assert content == test_output
    
    def test_save_output_csv(self):
        """Test saving CSV output."""
        test_output = "item_index,score\n0,0.5\n1,1.0"
        
        save_output(test_output, self.output_file, "csv")
        
        assert os.path.exists(self.output_file)
        
        with open(self.output_file, 'r') as f:
            content = f.read()
        
        assert content == test_output
    
    def test_save_output_table(self):
        """Test saving table output."""
        test_output = "Summary Table\n============\nMetric: test"
        
        save_output(test_output, self.output_file, "table")
        
        assert os.path.exists(self.output_file)
        
        with open(self.output_file, 'r') as f:
            content = f.read()
        
        assert content == test_output
    
    def test_save_output_overwrite(self):
        """Test overwriting existing output file."""
        # Create initial file
        with open(self.output_file, 'w') as f:
            f.write("initial content")
        
        # Overwrite with new content
        new_output = "new content"
        save_output(new_output, self.output_file, "json")
        
        with open(self.output_file, 'r') as f:
            content = f.read()
        
        assert content == new_output
    
    def test_save_output_directory_creation(self):
        """Test saving output when directory doesn't exist."""
        nested_dir = os.path.join(self.temp_dir, "nested", "deep")
        nested_file = os.path.join(nested_dir, "output.json")
        
        # This should fail if directory doesn't exist
        with pytest.raises(FileNotFoundError):
            save_output('{"test": "data"}', nested_file, "json")


class TestRunnerIntegration:
    """Test runner module integration scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test.json")
        self.output_file = os.path.join(self.temp_dir, "output.json")
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('runner.get_evaluator')
    def test_end_to_end_evaluation(self, mock_get_evaluator):
        """Test complete end-to-end evaluation flow."""
        # Create realistic test data
        test_data = [
            {
                "question": "What is the capital of France?",
                "context": [
                    "Paris is the capital and most populous city of France.",
                    "France is a country in Western Europe.",
                    "The population of Paris is about 2.2 million."
                ],
                "answer": "The capital of France is Paris.",
                "ground_truth": "Paris"
            }
        ]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Mock evaluator with realistic behavior
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.return_value = 0.92
        mock_get_evaluator.return_value = mock_evaluator
        
        # Run evaluation with all options
        result = run_evaluation(
            input_file=self.test_file,
            metric="faithfulness_ragas",
            output_file=self.output_file,
            output_format="json",
            verbose=True,
            model_config={
                "llm_model": "gpt-3.5-turbo",
                "api_key": "sk-test-key"
            }
        )
        
        # Verify result structure
        result_data = json.loads(result)
        assert result_data["metric"] == "faithfulness_ragas"
        assert result_data["summary"]["total_items"] == 1
        assert result_data["summary"]["average_score"] == 0.92
        assert result_data["scores"] == [0.92]
        assert "execution_time_seconds" in result_data["summary"]
        
        # Verify output file was created and matches
        assert os.path.exists(self.output_file)
        with open(self.output_file, 'r') as f:
            file_content = f.read()
        assert file_content == result
        
        # Verify evaluator was called correctly
        mock_get_evaluator.assert_called_once()
        mock_evaluator.evaluate.assert_called_once_with(test_data[0])
    
    @patch('runner.get_evaluator')
    def test_mixed_success_failure_evaluation(self, mock_get_evaluator):
        """Test evaluation with mix of successful and failed items."""
        test_data = [
            {"question": "Q1", "context": ["C1"], "answer": "A1"},
            {"question": "Q2", "context": ["C2"], "answer": "A2"},
            {"question": "Q3", "context": ["C3"], "answer": "A3"}
        ]
        
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f)
        
        # Mock evaluator with mixed results (success, failure, success)
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate.side_effect = [
            0.85,  # Success
            Exception("API error"),  # Failure
            0.75   # Success
        ]
        mock_get_evaluator.return_value = mock_evaluator
        
        result = run_evaluation(
            input_file=self.test_file,
            metric="context_precision_ragas",
            verbose=True
        )
        
        result_data = json.loads(result)
        
        # Should have 3 scores: 0.85, 0.0 (default on error), 0.75
        assert result_data["scores"] == [0.85, 0.0, 0.75]
        assert result_data["summary"]["total_items"] == 3
        assert result_data["summary"]["average_score"] == 0.533  # (0.85 + 0.0 + 0.75) / 3
        assert result_data["summary"]["min_score"] == 0.0
        assert result_data["summary"]["max_score"] == 0.85
        
        assert mock_evaluator.evaluate.call_count == 3