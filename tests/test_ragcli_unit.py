# tests/test_ragcli_unit.py
import json
import subprocess
import sys
import pytest
from pathlib import Path

CLI = [sys.executable, "cli.py"]  # adjust if your entrypoint differs

EXAMPLES = Path(__file__).parent.parent / "examples"

def run_cli(args):
    """Helper to run the CLI and return (exit_code, stdout, stderr)."""
    proc = subprocess.run(CLI + args, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

@pytest.mark.parametrize("metric,input_file,expected_score", [
    ("faithfulness_ragas", "multi_faithfulness_test.json", None),  # score array check below
    ("context_precision_ragas", "louvre_context_precision_test.json", None),
])
def test_test_mode_runs(metric, input_file, expected_score):
    code, out, err = run_cli(["test", "--input", str(EXAMPLES/input_file),
                               "--metric", metric, "--llm-model", "gpt-3.5-turbo"])
    assert code == 0, f"CLI exited non-zero:\n{err}"
    data = json.loads(out)
    assert "metric" in data
    assert data["metric"] == metric
    assert "scores" in data
    # Validate score structure
    assert isinstance(data["scores"], list)
    assert len(data["scores"]) > 0
    for score in data["scores"]:
        assert isinstance(score, (int, float))
        assert 0.0 <= score <= 1.0

def test_analyze_strips_numbering_and_phantom():
    code, out, err = run_cli([
        "analyze",
        "--input", str(EXAMPLES/"quick_test.json"),
        "--metric", "faithfulness",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code == 0, f"Analyze failed:\n{err}"
    lines = out.splitlines()
    # find claim lines
    claims = [ln for ln in lines if ln.strip().startswith("Claim ")]
    # none should contain a phantom intro or leading numbering
    for cl in claims:
        text = cl.split(":", 1)[1].strip()
        # disallow residual "Here are" or leading digits+ punctuation
        assert not text.startswith(("Here are", "1.", "2.", "(1)", "-", "•")), \
            f"Bad claim formatting: {cl}"

def test_hallucination_basic():
    # Test hallucination detection with known hallucinated data
    code, out, err = run_cli([
        "test", "--input", str(EXAMPLES/"hallucinated_test.json"),
        "--metric", "hallucination_ragchecker",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code == 0, f"Hallucination test failed:\n{err}"
    data = json.loads(out)
    assert "metric" in data
    assert data["metric"] == "hallucination_ragchecker"
    assert "scores" in data
    # Should detect high hallucination in deliberately false data
    assert any(score > 50.0 for score in data["scores"]), "Should detect hallucination in test data"

def test_context_precision_analysis():
    # Test context precision detailed analysis
    code, out, err = run_cli([
        "analyze",
        "--input", str(EXAMPLES/"louvre_context_precision_test.json"),
        "--metric", "context_precision",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code == 0, f"Context precision analyze failed:\n{err}"
    lines = out.splitlines()
    # Should contain precision calculation details
    precision_lines = [ln for ln in lines if "Precision@" in ln]
    assert len(precision_lines) > 0, "Should show Precision@k calculations"
    
    # Should show chunk relevance analysis
    relevance_lines = [ln for ln in lines if "RELEVANT" in ln or "NOT RELEVANT" in ln]
    assert len(relevance_lines) > 0, "Should show chunk relevance analysis"

def test_error_on_missing_api_key():
    # Test that CLI errors cleanly when API key is missing
    import os
    # Temporarily remove API key env vars
    old_keys = {}
    key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]
    for var in key_vars:
        if var in os.environ:
            old_keys[var] = os.environ.pop(var)
    
    try:
        code, out, err = run_cli([
            "test", "--input", str(EXAMPLES/"quick_test.json"),
            "--metric", "faithfulness_ragas",
            "--llm-model", "gpt-3.5-turbo"
        ])
        assert code != 0, "Should fail when API key is missing"
        assert "API key not found" in err or "API key not found" in out
    finally:
        # Restore API keys
        for var, val in old_keys.items():
            os.environ[var] = val

def test_list_metrics():
    # Test that list-metrics command works
    code, out, err = run_cli(["list-metrics"])
    assert code == 0, f"List metrics failed:\n{err}"
    
    # Should list all three metrics
    assert "faithfulness_ragas" in out
    assert "context_precision_ragas" in out  
    assert "hallucination_ragchecker" in out

def test_version_command():
    # Test version command
    code, out, err = run_cli(["version"])
    assert code == 0, f"Version command failed:\n{err}"
    assert "RAGTesterCLI" in out

def test_invalid_metric():
    # Test error handling for invalid metric
    code, out, err = run_cli([
        "test", "--input", str(EXAMPLES/"quick_test.json"),
        "--metric", "invalid_metric",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code != 0, "Should fail with invalid metric"
    
def test_invalid_input_file():
    # Test error handling for missing input file
    code, out, err = run_cli([
        "test", "--input", "nonexistent.json",
        "--metric", "faithfulness_ragas", 
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code != 0, "Should fail with missing input file"

def test_analyze_metric_validation():
    # Test that analyze command validates metric types
    code, out, err = run_cli([
        "analyze",
        "--input", str(EXAMPLES/"quick_test.json"),
        "--metric", "invalid_analyze_metric",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code != 0, "Should fail with invalid analyze metric"
    assert "faithfulness, context_precision, hallucination" in err or "faithfulness, context_precision, hallucination" in out

def test_fallback_indicators():
    # Test that fallback vs official library indicators work
    # This would need a way to force fallback mode, but we can at least check output format
    code, out, err = run_cli([
        "test", "--input", str(EXAMPLES/"quick_test.json"),
        "--metric", "faithfulness_ragas",
        "--llm-model", "gpt-3.5-turbo"
    ])
    assert code == 0, f"Test failed:\n{err}"
    # Should contain either success or fallback indicator
    assert "✓ Using official" in err or "→ Using fallback" in err

if __name__ == "__main__":
    # Run tests directly if called as script
    pytest.main([__file__, "-v"]) 