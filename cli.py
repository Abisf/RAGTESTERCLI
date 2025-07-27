#!/usr/bin/env python3
"""
🎯 RAGTesterCLI - Unified RAG Evaluation Tool
"""

import os
import typer
from typing import Optional
from runner import run_evaluation

app = typer.Typer()

# Provider detection mapping
PROVIDER_MAP = {
    "gpt-": "OPENAI",
    "claude-": "ANTHROPIC",
    "gemini": "GOOGLE",
    "mistral": "MISTRAL",
    "llama": "TOGETHER",
    "meta-llama": "TOGETHER",
}

def clean_model_name(model_name: str, api_base: Optional[str] = None) -> str:
    """Clean model name by removing provider prefixes, except for OpenRouter which needs them."""
    # For OpenRouter, keep provider prefixes as they're part of the model ID
    if api_base and "openrouter.ai" in api_base:
        return model_name  # Keep as-is for OpenRouter
    
    # Remove common provider prefixes for other APIs
    prefixes_to_remove = ["google/", "anthropic/", "openai/", "openrouter/"]
    
    for prefix in prefixes_to_remove:
        if model_name.startswith(prefix):
            return model_name[len(prefix):]
    
    return model_name

def detect_provider(model_name: str) -> str:
    """Detect provider from model name and return env var name."""
    for prefix, prov in PROVIDER_MAP.items():
        if model_name.lower().startswith(prefix):
            return prov + "_API_KEY"
    # Default to OpenAI for unknown models
    return "OPENAI_API_KEY"

@app.command()
def test(
    input: str = typer.Option(..., "--input", "-i", help="Path to input JSON file"),
    metric: str = typer.Option(..., "--metric", "-m", help="Evaluation metric to use"),
    llm_model: str = typer.Option(..., "--llm-model", help="Model name, e.g. gpt-4, claude-3-sonnet, gemini-pro"),
    api_key: str = typer.Option(..., "--api-key", help="API key for your chosen provider"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="(Optional) Custom API base URL"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (optional)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv, table"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    🎯 RAGTesterCLI - ONE API key, ANY provider, BOTH RAGAS and RAGChecker
    
    Examples:
      # OpenAI
      ragtester test --llm-model gpt-4 --api-key sk-...
      
      # OpenRouter (any model)
      ragtester test --llm-model google/gemini-pro --api-key sk-or-v1-... --api-base https://openrouter.ai/api/v1
      
      # Direct Anthropic
      ragtester test --llm-model claude-3-sonnet --api-key sk-ant-...
      
      # Direct Google
      ragtester test --llm-model gemini-pro --api-key AIzaSy... --api-base https://generativelanguage.googleapis.com/v1
    """
    
    # 1) Detect provider and set appropriate env var
    key_var = detect_provider(llm_model)
    os.environ[key_var] = api_key
    
    # 2) Mirror for OPENAI_API_KEY so RAGAS works
    os.environ["OPENAI_API_KEY"] = api_key
    
    # 3) Clean model name for actual API calls (remove provider prefixes)
    clean_model = clean_model_name(llm_model, api_base)
    os.environ["RAGCLI_LLM_MODEL"] = clean_model
    
    # 4) Set custom API base if provided
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
        
        # Special handling for OpenRouter - set LiteLLM-specific vars
        if "openrouter.ai" in api_base and api_key.startswith("sk-or-v1-"):
            os.environ["LITELLM_API_KEY"] = api_key
            os.environ["LITELLM_API_BASE"] = api_base
            os.environ["OPENROUTER_API_KEY"] = api_key  # Some versions use this
            print(f"✅ Set LiteLLM vars for OpenRouter")
        else:
            os.environ["OPENAI_API_BASE"] = api_base
        
    if verbose:
        typer.echo(f"🎯 RAGTesterCLI Configuration:")
        typer.echo(f"  Original Model: {llm_model}")
        typer.echo(f"  Clean Model: {clean_model}")
        typer.echo(f"  Provider Key: {key_var}")
        typer.echo(f"  API Base: {api_base or 'default'}")
        typer.echo(f"  Metric: {metric}")
        typer.echo()

    # 5) Run evaluation with unified config
    model_config = {
        "llm_model": llm_model,
        "api_key": api_key,
        "api_base": api_base
    }
    
    result = run_evaluation(
        input_file=input,
        metric=metric,
        output_file=output,
        output_format=format,
        verbose=verbose,
        model_config=model_config
    )
    
    if result:
        typer.echo(result)

@app.command()
def list_metrics():
    """List available evaluation metrics."""
    from evaluators import get_available_metrics
    
    typer.echo("📊 Available Metrics:")
    metrics = get_available_metrics()
    for name, description in metrics.items():
        typer.echo(f"  • {name}: {description}")

@app.command()  
def version():
    """Show version information."""
    typer.echo("🎯 RAGTesterCLI v0.1 - Universal RAG Evaluation")
    typer.echo("   ✅ Multi-provider LLM support (OpenAI, Anthropic, Google, OpenRouter)")
    typer.echo("   ✅ True RAGAS & RAGChecker integration")
    typer.echo("   ✅ One API key for any provider")

if __name__ == "__main__":
    app() 