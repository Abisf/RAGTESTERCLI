#!/usr/bin/env python3
"""
RAGTesterCLI - Unified RAG Evaluation Tool
"""

import os
import typer
from typing import Optional
from runner import run_evaluation
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for your chosen provider (or use .env)"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="(Optional) Custom API base URL (or use .env)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output file path (optional)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, csv, table"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    RAGTesterCLI - Unified RAG evaluation with support for multiple providers
    
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
    
    # 1) Get API key from CLI or environment
    if api_key:
        # Use CLI-provided API key
        key_var = detect_provider(llm_model)
        os.environ[key_var] = api_key
        os.environ["OPENAI_API_KEY"] = api_key  # Mirror for RAGAS
    else:
        # Use environment variables from .env
        key_var = detect_provider(llm_model)
        env_api_key = os.getenv(key_var)
        if not env_api_key:
            typer.echo(f"Error: No API key found. Set {key_var} in .env file or use --api-key")
            raise typer.Exit(1)
        api_key = env_api_key
    
    # 3) Clean model name for actual API calls (remove provider prefixes)
    clean_model = clean_model_name(llm_model, api_base)
    os.environ["RAGCLI_LLM_MODEL"] = clean_model
    
    # 4) Set custom API base if provided
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    else:
        # Try to get from environment
        env_api_base = os.getenv("OPENAI_API_BASE")
        if env_api_base:
            api_base = env_api_base
    
    # Special handling for OpenRouter - set LiteLLM-specific vars
    if api_base and "openrouter.ai" in api_base and api_key.startswith("sk-or-v1-"):
        os.environ["LITELLM_API_KEY"] = api_key
        os.environ["LITELLM_API_BASE"] = api_base
        os.environ["OPENROUTER_API_KEY"] = api_key  # Some versions use this
        print(f"Set LiteLLM vars for OpenRouter")
        
    if verbose:
        typer.echo(f"RAGTesterCLI Configuration:")
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
def analyze(
    input: str = typer.Option(..., "--input", "-i", help="Path to input JSON file"),
    metric: str = typer.Option("faithfulness", "--metric", "-m", help="Analysis type: faithfulness, context_precision, or hallucination"),
    llm_model: str = typer.Option("gpt-3.5-turbo", "--llm-model", help="Model name for analysis"),
    api_key: Optional[str] = typer.Option(None, "--api-key", help="API key for your chosen provider (or use .env)"),
    api_base: Optional[str] = typer.Option(None, "--api-base", help="(Optional) Custom API base URL (or use .env)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
):
    """
    Analyze RAG metrics with detailed breakdowns.
    
    Faithfulness: Shows claim-by-claim analysis
    Context Precision: Shows chunk-by-chunk relevance analysis with Precision@k
    Hallucination: Shows detailed hallucination detection analysis
    """
    
    # Setup API keys (same logic as test command)
    if api_key:
        key_var = detect_provider(llm_model)
        os.environ[key_var] = api_key
        os.environ["OPENAI_API_KEY"] = api_key
    else:
        key_var = detect_provider(llm_model)
        env_api_key = os.getenv(key_var)
        if not env_api_key:
            typer.echo(f"Error: No API key found. Set {key_var} in .env file or use --api-key")
            raise typer.Exit(1)
        api_key = env_api_key
    
    clean_model = clean_model_name(llm_model, api_base)
    os.environ["RAGCLI_LLM_MODEL"] = clean_model
    
    if api_base:
        os.environ["OPENAI_API_BASE"] = api_base
    
    if api_base and "openrouter.ai" in api_base and api_key.startswith("sk-or-v1-"):
        os.environ["LITELLM_API_KEY"] = api_key
        os.environ["LITELLM_API_BASE"] = api_base
        os.environ["OPENROUTER_API_KEY"] = api_key
        print(f"Set LiteLLM vars for OpenRouter")
    
    # Load input data
    from runner import load_input_data
    data_list = load_input_data(input)
    
    # Initialize appropriate evaluator based on metric
    if metric == "faithfulness":
        from evaluators.ragas_faithfulness import RagasFaithfulnessEvaluator
        evaluator = RagasFaithfulnessEvaluator()
        typer.echo("RAGAS Faithfulness - Detailed Claim Analysis")
        typer.echo("=" * 50)
        
        for i, data in enumerate(data_list, 1):
            typer.echo(f"\nTest Case {i}:")
            typer.echo("-" * 30)
            
            analysis = evaluator.get_detailed_analysis(data)
            
            typer.echo(f"Question: {data.get('question', '')}")
            typer.echo(f"Context: {' '.join(data.get('context', []))}")
            typer.echo(f"Answer: {data.get('answer', '')}")
            typer.echo()
            
            if "error" in analysis:
                typer.echo(f"Error: {analysis['error']}")
                continue
            
            typer.echo("Claim Analysis:")
            for claim_data in analysis['claim_analysis']:
                status = "✓ SUPPORTED" if claim_data['supported'] else "✗ NOT SUPPORTED"
                typer.echo(f"  Claim {claim_data['claim_number']}: {claim_data['claim_text']}")
                typer.echo(f"    Status: {status}")
            
            typer.echo()
            typer.echo(f"Summary:")
            typer.echo(f"  Total Claims: {analysis['total_claims']}")
            typer.echo(f"  Supported Claims: {analysis['supported_claims']}")
            typer.echo(f"  Faithfulness Score: {analysis['faithfulness_score']}")
            typer.echo(f"  Formula: {analysis['supported_claims']}/{analysis['total_claims']} = {analysis['faithfulness_score']}")
            typer.echo()
    
    elif metric == "context_precision":
        from evaluators.ragas_context_precision import RagasContextPrecisionEvaluator
        evaluator = RagasContextPrecisionEvaluator()
        typer.echo("RAGAS Context Precision - Detailed Chunk Analysis")
        typer.echo("=" * 50)
        
        for i, data in enumerate(data_list, 1):
            typer.echo(f"\nTest Case {i}:")
            typer.echo("-" * 30)
            
            analysis = evaluator.get_detailed_analysis(data)
            
            typer.echo(f"Question: {data.get('question', '')}")
            typer.echo(f"Answer: {data.get('answer', '')}")
            typer.echo(f"Retrieved Contexts ({analysis.get('total_chunks', 0)} chunks):")
            for j, context in enumerate(data.get('context', []), 1):
                typer.echo(f"  [{j}] {context}")
            typer.echo()
            
            if "error" in analysis:
                typer.echo(f"Error: {analysis['error']}")
                continue
            
            typer.echo("Chunk Relevance Analysis:")
            for chunk_data in analysis['chunk_analysis']:
                status = "✓ RELEVANT" if chunk_data['relevant'] else "✗ NOT RELEVANT"
                typer.echo(f"  Chunk {chunk_data['chunk_number']}: {status}")
                typer.echo(f"    Text: {chunk_data['chunk_text']}")
            
            typer.echo()
            typer.echo("Precision@k Calculation:")
            for precision_data in analysis['precision_at_k']:
                typer.echo(f"  Precision@{precision_data['position']}: {precision_data['true_positives']}/{precision_data['total_chunks']} = {precision_data['precision']}")
            
            typer.echo()
            typer.echo(f"Summary:")
            typer.echo(f"  Total Chunks: {analysis['total_chunks']}")
            typer.echo(f"  Relevant Chunks: {analysis['relevant_chunks']}")
            typer.echo(f"  Mean Precision: {analysis['mean_precision']}")
            typer.echo(f"  Formula: Mean of Precision@k values = {analysis['mean_precision']}")
            typer.echo()
    
    elif metric == "hallucination":
        from evaluators.ragchecker_hallucination import RAGCheckerHallucinationEvaluator
        evaluator = RAGCheckerHallucinationEvaluator()
        typer.echo("RAGChecker Hallucination - Detailed Analysis")
        typer.echo("=" * 50)
        
        for i, data in enumerate(data_list, 1):
            typer.echo(f"\nTest Case {i}:")
            typer.echo("-" * 30)
            
            analysis = evaluator.get_detailed_analysis(data)
            
            typer.echo(f"Question: {data.get('question', '')}")
            typer.echo(f"Answer: {data.get('answer', '')}")
            typer.echo(f"Context: {' '.join(data.get('context', []))}")
            typer.echo()
            
            if "error" in analysis:
                typer.echo(f"Error: {analysis['error']}")
                continue
            
            typer.echo("Hallucination Analysis:")
            typer.echo(f"  Score: {analysis['hallucination_score']}% hallucination detected")
            typer.echo(f"  Interpretation: {analysis['interpretation']}")
            typer.echo()
            
            typer.echo("Analysis Breakdown:")
            breakdown = analysis['analysis_breakdown']
            typer.echo(f"  Response Length: {breakdown.get('response_length', 0)} characters")
            typer.echo(f"  Context Count: {breakdown.get('context_count', 0)} chunks")
            typer.echo(f"  Metrics Analyzed: {', '.join(breakdown.get('metrics_analyzed', []))}")
            
            if 'generator_metrics' in breakdown:
                gen_metrics = breakdown['generator_metrics']
                typer.echo(f"  Generator Metrics:")
                typer.echo(f"    Hallucination Score: {gen_metrics.get('hallucination', 0.0)}%")
            
            typer.echo()
    
    else:
        typer.echo(f"Error: Unknown metric '{metric}'. Available: faithfulness, context_precision, hallucination")
        raise typer.Exit(1)

@app.command()
def list_metrics():
    """List available evaluation metrics."""
    from evaluators import get_available_metrics
    
    typer.echo("Available Metrics:")
    metrics = get_available_metrics()
    for name, description in metrics.items():
        typer.echo(f"  • {name}: {description}")

@app.command()  
def version():
    """Show version information."""
    typer.echo("RAGTesterCLI - Unified RAG Evaluation")
    typer.echo("  Multi-provider LLM support (OpenAI, Anthropic, Google, OpenRouter)")
    typer.echo("  True RAGAS & RAGChecker integration")
    typer.echo("  One API key for any provider")

if __name__ == "__main__":
    app() 