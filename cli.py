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
            
            # Display formula explanation
            if 'formula_explanation' in analysis:
                formula = analysis['formula_explanation']
                typer.echo("📊 Faithfulness Formula:")
                typer.echo(f"  {formula['faithfulness_formula']}")
                typer.echo(f"  Calculation: {formula['calculation']}")
                typer.echo(f"  Interpretation: {formula['interpretation']}")
                typer.echo()
            
            # Display step-by-step verification process
            if 'verification_steps' in analysis:
                typer.echo("🔍 Claim Analysis:")
                for step in analysis['verification_steps']:
                    status_icon = "✓" if step['is_supported'] else "✗"
                    status_text = "SUPPORTED" if step['is_supported'] else "NOT SUPPORTED"
                    typer.echo(f"  {status_icon} Claim {step['step_number']}: {step['claim']}")
                    typer.echo(f"    Status: {status_text}")
                    typer.echo(f"    Verification: {step['verification_response']}")
                    
                    # Add brief context explanation
                    if step['is_supported']:
                        typer.echo(f"    Context: Found supporting evidence in retrieved chunks")
                    else:
                        typer.echo(f"    Context: No supporting evidence found in retrieved chunks")
                    typer.echo()
            
            # Display enhanced context usage analysis
            if 'context_usage_analysis' in analysis and 'chunks' in analysis['context_usage_analysis']:
                chunks = analysis['context_usage_analysis']['chunks']
                typer.echo("Context Usage Analysis:")
                for chunk in chunks:
                    usage_icon = "✓" if chunk['usage_status'] != 'unused_or_irrelevant' else "✗"
                    typer.echo(f"  {usage_icon} Chunk {chunk['chunk_number']}: {chunk['usage_status'].replace('_', ' ').title()}")
                    typer.echo(f"    Text: {chunk['chunk_text'][:80]}{'...' if len(chunk['chunk_text']) > 80 else ''}")
                    if chunk['supporting_claims']:
                        typer.echo(f"    Supporting: {len(chunk['supporting_claims'])} claims")
                typer.echo()
            
            # Display diagnostic insights
            if 'diagnostic_insights' in analysis:
                insights = analysis['diagnostic_insights']
                typer.echo(f"Diagnostic Insights (Severity: {insights.get('severity', 'unknown').upper()}):")
                if insights.get('primary_issues'):
                    typer.echo("  Issues Found:")
                    for issue in insights['primary_issues']:
                        typer.echo(f"    • {issue}")
                if insights.get('recommendations'):
                    typer.echo("  Recommendations:")
                    for rec in insights['recommendations']:
                        typer.echo(f"    • {rec}")
                if insights.get('specific_problems'):
                    typer.echo("  Specific Unsupported Claims:")
                    for problem in insights['specific_problems']:
                        typer.echo(f"    • Claim {problem['claim_number']}: {problem['claim_text']}")
                typer.echo()
            
            # Display summary verdict prominently for casual users
            if 'summary_verdict' in analysis:
                typer.echo("📋 Summary Verdict:")
                typer.echo(f"  {analysis['summary_verdict']}")
                typer.echo()
            
            # Display action recommendations
            if 'action_recommendations' in analysis:
                action_rec = analysis['action_recommendations']
                typer.echo("🎯 Action Recommendations:")
                typer.echo(f"  Action: {action_rec.get('action', 'none').replace('_', ' ').title()}")
                typer.echo(f"  Reason: {action_rec.get('reason', 'unknown').replace('_', ' ').title()}")
                typer.echo(f"  Confidence: {action_rec.get('confidence', 'unknown').title()}")
                if action_rec.get('specific_actions'):
                    typer.echo("  Specific Actions:")
                    for action in action_rec['specific_actions']:
                        typer.echo(f"    • {action}")
                typer.echo()
            
            typer.echo(f"Summary:")
            typer.echo(f"  Total Claims: {analysis['total_claims']}")
            typer.echo(f"  Supported Claims: {analysis['supported_claims']}")
            typer.echo(f"  Unsupported Claims: {analysis['unsupported_claims']}")
            typer.echo(f"  Faithfulness Score: {analysis['faithfulness_score']:.3f}")
            
            # Display additional metrics
            if 'summary_statistics' in analysis:
                stats = analysis['summary_statistics']
                typer.echo()
                typer.echo(f"Additional Metrics:")
                typer.echo(f"  Support Rate: {stats.get('support_rate', 0.0):.1%}")
                typer.echo(f"  Unsupported Rate: {stats.get('unsupported_rate', 0.0):.1%}")
                typer.echo(f"  Claim Density: {stats.get('claim_density', 0.0):.3f} claims/word")
                typer.echo(f"  Average Claim Length: {stats.get('average_claim_length', 0.0):.1f} words")
            
            # Display context usage summary
            if 'context_usage_analysis' in analysis and 'summary' in analysis['context_usage_analysis']:
                ctx_summary = analysis['context_usage_analysis']['summary']
                typer.echo()
                typer.echo(f"Context Utilization:")
                typer.echo(f"  Used Chunks: {ctx_summary.get('used_chunks', 0)}/{ctx_summary.get('total_chunks', 0)}")
                typer.echo(f"  Utilization Rate: {ctx_summary.get('context_utilization_rate', 0.0):.1%}")
                typer.echo(f"  Total Claims Supported: {ctx_summary.get('total_claims_supported', 0)}")
                typer.echo(f"  Average Relevance Score: {ctx_summary.get('average_relevance_score', 0.0):.3f}")
            
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
                typer.echo(f"    Text: {chunk_data['chunk_text'][:80]}{'...' if len(chunk_data['chunk_text']) > 80 else ''}")
            
            typer.echo()
            typer.echo("Precision@k Calculation:")
            for precision_data in analysis['precision_at_k']:
                typer.echo(f"  Precision@{precision_data['position']}: {precision_data['true_positives']}/{precision_data['total_chunks']} = {precision_data['precision']}")
            
            typer.echo()
            
            # Display retrieval quality analysis
            if 'retrieval_quality_analysis' in analysis:
                quality = analysis['retrieval_quality_analysis']
                typer.echo("Retrieval Quality Analysis:")
                typer.echo(f"  Precision Trend: {quality.get('precision_trend', 'unknown').replace('_', ' ').title()}")
                typer.echo(f"  Ranking Quality: {quality.get('ranking_quality', 'unknown').replace('_', ' ').title()}")
                typer.echo(f"  Noise Pattern: {quality.get('noise_pattern', 'unknown').replace('_', ' ').title()}")
                typer.echo()
            
            # Display diagnostic insights
            if 'diagnostic_insights' in analysis:
                insights = analysis['diagnostic_insights']
                typer.echo(f"Diagnostic Insights (Severity: {insights.get('severity', 'unknown').upper()}):")
                if insights.get('primary_issues'):
                    typer.echo("  Issues Found:")
                    for issue in insights['primary_issues']:
                        typer.echo(f"    • {issue}")
                if insights.get('recommendations'):
                    typer.echo("  Recommendations:")
                    for rec in insights['recommendations']:
                        typer.echo(f"    • {rec}")
                if insights.get('specific_problems'):
                    typer.echo("  Irrelevant Chunks:")
                    for problem in insights['specific_problems']:
                        typer.echo(f"    • Chunk {problem['chunk_number']}: {problem['chunk_text']}")
                typer.echo()
            
            # Display summary verdict prominently for casual users
            if 'summary_verdict' in analysis:
                typer.echo("📋 Summary Verdict:")
                typer.echo(f"  {analysis['summary_verdict']}")
                typer.echo()
            
            typer.echo(f"Summary:")
            typer.echo(f"  Total Chunks: {analysis['total_chunks']}")
            typer.echo(f"  Relevant Chunks: {analysis['relevant_chunks']}")
            typer.echo()
            
            # Display both precision metrics with clear explanations
            ragas_precision = analysis.get('ragas_context_precision', analysis.get('context_precision', 0.0))
            simple_precision = analysis.get('simple_context_precision', 0.0)
            
            typer.echo(f"Context Precision Metrics:")
            typer.echo(f"  📊 RAGAS Context Precision: {ragas_precision}")
            typer.echo(f"      → Measures ranking quality: 'Did we rank relevant chunks early?'")
            typer.echo(f"      → Formula: Σ(Precision@k × v_k) / Σ(v_k)")
            typer.echo()
            typer.echo(f"  🔍 Simple Context Precision: {simple_precision}")
            typer.echo(f"      → Measures retrieval cleanliness: 'How much retrieved content is useful?'")
            typer.echo(f"      → Formula: Relevant Chunks / Total Chunks")
            
            # Display summary statistics
            if 'summary_statistics' in analysis:
                stats = analysis['summary_statistics']
                typer.echo()
                typer.echo(f"Additional Metrics:")
                typer.echo(f"  Noise Rate: {stats.get('noise_rate', 0.0):.1%}")
                typer.echo(f"  Early Precision: {stats.get('early_precision', 0.0):.3f}")
                typer.echo(f"  Late Precision: {stats.get('late_precision', 0.0):.3f}")
            
            # Interpretive guidance based on both metrics
            typer.echo()
            typer.echo("Interpretation:")
            if ragas_precision >= 0.8 and simple_precision >= 0.8:
                typer.echo("  ✅ Excellent: Good ranking AND low noise")
                typer.echo("      → Relevant chunks appear early with minimal irrelevant content")
            elif ragas_precision >= 0.8 and simple_precision < 0.5:
                typer.echo("  ⚠️  Good early ranking but noisy retrieval")
                typer.echo("      → Relevant evidence appears first, but overall retrieval contains significant noise")
                typer.echo("      → Recommendation: Keep top results, prune or rerank the rest to suppress noise")
            elif ragas_precision < 0.5 and simple_precision >= 0.8:
                typer.echo("  ⚠️  Clean retrieval but poor ranking")
                typer.echo("      → Low noise but relevant chunks appear late in ranking")
                typer.echo("      → Recommendation: Improve ranking algorithm - top results should be most relevant")
            else:
                typer.echo("  ❌ Both ranking and noise need improvement")
                typer.echo("      → Poor early precision AND high noise content")
                typer.echo("      → Recommendation: Overhaul both similarity matching and ranking algorithms")
            
            # Display action recommendations if available
            if 'action_recommendations' in analysis:
                action_rec = analysis['action_recommendations']
                typer.echo()
                typer.echo("🎯 Action Recommendations:")
                typer.echo(f"  Action: {action_rec.get('action', 'none')}")
                if action_rec.get('target_ranks'):
                    typer.echo(f"  Target Ranks: {action_rec['target_ranks']}")
                if action_rec.get('keep_ranks'):
                    typer.echo(f"  Keep Ranks: {action_rec['keep_ranks']}")
                typer.echo(f"  Reason: {action_rec.get('reason', 'unknown')}")
                typer.echo(f"  Confidence: {action_rec.get('confidence', 'unknown')}")
            
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
            if data.get('ground_truth'):
                typer.echo(f"Ground Truth: {data.get('ground_truth')}")
            typer.echo()
            
            if "error" in analysis:
                typer.echo(f"Error: {analysis['error']}")
                continue
            
            # Display formula explanation
            if 'formula_explanation' in analysis:
                formula = analysis['formula_explanation']
                typer.echo("📊 RAGChecker Hallucination Formula:")
                typer.echo(f"  {formula['ragchecker_formula']}")
                typer.echo(f"  Calculation: {formula['calculation']}")
                typer.echo(f"  Interpretation: {formula['interpretation']}")
                typer.echo(f"  Explanation: {formula['explanation']}")
                typer.echo(f"  Evaluation Mode: {formula['evaluation_mode'].replace('_', ' ').title()}")
                typer.echo()
            
            # Display enhanced claim-level analysis
            if 'claim_analysis' in analysis and 'claims' in analysis['claim_analysis']:
                claims = analysis['claim_analysis']['claims']
                typer.echo("🔍 Claim-Level Analysis:")
                for claim in claims:
                    status_icon = "✓" if claim['classification'] == 'correct_grounded' else "✗"
                    typer.echo(f"  {status_icon} Claim {claim['claim_number']}: {claim['claim_text']}")
                    typer.echo(f"    Classification: {claim['classification'].replace('_', ' ').title()}")
                    if claim['context_support']['supported']:
                        typer.echo(f"    Context Support: ✓ {claim['context_support']['explanation']}")
                    else:
                        typer.echo(f"    Context Support: ✗ {claim['context_support']['explanation']}")
                    if analysis.get('evaluation_mode') == 'full':
                        if claim['ground_truth_support']['supported']:
                            typer.echo(f"    Ground Truth: ✓ {claim['ground_truth_support']['explanation']}")
                        else:
                            typer.echo(f"    Ground Truth: ✗ {claim['ground_truth_support']['explanation']}")
                    typer.echo()
            
            # Display context analysis
            if 'context_analysis' in analysis and 'chunks' in analysis['context_analysis']:
                chunks = analysis['context_analysis']['chunks']
                typer.echo("Context Usage Analysis:")
                for chunk in chunks:
                    usage_icon = "✓" if chunk['usage_analysis'] != 'irrelevant_noise' else "✗"
                    typer.echo(f"  {usage_icon} Chunk {chunk['chunk_number']}: {chunk['usage_analysis'].replace('_', ' ').title()}")
                    typer.echo(f"    Text: {chunk['chunk_text'][:80]}{'...' if len(chunk['chunk_text']) > 80 else ''}")
                    if chunk['supports_claims']:
                        claim_nums = [str(c['claim_number']) for c in chunk['supports_claims']]
                        typer.echo(f"    Supports Claims: {', '.join(claim_nums)}")
                typer.echo()
            
            # Display diagnostic insights
            if 'diagnostic_insights' in analysis:
                insights = analysis['diagnostic_insights']
                typer.echo(f"Diagnostic Insights (Severity: {insights.get('severity', 'unknown').upper()}):")
                if insights.get('primary_issues'):
                    typer.echo("  Issues Found:")
                    for issue in insights['primary_issues']:
                        typer.echo(f"    • {issue}")
                if insights.get('recommendations'):
                    typer.echo("  Recommendations:")
                    for rec in insights['recommendations']:
                        typer.echo(f"    • {rec}")
                typer.echo()
            
            # Display summary verdict prominently for casual users
            if 'summary_verdict' in analysis:
                typer.echo("📋 Summary Verdict:")
                typer.echo(f"  {analysis['summary_verdict']}")
                typer.echo()
            
            # Display action recommendations
            if 'action_recommendations' in analysis:
                action_rec = analysis['action_recommendations']
                typer.echo("🎯 Action Recommendations:")
                typer.echo(f"  Action: {action_rec.get('action', 'none').replace('_', ' ').title()}")
                typer.echo(f"  Reason: {action_rec.get('reason', 'unknown').replace('_', ' ').title()}")
                typer.echo(f"  Confidence: {action_rec.get('confidence', 'unknown').title()}")
                if action_rec.get('specific_actions'):
                    typer.echo("  Specific Actions:")
                    for action in action_rec['specific_actions']:
                        typer.echo(f"    • {action}")
                typer.echo()
            
            typer.echo(f"Summary:")
            typer.echo(f"  Total Claims: {analysis.get('summary_statistics', {}).get('total_claims', 0)}")
            typer.echo(f"  Hallucinated Claims: {analysis.get('summary_statistics', {}).get('hallucinated_claims', 0)}")
            typer.echo(f"  Correct Grounded Claims: {analysis.get('summary_statistics', {}).get('correct_grounded_claims', 0)}")
            typer.echo(f"  Context Supported but Incorrect: {analysis.get('summary_statistics', {}).get('context_supported_but_incorrect', 0)}")
            typer.echo(f"  Missing Context Evidence: {analysis.get('summary_statistics', {}).get('missing_context_evidence', 0)}")
            typer.echo(f"  Hallucination Score: {analysis['hallucination_score']:.1f}%")
            
            # Display additional metrics
            if 'summary_statistics' in analysis:
                stats = analysis['summary_statistics']
                typer.echo()
                typer.echo(f"Additional Metrics:")
                typer.echo(f"  Hallucination Rate: {stats.get('hallucination_rate', 0.0):.1%}")
                typer.echo(f"  Grounding Rate: {stats.get('grounding_rate', 0.0):.1%}")
                typer.echo(f"  Context Support Rate: {stats.get('context_support_rate', 0.0):.1%}")
                typer.echo(f"  Claim Density: {stats.get('claim_density', 0.0):.3f} claims/word")
                typer.echo(f"  Average Claim Length: {stats.get('average_claim_length', 0.0):.1f} words")
            
            # Display context analysis summary
            if 'context_analysis' in analysis and 'summary' in analysis['context_analysis']:
                ctx_summary = analysis['context_analysis']['summary']
                typer.echo()
                typer.echo(f"Context Analysis:")
                typer.echo(f"  Total Chunks: {ctx_summary.get('total_chunks', 0)}")
                typer.echo(f"  Relevant Chunks: {ctx_summary.get('relevant_chunks', 0)}")
                typer.echo(f"  Context Precision: {ctx_summary.get('context_precision', 0.0):.3f}")
                typer.echo(f"  Noise Chunks: {ctx_summary.get('noise_chunks', 0)}")
                typer.echo(f"  Average Relevance Score: {ctx_summary.get('average_relevance_score', 0.0):.3f}")
            
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