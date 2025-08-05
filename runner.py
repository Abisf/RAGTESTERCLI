#!/usr/bin/env python3
"""
Runner module for RAGTESTERCLI

Handles loading input data, running evaluations, and formatting output.
"""

import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional
from tabulate import tabulate
from evaluators import get_evaluator
import config
import typer

def load_input_data(file_path: str) -> List[Dict[str, Any]]:
    """Load and validate input data from JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Ensure data is a list
        if isinstance(data, dict):
            data = [data]
        
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in input file: {e}")

def run_evaluation(
    input_file: str, 
    metric: str, 
    output_file: Optional[str] = None,
    output_format: str = "json",
    verbose: bool = False,
    model_config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Run evaluation on input data using specified metric.
    
    Args:
        input_file: Path to input JSON file
        metric: Name of the evaluation metric
        output_file: Optional output file path
        output_format: Output format (json, csv, table)
        verbose: Enable verbose output
        model_config: Model configuration for flexible LLM support
    """
    start_time = time.time()
    
    if verbose:
        print(f"Loading input data from {input_file}...")
    
    # Load input data
    data = load_input_data(input_file)
    
    if verbose:
        print(f"Running evaluation with metric: {metric}")
    
    # Get evaluator with model configuration
    evaluator = get_evaluator(metric, model_config=model_config)
    
    # Run evaluation on each item
    scores = []
    error_messages = []
    
    for i, item in enumerate(data):
        try:
            score = evaluator.evaluate(item)
            scores.append(score)
        except Exception as e:
            error_msg = str(e)
            
            # Check for quota/rate limit errors and stop execution
            if any(keyword in error_msg.lower() for keyword in ["quota", "rate limit", "insufficient_quota", "billing"]):
                print(f"\nError: API quota exceeded")
                print(f"Please check your billing and usage limits, or try again later.")
                print(f"Details: {error_msg}")
                raise typer.Exit(1)
            elif "authentication" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print(f"\nError: Authentication failed")
                print(f"Please check your API key and try again.")
                print(f"Details: {error_msg}")
                raise typer.Exit(1)
            else:
                friendly_error = f"Evaluation failed: {error_msg}"
                error_messages.append(f"Item {i+1}: {friendly_error}")
                scores.append(float('nan'))  # Use NaN for other errors
    
    end_time = time.time()
    execution_time = round(end_time - start_time, 2)
    
    if verbose:
        print(f"\nEvaluation completed in {execution_time}s")
    
    # Format results
    result = {
        "metric": metric,
        "summary": {
            "total_items": len(scores),
            "average_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "min_score": min(scores) if scores else 0.0,
            "max_score": max(scores) if scores else 0.0,
            "execution_time_seconds": execution_time
        },
        "scores": scores
    }
    
    # Add error messages if any occurred
    if error_messages:
        result["errors"] = error_messages
    
    # Format output
    output = format_output(result, output_format)
    
    # Save output if requested
    if output_file:
        save_output(output, output_file, output_format)
        if verbose:
            print(f"Results saved to {output_file}")
    
    return output

def format_output(result: Dict[str, Any], format_type: str = "json") -> str:
    """Format evaluation results in specified format."""
    if format_type == "json":
        return json.dumps(result, indent=2)
    
    elif format_type == "csv":
        # Create DataFrame for CSV output
        df = pd.DataFrame({
            'item_index': range(len(result['scores'])),
            'score': result['scores']
        })
        return df.to_csv(index=False)
    
    elif format_type == "table":
        # Create tabulated output
        summary = result['summary']
        table_data = [
            ["Metric", result['metric']],
            ["Total Items", summary['total_items']],
            ["Average Score", summary['average_score']],
            ["Min Score", summary['min_score']],
            ["Max Score", summary['max_score']],
            ["Execution Time (s)", summary['execution_time_seconds']]
        ]
        
        table = tabulate(table_data, headers=["Property", "Value"], tablefmt="grid")
        
        # Add individual scores
        score_data = [[i, score] for i, score in enumerate(result['scores'])]
        score_table = tabulate(score_data, headers=["Item", "Score"], tablefmt="grid")
        
        output = f"{table}\n\nIndividual Scores:\n{score_table}"
        
        # Add error messages if any
        if 'errors' in result:
            error_table = tabulate([[msg] for msg in result['errors']], headers=["Error"], tablefmt="grid")
            output += f"\n\nErrors:\n{error_table}"
        
        return output
    
    else:
        raise ValueError(f"Unsupported output format: {format_type}")

def save_output(output: str, file_path: str, format_type: str):
    """Save formatted output to file."""
    with open(file_path, 'w') as f:
        f.write(output) 