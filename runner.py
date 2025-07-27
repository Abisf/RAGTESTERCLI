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
    for item in data:
        try:
            score = evaluator.evaluate(item)
            scores.append(score)
        except Exception as e:
            if verbose:
                print(f"Error evaluating item: {e}")
            scores.append(0.0)  # Default score on error
    
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
        
        return f"{table}\n\nIndividual Scores:\n{score_table}"
    
    else:
        raise ValueError(f"Unsupported output format: {format_type}")

def save_output(output: str, file_path: str, format_type: str):
    """Save formatted output to file."""
    with open(file_path, 'w') as f:
        f.write(output) 