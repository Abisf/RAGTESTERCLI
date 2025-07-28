#!/usr/bin/env python3
"""
Test script to demonstrate RAGAS claim-based faithfulness analysis.

This script shows exactly how faithfulness is calculated:
1. Extract discrete claims from the answer
2. Verify each claim against the context
3. Compute ratio: supported_claims / total_claims
"""

import json
import os
from evaluators.ragas_faithfulness import RagasFaithfulnessEvaluator

def test_claim_based_faithfulness():
    """Test the claim-based faithfulness calculation with detailed analysis."""
    
    # Test cases from the RAGAS documentation
    test_cases = [
        {
            "name": "Einstein - High Faithfulness",
            "question": "Where and when was Einstein born?",
            "context": ["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist."],
            "answer": "Einstein was born in Germany on 14 March 1879."
        },
        {
            "name": "Einstein - Low Faithfulness (Date Error)", 
            "question": "Where and when was Einstein born?",
            "context": ["Albert Einstein (born 14 March 1879) was a German-born theoretical physicist."],
            "answer": "Einstein was born in Germany on 20 March 1879."
        },
        {
            "name": "Model X - Mixed Faithfulness",
            "question": "What is the battery life of Model X?",
            "context": [
                "Model X battery lasts up to 10 hours.",
                "Weight of Model X is 1.2 kg."
            ],
            "answer": "Model X has a battery life of 12 hours and weighs 1.2 kg."
        }
    ]
    
    # Initialize evaluator
    evaluator = RagasFaithfulnessEvaluator()
    
    print("RAGAS Faithfulness - Claim-Based Analysis")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        # Get detailed analysis
        analysis = evaluator.get_detailed_analysis(test_case)
        
        print(f"Question: {test_case['question']}")
        print(f"Context: {' '.join(test_case['context'])}")
        print(f"Answer: {test_case['answer']}")
        print()
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            continue
        
        print("Claim Analysis:")
        for claim_data in analysis['claim_analysis']:
            status = "✓ SUPPORTED" if claim_data['supported'] else "✗ NOT SUPPORTED"
            print(f"  Claim {claim_data['claim_number']}: {claim_data['claim_text']}")
            print(f"    Status: {status}")
        
        print()
        print(f"Summary:")
        print(f"  Total Claims: {analysis['total_claims']}")
        print(f"  Supported Claims: {analysis['supported_claims']}")
        print(f"  Faithfulness Score: {analysis['faithfulness_score']}")
        print(f"  Formula: {analysis['supported_claims']}/{analysis['total_claims']} = {analysis['faithfulness_score']}")
        print()

if __name__ == "__main__":
    test_claim_based_faithfulness() 