# RAGTesterCLI

A comprehensive CLI tool for evaluating Retrieval-Augmented Generation (RAG) pipelines with detailed diagnostic analysis, multi-provider LLM support, and professional reporting capabilities.

**Note: This is an MVP (Minimum Viable Product). We are actively seeking user feedback to guide future development. Please reach out with your suggestions and feature requests!**

## Vision

RAG Tester CLI is the first step toward a full-stack observability and testing platform for Retrieval-Augmented Generation (RAG) and Agentic AI systems.

- **Today**: Run evaluations by unifying different RAG testing tools targeting fragmentation (currently)
- **Next**: Capture telemetry, version RAG artifacts locally, hosted artifact registry, agent simulation, compliance tooling
- **Endgame**: The control center for intelligent systems — versioning, testing, and governing AI behavior at scale

## Contact & Feedback

We welcome your feedback and feature requests! Please reach out to our development team:

- **Abyesolome Assefa** (Developer): abistech@umich.edu
- **Jeremi Owusu** (Developer): jkowusu@umich.edu

## Purpose

This tool provides a unified interface for running RAG evaluation metrics from popular frameworks like RAGAS and RAGChecker, with comprehensive diagnostic analysis that goes beyond simple scores to provide actionable insights and detailed breakdowns.

## Key Features

- **Detailed Diagnostic Analysis**: Step-by-step claim verification, formula explanations, and context usage analysis
- **Professional Reporting**: Summary verdicts, action recommendations, and confidence assessments
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, OpenRouter, and more
- **Multiple Evaluation Metrics**: RAGAS Faithfulness, Context Precision, and RAGChecker Hallucination
- **Comprehensive Insights**: Claim-level analysis, context utilization, and diagnostic recommendations
- **Extensible Design**: Easy to add new metrics and providers
- **Flexible Output**: JSON, table, and detailed analysis formats

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd RAGTESTERCLI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys (optional - can also use CLI flags):
```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys
```

### Basic Usage

```bash
# Quick evaluation (scores only)
python cli.py test --input examples/test.json --metric faithfulness_ragas --llm-model gpt-4

# Comprehensive analysis with detailed diagnostics
python cli.py analyze --input examples/test.json --metric faithfulness --llm-model gpt-3.5-turbo

# Multi-provider support
python cli.py analyze --input examples/test.json --metric hallucination --llm-model anthropic/claude-3-haiku --api-base https://openrouter.ai/api/v1

# Context precision analysis with dual metrics
python cli.py analyze --input examples/louvre_context_precision_test.json --metric context_precision --llm-model gpt-3.5-turbo
```

## Supported Metrics

| Metric | Tool | Description | Enhanced Features |
|--------|------|-------------|------------------|
| `faithfulness` | RAGAS | Measures if answer is grounded in context | Step-by-step claim verification, Context usage analysis, Action recommendations |
| `context_precision` | RAGAS | Measures chunk relevance and ranking quality | Dual metrics (RAGAS + Simple), Precision@k calculations, Ranking quality analysis |
| `hallucination` | RAGChecker | Detects hallucinated information | Claim-level entailment analysis, Ground truth comparison, Pure invention detection |

## Detailed Analysis Features

### RAGAS Faithfulness
```
Faithfulness Formula:
  Supported Claims / Total Claims
  Calculation: 5 / 10 = 0.500
  Interpretation: 50.0% of claims are supported by context

Claim Analysis:
  Claim 1: The Louvre Museum is located in Paris.
    Status: SUPPORTED
    Verification: YES
    Context: Found supporting evidence in retrieved chunks

Summary Verdict:
  Moderate faithfulness (50.0%) - 5/10 claims supported

Action Recommendations:
  Action: Improve Generation
  Reason: High Unsupported Claims
  Confidence: High
```

### RAGAS Context Precision
```
RAGAS Context Precision: 1.0
  Measures ranking quality: 'Did we rank relevant chunks early?'
  Formula: Σ(Precision@k × v_k) / Σ(v_k)

Simple Context Precision: 0.667
  Measures retrieval cleanliness: 'How much retrieved content is useful?'
  Formula: Relevant Chunks / Total Chunks

Action Recommendations:
  Action: Prune
  Target Ranks: [2, 4, 5]
  Reason: High noise with good early ranking
```

### RAGChecker Hallucination
```
RAGChecker Hallucination Formula:
  np.mean(unfaithful & ~answer2response) * 100
  Calculation: Hallucinated Claims / Total Claims = 2 / 7 = 28.6%
  Interpretation: 28.6% of ALL response claims are pure hallucinations

Claim-Level Analysis:
  Claim 1: Tokyo is capital of Japan
    Classification: Correct Grounded
    Context Support: Supported by chunk 1 (Entailment)
    Ground Truth: Ground truth: Entailment (supported)

Summary Verdict:
  Moderate hallucination risk (28.6%) - 2/7 claims are pure inventions
```

## Input Format

The tool expects a JSON file with the following structure:

```json
[
  {
    "question": "What are the features of the Louvre Museum?",
    "context": [
      "The Louvre Museum is located in Paris, France. It is open from 9 AM to 6 PM daily.",
      "The museum houses the famous Mona Lisa painting and has over 35,000 works of art.",
      "The Louvre was originally a royal palace before becoming a museum."
    ],
    "answer": "The Louvre Museum is located in Paris and is open from 9 AM to 6 PM daily. It houses the famous Mona Lisa painting and has over 35,000 works of art.",
    "ground_truth": "The Louvre Museum is located in Paris with many artworks." // Optional for RAGChecker
  }
]
```

## Enhanced Output Features

### Comprehensive Analysis
- **Formula Explanations**: Clear mathematical breakdowns of how scores are calculated
- **Step-by-Step Verification**: Detailed claim-by-claim analysis with verification process
- **Context Usage Analysis**: How well retrieved chunks support the generated claims
- **Diagnostic Insights**: Severity levels, specific issues, and actionable recommendations
- **Summary Verdicts**: Human-readable assessments with clear messaging
- **Action Recommendations**: Structured suggestions with confidence levels and specific actions

### Professional Reporting
- **Dual Metrics**: Both ranking quality and retrieval cleanliness for context precision
- **Confidence Assessment**: High/Medium/Low confidence levels for recommendations
- **Additional Metrics**: Claim density, average claim length, context utilization rates
- **Structured Output**: Machine-readable recommendations for automation

## Project Structure

```
RAGTESTERCLI/
├── cli.py                          # Main CLI interface with enhanced output
├── config.py                       # Configuration management
├── runner.py                       # Evaluation orchestration
├── evaluators/
│   ├── base_evaluator.py          # Base evaluator class
│   ├── ragas_faithfulness.py      # Enhanced RAGAS faithfulness with detailed analysis
│   ├── ragas_context_precision.py # Dual-metric context precision evaluation
│   └── ragchecker_hallucination.py # RAGChecker hallucination with claim-level analysis
├── llm/
│   └── openai_client.py           # Multi-provider LLM wrapper
├── examples/
│   ├── faithfulness_test_with_issues.json
│   ├── louvre_context_precision_test.json
│   ├── detailed_claim_example.json
│   └── hallucination_context_only_test.json
└── requirements.txt               # Dependencies
```

## Advanced Usage

### Multi-Provider LLM Support
```bash
# OpenAI
python cli.py analyze --input test.json --metric faithfulness --llm-model gpt-4

# OpenRouter (Claude)
python cli.py analyze --input test.json --metric hallucination --llm-model anthropic/claude-3-haiku --api-base https://openrouter.ai/api/v1

# Google Gemini
python cli.py analyze --input test.json --metric context_precision --llm-model gemini-pro
```

### Environment Variables
```bash
# .env file
OPENAI_API_KEY=your_openai_key
OPENAI_API_BASE=https://api.openai.com/v1
RAGCLI_LLM_MODEL=gpt-4
```

### Output Formats
```bash
# JSON format
python cli.py test --input test.json --metric faithfulness_ragas --output results.json

# Table format
python cli.py test --input test.json --metric faithfulness_ragas --format table

# Detailed analysis (default)
python cli.py analyze --input test.json --metric faithfulness
```

## Key Improvements

### Enhanced Diagnostic Capabilities
- **Claim-Level Analysis**: Step-by-step verification of each claim
- **Context Utilization**: How well retrieved chunks support claims
- **Formula Transparency**: Clear explanations of how scores are calculated
- **Actionable Insights**: Specific recommendations with confidence levels

### Professional Reporting
- **Summary Verdicts**: Human-readable assessments with clear messaging
- **Dual Metrics**: Both ranking quality and retrieval cleanliness
- **Structured Recommendations**: Machine-readable action items
- **Confidence Assessment**: Reliability indicators for recommendations

### Streamlined Output
- **No Redundancy**: Eliminated duplicate information sections
- **Logical Flow**: Formula → Analysis → Summary → Actions
- **Comprehensive Coverage**: All evaluators provide equal detail level

## Future Enhancements

- **RAGLAB Integration**: Algorithmic analysis capabilities
- **Version Control**: Diff tracking for evaluation results
- **Artifact Tracking**: JFrog-style artifact management for RAG
- **Enterprise Features**: Multi-user support, audit trails
- **Web Interface**: GUI for non-technical users
- **CI/CD Integration**: Automated evaluation pipelines

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 