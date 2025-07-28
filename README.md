# RAGTESTERCLI

A unified CLI tool for evaluating Retrieval-Augmented Generation (RAG) pipelines with support for multiple LLM providers and evaluation metrics.

## Purpose

This tool provides a unified interface for running RAG evaluation metrics from popular open-source tools like RAGAS and RAGChecker, with support for multiple LLM providers.

## Features

- **Unified CLI Interface**: Typer-based command-line interface
- **Multiple Metrics**: Support for RAGAS and RAGChecker evaluations
- **Multi-Provider LLM Support**: OpenAI, Anthropic, Google, OpenRouter, and more
- **Environment Variable Support**: Secure API key management via .env files
- **Flexible Output**: JSON and table formats
- **Extensible Design**: Easy to add new metrics and providers

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
# Run faithfulness evaluation with OpenAI
python cli.py test --input examples/test.json --metric faithfulness_ragas --llm-model gpt-4

# Run hallucination detection with OpenRouter
python cli.py test --input examples/test.json --metric hallucination_ragchecker --llm-model anthropic/claude-3-haiku --api-base https://openrouter.ai/api/v1

# Using environment variables (.env file)
python cli.py test --input examples/test.json --metric faithfulness_ragas --llm-model gpt-3.5-turbo

# Detailed claim-by-claim analysis (shows exactly how faithfulness is calculated)
python cli.py analyze --input examples/einstein_faithfulness_test.json --metric faithfulness --llm-model gpt-3.5-turbo

# Context precision analysis (shows chunk-by-chunk relevance with Precision@k)
python cli.py analyze --input examples/louvre_context_precision_test.json --metric context_precision --llm-model gpt-3.5-turbo

# Output as table format
python cli.py test --input examples/test.json --metric faithfulness_ragas --output-format table

# Save results to file
python cli.py test --input examples/test.json --metric faithfulness_ragas --output results.json
```

### List Available Metrics

```bash
python cli.py list-metrics
```

## Supported Metrics

| Tool | Metric | Description |
|------|--------|-------------|
| RAGAS | `faithfulness_ragas` | Measures if answer is grounded in context |
| RAGAS | `context_precision_ragas` | Measures proportion of relevant chunks in retrieved contexts |
| RAGChecker | `hallucination_ragchecker` | Detects hallucinated information |

## Input Format

The tool expects a JSON file with the following structure:

```json
[
  {
    "question": "When was GPT-4 released?",
    "context": ["GPT-4 was released in March 2023 by OpenAI."],
    "answer": "GPT-4 came out in 2023."
  }
]
```

Required fields:
- `question`: The question being answered
- `context`: Array of context strings or single string
- `answer`: The answer to evaluate

## Configuration

Create a `ragtester.yaml` file for custom configuration:

```yaml
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
  temperature: 0.0
  max_tokens: 1000

output:
  format: json
  include_reasoning: false
  decimal_places: 3
```

## Project Structure

```
RAGTESTERCLI/
├── cli.py                          # Main CLI interface
├── config.py                       # Configuration management
├── runner.py                       # Evaluation orchestration
├── evaluators/
│   ├── base_evaluator.py          # Base evaluator class
│   ├── ragas_faithfulness.py      # RAGAS faithfulness metric
│   └── ragchecker_hallucination.py # RAGChecker hallucination metric
├── llm/
│   └── openai_client.py           # OpenAI API wrapper
├── examples/
│   └── test.json                  # Sample test data
└── requirements.txt               # Dependencies
```

## Adding New Metrics

1. Create a new evaluator in `evaluators/`:
```python
from .base_evaluator import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def evaluate(self, data):
        # Your evaluation logic here
        return score
```

2. Register it in `evaluators/__init__.py`:
```python
EVALUATORS = {
    'my_custom_metric': MyCustomEvaluator,
    # ... other evaluators
}
```

## Output Example

### JSON Format
```json
{
  "metric": "faithfulness_ragas",
  "summary": {
    "total_items": 3,
    "average_score": 0.867,
    "min_score": 0.750,
    "max_score": 0.950,
    "execution_time_seconds": 12.34
  },
  "scores": [0.950, 0.850, 0.750]
}
```

### Table Format
```
EVALUATION SUMMARY
==================================================
┌─────────────────────┬─────────────────────────┐
│ Metric              │ Value                   │
├─────────────────────┼─────────────────────────┤
│ Metric              │ faithfulness_ragas      │
│ Total Items         │ 3                       │
│ Average Score       │ 0.867                   │
│ Min Score           │ 0.750                   │
│ Max Score           │ 0.950                   │
│ Execution Time (s)  │ 12.34                   │
└─────────────────────┴─────────────────────────┘
```

## Development

This tool focuses on:
- Plug-and-play metric testing
- LLM-driven scoring
- Clean CLI interface
- Expandable backend

Future enhancements could include:
- Additional metric providers (LynX, HaluBench, DeepEval)
- More LLM providers (Claude, Mistral)
- Batch processing optimizations
- Web interface
- Integration with CI/CD pipelines

## License

[Add your license here]

## Contributing

[Add contribution guidelines here] 