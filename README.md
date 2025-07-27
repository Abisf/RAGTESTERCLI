# RAGTESTERCLI - MVP (v0.1)

A CLI tool for evaluating Retrieval-Augmented Generation (RAG) pipelines with plug-and-play metric testing and LLM-driven scoring.

## 🎯 Purpose

This MVP validates developer demand for a unified CLI wrapper that runs multiple types of RAG evaluation metrics from popular open-source tools like RAGAS and RAGChecker.

## ✨ Features

- **Simple CLI Interface**: Typer-based command-line interface
- **Multiple Metrics**: Support for RAGAS and RAGChecker evaluations
- **LLM-Powered**: Uses OpenAI GPT-4 for scoring (configurable)
- **Flexible Output**: JSON and table formats
- **Extensible Design**: Easy to add new metrics and providers

## 🚀 Quick Start

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

3. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

### Basic Usage

```bash
# Run faithfulness evaluation
python cli.py test --input examples/test.json --metric faithfulness_ragas

# Run hallucination detection  
python cli.py test --input examples/test.json --metric hallucination_ragchecker

# Output as table format
python cli.py test --input examples/test.json --metric faithfulness_ragas --output-format table

# Save results to file
python cli.py test --input examples/test.json --metric faithfulness_ragas --output results.json
```

### List Available Metrics

```bash
python cli.py list-metrics
```

## 📊 Supported Metrics

| Tool | Metric | Description |
|------|--------|-------------|
| RAGAS | `faithfulness_ragas` | Measures if answer is grounded in context |
| RAGChecker | `hallucination_ragchecker` | Detects hallucinated information |

## 📋 Input Format

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

## 🔧 Configuration

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

## 📁 Project Structure

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

## 🧩 Adding New Metrics

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

## 📈 Output Example

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

## 🛠️ Development

This is an MVP focused on:
- ✅ Plug-and-play metric testing
- ✅ LLM-driven scoring
- ✅ Clean CLI interface  
- ✅ Expandable backend

Future enhancements could include:
- Additional metric providers (LynX, HaluBench, DeepEval)
- More LLM providers (Claude, Mistral)
- Batch processing optimizations
- Web interface
- Integration with CI/CD pipelines

## 📄 License

[Add your license here]

## 🤝 Contributing

[Add contribution guidelines here] 