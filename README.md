# RAGTesterCLI

A comprehensive CLI tool for evaluating Retrieval-Augmented Generation (RAG) pipelines with detailed diagnostic analysis, multi-provider LLM support, and professional reporting capabilities.

**Note: This is an MVP (Minimum Viable Product). We are actively seeking user feedback to guide future development. Please reach out with your suggestions and feature requests!**

## Vision

RAG Tester CLI is the first step toward a full-stack observability and testing platform for Retrieval-Augmented Generation (RAG) and Agentic AI systems.

- Currently: Run evaluations by unifying different RAG testing tools targeting fragmentation
- Next: Capture telemetry, version RAG artifacts locally, hosted artifact registry, agent simulation, compliance tooling
- Endgame: The control center for intelligent systems — versioning, testing, and governing AI behavior at scale

## Contact & Feedback

We welcome your feedback and feature requests! Please reach out to our development team:

- **Abyesolome Assefa** (Developer): abistech@umich.edu
- **Jeremi Owusu** (Developer): jkowusu@umich.edu

## Purpose

This tool provides a unified interface for running RAG evaluation metrics from popular frameworks like RAGAS and RAGChecker, with comprehensive diagnostic analysis that goes beyond simple scores to provide actionable insights and detailed breakdowns.

## Key Features

- **Detailed Diagnostic Analysis**: Step-by-step claim verification, formula explanations, and context usage analysis
- **Professional Reporting**: Summary verdicts, action recommendations, and confidence assessments
- **Universal LLM Support**: OpenAI, Anthropic, Google, Meta Llama, Cohere, and any OpenAI-compatible API
- **Multiple Evaluation Metrics**: RAGAS Faithfulness, Context Precision, and RAGChecker Hallucination
- **Comprehensive Insights**: Claim-level analysis, context utilization, and diagnostic recommendations
- **Extensible Design**: Easy to add new metrics and providers
- **Flexible Output**: JSON, table, and detailed analysis formats

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Abisf/RAGTESTERCLI
cd RAGTESTERCLI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

**⚠️ Important Notes:**
- **Heavy Dependencies**: RAGChecker includes PyTorch, transformers, spaCy (~2GB+ download)
- **Installation Time**: First install may take 10-15 minutes due to ML dependencies
- **System Requirements**: Requires 4GB+ RAM for full functionality

**Installation Options:**
```bash
# Full installation (recommended)
pip install -r requirements.txt

# Minimal installation (no RAGChecker hallucination detection)
pip install -r requirements-minimal.txt

# Alternative: Install without heavy ML dependencies
pip install --no-deps ragchecker  # Limited functionality
```

3. Set up your API keys (optional - can also use CLI flags):
```bash
# Create .env file
cp .env.example .env
# Edit .env with your API keys

# Or set environment variables directly:
export OPENAI_API_KEY=sk-your-key-here
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Basic Usage

```bash
# OpenAI (Direct) - Most common
python cli.py test --input examples/perfect_faithfulness_test.json --metric faithfulness_ragas --llm-model gpt-4 --api-key YOUR_OPENAI_KEY

# Anthropic Claude (Direct)
python cli.py test --input examples/perfect_faithfulness_test.json --metric faithfulness_ragas --llm-model claude-3-haiku-20240307 --api-key YOUR_ANTHROPIC_KEY

# OpenRouter (100+ models including GPT, Claude, Llama, etc.)
python cli.py test --input examples/perfect_faithfulness_test.json --metric faithfulness_ragas --llm-model openai/gpt-3.5-turbo --api-key YOUR_OPENROUTER_KEY --api-base https://openrouter.ai/api/v1

# Context precision analysis (any provider)
python cli.py test --input examples/louvre_context_precision_test.json --metric context_precision_ragas --llm-model gpt-3.5-turbo --api-key YOUR_API_KEY

# Hallucination detection (any provider)
python cli.py test --input examples/hallucinated_test.json --metric hallucination_ragchecker --llm-model gpt-4 --api-key YOUR_API_KEY

# Using environment variables (no --api-key needed)
python cli.py test --input examples/perfect_faithfulness_test.json --metric faithfulness_ragas --llm-model gpt-4
```

## Supported Metrics

| Metric | Tool | Description | Enhanced Features |
|--------|------|-------------|------------------|
| `faithfulness` | RAGAS | Measures if answer is grounded in context | Step-by-step claim verification, Context usage analysis, Action recommendations |
| `context_precision` | RAGAS | Measures chunk relevance and ranking quality | Dual metrics (RAGAS + Simple), Precision@k calculations, Ranking quality analysis |
| `hallucination` | RAGChecker | Detects hallucinated information | Claim-level entailment analysis, Ground truth comparison, Pure invention detection |

## Scoring Metrics & Interpretation

### RAGAS Faithfulness Scoring

**Formula:** `Supported Claims / Total Claims`

| Score Range | Interpretation | Confidence Level | Action Required |
|-------------|----------------|------------------|-----------------|
| 0.9 - 1.0 | Excellent | High | None - System performing well |
| 0.7 - 0.89 | Good | Medium | Minor improvements recommended |
| 0.5 - 0.69 | Moderate | Medium | Review generation constraints |
| 0.3 - 0.49 | Poor | High | Significant improvements needed |
| 0.0 - 0.29 | Critical | High | Major overhaul required |

**Claim Classification:**
- **Supported**: Claim is directly supported by retrieved context
- **Unsupported**: Claim lacks sufficient context evidence
- **Contradicted**: Claim conflicts with retrieved context

### RAGAS Context Precision Scoring

**Dual Metrics System:**

1. **RAGAS Context Precision** (Ranking Quality)
   - **Formula:** `Σ(Precision@k × v_k) / Σ(v_k)`
   - **Measures:** How well relevant chunks are ranked early
   - **Range:** 0.0 - 1.0

2. **Simple Context Precision** (Retrieval Cleanliness)
   - **Formula:** `Relevant Chunks / Total Chunks`
   - **Measures:** Overall quality of retrieved content
   - **Range:** 0.0 - 1.0

| RAGAS Score | Simple Score | Interpretation | Ranking Quality | Noise Level | Action |
|-------------|--------------|----------------|-----------------|-------------|---------|
| 0.8+ | 0.8+ | Excellent | Good | Low | None |
| 0.8+ | <0.5 | Good ranking, noisy retrieval | Good | High | Prune irrelevant chunks |
| <0.5 | 0.8+ | Poor ranking, clean retrieval | Poor | Low | Improve ranking algorithm |
| <0.5 | <0.5 | Poor ranking, noisy retrieval | Poor | High | Overhaul retrieval system |

**Precision@k Calculation:**
- **Precision@1:** Relevance of first chunk
- **Precision@2:** Relevance of first two chunks
- **Precision@3:** Relevance of first three chunks
- **Weighted Average:** Early chunks weighted higher

### RAGChecker Hallucination Scoring

**Formula:** `np.mean(unfaithful) * 100`

| Score Range | Risk Level | Interpretation | Confidence | Action Required |
|-------------|------------|----------------|------------|-----------------|
| 0-10% | Low | Minimal hallucination | High | Monitor |
| 11-25% | Moderate | Some hallucination | Medium | Review prompts |
| 26-50% | High | Significant hallucination | High | Improve grounding |
| 51-75% | Critical | Major hallucination | High | Major fixes needed |
| 76-100% | Severe | Severe hallucination | High | Complete overhaul |

**Claim Classification:**
- **Correct Grounded**: Claim is both correct and supported by context
- **Hallucination**: Claim is not supported by context (regardless of correctness)
- **Context Supported but Incorrect**: Claim is supported by context but factually wrong
- **Missing Context Evidence**: Claim lacks supporting context

**Evaluation Modes:**
- **Context Only**: Evaluates against retrieved chunks only
- **Full Evaluation**: Evaluates against both context and ground truth

### Confidence Levels

| Level | Description | Reliability | Use Case |
|-------|-------------|-------------|----------|
| High | Strong evidence, clear patterns | 90%+ | Production decisions |
| Medium | Moderate evidence, some uncertainty | 70-89% | Development guidance |
| Low | Limited evidence, high uncertainty | <70% | Further investigation needed |

### Action Recommendations

**Faithfulness Actions:**
- **Improve Generation**: Enhance prompt engineering for better grounding
- **Review Constraints**: Adjust generation parameters
- **Critical Fix**: Major overhaul of generation pipeline

**Context Precision Actions:**
- **Prune**: Remove irrelevant chunks from retrieval
- **Rerank**: Improve ranking algorithm
- **Overhaul**: Complete retrieval system redesign

**Hallucination Actions:**
- **Monitor**: Continue current approach with observation
- **Review Prompts**: Enhance grounding instructions
- **Critical Fix**: Immediate intervention required
- **Complete Overhaul**: Fundamental system redesign

### Additional Metrics

**Claim-Level Metrics:**
- **Claim Density**: Claims per word in response
- **Average Claim Length**: Words per claim
- **Context Support Rate**: Percentage of claims with context support

**Context-Level Metrics:**
- **Noise Rate**: Percentage of irrelevant chunks
- **Early Precision**: Relevance of top-ranked chunks
- **Late Precision**: Relevance of lower-ranked chunks

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

## Scoring Metrics & Interpretation

### RAGAS Faithfulness

**What it measures:** How well the generated answer is supported by the retrieved context.

**Formula:** `Supported Claims / Total Claims`

**Score Interpretation:**
| Score Range | Level | Meaning | Action Required |
|-------------|-------|---------|-----------------|
| 0.9 - 1.0 | Excellent | Almost all claims supported | ✅ Maintain system |
| 0.7 - 0.89 | Good | Most claims supported | ⚠️ Minor improvements |
| 0.5 - 0.69 | Moderate | Some unsupported claims | 🔧 Improve generation |
| 0.3 - 0.49 | Poor | Many unsupported claims | 🚨 Major overhaul |
| 0.0 - 0.29 | Critical | Most claims unsupported | 🔥 Complete redesign |

**Example Calculation:**
```
Question: "What time does the Louvre open?"
Answer: "The Louvre opens at 9 AM and closes at 6 PM."
Context: ["The Louvre Museum is open 9 AM–6 PM daily."]

Claims: 2 (opening time, closing time)
Supported: 2 (both found in context)
Score: 2/2 = 1.0 (Perfect faithfulness)
```

**Action Recommendations:**
- **Improve Generation**: Focus on making responses more faithful to context
- **Enhance Retrieval**: Get better context to support claims
- **Maintain**: System is working well

---

### RAGAS Context Precision

**What it measures:** How well the retrieval system ranks relevant chunks early and filters out noise.

**Two Metrics:**

1. **RAGAS Context Precision** (Ranking Quality):
   - **Formula:** `Σ(Precision@k × v_k) / Σ(v_k)`
   - **Where:** `v_k = 1` for relevant chunks, `v_k = 0` for irrelevant chunks
   - **Measures:** "Did we rank relevant chunks early?"

2. **Simple Context Precision** (Noise Analysis):
   - **Formula:** `Relevant Chunks / Total Chunks`
   - **Measures:** "How much retrieved content is useful?"

**Detailed Calculation Example:**
```
Retrieved Chunks: [Relevant, Relevant, Irrelevant, Relevant, Irrelevant]
Precision@1: 1/1 = 1.0
Precision@2: 2/2 = 1.0  
Precision@3: 2/3 = 0.667
Precision@4: 3/4 = 0.75
Precision@5: 3/5 = 0.6

Weighted Calculation:
Position 1 (relevant): Precision@1 × 1 = 1.0
Position 2 (relevant): Precision@2 × 1 = 1.0
Position 4 (relevant): Precision@4 × 1 = 0.75
Final Score: (1.0 + 1.0 + 0.75) / 3 = 0.917

Simple Precision: 3 relevant / 5 total = 0.6
```

**Score Interpretation:**
| RAGAS Score | Simple Score | Meaning | Action Required |
|-------------|--------------|---------|-----------------|
| 0.9+ | 0.8+ | Excellent ranking & low noise | ✅ Maintain |
| 0.8+ | 0.5-0.7 | Good ranking, moderate noise | ⚠️ Reduce noise |
| 0.5-0.7 | 0.8+ | Poor ranking, low noise | 🔧 Improve ranking |
| <0.5 | <0.5 | Poor ranking & high noise | 🚨 Overhaul retrieval |

**Action Recommendations:**
- **Maintain**: Both ranking and noise are good
- **Reduce Noise**: Good ranking but too much irrelevant content
- **Improve Ranking**: Low noise but relevant chunks appear late
- **Overhaul Retrieval**: Both ranking and noise need fixing

---

### RAGChecker Hallucination

**What it measures:** How many claims in the response are pure inventions (not supported by context or ground truth).

**Formula:** `Hallucinated Claims / Total Claims × 100`

**Evaluation Modes:**
1. **Context-Only Mode** (no ground truth): Claims checked against retrieved context
2. **Full Evaluation Mode** (with ground truth): Claims checked against both context AND ground truth

**Score Interpretation:**
| Score Range | Risk Level | Meaning | Action Required |
|-------------|------------|---------|-----------------|
| 0-10% | Low | Minimal hallucinations | ✅ Monitor |
| 11-25% | Moderate | Some hallucinations | ⚠️ Improve generation |
| 26-50% | High | Many hallucinations | 🚨 Major fixes |
| 51%+ | Critical | Mostly hallucinations | 🔥 Complete overhaul |

**Claim Classifications:**
- **Correct Grounded**: Supported by context/ground truth
- **Correct Ungrounded**: True but not in context/ground truth
- **Incorrect Grounded**: False but supported by context/ground truth
- **Incorrect Ungrounded**: False and not supported (HALLUCINATION)

**Example Analysis:**
```
Question: "What is Tokyo's population?"
Answer: "Tokyo has 14 million people and beautiful cherry blossoms."
Ground Truth: "Tokyo has 14 million people."

Claims:
1. "Tokyo has 14 million people" → Correct Grounded ✅
2. "Tokyo has beautiful cherry blossoms" → Correct Ungrounded ⚠️

Hallucination Rate: 0/2 = 0% (no pure inventions)
```

**Action Recommendations:**
- **Monitor**: Low hallucination risk
- **Improve Generation**: Reduce unsupported claims
- **Enhance Retrieval**: Get better context to support claims
- **Complete Overhaul**: System generating mostly false claims

---

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

**The CLI supports ALL major LLM providers and models:**

```bash
# OpenAI Models (Direct API)
python cli.py test --input test.json --metric faithfulness_ragas --llm-model gpt-4 --api-key YOUR_OPENAI_KEY
python cli.py test --input test.json --metric faithfulness_ragas --llm-model gpt-3.5-turbo --api-key YOUR_OPENAI_KEY

# Anthropic Claude (Direct API)
python cli.py test --input test.json --metric faithfulness_ragas --llm-model claude-3-haiku-20240307 --api-key YOUR_ANTHROPIC_KEY
python cli.py test --input test.json --metric faithfulness_ragas --llm-model claude-3-sonnet-20240229 --api-key YOUR_ANTHROPIC_KEY

# Google Models (via OpenRouter)
python cli.py test --input test.json --metric faithfulness_ragas --llm-model google/gemini-pro --api-key YOUR_OPENROUTER_KEY --api-base https://openrouter.ai/api/v1

# Meta Llama Models (via OpenRouter)
python cli.py test --input test.json --metric faithfulness_ragas --llm-model meta-llama/llama-3.1-70b-instruct --api-key YOUR_OPENROUTER_KEY --api-base https://openrouter.ai/api/v1

# Cohere Models (via OpenRouter)
python cli.py test --input test.json --metric faithfulness_ragas --llm-model cohere/command-r-plus --api-key YOUR_OPENROUTER_KEY --api-base https://openrouter.ai/api/v1

# Any OpenAI-compatible API
python cli.py test --input test.json --metric faithfulness_ragas --llm-model custom-model --api-key YOUR_API_KEY --api-base https://your-custom-endpoint.com/v1
```

### Environment Variables

**Choose your preferred provider setup:**

```bash
# Option 1: OpenAI Direct (Most Common)
OPENAI_API_KEY=your_openai_key
RAGCLI_LLM_MODEL=gpt-4

# Option 2: Anthropic Direct  
ANTHROPIC_API_KEY=your_anthropic_key
RAGCLI_LLM_MODEL=claude-3-haiku-20240307

# Option 3: OpenRouter (Access to 100+ models)
OPENAI_API_KEY=your_openrouter_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
RAGCLI_LLM_MODEL=openai/gpt-3.5-turbo

# Option 4: Custom API Endpoint
OPENAI_API_KEY=your_custom_key
OPENAI_API_BASE=https://your-endpoint.com/v1
RAGCLI_LLM_MODEL=your-model-name
```

**Supported Environment Variables:**
- `OPENAI_API_KEY` - OpenAI API key (default, also used for OpenRouter)
- `ANTHROPIC_API_KEY` - Anthropic Claude API key (direct access)
- `OPENAI_API_BASE` - Custom API base URL (for OpenRouter, etc.)
- `RAGCLI_LLM_MODEL` - Default model to use

**Provider Detection:**
The CLI automatically detects the provider from the model name:
- `gpt-*` → `OPENAI_API_KEY`
- `claude-*` → `ANTHROPIC_API_KEY`
- Unknown models → `OPENAI_API_KEY` (default)

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
