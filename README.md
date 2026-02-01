# Self-Evolving Benchmark Generator

A Python-based system that automatically generates, evaluates, and tracks performance on AI benchmark questions. It creates questions across multiple reasoning types and difficulty levels, answers them using an LLM, and evaluates the answers while tracking performance metrics over time.


## Features

### Multi-Model & Multi-Endpoint Support
- **Separate models** for each task: generation, answering, and evaluation
- **Separate endpoints** for each model: use different API providers for each task
- Allows using a stronger model for question generation/evaluation and a faster model for answering
- Uses `POST /v1/chat/completions` (OpenAI-compatible)

### Round-Robin Difficulty Progression
- Questions are generated across **10 difficulty levels** (1 = easiest, 10 = hardest)
- For each difficulty level, the system cycles through **all reasoning types**
- Supports **continuation**: if interrupted, resumes from the last completed difficulty level

### Reasoning Types (8 default categories)
Customizable via `--reasoning-types`. Default types:
- `logical_deduction`
- `mathematical_reasoning`
- `commonsense_reasoning`
- `reading_comprehension`
- `abstraction_analogy`
- `scientific_reasoning`
- `data_interpretation`
- `computer_programming`

### Performance Tracking via Exponential Moving Average (EMA)
- **Overall EMA**: Tracks global performance across all runs
- **Per-reasoning-type EMA**: Identifies weak areas by reasoning category
- **Per-difficulty EMA**: Tracks performance at each difficulty level

### Reporting
- Console summary with reasoning type and difficulty breakdowns
- Per-run markdown report: `report_run_XXXX.md`


## Best uses for this self-Evolving Benchmark

This benchmark is best suited for flexible, iterative, and model-agnostic evaluation scenarios:

- No predefined or existing dataset is required for testing.
- Any `reasoning_type` can be defined and evaluated via configuration parameters.
- Evaluations can be run in small batches and resumed later without loss of continuity.

To get the most reliable and informative results:

1. Use a stronger, more capable model for **question generation** and **evaluation**.
2. Use a **domain-specific fine-tuned model** for both question generation and evaluation when assessing specialized knowledge or reasoning.


## Repository Contents

```
self_evolving_benchmark/
├── self_evolving_benchmark.py  # Core orchestration engine
├── llm_client.py               # OpenAI API-compatible HTTP client
├── persistence_layer.py        # File-based state management (JSON, JSONL)
├── reporting_layer.py          # Markdown report generation
├── requirements.txt            # Python dependencies
└── out_benchmark/              # Output directory (created at runtime)
    ├── runs.jsonl              # Append-only run log
    ├── state.json              # Global state (EMA, current difficulty)
    └── report_run_*.md         # Per-run reports
```


## Setup

### 1) Create a virtualenv (recommended)

```bash
python -m venv benchmark-env
source ./benchmark-env/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```


## Usage

### Single Endpoint for All Models

```bash
python self_evolving_benchmark.py \
  --base-url "https://api.openai.com" \
  --api-key "YOUR_KEY" \
  --model-generation "gpt-4o" \
  --model-answering "gpt-4o-mini" \
  --model-evaluation "gpt-4o" \
  --n 3 \
  --alpha 0.3 \
  --out-dir out_benchmark
```

### Different Endpoints per Model

Use different API providers for each task (e.g., OpenAI for generation/evaluation, local model for answering):

```bash
python self_evolving_benchmark.py \
  --base-url-generation "https://api.openai.com" \
  --base-url-answering "http://localhost:8000" \
  --base-url-evaluation "https://api.openai.com" \
  --api-key "YOUR_KEY" \
  --model-generation "gpt-4o" \
  --model-answering "local-llama-70b" \
  --model-evaluation "gpt-4o" \
  --n 3
```

### Using the Same Model for All Tasks

If you want to use the same model for all tasks, simply specify it for each parameter:

```bash
python self_evolving_benchmark.py \
  --base-url "https://api.openai.com" \
  --api-key "YOUR_KEY" \
  --model-generation "gpt-4o-mini" \
  --model-answering "gpt-4o-mini" \
  --model-evaluation "gpt-4o-mini" \
  --n 5
```

### Using Custom Reasoning Types

You can specify a custom subset of reasoning types to focus on specific areas:

```bash
python self_evolving_benchmark.py \
  --base-url "https://api.openai.com" \
  --api-key "YOUR_KEY" \
  --model-generation "gpt-4o" \
  --model-answering "gpt-4o-mini" \
  --model-evaluation "gpt-4o" \
  --reasoning-types "logical_deduction,mathematical_reasoning,computer_programming" \
  --n 5
```


## CLI Arguments

| Argument | Description | Required | Default |
|----------|-------------|----------|---------|
| `--base-url` | Default OpenAI-compatible API endpoint for all models | No* | - |
| `--base-url-generation` | API endpoint for question generation model | No* | `--base-url` |
| `--base-url-answering` | API endpoint for answering model | No* | `--base-url` |
| `--base-url-evaluation` | API endpoint for evaluation model | No* | `--base-url` |
| `--api-key` | API key for authentication | Yes | - |
| `--model-generation` | Model for question generation | Yes | - |
| `--model-answering` | Model for answering questions | Yes | - |
| `--model-evaluation` | Model for evaluation/judging | Yes | - |
| `--n` | Number of difficulty rounds to run (1-10) | No | 10 |
| `--num-questions` | Questions per (difficulty, reasoning_type) cell | No | 1 |
| `--alpha` | EMA smoothing factor (0-1) | No | 0.3 |
| `--out-dir` | Output directory path | No | `out_benchmark` |
| `--reasoning-types` | Comma-separated list of reasoning types | No | See below |
| `--log-level` | Logging level (DEBUG, INFO, WARNING, ERROR) | No | INFO |

\* At least `--base-url` or all three `--base-url-*` arguments must be provided.


## Execution Flow

1. **Load state**: Resume from `current_difficulty` (allows continuation)
2. **Determine range**: Run from `current_difficulty` to `min(current + n - 1, 10)`
3. **Round-robin generation**: For each difficulty level:
   - Iterate through all specified reasoning types (default: 8 types)
   - For each (difficulty, reasoning_type) cell, generate N questions (default: 1)
   - Answer each question using the answering model
   - Evaluate each answer using the evaluation model (0 or 1 score)
   - Average scores within each cell for EMA calculations
4. **Update EMAs**: Per-difficulty, per-reasoning-type, and overall
5. **Persist state**: Save to `state.json` after each difficulty round
6. **Generate report**: Create markdown report with results

### Example: Running in Multiple Sessions

```bash
# First session: Run difficulty levels 1-3
python self_evolving_benchmark.py --base-url ... --n 3

# Second session: Automatically continues with levels 4-6
python self_evolving_benchmark.py --base-url ... --n 3

# Third session: Continues with levels 7-9
python self_evolving_benchmark.py --base-url ... --n 3

# Fourth session: Completes level 10
python self_evolving_benchmark.py --base-url ... --n 3
```

### Example: Multiple Questions per Cell

Generate 3 questions for each (difficulty, reasoning_type) combination:

```bash
python self_evolving_benchmark.py \
  --base-url "https://api.openai.com" \
  --api-key "YOUR_KEY" \
  --model-generation "gpt-4o" \
  --model-answering "gpt-4o-mini" \
  --model-evaluation "gpt-4o" \
  --num-questions 3 \
  --n 2
```

This generates: 2 difficulties × 8 reasoning types × 3 questions = 48 total questions.


## Output Artifacts

### `runs.jsonl`

Each line is one run record containing:
- `run_id`, `started_at`, `ended_at`
- `difficulty_start`, `difficulty_end`, `num_questions_per_cell`
- `endpoint_generation`, `endpoint_answering`, `endpoint_evaluation`: API endpoints used
- `model_generation`, `model_answering`, `model_evaluation`: Models used
- `reasoning_types`: List of reasoning types used in this run
- `run_mean_score`, `ema_score`
- List of examples with: `question`, `tags`, `answer`, `score`, `rationale`, `created_at`

### `state.json`

Stores:
- `run_count`: Total number of completed runs
- `current_difficulty`: Next difficulty level to run (1-11, where 11 means complete)
- `ema`: Overall EMA score
- `ema_by_reasoning_type`: EMA per reasoning category
- `ema_by_difficulty`: EMA per difficulty level

### `report_run_XXXX.md`

Per-run markdown report containing:
- Run metadata (models used, difficulty range, timestamps)
- Overall and EMA scores
- Reasoning type performance breakdown
- Difficulty level performance breakdown
- Example previews with questions, answers, and evaluations


## Evaluation (LLM-as-Judge)

The evaluator uses a rubric-based LLM prompt with binary scoring:
- **Score 1.0**: Correct answer
- **Score 0.0**: Incorrect or non-answer

Output is parsed from JSON: `{ "score": number, "rationale": "..." }`


## EMA Scoring

For each metric (overall, per-reasoning-type, per-difficulty):

```
EMA_new = alpha * current_mean + (1 - alpha) * EMA_prev
```

- On the first update, `EMA_new = current_mean`
- Higher alpha = EMA reacts faster to recent performance
- Lower alpha = EMA is smoother and more stable


## Difficulty Progression

Questions are generated with explicit difficulty targeting:

| Difficulty | Description |
|------------|-------------|
| 1-2 | Very easy, suitable for beginners |
| 3-4 | Easy to moderate |
| 5-6 | Moderate, requiring solid understanding |
| 7-8 | Challenging, requiring deep reasoning |
| 9-10 | Very hard, expert-level complexity |

The system instructs the LLM to make questions progressively harder by adding:
- More steps
- Subtle constraints
- Edge cases
- Deeper analysis requirements


## Design Notes

Details related to the design of this project can be found in the [Design-Specific Document](DESIGN.md), but here is an overview of the main design decisions:

- **Three separate models**: Allows optimization for cost/quality tradeoffs
- **Per-model endpoints**: Each model can use a different API provider (e.g., OpenAI, local, Azure)
- **Round-robin approach**: Ensures balanced coverage across all reasoning types
- **Difficulty progression**: Systematic increase in complexity from 1 to 10
- **Continuation support**: State persisted after each difficulty round
- **Binary evaluation**: Simple 0/1 scoring for clear pass/fail metrics


## Limitations & Future Work

- The evaluation step/agent uses a single LLM judge (no multi-judge aggregation)
- Support only for models with the endpoint `/v1/chat/completions` (OpenAI-compatible endpoints)
- No embedding-based novelty checking (relies on LLM diversity)
- Add a test-only mode, which can use a previously created dataset to evaluate an LLM.
