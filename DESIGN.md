# Self-Evolving Benchmark Generator — Design Document

## 1. Executive Summary

The Self-Evolving Benchmark Generator is a closed-loop system that automatically generates AI benchmark questions, answers them, and evaluates performance—all using LLMs. The system "self-evolves" through:

1. **Progressive difficulty scaling** (1→10)
2. **Novelty enforcement** via prior question conditioning
3. **Performance tracking** with Exponential Moving Averages (EMA)
4. **Multi-dimensional analysis** across reasoning types and difficulty levels


## 2. System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELF-EVOLVING BENCHMARK LOOP                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│   │   GENERATION     │    │    ANSWERING     │    │   EVALUATION     │      │
│   │     MODEL        │───►│      MODEL       │───►│     MODEL        │      │
│   │                  │    │                  │    │  (LLM-as-Judge)  │      │
│   │  • Creates novel │    │  • Solves the    │    │  • Binary score  │      │
│   │    questions     │    │    question      │    │    (0 or 1)      │      │
│   │  • Targets       │    │                  │    │  • Rationale     │      │
│   │    difficulty    │    │                  │    │                  │      │
│   └────────┬─────────┘    └──────────────────┘    └────────┬─────────┘      │
│            │                                               │                │
│            │              ┌──────────────────┐             │                │
│            └─────────────►│  PERSISTENCE     │◄────────────┘                │
│                           │     LAYER        │                              │
│                           │                  │                              │
│                           │  • state.json    │                              │
│                           │  • runs.jsonl    │                              │
│                           │  • report_*.md   │                              │
│                           └──────────────────┘                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Component Breakdown

| Component | File | Responsibility |
|-----------|------|----------------|
| Orchestration Engine | `self_evolving_benchmark.py` | Main loop, difficulty progression, EMA calculation |
| LLM Client | `llm_client.py` | HTTP calls to OpenAI-compatible endpoints |
| Persistence Layer | `persistence_layer.py` | State management, run logs, question history |
| Reporting Layer | `reporting_layer.py` | Markdown report generation |


## 3. The "Self-Evolving" Mechanism

The benchmark evolves in three ways:

```
┌─────────────────────────────────────────────────────────────────────┐
│                       THREE AXES OF EVOLUTION                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. DIFFICULTY PROGRESSION                                          │
│     ────────────────────────                                        │
│     Questions automatically scale from easy (1) to expert (10)      │
│     based on explicit difficulty targeting in prompts               │
│                                                                     │
│  2. NOVELTY ENFORCEMENT                                             │
│     ────────────────────                                            │
│     Each new question is conditioned on ALL prior questions         │
│     for that reasoning type, preventing repetition                  │
│                                                                     │
│  3. PERFORMANCE TRACKING                                            │
│     ─────────────────────                                           │
│     EMA scores track improvement/degradation across:                │
│     • Overall performance                                           │
│     • Per reasoning type                                            │
│     • Per difficulty level                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 4. Key Pieces

### 4.1 Exponential Moving Average (EMA)

The EMA provides a smoothed performance signal that balances recent performance with historical trends:

```
EMA_new = α × current_score + (1 - α) × EMA_previous

Where:
  • α (alpha) = smoothing factor, typically 0.3
  • Higher α = more weight on recent performance
  • Lower α = more weight on historical performance
```

**Example with α = 0.3:**
```
Run 1: score = 0.8  → EMA = 0.8 (first run, EMA = score)
Run 2: score = 0.6  → EMA = 0.3×0.6 + 0.7×0.8 = 0.74
Run 3: score = 0.9  → EMA = 0.3×0.9 + 0.7×0.74 = 0.788
```

### 4.2 Novelty Selection (`select_focus`)

```python
def select_focus(reasoning_type, difficulty, prior_questions_for_type, prior_questions_in_cell):
    """
    Builds a prompt that ensures question novelty at two levels:

    1. CELL-LEVEL (same difficulty + reasoning_type):
       - MUST be completely different from these questions
       - Different scenarios, topics, entities, problem structures

    2. HISTORICAL (all prior questions for this reasoning type):
       - Avoid repetition of themes from other difficulty levels
       - Uses last 10 questions for context length management
    """
```

**Novelty Hierarchy:**
```
┌───────────────────────────────────────────────────────────────┐
│                     NOVELTY ENFORCEMENT                       │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  STRONGEST │ Cell-level questions (same difficulty)           │
│  CONSTRAINT│ "MUST be ENTIRELY NOVEL"                         │
│            │                                                  │
│            ▼                                                  │
│  MODERATE  │ Historical questions (other difficulties)        │
│  CONSTRAINT│ "avoid repetition"                               │
│            │                                                  │
│            ▼                                                  │
│  IMPLICIT  │ Difficulty progression hints                     │
│  CONSTRAINT│ "should be MORE DIFFICULT than previous"         │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 4.3 Difficulty Descriptors

The system uses semantic difficulty descriptions to guide question generation:

| Level | Descriptor | Expected Characteristics |
|-------|------------|-------------------------|
| 1-2 | Very easy, suitable for beginners | Basic recall, simple logic |
| 3-4 | Easy to moderate | Multi-step but straightforward |
| 5-6 | Moderate, requiring solid understanding | Requires domain knowledge |
| 7-8 | Challenging, requiring deep reasoning | Complex logic, edge cases |
| 9-10 | Very hard, expert-level complexity | Multi-domain synthesis, subtle constraints |


## 5. Three-Model Architecture

The system decouples three distinct cognitive tasks:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      THREE-MODEL ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐                                                │
│  │                 │  Purpose: Create novel, calibrated questions   │
│  │   GENERATION    │  Temperature: 0.8 (high creativity)            │
│  │      MODEL      │  Max tokens: 500                               │
│  │                 │  Focus: Creativity, diversity, difficulty      │
│  └─────────────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │                 │  Purpose: Solve the generated question         │
│  │    ANSWERING    │  Temperature: 0.5 (balanced)                   │
│  │      MODEL      │  Max tokens: 700                               │
│  │                 │  Focus: Accuracy, reasoning, completeness      │
│  └─────────────────┘                                                │
│           │                                                         │
│           ▼                                                         │
│  ┌─────────────────┐                                                │
│  │   EVALUATION    │  Purpose: Judge answer correctness             │
│  │     MODEL       │  Temperature: 0.3 (low variance)               │
│  │ (LLM-as-Judge)  │  Max tokens: 250                               │
│  │                 │  Focus: Consistency, objectivity               │
│  └─────────────────┘                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 6. Data Model

### 6.1 State Management

```
┌─────────────────────────────────────────────────────────────────────┐
│                             RunState                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ema: Optional[float]              # Overall EMA score              │
│  run_count: int                    # Number of completed runs       │
│  current_difficulty: int           # Resume point (1-10)            │
│  ema_by_reasoning_type: Dict       # Per-category EMA               │
│  ema_by_difficulty: Dict           # Per-level EMA                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Example Result

```
┌─────────────────────────────────────────────────────────────────────┐
│                           ExampleResult                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  question: str                     # The generated question         │
│  tags: {                           # Metadata                       │
│      reasoning_type: str,          #   Category of reasoning        │
│      difficulty: int,              #   1-10 scale                   │
│      question_index: int           #   Position in cell             │
│  }                                                                  │
│  answer: str                       # Model's answer                 │
│  score: float                      # 0.0 or 1.0 (binary)            │
│  rationale: str                    # Judge's explanation            │
│  created_at: str                   # ISO timestamp                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 7. Round-Robin Execution Model

The system uses a round-robin approach that ensures balanced coverage:

```
┌───────────────────────────────────────────────────────────────────┐
│                        ROUND-ROBIN MATRIX                         │
├───────────────────────────────────────────────────────────────────┤
│                                                                   │
│         │ logical │ math │ common │ reading │ abstract │ ... │    │
│         │ deduct. │ reas │ sense  │ compr.  │ analogy  │     │    │
│  ───────┼─────────┼──────┼────────┼─────────┼──────────┼─────┤    │
│  Diff 1 │   Q1    │  Q2  │   Q3   │   Q4    │    Q5    │ ... │    │
│  Diff 2 │   Q9    │ Q10  │  Q11   │  Q12    │   Q13    │ ... │    │
│  Diff 3 │  Q17    │ Q18  │  Q19   │  Q20    │   Q21    │ ... │    │
│   ...   │   ...   │ ...  │  ...   │   ...   │   ...    │ ... │    │
│  Diff10 │  Q73    │ Q74  │  Q75   │  Q76    │   Q77    │ ... │    │
│  ───────┴─────────┴──────┴────────┴─────────┴──────────┴─────┘    │
│                                                                   │
│  Default: 8 reasoning types × 10 difficulties × 1 question = 80   │
│  With --num-questions 3: 8 × 10 × 3 = 240 questions               │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```


## 8. LLM-as-Judge Evaluation

The evaluation model acts as a binary classifier with explanations:

```
┌─────────────────────────────────────────────────────────────────────┐
│                          EVALUATION PROTOCOL                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  INPUT:                                                             │
│    • Question text                                                  │
│    • Model's answer                                                 │
│                                                                     │
│  OUTPUT (JSON):                                                     │
│    {                                                                │
│      "score": "correct" | "incorrect",                              │
│      "rationale": "Explanation of why..."                           │
│    }                                                                │
│                                                                     │
│  SCORING:                                                           │
│    • "correct" → 1.0                                                │
│    • "incorrect" → 0.0                                              │
│    • Parse failure → 0.0 (conservative fallback)                    │
│                                                                     │
│  DESIGN CHOICE: Binary scoring for:                                 │
│    ✓ Clear pass/fail metrics                                        │
│    ✓ Reduced subjectivity                                           │
│    ✓ Easier trend analysis                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```


## 9. Persistence & Resumption

The system is designed for fault tolerance:

```
┌───────────────────────────────────────────────────────────────────────┐
│                         PERSISTENCE STRATEGY                          │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  STATE SAVE POINTS:                                                   │
│    • After completing each difficulty level                           │
│    • Stores current_difficulty for resumption                         │
│                                                                       │
│  FILES:                                                               │
│    out_benchmark/                                                     │
│    ├── state.json          # Current EMA, run count, next difficulty  │
│    ├── runs.jsonl          # Append-only log of all runs              │
│    └── report_run_*.md     # Human-readable per-run reports           │
│                                                                       │
│  RESUMPTION BEHAVIOR:                                                 │
│    1. Load state.json                                                 │
│    2. Read current_difficulty (e.g., 5)                               │
│    3. Continue from difficulty 5 onward                               │
│    4. Load prior questions for novelty from runs.jsonl                │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```


## 10. Default Reasoning Type Categories

The default 8 reasoning types cover diverse cognitive skills:

| Category | Description | Example Question Type |
|----------|-------------|----------------------|
| `logical_deduction` | Formal logic, syllogisms, puzzles | "If A implies B and B implies C..." |
| `mathematical_reasoning` | Numerical problems, proofs | "Calculate the probability that..." |
| `commonsense_reasoning` | Everyday knowledge, intuition | "Why would someone bring an umbrella..." |
| `reading_comprehension` | Text analysis, inference | "Based on the passage, what can be inferred..." |
| `abstraction_analogy` | Pattern recognition, metaphors | "Tree is to forest as drop is to..." |
| `scientific_reasoning` | Hypothesis, experimentation | "Design an experiment to test..." |
| `data_interpretation` | Charts, statistics, trends | "Given the following data table..." |
| `computer_programming` | Code analysis, algorithms | "What is the output of this function..." |


## 11. Workflow Diagram

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │   Load state.json     │
                    │   (or initialize)     │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Load runs.jsonl      │
                    │  (prior questions)    │
                    └───────────┬───────────┘
                                │
                                ▼
        ┌───────────────────────────────────────────────┐
        │         FOR difficulty = start → end          │
        │  ┌────────────────────────────────────────┐   │
        │  │    FOR each reasoning_type             │   │
        │  │  ┌─────────────────────────────────┐   │   │
        │  │  │  FOR q = 1 → num_questions      │   │   │
        │  │  │                                 │   │   │
        │  │  │  ┌──────────────────────────┐   │   │   │
        │  │  │  │ 1. Generate question     │   │   │   │
        │  │  │  │    (with novelty focus)  │   │   │   │
        │  │  │  └──────────┬───────────────┘   │   │   │
        │  │  │             ▼                   │   │   │
        │  │  │  ┌──────────────────────────┐   │   │   │
        │  │  │  │ 2. Answer question       │   │   │   │
        │  │  │  └──────────┬───────────────┘   │   │   │
        │  │  │             ▼                   │   │   │
        │  │  │  ┌──────────────────────────┐   │   │   │
        │  │  │  │ 3. Evaluate answer       │   │   │   │
        │  │  │  │    (LLM-as-judge)        │   │   │   │
        │  │  │  └──────────┬───────────────┘   │   │   │
        │  │  │             ▼                   │   │   │
        │  │  │  ┌──────────────────────────┐   │   │   │
        │  │  │  │ 4. Record result         │   │   │   │
        │  │  │  └──────────────────────────┘   │   │   │
        │  │  │                                 │   │   │
        │  │  └─────────────────────────────────┘   │   │
        │  └────────────────────────────────────────┘   │
        │                     │                         │
        │                     ▼                         │
        │         ┌────────────────────────┐            │
        │         │ Update difficulty EMA  │            │
        │         │ Save state.json        │            │
        │         └────────────────────────┘            │
        └───────────────────────────────────────────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Update overall EMA    │
                    │ Update per-RT EMA     │
                    │ Append to runs.jsonl  │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌───────────────────────┐
                    │ Generate report_*.md  │
                    └───────────┬───────────┘
                                │
                                ▼
                               END
```


## 12. Extension Points

The architecture supports several natural extensions:

1. **Adaptive Difficulty**: Use EMA trends to dynamically adjust difficulty progression
2. **Weak-Area Focus**: Generate more questions for low-EMA reasoning types
3. **Multi-Judge Ensemble**: Use multiple evaluation models and aggregate scores
4. **Question Quality Filter**: Add a validation step before answering
5. **Human-in-the-Loop**: Allow human override of evaluation scores


## 13. Summary

The Self-Evolving Benchmark Generator creates a **closed-loop evaluation system** where:

- **Questions automatically scale in difficulty** through explicit prompting
- **Novelty is enforced** by conditioning on prior questions
- **Performance is tracked** via multi-dimensional EMA
- **The system is resumable** through persistent state management
- **Three-model architecture** separates generation, solving, and judging concerns

This design enables continuous, automated assessment of LLM capabilities across diverse reasoning domains with minimal human intervention.
