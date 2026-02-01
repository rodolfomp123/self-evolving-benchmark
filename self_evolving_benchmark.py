#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

from llm_client import chat_completions
from persistence_layer import (
    ensure_dir,
    load_state,
    runs_path,
    save_state,
    append_run_record,
    load_questions_by_reasoning_type,
    state_path,
)
from reporting_layer import (
    format_reasoning_type_report,
    write_run_report_md,
)

# -------------------------
# Constants
# -------------------------

# Generation settings
MAX_QUESTION_LENGTH = 1800

# LLM temperature settings
TEMPERATURE_GENERATION = 0.8
TEMPERATURE_ANSWERING = 0.5
TEMPERATURE_EVALUATION = 0.3

# Token limits
MAX_TOKENS_GENERATION = 500
MAX_TOKENS_ANSWERING = 700
MAX_TOKENS_EVALUATION = 250

# Difficulty settings
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 10

# Default reasoning types
ALLOWED_REASONING = [
    "logical_deduction",
    "mathematical_reasoning",
    "commonsense_reasoning",
    "reading_comprehension",
    "abstraction_analogy",
    "scientific_reasoning",
    "data_interpretation",
    "computer_programming",
]

logger = logging.getLogger(__name__)

# -------------------------
# Data Structures
# -------------------------

@dataclasses.dataclass
class ExampleResult:
    question: str
    tags: Dict[str, Any]
    answer: str
    score: float
    rationale: str
    created_at: str

# -------------------------
# Novelty & Self-evolution
# -------------------------

def select_focus(
    reasoning_type: str,
    difficulty: int,
    prior_questions_for_type: List[str],
    prior_questions_in_cell: List[str],
) -> str:
    """
    Build a focus prompt for question generation based on reasoning type, difficulty,
    and prior questions generated for this reasoning type.

    Args:
        reasoning_type: The target reasoning type for the question.
        difficulty: Difficulty level from 1 (easiest) to 10 (hardest).
        prior_questions_for_type: Previously generated questions for this reasoning type (across all difficulties).
        prior_questions_in_cell: Questions already generated in this specific (difficulty, reasoning_type) cell.

    Returns:
        A prompt string guiding the LLM to generate an appropriate question.
    """
    # Difficulty descriptors
    if difficulty <= 2:
        diff_desc = "very easy, suitable for beginners"
    elif difficulty <= 4:
        diff_desc = "easy to moderate"
    elif difficulty <= 6:
        diff_desc = "moderate, requiring solid understanding"
    elif difficulty <= 8:
        diff_desc = "challenging, requiring deep reasoning"
    else:
        diff_desc = "very hard, expert-level complexity"

    base = (
        f"Generate a benchmark question that tests '{reasoning_type}' reasoning.\n"
        f"Target difficulty: {difficulty}/10 ({diff_desc}).\n\n"
    )

    # Handle cell-level novelty (questions at the same difficulty level)
    if prior_questions_in_cell:
        base += (
            "Questions already generated at THIS EXACT difficulty level (MUST be completely different from these):\n"
            + "\n".join([f"- {q}" for q in prior_questions_in_cell])
            + "\n\n"
            "CRITICAL: Your question must be ENTIRELY NOVEL compared to the above.\n"
            "Use completely different scenarios, topics, entities, and problem structures.\n"
            "Do NOT create variations or paraphrases of the above questions.\n\n"
        )

    # Handle historical questions across all difficulties
    if prior_questions_for_type:
        # Filter out cell questions to avoid duplication in the prompt
        historical_questions = [q for q in prior_questions_for_type if q not in prior_questions_in_cell]
        if historical_questions:
            base += (
                "Prior questions for this reasoning type at other difficulty levels (avoid repetition):\n"
                + "\n".join([f"- {q}" for q in historical_questions[-10:]])  # Limit to last 10 for context length
                + "\n\n"
            )

    if not prior_questions_for_type and not prior_questions_in_cell:
        base += (
            "This is the first question for this reasoning type.\n"
            "Focus on creating a clear, self-contained problem.\n"
        )

    # Add difficulty progression hints
    if difficulty > 1 and prior_questions_for_type:
        base += (
            f"\nIMPORTANT: This question should be MORE DIFFICULT than previous ones.\n"
            f"Add complexity through: more steps, subtle constraints, edge cases, "
            f"and/or requiring deeper analysis.\n"
        )

    return base


def generate_question(
    base_url: str,
    api_key: str,
    model: str,
    reasoning_type: str,
    difficulty: int,
    prior_questions_for_type: List[str],
    prior_questions_in_cell: List[str],
) -> str:
    """
    Generates one novel question for a specific reasoning type and difficulty.

    Args:
        base_url: OpenAI-compatible API base URL.
        api_key: API authentication key.
        model: Model identifier to use.
        reasoning_type: The target reasoning type for the question.
        difficulty: Difficulty level from 1 (easiest) to 10 (hardest).
        prior_questions_for_type: Previously generated questions for this reasoning type (across all difficulties).
        prior_questions_in_cell: Questions already generated in this specific (difficulty, reasoning_type) cell.

    Returns:
        The generated question text.
    """
    focus = select_focus(reasoning_type, difficulty, prior_questions_for_type, prior_questions_in_cell)

    system = (
        "You generate high-utility benchmark questions that test machine cognition. "
        "Each question must be self-contained, unambiguous, and answerable without external tools. "
        "Avoid trivia and avoid asking for personal opinions."
    )

    user = (
        f"{focus}\n\n"
        "Constraints:\n"
        "- Produce ONE question only.\n"
        "- Must be novel vs prior questions (if they exist).\n"
        "- Include any necessary rules, data, or examples inside the question.\n"
        f"- Keep it reasonably short (<= {MAX_QUESTION_LENGTH} chars).\n\n"
        "Return only the question text (no quotes, no labels)."
    )

    return chat_completions(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE_GENERATION,
        max_tokens=MAX_TOKENS_GENERATION,
    ).strip()


# -------------------------
# Answering & Evaluation
# -------------------------

def answer_question(base_url: str, api_key: str, model: str, question: str) -> str:
    """
    Generate an answer to a benchmark question using the specified LLM.

    Args:
        base_url: OpenAI-compatible API base URL.
        api_key: API authentication key.
        model: Model identifier to use.
        question: The question to answer.

    Returns:
        The model's answer to the question.
    """
    system = (
        "You are solving a benchmark question. "
        "Be correct and concise. "
        "If constraints exist, follow them."
    )
    user = f"Question:\n{question}\n\nAnswer:"
    return chat_completions(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE_ANSWERING,
        max_tokens=MAX_TOKENS_ANSWERING,
    ).strip()


def extract_json(text: str) -> str:
    """
    Extract the first JSON object from a text blob.
    Minimal helper to survive occasional extra tokens.
    """
    t = text.strip()
    if t.startswith("{") and t.endswith("}"):
        return t
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        return t[start : end + 1]
    return t


def evaluate_answer(base_url: str, api_key: str, model: str, question: str, answer: str) -> Tuple[float, str]:
    """
    LLM-as-judge.

    Args:
        base_url: OpenAI-compatible API base URL.
        api_key: API authentication key.
        model: Model identifier to use.
        question: The benchmark question.
        answer: The model's answer to evaluate.

    Returns:
        The score in 0 or 1 and the rationale for the evaluation.
    """
    system = (
        "You are a strict evaluator for benchmark answers. "
        "Score correctness of the answer given the question. "
        "Be consistent across examples."
    )
    rubric = (
        "Scoring rubric:\n"
        "- correct\n"
        "- incorrect\n"
    )
    user = (
        "Return ONLY valid JSON: {\"score\": string, \"rationale\": \"...\"}\n\n"
        f"{rubric}\n"
        "Question:\n"
        f"{question}\n\n"
        "Model answer:\n"
        f"{answer}\n"
    )

    raw = chat_completions(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=TEMPERATURE_EVALUATION,
        max_tokens=MAX_TOKENS_EVALUATION,
    )

    try:
        data = json.loads(extract_json(raw))
        score = 1.0 if data.get("score", "incorrect") == "correct" else 0.0
        rationale = data.get("rationale", "").strip()
        return score, rationale
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fallback if JSON parsing fails: treat as 0 with rationale
        return 0.0, f"Judge output parse failed. Raw: {raw[:300]}"

# -------------------------
# EMA
# -------------------------

def update_ema(prev: Optional[float], current: float, alpha: float) -> float:
    """
    Update exponential moving average.

    Args:
        prev: Previous EMA value, or None if first update.
        current: Current value to incorporate.
        alpha: Smoothing factor in (0, 1). Higher = more weight on current.

    Returns:
        Updated EMA value.
    """
    if prev is None:
        return current
    return alpha * current + (1.0 - alpha) * prev

# -------------------------
# Main run
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Self-evolving benchmark generator.")
    parser.add_argument("--base-url", help="Default OpenAI-compatible base URL for all models (can be overridden per-model)")
    parser.add_argument("--base-url-generation", help="Base URL for question generation model")
    parser.add_argument("--base-url-answering", help="Base URL for answering model")
    parser.add_argument("--base-url-evaluation", help="Base URL for evaluation model")
    parser.add_argument("--api-key", required=True, help="API key (or dummy if your endpoint ignores it)")
    parser.add_argument("--model-generation", type=str, required=True, help="Model for question generation")
    parser.add_argument("--model-answering", type=str, required=True, help="Model for answering questions")
    parser.add_argument("--model-evaluation", type=str, required=True, help="Model for evaluation/judging")
    parser.add_argument("--n", type=int, default=MAX_DIFFICULTY, help=f"Number of difficulty rounds to run (1-{MAX_DIFFICULTY}). Continues from last completed difficulty.")
    parser.add_argument("--num-questions", type=int, default=1, help="Number of questions to generate per (difficulty, reasoning_type) combination. Default: 1")
    parser.add_argument("--alpha", type=float, default=0.3, help="EMA smoothing factor (0-1)")
    parser.add_argument("--out-dir", type=str, default="out_benchmark", help="Output directory for state/artifacts")
    parser.add_argument("--reasoning-types", type=str, default=None, help=f"Comma-separated list of reasoning types to use. Defaults to: {', '.join(ALLOWED_REASONING)}")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Validate n is within bounds
    if args.n < 1 or args.n > MAX_DIFFICULTY:
        logger.error(f"--n must be between 1 and {MAX_DIFFICULTY}")
        return

    # Validate num_questions
    if args.num_questions < 1:
        logger.error("--num-questions must be at least 1")
        return

    num_questions_per_cell = args.num_questions

    # Parse reasoning types
    if args.reasoning_types:
        reasoning_types = [rt.strip() for rt in args.reasoning_types.split(",") if rt.strip()]
        if not reasoning_types:
            logger.error("--reasoning-types cannot be empty")
            return
        logger.info(f"Using custom reasoning types: {reasoning_types}")
    else:
        reasoning_types = ALLOWED_REASONING
        logger.info(f"Using default reasoning types: {reasoning_types}")

    # Resolve base URLs for each model (specific URL takes precedence over default)
    base_url_generation = args.base_url_generation or args.base_url
    base_url_answering = args.base_url_answering or args.base_url
    base_url_evaluation = args.base_url_evaluation or args.base_url

    # Validate that we have URLs for all models
    if not base_url_generation:
        logger.error("No base URL for generation model. Provide --base-url or --base-url-generation")
        return
    if not base_url_answering:
        logger.error("No base URL for answering model. Provide --base-url or --base-url-answering")
        return
    if not base_url_evaluation:
        logger.error("No base URL for evaluation model. Provide --base-url or --base-url-evaluation")
        return

    ensure_dir(args.out_dir)

    # Get models for each task
    model_generation = args.model_generation
    model_answering = args.model_answering
    model_evaluation = args.model_evaluation

    logger.info(f"Models - generation: {model_generation}, answering: {model_answering}, evaluation: {model_evaluation}")
    logger.info(f"Endpoints - generation: {base_url_generation}, answering: {base_url_answering}, evaluation: {base_url_evaluation}")

    # Load history
    state = load_state(args.out_dir)
    questions_by_rt = load_questions_by_reasoning_type(args.out_dir)

    # Determine difficulty range for this run
    start_difficulty = state.current_difficulty
    end_difficulty = min(start_difficulty + args.n - 1, MAX_DIFFICULTY)

    if start_difficulty > MAX_DIFFICULTY:
        logger.info(f"All {MAX_DIFFICULTY} difficulty levels already completed. Nothing to do.")
        return

    started_at = dt.datetime.now(dt.UTC).isoformat()
    logger.info(
        f"Run started: starting at difficulty {start_difficulty}, "
        f"running through difficulty {end_difficulty}, prev_ema={state.ema}"
    )

    all_examples: List[Dict[str, Any]] = []
    all_scores: List[float] = []
    scores_by_rt: Dict[str, List[float]] = {}
    scores_by_difficulty: Dict[int, List[float]] = {}

    num_reasoning_types = len(reasoning_types)
    total_questions = (end_difficulty - start_difficulty + 1) * num_reasoning_types * num_questions_per_cell
    question_num = 0

    # Round-robin: for each difficulty level, iterate through all reasoning types
    for difficulty in range(start_difficulty, end_difficulty + 1):
        difficulty_scores: List[float] = []

        for reasoning_type in reasoning_types:
            # Generate multiple questions for this (difficulty, reasoning_type) cell
            cell_scores: List[float] = []
            cell_questions: List[str] = []  # Track questions within this cell for novelty

            for q_idx in range(num_questions_per_cell):
                question_num += 1

                # Get prior questions for this reasoning type
                prior_questions = questions_by_rt.get(reasoning_type, [])

                # Generate question for this reasoning type and difficulty
                q = generate_question(
                    base_url=base_url_generation,
                    api_key=args.api_key,
                    model=model_generation,
                    reasoning_type=reasoning_type,
                    difficulty=difficulty,
                    prior_questions_for_type=prior_questions,
                    prior_questions_in_cell=cell_questions,
                )

                # Answer and evaluate
                a = answer_question(base_url_answering, args.api_key, model_answering, q)
                s, r = evaluate_answer(base_url_evaluation, args.api_key, model_evaluation, q, a)

                # Track scores
                cell_scores.append(s)
                all_scores.append(s)

                # Build example record
                tags = {"reasoning_type": reasoning_type, "difficulty": difficulty, "question_index": q_idx + 1}
                ex = ExampleResult(
                    question=q,
                    tags=tags,
                    answer=a,
                    score=s,
                    rationale=r,
                    created_at=dt.datetime.now(dt.UTC).isoformat(),
                )
                all_examples.append(dataclasses.asdict(ex))

                # Update local tracking for diversity in subsequent questions
                questions_by_rt.setdefault(reasoning_type, []).append(q)
                cell_questions.append(q)  # Track for cell-level novelty

                logger.info(
                    f"[{question_num}/{total_questions}] difficulty={difficulty} "
                    f"reasoning_type={reasoning_type} q={q_idx + 1}/{num_questions_per_cell} score={s:.2f} "
                    f"question_preview={q[:50].replace(chr(10), ' ')}..."
                )

            # Average the scores for this cell and add to tracking
            cell_mean = sum(cell_scores) / len(cell_scores)
            difficulty_scores.append(cell_mean)
            scores_by_rt.setdefault(reasoning_type, []).append(cell_mean)

        # Update EMA for this difficulty round
        difficulty_mean = sum(difficulty_scores) / len(difficulty_scores)
        scores_by_difficulty[difficulty] = difficulty_scores
        prev_diff_ema = state.ema_by_difficulty.get(difficulty)
        state.ema_by_difficulty[difficulty] = update_ema(prev_diff_ema, difficulty_mean, args.alpha)

        logger.info(f"Completed difficulty {difficulty}: mean_score={difficulty_mean:.4f}")

        # Update current_difficulty after completing each round
        state.current_difficulty = difficulty + 1
        save_state(args.out_dir, state)

    # Compute overall statistics
    run_mean = sum(all_scores) / max(1, len(all_scores))
    new_ema = update_ema(state.ema, run_mean, args.alpha)

    # Update per-reasoning-type EMA
    for rt, rt_scores in scores_by_rt.items():
        rt_mean = sum(rt_scores) / len(rt_scores)
        prev_rt_ema = state.ema_by_reasoning_type.get(rt)
        state.ema_by_reasoning_type[rt] = update_ema(prev_rt_ema, rt_mean, args.alpha)

    state.run_count += 1
    state.ema = new_ema
    save_state(args.out_dir, state)

    ended_at = dt.datetime.now(dt.UTC).isoformat()
    record = {
        "run_id": state.run_count,
        "started_at": started_at,
        "ended_at": ended_at,
        "difficulty_start": start_difficulty,
        "difficulty_end": end_difficulty,
        "n": args.n,
        "num_questions_per_cell": num_questions_per_cell,
        "alpha": args.alpha,
        "run_mean_score": run_mean,
        "ema_score": new_ema,
        "endpoint_generation": base_url_generation,
        "endpoint_answering": base_url_answering,
        "endpoint_evaluation": base_url_evaluation,
        "model_generation": model_generation,
        "model_answering": model_answering,
        "model_evaluation": model_evaluation,
        "reasoning_types": reasoning_types,
        "examples": all_examples,
    }
    append_run_record(args.out_dir, record)

    logger.info("Reasoning type report:\n%s", format_reasoning_type_report(state))

    report_path = write_run_report_md(args.out_dir, record, state)
    logger.info(f"Run report: {report_path}")

    logger.info(
        f"Summary: run_mean_score={run_mean:.4f}, ema_score={new_ema:.4f} (alpha={args.alpha})"
    )
    logger.info(f"Saved runs: {runs_path(args.out_dir)}")
    logger.info(f"Saved state: {state_path(args.out_dir)}")


if __name__ == "__main__":
    main()
