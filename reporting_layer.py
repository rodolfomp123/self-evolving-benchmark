# -------------------------
# Reporting Generation
# -------------------------

import os
from typing import Any, Dict, List, Tuple
from persistence_layer import RunState


def _get_sorted_reasoning_stats(state: RunState) -> List[Tuple[float, str]]:
    """
    Returns reasoning types sorted by EMA - weakest first.

    Args:
        state: The current run state with EMA data.

    Returns:
        List of (ema, reasoning_type) tuples, sorted by EMA ascending.
    """
    rows = []
    for rt, ema in state.ema_by_reasoning_type.items():
        rows.append((ema, rt))
    rows.sort(key=lambda x: x[0])
    return rows


def _get_sorted_difficulty_stats(state: RunState) -> List[Tuple[int, float]]:
    """
    Returns difficulty levels sorted by level.

    Args:
        state: The current run state with EMA data by difficulty.

    Returns:
        List of (difficulty, ema) tuples, sorted by difficulty ascending.
    """
    rows = []
    for diff, ema in state.ema_by_difficulty.items():
        rows.append((diff, ema))
    rows.sort(key=lambda x: x[0])
    return rows

def format_reasoning_type_report(state: RunState) -> str:
    """
    Returns a human-readable report of weakest reasoning types by EMA.
    """
    if not state.ema_by_reasoning_type:
        return "No per-reasoning-type stats yet."

    rows = _get_sorted_reasoning_stats(state)

    lines = []
    lines.append(f"Weakest reasoning types (by EMA):")
    lines.append("  EMA    Reasoning Type")
    lines.append("  -----  --------------")
    for ema, rt in rows:
        lines.append(f"  {ema:0.3f}  {rt}")

    # Also show difficulty stats if available
    if state.ema_by_difficulty:
        diff_rows = _get_sorted_difficulty_stats(state)
        lines.append("")
        lines.append("EMA by difficulty level:")
        lines.append("  Difficulty  EMA")
        lines.append("  ----------  -----")
        for diff, ema in diff_rows:
            lines.append(f"  {diff:10d}  {ema:0.3f}")

    return "\n".join(lines)


def write_run_report_md(out_dir: str, record: Dict[str, Any], state: RunState) -> str:
    """
    Writes a simple Markdown report for the run and returns the filepath.
    """
    report_path = os.path.join(out_dir, f"report_run_{record['run_id']:04d}.md")

    # Summaries
    run_id = record["run_id"]
    started_at = record["started_at"]
    ended_at = record["ended_at"]
    run_mean = record["run_mean_score"]
    ema = record["ema_score"]
    # Support both old single endpoint and new per-model endpoints
    endpoint_generation = record.get("endpoint_generation") or record.get("endpoint", "N/A")
    endpoint_answering = record.get("endpoint_answering") or record.get("endpoint", "N/A")
    endpoint_evaluation = record.get("endpoint_evaluation") or record.get("endpoint", "N/A")
    alpha = record["alpha"]
    difficulty_start = record.get("difficulty_start", 1)
    difficulty_end = record.get("difficulty_end", 1)
    model_generation = record.get("model_generation", "N/A")
    model_answering = record.get("model_answering", "N/A")
    model_evaluation = record.get("model_evaluation", "N/A")
    num_questions_per_cell = record.get("num_questions_per_cell", 1)
    reasoning_types = record.get("reasoning_types", [])

    # Per-type snapshot (weakest first)
    rows = _get_sorted_reasoning_stats(state)

    # Per-difficulty snapshot
    diff_rows = _get_sorted_difficulty_stats(state)

    # Per-run breakdown by reasoning_type
    run_examples = record.get("examples", [])
    by_rt: Dict[str, List[float]] = {}
    by_diff: Dict[int, List[float]] = {}
    for ex in run_examples:
        tags = ex.get("tags") or {}
        if isinstance(tags, str):
            rt = tags
            diff = 1
        else:
            rt = tags.get("reasoning_type", "other")
            diff = int(tags.get("difficulty", 1))
        by_rt.setdefault(rt, []).append(float(ex.get("score", 0.0)))
        by_diff.setdefault(diff, []).append(float(ex.get("score", 0.0)))

    by_rt_rows = []
    for rt, scores in by_rt.items():
        mean_s = sum(scores) / max(1, len(scores))
        by_rt_rows.append((mean_s, len(scores), rt))
    by_rt_rows.sort(key=lambda x: x[0])  # weakest this run first

    by_diff_rows = []
    for diff, scores in by_diff.items():
        mean_s = sum(scores) / max(1, len(scores))
        by_diff_rows.append((diff, mean_s, len(scores)))
    by_diff_rows.sort(key=lambda x: x[0])  # by difficulty ascending

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Benchmark Report — Run {run_id}\n\n")
        f.write("## Run Metadata\n\n")
        f.write(f"- **Started:** {started_at}\n")
        f.write(f"- **Ended:** {ended_at}\n")
        f.write(f"- **Model (Generation):** `{model_generation}` @ `{endpoint_generation}`\n")
        f.write(f"- **Model (Answering):** `{model_answering}` @ `{endpoint_answering}`\n")
        f.write(f"- **Model (Evaluation):** `{model_evaluation}` @ `{endpoint_evaluation}`\n")
        f.write(f"- **Difficulty range:** {difficulty_start} to {difficulty_end}\n")
        f.write(f"- **Questions per cell:** {num_questions_per_cell}\n")
        f.write(f"- **Reasoning types:** {len(reasoning_types)} ({', '.join(reasoning_types)})\n")
        f.write(f"- **EMA alpha:** {alpha}\n")
        f.write("\n")

        f.write("## Scores\n\n")
        f.write(f"- **Run mean score:** `{run_mean:.4f}`\n")
        f.write(f"- **Overall EMA score:** `{ema:.4f}`\n\n")

        f.write("## Reasoning Types (Global EMA)\n\n")
        if rows:
            f.write("| Reasoning Type | EMA |\n")
            f.write("|---|---:|\n")
            for rt_ema, rt in rows:
                f.write(f"| `{rt}` | {rt_ema:.4f} |\n")
        else:
            f.write("_No per-reasoning-type stats yet._\n")
        f.write("\n")

        f.write("## Difficulty Levels (Global EMA)\n\n")
        if diff_rows:
            f.write("| Difficulty | EMA |\n")
            f.write("|---:|---:|\n")
            for diff, diff_ema in diff_rows:
                f.write(f"| {diff} | {diff_ema:.4f} |\n")
        else:
            f.write("_No per-difficulty stats yet._\n")
        f.write("\n")

        f.write("## This Run: Mean Score by Reasoning Type\n\n")
        if by_rt_rows:
            f.write("| Reasoning Type | Mean Score | N |\n")
            f.write("|---|---:|---:|\n")
            for mean_s, cnt, rt in by_rt_rows:
                f.write(f"| `{rt}` | {mean_s:.4f} | {cnt} |\n")
        else:
            f.write("_No examples in this run._\n")
        f.write("\n")

        f.write("## This Run: Mean Score by Difficulty\n\n")
        if by_diff_rows:
            f.write("| Difficulty | Mean Score | N |\n")
            f.write("|---:|---:|---:|\n")
            for diff, mean_s, cnt in by_diff_rows:
                f.write(f"| {diff} | {mean_s:.4f} | {cnt} |\n")
        else:
            f.write("_No examples in this run._\n")
        f.write("\n")

        f.write("## Examples (Preview)\n\n")
        for i, ex in enumerate(run_examples, start=1):
            q = (ex.get("question") or "").strip().replace("\n", " ")
            a = (ex.get("answer") or "").strip().replace("\n", " ")
            score = float(ex.get("score", 0.0))
            tags = ex.get("tags") or {}
            rationale = (ex.get("rationale") or "").strip().replace("\n", " ")

            f.write(f"### {i}) Score: {score:.2f} — Tags: `{tags}`\n\n")
            f.write(f"- **Q:** {q}\n")
            f.write(f"- **A:** {a}\n")
            f.write(f"- **Judge:** {rationale}\n\n")

    return report_path
