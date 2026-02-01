# -------------------------
# Persistence Layer
# -------------------------

import dataclasses
import json
import os
from typing import Any, Dict, List, Optional

# Constants
STATE_FILENAME = "state.json"
RUNS_FILENAME = "runs.jsonl"


@dataclasses.dataclass
class RunState:
    ema: Optional[float] = None
    run_count: int = 0
    current_difficulty: int = 1  # Tracks which difficulty level we're on (1-10)
    ema_by_reasoning_type: Dict[str, float] = dataclasses.field(default_factory=dict)
    ema_by_difficulty: Dict[int, float] = dataclasses.field(default_factory=dict)


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def state_path(out_dir: str) -> str:
    """Return the path to the state file."""
    return os.path.join(out_dir, STATE_FILENAME)


def runs_path(out_dir: str) -> str:
    """Return the path to the runs log file."""
    return os.path.join(out_dir, RUNS_FILENAME)


def load_state(out_dir: str) -> RunState:
    """Load the benchmark state from disk, or return default state if not found."""
    path = state_path(out_dir)
    if not os.path.exists(path):
        return RunState()
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert difficulty keys from strings (JSON limitation) back to integers
    ema_by_difficulty_raw = data.get("ema_by_difficulty", {})
    ema_by_difficulty = {int(k): v for k, v in ema_by_difficulty_raw.items()}
    return RunState(
        ema=data.get("ema", None),
        run_count=int(data.get("run_count", 0)),
        current_difficulty=int(data.get("current_difficulty", 1)),
        ema_by_reasoning_type=dict(data.get("ema_by_reasoning_type", {})),
        ema_by_difficulty=ema_by_difficulty,
    )


def save_state(out_dir: str, state: RunState) -> None:
    """Save the benchmark state to disk."""
    path = state_path(out_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(state), f, indent=2)


def append_run_record(out_dir: str, record: Dict[str, Any]) -> None:
    """Append a run record to the runs log file."""
    path = runs_path(out_dir)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_questions_by_reasoning_type(out_dir: str) -> Dict[str, List[str]]:
    """
    Load all questions from the runs log file, grouped by reasoning_type.

    Returns:
        Dict mapping reasoning_type -> list of questions for that type.
    """
    path = runs_path(out_dir)
    if not os.path.exists(path):
        return {}

    questions_by_rt: Dict[str, List[str]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                for ex in rec.get("examples", []):
                    q = ex.get("question")
                    # tags can be a string (reasoning_type) or a dict with reasoning_type key
                    tags = ex.get("tags")
                    if isinstance(tags, str):
                        rt = tags
                    elif isinstance(tags, dict):
                        rt = tags.get("reasoning_type", "other")
                    else:
                        rt = "other"

                    if isinstance(q, str) and q.strip():
                        questions_by_rt.setdefault(rt, []).append(q.strip())
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
    return questions_by_rt
