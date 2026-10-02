from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

from .discover_sessions import SessionInfo


def make_session_split(
    sessions: Sequence[SessionInfo],
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    seed: int = 7,
) -> dict:
    ids = [s.session_id for s in sessions]
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)
    if n and n_train == 0:
        n_train = 1
    if n - n_train > 1 and n_valid == 0:
        n_valid = 1

    return {
        "train": ids[:n_train],
        "valid": ids[n_train : n_train + n_valid],
        "test": ids[n_train + n_valid :],
    }


def load_or_create_split(
    split_path: str | Path,
    sessions: Sequence[SessionInfo],
    train_ratio: float,
    valid_ratio: float,
    seed: int,
) -> dict:
    path = Path(split_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    split = make_session_split(sessions, train_ratio, valid_ratio, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    return split

