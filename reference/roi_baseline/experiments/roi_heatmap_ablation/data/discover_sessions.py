from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    artifacts_dir: str
    db_dir: str
    agent6_csv: str
    heatmap_mat: str
    heatmap_figures_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


def _first_csv(agent6_dir: Path) -> Optional[Path]:
    csvs = sorted(agent6_dir.glob("*.csv"))
    if csvs:
        return csvs[0]
    csvs = sorted(agent6_dir.rglob("*.csv"))
    return csvs[0] if csvs else None


def discover_sessions(dataset_root: str | Path) -> List[SessionInfo]:
    root = Path(dataset_root)
    artifacts_root = root / "artifacts"
    db_root = root / "db"
    heatmap_root = root / "heatmaps" / "heatmap_result"
    sessions: list[SessionInfo] = []

    if not artifacts_root.exists():
        return sessions

    for artifacts_dir in sorted(p for p in artifacts_root.iterdir() if p.is_dir()):
        session_id = artifacts_dir.name
        db_dir = db_root / session_id
        if not db_dir.is_dir():
            continue

        agent6_dirs = sorted(p for p in db_dir.iterdir() if p.is_dir() and p.name.startswith("agent6"))
        if not agent6_dirs:
            continue
        agent6_csv = _first_csv(agent6_dirs[0])
        if agent6_csv is None:
            continue

        figures_dir = heatmap_root / "figures" / session_id / "ToF-Doppler"
        mat_path = heatmap_root / "mat" / session_id / "ToF-Doppler" / "smoothed_CSI_avg.mat"
        if not figures_dir.is_dir() or not mat_path.is_file():
            continue

        sessions.append(
            SessionInfo(
                session_id=session_id,
                artifacts_dir=str(artifacts_dir),
                db_dir=str(db_dir),
                agent6_csv=str(agent6_csv),
                heatmap_mat=str(mat_path),
                heatmap_figures_dir=str(figures_dir),
            )
        )
    return sessions


def sessions_from_ids(all_sessions: Iterable[SessionInfo], ids: Iterable[str]) -> list[SessionInfo]:
    by_id = {s.session_id: s for s in all_sessions}
    return [by_id[sid] for sid in ids if sid in by_id]

