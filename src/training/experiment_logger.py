from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from config import settings


class FileExperimentLogger:

    def __init__(self, runs_dir: Optional[Path] = None) -> None:
        self.runs_dir = Path(runs_dir or settings.RUNS_DIR)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.runs_dir / "runs.jsonl"

    def start_run(
        self, model_type: str, params: Dict[str, Any], run_name: Optional[str] = None
    ) -> str:
        run_id = run_name or f"{model_type}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:6]}"
        entry = {
            "run_id": run_id,
            "model_type": model_type,
            "status": "started",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "params": params,
            "metrics": {},
            "finished_at": None,
        }
        self._append(entry)
        return run_id

    def log_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        self._append(
            {"run_id": run_id, "type": "metrics", "metrics": metrics,
             "timestamp": datetime.now(timezone.utc).isoformat()}
        )

    def end_run(self, run_id: str, metrics: Dict[str, Any], status: str = "completed") -> None:
        self._append(
            {
                "run_id": run_id,
                "type": "end",
                "status": status,
                "metrics": metrics,
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _append(self, entry: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
