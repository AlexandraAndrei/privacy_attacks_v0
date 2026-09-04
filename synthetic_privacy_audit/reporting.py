from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable

from synthetic_privacy_audit.types import AttackResult


def write_reports(results: Iterable[AttackResult], output_dir: str | Path) -> tuple[Path, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = [result.as_dict() for result in results]
    json_path = output / "results.json"
    csv_path = output / "results.csv"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    metric_names = sorted({key for result in payload for key in result["metrics"]})
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["attack", "status", "reason", *metric_names])
        writer.writeheader()
        for result in payload:
            writer.writerow(
                {
                    "attack": result["attack"],
                    "status": result["status"],
                    "reason": result["reason"],
                    **result["metrics"],
                }
            )
    return json_path, csv_path

