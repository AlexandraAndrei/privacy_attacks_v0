from __future__ import annotations

import argparse

from synthetic_privacy_audit.data import load_context
from synthetic_privacy_audit.registry import all_attacks
from synthetic_privacy_audit.reporting import write_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic tabular privacy attacks.")
    parser.add_argument("--config", required=True, help="Path to a dataset JSON configuration.")
    parser.add_argument("--output", required=True, help="Directory for JSON and CSV results.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    context = load_context(args.config)
    results = [attack.execute(context) for attack in all_attacks()]
    json_path, csv_path = write_reports(results, args.output)
    completed = sum(result.status.value == "completed" for result in results)
    print(f"Completed {completed}/{len(results)} attacks.")
    print(f"Reports: {json_path} | {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

