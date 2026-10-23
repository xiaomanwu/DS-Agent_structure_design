"""Golden-QA benchmark runner for DS-Agent.

Executes every case in ``benchmarks/golden_qa.jsonl`` against a freshly
built demo warehouse and checks that the deterministic agent output
contains the expected substrings (and, optionally, routes to the expected
capability). This is the public, statistics-credible evaluation set: the
same questions always produce the same verified answers.

Usage::

    python scripts/run_benchmark.py                 # run all, print table
    python scripts/run_benchmark.py --update-what-changed   # not supported on purpose

Expected values are FROZEN. If a case fails, either a real regression was
introduced (fix the code) or the methodology changed deliberately (update
the golden file in the same commit, with a justification in the note).

Exit code: 0 when every case passes, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "benchmarks" / "golden_qa.jsonl"


def load_cases() -> list[dict]:
    cases = []
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            cases.append(json.loads(line))
    return cases


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filter", default="", help="run only cases whose id contains this substring")
    args = parser.parse_args()

    from ds_agent.agent import DSQueryAgent
    from ds_agent.data.synthetic import build_demo_warehouse

    cases = [c for c in load_cases() if args.filter in c["id"]]
    if not cases:
        print(f"no cases match filter {args.filter!r}")
        return 1

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        build_demo_warehouse(tmp / "wh.duckdb", force=True)
        agent = DSQueryAgent(warehouse=tmp / "wh.duckdb", artifact_dir=tmp / "art")

        passed, failed = 0, 0
        print(f"{'case':28s} {'cap':18s} {'ms':>6s}  result")
        print("-" * 90)
        for case in cases:
            t0 = time.perf_counter()
            try:
                reply = agent.ask(case["question"])
                elapsed = int((time.perf_counter() - t0) * 1000)
                problems = []
                intent = reply.data.get("intent", "")
                if case.get("capability") and intent != case["capability"]:
                    problems.append(f"intent={intent}!={case['capability']}")
                for sub in case.get("expect", []):
                    if sub not in reply.text:
                        problems.append(f"missing {sub!r}")
            except Exception as exc:
                elapsed = int((time.perf_counter() - t0) * 1000)
                problems = [f"{type(exc).__name__}: {exc}"]

            if problems:
                failed += 1
                print(f"{case['id']:28s} {case['capability']:18s} {elapsed:6d}  FAIL  {'; '.join(problems)}")
            else:
                passed += 1
                print(f"{case['id']:28s} {case['capability']:18s} {elapsed:6d}  ok")

        agent.close()

    total = passed + failed
    print("-" * 90)
    print(f"{passed}/{total} passed ({passed / total:.0%})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
