"""Statcast projection sample, from the command line:

    python -m sample download     fetch 2021-2026 Statcast plate appearances (resumable)
    python -m sample build        season totals, league numbers, validation
    python -m sample evaluate     holdout test: predict 2024, 2025, 2026 against Marcel
    python -m sample project      fit on 2024-2026, project 2027, Nationals table
    python -m sample all          everything above, in order
"""

import sys


def main(argv: list[str]) -> int:
    command = argv[0] if argv else ""
    if command in ("download", "all"):
        from sample import download
        download.run()
    if command in ("build", "all"):
        from sample import aggregate
        if not aggregate.run():
            return 1
    if command in ("evaluate", "all"):
        from sample import evaluate
        evaluate.run()
    if command in ("project", "all"):
        from sample import project
        project.run()
    if command not in ("download", "build", "evaluate", "project", "all"):
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
