"""Team and issue label tables for Decision_Records (Req 1.5).

The tables live in the committed file ``label_tables.json`` next to this module,
because Platform images copy ``src/`` but not ``data/processed/``. When the JSON
file is missing or unreadable, the loader falls back to the CSVs in
``data/processed``.

This module uses only the standard library and does not import
``src.utils.labels`` (which needs pandas and ``data/processed`` at import time).

Regenerate the JSON after the CSVs change::

    python -m src.explain.labels --write
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

TABLES_VERSION = 1
TABLES_PATH = Path(__file__).with_name("label_tables.json")
_ROOT = Path(__file__).resolve().parents[2]
TEAM_CSV = _ROOT / "data" / "processed" / "team_mapping.csv"
ISSUE_CSV = _ROOT / "data" / "processed" / "issue_canonical.csv"

_SEP = " :: "


def display_label(label: str) -> str:
    """The part of an issue label after ``" :: "``, as the Streamlit app shows it."""
    return label.split(_SEP, 1)[-1]


@dataclass(frozen=True)
class IssueEntry:
    issue_id: str
    label: str
    display_label: str
    team_id: str


@dataclass(frozen=True)
class LabelTables:
    teams: Mapping[str, str]          # team id -> team name
    issues: Mapping[str, IssueEntry]  # canonical issue id -> entry
    source: str                       # "json" or "csv"

    def team_name(self, team_id: str) -> str:
        return self.teams[team_id]

    def issue(self, issue_id: str) -> IssueEntry:
        return self.issues[issue_id]

    def to_json_dict(self) -> dict:
        return {
            "version": TABLES_VERSION,
            "teams": {t: {"name": n} for t, n in sorted(self.teams.items())},
            "issues": {
                i: {"label": e.label, "display_label": e.display_label,
                    "team_id": e.team_id}
                for i, e in sorted(self.issues.items())
            },
        }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tables_from_csv(team_csv: Path = TEAM_CSV,
                    issue_csv: Path = ISSUE_CSV) -> LabelTables:
    """Build the tables from the CSVs.

    ``issue_canonical.csv`` can hold several product issues merged into one
    canonical ``issue_id``. As in ``src.utils.labels.issue_labels`` (the source
    of the labels in prediction output), the last row for an id wins.
    """
    teams: dict[str, str] = {}
    for row in _read_csv(team_csv):
        tid = row["team_id"]
        if tid in teams:
            raise ValueError(f"{team_csv.name}: duplicate team_id {tid!r}")
        teams[tid] = row["team_name"]
    issues: dict[str, IssueEntry] = {}
    for row in _read_csv(issue_csv):
        iid, tid, label = row["issue_id"], row["team_id"], row["issue_label"]
        prev = issues.get(iid)
        if prev is not None and prev.team_id != tid:
            raise ValueError(
                f"{issue_csv.name}: issue_id {iid!r} maps to teams "
                f"{prev.team_id!r} and {tid!r}")
        if tid not in teams:
            raise ValueError(
                f"{issue_csv.name}: issue_id {iid!r} has unknown team {tid!r}")
        issues[iid] = IssueEntry(iid, label, display_label(label), tid)
    return LabelTables(MappingProxyType(teams), MappingProxyType(issues), "csv")


def tables_from_json(path: Path = TABLES_PATH) -> LabelTables:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != TABLES_VERSION:
        raise ValueError(f"{path.name}: unsupported version {data.get('version')!r}")
    teams = {t: v["name"] for t, v in data["teams"].items()}
    issues = {
        i: IssueEntry(i, v["label"], v["display_label"], v["team_id"])
        for i, v in data["issues"].items()
    }
    return LabelTables(MappingProxyType(teams), MappingProxyType(issues), "json")


@lru_cache(maxsize=1)
def load_tables() -> LabelTables:
    """Read ``label_tables.json``; fall back to the CSVs if it cannot be read."""
    try:
        return tables_from_json()
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return tables_from_csv()


def team_names() -> Mapping[str, str]:
    return load_tables().teams


def issue_labels() -> Mapping[str, str]:
    return MappingProxyType(
        {i: e.label for i, e in load_tables().issues.items()})


def team_name(team_id: str) -> str:
    return load_tables().team_name(team_id)


def issue(issue_id: str) -> IssueEntry:
    return load_tables().issue(issue_id)


def render_json(tables: LabelTables) -> str:
    return json.dumps(tables.to_json_dict(), ensure_ascii=False, indent=2,
                      sort_keys=True) + "\n"


def write_tables(path: Path = TABLES_PATH) -> LabelTables:
    tables = tables_from_csv()
    path.write_text(render_json(tables), encoding="utf-8", newline="\n")
    return tables


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true",
                    help=f"regenerate {TABLES_PATH.name} from the CSVs")
    ap.add_argument("--check", action="store_true",
                    help=f"exit 1 if {TABLES_PATH.name} differs from the CSVs")
    args = ap.parse_args(argv)
    if args.write:
        t = write_tables()
        print(f"[explain.labels] wrote {TABLES_PATH.name}: "
              f"{len(t.teams)} teams, {len(t.issues)} issues")
        return 0
    if args.check:
        want = render_json(tables_from_csv())
        have = TABLES_PATH.read_text(encoding="utf-8") if TABLES_PATH.exists() else ""
        ok = want == have
        print(f"[explain.labels] {TABLES_PATH.name} "
              f"{'is up to date' if ok else 'is stale; run --write'}")
        return 0 if ok else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
