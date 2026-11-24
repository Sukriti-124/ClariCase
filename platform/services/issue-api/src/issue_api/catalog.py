"""Read the original product-specific labels, without applying ML label merges."""
import csv
import hashlib
import json
from pathlib import Path


class Catalog:
    def __init__(self, directory: Path):
        with (directory / "team_mapping.csv").open(newline="", encoding="utf-8-sig") as f:
            teams = list(csv.DictReader(f))
        self.teams = {r["team_id"]: {k: r[k] for k in ("team_id", "team_name", "product")}
                      for r in teams}
        if not self.teams or len(self.teams) != len(teams):
            raise ValueError("empty or duplicate team registry")
        self.issues: dict[str, dict] = {}
        with (directory / "issue_mapping.csv").open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                iid, tid = row["product_issue_id"], row["team_id"]
                if iid in self.issues or tid not in self.teams or not iid.startswith(tid + "_"):
                    raise ValueError("invalid issue registry ownership")
                if row["product"] != self.teams[tid]["product"]:
                    raise ValueError("product registry mismatch")
                self.issues[iid] = {"issue_id": iid, "team": tid,
                                    "label": row["product_issue_label"], "issue": row["issue"]}
        if any(not self.for_team(t) for t in self.teams):
            raise ValueError("each product must have issues")
        self.version = hashlib.sha256(json.dumps(
            [self.teams, self.issues], sort_keys=True).encode()).hexdigest()[:16]

    def for_team(self, team: str) -> list[dict]:
        return [row for _, row in sorted(self.issues.items()) if row["team"] == team]

    def view(self, team: str) -> dict:
        return {**self.teams[team], "issues": self.for_team(team)}
