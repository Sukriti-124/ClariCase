"""Template_Explainer: the deterministic fallback narration (design §17).

The template turns a Decision_Record into a ``Narration`` without any network
call, backend, clock, randomness or locale-dependent formatting (Req 14.1,
14.2). Everything is read from the record's canonical JSON form, so two
records with the same canonical JSON give byte-identical output, and every
number and id it writes is exactly a stored value.

The output is built to pass the Faithfulness_Validator (design §16):

* The Consumer_Summary is built only from ``outcome``: the Chosen_Team name,
  the routing outcome ("auto-routed" or "sent to human review") and the rank-1
  issue label. It has no numbers, no model names and no banned words, and is
  at most 120 words (Req 14.4, 15.1 to 15.3, 15.9).
* The Agent_Summary has an outcome line, then one line per ``ok`` section in
  ``SECTION_NAMES`` order (Req 15.4). Sections that are skipped or errored and
  fields that are absent or null are left out (Req 14.5).
* Quotes are used only around a Driver n-gram (prefix stripped) or a Driver
  sentence of 1 to 500 characters with no quote characters. Any other text
  written into the Agent_Summary must be free of numbers, team or issue ids
  and names that are not stored record values, and quote characters.
* Numbers are written with ``format_fixed``: contributions, scores and
  thresholds with 3 decimals, counts and offsets as integers. ``format_fixed``
  rounds ``Decimal(repr(v))`` half-even, which is the rule the validator uses to
  match a written number against a record value.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from typing import Any, Iterable

from src.explain.explainers import SECTION_NAMES
from src.explain.narration import (AGENT_MAX, CONSUMER_MAX, FACTS_MAX, PATH_MAX,
                                   CitedFact, Explanation, Narration,
                                   ValidationFailure, ValidationSummary)
from src.explain.record import DecisionRecord, canonical_json, hash_view

__all__ = ["SummaryFacts", "summary_facts", "consumer_summary", "agent_summary",
           "template_narration", "template_explanation", "format_fixed",
           "CONSUMER_WORD_MAX", "QUOTE_MAX"]

CONSUMER_WORD_MAX = 120
QUOTE_MAX = 500
DRIVERS_PER_SECTION = 3
FREE_TEXT_MAX = 80
DP = 3                                    # decimals for contributions and scores

AUTO_PHRASES = ("auto-routed", "auto routed", "automatically routed")
REVIEW_PHRASES = ("sent to human review", "human review", "manual review")
_QUOTE_CHARS = ('"', "\u201c", "\u201d")
_NGRAM_PREFIXES = ("char__", "word__")

# The same patterns as the Faithfulness_Validator (design §16).
_NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?(?![\w])")
_TEAM_ID_RE = re.compile(r"\bT\d{2}\b")
_ISSUE_ID_RE = re.compile(r"\bT\d{2}_I\d{3}\b")

_FALLBACK_ALIASES = {
    "backend_failure": "backend_failure", "backend failure": "backend_failure",
    "timeout": "timeout",
    "no_api_key": "no_api_key", "no api key": "no_api_key",
    "missing api key": "no_api_key",
    "invalid_config": "invalid_config", "invalid config": "invalid_config",
    "invalid narrator config": "invalid_config",
    "validator_rejection": "validator_rejection",
    "validator rejection": "validator_rejection",
    "parse_error": "parse_error", "parse error": "parse_error",
}


# --------------------------------------------------------------------------
# Small pure helpers
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().casefold()


def format_fixed(v: int | float, places: int = DP) -> str:
    """``v`` written with ``places`` decimals, rounded half-even from ``repr(v)``.

    Integers (and ``places == 0``) are written without a decimal point. The
    precision grows with the magnitude, so huge values never raise. A value
    that rounds to zero is written without a minus sign.
    """
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f"not a number: {v!r}")
    if isinstance(v, int):
        return str(v)
    d = Decimal(repr(v))
    with localcontext() as ctx:
        ctx.prec = max(28, d.adjusted() + places + 4)
        q = d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)
    if q == 0:
        q = abs(q)
    return format(q, "f")


@lru_cache(maxsize=1)
def _tables() -> tuple[dict[str, str], dict[str, str], tuple[str, ...]]:
    """Team names, issue labels and every normalised name/label, longest first."""
    try:
        from src.explain.labels import load_tables
        t = load_tables()
        teams = {k: t.teams[k] for k in sorted(t.teams)}
        issues = {k: t.issues[k].label for k in sorted(t.issues)}
    except Exception:  # tables missing: no name screening beyond the regexes
        teams, issues = {}, {}
    names = sorted({_norm(n) for n in (*teams.values(), *issues.values()) if _norm(n)},
                   key=lambda n: (-len(n), n))
    return teams, issues, tuple(names)


def _mentions_table_name(s: str) -> bool:
    ns = _norm(s)
    for name in _tables()[2]:
        if re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", ns):
            return True
    return False


def _has_outcome_phrase(s: str) -> bool:
    ns = _norm(s)
    return any(p in ns for p in AUTO_PHRASES + REVIEW_PHRASES)


def _safe_free_text(s: Any) -> bool:
    """True when ``s`` may be written unquoted in the Agent_Summary.

    It must be short printable text with no quote characters, no number, no
    ``%``, no team or issue id, no team name or issue label and no routing
    outcome phrase, so it can never trip a validator check.
    """
    if not isinstance(s, str):
        return False
    t = s.strip()
    if not t or len(t) > FREE_TEXT_MAX or not t.isprintable():
        return False
    if any(q in t for q in _QUOTE_CHARS) or "%" in t:
        return False
    if _NUM_RE.search(t) or _TEAM_ID_RE.search(t) or _ISSUE_ID_RE.search(t):
        return False
    return not _mentions_table_name(t) and not _has_outcome_phrase(t)


def _safe_team_id(s: Any) -> bool:
    return isinstance(s, str) and (s in _tables()[0] or _safe_free_text(s))


def _safe_issue_id(s: Any) -> bool:
    return isinstance(s, str) and (s in _tables()[1] or _safe_free_text(s))


def _safe_key(k: Any) -> bool:
    """A dict key that can appear in a dotted Cited_Fact path."""
    return isinstance(k, str) and bool(k) and "." not in k and not k.isdigit()


def _quotable(s: Any) -> bool:
    """Text that may be quoted: 1..500 chars, no quote characters."""
    if not isinstance(s, str) or not s.strip() or len(s) > QUOTE_MAX:
        return False
    if any(q in s for q in _QUOTE_CHARS):
        return False
    return not _has_outcome_phrase(s)


def _one_line(s: str) -> str:
    """Whitespace runs collapsed to one space; the validator's ``norm`` is unchanged."""
    return re.sub(r"\s+", " ", s)


def _strip_prefix(label: str) -> str:
    for p in _NGRAM_PREFIXES:
        if label.startswith(p):
            return label[len(p):]
    return label


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_absent(v: Any) -> bool:
    return isinstance(v, dict) and v.get("absent") is True


def _top_key(dist: Any) -> str | None:
    """The key with the largest value; ties go to the smallest key."""
    if not isinstance(dist, dict):
        return None
    items = [(k, v) for k, v in sorted(dist.items()) if _is_num(v)]
    if not items:
        return None
    return min(items, key=lambda kv: (-kv[1], kv[0]))[0]


def _canonical(record: DecisionRecord | dict) -> dict:
    return json.loads(canonical_json(record))


# --------------------------------------------------------------------------
# Lines with their Cited_Facts
# --------------------------------------------------------------------------

@dataclass
class _Line:
    parts: list[str] = field(default_factory=list)
    facts: list[tuple[str, Any]] = field(default_factory=list)
    protected: bool = False

    def fact(self, path: str, value: Any) -> None:
        if len(path) <= PATH_MAX:
            self.facts.append((path, value))

    def num(self, path: str, v: Any, places: int = DP) -> str:
        self.fact(path, v)
        return format_fixed(v, places)

    def text(self) -> str:
        return " ".join(p for p in self.parts if p).strip()


def _driver_phrases(line: _Line, drivers: Any, base: str) -> list[str]:
    """The top Drivers by absolute contribution, original index as tie-break."""
    if not isinstance(drivers, list):
        return []
    idx = [i for i, d in enumerate(drivers)
           if isinstance(d, dict) and _is_num(d.get("contribution"))]
    idx.sort(key=lambda i: (-abs(drivers[i]["contribution"]), i))
    out: list[str] = []
    for i in idx[:DRIVERS_PER_SECTION]:
        d, path = drivers[i], f"{base}.{i}"
        kind = d.get("kind")
        if kind == "ngram":
            label = d.get("label")
            gram = _one_line(_strip_prefix(label)) if isinstance(label, str) else None
            what = f'"{gram}"' if _quotable(gram) else "an unquotable n-gram"
        elif kind == "sentence":
            span = d.get("span")
            text = _one_line(d["text"]) if isinstance(d.get("text"), str) else None
            if _quotable(text):
                what = f'"{text}"'
            elif isinstance(span, dict) and all(_is_num(span.get(k))
                                                for k in ("index", "start", "end")):
                what = (f"sentence {line.num(path + '.span.index', span['index'], 0)}"
                        f" (characters {line.num(path + '.span.start', span['start'], 0)}"
                        f" to {line.num(path + '.span.end', span['end'], 0)})")
            else:
                what = "a sentence"
        elif kind == "model":
            label = d.get("label")
            what = f"model {label.strip()}" if _safe_free_text(label) else "a model"
        elif kind == "feature_group":
            label = d.get("label")
            what = (f"feature group {label.strip()}" if _safe_free_text(label)
                    else "a feature group")
        else:
            what = "redacted text"
        out.append(f"{what} {line.num(path + '.contribution', d['contribution'])}")
    return out


def _drivers_clause(line: _Line, drivers: Any, base: str, title: str = "drivers") -> str:
    phrases = _driver_phrases(line, drivers, base)
    return f"{title}: " + ", ".join(phrases) if phrases else ""


def _team(line: _Line, path: str, v: Any) -> str | None:
    if _safe_team_id(v):
        line.fact(path, v)
        return v.strip() if v not in _tables()[0] else v
    return None


def _score_vs_threshold(line: _Line, base: str, details: dict) -> str:
    score, thr = details.get("score"), details.get("threshold")
    if not _is_num(score):
        return ""
    s = f"score {line.num(base + '.score', score)}"
    if _is_num(thr):
        op = ">=" if score >= thr else "<"
        return f"{s} {op} threshold {line.num(base + '.threshold', thr)}"
    return f"{s}, threshold absent"


def _join_clauses(head: str, clauses: Iterable[str]) -> str:
    body = "; ".join(c for c in clauses if c)
    return f"{head}: {body}." if body else f"{head}: no further details."


# --------------------------------------------------------------------------
# Per-section lines (only called for status "ok")
# --------------------------------------------------------------------------

def _im1_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    t = _team(line, f"{base}.details.top_team", d.get("top_team"))
    if t:
        out.append(f"top team {t}")
    r = _team(line, f"{base}.details.runner_up", d.get("runner_up"))
    if r:
        out.append(f"runner-up {r}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    issue = d.get("issue")
    if isinstance(issue, dict):
        ib = f"{base}.details.issue"
        iid = issue.get("issue_id")
        if _safe_issue_id(iid):
            line.fact(f"{ib}.issue_id", iid)
            head = f"issue {iid.strip() if iid not in _tables()[1] else iid}"
            cmp_ = issue.get("compared_issue")
            if _safe_issue_id(cmp_):
                line.fact(f"{ib}.compared_issue", cmp_)
                head += f" versus {cmp_.strip() if cmp_ not in _tables()[1] else cmp_}"
            out.append(head)
        out.append(_drivers_clause(line, issue.get("drivers"), f"{ib}.drivers",
                                   "issue drivers"))
    return out


def _bm2_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    t = _team(line, f"{base}.details.chosen_team", d.get("chosen_team"))
    if t:
        out.append(f"explains team {t}")
    r = _team(line, f"{base}.details.runner_up", d.get("runner_up"))
    if r:
        out.append(f"runner-up {r}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    return out


def _im2_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    top = _top_key(d.get("team_probs"))
    if top is not None and _safe_team_id(top):
        out.append(f"top team {top.strip()}")
    t = _team(line, f"{base}.details.chosen_team", d.get("chosen_team"))
    if t:
        out.append(f"explains team {t}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    return out


def _knn_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    top = _top_key(d.get("team_distribution"))
    if top is not None and _safe_team_id(top):
        out.append(f"top team {top.strip()}")
    if _is_num(d.get("effective_k")):
        out.append(f"neighbours used {line.num(base + '.details.effective_k', d['effective_k'], 0)}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    return out


def _stacker_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    if d.get("case") in ("stacker_agrees", "stacker_disagrees"):
        out.append(f"case {d['case']}")
    top = _team(line, f"{base}.details.top_after", d.get("top_after"))
    if top is None and isinstance(d.get("ranking"), list) and d["ranking"]:
        top = _team(line, f"{base}.details.ranking.0", d["ranking"][0])
    if top:
        out.append(f"top team {top}")
    c = _team(line, f"{base}.details.comparison_team", d.get("comparison_team"))
    if c:
        out.append(f"comparison team {c}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    return out


def _gate_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = [_score_vs_threshold(line, f"{base}.details", d)]
    if d.get("unit") in ("log-odds", "change in gate score"):
        out.append(f"unit {d['unit']}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))
    return out


def _routing_line(line: _Line, base: str, sec: dict, chosen: Any) -> list[str]:
    d = sec.get("details") or {}
    out = []
    teams = d.get("model_teams")
    if isinstance(teams, dict):
        shown, disagree = [], []
        for model in sorted(teams):
            team = teams[model]
            if not _safe_free_text(model):
                continue
            if _is_absent(team):
                shown.append(f"{model} absent")
                continue
            path = f"{base}.details.model_teams.{model}"
            t = _team(line, path, team) if _safe_key(model) else (
                team if _safe_team_id(team) else None)
            if t is None:
                continue
            shown.append(f"{model} {t}")
            if team != chosen:
                disagree.append(model)
        if shown:
            out.append("model top teams: " + ", ".join(shown))
            out.append("models disagreeing with the chosen team: " +
                       (", ".join(disagree) if disagree else "none"))
    out.append(_score_vs_threshold(line, f"{base}.details", d))
    if d.get("threshold_kind") in ("base", "per-team"):
        out.append(f"threshold kind {d['threshold_kind']}")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))  # Req 15.4
    return out


def _agentic_line(line: _Line, base: str, sec: dict) -> list[str]:
    d = sec.get("details") or {}
    out = []
    t = _team(line, f"{base}.details.team_choice", d.get("team_choice"))
    if t:
        out.append(f"team choice {t}")
    if d.get("team_status") in ("chosen", "failed", "not_asked"):
        out.append(f"team task {d['team_status']}")
    iid = d.get("issue_choice")
    if _safe_issue_id(iid):
        line.fact(f"{base}.details.issue_choice", iid)
        out.append(f"issue choice {iid.strip() if iid not in _tables()[1] else iid}")
    if d.get("issue_status") in ("chosen", "failed", "not_asked"):
        out.append(f"issue task {d['issue_status']}")
    if isinstance(d.get("team_changed"), bool):
        out.append("team changed" if d["team_changed"] else "team kept")
    out.append(_drivers_clause(line, sec.get("drivers"), f"{base}.drivers"))  # Req 15.4
    return out


_SECTION_LINES = {
    "improved_model_1": _im1_line, "baseline_model_2": _bm2_line,
    "improved_model_2": _im2_line, "knn_embed": _knn_line,
    "stacker": _stacker_line, "gate": _gate_line, "agentic": _agentic_line,
}


# --------------------------------------------------------------------------
# Summary facts and the consumer summary
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class SummaryFacts:
    team: str
    team_name: str
    issues: list[dict]
    route: str
    auto_route: bool
    scenario: str


def summary_facts(record: DecisionRecord | dict) -> SummaryFacts:
    """Chosen team, ranked issues, route and scenario from ``outcome``."""
    out = _canonical(record)["outcome"]
    issues = [{k: i.get(k) for k in ("rank", "issue_id", "label", "display_label", "prob")}
              for i in out.get("top_issues") or []]
    return SummaryFacts(team=out["chosen_team"]["id"], team_name=out["chosen_team"]["name"],
                        issues=issues, route=out["route"],
                        auto_route=bool(out["auto_route"]), scenario=out["scenario"])


def _consumer(outcome: dict) -> str:
    name = outcome["chosen_team"]["name"].strip()
    verb = "auto-routed to" if outcome["auto_route"] else "sent to human review for"
    first = f"Your complaint was {verb} the {name}."
    issues = outcome.get("top_issues") or []
    if issues:
        top = issues[0]
        label = (top.get("display_label") or "").strip() or (top.get("label") or "").strip()
        if label:
            full = f"{first} The main issue we identified is: {label}."
            if len(full.split()) <= CONSUMER_WORD_MAX and len(full) <= CONSUMER_MAX:
                return full
    return first


def consumer_summary(record: DecisionRecord | dict) -> str:
    """The Consumer_Summary, built only from ``outcome`` (Req 14.4, 15.1 to 15.3)."""
    return _consumer(_canonical(record)["outcome"])


# --------------------------------------------------------------------------
# Agent summary
# --------------------------------------------------------------------------

def _outcome_line(outcome: dict, *, minimal: bool = False) -> _Line:
    line = _Line(protected=True)
    team = outcome["chosen_team"]
    tid, tname = team["id"], team["name"]
    teams = _tables()[0]
    id_ok = _safe_team_id(tid)
    name_ok = (teams.get(tid) == tname) or _safe_free_text(tname)
    if id_ok or not name_ok:
        ref = tid if tid in teams else _one_line(tid).strip() or tid
        line.fact("outcome.chosen_team.id", tid)
        if name_ok:
            ref += f" ({tname.strip()})"
            line.fact("outcome.chosen_team.name", tname)
    else:
        ref = tname.strip()
        line.fact("outcome.chosen_team.name", tname)
    verb = "auto-routed to" if outcome["auto_route"] else "sent to human review for"
    line.fact("outcome.route", outcome["route"])
    text = f"Outcome: the complaint was {verb} chosen team {ref}"
    if not minimal:
        line.fact("outcome.scenario", outcome["scenario"])
        text += f"; scenario {outcome['scenario']}; issue order {outcome['issue_order']}"
        issues = []
        labels = _tables()[1]
        for i, iss in enumerate(outcome.get("top_issues") or []):
            p = f"outcome.top_issues.{i}"
            iid, lab = iss.get("issue_id"), iss.get("label")
            bits = [str(iss.get("rank", i + 1))]
            if _safe_issue_id(iid):
                line.fact(f"{p}.issue_id", iid)
                bits.append(iid if iid in labels else iid.strip())
            if isinstance(lab, str) and (labels.get(iid) == lab or _safe_free_text(lab)):
                bits.append(f"({lab.strip()})")
            if _is_num(iss.get("prob")):
                bits.append(line.num(f"{p}.prob", iss["prob"]))
            issues.append(" ".join(bits))
        if issues:
            text += "; top issues: " + ", ".join(issues)
    line.parts.append(text + ".")
    return line


def _agent_lines(rec: dict) -> list[_Line]:
    outcome = rec["outcome"]
    lines = [_outcome_line(outcome)]
    sections = rec.get("sections") or {}
    for name in SECTION_NAMES:
        sec = sections.get(name)
        if not isinstance(sec, dict) or sec.get("status") != "ok":
            continue
        base = f"sections.{name}"
        line = _Line(protected=(name == "routing"))
        if name == "routing":
            clauses = _routing_line(line, base, sec, outcome["chosen_team"]["id"])
        else:
            clauses = _SECTION_LINES[name](line, base, sec)
        head = f"{name} ({sec.get('method')} {sec.get('technique')})"
        line.parts.append(_join_clauses(head, clauses))
        lines.append(line)
    return lines


def _fit(lines: list[_Line], outcome: dict) -> list[_Line]:
    """Drop unprotected lines from the end until the text fits AGENT_MAX."""
    kept = list(lines)

    def size(ls: list[_Line]) -> int:
        return len("\n".join(l.text() for l in ls).strip())

    i = len(kept) - 1
    while size(kept) > AGENT_MAX and i >= 0:
        if not kept[i].protected:
            del kept[i]
        i -= 1
    if size(kept) > AGENT_MAX:              # last resort: the bare outcome line
        kept[0] = _outcome_line(outcome, minimal=True)
    return kept


def _collect_facts(lines: list[_Line]) -> list[CitedFact]:
    seen: set[str] = set()
    facts: list[CitedFact] = []
    for line in lines:
        for path, value in line.facts:
            if path in seen or len(facts) >= FACTS_MAX:
                continue
            seen.add(path)
            facts.append(CitedFact(path=path, value=value))
    return facts


def _build(rec: dict) -> tuple[str, str, list[CitedFact]]:
    lines = _fit(_agent_lines(rec), rec["outcome"])
    agent = "\n".join(l.text() for l in lines).strip()
    return _consumer(rec["outcome"]), agent, _collect_facts(lines)


def agent_summary(record: DecisionRecord | dict) -> str:
    """The Agent_Summary text (Req 15.4), at most AGENT_MAX characters."""
    return _build(_canonical(record))[1]


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def template_narration(record: DecisionRecord | dict) -> Narration:
    """The deterministic Narration for ``record`` (Req 14.1, 14.2)."""
    consumer, agent, facts = _build(_canonical(record))
    return Narration(consumer_summary=consumer, agent_summary=agent, cited_facts=facts)


def _fallback_reason(reason: str) -> str:
    key = _norm(reason).replace("-", " ") if isinstance(reason, str) else ""
    if key in _FALLBACK_ALIASES:
        return _FALLBACK_ALIASES[key]
    if key.replace(" ", "_") in _FALLBACK_ALIASES:
        return _FALLBACK_ALIASES[key.replace(" ", "_")]
    raise ValueError(f"unknown fallback reason {reason!r}")


def _failure(fc: Any) -> ValidationFailure:
    if isinstance(fc, ValidationFailure):
        return fc
    if isinstance(fc, dict):
        return ValidationFailure(check=str(fc["check"]), offending=str(fc["offending"]),
                                 reason=str(fc["reason"]))
    return ValidationFailure(check=str(fc.check), offending=str(fc.offending),
                             reason=str(fc.reason))


def template_explanation(record: DecisionRecord | dict, reason: str,
                         failed_checks: Iterable[Any] = (), *,
                         record_hash: str | None = None,
                         created_at: str | None = None) -> Explanation:
    """An Explanation with ``source="template"`` (Req 14.6).

    ``reason`` is a fallback reason, either the literal (``no_api_key``) or its
    prose form (``"no API key"``). ``failed_checks`` holds the validator's
    ``FailedCheck`` objects (or dicts, or ``ValidationFailure``) and is stored
    in order. ``validation.accepted`` is always false: no LLM narration was
    accepted. ``record_hash`` and ``created_at`` default to values derived from
    the record (SHA-256 of the canonical hash view, and the record's own
    ``created_at``), so the template never reads a clock.
    """
    rec = _canonical(record)
    narration = template_narration(rec)
    if record_hash is None:
        record_hash = hashlib.sha256(
            canonical_json(hash_view(rec)).encode("utf-8")).hexdigest()
    if created_at is None:
        created_at = str(rec.get("created_at", ""))
    return Explanation(
        consumer_summary=narration.consumer_summary,
        agent_summary=narration.agent_summary,
        cited_facts=list(narration.cited_facts),
        source="template", fallback_reason=_fallback_reason(reason),
        validation=ValidationSummary(accepted=False,
                                     failed=[_failure(f) for f in failed_checks]),
        record_hash=record_hash, created_at=created_at)
