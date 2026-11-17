"""Faithfulness_Validator: checks a Narration against its Decision_Record (design §16).

``validate(n, record, ctx)`` is a pure function of the Narration, the canonical
hash view of the record (no request_id, timestamps or timings, so a result is
valid for every record with the same Record_Hash) and the ``ValidatorContext``.
The whole body runs inside ``try``; an internal exception gives
``accepted=False`` with the single check ``internal_error`` (Req 13.10). Failed
checks are de-duplicated and sorted by ``(check, offending, reason)``, so two
runs return identical results (Req 13.11).

Steps, applied to each summary unless stated otherwise:

1. **Quoted spans** (``QUOTE_RE``) are recorded and blanked in a working copy,
   so mentions and numbers inside quotes are checked only as quoted phrases
   (Req 13.5c).
2. **Names** (Req 13.1, 13.2). Team names, issue labels and the *outcome
   values* (Chosen_Team id and name, and the id, label and display label of
   each served top-3 issue) are searched longest first, case-folded with
   whitespace runs collapsed and word guards ``(?<!\\w)…(?!\\w)``, and blanked.
   Outcome values are always allowed (Req 13.8, 15.2, F2). A table name or
   label is allowed when one of its ids, or the name itself, is a string (key
   or value) of the record. Names run before the id regexes so the longest
   match wins and an outcome value is never re-read as something else; the
   result is the same as the design order for every real label table, because
   no team name or issue label contains an id.
3. **Ids**: ``ISSUE_ID_RE`` and then ``TEAM_ID_RE`` (case-sensitive capital T)
   on the working copy. An id is allowed when it is a string of the record.
4. **Numbers** (Req 13.4, 13.5): ``NUM_RE`` on the working copy. Integers 0..11
   without ``.`` or ``%`` are exempt. A number with d decimals matches a record
   value v when ``Decimal(repr(v))`` rounded half-even to d decimals equals it;
   ``X%`` matches p in [0, 1] when ``p * 100`` rounded the same way equals X.
   Rounding uses a local Decimal context whose precision grows with the
   magnitude, so values such as 1e300 never raise ``InvalidOperation``.
5. **Quoted phrases** (Req 13.3): ``norm(q)`` must equal a Driver n-gram
   (``word__``/``char__`` prefix removed) or Driver sentence, or, when
   ``ctx.include_text`` is true, be a substring of ``norm(masked_text)`` or of
   ``norm(clean_text(masked_text))``. ``clean_text`` is imported lazily; where
   it cannot be imported (an image without ``data/preparation``) only that
   option is unavailable.
6. **Required statements** (Req 13.7): each summary names the Chosen_Team (id
   or name) and contains a phrase of the recorded outcome and no phrase of the
   opposite outcome (searched outside quotes). When the recorded Chosen_Team
   name is blank, the Consumer_Summary (which names the team only by name,
   Req 15.3) is exempt from the team part.

Quote characters inside an occurrence of an outcome value do not open or close
a quoted span, so a team name or issue label holding ``"`` is read as a name.

Then the **Cited_Facts** (Req 13.6) and the **Consumer_Summary rules** (Req
13.8, 15.1 to 15.3). Before the consumer bans run, the outcome values are
blanked in a separate copy of the Consumer_Summary. This keeps the Chosen_Team
name and the served issue labels allowed even when they contain a banned word:
the rank-1 label of issue T01_I012 is "Unable to get your credit report or
credit score", and the Consumer_Summary must state it (Req 15.3) although
"score" is a banned word (Req 15.1). Outside the outcome values the bans apply
in full, including inside quotes.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from functools import lru_cache
from typing import Any, Iterable, Iterator, Mapping

from src.explain.narration import CitedFact, Narration
from src.explain.record import DecisionRecord, canonical_json, hash_view

__all__ = ["FailedCheck", "ValidationResult", "ValidatorContext", "validate",
           "default_context", "norm", "number_matches", "resolve_path",
           "QUOTE_RE", "ISSUE_ID_RE", "TEAM_ID_RE", "NUM_RE", "BANNED_WORD_RE",
           "AUTO_PHRASES", "REVIEW_PHRASES", "MODEL_NAMES",
           "BANNED_NUMERIC_KEYS", "CONSUMER_WORD_MAX", "NUMBER_TOL", "CHECKS"]

QUOTE_RE = re.compile(r'"([^"]{1,500})"|\u201c([^\u201d]{1,500})\u201d')
ISSUE_ID_RE = re.compile(r"\bT\d{2}_I\d{3}\b")
TEAM_ID_RE = re.compile(r"\bT\d{2}\b")
NUM_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?%?(?![\w])")
BANNED_WORD_RE = re.compile(r"\b(?:probabilit(?:y|ies)|scores?|thresholds?|confidence)\b")

AUTO_PHRASES = ("auto-routed", "auto routed", "automatically routed")
REVIEW_PHRASES = ("sent to human review", "human review", "manual review")

# Req 15.1 list, plus spelling variants of the same models.
MODEL_NAMES: tuple[str, ...] = (
    "improved_model_1", "baseline_model_2", "improved_model_2", "knn_embed",
    "stacker", "gate", "LinearSVC", "naive Bayes", "na\u00efve Bayes",
    "LogisticRegression", "logistic regression", "HistGradientBoosting",
    "bge-small")

# A numeric leaf is banned in the Consumer_Summary when any key on its path is
# one of these (probability, score, contribution and threshold fields, Req 15.1).
BANNED_NUMERIC_KEYS: frozenset[str] = frozenset({
    "prob", "probs", "probability", "score", "contribution", "threshold",
    "confidence", "calibrated_prob", "isotonic_pre_norm", "raw_decision",
    "p_full", "isotonic", "team_probs", "team_distribution", "p_before",
    "p_after", "calibrated_prob_chosen", "uncalibrated_llr", "raw_diff",
    "served_diff", "served_decision_deviation", "redacted_contribution",
    "constant", "residual", "logit_diff", "intercept_diff", "similarity",
    "value", "primary_score", "fallback_threshold", "rescue_threshold",
    "max_abs_diff_vs_served",
})

CONSUMER_WORD_MAX = 120
NUMBER_TOL = Decimal("0.000001")
EXEMPT_INT_MAX = 11
_NGRAM_PREFIXES = ("char__", "word__")
_QUOTE_CHARS = ('"', "\u201c", "\u201d")
_WS = re.compile(r"\s+")
_LABEL_SEP = " :: "

CHECKS: tuple[str, ...] = (
    "internal_error", "quoted_phrase", "team_mention", "issue_mention", "number",
    "cited_fact_path", "cited_fact_value", "required_team", "required_outcome",
    "opposite_outcome", "consumer_percent", "consumer_number",
    "consumer_banned_word", "consumer_model_name", "consumer_gate_feature",
    "consumer_neighbour", "consumer_length", "consumer_team_name",
    "consumer_issue_label")


# --------------------------------------------------------------------------
# Public types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class FailedCheck:
    check: str
    offending: str
    reason: str


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    failed: tuple[FailedCheck, ...] = ()


def _team_table() -> dict[str, str]:
    from src.explain.labels import load_tables
    return dict(sorted(load_tables().teams.items()))


def _issue_table() -> dict[str, str]:
    from src.explain.labels import load_tables
    return {k: e.label for k, e in sorted(load_tables().issues.items())}


@dataclass(frozen=True)
class ValidatorContext:
    """What the validator needs besides the Narration and the record.

    ``team_names`` maps team id -> name and ``issue_labels`` maps issue id ->
    full label; both default to ``src/explain/label_tables.json``.
    ``narrator_model`` is the Narrator_Config model id (banned in the
    Consumer_Summary). ``masked_text`` counts for quoted phrases only while
    ``include_text`` is true (Req 13.3).
    """
    masked_text: str | None = None
    include_text: bool = True
    team_names: Mapping[str, str] = field(default_factory=_team_table)
    issue_labels: Mapping[str, str] = field(default_factory=_issue_table)
    narrator_model: str = ""


def default_context(masked_text: str | None = None, include_text: bool = True,
                    narrator_model: str = "") -> ValidatorContext:
    """A context with the committed label tables."""
    return ValidatorContext(masked_text=masked_text, include_text=include_text,
                            narrator_model=narrator_model)


# --------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------

def norm(s: str) -> str:
    """``re.sub(r"\\s+", " ", s).strip().casefold()`` (design §16)."""
    return _WS.sub(" ", s).strip().casefold()


def _fold_char(c: str) -> str:
    """Case-fold one character, keeping the length at one character."""
    f = c.casefold()
    if len(f) == 1:
        return f
    low = c.lower()
    return low if len(low) == 1 else c


def _fold(s: str) -> str:
    """Whitespace collapsed and stripped, then case-folded char by char.

    Unlike ``norm`` this keeps one output character per input character, so
    positions found in the folded text are positions in the collapsed text.
    """
    return "".join(_fold_char(c) for c in _WS.sub(" ", s).strip())


@lru_cache(maxsize=4096)
def _guarded(term: str) -> re.Pattern:
    return re.compile(r"(?<!\w)" + re.escape(term) + r"(?!\w)")


def _contains(folded_text: str, folded_term: str) -> bool:
    return bool(folded_term) and _guarded(folded_term).search(folded_text) is not None


class _Work:
    """A summary with whitespace collapsed, as aligned raw and folded copies."""

    def __init__(self, text: str):
        collapsed = _WS.sub(" ", text).strip()
        self.raw = list(collapsed)
        self.fold = [_fold_char(c) for c in collapsed]

    def raw_str(self) -> str:
        return "".join(self.raw)

    def fold_str(self) -> str:
        return "".join(self.fold)

    def blank(self, start: int, end: int) -> None:
        for k in range(start, end):
            self.raw[k] = " "
            self.fold[k] = " "

    def copy(self) -> "_Work":
        w = _Work.__new__(_Work)
        w.raw, w.fold = list(self.raw), list(self.fold)
        return w


def _strip_prefix(label: str) -> str:
    for p in _NGRAM_PREFIXES:
        if label.startswith(p):
            return label[len(p):]
    return label


def _display(label: str) -> str:
    return label.split(_LABEL_SEP, 1)[-1]


@lru_cache(maxsize=1)
def _clean_text():
    try:
        from src.utils.text import clean_text
        return clean_text
    except Exception:          # an image without data/preparation
        return None


# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------

def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _quantize(x: Decimal, places: int) -> Decimal:
    with localcontext() as c:
        c.prec = max(28, x.adjusted() + places + 4)
        return x.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_EVEN)


def _decimals(body: str) -> int:
    return len(body.split(".", 1)[1]) if "." in body else 0


def _exempt(token: str) -> bool:
    """Integers 0..11 written without a decimal point or percent sign (Req 13.5a)."""
    if "." in token or "%" in token:
        return False
    return 0 <= int(token) <= EXEMPT_INT_MAX


class _Numbers:
    """Record numbers, rounded lazily per number of decimals."""

    def __init__(self, values: Iterable[int | float]):
        self.values = tuple(sorted({Decimal(repr(v)) for v in values if _is_num(v)}))
        self.pcts = tuple(v for v in self.values if 0 <= v <= 1)
        self._plain: dict[int, frozenset[Decimal]] = {}
        self._pct: dict[int, frozenset[Decimal]] = {}

    def _rounded(self, places: int, pct: bool) -> frozenset[Decimal]:
        cache = self._pct if pct else self._plain
        got = cache.get(places)
        if got is None:
            if pct:
                with localcontext() as c:
                    c.prec = 60
                    scaled = [v * 100 for v in self.pcts]
            else:
                scaled = list(self.values)
            got = frozenset(_quantize(v, places) for v in scaled)
            cache[places] = got
        return got

    def matches(self, token: str) -> bool:
        pct = token.endswith("%")
        body = token[:-1] if pct else token
        return Decimal(body) in self._rounded(_decimals(body), pct)


def number_matches(token: str, values: Iterable[int | float]) -> bool:
    """True when the written number ``token`` (``"0.878"``, ``"87.8%"``) matches a value."""
    return _Numbers(values).matches(token)


# --------------------------------------------------------------------------
# Record facts (computed once per call)
# --------------------------------------------------------------------------

def _walk(obj: Any, keys: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], Any]]:
    """Every leaf with the dict keys on its path; dict keys are yielded as ``str`` leaves too."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield keys, k
            yield from _walk(v, keys + (k,))
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v, keys)
    else:
        yield keys, obj


def _get(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _drivers(rec: dict) -> Iterator[dict]:
    sections = rec.get("sections") or {}
    for name in sorted(sections):
        sec = sections[name]
        lists = [_get(sec, "drivers")]
        if name == "improved_model_1":
            lists.append(_get(sec, "details", "issue", "drivers"))
        for ds in lists:
            for d in ds if isinstance(ds, list) else ():
                if isinstance(d, dict):
                    yield d


@dataclass
class _Name:
    team_ids: set[str] = field(default_factory=set)
    issue_ids: set[str] = field(default_factory=set)
    outcome: bool = False
    in_record: bool = False


class _Facts:
    def __init__(self, record: DecisionRecord | dict, ctx: ValidatorContext):
        rec = json.loads(canonical_json(hash_view(record)))
        self.rec = rec
        strings: set[str] = set()
        numbers: list[int | float] = []
        banned: list[int | float] = []
        for keys, leaf in _walk(rec):
            if isinstance(leaf, str):
                strings.add(leaf)
            elif _is_num(leaf):
                numbers.append(leaf)
                if any(k in BANNED_NUMERIC_KEYS for k in keys):
                    banned.append(leaf)
        self.strings = strings
        self.numbers = _Numbers(numbers)
        self.banned_numbers = _Numbers(banned)
        folded_strings = {_fold(s) for s in strings}

        out = rec["outcome"]
        team = out["chosen_team"]
        self.auto = bool(out["auto_route"])
        self.chosen_id = _WS.sub(" ", team["id"]).strip()
        self.chosen_name = _fold(team["name"])
        self.chosen_name_raw = team["name"]
        issues = out.get("top_issues") or []
        self.rank1 = ((_fold(issues[0]["label"]), _fold(issues[0]["display_label"]),
                       issues[0]["label"]) if issues else None)
        literals = {_fold(team["id"]), self.chosen_name}
        for iss in issues:
            literals |= {_fold(iss["issue_id"]), _fold(iss["label"]),
                         _fold(iss["display_label"])}
        self.outcome_literals = sorted((x for x in literals if x), key=lambda s: (-len(s), s))
        self.quoted_literals = [x for x in self.outcome_literals
                                if any(q in x for q in _QUOTE_CHARS)]

        names: dict[str, _Name] = {}
        for tid, tname in ctx.team_names.items():
            names.setdefault(_fold(tname), _Name()).team_ids.add(tid)
        for iid, label in ctx.issue_labels.items():
            names.setdefault(_fold(label), _Name()).issue_ids.add(iid)
        for lit in self.outcome_literals:
            names.setdefault(lit, _Name()).outcome = True
        names.pop("", None)
        for key, entry in names.items():
            entry.in_record = (entry.outcome or key in folded_strings
                               or bool((entry.team_ids | entry.issue_ids) & strings))
        self.names = sorted(names.items(), key=lambda kv: (-len(kv[0]), kv[0]))

        texts: set[str] = set()
        for d in _drivers(rec):
            if d.get("kind") == "ngram" and isinstance(d.get("label"), str):
                texts.add(norm(_strip_prefix(d["label"])))
            elif d.get("kind") == "sentence" and isinstance(d.get("text"), str):
                texts.add(norm(d["text"]))
        self.driver_texts = texts

        self.masked: tuple[str, ...] = ()
        mt = ctx.masked_text
        if ctx.include_text and isinstance(mt, str):
            forms = [norm(mt)]
            clean = _clean_text()
            if clean is not None:
                forms.append(norm(clean(mt)))
            self.masked = tuple(forms)

        gate = _get(rec, "sections", "gate", "drivers")
        self.gate_names = sorted({_fold(d["label"]) for d in gate or ()
                                  if isinstance(d, dict) and isinstance(d.get("label"), str)
                                  and _fold(d["label"])})

        terms: set[str] = set()
        for nb in _get(rec, "sections", "knn_embed", "details", "neighbours") or ():
            if not isinstance(nb, dict):
                continue
            for k in ("complaint_id", "issue_id"):
                if isinstance(nb.get(k), str):
                    terms.add(_fold(nb[k]))
            label = ctx.issue_labels.get(nb.get("issue_id")) if isinstance(
                nb.get("issue_id"), str) else None
            if isinstance(label, str):
                terms |= {_fold(label), _fold(_display(label))}
        terms.discard("")
        self.neighbour_terms = sorted(terms, key=lambda s: (-len(s), s))


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def _fc(check: str, offending: str, reason: str) -> FailedCheck:
    return FailedCheck(check=check, offending=offending, reason=reason)


def _blank_terms(work: _Work, terms: Iterable[str]) -> None:
    for term in terms:
        s = work.fold_str()
        for m in _guarded(term).finditer(s):
            work.blank(m.start(), m.end())


def _check_summary(field_name: str, text: str, facts: _Facts) -> list[FailedCheck]:
    out: list[FailedCheck] = []
    full = _Work(text)
    work = full.copy()

    # Step 1: quoted spans. Quote characters that belong to an outcome value
    # (a team name or issue label holding a quote) do not open or close a quote.
    scan = list(work.raw_str())
    for lit in facts.quoted_literals:
        for m in _guarded(lit).finditer(work.fold_str()):
            for k in range(m.start(), m.end()):
                if scan[k] in _QUOTE_CHARS:
                    scan[k] = "\x00"
    quotes: list[str] = []
    raw = work.raw_str()
    for m in QUOTE_RE.finditer("".join(scan)):
        g = 1 if m.group(1) is not None else 2
        quotes.append(raw[m.start(g):m.end(g)])
        work.blank(m.start(), m.end())
    outside_quotes = work.fold_str()

    # Step 2: team names, issue labels and outcome values, longest first.
    for key, entry in facts.names:
        s = work.fold_str()
        for m in _guarded(key).finditer(s):
            if not entry.in_record:
                kind = "team_mention" if entry.team_ids else "issue_mention"
                what = "team name" if entry.team_ids else "issue label"
                out.append(_fc(kind, m.group(),
                               f"{field_name}: {what} is not in the Decision_Record"))
            work.blank(m.start(), m.end())

    # Step 3: issue ids, then team ids.
    for regex, kind in ((ISSUE_ID_RE, "issue_mention"), (TEAM_ID_RE, "team_mention")):
        for m in regex.finditer(work.raw_str()):
            if m.group() not in facts.strings:
                out.append(_fc(kind, m.group(),
                               f"{field_name}: identifier is not in the Decision_Record"))
            work.blank(m.start(), m.end())

    # Step 4: numbers.
    for m in NUM_RE.finditer(work.fold_str()):
        tok = m.group()
        if not _exempt(tok) and not facts.numbers.matches(tok):
            out.append(_fc("number", tok, f"{field_name}: number matches no "
                           "Decision_Record value at the precision shown"))

    # Step 5: quoted phrases.
    for q in quotes:
        nq = norm(q)
        if nq in facts.driver_texts or any(nq in m for m in facts.masked):
            continue
        out.append(_fc("quoted_phrase", q, f"{field_name}: quoted text is not a "
                       "Driver or part of the complaint text"))

    # Step 6: Chosen_Team and routing outcome.
    # A blank Chosen_Team name cannot be stated; the Consumer_Summary names the
    # team only by name (Req 15.3), so it is then exempt from this check.
    cid, cname = facts.chosen_id, facts.chosen_name
    if (cid or cname) and not (field_name == "consumer_summary" and not cname):
        has_id = bool(cid) and _guarded(cid).search(full.raw_str()) is not None
        if not has_id and not _contains(full.fold_str(), cname):
            out.append(_fc("required_team", field_name,
                           "the summary does not name the Chosen_Team"))
    own, other = ((AUTO_PHRASES, REVIEW_PHRASES) if facts.auto
                  else (REVIEW_PHRASES, AUTO_PHRASES))
    if not any(p in outside_quotes for p in own):
        out.append(_fc("required_outcome", field_name,
                       f"the summary does not state the recorded outcome ({own[0]})"))
    for p in other:
        if p in outside_quotes:
            out.append(_fc("opposite_outcome", p,
                           f"{field_name}: states the opposite routing outcome"))
    return out


def _check_consumer(text: str, facts: _Facts, ctx: ValidatorContext) -> list[FailedCheck]:
    out: list[FailedCheck] = []
    full = _Work(text)
    work = full.copy()
    _blank_terms(work, facts.outcome_literals)      # outcome values are always allowed
    raw, folded = work.raw_str(), work.fold_str()

    if "%" in raw:
        out.append(_fc("consumer_percent", "%", "the Consumer_Summary contains '%'"))
    for m in NUM_RE.finditer(folded):
        tok = m.group()
        if not _exempt(tok) and facts.banned_numbers.matches(tok):
            out.append(_fc("consumer_number", tok, "number from a probability, score, "
                           "contribution or threshold field"))
    for m in BANNED_WORD_RE.finditer(folded):
        out.append(_fc("consumer_banned_word", m.group(), "banned word"))
    models = list(MODEL_NAMES)
    if isinstance(ctx.narrator_model, str) and ctx.narrator_model.strip():
        models.append(ctx.narrator_model)
    for name in models:
        f = _fold(name)
        if _contains(folded, f):
            out.append(_fc("consumer_model_name", name, "model name"))
    for g in facts.gate_names:
        if _contains(folded, g):
            out.append(_fc("consumer_gate_feature", g, "gate feature name"))
    for t in facts.neighbour_terms:
        if _contains(folded, t):
            out.append(_fc("consumer_neighbour", t,
                           "kNN neighbour complaint id or issue label"))

    n_words = len(text.split())
    if n_words > CONSUMER_WORD_MAX:
        out.append(_fc("consumer_length", f"{n_words} words",
                       f"the Consumer_Summary is longer than {CONSUMER_WORD_MAX} words"))
    whole = full.fold_str()
    if facts.chosen_name and not _contains(whole, facts.chosen_name):
        out.append(_fc("consumer_team_name", facts.chosen_name_raw,
                       "the Consumer_Summary does not state the Chosen_Team name"))
    if facts.rank1 is not None:
        label, disp, raw_label = facts.rank1
        if (label or disp) and not (_contains(whole, label) or _contains(whole, disp)):
            out.append(_fc("consumer_issue_label", raw_label, "the Consumer_Summary does "
                           "not state the rank-1 issue label"))
    return out


_MISSING = object()


def resolve_path(rec: Any, path: str) -> Any:
    """Follow a dotted path; digit segments index lists. ``_MISSING`` if absent."""
    cur = rec
    for seg in path.split("."):
        if isinstance(cur, dict):
            if seg not in cur:
                return _MISSING
            cur = cur[seg]
        elif isinstance(cur, list):
            if not (seg.isascii() and seg.isdigit()) or int(seg) >= len(cur):
                return _MISSING
            cur = cur[int(seg)]
        else:
            return _MISSING
    return cur


def _same_value(cited: Any, stored: Any) -> bool:
    if isinstance(cited, bool) or isinstance(stored, bool):
        return isinstance(cited, bool) and isinstance(stored, bool) and cited == stored
    if _is_num(cited) and _is_num(stored):
        with localcontext() as c:
            c.prec = 60
            return abs(Decimal(repr(cited)) - Decimal(repr(stored))) <= NUMBER_TOL
    if cited is None or stored is None:
        return cited is None and stored is None
    if isinstance(cited, str) or isinstance(stored, str):
        return isinstance(cited, str) and isinstance(stored, str) and cited == stored
    if isinstance(cited, list) and isinstance(stored, list):
        return canonical_json(cited) == canonical_json(stored)
    return False


def _check_cited(facts_list: Iterable[CitedFact], facts: _Facts) -> list[FailedCheck]:
    out: list[FailedCheck] = []
    for cf in facts_list:
        stored = resolve_path(facts.rec, cf.path)
        if stored is _MISSING:
            out.append(_fc("cited_fact_path", cf.path,
                           "the field path does not exist in the Decision_Record"))
        elif not _same_value(cf.value, stored):
            out.append(_fc("cited_fact_value", cf.path,
                           "the cited value differs from the stored value"))
    return out


def _validate(n: Narration, record: DecisionRecord | dict,
              ctx: ValidatorContext) -> list[FailedCheck]:
    facts = _Facts(record, ctx)
    out: list[FailedCheck] = []
    out += _check_summary("consumer_summary", n.consumer_summary, facts)
    out += _check_summary("agent_summary", n.agent_summary, facts)
    out += _check_cited(n.cited_facts, facts)
    out += _check_consumer(n.consumer_summary, facts, ctx)
    return out


def validate(n: Narration, record: DecisionRecord | dict,
             ctx: ValidatorContext) -> ValidationResult:
    """Check ``n`` against ``record`` (Req 13.1 to 13.11, 15.1 to 15.3, 15.5)."""
    try:
        failed = _validate(n, record, ctx)
        ordered = tuple(sorted(set(failed), key=lambda f: (f.check, f.offending, f.reason)))
    except Exception as e:  # Req 13.10: an internal error rejects the Narration
        return ValidationResult(accepted=False, failed=(
            FailedCheck("internal_error", type(e).__name__,
                        "the validator raised an internal error"),))
    return ValidationResult(accepted=not ordered, failed=ordered)
