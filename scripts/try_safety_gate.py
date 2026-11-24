"""Try the content safety gate (src/safety) by hand.

Run from the repo root:

    python scripts/try_safety_gate.py                  # built-in examples, offline
    python scripts/try_safety_gate.py --llm            # also the false-information check
    python scripts/try_safety_gate.py "some text"      # screen your own text
    python scripts/try_safety_gate.py --llm "some text"

Without --llm nothing leaves the machine. With --llm every complaint that
passes the word checks costs one call to the model configured by the
AGENTIC_* variables in the repo-root .env (see .env.example).

With the built-in examples the exit code is 1 if any result differs from
what is expected, so the script also works as a quick smoke test.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.safety import MisinformationJudge, SafetyGate  # noqa: E402

# (text, expected decision, expected first category or None)
OFFLINE_EXAMPLES = [
    ("My card was charged twice and the bank won't refund me.", "allow", None),
    ("This is fraud. They scammed me and lied about the interest rate!", "allow", None),
    ("What the hell is going on? This damn bank sucks.", "allow", None),
    ("This letter serves as a** formal demand for a$5 refund.", "allow", None),
    ("This is f*cking ridiculous.", "block", "profanity"),
    ("F u c k this bank.", "block", "profanity"),
    ("You are an idiot and so is your manager.", "block", "harassment"),
    ("I will kill the manager if they don't fix this.", "block", "threat"),
]

LLM_EXAMPLES = [
    ("A 2024 federal law says banks must forgive all credit card debt over "
     "$1,000, and my bank refuses to follow it.", "block", "misinformation"),
    ("The Federal Reserve was shut down in 2020, so my mortgage interest is "
     "illegal.", "block", "misinformation"),
    ("My bank charged me a $35 overdraft fee even though I had money in my "
     "account.", "allow", None),
    ("The debt collector keeps calling me about a loan I never took out.",
     "allow", None),
]


def build_gate(use_llm: bool, model: str | None, max_tokens: int,
               fail_closed: bool) -> SafetyGate:
    if not use_llm:
        return SafetyGate()
    from src.agentic.llm import get_backend
    kw = {"model": model} if model else {}
    backend = get_backend("openai_compat", **kw)
    print(f"LLM judge: model={backend.model}\n")
    return SafetyGate(judge=MisinformationJudge(backend, max_tokens=max_tokens),
                      fail_closed=fail_closed)


def show(text: str, result) -> None:
    print(f"[{result.decision.upper():5}] {text}")
    if result.categories:
        print(f"        categories: {', '.join(result.categories)}")
        print(f"        matched:    {', '.join(f.term for f in result.findings)}")
        print(f"        message:    {result.message}")
    if result.misinformation_checked:
        print(f"        llm reason: {result.judge_reason or '-'}")
    if result.judge_error:
        print(f"        llm ERROR:  {result.judge_error}")


def run_examples(gate: SafetyGate, examples) -> int:
    failures = 0
    for text, want_decision, want_category in examples:
        result = gate.screen(text)
        show(text, result)
        got_category = result.categories[0] if result.categories else None
        if (result.decision, got_category) != (want_decision, want_category):
            failures += 1
            print(f"        !! expected {want_decision}"
                  f"{f' ({want_category})' if want_category else ''}")
        print()
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="*", help="text to screen (default: built-in examples)")
    ap.add_argument("--llm", action="store_true",
                    help="also run the false-information check (calls the model)")
    ap.add_argument("--model", help="model id, overrides AGENTIC_MODEL")
    ap.add_argument("--max-tokens", type=int, default=512,
                    help="reply token limit for the judge (default 512)")
    ap.add_argument("--fail-closed", action="store_true",
                    help="block the complaint when the model check fails")
    args = ap.parse_args(argv)

    try:
        gate = build_gate(args.llm, args.model, args.max_tokens, args.fail_closed)
    except RuntimeError as e:   # no API key
        print(f"Cannot start the LLM judge: {e}", file=sys.stderr)
        return 2

    if args.text:
        show(" ".join(args.text), gate.screen(" ".join(args.text)))
        return 0

    examples = OFFLINE_EXAMPLES + (LLM_EXAMPLES if args.llm else [])
    failures = run_examples(gate, examples)
    print(f"{len(examples) - failures}/{len(examples)} examples as expected")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
