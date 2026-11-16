"""LLM candidate choice for the complaints the ML models are unsure about.

Two tasks, both "pick one of these candidates":
  team   on the review queue: the stacker's top-3 teams
  issue  where the ML issue head's top-2 gap is small: its top-3 issues

The ML decides *whether* to ask and *what the options are*; the LLM only
chooses among them, grounded in retrieved labelled examples per candidate
and the team descriptions written for the Jev work. Every call is cached by
prompt hash, so re-running an evaluation costs nothing.
"""
