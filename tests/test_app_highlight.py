"""Phrase highlighting in the complaint excerpt."""

from src.app.highlight import highlight
from src.app.theme import team_code


def test_marks_phrase_regardless_of_case_and_punctuation():
    out = highlight("My Escrow, payment was late.", ["escrow payment"])
    assert out == "My <mark>Escrow, payment</mark> was late."


def test_overlapping_phrases_merge_into_one_mark():
    out = highlight("the escrow payment shortage", ["escrow payment",
                                                     "payment shortage"])
    assert out == "the <mark>escrow payment shortage</mark>"


def test_whole_words_only():
    assert "<mark>" not in highlight("escrowed payments", ["escrow payment"])


def test_text_is_escaped_and_masked_tokens_styled():
    out = highlight("<b>call</b> XXXX on XX/XX/XXXX", [])
    assert "<b>" not in out and "&lt;b&gt;" in out
    assert out.count('class="cc-x"') == 2


def test_team_codes_are_unique_for_known_teams():
    from src.app.theme import TEAM_CODES
    assert len(set(TEAM_CODES.values())) == len(TEAM_CODES)
    assert team_code("T06", "Mortgage Team") == "MG"
    assert team_code(None, "Some New Team") == "SN"
