"""Unit tests for src/explain/explainers/__init__.py (task 4.2)."""
import pytest

from src.explain import record
from src.explain.explainers import (NOMINAL, SECTION_NAMES, error_section, run_explainer,
                                    select_drivers, skipped_section, validate_n)


def test_section_names_reexported():
    assert SECTION_NAMES is record.SECTION_NAMES
    assert set(NOMINAL) == set(SECTION_NAMES)


@pytest.mark.parametrize("name", SECTION_NAMES)
def test_skipped_section_carries_nominal(name):
    s = skipped_section(name, "not available on this serving path")
    assert (s.status, s.drivers, s.reason) == ("skipped", [], "not available on this serving path")
    assert (s.method, s.technique) == NOMINAL[name]
    assert isinstance(s, record.SECTION_MODELS[name])


def test_skipped_section_keeps_details_and_rejects_empty_reason():
    s = skipped_section("gate", "no reference medians", gate_kind="hgb")
    assert s.details.gate_kind == "hgb"
    with pytest.raises(ValueError):
        skipped_section("gate", "  ")


def test_error_section_reason_truncated():
    s = error_section("stacker", KeyError("x" * 900))
    assert s.status == "error" and s.drivers == []
    assert s.reason.startswith("KeyError: ")
    assert len(s.reason) == len("KeyError: ") + 500


def test_error_section_drops_invalid_details():
    s = error_section("routing", RuntimeError("boom"), not_a_field=1)
    assert s.reason == "RuntimeError: boom"


def test_run_explainer_never_reraises():
    def boom():
        raise ZeroDivisionError("division by zero")
    s = run_explainer("baseline_model_2", boom)
    assert (s.status, s.reason) == ("error", "ZeroDivisionError: division by zero")
    assert (s.method, s.technique) == ("exact", "nb_llr")

    wrong = run_explainer("knn_embed", lambda: skipped_section("gate", "x"))
    assert wrong.status == "error" and wrong.reason.startswith("TypeError")

    ok = run_explainer("gate", skipped_section, "gate", "why")
    assert ok.status == "skipped"


def test_select_drivers_per_sign_order_no_padding():
    contribs = [0.5, -0.5, 0.0, 0.2, -0.9, 0.5, 0.1]
    labels = ["b", "a", "z", "c", "d", "a", "e"]
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    out = select_drivers(contribs, labels, values, 2)
    assert [(d.label, d.contribution) for d in out] == [
        ("d", -0.9), ("a", 0.5), ("a", -0.5), ("b", 0.5)]
    assert out[1].value == 6.0
    # fewer than N of one sign: all reported, no padding; zeros dropped
    out = select_drivers(contribs, labels, values, 10)
    assert len(out) == 6 and all(d.contribution != 0 for d in out)
    assert select_drivers([0.0], ["x"], None, 3) == []


def test_select_drivers_sentence_kind():
    (d,) = select_drivers([0.3], ["A sentence."], None, 1, kind="sentence")
    assert (d.kind, d.text, d.label, d.value) == ("sentence", "A sentence.", None, None)


@pytest.mark.parametrize("name,ok,bad", [
    ("improved_model_1", [1, 10, 100], [0, 101, True, 2.0, "5"]),
    ("baseline_model_2", [1, 50], [0, 51, None]),
])
def test_validate_n(name, ok, bad):
    for n in ok:
        assert validate_n(name, n) == n
    for n in bad:
        with pytest.raises(ValueError):
            validate_n(name, n)
    with pytest.raises(ValueError):
        validate_n("improved_model_2", 5)
