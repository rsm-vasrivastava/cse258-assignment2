import pytest

from a2 import evaluate

Y = ["fit", "fit", "small", "large"]


def test_perfect_predictions():
    assert evaluate.accuracy(Y, Y) == 1
    assert evaluate.macro_f1(Y, Y) == 1


def test_always_fit_has_high_accuracy_but_low_macro_f1():
    pred = ["fit"] * 4
    assert evaluate.accuracy(Y, pred) == 0.5
    f1 = evaluate.f1_per_class(Y, pred)
    assert f1 == {"small": 0.0, "fit": pytest.approx(2 * 2 / (2 * 2 + 2 + 0)), "large": 0.0}
    assert evaluate.macro_f1(Y, pred) == pytest.approx((2 / 3) / 3)


@pytest.mark.parametrize("y,p", [([], []), (["fit"], [])])
def test_bad_inputs_raise(y, p):
    with pytest.raises(ValueError):
        evaluate.accuracy(y, p)


def test_report_and_table_sorted_by_macro_f1():
    rows = [evaluate.report("always fit", Y, ["fit"] * 4), evaluate.report("perfect", Y, Y)]
    assert set(rows[0]) == {"model", "Macro-F1", "Accuracy", "F1 small", "F1 fit", "F1 large"}
    assert list(evaluate.results_table(rows)["model"]) == ["perfect", "always fit"]
