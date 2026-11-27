import numpy as np
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


# ---------- probabilistic metrics ----------
# column order is FIT_LABELS = small, fit, large

def test_average_auc_perfect_and_constant():
    perfect = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    assert evaluate.average_auc(Y, perfect) == pytest.approx(1.0)
    constant = np.full((4, 3), 1 / 3)
    assert evaluate.average_auc(Y, constant) == pytest.approx(0.5)


def test_average_auc_known_value():
    # "small" column ranks the small row 2nd of 4 -> AUC_small = 2/3 ; others perfect
    p = np.array([[0.9, 0.9, 0.0], [0.1, 0.8, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])
    assert evaluate.average_auc(Y, p) == pytest.approx((2 / 3 + 1 + 1) / 3)


def test_average_auc_shape_mismatch_raises():
    with pytest.raises(ValueError):
        evaluate.average_auc(Y, np.ones((3, 3)))


def test_decide_applies_class_weights():
    p = np.array([[0.3, 0.6, 0.1]])
    assert evaluate.decide(p) == ["fit"]
    assert evaluate.decide(p, weights=(3.0, 1.0, 1.0)) == ["small"]


def test_tune_weights_beats_argmax_on_imbalanced_scores():
    # minority classes never win argmax, but rank correctly -> tuning should recover them
    y = ["fit"] * 6 + ["small", "large"]
    p = np.array([[0.1, 0.8, 0.1]] * 6 + [[0.35, 0.6, 0.05], [0.05, 0.6, 0.35]])
    assert evaluate.macro_f1(y, evaluate.decide(p)) < 0.5
    w = evaluate.tune_weights(y, p)
    assert evaluate.macro_f1(y, evaluate.decide(p, w)) == pytest.approx(1.0)


def test_paired_bootstrap_difference():
    rng = np.random.default_rng(0)
    y = list(rng.choice(["small", "fit", "large"], 400, p=[0.15, 0.7, 0.15]))
    onehot = np.array([[c == lab for lab in ["small", "fit", "large"]] for c in y], dtype=float)
    good = 0.7 * onehot + 0.3 * rng.random((400, 3))
    noise = rng.random((400, 3))
    lo, hi = evaluate.bootstrap_auc_difference(y, good, noise, n_boot=200)
    assert 0 < lo < hi  # clearly better -> interval excludes 0
    lo, hi = evaluate.bootstrap_auc_difference(y, good, good, n_boot=50)
    assert lo == pytest.approx(0) and hi == pytest.approx(0)


def test_report_with_proba_and_table_sorted():
    proba = np.full((4, 3), 1 / 3)
    rows = [evaluate.report("always fit", Y, ["fit"] * 4, proba), evaluate.report("perfect", Y, Y)]
    assert set(rows[0]) == {"model", "Avg AUC", "Macro-F1", "Accuracy", "F1 small", "F1 fit", "F1 large"}
    assert np.isnan(rows[1]["Avg AUC"])
    assert list(evaluate.results_table(rows)["model"]) == ["perfect", "always fit"]
