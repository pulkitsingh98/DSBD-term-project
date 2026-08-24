"""
Turning a probability into a decision: what does each threshold actually cost?

Statistical quality (ROC-AUC, F1) says how well the model separates claims from
non-claims. It cannot say where to draw the line, because that depends on two
numbers the model knows nothing about: what it costs to act on a flagged
shipment, and how much damage acting actually prevents. Neither exists in the
dataset, so both arrive as arguments and are exposed as sliders on screen.

Everything here is a backtest against *realised* outcomes on the held-out test
set - what each shipment actually cost, not what a model predicted it would.
"""

from __future__ import annotations

import numpy as np


def total_cost(flagged, y_true, amounts, cost_per_flagged: float,
               effectiveness: float) -> float:
    """
    What one decision rule would have cost over the test set.

        cost = intervention spend
             + full loss on claims we did not flag
             + residual loss on claims we did flag

    A flagged shipment costs the intervention whether or not it was going to be
    damaged; a flagged shipment that *was* damaged still costs the share of the
    loss the intervention failed to prevent.
    """
    flagged = np.asarray(flagged, dtype=bool)
    claimed = np.asarray(y_true).astype(bool)
    amounts = np.asarray(amounts, dtype=float)

    spend = cost_per_flagged * float(flagged.sum())
    missed = float(amounts[claimed & ~flagged].sum())
    residual = (1.0 - effectiveness) * float(amounts[claimed & flagged].sum())
    return spend + missed + residual


def cost_curve(y_true, amounts, probs, cost_per_flagged: float,
               effectiveness: float, step: float = 0.005) -> dict:
    """
    Sweep every threshold and find the cheapest one.

    Returns the whole curve so the page can draw it, plus the do-nothing
    baseline - flagging nothing is always an available option, and at a high
    enough intervention cost it is the right one.
    """
    probs = np.asarray(probs, dtype=float)
    thresholds = np.round(np.arange(step, 1.0 + step, step), 4)
    costs = np.array([
        total_cost(probs >= t, y_true, amounts, cost_per_flagged, effectiveness)
        for t in thresholds
    ])

    best = int(np.argmin(costs))
    do_nothing = total_cost(np.zeros(len(probs), dtype=bool), y_true, amounts,
                            cost_per_flagged, effectiveness)
    return {
        "thresholds": thresholds.tolist(),
        "costs": costs.round(2).tolist(),
        "optimal_threshold": float(thresholds[best]),
        "optimal_cost": float(costs[best]),
        "optimal_flagged": int((probs >= thresholds[best]).sum()),
        "optimal_flagged_pct": float((probs >= thresholds[best]).mean()),
        "do_nothing_cost": float(do_nothing),
        "saving_vs_do_nothing": float(do_nothing - costs[best]),
        "saving_pct": float((do_nothing - costs[best]) / do_nothing) if do_nothing else 0.0,
    }


def closed_form_threshold(cost_per_flagged: float, effectiveness: float,
                          mean_claim: float) -> float:
    """
    The optimal threshold in one line, for a shipment of average claim size.

    Act when the damage you expect to prevent is worth more than the action:

        effectiveness x p x claim  >  cost      =>      p  >  cost / (effectiveness x claim)

    This is the textbook result, and it lands within a couple of hundredths of
    the empirical minimum above - worth showing side by side, because agreement
    between the formula and the backtest is the reassurance that neither is a
    fluke.
    """
    denominator = effectiveness * mean_claim
    if denominator <= 0:
        return 1.0
    return float(min(cost_per_flagged / denominator, 1.0))


def expected_loss_rule(y_true, amounts, probs, values, cost_per_flagged: float,
                       effectiveness: float) -> dict:
    """
    Rank by expected loss rather than probability alone.

    A single probability cut-off treats a 10% chance of a Rs 100,000 claim as
    less urgent than a 30% chance of a Rs 2,000 one, which is backwards in
    money terms. Combining the two models fixes that: act when

        effectiveness x p x predicted_value  >  cost

    There is no threshold to choose here - the rule falls out of the same two
    business assumptions.
    """
    probs = np.asarray(probs, dtype=float)
    values = np.asarray(values, dtype=float)
    expected = probs * values
    flagged = (effectiveness * expected) > cost_per_flagged
    return {
        "cost": total_cost(flagged, y_true, amounts, cost_per_flagged, effectiveness),
        "flagged": int(flagged.sum()),
        "flagged_pct": float(flagged.mean()),
        "min_expected_loss_acted_on": (float(expected[flagged].min())
                                       if flagged.any() else None),
    }


def compare(y_true, amounts, probs, values, cost_per_flagged: float,
            effectiveness: float) -> dict:
    """Full picture for one pair of business assumptions."""
    curve = cost_curve(y_true, amounts, probs, cost_per_flagged, effectiveness)
    el = expected_loss_rule(y_true, amounts, probs, values,
                            cost_per_flagged, effectiveness)
    best_threshold_cost = curve["optimal_cost"]
    improvement = best_threshold_cost - el["cost"]
    return {
        "curve": curve,
        "expected_loss_rule": el,
        "el_improvement": float(improvement),
        "el_improvement_pct": (float(improvement / best_threshold_cost)
                               if best_threshold_cost else 0.0),
        "el_wins": bool(improvement > 0),
    }
