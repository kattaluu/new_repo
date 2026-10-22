"""Shared appearance of vision study reports and comparisons."""

import math

PURPLE = "#7570b3"


def validate_panel_positions(positions):
    if positions is None:
        return None
    try:
        normalized = tuple(tuple(float(value) for value in pair) for pair in positions)
        if len(normalized) != 2 or any(len(pair) != 2 for pair in normalized):
            raise ValueError
        if not all(math.isfinite(value) for pair in normalized for value in pair):
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError("panel_positions: ((x_left, y_left), (x_right, y_right)) или None") from None
    return normalized


def add_panel_label(axis, index, positions):
    if positions is not None:
        axis.text(*positions[index], "(а)" if index == 0 else "(б)", transform=axis.transAxes,
                  ha="left", va="bottom", fontsize=16, clip_on=False)


def vision_rc(font="Times New Roman"):
    return {"font.family": "serif", "font.serif": [font], "font.size": 14,
            "figure.titlesize": 16, "axes.titlesize": 16, "axes.labelsize": 16,
            "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 14,
            "axes.unicode_minus": False}
