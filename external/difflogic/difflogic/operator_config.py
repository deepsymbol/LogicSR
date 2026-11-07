"""Operator sets of LogicLayer.

A set lists the gates a LogicLayer may learn. The names and their order match the
table in functional.py and ALL_OPERATIONS in compiled_model.py.
"""

OPERATOR_SETS = {
    "minimal": ["zero", "and", "not_a", "not_b", "one"],
    "basic": ["zero", "and", "or", "not_a", "not_b", "one"],
    "extended": [
        "zero",
        "and",
        "not_implies",
        "a",
        "not_implied_by",
        "b",
        "xor",
        "or",
        "not_or",
        "not_xor",
        "not_b",
        "implied_by",
        "not_a",
        "implies",
        "not_and",
        "one",
    ],
}

ALL_OPERATOR_NAMES = OPERATOR_SETS["extended"]

ACTIVE_SET_NAME = "extended"
_config_version = 0


def set_operator_set(name):
    """Select the operator set used by LogicLayer."""
    global ACTIVE_SET_NAME, _config_version
    if name not in OPERATOR_SETS:
        raise ValueError(
            "Unknown operator set '{}'. Available: {}".format(name, sorted(OPERATOR_SETS)))
    ACTIVE_SET_NAME = name
    _config_version += 1


def get_active_operator_indices():
    """Return the indices (0-15) of the gates in the active operator set."""
    return [ALL_OPERATOR_NAMES.index(op) for op in OPERATOR_SETS[ACTIVE_SET_NAME]]


def get_config_version():
    """Return a counter that changes whenever the operator set is selected."""
    return _config_version
