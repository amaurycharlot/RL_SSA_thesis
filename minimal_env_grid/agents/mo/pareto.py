"""Pure two-objective maximization geometry, with exact comparisons.

NumPy is lazy; no environments, plotting, training or external services are
imported. Input coordinates and policy identities are never mutated or rounded.
"""
from collections.abc import Sequence
import math
from numbers import Integral


# Params: points; returns: points.
def _validate_points(points):
    import numpy as np
    if not isinstance(points, np.ndarray) or points.dtype.kind != "f":
        raise TypeError("points must be a floating NumPy array; no implicit cast is performed")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (M,2), including (0,2) for an empty set")
    if not np.all(np.isfinite(points)):
        raise ValueError("points must be finite")
    return points


def non_dominated_mask(points):
    """Return bool[M]: no other row is >= in both and > in at least one.

    Equal points never strictly dominate each other. Thus all nondominated
    duplicate rows remain in the all-points mask, preserving every policy ID.
    Uses O(M**2) comparisons and O(M) temporary storage for the small policy set.
    """
    import numpy as np
    points = _validate_points(points)
    result = np.ones(len(points), dtype=bool)
    for index, point in enumerate(points):
        dominates = np.all(points >= point, axis=1) & np.any(points > point, axis=1)
        result[index] = not np.any(dominates)
    return result


def _validate_identifier(value):
    # Stable policy IDs may be weight indices, bundle IDs or compound slot IDs.
    # Avoid NaN/object identities and randomized hashes as ordering keys.
    if isinstance(value, Integral) and not isinstance(value, bool):
        if value < 0:
            raise ValueError("Integer stable IDs must be nonnegative")
        return
    if isinstance(value, str) and value:
        return
    if isinstance(value, tuple) and value:
        for component in value:
            _validate_identifier(component)
        return
    raise TypeError("Stable IDs must be nonnegative integers, nonempty strings or tuples of them")


# Params: stable_ids, count; returns: identifiers.
def _validate_ids(stable_ids, count):
    import numpy as np
    if isinstance(stable_ids, np.ndarray):
        if stable_ids.ndim != 1:
            raise ValueError("stable_ids must be one-dimensional")
    elif not isinstance(stable_ids, Sequence) or isinstance(stable_ids, (str, bytes)):
        raise TypeError("stable_ids must be a sequence of policy IDs")
    identifiers = tuple(stable_ids)
    if len(identifiers) != count:
        raise ValueError("One stable ID is required per point row")
    for identifier in identifiers:
        _validate_identifier(identifier)
    if len(set(identifiers)) != count:
        raise ValueError("Stable policy IDs must be unique")
    try:
        sorted(identifiers)
    except TypeError as error:
        raise TypeError("Stable policy IDs must be mutually orderable") from error
    return identifiers


def unique_front_indices(points, stable_ids):
    """Return original row indices of the unique front, ascending objective 0.

    For exact duplicate coordinates choose the lowest stable policy ID, never
    the first input row. Distinct nearby points remain distinct; no tolerance.
    Returned indices refer to the original all-points/ID tables, which callers
    retain for duplicate associations. This does not make statistical claims.
    """
    import numpy as np
    points = _validate_points(points)
    identifiers = _validate_ids(stable_ids, len(points))
    front_rows = np.flatnonzero(non_dominated_mask(points))
    representatives = {}
    for index in front_rows:
        coordinates = tuple(points[index])
        existing = representatives.get(coordinates)
        if existing is None or identifiers[index] < identifiers[existing]:
            representatives[coordinates] = int(index)
    selected = sorted(representatives.values(), key=lambda index: points[index, 0])
    return np.asarray(selected, dtype=np.intp)


# Params: reference; returns: values.
def _validate_reference(reference):
    import numpy as np
    if isinstance(reference, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in reference):
        raise TypeError("reference components must be real numbers, not bools")
    values = np.asarray(reference)
    if values.shape != (2,):
        raise ValueError("reference must have exactly two components")
    if values.dtype.kind not in "fiu":
        raise TypeError("reference must contain real numeric components")
    if not np.all(np.isfinite(values)):
        raise ValueError("reference must be finite")
    return values


# Params: points, reference; returns: (points, reference, bool).
def _reference_contributors(points, reference):
    import numpy as np
    points = _validate_points(points)
    reference = _validate_reference(reference)
    return points, reference, np.all(points > reference, axis=1)


def hypervolume_reference_diagnostics(points, reference) -> dict:
    """Report zero-area reference exclusions using original all-points indices.

    A point must be strictly above the fixed reference in both coordinates to
    have positive rectangle area. Dominated/duplicate points above it are not
    reference exclusions; their overlapping area is handled by the union.
    Returned lists let reporting retain the matching original policy IDs.
    """
    import numpy as np
    points, _, contributes = _reference_contributors(points, reference)
    included = np.flatnonzero(contributes).tolist()
    excluded = np.flatnonzero(~contributes).tolist()
    return {"n_points": len(points), "n_reference_contributors": len(included),
            "n_noncontributing_points": len(excluded),
            "contributing_indices": included, "noncontributing_indices": excluded}


def hypervolume_2d_max(points, reference) -> float:
    """Exact rectangle-union geometry for two maximized objectives.

    Use the caller's fixed reference unchanged. Outside/boundary points have
    zero area; duplicate/dominated rectangles add no area. Arithmetic uses
    Python floats and compensated summation, avoiding float32 area overflow.
    Return 0.0 for empty contributors; reject a result outside finite float
    range rather than claiming a valid infinite hypervolume. Diagnostics are
    available separately through hypervolume_reference_diagnostics().
    """
    points, reference, contributes = _reference_contributors(points, reference)
    contributors = points[contributes]
    if not len(contributors):
        return 0.0
    front = contributors[unique_front_indices(contributors, range(len(contributors)))]
    previous_x, reference_y = float(reference[0]), float(reference[1])
    rectangles = []
    for x, y in front:
        current_x = float(x)
        rectangles.append((current_x - previous_x) * (float(y) - reference_y))
        previous_x = current_x
    try:
        area = math.fsum(rectangles)
    except OverflowError as error:
        raise OverflowError("Hypervolume exceeds finite float range") from error
    if not math.isfinite(area):
        raise OverflowError("Hypervolume exceeds finite float range")
    return float(area)


# Params: actual_steps, n_budgets; returns: steps.
def _validate_actual_steps(actual_steps, n_budgets):
    import numpy as np
    if isinstance(actual_steps, (list, tuple)) and any(isinstance(value, (bool, np.bool_)) for value in actual_steps):
        raise TypeError("actual_steps must be integer budgets, not bools")
    steps = np.asarray(actual_steps)
    if steps.shape != (n_budgets,):
        raise ValueError("One actual training step is required per budget column")
    if steps.dtype.kind not in "iu":
        raise TypeError("actual_steps must contain integers")
    if np.any(steps <= 0) or np.any(steps[1:] <= steps[:-1]):
        raise ValueError("actual_steps must be positive, sorted and unique")
    return steps


def compute_hv_curve(results, reference):
    """Return float64[S,K] union areas of complete scaled policy sets.

    All W policy slots must be valid and finite within the same training seed
    and actual budget. Missing sets stay NaN; real zero returns remain valid.
    Recompute from scaled means, never trust cached HV, average policy areas,
    pool seeds, or mutate results/cache/identity tables.
    """
    import numpy as np
    from .types import ExperimentResults
    if not isinstance(results, ExperimentResults):
        raise TypeError("results must be ExperimentResults")
    results.validate()
    reference = _validate_reference(reference)
    n_seeds, n_budgets, _ = results.validity.shape
    _validate_actual_steps(results.actual_budgets, n_budgets)
    curve = np.full((n_seeds, n_budgets), np.nan, dtype=np.float64)
    for seed in range(n_seeds):
        for budget in range(n_budgets):
            if results.validity[seed, budget].all():
                curve[seed, budget] = hypervolume_2d_max(results.means_scaled[seed, budget], reference)
    return curve


def summarize_hv_across_seeds(hv, actual_steps):
    """Return HVSummary for the full declared float[S,K] seed matrix.

    actual_steps is explicit: never invent budget labels from column indices.
    NaN marks missing seeds, including wholly absent replicate rows. The main
    mean/sample SD require all S seeds, with SD zero for S=1. Complete counts
    are always reported; no available-seed mean is substituted or interpolated.
    """
    import numpy as np
    from .types import HVSummary
    if not isinstance(hv, np.ndarray) or hv.dtype.kind != "f":
        raise TypeError("hv must be a floating NumPy array")
    if hv.ndim != 2 or min(hv.shape) < 1:
        raise ValueError("hv must have nonempty shape (S,K)")
    if np.any(np.isinf(hv)) or np.any(hv[np.isfinite(hv)] < 0):
        raise ValueError("Hypervolume must be nonnegative and finite, or NaN when missing")
    n_seeds, n_budgets = hv.shape
    steps = _validate_actual_steps(actual_steps, n_budgets)
    counts = np.isfinite(hv).sum(axis=0, dtype=np.int64)
    complete = counts == n_seeds
    means = np.full(n_budgets, np.nan, dtype=np.float64)
    sds = np.full(n_budgets, np.nan, dtype=np.float64)
    for budget in np.flatnonzero(complete):
        values = hv[:, budget]
        # Scale before summation/squaring to avoid unnecessary overflow of
        # finite means/SDs. hypot computes the Euclidean norm without squaring
        # large deviations directly; dividing first keeps the sample SD finite.
        means[budget] = math.fsum(float(value) / n_seeds for value in values)
        sds[budget] = (math.hypot(*((float(value) - means[budget]) / math.sqrt(n_seeds - 1)
                                  for value in values)) if n_seeds > 1 else 0.0)
    return HVSummary(actual_steps=steps, mean=means, sd=sds, complete_seed_counts=counts,
                     declared_seed_count=n_seeds, completeness=complete)
