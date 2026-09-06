"""Explicit training-subsample IC stability, with no hidden grid selection.

Reference: Han, Chen and Zhang (2022), Rank Determination in Tensor Factor
Model, https://arxiv.org/html/2011.07131v3, Remark 8 and Section 5.4.
"""

import numpy as np

from mavats.rank_selection import select_tensor_rank
from mavats.rank_stability import tensor_rank_stability


def main():
    rng = np.random.default_rng(97)
    factors = np.zeros(150)
    for t in range(1, len(factors)):
        factors[t] = 0.8 * factors[t - 1] + rng.normal()
    data = factors[:, None, None] * np.ones((1, 6, 7))
    data += 0.1 * rng.normal(size=data.shape)
    path = tensor_rank_stability(
        data,
        np.geomspace(1e-6, 100, 40),
        spatial_subsets=[
            ([4, 0, 3], [6, 2, 0]),
            ([4, 0, 3, 1], [6, 2, 0, 4]),
            (list(range(6)), list(range(7))),
        ],
        time_prefixes=[100, 125, 150],
        max_ranks=(2, 2),
        iterative=False,
    )
    choice = path.choose_plateaus(variance_tolerance=0, minimum_grid_points=3)
    print("Rank path shape (mode, c, subsample):", path.ranks.shape)
    print("All cell fits converged:", bool(path.converged.all()))
    print("Selected grid indices:", choice.selected_indices)
    print("Selected modewise multipliers:", choice.penalty_multipliers)
    print("Selected full-sample path ranks:", choice.selected_ranks)
    if choice.complete:
        fitted = select_tensor_rank(
            data, penalty_multiplier=choice.penalty_multipliers, **path.options
        )
        print("Joint refit ranks/convergence:", fitted.ranks, fitted.converged)
    else:
        print("No admissible region in every mode:", choice.reasons)
    print("A finite-grid stability choice is not a certificate of the true rank.")


if __name__ == "__main__":
    main()
