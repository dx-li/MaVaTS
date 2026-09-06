"""Published dynamic rank criteria, not an adjacent-gap heuristic.

Run from the repository root: python examples/factor_rank_selection.py
Reference: Han, Chen and Zhang (2022), Rank Determination in Tensor Factor
Model. https://doi.org/10.1214/22-EJS1991
"""

import numpy as np

from mavats.rank_selection import select_tensor_rank


def main():
    rng = np.random.default_rng(85)
    row = np.linalg.qr(rng.normal(size=(8, 2)))[0] * np.sqrt(8)
    column = np.linalg.qr(rng.normal(size=(9, 2)))[0] * 3
    core = np.zeros((600, 2, 2))
    for t in range(1, len(core)):
        core[t] = 0.8 * core[t - 1] + rng.normal(size=(2, 2))
    signal = np.einsum("ia,tab,jb->tij", row, core, column)
    observed = signal + 0.4 * rng.normal(size=signal.shape)
    train = observed[:450]
    for method in ("topup", "tipup"):
        for criterion, penalty in (("ic", 2), ("er", 1)):
            fit = select_tensor_rank(
                train,
                method=method,
                criterion=criterion,
                penalty=penalty,
                iterative=True,
            )
            denoised = fit.inverse_transform(fit.transform(observed[450:]))
            error = np.linalg.norm(denoised - signal[450:]) / np.linalg.norm(
                signal[450:]
            )
            print(
                f"{fit.method} {criterion}{penalty}: ranks={fit.ranks}, "
                f"sweeps={fit.n_iter}, converged={fit.converged}, "
                f"held-out denoising error={error:.4f}"
            )
    # Denoising uses the current held-out observation; this is NOT a forecast.
    print("initial criterion ranks:", fit.initial_ranks)
    print("conservative projection ranks:", fit.starting_ranks)
    print("rank path:", fit.rank_history)
    print("final physical log penalties:", [s.log_penalty for s in fit.diagnostics])
    zero = select_tensor_rank(np.zeros((20, 4, 5)), criterion="ic")
    print("IC no-factor example:", zero.ranks, "core shape:", zero.factors.shape)


if __name__ == "__main__":
    main()
