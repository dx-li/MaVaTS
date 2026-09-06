"""Separate matrix components, forecast each block, and invert the forecasts.

Run from the repository root: python -m examples.decorrelation
"""

import numpy as np

from mavats.decorrelation import fit_matrix_decorrelation


def main():
    rng = np.random.default_rng(271)
    n_time = 10000
    coefficients = np.array([[0.92, 0.6, 0.2], [0.75, 0.4, 0.1], [0.55, 0.25, -0.1]])
    latent = np.zeros((n_time, 3, 3))
    noise = rng.normal(size=latent.shape) * np.sqrt(1 - coefficients**2)
    for t in range(1, n_time):
        latent[t] = coefficients * latent[t - 1] + noise[t]
    row_mixing = np.array([[1, 0.3, -0.2], [0.1, 1, 0.4], [0.2, -0.2, 1]])
    column_mixing = np.array([[1, 0.4, 0.1], [-0.2, 1, 0.3], [0.1, 0.2, 1]])
    observations = row_mixing @ latent @ column_mixing.T + 3

    # Choose all tuning parameters using training observations only. These
    # values suit this planted example; 0.12 is not a significance threshold.
    train, test = observations[:-100], observations[-100:]
    model = fit_matrix_decorrelation(
        train, lags=1, correlation_lags=5, correlation_threshold=0.12
    )
    print("Row groups:", model.row_groups)
    print("Column groups:", model.column_groups)
    # The paper's ratio heuristic must select at least one edge in each
    # nontrivial mode, so cannot recover all singletons in this example.
    ratio_model = fit_matrix_decorrelation(
        train, lags=1, correlation_lags=5, grouping="ratio"
    )
    print("Ratio-selected row groups:", ratio_model.row_groups)
    print("Ratio-selected column groups:", ratio_model.column_groups)
    print(
        "Round-trip error:",
        np.max(np.abs(model.inverse_blocks(model.blocks()) - train)),
    )

    # Each block gets an ordinary least-squares VAR(1). This example deliberately
    # keeps forecasting separate from transformation and uses lagged observed
    # test data for rolling one-step predictions, with no parameter refitting.
    training_blocks = model.blocks()
    predictor_blocks = model.blocks(np.concatenate([train[-1:], test[:-1]]))
    predictions = []
    for block_row, predictors_row in zip(training_blocks, predictor_blocks):
        row_predictions = []
        for block, predictors in zip(block_row, predictors_row):
            flattened = block.reshape(len(block), -1)
            design = np.column_stack([flattened[:-1], np.ones(len(block) - 1)])
            coefficient = np.linalg.lstsq(design, flattened[1:], rcond=None)[0]
            forecast_design = np.column_stack(
                [predictors.reshape(len(predictors), -1), np.ones(len(predictors))]
            )
            row_predictions.append(
                (forecast_design @ coefficient).reshape(predictors.shape)
            )
        predictions.append(row_predictions)
    forecast = model.inverse_blocks(predictions)
    print("Rolling one-step matrix MSE:", np.mean((forecast - test) ** 2))
    print("Training-mean baseline MSE:", np.mean((model.mean - test) ** 2))


if __name__ == "__main__":
    main()
