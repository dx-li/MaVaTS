# ITOPUP tensor factors: an air-quality walkthrough

[Gallery and data protocol](../real-world-examples.md) · [All examples](index.md) · [API](https://dx-li.github.io/MaVaTS/mavats.html#fit_tensor_factor)

## Question and applicability

Can shared station, pollutant and half-day spaces summarize the daily tensor?

Scores use each held-out observation itself: this is reconstruction, not forecasting. The unobserved clean signal is unknown, so discrepancy is not denoising accuracy. Lagged identification requires temporal factor signal; TIPUP can suffer signal cancellation.

Paper attribution: [chen2022tensorfactor](../citations.md#chen2022tensorfactor), [han2024iterative](../citations.md#han2024iterative). This is a new teaching application,
not reproduction of a paper's empirical results or endorsement by its authors.

## Data and execution

Observed Beijing data: (365, 4, 3, 2) = day × station × pollutant × half-day.
Train: January–September 2014 (273 days). Held out: October–December (92 days).
The [shared protocol](../real-world-examples.md#data-and-preprocessing) specifies
provenance, missingness, training-only transformations and observational limits.

From a checkout with the examples extra installed:

```bash
python -m examples.air_quality --methods itopup --output /tmp/mavats-gallery
```

The snippet includes shared preprocessing and the core call. Special setups
use `fit_case`, whose source explicitly defines constraints, lag alignment or
stream state. The executable [runner](../../examples/air_quality/__main__.py)
also contains the complete evaluation and plotting recipe.

```python
from examples.air_quality.data import load, preprocess
from examples.air_quality.cases import CASES
case = next(c for c in CASES if c.id == "itopup")
dates, values, counts = load()
x, observed, preprocessing = preprocess(values, counts, train=273, tensor=True)
from mavats import fit_tensor_factor
result = fit_tensor_factor(x[:273], ranks=(1, 1, 1), method='topup', iterative=True)
```

## Computed result

Run status: **completed**. This records numerical execution, not
scientific validity. Unconverged output is diagnostic only; rejected fits
have no substituted estimate. [Full diagnostics](itopup.json) include
warnings, convergence, preprocessing counts and method-specific quantities.

Training cells imputed: 233; held-out cells imputed: 129. Gaps in the observed trace mark insufficient readings; predictions over those gaps are not scored.

Computed reconstruction_discrepancy: **0.5898** over 2079 observed target cells. 

![ITOPUP tensor factors: computed output and diagnostics; see adjacent interpretation and JSON data](itopup.png)

The first-cell display is predeclared (Aotizhongxin PM2.5, morning for tensors),
not selected for visual appeal. Forecast/reconstruction metrics use all observed
held-out cells, excluding imputed targets. Reconstruction sees the target day;
it must not be read as a forecast or measured recovery of an unknown clean signal.
Diagnostic-only plots can instead use training data as explicitly labelled.

[Numerical plot/score arrays](itopup.npz) and [run provenance](provenance.json)
allow checking displayed results. No causal, health or regulatory conclusion follows.
