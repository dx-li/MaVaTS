# Tensor AR: PROJECTION: an air-quality walkthrough

[Gallery and data protocol](../real-world-examples.md) · [All examples](index.md) · [API](https://dx-li.github.io/MaVaTS/mavats.html#fit_tensor_ar)

## Question and applicability

Can station × pollutant × half-day patterns predict the next day's two half-days?

The third spatial mode is actual recorded half-day, not an arbitrary reshape. One term is fixed; no automatic term selection or inference is claimed.

Paper attribution: [li2021tenar](../citations.md#li2021tenar). This is a new teaching application,
not reproduction of a paper's empirical results or endorsement by its authors.

## Data and execution

Observed Beijing data: (365, 4, 3, 2) = day × station × pollutant × half-day.
Train: January–September 2014 (273 days). Held out: October–December (92 days).
The [shared protocol](../real-world-examples.md#data-and-preprocessing) specifies
provenance, missingness, training-only transformations and observational limits.

From a checkout with the examples extra installed:

```bash
python -m examples.air_quality --methods tensor-ar-projection --output /tmp/mavats-gallery
```

The snippet includes shared preprocessing and the core call. Special setups
use `fit_case`, whose source explicitly defines constraints, lag alignment or
stream state. The executable [runner](../../examples/air_quality/__main__.py)
also contains the complete evaluation and plotting recipe.

```python
from examples.air_quality.data import load, preprocess
from examples.air_quality.cases import CASES
case = next(c for c in CASES if c.id == "tensor-ar-projection")
dates, values, counts = load()
x, observed, preprocessing = preprocess(values, counts, train=273, tensor=True)
from mavats import fit_tensor_ar
result = fit_tensor_ar(x[:273], method='projection', terms=(1,), max_iter=200)
```

## Computed result

Run status: **completed**. This records numerical execution, not
scientific validity. Unconverged output is diagnostic only; rejected fits
have no substituted estimate. [Full diagnostics](tensor-ar-projection.json) include
warnings, convergence, preprocessing counts and method-specific quantities.

Training cells imputed: 233; held-out cells imputed: 129. Gaps in the observed trace mark insufficient readings; predictions over those gaps are not scored.

Computed forecast_mse: **1.6992** over 2079 observed target cells. Previous-day MSE: 2.3205; fixed-training-mean MSE: 1.7401. Lower is better on this split only; no uncertainty interval is estimated.

![Tensor AR: PROJECTION: computed output and diagnostics; see adjacent interpretation and JSON data](tensor-ar-projection.png)

The first-cell display is predeclared (Aotizhongxin PM2.5, morning for tensors),
not selected for visual appeal. Forecast/reconstruction metrics use all observed
held-out cells, excluding imputed targets. Reconstruction sees the target day;
it must not be read as a forecast or measured recovery of an unknown clean signal.
Diagnostic-only plots can instead use training data as explicitly labelled.

[Numerical plot/score arrays](tensor-ar-projection.npz) and [run provenance](provenance.json)
allow checking displayed results. No causal, health or regulatory conclusion follows.
