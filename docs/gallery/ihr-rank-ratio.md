# IHR rank selection: ratio: an air-quality walkthrough

[Gallery and data protocol](../real-world-examples.md) · [All examples](index.md) · [API](https://dx-li.github.io/MaVaTS/mavats.html#select_ihr_ranks)

## Question and applicability

What ranks does an oversized robust training pilot suggest?

The pilot rank is fixed at (2, 2); unknown true ranks may violate the oversized-pilot assumption. Pilot convergence and rank validity are different questions.

Paper attribution: [he2023ihr](../citations.md#he2023ihr). This is a new teaching application,
not reproduction of a paper's empirical results or endorsement by its authors.

## Data and execution

Observed Beijing data: (365, 4, 3) = day × station × pollutant.
Train: January–September 2014 (273 days). Held out: October–December (92 days).
The [shared protocol](../real-world-examples.md#data-and-preprocessing) specifies
provenance, missingness, training-only transformations and observational limits.

From a checkout with the examples extra installed:

```bash
python -m examples.air_quality --methods ihr-rank-ratio --output /tmp/mavats-gallery
```

The snippet includes shared preprocessing and the core call. Special setups
use `fit_case`, whose source explicitly defines constraints, lag alignment or
stream state. The executable [runner](../../examples/air_quality/__main__.py)
also contains the complete evaluation and plotting recipe.

```python
from examples.air_quality.data import load, preprocess
from examples.air_quality.cases import CASES
case = next(c for c in CASES if c.id == "ihr-rank-ratio")
dates, values, counts = load()
x, observed, preprocessing = preprocess(values, counts, train=273, tensor=False)
from mavats import select_ihr_ranks
result = select_ihr_ranks(x[:273], max_ranks=(2, 2), method='ratio')
```

## Computed result

Run status: **completed**. This records numerical execution, not
scientific validity. Unconverged output is diagnostic only; rejected fits
have no substituted estimate. [Full diagnostics](ihr-rank-ratio.json) include
warnings, convergence, preprocessing counts and method-specific quantities.

Training cells imputed: 142; held-out cells imputed: 62. Gaps in the observed trace mark insufficient readings; predictions over those gaps are not scored.



![IHR rank selection: ratio: computed output and diagnostics; see adjacent interpretation and JSON data](ihr-rank-ratio.png)

The first-cell display is predeclared (Aotizhongxin PM2.5, morning for tensors),
not selected for visual appeal. Forecast/reconstruction metrics use all observed
held-out cells, excluding imputed targets. Reconstruction sees the target day;
it must not be read as a forecast or measured recovery of an unknown clean signal.
Diagnostic-only plots can instead use training data as explicitly labelled.

[Numerical plot/score arrays](ihr-rank-ratio.npz) and [run provenance](provenance.json)
allow checking displayed results. No causal, health or regulatory conclusion follows.
