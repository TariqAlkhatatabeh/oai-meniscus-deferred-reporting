# Reproducibility limitation

The repository contains the final code and aggregate results, but not the
row-level tables that would allow every count to be recomputed. The withheld
files are per-knee predictions, per-report audit rows, generated reports,
per-exam MRNet scores, Grad-CAM case selections, the fixed participant split
file and the MRNet adaptation split file. They remain subject to the OAI and
MRNet data-use terms.

The aggregate outputs are in `results/metrics/`, `results/reporting/`,
`results/vlm_eval/`, `results/fusion/`, `results/source_data/` and
`results/cohort/`. Every reported estimate and confidence interval can be read
from those files. Counts that require row linkage cannot be independently
recounted from this repository alone.

| Claim or summary | Needs |
|---|---|
| 177/224 and 313/342 test classifications; 29 FP, 47 FN | locked-test scores per knee |
| 43/459 (9.4%) and 33/107 (30.8%) branch misclassification | locked routing table |
| 38/224 positives deferred; 28/47 missed tears in the automated branch | locked routing table |
| Conformal single-label and set composition counts | per-knee conformal sets |
| Severity per-class counts | row-level severity table |
| Per-report audit rows and the split of workflow flags by routed action | paired report/audit table |
| MRNet overlap 92/13/15 and per-exam paired predictions | adaptation split and score tables |
| Grad-CAM case selection | selected-case table |

The code to generate these tables is included. The data can be supplied by the
corresponding author under the applicable data-use terms. No source images,
weights or row-level tables are redistributed here.
