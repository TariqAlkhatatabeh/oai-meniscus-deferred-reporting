"""Reproduce the deployment-workload figures quoted in the npjDM Discussion.

Emitted numbers: 459/107 branch split, 9.4% (43/459) misclassification inside the automated branch,
13.4% (76/566) across the full cohort, 30.8% (33/107) among deferred knees, 38/224 (17.0%) tear-positive
knees deferred, and 28 of the 47 full-coverage missed tears falling inside the automated branch.

Classification uses the unrounded prob_raw; the facts packet carries 3-decimal values: the frozen per-knee actions and structured facts from the
paired report generation run, cross-checked against the calibration test predictions. Re-running this
must reproduce every figure exactly; if it does not, the manuscript text is wrong.
"""
import sys, csv, json
sys.stdout.reconfigure(encoding="utf8", errors="replace")

RUNS = "results/reporting/qwen35_9b_primary16s_paired_reports.csv"
PRED = "results/convnext_safe_strong_16s_tta_20260629_calibration_test_predictions.csv"

knee = {}
for r in csv.DictReader(open(RUNS, encoding="utf8", errors="replace")):
    if r["arm"] != "constrained":
        continue
    f = json.loads(r["facts_json"])
    knee[(r["ID"], r["SIDE"])] = {"label": int(r["label"]), "action": r["action"], "p": float(f["tta_tear_probability"])}

pred = {}
for r in csv.DictReader(open(PRED, encoding="utf8", errors="replace")):
    pred[(r["ID"], r["SIDE"])] = (float(r["prob_raw"]), int(r["label"]))

assert set(knee) == set(pred), "knee sets differ between the routing run and the calibration predictions"
for k in knee:
    # the facts packet carries three-decimal probabilities, as documented in the frozen protocol
    assert abs(knee[k]["p"] - round(pred[k][0], 3)) < 1e-9, f"TTA probability differs for {k}"
    assert knee[k]["label"] == pred[k][1], f"label differs for {k}"
print(f"joined {len(knee)} locked test knees; facts probabilities equal round(raw,3) and labels agree")

def branch(sel):
    n = len(sel)
    err = sum(1 for k in sel if (1 if pred[k][0] >= 0.5 else 0) != knee[k]["label"])
    fn = sum(1 for k in sel if pred[k][0] < 0.5 and knee[k]["label"] == 1)
    fp = sum(1 for k in sel if pred[k][0] >= 0.5 and knee[k]["label"] == 0)
    pos = sum(1 for k in sel if knee[k]["label"] == 1)
    return n, err, fn, fp, pos

allk = list(knee)
auto = [k for k in allk if knee[k]["action"] == "AUTO_READ"]
dfr = [k for k in allk if knee[k]["action"] == "DEFER"]

for name, sel in (("FULL", allk), ("AUTO_READ", auto), ("DEFER", dfr)):
    n, err, fn, fp, pos = branch(sel)
    print(f"  {name:9s} n={n:3d}  errors={err:3d} ({err/n*100:4.1f}%)  FP={fp:3d}  FN={fn:3d}  tear-positive={pos:3d}")

_, _, fn_all, _, pos_all = branch(allk)
_, _, fn_auto, _, _ = branch(auto)
_, _, _, _, pos_def = branch(dfr)

print("\n=== figures as they appear in the Discussion ===")
print(f"  459 entered automatic drafting and 107 were read by a radiologist first -> {len(auto)}/{len(dfr)}")
print(f"  per 1,000 MRIs: {len(auto)/len(allk)*1000:.0f} drafted, {len(dfr)/len(allk)*1000:.0f} read first")
print(f"  9.4% (43 of 459) inside the automated branch -> {branch(auto)[1]}/{len(auto)} = {branch(auto)[1]/len(auto)*100:.1f}%")
print(f"  13.4% across the full cohort -> {branch(allk)[1]}/{len(allk)} = {branch(allk)[1]/len(allk)*100:.1f}%")
print(f"  30.8% (33 of 107) among deferred -> {branch(dfr)[1]}/{len(dfr)} = {branch(dfr)[1]/len(dfr)*100:.1f}%")
print(f"  38 of the 224 tear-positive knees (17.0%) deferred -> {pos_def}/{pos_all} = {pos_def/pos_all*100:.1f}%")
print(f"  28 of the 47 missed tears still inside the automated branch -> {fn_auto}/{fn_all}")
