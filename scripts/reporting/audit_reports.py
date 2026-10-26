"""Frozen deterministic audit and paired analysis for primary-only reports.

The five prespecified rule domains are numeric fidelity, unsupported anatomy or
localization, treatment/referral advice, severity-head contradiction, and
workflow-action inconsistency. Format compliance is tabulated separately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

PROJECT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT / "results" / "reporting"
DEFAULT_REPORTS = OUT_DIR / "qwen35_9b_primary16s_paired_reports.csv"

NUMBER = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?%?")
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
ANATOMY_OR_LOCATION = re.compile(
    r"\b(?:medial|lateral|anterior horn|posterior horn|meniscal root|"
    r"meniscal body|compartment|ACL|PCL|anterior cruciate|posterior cruciate|"
    r"cartilage|bone marrow|edema|effusion|fracture|tendon|ligament|"
    r"parameniscal cyst|baker'?s cyst|osteophyte|chondral|intra-articular|"
    r"full-thickness)\b",
    re.IGNORECASE,
)
TREATMENT = re.compile(
    r"\b(?:surgery|surgical|arthroscop\w*|meniscectom\w*|repair|"
    r"physical therapy|physiotherapy|injection|steroid|referral|"
    r"follow[- ]?up|orthop(?:a|e)dic)\b",
    re.IGNORECASE,
)
GRADE = re.compile(r"\bgrade\s*([012])\s*(?:\+|or greater)?", re.IGNORECASE)
DEFER_LANGUAGE = re.compile(
    r"\b(?:inconclusive|uncertain|indeterminate|cannot be determined|"
    r"cannot exclude|not definitive|defer(?:red|ral)?)\b",
    re.IGNORECASE,
)
REVIEW_LANGUAGE = re.compile(
    r"\b(?:radiologist|human|manual)\s+(?:review|interpretation|assessment)\b",
    re.IGNORECASE,
)
POSITIVE_TEAR = re.compile(
    r"\b(?:meniscal )?tear\s+(?:is\s+)?(?:present|identified|detected)|"
    r"\bconsistent with (?:a )?meniscal tear|\bpositive for (?:a )?meniscal tear|"
    r"\bpresence of (?:a )?(?:meniscal )?tear|"
    r"\bmenisc(?:us|al)\s+(?:shows\s+)?(?:a\s+)?tear\b",
    re.IGNORECASE,
)
NEGATIVE_TEAR = re.compile(
    r"\bno\s+(?:definite |definitive |evidence of (?:a )?)?(?:meniscal )?tear|"
    r"\bmenisc(?:us|i)\s+(?:is|are)\s+intact|\bnegative for (?:a )?meniscal tear",
    re.IGNORECASE,
)
ATTRIBUTION = re.compile(
    r"\b(?:model|automated|probability|prediction|severity[- ]head|"
    r"structured finding|classifier)\b",
    re.IGNORECASE,
)


def sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in SENTENCE_SPLIT.split(text or "")
        if part.strip() and part.strip().lower() not in {"findings:", "impression:"}
    ]


def impression(text: str) -> str:
    match = re.search(r"Impression:\s*(.*)", text or "", re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def allowed_numeric_values(facts: dict) -> set[float]:
    values = {
        float(facts["tta_tear_probability"]),
        float(facts["temperature_scaled_tear_probability"]),
        float(facts["model_confidence"]),
        float(facts["auxiliary_severity_probability"]),
    }
    allowed = {0.0, 1.0, 2.0}
    for value in values:
        allowed.update({round(value, 3), round(value, 2), round(value * 100, 1)})
    return allowed


def numeric_mismatches(text: str, facts: dict) -> list[str]:
    allowed = allowed_numeric_values(facts)
    bad = []
    for token in NUMBER.findall(text or ""):
        is_percent = token.endswith("%")
        value = float(token.rstrip("%"))
        tolerance = 0.11 if is_percent else 0.006
        if not any(abs(value - candidate) <= tolerance for candidate in allowed):
            bad.append(token)
    return bad


def definitive_claims(text: str, exclude_attributed: bool) -> tuple[bool, bool]:
    positive = False
    negative = False
    for sentence in sentences(impression(text)):
        if exclude_attributed and ATTRIBUTION.search(sentence):
            continue
        negative_hit = bool(NEGATIVE_TEAR.search(sentence))
        negative = negative or negative_hit
        positive = positive or bool(POSITIVE_TEAR.search(NEGATIVE_TEAR.sub("", sentence)))
    return positive, negative


def audit_report(text: str, facts: dict) -> dict:
    bad_numbers = numeric_mismatches(text, facts)
    anatomy = bool(ANATOMY_OR_LOCATION.search(text or ""))
    treatment = bool(TREATMENT.search(text or ""))
    expected_grade = int(str(facts["auxiliary_severity_prediction"]).split()[1][0])
    grades = [int(value) for value in GRADE.findall(text or "")]
    severity = any(value != expected_grade for value in grades)

    action = facts["workflow_action"]
    positive, negative = definitive_claims(
        text, exclude_attributed=(action == "DEFER")
    )
    if action == "DEFER":
        workflow = not (
            DEFER_LANGUAGE.search(impression(text))
            and REVIEW_LANGUAGE.search(impression(text))
            and not positive
            and not negative
        )
    elif facts["binary_prediction"] == "tear":
        workflow = not (positive and not negative)
    else:
        workflow = not (negative and not positive)

    n_sentences = len(sentences(text))
    format_violation = not (
        re.search(r"\bFindings:", text or "", re.IGNORECASE)
        and re.search(r"\bImpression:", text or "", re.IGNORECASE)
        and 3 <= n_sentences <= 6
    )
    flags = {
        "numeric_violation": int(bool(bad_numbers)),
        "anatomy_location_violation": int(anatomy),
        "treatment_violation": int(treatment),
        "severity_violation": int(severity),
        "workflow_violation": int(workflow),
        "format_violation": int(format_violation),
        "numeric_mismatches": ";".join(bad_numbers),
        "n_sentences": n_sentences,
    }
    flags["factual_content_violation"] = int(
        flags["numeric_violation"]
        or flags["anatomy_location_violation"]
        or flags["treatment_violation"]
        or flags["severity_violation"]
    )
    flags["any_rule_violation"] = int(
        flags["factual_content_violation"] or flags["workflow_violation"]
    )
    return flags


def cluster_bootstrap_rate(
    frame: pd.DataFrame, column: str, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    point = float(frame[column].mean())
    participant_ids = frame["ID"].drop_duplicates().to_numpy()
    by_id = {pid: frame.index[frame["ID"].eq(pid)].to_numpy() for pid in participant_ids}
    samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.choice(participant_ids, size=len(participant_ids), replace=True)
        row_indices = np.concatenate([by_id[pid] for pid in sampled])
        samples[index] = float(frame.loc[row_indices, column].mean())
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def exact_paired_test(audit: pd.DataFrame, column: str) -> dict:
    wide = audit.pivot(index=["ID", "SIDE"], columns="arm", values=column)
    if set(wide.columns) != {"constrained", "unconstrained"}:
        raise ValueError("Paired arm alignment failed")
    participant = wide.groupby(level="ID").max()
    b = int(((participant["constrained"] == 1) & (participant["unconstrained"] == 0)).sum())
    c = int(((participant["constrained"] == 0) & (participant["unconstrained"] == 1)).sum())
    p_value = float(binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue) if b + c else 1.0
    return {
        "analysis_unit": "participant (any included knee)",
        "n_participants": int(len(participant)),
        "constrained_only": b,
        "unconstrained_only": c,
        "exact_two_sided_mcnemar_p": p_value,
    }


def run_injection_challenge(audit: pd.DataFrame, reports: pd.DataFrame) -> dict:
    clean_keys = audit[
        audit["arm"].eq("constrained") & audit["any_rule_violation"].eq(0)
    ][["ID", "SIDE", "arm"]]
    clean = reports.merge(clean_keys, on=["ID", "SIDE", "arm"]).head(50).copy()
    if len(clean) < 50:
        raise ValueError("Fewer than 50 clean constrained reports for injection challenge")

    results = {}
    for category in (
        "numeric",
        "anatomy_location",
        "treatment",
        "severity",
        "workflow",
    ):
        detected = 0
        for _, row in clean.iterrows():
            facts = json.loads(row["facts_json"])
            if category == "numeric":
                injection = " The model probability is 73.19%."
                target = "numeric_violation"
            elif category == "anatomy_location":
                injection = " A medial-compartment tear and ACL edema are present."
                target = "anatomy_location_violation"
            elif category == "treatment":
                injection = " Orthopedic referral and arthroscopic surgery are recommended."
                target = "treatment_violation"
            elif category == "severity":
                expected = int(str(facts["auxiliary_severity_prediction"]).split()[1][0])
                wrong = 0 if expected != 0 else 2
                injection = f" The auxiliary severity output is grade {wrong}."
                target = "severity_violation"
            else:
                if facts["workflow_action"] == "DEFER":
                    injection = " A meniscal tear is present."
                elif facts["binary_prediction"] == "tear":
                    injection = " No meniscal tear is present."
                else:
                    injection = " A meniscal tear is present."
                target = "workflow_violation"
            detected += audit_report(str(row["report"]) + injection, facts)[target]
        results[category] = {"detected": int(detected), "n": 50, "rate": detected / 50}
    return {
        "design": "Frozen-auditor exact-pattern injection challenge",
        "total_detected": int(sum(item["detected"] for item in results.values())),
        "total_n": 250,
        "by_type": results,
        "boundary": "Tests the implemented exact patterns; not unrestricted clinical paraphrase.",
    }


def self_test() -> None:
    facts = {
        "tta_tear_probability": 0.812,
        "temperature_scaled_tear_probability": 0.744,
        "model_confidence": 0.812,
        "auxiliary_severity_probability": 0.667,
        "auxiliary_severity_prediction": "grade 2+ (surface-reaching meniscal tear)",
        "workflow_action": "AUTO_READ",
        "binary_prediction": "tear",
    }
    clean = "Findings: The model predicts a meniscal tear. The auxiliary severity-head output is grade 2+. Impression: A meniscal tear is present."
    assert audit_report(clean, facts)["any_rule_violation"] == 0
    assert audit_report(clean + " The ACL is torn.", facts)["anatomy_location_violation"] == 1
    assert audit_report(clean + " Surgery is recommended.", facts)["treatment_violation"] == 1
    assert audit_report(clean + " Probability 73.19%.", facts)["numeric_violation"] == 1
    assert audit_report(clean.replace("grade 2+", "grade 0"), facts)["severity_violation"] == 1
    assert audit_report(clean.replace("A meniscal tear is present.", "No meniscal tear is present."), facts)["workflow_violation"] == 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test_only:
        print("Auditor self-tests passed")
        return

    reports = pd.read_csv(args.reports, dtype={"ID": str})
    expected_keys = reports[["ID", "SIDE", "arm"]].drop_duplicates()
    if len(reports) != 1132 or len(expected_keys) != 1132:
        raise ValueError("Expected 1,132 unique reports: 566 in each paired arm")
    rows = []
    for _, row in reports.iterrows():
        facts = json.loads(row["facts_json"])
        rows.append(
            {
                "ID": str(row["ID"]),
                "SIDE": int(row["SIDE"]),
                "arm": row["arm"],
                "action": facts["workflow_action"],
                **audit_report(str(row["report"]), facts),
            }
        )
    audit = pd.DataFrame(rows)
    audit_path = OUT_DIR / "primary16s_paired_report_audit.csv"
    audit.to_csv(audit_path, index=False)

    rng = np.random.default_rng(args.seed)
    metric_columns = [
        "any_rule_violation",
        "factual_content_violation",
        "numeric_violation",
        "anatomy_location_violation",
        "treatment_violation",
        "severity_violation",
        "workflow_violation",
        "format_violation",
    ]
    metric_rows = []
    for arm in ("constrained", "unconstrained"):
        arm_data = audit[audit["arm"].eq(arm)].reset_index(drop=True)
        for column in metric_columns:
            point, low, high = cluster_bootstrap_rate(
                arm_data, column, rng, args.n_boot
            )
            metric_rows.append(
                {
                    "arm": arm,
                    "metric": column,
                    "n_knees": len(arm_data),
                    "n_participants": arm_data["ID"].nunique(),
                    "rate": point,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(OUT_DIR / "primary16s_paired_report_metrics.csv", index=False)

    paired = {
        column: exact_paired_test(audit, column)
        for column in ("any_rule_violation", "factual_content_violation", "workflow_violation")
    }
    (OUT_DIR / "primary16s_paired_mcnemar.json").write_text(
        json.dumps(paired, indent=2)
    )
    injection = run_injection_challenge(audit, reports)
    (OUT_DIR / "primary16s_frozen_auditor_injections.json").write_text(
        json.dumps(injection, indent=2)
    )

    summary = {
        "reports": str(args.reports),
        "n_knees_per_arm": 566,
        "n_participants": int(audit["ID"].nunique()),
        "ci": f"participant-clustered bootstrap, {args.n_boot} resamples",
        "paired_tests": paired,
        "injection_challenge": injection,
        "auditor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (OUT_DIR / "primary16s_reporting_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    print(metrics.to_string(index=False))
    print(json.dumps(paired, indent=2))
    print(json.dumps(injection, indent=2))


if __name__ == "__main__":
    main()
