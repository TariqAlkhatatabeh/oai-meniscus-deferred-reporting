# Final frozen primary-only reporting evaluation protocol

This protocol supersedes `FROZEN_PROTOCOL_20260910.md`. Before formal report
generation, a four-report engineering smoke test showed that the initial
auditor did not recognize the valid phrase "presence of a tear" and did not
match two explicit unsupported descriptors ("full-thickness" and
"intra-articular"). Those lexical patterns were corrected, unit tests and the
smoke audit were rerun, and the auditor was then frozen. No formal report was
generated or inspected before this final freeze.

## Locked design

- Locked OAI test cohort: all 566 knees from 450 participants.
- Paired arms: one constrained and one unconstrained report per knee.
- Structured facts: only the primary 16-slice ConvNeXt binary probability,
  validation-fitted temperature-scaled probability, confidence,
  validation-locked workflow action, and the same model's auxiliary
  severity-head output.
- Excluded: all historical 9-slice-model outputs, including compartment and
  medial/lateral support variables.
- Text model: local Qwen3.5-9B in 4-bit NF4; deterministic decoding
  (`do_sample=False`, thinking disabled).
- Primary endpoint: any violation across five frozen rule domains, aggregated
  to participant level (any included knee), compared between paired arms with
  an exact two-sided McNemar test.
- Descriptive uncertainty: report-level rates with 95% participant-clustered
  bootstrap confidence intervals (5,000 resamples).
- Format/length compliance: recorded separately; excluded from the primary
  rule-violation endpoint.
- Controlled challenge: 50 exact-pattern injections per rule domain (250
  total), performed after natural-report generation without changing the
  auditor. This evaluates the implemented patterns, not unrestricted clinical
  paraphrase.

## Frozen rule domains

1. Numeric mismatch.
2. Unsupported anatomy or compartment localization.
3. Treatment, referral, or follow-up advice.
4. Contradiction of the supplied primary-model severity class.
5. Inconsistency with the validation-locked AUTO_READ/DEFER action.

## Final frozen script hashes (SHA-256)

- `audit_reports.py`: `4135D5D7A7A9910557EBFB4087E8CFC48776057D42329AE3CB6E2FD61C112310`
- `generate_reports.py`: `FD849C10379A455B35E6D1FE084B125C820DFB22A36D43F222213249C43B4CA9`

## Locked artifact hashes (SHA-256)

- Primary checkpoint: `3720B157520C72A168DA8E489AE6EACC6225E2E4627A2C3A41CA3891B55C6965`
- Raw TTA/coverage test predictions: `3328C9960001D72D0A5D63FBA29529D71772821D66ABC75463002A0384E11396`
- Temperature-scaled test predictions: `2EC255D579C60C9EB2AA837DBE9ADEC06DEE47139134305BA6DAD3C40CAF6FE2`
- Validation-locked coverage table: `D5B65A68494E0E7C48F9787D9F8E8970852A5C6A6E7F4E697A7EB887F629022F`
- Primary severity-output CSV: `ACDA91F355811EEA96C7AF1EB23C50215CEF9503B1C351DB3B9E03AAD099011A`
