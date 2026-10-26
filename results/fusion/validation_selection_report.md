# ConvNeXt-VLM fusion validation selection

This artifact was produced without a test-set input.

- Selected candidate: `weighted_platt_blend_conv70`
- Validation knees/participants: 540/449
- Selected OOF AUC: 0.9229
- ConvNeXt OOF AUC: 0.9179
- VLM OOF AUC: 0.8920

| Candidate | Eligible | AUC | PR-AUC | F1 | Brier |
|---|---:|---:|---:|---:|---:|
| equal_platt_blend | True | 0.9203 | 0.9069 | 0.8264 | 0.1043 |
| logistic_stack | True | 0.9216 | 0.9108 | 0.8284 | 0.0995 |
| weighted_platt_blend_conv60 | True | 0.9219 | 0.9104 | 0.8273 | 0.1019 |
| weighted_platt_blend_conv70 | True | 0.9229 | 0.9120 | 0.8263 | 0.1005 |
| weighted_platt_blend_conv80 | True | 0.9219 | 0.9116 | 0.8249 | 0.1000 |
| weighted_platt_blend_conv90 | True | 0.9205 | 0.9104 | 0.8313 | 0.1004 |
