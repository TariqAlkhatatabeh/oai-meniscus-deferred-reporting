# Locked-test ConvNeXt-VLM fusion evaluation

- Frozen candidate: `weighted_platt_blend_conv70`
- Knees/participants: 566/450
- Participant-clustered bootstrap replicates: 5000

| Model | AUC (95% CI) | PR-AUC (95% CI) | F1 | Brier |
|---|---:|---:|---:|---:|
| convnext | 0.9090 (0.8808-0.9349) | 0.8920 (0.8546-0.9250) | 0.8188 | 0.1156 |
| vlm | 0.8756 (0.8429-0.9054) | 0.8552 (0.8118-0.8920) | 0.7830 | 0.1499 |
| fusion | 0.9095 (0.8813-0.9355) | 0.9000 (0.8658-0.9297) | 0.8113 | 0.1054 |

## Paired fusion gains

| Comparison | AUC difference (95% CI) | PR-AUC difference (95% CI) |
|---|---:|---:|
| fusion_minus_convnext | +0.0006 (-0.0054-+0.0069) | +0.0081 (-0.0032-+0.0215) |
| fusion_minus_vlm | +0.0340 (+0.0172-+0.0522) | +0.0449 (+0.0260-+0.0661) |
