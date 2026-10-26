# Arm C: Qwen2.5-VL-7B zero-shot meniscus tear detection

- Locked OAI test set: 566 knees; threshold 0.5
- Bootstrap: participant-clustered, 5000 replicates
- Reference: supervised ConvNeXt test AUC 0.9090 [0.8808-0.9349]

| Input variant | AUC | 95% CI | Bal. acc @0.5 | Mean p(tear|label=1) | Mean p(tear|label=0) | Parse fails |
|---|---:|---:|---:|---:|---:|---:|
| Single mid-sagittal DESS slice | 0.5103 | 0.4739-0.5472 | 0.500 | 0.875 | 0.873 | 0 |
| 3x3 tri-plane grid | 0.5078 | 0.4665-0.5497 | 0.500 | 0.838 | 0.837 | 0 |
| 4x4 sagittal montage (16 slices) | 0.4769 | 0.4473-0.5067 | 0.500 | 0.812 | 0.817 | 0 |
| Three plane montages (all 48 slices) | 0.4893 | 0.4497-0.5280 | 0.500 | 0.869 | 0.871 | 0 |
