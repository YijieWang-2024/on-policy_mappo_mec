# Single-normalization comparison

Interpretation of the three curves:
- FeatureNorm only = FeatureNorm on + no RMS.
- Flat RMS only = FeatureNorm off + flat RMS.
- Component RMS only = FeatureNorm off + component RMS.

## Summary

| Metric | Method | Last | Best | Last5 mean | Early500 mean | AUC mean |
|---|---|---:|---:|---:|---:|---:|
| train | FeatureNorm only | -777.3 | -698.5 | -792.9 | -1305.1 | -987.3 |
| train | Flat RMS only | -868.2 | -776.6 | -845.5 | -1214.2 | -1000.7 |
| train | Component RMS only | -846.0 | -790.6 | -870.4 | -1355.8 | -1088.1 |
| eval | FeatureNorm only | -730.0 | -696.8 | -715.3 | -1247.7 | -879.1 |
| eval | Flat RMS only | -815.9 | -799.5 | -872.3 | -1086.6 | -907.7 |
| eval | Component RMS only | -829.7 | -822.5 | -845.5 | -1265.7 | -978.0 |

## Takeaway

For final eval, FeatureNorm only has the best last and best reward. Flat RMS only has the best eval AUC. Component RMS only has the best eval last5 mean among the two RMS-only settings, but it does not beat FeatureNorm only on final eval quality.