# MEC mean seed1 normalization comparison

Protocol:
- Scenario: `v6_hap_loadbearing`
- Policy: `mean`
- Seed: `1`
- Budget: `1.5M` env steps
- Higher `episode_reward` is better because rewards are negative costs.
- `FeatureNorm off` means the run passed `--use_feature_normalization`, because the upstream flag is `store_false`.

Runs:
- FeatureNorm on + no RMS: `postchange_resetperm1500_mean_seed1/run1`
- FeatureNorm on + flat RMS: `postchange_rms_resetperm1500_mean_seed1/run2`
- FeatureNorm on + component RMS: `postchange_component_rms_resetperm1500_mean_seed1/run1`
- FeatureNorm off + no RMS: `postchange_nofeat_resetperm1500_mean_seed1/run1`
- FeatureNorm off + flat RMS: `postchange_rms_nofeat_resetperm1500_mean_seed1/run1`
- FeatureNorm off + component RMS: `postchange_component_rms_nofeat_resetperm1500_mean_seed1/run1`

## Train episode_reward

| FeatureNorm | RMS | Last | Best | Last5 mean | Early500 mean | AUC mean |
|---|---|---:|---:|---:|---:|---:|
| on | none | -777.3 | -698.5 | -792.9 | -1305.1 | -987.3 |
| on | flat | -1046.6 | -804.5 | -949.6 | -1130.6 | -988.2 |
| on | component | -851.5 | -777.1 | -857.9 | -1252.2 | -1024.2 |
| off | none | -986.5 | -806.0 | -1006.9 | -1569.4 | -1265.4 |
| off | flat | -868.2 | -776.6 | -845.5 | -1214.2 | -1000.7 |
| off | component | -846.0 | -790.6 | -870.4 | -1355.8 | -1088.1 |

## Eval episode_reward

| FeatureNorm | RMS | Last | Best | Last5 mean | Early500 mean | AUC mean |
|---|---|---:|---:|---:|---:|---:|
| on | none | -730.0 | -696.8 | -715.3 | -1247.7 | -879.1 |
| on | flat | -936.1 | -735.8 | -897.7 | -1067.2 | -910.9 |
| on | component | -936.7 | -877.6 | -936.5 | -1167.3 | -1007.3 |
| off | none | -874.6 | -866.9 | -1022.8 | -1533.0 | -1222.1 |
| off | flat | -815.9 | -799.5 | -872.3 | -1086.6 | -907.7 |
| off | component | -829.7 | -822.5 | -845.5 | -1265.7 | -978.0 |

## Discussion

FeatureNorm on:
- No RMS is the strongest setting on final eval: last `-730.0`, best `-696.8`, last5 `-715.3`, AUC `-879.1`.
- Flat RMS improves early training/eval relative to no RMS in the first part of training, but it finishes worse.
- Component RMS does not fix this interaction. It is worse than no RMS on every eval summary metric, and worse than flat RMS on best/AUC.
- Conclusion: do not use RMS together with FeatureNorm on for this current mean baseline.

FeatureNorm off:
- Both RMS variants clearly improve over no RMS.
- Flat RMS has the best eval best and AUC: best `-799.5`, AUC `-907.7`.
- Component RMS has the best eval last5 mean: `-845.5` versus flat RMS `-872.3`, but its best and AUC are weaker than flat RMS.
- Conclusion: component RMS is competitive in late training when FeatureNorm is off, but this single seed does not show a clean win over flat RMS.

Overall:
- The new component RMS is not a general improvement.
- The strongest final eval setting remains `FeatureNorm on + no RMS`.
- If the research question is whether RMS helps training, the answer is conditional:
  - With FeatureNorm on: RMS hurts final eval.
  - With FeatureNorm off: RMS helps substantially.
- If the research question is whether component RMS is better than flat RMS, current evidence says no for AUC/best, but component RMS may be worth one confirmatory seed only for `FeatureNorm off` because its late eval last5 is better.

Recommended next step:
- Keep default as no RMS with FeatureNorm on unless there is a stronger reason to remove FeatureNorm.
- For RMS ablation, report two panels by FeatureNorm state rather than one six-line figure.
- If continuing this branch, run only `FeatureNorm off + flat RMS` vs `FeatureNorm off + component RMS` with seeds `2,3`, and compare held-out eval, AUC, and last5. Do not expand the full 2x3 matrix yet.
