# =============================================================================
# ci_diagnostic.py
# Run this standalone to understand why q90=±118 cycles
# Takes ~10 seconds — loads val_calibration_data.json only
# =============================================================================
import numpy as np
import json
import os

RESULTS_DIR = 'results'

with open(os.path.join(RESULTS_DIR, 'val_calibration_data.json')) as f:
    val_cal = json.load(f)

residuals = np.array(val_cal['all_val_residuals'])
within_std = np.array(val_cal['all_val_within_std'])

print("=" * 60)
print("  CI DIAGNOSTIC REPORT")
print("=" * 60)

print(f"\n  Total val calibration cycles : {len(residuals)}")
print(f"  q90 stored                   : {val_cal['q90']:.4f} cycles")
print(f"  k90 stored                   : {val_cal['k90']:.4f}")

print(f"\n  ── Residual Distribution ──")
for p in [10, 25, 50, 75, 90, 95, 99]:
    print(f"  p{p:>2} : {np.percentile(residuals, p):.4f} cycles")

print(f"\n  Mean  : {residuals.mean():.4f}")
print(f"  Std   : {residuals.std():.4f}")
print(f"  Max   : {residuals.max():.4f}")
print(f"  Min   : {residuals.min():.4f}")

# Check for outliers driving up q90
extreme_mask = residuals > 50
print(f"\n  Residuals > 50 cycles : {extreme_mask.sum()} / {len(residuals)}")
print(f"  Residuals > 100 cycles: {(residuals > 100).sum()} / {len(residuals)}")
print(f"  Residuals > 200 cycles: {(residuals > 200).sum()} / {len(residuals)}")

if extreme_mask.sum() > 0:
    print(f"\n  Mean of extreme residuals (>50): {residuals[extreme_mask].mean():.2f}")
    print(f"  These are likely from early val cycles where DL models overfit badly")
    print(f"  → This inflates q90 far beyond the test engine prediction range")

# Show what q90 would be WITHOUT extreme residuals
for cutoff in [50, 100, 150]:
    clean = residuals[residuals <= cutoff]
    q90_clean = np.percentile(clean, 90)
    print(f"\n  q90 excluding residuals > {cutoff}: {q90_clean:.4f} cycles "
          f"({len(clean)}/{len(residuals)} cycles used)")

# Check if val residuals are dominated by specific models/phases
print(f"\n  ── Within-Config Std Distribution ──")
for p in [10, 25, 50, 75, 90]:
    print(f"  p{p:>2} : {np.percentile(within_std, p):.4f}")

print(f"\n  Valid std cycles (>0.1): {(within_std > 0.1).sum()} / {len(within_std)}")

# Root cause summary
print("\n" + "=" * 60)
print("  ROOT CAUSE ASSESSMENT")
print("=" * 60)
if val_cal['q90'] > 50:
    print("""
  q90 > 50 cycles indicates the calibration pool contains
  large-residual validation predictions that are not
  representative of test-engine behavior.

  Most likely cause:
  - Validation fleet contains engines with very different
    lifecycle lengths or degradation patterns from Engine 52
  - DL models (LSTM/CNN/Transformer) produce extreme predictions
    on early validation cycles before learning good representations
  - These extreme residuals are pooled across all 100 experiments
    and all 12 val engines, inflating the 90th percentile

  Recommended fix options:
  1. Cap residuals at max_expected_RUL before computing q90
     (e.g., clip at 2 * TOTAL_CYCLES = 426)
  2. Use per-experiment median of q90 instead of global pool
  3. Report conformal CI as supplementary — lean on variance CI
     as primary interval in the paper (it's better calibrated here)
  4. Add honest one-sentence limitation in Section 5
""")
else:
    print("  q90 is reasonable — no action needed.")