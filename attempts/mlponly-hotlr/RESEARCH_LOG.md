# MLP-only crystallization at a warmer learning rate — spending retention slack

The score is test-edge accuracy x min(1, decisiveness_FT / decisiveness_base). The
fleet best #140 (MLP-only, freeze attention + embeddings, L2-SP 1e-2, lr 1e-4, 1000
steps) got test 0.6776 x retention 0.9388 = 0.6362. Retention 0.94 means the min()
cap is not saturated, but there is slack between it and 1.0. Test accuracy is the
binding term.

The cool-LR siblings push retention up, which cannot help once the ratio is already
below the cap unless test also holds — and above the cap (ratio >= 1) extra
retention is wasted. This attempt takes the opposite bet: warm the learning rate to
1.2e-4 so the concept installs more fully in 1000 steps, raising test accuracy, and
accept that retention falls somewhat. As long as the retention ratio stays high, the
min() term barely moves and a test gain flows straight to the score.

This is deliberately distinct from #152, which also used a hotter LR but stacked it
with a strong L2-SP anchor (2e-2) that prior attempts (#133, #151) showed
over-anchors MLP-only training; here the anchor stays at the value that worked in
#140 (1e-2), so the LR is the only change.

If retention collapses (the hotter LR cooks the MLP memories faster than expected),
that bounds the safe MLP-only LR just above 1e-4. If test rises with retention
holding, the optimum is hotter than #140 and worth pushing toward 1.4e-4.

Shipped config-only near the pod's wall-clock deadline; held-out CI trains and
scores from scratch, so the PR comment score is authoritative.
