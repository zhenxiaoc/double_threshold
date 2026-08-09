"""Parameter-free test of the cross-half off-diagonal order (App. B.2) using
only the numbers already reported in Table 1 of the draft.

Theory:  Var(SS quadratic correction)/Var(score) = c * K^{(d+2)/d} / n,
         c constant in n, so SD(SS)/SD(plug-in) = sqrt(1 + c K^{(d+2)/d}/n).
Design:  d = 2, K = 25 fixed at every n.
"""
K, d = 25, 2
NS = [1000, 2000, 4000, 8000]
# (SD plug-in, SD SS corner-aware, SE/SD plug-in, SE/SD SS)  from Table 1
DESIGNS = {
    "Affine (exact truth, assumptions hold)":
        ([0.0546, 0.0339, 0.0249, 0.0171], [0.0681, 0.0393, 0.0268, 0.0178],
         [0.97, 1.07, 1.00, 1.00], [0.78, 0.92, 0.93, 0.96]),
    "KRR (smooth)":
        ([0.0362, 0.0259, 0.0185, 0.0133], [0.0446, 0.0293, 0.0204, 0.0134],
         [1.11, 1.06, 1.00, 0.94], [0.90, 0.94, 0.91, 0.93]),
    "WGAN scalar (s~1; VIOLATES smoothness)":
        ([0.1008, 0.0884, 0.0618, 0.0453], [0.1370, 0.1213, 0.0780, 0.0535],
         [1.34, 1.15, 1.23, 1.13], [0.98, 0.84, 0.98, 0.96]),
    "WGAN degenerate (empty margins; VIOLATES)":
        ([0.0432, 0.0243, 0.0113, 0.0061], [0.0643, 0.0347, 0.0183, 0.0091],
         [1.60, 1.52, 1.73, 1.73], [1.07, 1.06, 1.06, 1.16]),
}
print("Implied c = n[(SD_SS/SD_plugin)^2 - 1] / K^{(d+2)/d};  theory: FLAT in n\n")
print(f"{'design':43s}" + "".join(f"{n:>9d}" for n in NS) + "   spread")
for lbl, (sdp, sds, sep, ses) in DESIGNS.items():
    cs = [n * ((b / a) ** 2 - 1) / K ** ((d + 2) / d)
          for n, a, b in zip(NS, sdp, sds)]
    print(f"{lbl:43s}" + "".join(f"{c:9.2f}" for c in cs)
          + f"   {max(cs)/min(cs):5.1f}x")

print("\nIdentity check: one SE is computed from the full-sample fit and used for")
print("both estimators, so SE/SD(SS) must equal (SE/SD plug-in)/(SD ratio).")
worst = 0.0
for lbl, (sdp, sds, sep, ses) in DESIGNS.items():
    impl = [e * a / b for e, a, b in zip(sep, sdp, sds)]
    err = max(abs(x - y) for x, y in zip(impl, ses))
    worst = max(worst, err)
    print(f"  {lbl:43s} max |implied - reported| = {err:.4f}")
print(f"  -> worst discrepancy over all designs and n: {worst:.4f}")
print("\nInflation factor K^{(d+2)/d}/n at the design's fixed K:",
      "  ".join(f"{K**((d+2)/d)/n:.2f}" for n in NS))
print("It vanishes iff a < d/(d+2), the upper edge of the SS window (4.10).")
