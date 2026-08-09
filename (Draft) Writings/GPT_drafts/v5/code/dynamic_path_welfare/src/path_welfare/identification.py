"""Run and serialize the identification audit (JSON + Markdown)."""

from __future__ import annotations

import json
from pathlib import Path

from .diagnostics import full_identification_audit


def run_and_write(df, cfg, outdir: str | Path, *, e1=None, e2=None, availability_col=None) -> dict:
    audit = full_identification_audit(df, cfg, e1=e1, e2=e2, availability_col=availability_col)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "identification_audit.json").write_text(
        json.dumps(audit, indent=2, default=float), encoding="utf-8"
    )
    (outdir / "identification_audit.md").write_text(_to_markdown(audit, cfg), encoding="utf-8")
    return audit


def _to_markdown(audit: dict, cfg) -> str:
    L = ["# Identification Audit", ""]
    L.append(f"Dataset: **{getattr(cfg, 'dataset', 'n/a')}**  (config: {getattr(cfg,'name','')})")
    L.append("")
    pos = audit["positivity"]
    L.append("## Positivity (design-based)")
    L.append(f"- Path counts: {pos['path_counts']}")
    L.append(f"- P(T1=1) empirical={pos['P(T1=1)_empirical']:.3f} design={pos['P(T1=1)_design']}")
    L.append(f"- P(T2=1) empirical={pos['P(T2=1)_empirical']:.3f} design={pos['P(T2=1)_design']}")
    L.append(f"- Smallest assignment prob: {pos['smallest_assignment_prob']:.3f} "
             f"(>=0.10: {pos['positivity_holds_at_0.10']})")
    L.append("")
    ms = audit["markov_sufficiency"]
    L.append("## Markov sufficiency (testable modelling restriction)")
    L.append(f"- Held-out MSE restricted Y~f(X,T2): {ms['mse_restricted_YXT2']:.4f}")
    L.append(f"- Held-out MSE rich Y~f(S,T1,X,T2): {ms['mse_rich_YSXT1T2']:.4f}")
    L.append(f"- Incremental MSE reduction: {ms['incremental_mse_reduction']*100:.1f}%")
    L.append(f"- **Markov questionable: {ms['markov_questionable']}** -- {ms['interpretation']}")
    L.append("")
    bal = audit["balance"]
    L.append("## Randomization / balance diagnostics")
    L.append(f"- AUC(T1 | S) = {bal['T1_from_S_auc']:.3f}")
    L.append(f"- AUC(T2 | S,T1,X) = {bal['T2_from_history_auc']:.3f}")
    L.append(f"- {bal['note']}")
    L.append("")
    att = audit["attrition"]
    L.append("## Attrition / missingness")
    L.append(f"- Missing S/X/Y: {att['missing_S']:.3f} / {att['missing_X']:.3f} / {att['missing_Y']:.3f}")
    L.append("- Missing Y by path: " +
             ", ".join(f"{k}:{v['missing_Y']:.3f}" for k, v in att["missing_Y_by_path"].items()))
    L.append("")
    cs = audit["continuous_state"]
    L.append("## Continuous-state diagnostics")
    for name in ("S", "X"):
        c = cs[name]
        L.append(f"- {name}: unique={c['n_unique']}, max point mass={c['max_point_mass']:.3f}, "
                 f"effectively_continuous={c['effectively_continuous']}, "
                 f"genuinely_continuous={c['genuinely_continuous']}")
    L.append("")
    L.append("_Language key: design-based (positivity, sequential ignorability from "
             "randomization) vs. testable modelling restriction (Markov) vs. diagnostic "
             "(balance, continuity)._")
    return "\n".join(L)
