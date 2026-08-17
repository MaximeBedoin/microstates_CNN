#!/usr/bin/env python
"""Synthese lisible d'un results.json (markdown)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def fmt(x, n=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{n}f}"


def table(rows, headers):
    widths = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    out = [line, sep]
    for r in rows:
        out.append("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("results", help="chemin vers results.json")
    p.add_argument("--out", default=None, help="ecrit le markdown dans ce fichier")
    a = p.parse_args()

    r = json.loads(Path(a.results).read_text())
    cfg = r["config"]
    lines = []
    add = lines.append

    add(f"# Resultats — {cfg['name']} ({cfg['dataset']})\n")
    add(f"K={cfg['k']}, latent={cfg['latent_dim']}, image={cfg['image_size']}x"
        f"{cfg['image_size']}, {r['peaks']['n_subjects']} sujets, "
        f"{r['peaks']['total']} pics "
        f"(mediane {r['peaks']['per_subject_median']:.0f}/sujet)")
    add(f"Fidelite topo->image->topo : r = {fmt(r['image_roundtrip_r'], 4)}")
    add(f"Duree totale : {r['seconds_total'] / 60:.0f} min\n")

    add("## Modeles\n")
    rows = []
    for k, m in r["models"].items():
        f = m["final"]
        rows.append([k, m["n_params"], fmt(m["best_val"], 4),
                     fmt(f.get("val_recon"), 4), fmt(f.get("train_pol_rel"), 6),
                     f"{m['seconds']:.0f}s"])
    add(table(rows, ["modele", "params", "val_loss", "val_recon",
                     "invariance res.", "temps"]))
    add("")

    if "arch_search" in r:
        add("## Recherche d'architecture (top 5)\n")
        rows = [[s["cfg"]["latent_dim"], s["cfg"]["beta"], s["cfg"]["base_width"],
                 fmt(s["val_loss"], 4), fmt(s["val_recon"], 4)]
                for s in r["arch_search"][:5]]
        add(table(rows, ["latent", "beta", "width", "val_loss", "val_recon"]))
        add("")

    add("## Accord entre methodes (correlation absolue moyenne, appariement hongrois)\n")
    keys = sorted(r["cross_method_corr"])
    refs = [k for k in keys if k.startswith(("ground_truth", "pycrostates"))]
    rows = [[k, fmt(np.mean(r["cross_method_corr"][k])),
             " ".join(fmt(v, 2) for v in r["cross_method_corr"][k])]
            for k in (refs or keys)]
    add(table(rows, ["paire", "corr abs moy", "par classe"]))
    add("")

    add("## Sanity check canonique (proxys geometriques A/B/C/D)\n")
    rows = [[k, fmt(v["mean_corr"]), " ".join(fmt(x, 2) for x in v["corrs"])]
            for k, v in r["sanity_canonical"].items()]
    add(table(rows, ["methode", "corr abs moy", "A B C D"]))
    add("")

    add("## Back-fitting : parametres microstates\n")
    rows = []
    for k, b in r["backfit"].items():
        rows.append([k, fmt(b["gev_total_mean"]),
                     fmt(np.nanmean(b["duration_ms"]), 1),
                     fmt(np.mean(b["occurrence"]), 2),
                     fmt(b["entropy_bits"], 2), fmt(b["lzc"], 3)])
    add(table(rows, ["methode", "GEV", "duree (ms)", "occ/s", "H (bits)", "LZC"]))
    add("")

    add("## Stabilite split-half (correlation absolue entre deux moities)\n")
    rows = []
    for k, v in r["stability"].items():
        if isinstance(v, dict):
            rows.append([k, fmt(v["mean"]), fmt(v["std"]),
                         "oui" if v.get("refit") else "non"])
    add(table(rows, ["methode", "corr abs moy", "ecart-type", "VAE re-entraine"]))
    add("")

    if r.get("group_comparison"):
        add("## Comparaison de groupes\n")
        rows = []
        for meth, res in r["group_comparison"].items():
            if not isinstance(res, dict):
                continue
            for key, v in res.items():
                stats = np.array(v["stats"], dtype=float)
                if stats.size == 0 or not np.isfinite(stats).any():
                    continue
                best = int(np.nanargmin(stats[:, 1]))
                rows.append([meth, key, f"K{best + 1}", fmt(stats[best, 1], 4),
                             fmt(stats[best, 2], 2),
                             "apparie" if v.get("paired") else "independant"])
        add(table(rows, ["methode", "parametre", "classe la + discriminante",
                         "p", "effet", "test"]))
        add("")

    md = "\n".join(lines)
    print(md)
    if a.out:
        Path(a.out).write_text(md)
        print(f"\n-> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
