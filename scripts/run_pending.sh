#!/bin/bash
# Enchaine les runs restants et commite les resultats apres chacun, pour
# survivre a un redemarrage de conteneur.
cd "$(dirname "$0")/.."
set -x

python scripts/diagnose_overtraining.py --threads 4 > results/overtraining.log 2>&1
git add -A -f results/diagnostics results/overtraining.log 2>/dev/null
git commit -q -m "Diagnostic : courbes reconstruction vs GEV aval au fil de l'entrainement" || true
git push -q origin claude/eeg-microstate-vae-c6lz01 || true

python scripts/run_eegbci.py --n-subjects 60 --epochs 30 --latent-dim 8 \
    --n-per-subject 1000 --no-arch-search --stability-repeats 4 --threads 4 \
    --out results/eegbci > results/eegbci_run.log 2>&1
python scripts/report.py results/eegbci/results.json --out results/eegbci/RESULTS.md
git add -A -f results/eegbci results/eegbci_run.log 2>/dev/null
git commit -q -m "Resultats EEGBCI (60 sujets, 194k pics de GFP)" || true
git push -q origin claude/eeg-microstate-vae-c6lz01 || true
echo "TOUT TERMINE"
