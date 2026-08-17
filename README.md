# Microstates EEG par VAE convolutionnel sur images topographiques

Alternative aux pipelines de microstates classiques (modified k-means sur les
vecteurs d'électrodes) où **la structure spatiale du scalp est intégrée par
construction** : chaque topographie est rendue sous forme d'image 2D, un VAE
convolutionnel apprend un espace latent de faible dimension, et le clustering
des microstates se fait dans cet espace latent.

Les choix de conception, les alternatives écartées et les points fragiles sont
documentés dans [`CHOIX_METHODO.md`](CHOIX_METHODO.md).

## Chaîne de traitement

```
EEG  ──filtre 1–40 Hz, réf. moyenne──►  pics de GFP  ──projection + spline sphérique──►  images 32×32
                                                                                              │
                                                          VAE conv (latent 8, invariant polarité)
                                                                                              │
                                              k-means dans le latent (2 temps ou 1 temps pondéré)
                                                                                              │
                     décodage des centroïdes ──►  topographies interprétables  ──►  back-fitting
                                                                                              │
                            durée / occurrence / coverage / GEV / transitions / entropie / LZC
```

## Installation

```bash
pip install numpy scipy scikit-learn matplotlib mne pycrostates torch pytest
```

## Utilisation

```bash
# cohorte synthétique (dipôles connus, ground truth disponible)
python scripts/run_synthetic.py --n-subjects 20 --duration 60 --epochs 60 --stability-refit

# EEG réel : EEGBCI / PhysioNet eegmmidb (repos yeux ouverts vs yeux fermés)
python scripts/download_eegbci.py --first 1 --last 60
python scripts/run_eegbci.py --n-subjects 60 --epochs 40 --n-per-subject 1500

# tests
python -m pytest tests -q
```

Chaque run écrit dans `--out` : `results.json` (toutes les métriques),
`maps.npz` (cartes de chaque méthode), `model_{conv,dense}.pt` et `figures/`.

## Modules

| Module | Rôle |
|---|---|
| `msvae/topo.py` | projection azimuthale + splines sphériques, topo ↔ image |
| `msvae/preprocess.py` | filtrage, référence moyenne, pics de GFP |
| `msvae/synthetic.py` | cohorte simulée à dipôles connus (sphère 3 couches) |
| `msvae/data.py` | téléchargement EEGBCI (miroir S3) et cache disque |
| `msvae/features.py` | banque de pics, split par sujet, équilibrage |
| `msvae/models.py` | VAE conv et VAE dense, losses invariantes en polarité |
| `msvae/train.py` | boucle d'entraînement, recherche d'architecture |
| `msvae/cluster.py` | k-means latent (2 temps / pondéré), décodage des centroïdes |
| `msvae/microstates.py` | back-fitting et paramètres standards |
| `msvae/baseline.py` | pipeline Pycrostates (modified k-means) |
| `msvae/evaluate.py` | appariement hongrois, stabilité, comparaison de groupes |
| `msvae/pipeline.py` | orchestration bout-en-bout |

## État d'avancement et reprise en local

**Fait et versionné** : pipeline complet, 32 tests, un run synthétique de
référence (`results/synthetic/`, avec `RESULTS.md`, les cartes et les modèles
entraînés).

**Résultat principal et sa mise en garde** : voir
[`CHOIX_METHODO.md` §9](CHOIX_METHODO.md). En bref, le bras convolutionnel
sous-performe (GEV 0.61 vs 0.69), mais trois diagnostics montrent que c'est
l'objectif d'entraînement qui est en cause, pas la représentation — la même
configuration conv atteignait 0.695 à 25 époques contre 0.622 à 60, sa
reconstruction continuant de s'améliorer.

**Reste à lancer**, dans cet ordre (les correctifs sont implémentés, il n'y a
qu'à exécuter) :

```bash
# 0. données réelles : le cache n'est pas versionné, à re-télécharger (~150 Mo)
python scripts/download_eegbci.py --first 1 --last 60

# 1. mécanisme : les courbes reconstruction vs GEV aval divergent-elles ?
#    ~15 min CPU / ~2 min GPU  ->  results/diagnostics/figures/overtraining.png
python scripts/diagnose_overtraining.py

# 2. objectif corrigé : loss sur les électrodes + époque choisie sur la GEV
#    ~1 h 15 CPU / ~6 min GPU
python scripts/run_synthetic.py --n-subjects 20 --duration 60 --epochs 60 \
    --loss-space topo --select-epoch-by-score --stability-refit \
    --out results/synthetic_topoloss

# 3. grille d'architecture élargie (β jusqu'à 1e-1, profondeur 2/3, noyaux 4/6)
#    32 configurations : à réserver au GPU
python scripts/run_synthetic.py --grid extended --loss-space topo \
    --select-epoch-by-score --out results/synthetic_extended

# 4. données réelles, mêmes réglages  (~50 min CPU / ~5 min GPU)
python scripts/run_eegbci.py --n-subjects 60 --epochs 30 --n-per-subject 1000 \
    --no-arch-search --loss-space topo --select-epoch-by-score

# synthèse lisible de n'importe quel run
python scripts/report.py results/<run>/results.json --out results/<run>/RESULTS.md
```

Le code détecte le GPU automatiquement (`device='auto'`), aucun changement
n'est nécessaire. Comptez un facteur ~15 : l'entraînement est le poste de coût
dominant et le modèle ne fait que ~110 k paramètres sur des images 32×32.

`scripts/run_pending.sh` enchaîne plusieurs runs en commitant les résultats
après chacun — utile si l'exécution risque d'être interrompue.

## Trois volets de validation

1. **Sanity check** : à K = 4, les cartes de groupe doivent ressembler aux
   microstates canoniques A/B/C/D (proxys géométriques + comparaison avec
   Pycrostates sur les mêmes données).
2. **Ablation centrale** : VAE dense sur vecteurs vs VAE conv sur images, à
   dimension latente et **budget de paramètres appariés**. L'interpolation
   étant déterministe, elle n'ajoute aucune information : tout écart mesure le
   biais inductif de l'architecture convolutionnelle.
3. **Comparaison externe** : Pycrostates vs VAE sur les mêmes epochs, mêmes
   pics de GFP, même K — stabilité split-half, GEV, pouvoir discriminant entre
   groupes.
