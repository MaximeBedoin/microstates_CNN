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

### Reprise en local sur GPU

```bash
git clone <url-du-depot> && cd microstates_CNN
git checkout claude/eeg-microstate-vae-c6lz01

python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt                     # torch CUDA : voir pytorch.org
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

python -m pytest tests -q        # 33 tests, ~30 s
```

Le GPU est détecté seul (`--device auto`) ; chaque script affiche le
périphérique retenu au démarrage. Sur GPU, montez la taille de batch :
`--batch-size 1024` (voire 2048), le modèle ne fait que ~110 k paramètres.

**Les runs à lancer**, dans cet ordre — les correctifs sont implémentés, il n'y
a qu'à exécuter. Durées indicatives sur un GPU de bureau :

```bash
# 0. données réelles : le cache n'est pas versionné, à re-télécharger (~150 Mo)
python scripts/download_eegbci.py --first 1 --last 60

# 1. mécanisme : les courbes reconstruction et GEV aval divergent-elles ?  (~2 min)
python scripts/diagnose_overtraining.py --batch-size 1024
#    -> results/diagnostics/figures/overtraining.png

# 2. objectif corrigé : loss sur les électrodes + époque choisie sur la GEV  (~6 min)
python scripts/run_synthetic.py --n-subjects 20 --duration 60 --epochs 60 \
    --loss-space topo --select-epoch-by-score --stability-refit \
    --batch-size 1024 --out results/synthetic_topoloss

# 3. grille d'architecture élargie : 32 configurations  (~40 min)
python scripts/run_synthetic.py --grid extended --loss-space topo \
    --select-epoch-by-score --batch-size 1024 --out results/synthetic_extended

# 4. données réelles, mêmes réglages  (~10 min)
python scripts/run_eegbci.py --n-subjects 60 --epochs 30 --n-per-subject 1000 \
    --no-arch-search --loss-space topo --select-epoch-by-score --batch-size 1024

# synthèse lisible de n'importe quel run
python scripts/report.py results/<run>/results.json --out results/<run>/RESULTS.md
```

Le run 2 est le plus informatif : il se compare directement à
`results/synthetic/` (même graine, même protocole, seul l'objectif change).

**Piège à éviter en lisant les résultats** : ne pas sélectionner sur la GEV
puis comparer les bras sur la GEV. Les bras `pycrostates` et `pca8_modkmeans`
ne reçoivent aucun réglage, donc la comparaison doit porter sur une métrique
tenue hors de la sélection — corrélation au ground truth, stabilité
split-half, pouvoir discriminant. Voir `CHOIX_METHODO.md` §9.

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
