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
