# Hypothèses en cours — pourquoi le bras convolutionnel sous-performe

Document de passation. Il récapitule l'observation à expliquer, toutes les
hypothèses formulées, leur statut, les preuves accumulées et la façon de
tester celles qui restent ouvertes.

Contexte complet : `CHOIX_METHODO.md` §9. Résultats bruts :
`results/synthetic/RESULTS.md`.

---

## L'observation à expliquer

Cohorte synthétique, 20 sujets × 60 s, K = 4, latent = 8, 42 364 pics de GFP,
ground truth connu (dipôles simulés).

| Bras | GEV | corrélation au ground truth | stabilité split-half |
|---|---:|---:|---:|
| `pycrostates` (modified k-means, 64 valeurs) | 0.688 | 0.985 | 0.992 |
| `pca8_modkmeans` (compression linéaire rang 8) | 0.688 | 0.983 | 0.992 |
| `vae_dense` (goulot non linéaire, vecteurs) | 0.678 | 0.976 | 0.876 |
| `vae_conv` (goulot non linéaire, **images**) | 0.609–0.621 | 0.860–0.893 | 0.937 |

Deux faits à garder en tête, ils orientent tout le diagnostic :

1. **Le bras conv reconstruit MIEUX** que le dense (erreur de validation 0.033
   contre 0.045) tout en clusterisant moins bien. Ce n'est donc pas un défaut
   de capacité ni d'optimisation au sens habituel.
2. **La compression à 8 dimensions ne coûte rien en soi** : le bras PCA rang 8
   égale exactement le pipeline classique (0.688 vs 0.688). Le goulot
   contraint n'est pas le problème.

---

## Hypothèses écartées

### H1 — Le décodage image → électrodes dégrade les cartes
**ÉCARTÉE.** Les cartes ont été recalculées sans décodeur du tout (premier
vecteur propre des topographies assignées à chaque cluster) :

| | carte décodée | carte en espace capteur |
|---|---:|---:|
| conv | 0.860 | 0.873 |
| dense | 0.976 | 0.978 |

L'inversion coûte 0.013, pas les 0.11 d'écart observés. Le déficit est dans la
**partition latente elle-même**.

### H1bis — La pseudo-inverse image → électrodes est instable
**CONFIRMÉE ET DÉJÀ CORRIGÉE** (avant le run de référence). W est mal
conditionnée (cond ≈ 4·10³) : la pseudo-inverse exacte amplifiait le résidu
hors-variété des images décodées (r = 0.19), produisant des cartes de bruit et
une GEV de 0.04 — **sans qu'aucune erreur ne soit levée**. Corrigé par une
ridge calibrée (λ = 0.1·tr), qui rétablit r = 0.87 sur les images décodées et
reste exacte à 0.997 sur les images vraies.

### H8 — La largeur du réseau
**TESTÉE, EFFET MODESTE.** À latent = 8, β = 10⁻³ : largeur 8 → 0.6780,
16 → 0.6947, 32 → 0.6682. Amplitude 0.027, soit environ trois fois moins que
l'effet de la durée d'entraînement (0.073). L'optimum est *intérieur* à la
grille, donc correctement encadré. Ce n'est pas là qu'il manque d'exploration.

### H0 — L'invariance de polarité serait mal imposée
**VÉRIFIÉE, NON PROBLÉMATIQUE.** L'écart latent résiduel ‖μ(x) − μ(−x)‖² /
(‖μ(x)‖² + ‖μ(−x)‖²) est journalisé à chaque époque et tombe sous 10⁻³ en
quelques époques (5·10⁻⁵ pour le conv du run final). La contrainte de
consistance latente fait son travail.

---

## Hypothèse principale, fortement soutenue

### H2 — L'objectif entraîné n'est pas aligné sur la tâche : sur-entraînement
La **même configuration conv** évaluée sur les **mêmes sujets de validation** :

| | GEV validation |
|---|---:|
| conv, 25 époques (pendant la recherche d'archi) | **0.6947** |
| conv, 60 époques (modèle final) | **0.6224** |
| dense, 60 époques | 0.6931 |

À 25 époques, **le conv est à égalité avec le dense**. C'est l'entraînement
prolongé qui détruit le résultat, pendant que l'erreur de reconstruction
continue de baisser. Mécanisme cohérent : l'early stopping surveillait la loss
de validation, dominée par la reconstruction — le modèle optimise une quantité
qui n'est pas celle qui nous intéresse, et la géométrie latente se dégrade
pour le clustering.

*Réserve d'honnêteté* : les deux modèles comparés diffèrent aussi par la
taille du jeu d'entraînement et par le fait que le modèle final a vu les
sujets de validation. Ces deux différences jouent **en faveur** du modèle
final, ce qui rend la chute d'autant plus significative — mais le test direct
reste à faire.

**Test** : `python scripts/diagnose_overtraining.py` → trace les deux courbes
(reconstruction et GEV aval) époque par époque pour les deux bras.
**Correctif implémenté** : `--select-epoch-by-score`.

---

## Hypothèses ouvertes, toutes implémentées et prêtes à tester

### H3 — β trop faible, grille trop étroite
Tendance monotone : plus de pression KL = moins bonne reconstruction,
**meilleure** GEV.

| latent | β=10⁻⁴ | β=10⁻³ | β=10⁻² |
|---:|---:|---:|---:|
| 4 | 0.6642 | 0.6742 | **0.6797** ← optimum au bord |
| 8 | 0.6819 | **0.6947** | 0.6940 |
| 16 | 0.6778 | 0.6908 | **0.6935** ← optimum au bord |

Pour deux dimensions latentes sur trois, l'optimum est à la borne : la grille
s'arrête trop tôt. **Test** : `--grid extended` (β jusqu'à 10⁻¹).

### H4 — La loss en espace image pondère mal le scalp
La MSE sur les pixels est une **intégrale sur l'aire du disque**, pas sur les
électrodes. Le modèle dépense donc sa capacité à bien reproduire des valeurs
**interpolées — donc inventées** — dans les régions à faible densité de
capteurs, au lieu des mesures réelles. Le bras dense, lui, optimise exactement
les 64 mesures.

**Test** : `--loss-space topo`. La sortie du décodeur est ramenée aux 64
électrodes par l'opérateur linéaire fixe (matrice de lecture, différentiable),
ce qui rend l'objectif **identique** à celui du bras dense : l'ablation ne
porte alors plus que sur l'encodeur. C'est la version la plus propre du test.

### H5 — Sous-échantillonnage trop agressif
Trois blocs stride-2 réduisent 32×32 à **4×4**, soit 16 positions spatiales.
Or ce qui distingue une carte A d'une carte B, c'est l'**orientation d'un
gradient diagonal** — information géométrique fine que 4×4 peut détruire.
**Test** : `--grid extended` (profondeur 2 ou 3 blocs).

### H6 — Champ réceptif inadapté
Noyaux 4×4 sur une image dont le contenu est dominé par les **plus basses**
fréquences spatiales. Pour « voir » un gradient antéro-postérieur global, un
convnet à petits noyaux doit empiler des couches, donc perdre de la
résolution — ce qui ramène à H5. **Test** : `--grid extended` (noyaux 4 ou 6).

### H7 — Gestion du bord du masque *(non implémentée)*
~40 % de la grille 32×32 est à zéro hors du disque. Les convolutions à padding
zéro traitent la frontière du masque comme un contour très saillant : une part
de la capacité part probablement à encoder la **forme du masque**, qui ne porte
aucune information. Pistes : convolutions masquées, extrapolation au lieu de
zéros (MNE le fait pour ses topomaps), ou recadrage rectangulaire.

---

## Hypothèse de fond, à trancher par élimination

### H9 — Le prior convolutionnel est intrinsèquement inadapté aux topographies
Le biais inductif convolutionnel, c'est **localité + équivariance par
translation**. Or :

- une topographie EEG est dominée par les plus basses fréquences spatiales
  (essentiellement des harmoniques sphériques de bas ordre) : **la localité
  n'est pas le bon prior** ;
- **la translation n'est pas une symétrie des données** — déplacer un motif
  sur le scalp change son sens anatomique, alors que déplacer un chat dans une
  photo n'en change pas.

Cette hypothèse n'est pas testable directement : elle se conclut si le bras
conv reste derrière **après** correction de H2, H4, H5, H6. Ce serait alors un
résultat publiable en soi — *la structure spatiale intégrée par construction
n'aide pas si la construction encode la mauvaise symétrie* — et la suite
naturelle serait une famille d'architectures respectant la géométrie réelle :
**convolutions sphériques**, ou **réseau de graphe sur le graphe des
électrodes**, où le voisinage est celui du montage et non celui d'une grille
carrée.

### H10 — L'objectif VAE n'est pas clustering-aware, quelle que soit la loss
Arrêter sur la GEV est un correctif, pas une solution : on reconnaît que
l'objectif entraîné n'est pas celui qui nous intéresse et on compense à la
sortie. Route plus propre : remplacer le prior gaussien par un **mélange de
gaussiennes** (VaDE / deep clustering). Le modèle optimise alors une
vraisemblance qui contient la structure en K classes, le k-means aval devient
une lecture du modèle, et **K devient un paramètre du modèle sélectionnable
par vraisemblance** au lieu d'être fixé a priori. Non implémenté.

---

## Pièges méthodologiques à ne pas rouvrir

- **P1 — Ne pas sélectionner sur la GEV puis comparer sur la GEV.** Les bras
  `pycrostates` et `pca8_modkmeans` ne reçoivent aucun réglage ; donner un
  budget de tuning au seul bras conv, sur la métrique de comparaison, fabrique
  un avantage. Comparer sur une métrique tenue hors sélection : corrélation au
  ground truth, stabilité split-half, pouvoir discriminant.
- **P2 — Les sujets de validation servent deux fois** (sélection d'archi puis
  sélection d'époque). Pour un chiffre non biaisé sur données réelles, il faut
  un split à trois : entraînement / sélection / test.
- **P3 — Une seule graine partout.** Aucun écart rapporté n'a d'intervalle de
  confiance. L'effet de largeur (0.027) pourrait n'être que du bruit
  d'initialisation. À corriger en priorité sur GPU, le coût devient
  négligeable.
- **P4 — Les runs courts mentent.** Le run de rodage (6 sujets, 4 époques)
  donnait l'ordre **inverse** : conv 0.96 vs dense 0.82 face au ground truth.
  Aucun run court ne doit servir à trancher cette question.
- **P5 — Le contraste PCA → dense reste confondu.** Le bras PCA utilise le
  clustering de Pycrostates, donc l'écart mélange « non-linéarité » et
  « changement d'algorithme de clustering ». Un bras k-means euclidien sur les
  scores PCA (après canonicalisation du signe) les séparerait.

---

## Ordre de test recommandé

1. `scripts/diagnose_overtraining.py` — établit ou infirme H2, produit une
   figure directement utilisable.
2. `--loss-space topo --select-epoch-by-score` — corrige H2 et H4 ensemble, se
   compare terme à terme à `results/synthetic/` (même graine, même protocole).
3. `--grid extended` — teste H3, H5, H6 en une passe (32 configurations).
4. Plusieurs graines sur la meilleure configuration — répond à P3.
5. Si l'écart persiste : H9 est la conclusion, et le projet bascule vers les
   convolutions sphériques ou de graphe.

**Ne pas faire (3) avant (2)** : on classerait des architectures optimisées
pour le mauvais objectif.
