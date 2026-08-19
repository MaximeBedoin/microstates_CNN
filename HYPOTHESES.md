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

#### Mesure effectuée (GPU, `--batch-size 1024`) — mécanisme confirmé, attribution fausse

`results/diagnostics/` :

| Bras | recon ép. 0 → 59 | GEV max (époque) | GEV ép. 59 | Perte |
|---|---|---:|---:|---:|
| `conv` | 0.341 → 0.061 | 0.6906 (ép. 15) | 0.6807 | −0.010 |
| `dense` | 0.498 → 0.047 | 0.6911 (ép. 10) | 0.6341 | −0.057 |

La divergence annoncée existe bien : la reconstruction baisse jusqu'à la
dernière époque pendant que la GEV aval culmine tôt puis se dégrade. **Mais
c'est le bras dense qui en souffre le plus**, à l'inverse de ce qui est écrit
ci-dessus, et à leur optimum respectif les deux bras sont à égalité (0.6906 vs
0.6911, soit le bruit). L'écart de 0.07 du run de référence est donc un
artefact d'époque d'arrêt, pas un écart de représentation.

*Réserve* : ce run utilisait `--batch-size 1024` contre 256 par défaut, soit 4×
moins de pas de gradient par époque — 60 époques y valent environ 15 époques du
run d'origine. Le renversement conv/dense peut n'être qu'un effet de
trajectoire. À refaire à `--batch-size 256` pour comparer à budget de pas égal.

*Correction au « fait 1 » de l'observation à expliquer* : « le bras conv
reconstruit mieux que le dense, 0.033 contre 0.045 » n'est pas une comparaison
valide, alors qu'elle sert à écarter un défaut de capacité. Les deux nombres ne
mesurent pas la même chose — `models.polarity_invariant_recon` normalise le
conv par les ~620 pixels du masque (des valeurs interpolées) et le dense par
les 64 électrodes mesurées. C'est exactement l'objet de H4. Le run ci-dessus
donne d'ailleurs l'ordre inverse (0.061 contre 0.047).

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
- **P3 — Une seule graine partout. LE PLANCHER DE BRUIT EST MESURÉ, ET IL EST
  PLUS GRAND QUE PRESQUE TOUS LES EFFETS DU DOCUMENT.** Quatre graines
  d'entraînement, tout le reste tenu constant, sur le bras token :

  | γ | GEV aval par graine | moyenne ± écart-type |
  |---:|---|---|
  | 0.05 | 0.663 · 0.589 · 0.609 · 0.654 | 0.629 ± 0.036 |
  | 0.693 | 0.609 · 0.648 · 0.678 · 0.627 | 0.641 ± 0.030 |
  | 3.00 | 0.649 · 0.511 · 0.636 · 0.563 | 0.590 ± 0.065 |

  **La variabilité entre graines vaut 0.03 à 0.065 de GEV.** À comparer aux
  effets que ce document cherche à expliquer : largeur du réseau 0.027 (H8),
  β 0.01–0.02 (H3), dégradation du bras conv au fil de l'entraînement 0.010
  (H2). Tous sont sous le plancher de bruit. Le seul écart qui le dépasse
  franchement est celui du run de référence entre `vae_conv` et les autres
  (0.07), et le diagnostic H2 a montré qu'il s'agissait d'un artefact d'époque
  d'arrêt.

  Conséquence pratique : **aucun écart de GEV inférieur à ~0.07 ne doit être
  interprété sans plusieurs graines.** Le coût est négligeable sur GPU.
- **P4 — Les runs courts mentent.** Le run de rodage (6 sujets, 4 époques)
  donnait l'ordre **inverse** : conv 0.96 vs dense 0.82 face au ground truth.
  Aucun run court ne doit servir à trancher cette question.
- **P5 — Le contraste PCA → dense reste confondu.** Le bras PCA utilise le
  clustering de Pycrostates, donc l'écart mélange « non-linéarité » et
  « changement d'algorithme de clustering ». Un bras k-means euclidien sur les
  scores PCA (après canonicalisation du signe) les séparerait.
- **P6 — La GEV ne peut pas servir à classer les méthodes.** Les cartes vraies
  y obtiennent 0.6745, *moins* que `pycrostates` (0.6882), `pca8` (0.6877) et
  `vae_dense` (0.6779). La GEV récompense l'explication de variance, bruit de
  fond 1/f compris : une méthode ajustée aux données en capte plus que la
  vérité. Le sanity check canonique a le même défaut en pire, le ground truth
  y étant *dernier* (0.746 contre 0.79–0.84). Ces deux métriques sont des
  diagnostics, pas des critères de comparaison.
- **P7 — Un effet de groupe porté par les durées ne démontre rien.** La durée
  moyenne des segments est une propriété temporelle que la puissance relative
  par bande lit directement : sur l'effet historique du générateur (85 ms
  contre 65 ms), le bras spectral atteint AUC = 1.000 sans aucun microstate.
  Seul un effet porté par la matrice de transition, **à durées appariées**,
  permet de démontrer un apport propre des microstates.
- **P8 — Les features sensibles à la longueur d'enregistrement fuient.** La
  complexité de Lempel-Ziv et le taux d'entropie sont biaisés par la longueur
  de la séquence. Le rejet d'epochs artefactées n'en retire pas la même
  proportion chez tous, et rien ne garantit que patients et témoins soient
  également artefactés — l'agitation est un symptôme. Sans troncature à durée
  égale, un classifieur peut séparer les groupes en lisant la durée de signal
  exploitable. Corrigé par `max_duration` dans `iter_ds004504`.
- **P9 — L'erreur-type d'une AUC ne s'obtient pas par la formule analytique.**
  Hanley–McNeil donne 0.053 à 120 sujets ; la valeur empirique, validation
  croisée comprise, est **0.066** (mesure : 200 réplicats de bruit pur). La
  formule sous-estime d'un quart. Utiliser la valeur empirique pour les barres
  d'erreur et les seuils de détectabilité.

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

---

## Ce qui a changé : un critère de comparaison, et deux nouveaux bras

### Le problème que tout ce qui précède avait en commun

Les hypothèses H2 à H9 sont formulées comme des explications d'un écart mesuré
en GEV. Or aucune des métriques disponibles ne pouvait départager les méthodes
(P6) : la GEV est saturée et classe la vérité en quatrième position, le sanity
check canonique la classe dernière, la stabilité split-half mesure la
reproductibilité et non la qualité — une méthode dégénérée qui renvoie toujours
les mêmes cartes est parfaitement stable — et la corrélation aux cartes vraies
n'existe pas sur données réelles.

D'où le banc de classification (`msvae/classify.py`) : on note les méthodes sur
leur capacité à séparer deux groupes de sujets. Le critère a de la marge
(0.5 → 1.0), existe sur simulé comme sur réel, se tient **hors de la boucle de
sélection** — ce qui règle P1 — et se compare entre des méthodes qui n'ont rien
d'autre en commun.

**Bras de référence obligatoire** : la puissance relative par bande. L'EEG
Alzheimer est globalement ralenti et la durée des microstates est couplée au
rythme dominant ; sans ce point zéro, une AUC n'est pas interprétable. Si les
microstates ne battent pas la puissance par bande, la représentation n'apporte
rien, et il vaut mieux le découvrir soi-même.

### Le résultat qui justifie la démarche du projet

Courbes de sensibilité, 120 sujets, effet de groupe d'amplitude croissante
(`results/sensitivity_transition/`, `results/sensitivity_duration/`) :

| Amplitude | transition : microstates | transition : spectral | durée : microstates | durée : spectral |
|---:|---:|---:|---:|---:|
| 0 | 0.399 | 0.475 | 0.399 | 0.475 |
| 0.25 | 0.663 | 0.478 | 0.799 | 0.752 |
| 0.50 | 0.831 | 0.473 | 0.983 | 0.928 |
| 1.00 | **0.957** | **0.434** | 1.000 | 1.000 |

À durées appariées, les microstates montent jusqu'à 0.957 pendant que le
spectral reste au hasard sur toute la plage : **il existe un régime où les
microstates capturent une information que la puissance par bande ne capture
pas.** Le régime « durée » sert de contrôle et se comporte à l'opposé, les deux
bras montant ensemble — ce qui établit que la séparation des deux composantes
de l'effet est réelle et non un artefact.

Ces chiffres sont ceux des cartes **vraies**, donc un plafond : ils disent que
l'information est accessible en principe, pas qu'une méthode donnée la
récupère. Comparer les méthodes se fait à t = 0.25–0.5, là où la courbe a de la
pente.

*Les AUC sous 0.5 à effet nul ont été investiguées et closes* : l'estimateur
est non biaisé (0.5040 ± 0.0047 sur 200 réplicats de bruit pur), le protocole
appliqué aux vraies features l'est aussi (étiquettes permutées : 0.4980 ±
0.0083). Le 0.399 observé est au 5ᵉ percentile de sa propre distribution nulle,
ce qui est banal une fois pris en compte qu'il s'agit du minimum de trois bras
corrélés. Aucune correction nécessaire ; voir P9 pour le seul enseignement à
retenir.

### H11 — Encodeur à attention sur les électrodes (`TokenVAE`, implémenté)

Réponse directe à H9, et supérieure à la démarche par élimination qu'il
prévoyait. Chaque électrode est un token `(valeur, position 3D)`, l'attention
est biaisée par la distance :

    logit_ij = q_i · k_j / sqrt(d) − softplus(γ_h) · d_ij

**La localité cesse d'être une hypothèse d'architecture câblée pour devenir un
paramètre appris et lisible.** γ grand = traitement local ; γ → 0 = mélange
global, la localité ne sert à rien. H9 passe donc d'une conclusion par défaut,
toujours contestable, à une **mesure** avec une prédiction falsifiable : si
l'argument des basses fréquences spatiales est correct, γ doit tendre vers 0.
`TokenVAE.learned_locality()` est exporté dans `results.json`.

Effets de bord : sans image, **H4, H7 et H1bis disparaissent par construction**
plutôt que par correctif — ni interpolation, ni masque, ni bord à padding zéro,
ni ridge à calibrer — et l'objectif est nativement en espace capteur, donc
identique à celui du bras dense. Le décodeur est celui du bras dense, pour que
l'ablation porte sur le seul encodeur. Le modèle est agnostique au montage.

Vérifié : budget de paramètres apparié (conv 108 529 / dense 110 960 / token
109 316), invariance par permutation des électrodes à 1.2e-7, invariance de
polarité héritée sans modification.

### H10 (VaDE) — implémenté, avec son mode de défaillance mesuré et corrigé

`msvae/vade.py`. Le prior est posé sur un `BaseVAE` quelconque, donc il compose
avec `TokenVAE` : encodeur à attention + prior en mélange est le modèle final
cohérent, et rien n'oblige à choisir maintenant.

L'effondrement de composantes, annoncé comme risque, s'est produit : sur
données jouet à quatre prototypes connus, 2 composantes sur 4 utilisées
(pureté 0.672) malgré l'initialisation par k-means. Corrigé par `batch_balance`
— entropie de la responsabilité moyennée sur le lot :

| `lambda_balance` | composantes | pureté | H(q̄) / max |
|---:|---:|---:|---|
| 0.0 | 2/4 | 0.672 | 0.568 / 1.386 |
| 0.5 | 4/4 | 1.000 | 1.377 / 1.386 |
| 2.0 | 4/4 | 1.000 | 1.377 / 1.386 |

Quadrupler λ ne change rien : le terme empêche la mort d'une composante sans
peser sur la vraisemblance.

**Piège verrouillé** : `elbo()` appelle volontairement `vade_loss` avec
`lambda_balance=0`. Le terme d'équilibrage n'appartient pas à l'ELBO et son
maximum vaut log K ; l'inclure ferait croître le critère mécaniquement avec K
et sélectionnerait toujours le plus grand nombre de composantes, par pur
artefact et sans que rien ne le signale.

### Ordre de test, révisé

1. ~~`diagnose_overtraining.py`~~ — fait, voir H2.
2. **Banc de classification sur les bras existants**, à t = 0.25–0.5 sur la
   cohorte synthétique élargie, avec pycrostates. C'est là que les méthodes se
   classent : ds004504 n'a pas la puissance pour ça (65 sujets non appariés,
   écart d'AUC minimal détectable ~0.07–0.08).
3. `--loss-space topo --select-epoch-by-score` — corrige H2 et H4 ensemble.
4. `TokenVAE`, jugé sur le banc, et **lecture de γ**.
5. VaDE, jugé sur le banc *et* sur la sélection de K (ELBO confronté à la
   stabilité split-half, l'ELBO seul surestimant K).
6. ds004504 comme validité externe, pas comme instrument de classement.

**Ne pas faire `--grid extended`** avant que le banc n'ait tranché : classer 32
configurations sur une métrique saturée ne produirait que du bruit bien rangé.

---

## H11, suite : ce que le bras token a réellement montré

### Le mécanisme « localité apprise » ne fonctionne pas

L'argument central de H11 était que γ, appris par descente de gradient,
mesurerait la localité optimale. **C'est faux, et vérifié trois fois.** Avec un
modèle qui apprend correctement (reconstruction 0.754 → 0.07), trois
initialisations séparées d'un facteur 60 restent exactement où on les pose :

| init de γ | γ final | reconstruction |
|---:|---:|---:|
| 0.05 | 0.056 | 0.0715 |
| 0.693 | 0.707 | 0.0825 |
| 3.00 | 2.982 | 0.0810 |

La surface de loss est plate en γ. Lire γ après entraînement ne renseigne que
sur son initialisation. (Vérifié séparément que γ agit bien sur la sortie —
|μ| passe de 0.432 à 0.417 entre γ = 0.001 et γ = 10 — donc il ne s'agit pas
d'un bug mais d'un gradient non informatif.)

### Le balayage de γ ne montre pas d'effet non plus

Traiter γ en hyperparamètre balayé était la parade. À une graine, la courbe
semblait décroissante puis remontait au dernier point (0.663, 0.670, 0.609,
0.576, 0.649) : non monotone, donc suspecte. À quatre graines, voir P3 — les
moyennes se recouvrent, la variabilité entre graines est du même ordre que
l'écart entre valeurs de γ.

**Conclusion pour H9 : à ce protocole, la localité n'a pas d'effet mesurable
sur la qualité aval.** Ce n'est ni une confirmation ni une réfutation de
l'argument des basses fréquences spatiales — c'est un résultat nul, et il
signifie que ce dispositif ne tranchera pas la question.

### Le piège d'optimisation, à ne pas rouvrir

Le bras token **plateaute pendant ~30 époques** au taux d'apprentissage des
bras conv et dense (2e-3) avant de décrocher :

    lr=2e-3 : 0.755 0.754 0.754 0.496 0.108 0.081 0.078 0.073  (par 10 époques)
    lr=5e-4 : 0.754 0.153 0.123 0.103 0.096 0.088 0.085 0.083

Avec `patience=10`, l'early stopping l'arrête en plein plateau. Tous les
chiffres du bras token antérieurs au correctif mesurent donc un modèle qui n'a
rien appris — GEV 0.584, AUC 0.603, deux cartes sur quatre, et une stabilité
split-half de **1.000 ± 0.000** qui en est la signature : un modèle figé est
parfaitement reproductible. Corrigé par `token_lr` et `token_patience` dans
`ExperimentConfig`.

*Leçon générale* : la stabilité split-half ne distingue pas un bon modèle d'un
modèle mort. Elle doit toujours être lue à côté d'une mesure de qualité.

### Ce qui reste de solide

Le seul enseignement du balayage qui ne dépende pas de la monotonie : **la
reconstruction est plate sur toute la plage de γ (0.0696–0.0825, sans ordre)
pendant que la GEV aval varie de 0.094.** Une seule variable change, tout le
reste est tenu constant. C'est la démonstration la plus propre du désalignement
d'objectif de tout le projet — plus propre que les courbes de sur-entraînement,
où la durée d'entraînement faisait bouger plusieurs choses à la fois.
