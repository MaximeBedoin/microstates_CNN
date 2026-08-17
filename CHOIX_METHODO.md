# Choix méthodologiques

Journal des décisions de conception, des alternatives écartées et des points
fragiles. Destiné à servir de base à la section méthodes d'un article et à
identifier ce qui doit être testé plus rigoureusement.

Statut : pipeline fonctionnel de bout en bout (synthétique + EEGBCI).
Dernière mise à jour : voir historique git.

---

## 0. Contexte d'exécution

Environnement conteneurisé sans GPU (4 CPU, 15 Go RAM), accès réseau
restreint : PhysioNet, OSF, Zenodo et OpenNeuro sont bloqués par le proxy,
mais le miroir AWS Open Data de PhysioNet
(`physionet-open.s3.amazonaws.com`) répond. Toutes les décisions de coût de
calcul ci-dessous découlent de cette contrainte (CPU seul).

---

## 1. Jeux de données

### 1.1 Décision

Deux jeux, utilisés pour des rôles différents :

| Jeu | Rôle | Détail |
|---|---|---|
| **Synthétique** | validation contre ground truth | dipôles connus, K vrai connu, séquence d'états connue, SNR contrôlé |
| **EEGBCI** (PhysioNet `eegmmidb`) | validation sur données réelles | 109 sujets, 64 électrodes, 160 Hz, repos yeux ouverts (run 1) et yeux fermés (run 2), 1 min chacun |

`mne.datasets.eegbci` pointe sur `physionet.org` (bloqué) ; on télécharge donc
directement depuis le miroir S3 (`msvae/data.py:download_eegbci`).

### 1.2 Alternatives écartées

- **LEMON** (utilisé par les tutoriels Pycrostates) : hébergé sur un serveur
  MPI/OSF, inaccessible ici.
- **Un dataset clinique** (TDBRAIN, OpenNeuro) : aurait donné de vrais groupes
  cliniques, mais tous les hôtes testés sont bloqués. Le contraste
  **yeux ouverts / yeux fermés** sert de substitut : il est intra-sujet donc
  apparié, bien documenté sur les microstates (B et C notamment), et suffit
  pour valider *la machinerie* du volet « pouvoir discriminant ». Il ne
  valide évidemment pas la sensibilité à une pathologie.

### 1.3 Génération synthétique

- **Modèle direct** : sphère 3 couches de MNE (`make_sphere_model`,
  approximation de Berg). Aucun téléchargement (fsaverage est sur OSF, bloqué),
  et la physique reste correcte.
- **Dipôles des états** : choisis par recherche numérique pour minimiser la
  corrélation absolue maximale entre topographies (max |r| = 0.34) tout en
  respectant la description des microstates canoniques : A et B tangentiels
  excentrés (gradients diagonaux), C radial occipital, D radial fronto-central.
  *Hésitation* : quatre dipôles centraux purement tangentiels étaient plus
  élégants, mais les champs en gradient vivent dans un espace de dimension 3,
  donc deux des quatre cartes se retrouvaient corrélées à 0.71 — trop pour un
  ground truth exploitable.
- **Décours temporel** : semi-markovien, durées Gamma(shape=4, moyenne 80 ms),
  pas de self-transition, **polarité tirée au hasard à chaque segment** (sans
  quoi l'invariance en polarité ne serait jamais mise à l'épreuve).
- **Bruit** : dipôles aléatoires à spectre 1/f — donc spatialement corrélé.
  Un bruit blanc capteur aurait été trivialement éliminé par l'interpolation
  et aurait avantagé artificiellement le VAE convolutionnel dans l'ablation.
- **Variabilité inter-sujets** : jitter de position (σ = 6 mm) et d'orientation
  des dipôles ; deux groupes avec durées de segment et matrice de transition
  différentes (effet « clinique » connu).

---

## 2. Prétraitement

- **Filtrage 1–40 Hz** au lieu du seul passe-bas 40 Hz spécifié. Le passe-haut
  est indispensable : sans lui les dérives lentes dominent la GFP et les
  « pics » ne correspondent plus à des topographies stables. 1 Hz est le
  compromis usuel de la littérature microstates.
  *Point à revoir* : une partie de la littérature utilise 2–20 Hz. Le choix de
  la bande a un effet documenté sur les cartes obtenues — à tester en
  sensibilité.
- **Référence moyenne** avant tout calcul de GFP (la GFP n'a de sens que sur
  données re-référencées en moyenne).
- **Pics de GFP** : maxima locaux avec distance minimale de 3 échantillons
  (spécification). À 160 Hz cela vaut 18.75 ms, à 250 Hz 12 ms.
  *Point fragile* : ce critère dépend de la fréquence d'échantillonnage, donc
  le nombre de pics par seconde n'est pas directement comparable entre EEGBCI
  (160 Hz) et le synthétique (250 Hz).
- **Rejet des pics au-delà du 99e percentile de GFP** (artefacts résiduels),
  comme Pycrostates.
- **Normalisation par la GFP** de chaque topographie : le VAE apprend des
  motifs spatiaux et non des amplitudes. Effet secondaire utile : chaque
  vecteur de 64 canaux a exactement une variance de 1 par construction, ce qui
  rend les MSE du VAE dense et du VAE conv directement comparables.

---

## 3. Topographie → image 2D

### 3.1 Décisions

- **Projection azimuthale équidistante** (celle des topomaps MNE) : l'angle
  polaire depuis le vertex devient le rayon, l'azimut est conservé. Rayon
  normalisé pour que l'électrode la plus basse tombe à 0.95 du disque unité.
- **Interpolation par splines sphériques** (Perrin et al. 1989, ordre m = 4,
  50 termes de Legendre), calculée **sur la sphère** ; seule la grille de
  sortie est plane.
  *Alternative écartée* : triangulation de Clough-Tocher en 2D (ce que fait
  `mne.viz.plot_topomap` par défaut). Elle est plus rapide mais interpole dans
  le plan déjà déformé par la projection, ce qui n'a pas de sens physique près
  du bord du casque.
- **Résolution 32×32.** Justification empirique : le round-trip
  topo → image → topo donne r = 0.997 en 32×32 **comme** en 64×64 (mesuré sur
  du bruit blanc à 64 canaux, cas le plus défavorable ; sur des topographies
  réelles r > 0.9999). Passer à 64×64 quadruple le coût de calcul sans ajouter
  d'information.
- **Extérieur du disque mis à zéro et masqué dans la loss** : sinon le modèle
  dépense de la capacité à reconstruire des zéros.
- **Toute l'opération est linéaire** : `image = W · v`, avec W précalculée une
  fois. L'interpolation de 40 000 pics est un seul produit matriciel.

### 3.2 Le point le plus délicat : l'inversion image → électrodes

Nécessaire pour décoder les centroïdes en topographies interprétables et pour
le back-fitting. Trois options implémentées :

| Méthode | Sur images issues de `to_image` | Sur images **décodées** par le VAE |
|---|---|---|
| pseudo-inverse exacte | r = 1.0000 | **r = 0.19** |
| ridge (λ = 0.1·tr) | r = 0.997 | **r = 0.87** |
| échantillonnage bilinéaire | r = 0.62 | r = 0.87 |

W est mal conditionnée (cond ≈ 4·10³) : la pseudo-inverse exacte amplifie
violemment le résidu hors-variété des images décodées. **C'est un piège
sérieux** — avec la pseudo-inverse, le pipeline complet tournait sans erreur
et produisait des cartes de bruit (GEV = 0.04 au lieu de 0.68). La
régularisation ridge a été calibrée empiriquement (plateau entre 3·10⁻² et
3·10⁻¹ de la trace, valeur retenue 0.1) : elle est équivalente au bilinéaire
sur les images décodées, mais reste quasi exacte sur les images vraies.

*Point à revoir* : λ est calibré sur un seul modèle entraîné. Il faudrait
vérifier que l'optimum ne dépend pas fortement de l'architecture ou du jeu de
données, ou remplacer la ridge par une projection explicite sur la variété
des images admissibles (`W · pinv(W)`).

---

## 4. VAE

### 4.1 Invariance en polarité

Deux mécanismes **complémentaires**, et le second est le plus important :

1. **Loss de reconstruction invariante** :
   `L = min(‖D(E(x)) − x‖², ‖D(E(x)) + x‖²)`, plus augmentation (x ou −x tiré
   au hasard à chaque batch).
2. **Loss de consistance latente** : `λ · ‖μ(x) − μ(−x)‖²`, λ = 1 par défaut.

**Justification du point 2** (décision prise en concertation) : le point 1 seul
n'impose *rien* à l'encodeur. Il autorise μ(x) ≠ μ(−x) du moment que le
décodeur sait produire l'une ou l'autre polarité. Or tout le pipeline aval
(k-means euclidien dans le latent) suppose que deux pics de polarité opposée
tombent au même endroit. Sans le point 2, on risque un latent à deux nuages
miroirs par microstate, et K = 4 se met à capturer des polarités plutôt que des
topographies — le sanity check A/B/C/D échouerait pour une raison purement
architecturale. Le coût est d'environ un passage encodeur supplémentaire par
batch.

L'invariance résiduelle est **mesurée et journalisée** à chaque époque
(`pol_rel` = ‖μ(x) − μ(−x)‖² / (‖μ(x)‖² + ‖μ(−x)‖²)) : elle tombe sous 10⁻³ en
quelques époques.

*Alternatives écartées* :
- **Canonicalisation déterministe du signe** (fixer le signe par la projection
  sur le premier PC) : exactement invariante et gratuite, mais la règle est
  discontinue — deux topographies quasi identiques près de la frontière de
  décision reçoivent des signes opposés, ce qui injecte du bruit dans les
  données d'entraînement.
- **Invariance architecturale stricte** (non-linéarité paire type |·| après la
  première convolution) : exacte par construction, mais ampute la capacité
  expressive des premières couches et risquait de faire perdre l'ablation au
  VAE conv pour une mauvaise raison.

### 4.2 Architecture et goulot

- **Latent 8 par défaut**, balayé sur {4, 8, 16} dans la recherche
  d'architecture. Avec 64 électrodes, 8 dimensions forcent bien une
  compression non triviale.
- **Conv** : 3 blocs stride-2 (32→16→8→4), largeur de base 16, GroupNorm +
  SiLU, sortie linéaire (les données sont signées).
- **KL** : β balayé sur {10⁻⁴, 10⁻³, 10⁻²} avec montée linéaire sur 10 époques
  (évite le posterior collapse en début d'entraînement).

### 4.3 Équité de l'ablation dense vs conv

C'est le test central du projet, donc les conditions sont appariées :
- même dimension latente, même β, même optimiseur, même nombre d'époques,
  même early stopping, mêmes pics ;
- **même budget de paramètres** : `match_dense_to_conv` choisit la largeur des
  couches denses par recherche binaire pour tomber à moins de 5 % du nombre de
  paramètres du conv (typiquement 108 k vs 111 k) ;
- MSE comparables car les deux représentations ont une variance ≈ 1 par
  dimension.

*Point fragile assumé* : un VAE dense « à budget apparié » est
sur-paramétré par rapport à son entrée (64 dimensions), alors qu'un dense
plus petit serait peut-être meilleur. Il faudrait ajouter un dense « de taille
raisonnable » comme troisième bras pour vérifier que la conclusion ne dépend
pas de ce choix.

*Rappel théorique à conserver dans l'article* : l'interpolation est
déterministe et bijective en pratique (r = 0.997) — elle **n'ajoute aucune
information**. Tout écart de performance entre les deux bras vient du biais
inductif convolutionnel (localité, partage de poids, invariance par
translation approximative sur le scalp) et de la géométrie de l'espace latent
qui en résulte, pas des données.

---

## 5. Protocole d'entraînement

- **Split train/val par sujet** (jamais par image). Le split par image serait
  une fuite massive : deux pics consécutifs du même sujet sont quasi
  identiques.
- **Équilibrage** : n pics par sujet, n fixé au **10e percentile** de la
  distribution du nombre de pics par sujet. Compromis entre « un sujet long ne
  doit pas dominer » et « ne pas jeter 80 % des données ».
- **Recherche d'architecture sur le split train/val**, avec entraînement
  raccourci (25 époques) et sous-échantillonné (12 000 pics) : on compare des
  architectures, on n'a pas besoin de la convergence finale. Puis
  **ré-entraînement final sur tous les sujets** avec la meilleure configuration
  (60 époques).
  *Point fragile* : le ré-entraînement final inclut les sujets de validation,
  donc la loss de validation rapportée pour le modèle final n'est plus une
  estimation honnête. Elle ne sert qu'à la sélection d'architecture ; les
  métriques rapportées ensuite (GEV, stabilité) sont calculées autrement.

---

## 6. Clustering et décodage

- **Deux stratégies implémentées et comparées** :
  - `two_stage` : k-means par sujet puis k-means de groupe sur les prototypes
    individuels (logique Pycrostates, poids égal par sujet) ;
  - `weighted` : un seul k-means sur tous les pics, chaque pic pondéré par
    1/n_pics du sujet.
  En pratique les deux donnent des cartes appariées à |r| > 0.99 sur le
  synthétique — la question du « deux temps vs un temps » semble donc peu
  critique ici, ce qui est en soi un résultat à rapporter.
- **k-means euclidien standard** dans le latent (et non un k-means
  polarity-invariant) : c'est légitime *parce que* l'encodeur est contraint à
  l'invariance (§4.1). Sans cette contrainte, il aurait fallu clusteriser sur
  |corrélation|.
- **Décodage des centroïdes** : centroïde → décodeur → image → topographie
  (ridge) → re-référence moyenne → norme 1. C'est ce qui rend la méthode
  interprétable et comparable aux cartes de la littérature.

---

## 7. Back-projection et paramètres

- **Back-fitting sur tous les échantillons** (pas seulement les pics) par
  corrélation spatiale avec les cartes **décodées**, classe = argmax |r|.
  *Alternative implémentée mais secondaire* : encoder chaque échantillon et
  assigner dans le latent. Le choix du backfit topographique comme méthode
  primaire est délibéré : il garde GEV et durées directement comparables à la
  littérature, et il place les deux bras de l'ablation sur le même terrain.
- **Rejet des segments < 30 ms** : réaffectation à la classe voisine de plus
  forte corrélation, par ordre de durée croissante (liste chaînée + tas
  binaire, O(n log n)).
  *Piège rencontré* : une implémentation naïve avec un nombre fixe d'itérations
  laisse passer des segments d'un échantillon sur des données bruitées.
- **Paramètres calculés** : durée moyenne, occurrence (par s), coverage, GEV
  par classe et globale, matrice de transition **sur la séquence de segments**
  (les self-transitions n'existent alors pas par construction), entropie de
  Shannon de la distribution des classes, taux d'entropie markovien d'ordre 1,
  complexité de Lempel-Ziv (LZ76, normalisation c(n)·log_K(n)/n — vérifiée :
  ≈ 1 sur une séquence aléatoire, ≈ 0 sur une séquence périodique).
- **Bords de blocs** : les segments tronqués par un bord d'enregistrement sont
  comptés dans la couverture mais pas dans les durées moyennes.

---

## 8. Validation

### 8.1 Sanity check canonique

Les cartes publiées de Koenig et al. (2002) ne sont pas téléchargeables ici.
On utilise donc des **proxys géométriques** construits à partir des
coordonnées des électrodes (gradients diagonaux pour A et B, gradient
antéro-postérieur pour C, maximum fronto-central pour D).
**C'est un check grossier, pas une vérité terrain** — un |r| de 0.85 avec ces
proxys ne prouve pas qu'on a retrouvé A/B/C/D au sens strict. Le check le plus
informatif reste la comparaison avec les cartes obtenues par Pycrostates
**sur exactement les mêmes données**.

### 8.2 Ablation dense vs conv

Voir §4.3. Métriques comparées : GEV globale après back-fitting, stabilité
split-half des prototypes, corrélation avec le ground truth (synthétique),
corrélation avec les cartes Pycrostates (réel).

### 8.3 Comparaison externe Pycrostates

Modified k-means (`pycrostates.cluster.ModKMeans`) alimenté par **exactement
les mêmes pics de GFP** (mêmes sujets, mêmes indices temporels) et le même K,
en deux temps (individuel puis groupe) comme les tutoriels officiels.

### 8.4 Stabilité

Split-half répété sur les **sujets** (jamais sur les images), appariement
hongrois sur |r|. Deux modes :
- `stability_refit=False` : seul le clustering est refait, l'encodeur reste
  celui entraîné sur toute la cohorte. Rapide, mais **optimiste** : la
  représentation a vu tous les sujets.
- `stability_refit=True` : le VAE est **ré-entraîné** sur chaque moitié. C'est
  la comparaison honnête avec Pycrostates (qui refait tout à chaque fois), et
  c'est le mode utilisé sur le synthétique. Trop coûteux sur la cohorte réelle
  complète (CPU seul).

---

## 9. Points fragiles / à revoir en priorité

1. **λ de l'inversion ridge** calibré sur un seul modèle (§3.2).
2. **Proxys canoniques géométriques** au lieu des cartes publiées (§8.1).
3. **Pas de vrai groupe clinique** : EO/EC n'est qu'un substitut (§1.2).
4. **Bande passante 1–40 Hz** non testée en sensibilité (§2).
5. **VAE dense sur-paramétré** par l'appariement du budget (§4.3) : ajouter un
   troisième bras dense « de taille naturelle ».
6. **Stabilité sans réentraînement** sur les données réelles : optimiste
   pour le VAE, à requalifier si on veut publier ce chiffre (§8.4).
7. **Nombre de pics dépendant de la fréquence d'échantillonnage** (§2) : la
   comparaison inter-datasets des occurrences n'est pas directe.
8. **Choix de K non traité** : tout est fait à K fixé (4 pour le sanity check).
   Les critères usuels (GEV coude, cross-validation, silhouette dans le latent)
   ne sont pas encore implémentés.
