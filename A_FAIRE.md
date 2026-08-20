# Ce qu'il reste à faire

Écrit le 20/08/2026. Ordre recommandé, avec les commandes prêtes à lancer.

Total : environ **20 h de calcul**, soit deux nuits.

---

## Où on en est

**Acquis, sur 3 graines** — le `TokenVAE` devance toutes les méthodes, avec un
signe positif sur les trois graines :

| contre | graine 0 | graine 1 | graine 2 | moyenne |
|---|---:|---:|---:|---:|
| `pycrostates` | +0.082 | +0.025 | +0.063 | **+0.057** |
| `pca8_modkmeans` | +0.096 | +0.027 | +0.062 | **+0.062** |
| `vae_conv` | +0.114 | +0.084 | +0.049 | +0.082 |
| `vae_dense` | +0.112 | +0.071 | +0.051 | +0.078 |
| `spectral_global` | +0.362 | +0.343 | +0.379 | +0.361 |

Contre les cartes vraies, le signe change selon la graine : le token les égale
sans les dépasser.

**Le fait le plus intéressant** : le token obtient ces AUC avec une GEV
*systématiquement plus basse* que pycrostates (0.613 / 0.630 / 0.649 contre
0.697 / 0.690 / 0.692). La métrique historique du domaine l'aurait classé
dernier alors qu'il est le meilleur sur le seul critère qui compte cliniquement.

---

## 1. Les 2 graines restantes — 9 h

Le pilote saute les 3 déjà faites.

```bash
python scripts/sweep.py --dataset synthetic --effect transition --effect-t 0.35 \
    --n-subjects 120 --seeds 0 1 2 3 4 --k 4 --epochs 600 --jobs 2 \
    --out results/sweep_seeds_t035
```

> Une cellule consomme ~4 cœurs à elle seule (numpy parallélise le
> back-fitting). Ne pas lancer autre chose de lourd en même temps : trois
> cellules plus un autre travail saturent les 16 cœurs et tout ralentit.

---

## 2. Le test des taux d'apprentissage — 2 h

**La principale objection contre le résultat.** Le bras token tourne à
`lr = 5e-4`, les autres à `2e-3`. C'est le contournement d'un plateau qui lui
est propre, mais personne n'a vérifié que conv et dense ne feraient pas mieux au
même taux. Le test tourne dans une copie du run, donc les résultats actuels
restent intacts.

```bash
python scripts/retrain_arm_in_run.py --run results/synth_t035_lr5e4 --arm conv  --lr 5e-4 --epochs 600 --score-every 25 --force
python scripts/retrain_arm_in_run.py --run results/synth_t035_lr5e4 --arm dense --lr 5e-4 --epochs 600 --score-every 25 --force
python scripts/run_classification.py --run results/synth_t035_lr5e4 --n-repeats 10
```

> **Piège** : choisir le taux qui rend le token meilleur, ce serait régler un
> hyperparamètre sur la métrique de comparaison. Fixer les taux sur un critère
> qui n'est PAS l'AUC — par exemple « le taux auquel le bras converge sans
> plateau » — puis ne plus y revenir.

---

## 3. Le contrôle sans la GEV dans les features — 40 min

La GEV par classe fait partie des 31 descripteurs, et c'est une mesure de
qualité d'ajustement. Si le token doit une partie de son avance au fait qu'il
ajuste différemment les deux groupes, ce n'est pas un résultat sur les
microstates. C'est l'objection qu'un relecteur trouve en cinq minutes.

```bash
python scripts/run_classification.py --run results/synth_t035_transition \
    --no-gev --n-repeats 10 --out results/synth_t035_transition/classification_nogev
```

---

## 4. La stabilité prédit-elle le pouvoir discriminant ? — 1 h

Détermine **comment régler les hyperparamètres sans tricher**.

L'idée : un critère qui ne regarde jamais les étiquettes cliniques ne contamine
pas la comparaison, même appliqué aux données cibles. Deux candidats — la
stabilité split-half et la convergence de la reconstruction. Reste à vérifier
qu'ils prédisent le pouvoir discriminant.

À mesurer sur le synthétique, où l'on dispose des deux : les configurations les
plus stables sont-elles les plus discriminantes ?

- **Si oui** : on règle sur ds004504 lui-même, sans sacrifier un seul sujet, et
  le jeu clinique reste un test pur.
- **Si non** : il faut une base externe à 19 électrodes pour le réglage.

> **Piège** : un modèle mort est parfaitement stable. Le bras token bloqué
> affichait 1.000 ± 0.000. La stabilité ne s'utilise jamais seule, toujours
> avec une exigence de convergence.

---

## 5. Cohorte synthétique en 19 canaux — 3 h

Le pont entre le simulé et le clinique : mêmes 19 électrodes que ds004504, mais
avec un ground truth. Si le token gagne encore là, on sait *avant* de toucher au
réel que le résultat devrait transférer — et s'il échoue, on sait pourquoi, ce
que le réel ne dira jamais.

L'outillage existe : `montage="ds004504_19"` dans `ExperimentConfig`.

> Sur 19 électrodes, le rendu image compte 39 pixels par électrode contre 8 en
> 64 canaux. L'interpolation y est cinq fois plus lourde — c'est la condition
> réelle du jeu clinique, et c'est là que le bras conv devrait le plus souffrir.

---

## 6. VaDE : la sélection de K — 3 h

**Le seul résultat positif encore inaccessible aux autres méthodes.**
Pycrostates et l'ACP reçoivent K en entrée ; VaDE peut le trouver. Sur le
synthétique la vérité est K = 4, donc la réponse est vérifiable.

```bash
python scripts/sweep.py --dataset synthetic --effect transition --effect-t 0.35 \
    --n-subjects 120 --seeds 0 1 2 3 4 5 6 7 8 9 --k 2 3 4 5 6 7 \
    --arms vade --vade-lambda-balance 0 --jobs 4 --out results/sweep_k_synth
```

> **`--vade-lambda-balance 0` est indispensable ici.** Le garde-fou
> anti-effondrement maintient toutes les composantes en vie ; or la mort d'une
> composante est précisément le signal qu'on cherche quand on cherche K. Le
> laisser actif fabriquerait le biais de surestimation qu'on veut mesurer.
> Le garder à 0.5 pour un run à K fixé.

---

## 7. Tout refaire sur ds004504, les trois contrastes — 4 h

En dernier, quand le protocole ne bouge plus. Le banc réel actuel est périmé :
ses bras conv et dense n'ont eu que 30 époques et son bras token n'avait pas
appris. Ré-entraîner les trois bras à convergence, puis le banc, **avec
plusieurs graines** — la cohorte est fixe mais l'initialisation non.

Les trois comparaisons deux à deux, qui ne se valent pas :

| Contraste | n | Intérêt |
|---|---:|---|
| AD vs CTR | 65 | facile, dominé par le ralentissement global |
| **AD vs FTD** | 59 | **difficile et cliniquement utile** — les deux groupes sont ralentis, le spectral y a peu de prise |
| FTD vs CTR | 52 | intermédiaire |

Le second est celui qui vaut l'article : le diagnostic différentiel entre ces
deux démences est réellement difficile, et c'est le seul contraste où les
microstates ont une chance d'apporter quelque chose que la puissance par bande
n'apporte pas.

Montrer AD/CTR **à côté** renforce le propos plutôt que de l'affaiblir : un cas
où la méthode n'apporte rien rend beaucoup plus crédible le cas où elle apporte.

*EEGBCI est écarté* : yeux ouverts contre yeux fermés est une manipulation
physiologique, pas une question clinique. L'effet est trivialement porté par
l'alpha et personne n'a besoin d'un classifieur pour ça.

### Le verrou d'effectif, et ce que les graines ne peuvent pas faire

Deux sources de bruit, une seule réductible :

- **le bruit d'entraînement** (initialisation) — moyenner sur les graines le
  réduit ;
- **le bruit d'échantillonnage des sujets** — le bootstrap l'ESTIME, il ne le
  réduit pas. Avec une cohorte figée, il est irréductible.

Le bootstrap apparié aide beaucoup, parce qu'une grande part de la variabilité
sujet s'annule quand deux méthodes sont comparées sur les mêmes gens : la
demi-largeur passe de ~0.09 pour une AUC isolée à ~0.06–0.09 pour une
différence appariée.

Conséquence : graines et bootstrap ensemble établissent solidement **« sur
cette cohorte, A bat B »**. Ils n'établiront jamais **« dans la population, A
bat B »** — ça reste borné par le nombre de sujets. Le classement fin des
méthodes doit donc venir du synthétique, où le nombre de sujets est un
paramètre libre.

---

## 8. ~~Mettre `HYPOTHESES.md` à jour~~ — FAIT le 20/08

Le document de passation est en retard sur trois points importants :

- **H2 est confirmée** avec l'attribution d'origine au bras conv. J'avais
  corrigé en affirmant l'inverse, sur la foi d'un run de 60 époques trop court.
  Mesure à 600 époques : conv −0.072, token −0.044, dense −0.006.
- **H4 n'explique pas H2.** Avec `loss_space=topo`, donc un objectif identique à
  celui du dense, le conv se dégrade quand même. C'est l'encodeur lui-même.
- **Le résultat du token n'y figure pas.**

---

## Défauts connus, non corrigés

**La GEV sert encore de critère** à deux endroits — la recherche d'architecture
et `--select-epoch-by-score`. Or les cartes vraies y sont battues par trois
méthodes sur quatre : elle ne mesure pas ce qu'on croit. Légitime comme
descriptif, illégitime comme critère de choix.

**P10 — le jeu de validation du modèle final est inclus dans son jeu
d'entraînement.** L'early stopping ne peut donc pas détecter de
sur-apprentissage. L'impact est réduit depuis qu'on entraîne sans early stopping
jusqu'à convergence, mais `best_val` reste rapporté comme s'il était
hors-échantillon.

**Asymétrie des hyperparamètres** — `latent_dim` et β ont été choisis sur le
bras conv puis imposés au dense et au token. Rien n'a jamais été réglé pour le
token ni pour VaDE.

**Aucun test ne couvre le pipeline de bout en bout**, ce qui a laissé passer un
`--vade` qui plantait.

---

## Deux règles à ne pas oublier

**Le plancher de bruit vaut 0.03 à 0.065 de GEV entre graines.** Aucun écart
inférieur à ~0.07 ne doit être interprété sur une seule graine.

**Une stabilité split-half parfaite n'est pas un bon signe.** Un modèle qui n'a
rien appris produit des cartes rigoureusement identiques d'une moitié de cohorte
à l'autre. Toujours la lire à côté d'une mesure de qualité.

---

## Pistes non retenues pour ce projet

Notées pour qu'elles ne se reperdent pas, sans rang ni priorité.

### La discrétisation en K états fait-elle perdre de l'information ?

Le pipeline clusterise dans le latent, décode les centroïdes en cartes, puis
n'utilise plus que ces cartes. Le latent est abandonné en route.

Or les paramètres de microstates **sont déjà une agrégation du latent** : la
couverture est l'histogramme des affectations, la durée et l'occurrence
ajoutent l'ordre temporel que l'histogramme perd. Ce n'est donc pas une
agrégation arbitraire, c'est celle qui garde la séquence.

La seule chose qu'elle jette, c'est la **forme de la distribution** latente. Ce
qui donne une question précise : cette forme porte-t-elle de l'information de
groupe que les états discrets ne portent pas ?

**La version la moins chère ne demande pas de nouveau bras** : concaténer les
31 features de microstates avec 16 valeurs de latent agrégé (moyenne et
variance par dimension, pour un latent de 8) et regarder si l'AUC bouge. Si
elle ne bouge pas, la discrétisation ne perd rien.

Cette formulation évite aussi le problème de comparabilité : on ne prétend pas
comparer un bras latent à `pycrostates`, qui n'a aucun latent — on mesure juste
si un ajout améliore un bras existant.

*Coût : encoder avec les modèles déjà entraînés, agréger, relancer un banc.
Environ une heure de code, quarante minutes de calcul.*


---

## Perspective : CAUEEG

Jeu clinique de dementia nettement plus grand que ds004504 (de l'ordre du
millier de sujets, à vérifier), avec du MCI — le contraste à la fois le plus
difficile et le plus utile. Accessible plus tard.

Ce qu'il changerait n'est pas méthodologique mais **d'effectif**, et c'est
décisif. À 59 ou 65 sujets, l'écart d'AUC minimal détectable est de 0.07–0.08 :
on peut affirmer « les microstates fonctionnent », jamais « cette méthode est
meilleure que celle-là ». Le seuil tombe d'un facteur trois ou quatre sur un
millier de sujets.

**Le classement des méthodes deviendrait mesurable sur du clinique réel**, ce
qui est aujourd'hui hors d'atteinte et oblige à faire porter tout le classement
par la cohorte simulée.

Ce serait aussi le bon jeu pour la sélection de K : trouver K sur mille sujets
a un tout autre poids que sur soixante.
