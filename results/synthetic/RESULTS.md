# Resultats — synthetic (synthetic)

K=4, latent=8, image=32x32, 20 sujets, 42364 pics (mediane 2118/sujet)
Fidelite topo->image->topo : r = 1.0000
Duree totale : 65 min

## Modeles

| modele | params | val_loss | val_recon | invariance res. | temps |
|--------|--------|----------|-----------|-----------------|-------|
| conv   | 108529 | 0.0538   | 0.0331    | 0.000051        | 685s  |
| dense  | 110960 | 0.0726   | 0.0452    | 0.004242        | 28s   |

## Recherche d'architecture (top 5)

| latent | beta   | width | val_loss | val_recon |
|--------|--------|-------|----------|-----------|
| 8      | 0.001  | 16    | 0.1223   | 0.1000    |
| 8      | 0.01   | 16    | 0.2083   | 0.1190    |
| 16     | 0.01   | 16    | 0.2086   | 0.1089    |
| 16     | 0.001  | 16    | 0.1109   | 0.0783    |
| 8      | 0.0001 | 16    | 0.1009   | 0.0963    |

## Accord entre methodes (correlation absolue moyenne, appariement hongrois)

| paire                            | corr abs moy | par classe          |
|----------------------------------|--------------|---------------------|
| ground_truth|pca8_modkmeans      | 0.983        | 0.98 0.98 0.99 0.99 |
| ground_truth|pycrostates         | 0.985        | 0.98 0.98 0.99 0.99 |
| ground_truth|vae_conv_two_stage  | 0.860        | 0.87 0.87 0.81 0.88 |
| ground_truth|vae_conv_weighted   | 0.893        | 0.88 0.80 0.99 0.90 |
| ground_truth|vae_dense_two_stage | 0.976        | 0.96 0.98 0.99 0.97 |
| ground_truth|vae_dense_weighted  | 0.972        | 0.95 0.98 0.99 0.97 |
| pycrostates|vae_conv_two_stage   | 0.899        | 0.90 0.86 0.93 0.90 |
| pycrostates|vae_conv_weighted    | 0.922        | 0.84 0.99 0.94 0.91 |
| pycrostates|vae_dense_two_stage  | 0.992        | 1.00 1.00 0.99 0.99 |
| pycrostates|vae_dense_weighted   | 0.990        | 1.00 1.00 0.99 0.98 |

## Sanity check canonique (proxys geometriques A/B/C/D)

| methode             | corr abs moy | A B C D             |
|---------------------|--------------|---------------------|
| vae_conv_two_stage  | 0.798        | 0.87 0.76 0.68 0.88 |
| vae_conv_weighted   | 0.790        | 0.85 0.81 0.60 0.90 |
| vae_dense_two_stage | 0.833        | 0.89 0.84 0.68 0.92 |
| vae_dense_weighted  | 0.842        | 0.90 0.85 0.68 0.93 |
| pycrostates         | 0.825        | 0.85 0.86 0.67 0.93 |
| pca8_modkmeans      | 0.827        | 0.85 0.86 0.67 0.93 |
| ground_truth        | 0.746        | 0.76 0.76 0.58 0.89 |

## Back-fitting : parametres microstates

| methode                       | GEV   | duree (ms) | occ/s | H (bits) | LZC   |
|-------------------------------|-------|------------|-------|----------|-------|
| vae_conv_two_stage            | 0.609 | 81.7       | 3.05  | 1.97     | 0.177 |
| vae_conv_weighted             | 0.621 | 80.9       | 3.06  | 1.96     | 0.176 |
| vae_dense_two_stage           | 0.678 | 80.5       | 3.10  | 1.99     | 0.179 |
| vae_dense_weighted            | 0.675 | 80.4       | 3.11  | 1.99     | 0.179 |
| pycrostates                   | 0.688 | 80.8       | 3.10  | 1.99     | 0.178 |
| pca8_modkmeans                | 0.688 | 80.6       | 3.10  | 1.99     | 0.178 |
| ground_truth                  | 0.674 | 81.2       | 3.09  | 1.99     | 0.178 |
| vae_conv_two_stage_latentfit  | 0.561 | 82.1       | 2.96  | 1.91     | 0.171 |
| vae_dense_two_stage_latentfit | 0.655 | 81.8       | 3.04  | 1.97     | 0.176 |

## Stabilite split-half (correlation absolue entre deux moities)

| methode       | corr abs moy | ecart-type | VAE re-entraine |
|---------------|--------------|------------|-----------------|
| vae_conv      | 0.937        | 0.008      | oui             |
| vae_dense     | 0.876        | 0.104      | oui             |
| pycrostates   | 0.992        | 0.002      | non             |
| pca_modkmeans | 0.992        | 0.002      | non             |

## Comparaison de groupes

| methode                       | parametre        | classe la + discriminante | p      | effet | test        |
|-------------------------------|------------------|---------------------------|--------|-------|-------------|
| vae_conv_two_stage            | mean_duration_ms | K3                        | 0.0006 | 0.92  | independant |
| vae_conv_two_stage            | occurrence_per_s | K2                        | 0.0009 | -0.89 | independant |
| vae_conv_two_stage            | coverage         | K2                        | 0.0211 | -0.62 | independant |
| vae_conv_weighted             | mean_duration_ms | K4                        | 0.0008 | 0.90  | independant |
| vae_conv_weighted             | occurrence_per_s | K2                        | 0.0002 | -1.00 | independant |
| vae_conv_weighted             | coverage         | K2                        | 0.0640 | -0.50 | independant |
| vae_dense_two_stage           | mean_duration_ms | K2                        | 0.0003 | 0.96  | independant |
| vae_dense_two_stage           | occurrence_per_s | K4                        | 0.0002 | -1.00 | independant |
| vae_dense_two_stage           | coverage         | K4                        | 0.0008 | -0.90 | independant |
| vae_dense_weighted            | mean_duration_ms | K2                        | 0.0004 | 0.94  | independant |
| vae_dense_weighted            | occurrence_per_s | K4                        | 0.0002 | -1.00 | independant |
| vae_dense_weighted            | coverage         | K4                        | 0.0022 | -0.82 | independant |
| pycrostates                   | mean_duration_ms | K1                        | 0.0003 | 0.96  | independant |
| pycrostates                   | occurrence_per_s | K3                        | 0.0002 | -1.00 | independant |
| pycrostates                   | coverage         | K3                        | 0.0008 | -0.90 | independant |
| pca8_modkmeans                | mean_duration_ms | K4                        | 0.0003 | 0.96  | independant |
| pca8_modkmeans                | occurrence_per_s | K3                        | 0.0002 | -1.00 | independant |
| pca8_modkmeans                | coverage         | K3                        | 0.0008 | -0.90 | independant |
| ground_truth                  | mean_duration_ms | K2                        | 0.0002 | 1.00  | independant |
| ground_truth                  | occurrence_per_s | K4                        | 0.0002 | -1.00 | independant |
| ground_truth                  | coverage         | K4                        | 0.0004 | -0.94 | independant |
| vae_conv_two_stage_latentfit  | mean_duration_ms | K4                        | 0.0091 | 0.70  | independant |
| vae_conv_two_stage_latentfit  | occurrence_per_s | K2                        | 0.0010 | -0.88 | independant |
| vae_conv_two_stage_latentfit  | coverage         | K2                        | 0.0539 | -0.52 | independant |
| vae_dense_two_stage_latentfit | mean_duration_ms | K3                        | 0.0010 | 0.88  | independant |
| vae_dense_two_stage_latentfit | occurrence_per_s | K4                        | 0.0002 | -1.00 | independant |
| vae_dense_two_stage_latentfit | coverage         | K4                        | 0.0058 | -0.74 | independant |
