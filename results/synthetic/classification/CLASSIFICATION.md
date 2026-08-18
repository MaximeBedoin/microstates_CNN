# Classification — synthetic (K = 4)

20 sujets, positif = `G2` (10 sujets). Validation croisee 5 plis x 10 repetitions.

| Methode | AUC | IC 95 % |
|---|---:|---|
| `ground_truth` | 1.000 | [1.000, 1.000] |
| `spectral_global` | 1.000 | [1.000, 1.000] |
| `spectral_channel` | 1.000 | [1.000, 1.000] |
| `vae_dense_two_stage` | 0.990 | [0.939, 1.000] |
| `vae_dense_weighted` | 0.990 | [0.939, 1.000] |
| `pycrostates` | 0.990 | [0.939, 1.000] |
| `pca8_modkmeans` | 0.980 | [0.910, 1.000] |
| `vae_conv_two_stage` | 0.970 | [0.889, 1.000] |
| `vae_conv_weighted` | 0.940 | [0.790, 1.000] |

## Comparaisons appariees (bootstrap sur les sujets)

| Paire | delta AUC | IC 95 % | p |
|---|---:|---|---:|
| vae_conv_weighted - ground_truth | -0.060 | [-0.210, +0.000] | 0.411 |
| vae_conv_weighted - spectral_global | -0.060 | [-0.210, +0.000] | 0.415 |
| vae_conv_weighted - spectral_channel | -0.060 | [-0.210, +0.000] | 0.412 |
| vae_conv_weighted - vae_dense_two_stage | -0.050 | [-0.188, +0.000] | 0.486 |
| vae_conv_weighted - vae_dense_weighted | -0.050 | [-0.188, +0.000] | 0.483 |
| vae_conv_weighted - pycrostates | -0.050 | [-0.188, +0.000] | 0.481 |
| vae_conv_weighted - pca8_modkmeans | -0.040 | [-0.162, +0.010] | 0.542 |
| vae_conv_two_stage - vae_conv_weighted | +0.030 | [-0.060, +0.160] | 0.701 |
| vae_conv_two_stage - ground_truth | -0.030 | [-0.111, +0.000] | 0.588 |
| vae_conv_two_stage - spectral_global | -0.030 | [-0.111, +0.000] | 0.589 |
| vae_conv_two_stage - spectral_channel | -0.030 | [-0.111, +0.000] | 0.588 |
| vae_conv_two_stage - vae_dense_two_stage | -0.020 | [-0.095, +0.036] | 0.807 |
| vae_conv_two_stage - vae_dense_weighted | -0.020 | [-0.095, +0.036] | 0.815 |
| vae_conv_two_stage - pycrostates | -0.020 | [-0.095, +0.036] | 0.817 |
| pca8_modkmeans - ground_truth | -0.020 | [-0.090, +0.000] | 0.843 |
| pca8_modkmeans - spectral_global | -0.020 | [-0.090, +0.000] | 0.848 |
| pca8_modkmeans - spectral_channel | -0.020 | [-0.090, +0.000] | 0.842 |
| vae_conv_two_stage - pca8_modkmeans | -0.010 | [-0.080, +0.048] | 1.079 |
| vae_dense_two_stage - ground_truth | -0.010 | [-0.061, +0.000] | 1.139 |
| vae_dense_two_stage - spectral_global | -0.010 | [-0.061, +0.000] | 1.139 |
| vae_dense_two_stage - spectral_channel | -0.010 | [-0.061, +0.000] | 1.134 |
| vae_dense_weighted - ground_truth | -0.010 | [-0.061, +0.000] | 1.131 |
| vae_dense_weighted - spectral_global | -0.010 | [-0.061, +0.000] | 1.132 |
| vae_dense_weighted - spectral_channel | -0.010 | [-0.061, +0.000] | 1.128 |
| pycrostates - ground_truth | -0.010 | [-0.061, +0.000] | 1.134 |
| pycrostates - spectral_global | -0.010 | [-0.061, +0.000] | 1.136 |
| pycrostates - spectral_channel | -0.010 | [-0.061, +0.000] | 1.137 |
| vae_dense_two_stage - pca8_modkmeans | +0.010 | [-0.000, +0.053] | 1.115 |
| vae_dense_weighted - pca8_modkmeans | +0.010 | [-0.000, +0.053] | 1.130 |
| pycrostates - pca8_modkmeans | +0.010 | [-0.000, +0.053] | 1.135 |
| vae_dense_two_stage - vae_dense_weighted | +0.000 | [-0.000, +0.000] | 1.833 |
| vae_dense_two_stage - pycrostates | +0.000 | [-0.000, +0.000] | 1.832 |
| vae_dense_weighted - pycrostates | +0.000 | [-0.000, +0.000] | 1.855 |
| ground_truth - spectral_global | +0.000 | [-0.000, +0.000] | 1.934 |
| ground_truth - spectral_channel | +0.000 | [-0.000, +0.000] | 1.949 |
| spectral_global - spectral_channel | +0.000 | [-0.000, +0.000] | 1.938 |

## Controle par permutation (doit valoir ~0.5)

| Methode | AUC permutee |
|---|---:|
| `vae_conv_two_stage` | 0.470 ± 0.089 |
| `vae_conv_weighted` | 0.442 ± 0.154 |
| `vae_dense_two_stage` | 0.544 ± 0.122 |
| `vae_dense_weighted` | 0.532 ± 0.135 |
| `pycrostates` | 0.552 ± 0.166 |
| `pca8_modkmeans` | 0.532 ± 0.169 |
| `ground_truth` | 0.582 ± 0.159 |
| `spectral_global` | 0.618 ± 0.172 |
| `spectral_channel` | 0.514 ± 0.145 |
