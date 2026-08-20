# Classification — synthetic (K = 4)

120 sujets, positif = `G2` (60 sujets). Validation croisee 5 plis x 10 repetitions.

| Methode | AUC | IC 95 % |
|---|---:|---|
| `vae_token_weighted` | 0.848 | [0.770, 0.914] |
| `vae_token_two_stage` | 0.842 | [0.765, 0.907] |
| `vae_dense_two_stage` | 0.764 | [0.677, 0.843] |
| `pycrostates` | 0.760 | [0.669, 0.841] |
| `ground_truth` | 0.749 | [0.657, 0.833] |
| `pca8_modkmeans` | 0.746 | [0.651, 0.829] |
| `vae_dense_weighted` | 0.745 | [0.655, 0.828] |
| `vade_dense` | 0.733 | [0.641, 0.825] |
| `vae_conv_two_stage` | 0.715 | [0.624, 0.805] |
| `vae_conv_weighted` | 0.679 | [0.581, 0.776] |
| `spectral_global` | 0.480 | [0.378, 0.585] |
| `spectral_channel` | 0.460 | [0.355, 0.565] |

## Comparaisons appariees (bootstrap sur les sujets)

| Paire | delta AUC | IC 95 % | p |
|---|---:|---|---:|
| vae_token_weighted - spectral_channel | +0.388 | [+0.250, +0.518] | 0.000 |
| vae_token_two_stage - spectral_channel | +0.382 | [+0.250, +0.507] | 0.000 |
| vae_token_weighted - spectral_global | +0.368 | [+0.248, +0.489] | 0.000 |
| vae_token_two_stage - spectral_global | +0.362 | [+0.238, +0.487] | 0.000 |
| vae_dense_two_stage - spectral_channel | +0.304 | [+0.169, +0.428] | 0.000 |
| pycrostates - spectral_channel | +0.300 | [+0.163, +0.431] | 0.001 |
| ground_truth - spectral_channel | +0.289 | [+0.147, +0.423] | 0.000 |
| pca8_modkmeans - spectral_channel | +0.286 | [+0.147, +0.418] | 0.002 |
| vae_dense_weighted - spectral_channel | +0.285 | [+0.148, +0.411] | 0.000 |
| vae_dense_two_stage - spectral_global | +0.283 | [+0.146, +0.420] | 0.000 |
| pycrostates - spectral_global | +0.280 | [+0.146, +0.413] | 0.000 |
| vade_dense - spectral_channel | +0.273 | [+0.140, +0.402] | 0.000 |
| ground_truth - spectral_global | +0.268 | [+0.140, +0.401] | 0.000 |
| pca8_modkmeans - spectral_global | +0.265 | [+0.126, +0.400] | 0.000 |
| vae_dense_weighted - spectral_global | +0.265 | [+0.130, +0.396] | 0.000 |
| vae_conv_two_stage - spectral_channel | +0.255 | [+0.119, +0.384] | 0.001 |
| vade_dense - spectral_global | +0.253 | [+0.125, +0.388] | 0.000 |
| vae_conv_two_stage - spectral_global | +0.234 | [+0.104, +0.367] | 0.000 |
| vae_conv_weighted - spectral_channel | +0.219 | [+0.084, +0.356] | 0.003 |
| vae_conv_weighted - spectral_global | +0.199 | [+0.053, +0.354] | 0.010 |
| vae_conv_weighted - vae_token_weighted | -0.169 | [-0.276, -0.058] | 0.004 |
| vae_conv_weighted - vae_token_two_stage | -0.162 | [-0.258, -0.069] | 0.002 |
| vae_conv_two_stage - vae_token_weighted | -0.133 | [-0.209, -0.062] | 0.000 |
| vae_conv_two_stage - vae_token_two_stage | -0.127 | [-0.193, -0.061] | 0.000 |
| vae_token_weighted - vade_dense | +0.115 | [+0.033, +0.201] | 0.006 |
| vae_token_two_stage - vade_dense | +0.109 | [+0.033, +0.188] | 0.004 |
| vae_dense_weighted - vae_token_weighted | -0.103 | [-0.176, -0.036] | 0.001 |
| vae_token_weighted - pca8_modkmeans | +0.103 | [+0.040, +0.170] | 0.000 |
| vae_token_weighted - ground_truth | +0.099 | [+0.040, +0.161] | 0.000 |
| vae_dense_weighted - vae_token_two_stage | -0.097 | [-0.165, -0.035] | 0.002 |
| vae_token_two_stage - pca8_modkmeans | +0.096 | [+0.036, +0.162] | 0.000 |
| vae_token_two_stage - ground_truth | +0.093 | [+0.035, +0.157] | 0.001 |
| vae_token_weighted - pycrostates | +0.088 | [+0.027, +0.154] | 0.005 |
| vae_dense_two_stage - vae_token_weighted | -0.084 | [-0.149, -0.025] | 0.003 |
| vae_conv_weighted - vae_dense_two_stage | -0.084 | [-0.183, +0.016] | 0.097 |
| vae_token_two_stage - pycrostates | +0.082 | [+0.019, +0.146] | 0.005 |
| vae_conv_weighted - pycrostates | -0.081 | [-0.175, +0.017] | 0.102 |
| vae_dense_two_stage - vae_token_two_stage | -0.078 | [-0.142, -0.020] | 0.008 |
| vae_conv_weighted - ground_truth | -0.069 | [-0.172, +0.038] | 0.196 |
| vae_conv_weighted - pca8_modkmeans | -0.066 | [-0.163, +0.035] | 0.184 |
| vae_conv_weighted - vae_dense_weighted | -0.066 | [-0.161, +0.031] | 0.177 |
| vae_conv_weighted - vade_dense | -0.054 | [-0.163, +0.057] | 0.304 |
| vae_conv_two_stage - vae_dense_two_stage | -0.049 | [-0.118, +0.023] | 0.182 |
| vae_conv_two_stage - pycrostates | -0.046 | [-0.110, +0.022] | 0.183 |
| vae_conv_two_stage - vae_conv_weighted | +0.035 | [-0.060, +0.130] | 0.448 |
| vae_conv_two_stage - ground_truth | -0.034 | [-0.101, +0.035] | 0.338 |
| vae_conv_two_stage - pca8_modkmeans | -0.031 | [-0.096, +0.035] | 0.345 |
| vae_conv_two_stage - vae_dense_weighted | -0.031 | [-0.096, +0.038] | 0.396 |
| vae_dense_two_stage - vade_dense | +0.031 | [-0.061, +0.120] | 0.504 |
| pycrostates - vade_dense | +0.027 | [-0.057, +0.108] | 0.532 |
| spectral_global - spectral_channel | +0.020 | [-0.115, +0.145] | 0.776 |
| vae_conv_two_stage - vade_dense | -0.018 | [-0.110, +0.073] | 0.689 |
| vae_dense_two_stage - vae_dense_weighted | +0.018 | [-0.017, +0.055] | 0.295 |
| vae_dense_two_stage - pca8_modkmeans | +0.018 | [-0.024, +0.063] | 0.431 |
| ground_truth - vade_dense | +0.016 | [-0.069, +0.097] | 0.741 |
| vae_dense_two_stage - ground_truth | +0.015 | [-0.030, +0.059] | 0.539 |
| vae_dense_weighted - pycrostates | -0.015 | [-0.064, +0.036] | 0.546 |
| pycrostates - pca8_modkmeans | +0.015 | [-0.003, +0.034] | 0.125 |
| pca8_modkmeans - vade_dense | +0.012 | [-0.075, +0.096] | 0.774 |
| vae_dense_weighted - vade_dense | +0.012 | [-0.079, +0.102] | 0.808 |
| pycrostates - ground_truth | +0.012 | [-0.030, +0.056] | 0.583 |
| vae_token_two_stage - vae_token_weighted | -0.006 | [-0.053, +0.038] | 0.797 |
| vae_dense_two_stage - pycrostates | +0.003 | [-0.039, +0.048] | 0.907 |
| vae_dense_weighted - ground_truth | -0.003 | [-0.056, +0.046] | 0.903 |
| pca8_modkmeans - ground_truth | -0.003 | [-0.046, +0.039] | 0.899 |
| vae_dense_weighted - pca8_modkmeans | -0.000 | [-0.053, +0.051] | 0.982 |

## Controle par permutation (doit valoir ~0.5)

| Methode | AUC permutee |
|---|---:|
| `vae_conv_two_stage` | 0.478 ± 0.088 |
| `vae_conv_weighted` | 0.478 ± 0.072 |
| `vae_dense_two_stage` | 0.482 ± 0.070 |
| `vae_dense_weighted` | 0.466 ± 0.071 |
| `vae_token_two_stage` | 0.499 ± 0.059 |
| `vae_token_weighted` | 0.481 ± 0.063 |
| `pycrostates` | 0.482 ± 0.075 |
| `pca8_modkmeans` | 0.482 ± 0.074 |
| `ground_truth` | 0.474 ± 0.068 |
| `vade_dense` | 0.485 ± 0.077 |
| `spectral_global` | 0.485 ± 0.092 |
| `spectral_channel` | 0.511 ± 0.071 |
