# Classification — synthetic (K = 4)

120 sujets, positif = `G2` (60 sujets). Validation croisee 5 plis x 10 repetitions.

| Methode | AUC | IC 95 % |
|---|---:|---|
| `vae_dense_weighted` | 0.793 | [0.707, 0.874] |
| `pycrostates` | 0.760 | [0.669, 0.841] |
| `ground_truth` | 0.749 | [0.657, 0.833] |
| `vae_conv_two_stage` | 0.746 | [0.659, 0.834] |
| `pca8_modkmeans` | 0.746 | [0.651, 0.829] |
| `vae_conv_weighted` | 0.744 | [0.654, 0.834] |
| `vae_dense_two_stage` | 0.741 | [0.642, 0.831] |
| `vae_token_two_stage` | 0.603 | [0.494, 0.700] |
| `vae_token_weighted` | 0.597 | [0.488, 0.702] |
| `spectral_global` | 0.480 | [0.378, 0.585] |
| `spectral_channel` | 0.460 | [0.355, 0.565] |

## Comparaisons appariees (bootstrap sur les sujets)

| Paire | delta AUC | IC 95 % | p |
|---|---:|---|---:|
| vae_dense_weighted - spectral_channel | +0.333 | [+0.206, +0.458] | 0.000 |
| vae_dense_weighted - spectral_global | +0.312 | [+0.181, +0.444] | 0.000 |
| pycrostates - spectral_channel | +0.300 | [+0.163, +0.431] | 0.001 |
| ground_truth - spectral_channel | +0.289 | [+0.147, +0.423] | 0.000 |
| vae_conv_two_stage - spectral_channel | +0.286 | [+0.150, +0.416] | 0.001 |
| pca8_modkmeans - spectral_channel | +0.286 | [+0.147, +0.418] | 0.002 |
| vae_conv_weighted - spectral_channel | +0.284 | [+0.142, +0.416] | 0.001 |
| vae_dense_two_stage - spectral_channel | +0.281 | [+0.138, +0.421] | 0.001 |
| pycrostates - spectral_global | +0.280 | [+0.146, +0.413] | 0.000 |
| ground_truth - spectral_global | +0.268 | [+0.140, +0.401] | 0.000 |
| vae_conv_two_stage - spectral_global | +0.266 | [+0.137, +0.394] | 0.000 |
| pca8_modkmeans - spectral_global | +0.265 | [+0.126, +0.400] | 0.000 |
| vae_conv_weighted - spectral_global | +0.264 | [+0.131, +0.396] | 0.000 |
| vae_dense_two_stage - spectral_global | +0.261 | [+0.119, +0.395] | 0.000 |
| vae_dense_weighted - vae_token_weighted | +0.196 | [+0.082, +0.311] | 0.000 |
| vae_dense_weighted - vae_token_two_stage | +0.189 | [+0.086, +0.299] | 0.000 |
| vae_token_weighted - pycrostates | -0.163 | [-0.265, -0.058] | 0.001 |
| vae_token_two_stage - pycrostates | -0.157 | [-0.259, -0.057] | 0.000 |
| vae_token_weighted - ground_truth | -0.152 | [-0.253, -0.052] | 0.004 |
| vae_conv_two_stage - vae_token_weighted | +0.149 | [+0.051, +0.254] | 0.003 |
| vae_token_weighted - pca8_modkmeans | -0.149 | [-0.251, -0.047] | 0.004 |
| vae_conv_weighted - vae_token_weighted | +0.147 | [+0.044, +0.256] | 0.005 |
| vae_token_two_stage - ground_truth | -0.145 | [-0.242, -0.050] | 0.000 |
| vae_dense_two_stage - vae_token_weighted | +0.144 | [+0.044, +0.241] | 0.004 |
| vae_token_two_stage - spectral_channel | +0.143 | [+0.003, +0.273] | 0.046 |
| vae_conv_two_stage - vae_token_two_stage | +0.143 | [+0.047, +0.239] | 0.002 |
| vae_token_two_stage - pca8_modkmeans | -0.142 | [-0.241, -0.049] | 0.002 |
| vae_conv_weighted - vae_token_two_stage | +0.141 | [+0.048, +0.236] | 0.002 |
| vae_dense_two_stage - vae_token_two_stage | +0.137 | [+0.041, +0.237] | 0.003 |
| vae_token_weighted - spectral_channel | +0.137 | [-0.015, +0.277] | 0.065 |
| vae_token_two_stage - spectral_global | +0.123 | [-0.024, +0.278] | 0.103 |
| vae_token_weighted - spectral_global | +0.117 | [-0.031, +0.269] | 0.124 |
| vae_dense_two_stage - vae_dense_weighted | -0.052 | [-0.123, +0.011] | 0.106 |
| vae_conv_weighted - vae_dense_weighted | -0.049 | [-0.126, +0.033] | 0.223 |
| vae_dense_weighted - pca8_modkmeans | +0.047 | [-0.022, +0.118] | 0.193 |
| vae_conv_two_stage - vae_dense_weighted | -0.046 | [-0.124, +0.034] | 0.246 |
| vae_dense_weighted - ground_truth | +0.044 | [-0.033, +0.124] | 0.282 |
| vae_dense_weighted - pycrostates | +0.032 | [-0.038, +0.101] | 0.351 |
| spectral_global - spectral_channel | +0.020 | [-0.115, +0.145] | 0.776 |
| vae_dense_two_stage - pycrostates | -0.019 | [-0.092, +0.051] | 0.560 |
| vae_conv_weighted - pycrostates | -0.016 | [-0.061, +0.031] | 0.508 |
| pycrostates - pca8_modkmeans | +0.015 | [-0.003, +0.034] | 0.125 |
| vae_conv_two_stage - pycrostates | -0.014 | [-0.062, +0.034] | 0.592 |
| pycrostates - ground_truth | +0.012 | [-0.030, +0.056] | 0.583 |
| vae_dense_two_stage - ground_truth | -0.008 | [-0.082, +0.066] | 0.805 |
| vae_token_two_stage - vae_token_weighted | +0.006 | [-0.049, +0.060] | 0.808 |
| vae_conv_two_stage - vae_dense_two_stage | +0.006 | [-0.072, +0.090] | 0.858 |
| vae_conv_weighted - ground_truth | -0.005 | [-0.061, +0.054] | 0.875 |
| vae_dense_two_stage - pca8_modkmeans | -0.005 | [-0.077, +0.065] | 0.860 |
| pca8_modkmeans - ground_truth | -0.003 | [-0.046, +0.039] | 0.899 |
| vae_conv_weighted - vae_dense_two_stage | +0.003 | [-0.075, +0.085] | 0.910 |
| vae_conv_two_stage - vae_conv_weighted | +0.003 | [-0.031, +0.033] | 0.861 |
| vae_conv_two_stage - ground_truth | -0.002 | [-0.059, +0.055] | 0.947 |
| vae_conv_weighted - pca8_modkmeans | -0.002 | [-0.046, +0.045] | 0.945 |
| vae_conv_two_stage - pca8_modkmeans | +0.001 | [-0.049, +0.050] | 0.965 |

## Controle par permutation (doit valoir ~0.5)

| Methode | AUC permutee |
|---|---:|
| `vae_conv_two_stage` | 0.515 ± 0.062 |
| `vae_conv_weighted` | 0.505 ± 0.068 |
| `vae_dense_two_stage` | 0.497 ± 0.086 |
| `vae_dense_weighted` | 0.499 ± 0.087 |
| `vae_token_two_stage` | 0.579 ± 0.037 |
| `vae_token_weighted` | 0.555 ± 0.081 |
| `pycrostates` | 0.523 ± 0.057 |
| `pca8_modkmeans` | 0.527 ± 0.050 |
| `ground_truth` | 0.498 ± 0.081 |
| `spectral_global` | 0.558 ± 0.089 |
| `spectral_channel` | 0.520 ± 0.065 |
