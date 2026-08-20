# Classification — synthetic (K = 4)

120 sujets, positif = `G2` (60 sujets). Validation croisee 5 plis x 10 repetitions.

| Methode | AUC | IC 95 % |
|---|---:|---|
| `vae_dense_weighted` | 0.793 | [0.707, 0.874] |
| `vae_token_two_stage` | 0.763 | [0.670, 0.849] |
| `pycrostates` | 0.760 | [0.669, 0.841] |
| `vae_token_weighted` | 0.750 | [0.660, 0.836] |
| `ground_truth` | 0.749 | [0.657, 0.833] |
| `vae_conv_two_stage` | 0.746 | [0.659, 0.834] |
| `pca8_modkmeans` | 0.746 | [0.651, 0.829] |
| `vae_conv_weighted` | 0.744 | [0.654, 0.834] |
| `vae_dense_two_stage` | 0.741 | [0.642, 0.831] |
| `vade_dense` | 0.654 | [0.552, 0.749] |
| `spectral_global` | 0.480 | [0.378, 0.585] |
| `spectral_channel` | 0.460 | [0.355, 0.565] |

## Comparaisons appariees (bootstrap sur les sujets)

| Paire | delta AUC | IC 95 % | p |
|---|---:|---|---:|
| vae_dense_weighted - spectral_channel | +0.333 | [+0.206, +0.458] | 0.000 |
| vae_dense_weighted - spectral_global | +0.312 | [+0.181, +0.444] | 0.000 |
| vae_token_two_stage - spectral_channel | +0.303 | [+0.167, +0.429] | 0.000 |
| pycrostates - spectral_channel | +0.300 | [+0.163, +0.431] | 0.001 |
| vae_token_weighted - spectral_channel | +0.290 | [+0.146, +0.418] | 0.001 |
| ground_truth - spectral_channel | +0.289 | [+0.147, +0.423] | 0.000 |
| vae_conv_two_stage - spectral_channel | +0.286 | [+0.150, +0.416] | 0.001 |
| pca8_modkmeans - spectral_channel | +0.286 | [+0.147, +0.418] | 0.002 |
| vae_conv_weighted - spectral_channel | +0.284 | [+0.142, +0.416] | 0.001 |
| vae_token_two_stage - spectral_global | +0.282 | [+0.148, +0.415] | 0.000 |
| vae_dense_two_stage - spectral_channel | +0.281 | [+0.138, +0.421] | 0.001 |
| pycrostates - spectral_global | +0.280 | [+0.146, +0.413] | 0.000 |
| vae_token_weighted - spectral_global | +0.270 | [+0.136, +0.405] | 0.000 |
| ground_truth - spectral_global | +0.268 | [+0.140, +0.401] | 0.000 |
| vae_conv_two_stage - spectral_global | +0.266 | [+0.137, +0.394] | 0.000 |
| pca8_modkmeans - spectral_global | +0.265 | [+0.126, +0.400] | 0.000 |
| vae_conv_weighted - spectral_global | +0.264 | [+0.131, +0.396] | 0.000 |
| vae_dense_two_stage - spectral_global | +0.261 | [+0.119, +0.395] | 0.000 |
| vade_dense - spectral_channel | +0.194 | [+0.066, +0.315] | 0.002 |
| vade_dense - spectral_global | +0.174 | [+0.041, +0.309] | 0.012 |
| vae_dense_weighted - vade_dense | +0.138 | [+0.041, +0.236] | 0.006 |
| vae_token_two_stage - vade_dense | +0.108 | [+0.014, +0.204] | 0.023 |
| pycrostates - vade_dense | +0.106 | [+0.013, +0.205] | 0.026 |
| vae_token_weighted - vade_dense | +0.096 | [+0.003, +0.191] | 0.044 |
| ground_truth - vade_dense | +0.094 | [-0.002, +0.190] | 0.054 |
| vae_conv_two_stage - vade_dense | +0.092 | [-0.003, +0.192] | 0.058 |
| pca8_modkmeans - vade_dense | +0.091 | [-0.003, +0.192] | 0.055 |
| vae_conv_weighted - vade_dense | +0.089 | [-0.008, +0.188] | 0.062 |
| vae_dense_two_stage - vade_dense | +0.086 | [-0.008, +0.180] | 0.065 |
| vae_dense_two_stage - vae_dense_weighted | -0.052 | [-0.123, +0.011] | 0.106 |
| vae_conv_weighted - vae_dense_weighted | -0.049 | [-0.126, +0.033] | 0.223 |
| vae_dense_weighted - pca8_modkmeans | +0.047 | [-0.022, +0.118] | 0.193 |
| vae_conv_two_stage - vae_dense_weighted | -0.046 | [-0.124, +0.034] | 0.246 |
| vae_dense_weighted - ground_truth | +0.044 | [-0.033, +0.124] | 0.282 |
| vae_dense_weighted - vae_token_weighted | +0.043 | [-0.040, +0.123] | 0.304 |
| vae_dense_weighted - pycrostates | +0.032 | [-0.038, +0.101] | 0.351 |
| vae_dense_weighted - vae_token_two_stage | +0.030 | [-0.050, +0.111] | 0.456 |
| vae_dense_two_stage - vae_token_two_stage | -0.022 | [-0.100, +0.049] | 0.530 |
| spectral_global - spectral_channel | +0.020 | [-0.115, +0.145] | 0.776 |
| vae_dense_two_stage - pycrostates | -0.019 | [-0.092, +0.051] | 0.560 |
| vae_conv_weighted - vae_token_two_stage | -0.019 | [-0.069, +0.035] | 0.494 |
| vae_token_two_stage - pca8_modkmeans | +0.017 | [-0.037, +0.070] | 0.552 |
| vae_conv_weighted - pycrostates | -0.016 | [-0.061, +0.031] | 0.508 |
| vae_conv_two_stage - vae_token_two_stage | -0.016 | [-0.071, +0.039] | 0.583 |
| pycrostates - pca8_modkmeans | +0.015 | [-0.003, +0.034] | 0.125 |
| vae_token_two_stage - ground_truth | +0.014 | [-0.045, +0.075] | 0.663 |
| vae_conv_two_stage - pycrostates | -0.014 | [-0.062, +0.034] | 0.592 |
| vae_token_two_stage - vae_token_weighted | +0.013 | [-0.021, +0.047] | 0.469 |
| pycrostates - ground_truth | +0.012 | [-0.030, +0.056] | 0.583 |
| vae_token_weighted - pycrostates | -0.010 | [-0.071, +0.049] | 0.739 |
| vae_dense_two_stage - vae_token_weighted | -0.009 | [-0.084, +0.064] | 0.778 |
| vae_dense_two_stage - ground_truth | -0.008 | [-0.082, +0.066] | 0.805 |
| vae_conv_weighted - vae_token_weighted | -0.006 | [-0.060, +0.050] | 0.839 |
| vae_conv_two_stage - vae_dense_two_stage | +0.006 | [-0.072, +0.090] | 0.858 |
| vae_conv_weighted - ground_truth | -0.005 | [-0.061, +0.054] | 0.875 |
| vae_dense_two_stage - pca8_modkmeans | -0.005 | [-0.077, +0.065] | 0.860 |
| vae_token_weighted - pca8_modkmeans | +0.004 | [-0.054, +0.062] | 0.886 |
| vae_conv_two_stage - vae_token_weighted | -0.004 | [-0.066, +0.056] | 0.930 |
| pca8_modkmeans - ground_truth | -0.003 | [-0.046, +0.039] | 0.899 |
| vae_conv_weighted - vae_dense_two_stage | +0.003 | [-0.075, +0.085] | 0.910 |
| vae_conv_two_stage - vae_conv_weighted | +0.003 | [-0.031, +0.033] | 0.861 |
| vae_token_two_stage - pycrostates | +0.002 | [-0.052, +0.057] | 0.953 |
| vae_conv_two_stage - ground_truth | -0.002 | [-0.059, +0.055] | 0.947 |
| vae_conv_weighted - pca8_modkmeans | -0.002 | [-0.046, +0.045] | 0.945 |
| vae_token_weighted - ground_truth | +0.001 | [-0.063, +0.063] | 0.962 |
| vae_conv_two_stage - pca8_modkmeans | +0.001 | [-0.049, +0.050] | 0.965 |

## Controle par permutation (doit valoir ~0.5)

| Methode | AUC permutee |
|---|---:|
| `vae_conv_two_stage` | 0.484 ± 0.080 |
| `vae_conv_weighted` | 0.484 ± 0.068 |
| `vae_dense_two_stage` | 0.500 ± 0.069 |
| `vae_dense_weighted` | 0.500 ± 0.081 |
| `vae_token_two_stage` | 0.496 ± 0.079 |
| `vae_token_weighted` | 0.489 ± 0.083 |
| `pycrostates` | 0.482 ± 0.075 |
| `pca8_modkmeans` | 0.482 ± 0.074 |
| `ground_truth` | 0.474 ± 0.068 |
| `vade_dense` | 0.499 ± 0.079 |
| `spectral_global` | 0.485 ± 0.092 |
| `spectral_channel` | 0.511 ± 0.071 |
