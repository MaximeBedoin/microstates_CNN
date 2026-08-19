# Classification — ds004504 (K = 4)

65 sujets, positif = `AD` (36 sujets). Validation croisee 5 plis x 10 repetitions.

| Methode | AUC | IC 95 % |
|---|---:|---|
| `pca8_modkmeans` | 0.898 | [0.809, 0.970] |
| `pycrostates` | 0.859 | [0.767, 0.939] |
| `spectral_global` | 0.856 | [0.751, 0.939] |
| `vae_dense_two_stage` | 0.849 | [0.749, 0.934] |
| `vae_dense_weighted` | 0.821 | [0.711, 0.917] |
| `spectral_channel` | 0.819 | [0.699, 0.920] |
| `vae_conv_weighted` | 0.807 | [0.699, 0.901] |
| `vae_conv_two_stage` | 0.792 | [0.674, 0.897] |
| `vae_token_two_stage` | 0.710 | [0.578, 0.837] |
| `vae_token_weighted` | 0.660 | [0.522, 0.794] |

## Comparaisons appariees (bootstrap sur les sujets)

| Paire | delta AUC | IC 95 % | p |
|---|---:|---|---:|
| vae_token_weighted - pca8_modkmeans | -0.238 | [-0.364, -0.130] | 0.000 |
| vae_token_weighted - pycrostates | -0.199 | [-0.327, -0.092] | 0.000 |
| vae_token_weighted - spectral_global | -0.196 | [-0.337, -0.056] | 0.005 |
| vae_dense_two_stage - vae_token_weighted | +0.189 | [+0.053, +0.336] | 0.009 |
| vae_token_two_stage - pca8_modkmeans | -0.188 | [-0.307, -0.076] | 0.001 |
| vae_dense_weighted - vae_token_weighted | +0.161 | [+0.032, +0.308] | 0.011 |
| vae_token_weighted - spectral_channel | -0.159 | [-0.299, -0.024] | 0.018 |
| vae_token_two_stage - pycrostates | -0.149 | [-0.265, -0.044] | 0.008 |
| vae_conv_weighted - vae_token_weighted | +0.148 | [+0.007, +0.301] | 0.042 |
| vae_token_two_stage - spectral_global | -0.147 | [-0.278, -0.013] | 0.028 |
| vae_dense_two_stage - vae_token_two_stage | +0.139 | [+0.011, +0.270] | 0.037 |
| vae_conv_two_stage - vae_token_weighted | +0.132 | [+0.002, +0.284] | 0.048 |
| vae_dense_weighted - vae_token_two_stage | +0.111 | [-0.015, +0.241] | 0.097 |
| vae_token_two_stage - spectral_channel | -0.109 | [-0.248, +0.020] | 0.125 |
| vae_conv_two_stage - pca8_modkmeans | -0.105 | [-0.194, -0.024] | 0.010 |
| vae_conv_weighted - vae_token_two_stage | +0.098 | [-0.042, +0.239] | 0.164 |
| vae_conv_weighted - pca8_modkmeans | -0.090 | [-0.186, -0.010] | 0.030 |
| vae_conv_two_stage - vae_token_two_stage | +0.082 | [-0.047, +0.219] | 0.209 |
| pca8_modkmeans - spectral_channel | +0.079 | [-0.027, +0.195] | 0.135 |
| vae_dense_weighted - pca8_modkmeans | -0.077 | [-0.175, +0.016] | 0.118 |
| vae_conv_two_stage - pycrostates | -0.067 | [-0.149, +0.009] | 0.098 |
| vae_conv_two_stage - spectral_global | -0.064 | [-0.176, +0.052] | 0.295 |
| vae_conv_two_stage - vae_dense_two_stage | -0.057 | [-0.142, +0.027] | 0.169 |
| vae_conv_weighted - pycrostates | -0.052 | [-0.134, +0.021] | 0.173 |
| vae_token_two_stage - vae_token_weighted | +0.050 | [-0.027, +0.139] | 0.235 |
| vae_conv_weighted - spectral_global | -0.049 | [-0.171, +0.079] | 0.448 |
| vae_dense_two_stage - pca8_modkmeans | -0.049 | [-0.132, +0.028] | 0.216 |
| vae_conv_weighted - vae_dense_two_stage | -0.041 | [-0.129, +0.040] | 0.302 |
| pca8_modkmeans - spectral_global | +0.041 | [-0.061, +0.160] | 0.418 |
| pycrostates - spectral_channel | +0.040 | [-0.070, +0.154] | 0.449 |
| pycrostates - pca8_modkmeans | -0.038 | [-0.088, +0.004] | 0.078 |
| vae_dense_weighted - pycrostates | -0.038 | [-0.114, +0.036] | 0.304 |
| spectral_global - spectral_channel | +0.037 | [-0.011, +0.096] | 0.183 |
| vae_dense_weighted - spectral_global | -0.035 | [-0.158, +0.093] | 0.592 |
| vae_dense_two_stage - spectral_channel | +0.030 | [-0.090, +0.151] | 0.608 |
| vae_conv_two_stage - vae_dense_weighted | -0.029 | [-0.129, +0.068] | 0.555 |
| vae_dense_two_stage - vae_dense_weighted | +0.028 | [-0.074, +0.134] | 0.597 |
| vae_conv_two_stage - spectral_channel | -0.027 | [-0.140, +0.086] | 0.640 |
| vae_conv_two_stage - vae_conv_weighted | -0.015 | [-0.075, +0.042] | 0.597 |
| vae_conv_weighted - vae_dense_weighted | -0.013 | [-0.112, +0.078] | 0.790 |
| vae_conv_weighted - spectral_channel | -0.011 | [-0.132, +0.119] | 0.887 |
| vae_dense_two_stage - pycrostates | -0.011 | [-0.089, +0.061] | 0.772 |
| vae_dense_two_stage - spectral_global | -0.008 | [-0.115, +0.110] | 0.917 |
| pycrostates - spectral_global | +0.003 | [-0.101, +0.119] | 0.934 |
| vae_dense_weighted - spectral_channel | +0.002 | [-0.124, +0.132] | 0.969 |

## Controle par permutation (doit valoir ~0.5)

| Methode | AUC permutee |
|---|---:|
| `vae_conv_two_stage` | 0.502 ± 0.073 |
| `vae_conv_weighted` | 0.521 ± 0.060 |
| `vae_dense_two_stage` | 0.527 ± 0.067 |
| `vae_dense_weighted` | 0.488 ± 0.068 |
| `vae_token_two_stage` | 0.455 ± 0.087 |
| `vae_token_weighted` | 0.478 ± 0.094 |
| `pycrostates` | 0.525 ± 0.081 |
| `pca8_modkmeans` | 0.536 ± 0.091 |
| `spectral_global` | 0.455 ± 0.112 |
| `spectral_channel` | 0.494 ± 0.093 |
