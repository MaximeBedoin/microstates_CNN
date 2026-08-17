"""msvae — microstates EEG via VAE convolutionnel sur images topographiques.

Modules
-------
topo         : projection 2D des electrodes + interpolation spline spherique -> images
preprocess   : filtrage, reference moyenne, extraction des pics de GFP
synthetic    : generateur de donnees avec dipoles connus (ground truth)
models       : VAE convolutionnel (images) et VAE dense (vecteurs) — ablation
train        : protocole d'entrainement (split par sujet, equilibrage)
cluster      : k-means dans l'espace latent (2 temps / 1 temps pondere)
microstates  : back-projection et parametres (duree, occurrence, GEV, transitions, LZC)
baseline     : pipeline Pycrostates classique
evaluate     : appariement de cartes, stabilite, cartes canoniques
"""

__all__ = [
    "topo",
    "preprocess",
    "synthetic",
    "models",
    "train",
    "cluster",
    "microstates",
    "baseline",
    "evaluate",
]
