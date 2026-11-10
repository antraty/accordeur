"""
Transformée de Fourier Discrète (DFT) écrite « à la main », d'après le cours.

Définition du cours (indices MATLAB/cours : X(k+1), x(n+1), k et n de 0 à N-1) :

            N-1
    X(k) =  Σ   x(n) · e^(-i·2π·k·n / N)
            n=0

Trois versions de la MÊME formule (aucune FFT, tout est en O(N²)) :

* dft_naive   : double boucle pure Python, recopie littérale du cours
                (sert de référence pédagogique et de test).
* BandDFT     : la même somme, mais évaluée d'un coup sous forme de produit
                matrice × vecteur NumPy, et seulement pour les fréquences
                utiles à un accordeur (≈ 55 Hz … 1600 Hz). C'est ce qui rend
                le calcul assez rapide pour le temps réel.
* BandDFT.at  : la formule évaluée à des fréquences quelconques (non
                entières en « numéro de bin ») pour affiner le pic (« zoom »).
"""

from __future__ import annotations

import cmath
import math

import numpy as np


# --------------------------------------------------------------------------
# 1) Version littérale du cours
# --------------------------------------------------------------------------
def dft_naive(x):
    """DFT de la séquence x (liste ou tableau), double boucle, O(N²).

    Reproduit l'algorithme du cours :
        X = (0, 0, ..., 0)                  # même taille que x
        pour k de 0 à N-1
            pour n de 0 à N-1
                X[k] = X[k] + x[n] · e^(-i 2π k n / N)
    """
    N = len(x)
    X = [0j] * N
    for k in range(N):
        for n in range(N):
            X[k] = X[k] + x[n] * cmath.exp(-1j * 2 * math.pi * k * n / N)
    return X


# --------------------------------------------------------------------------
# 2) Version « bande de fréquences » pour le temps réel
# --------------------------------------------------------------------------
class BandDFT:
    """DFT de N échantillons, calculée uniquement pour les bins k_min … k_max.

    La matrice W[j, n] = e^(-i 2π k_j n / N) est calculée UNE fois (à la
    construction) ; chaque trame coûte ensuite un produit matrice-vecteur
    de taille (K × N), soit O(K·N) = O(N²) comme la DFT du cours.
    """

    def __init__(self, n: int, fs: float, f_min: float, f_max: float):
        self.n = int(n)
        self.fs = float(fs)
        self.df = self.fs / self.n                       # résolution (Hz/bin)
        nyquist_bin = self.n // 2
        self.k_min = max(1, int(math.ceil(f_min / self.df)))
        self.k_max = min(nyquist_bin, int(math.floor(f_max / self.df)))
        self.k = np.arange(self.k_min, self.k_max + 1)   # numéros de bins
        self.freqs = self.k * self.df                    # fréquences en Hz

        n_idx = np.arange(self.n)
        # (k·n) mod N en entiers : exact, puis exponentielle complexe.
        kn = np.outer(self.k, n_idx) % self.n
        self._w = np.exp(-2j * np.pi * kn / self.n).astype(np.complex64)
        self._n_idx = n_idx.astype(np.float64)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """X(k) pour k ∈ [k_min, k_max]  (tableau complexe)."""
        x = np.asarray(x, dtype=np.float32)
        return self._w @ x.astype(np.complex64)

    def at(self, x: np.ndarray, freqs_hz: np.ndarray) -> np.ndarray:
        """Même somme Σ x(n)·e^(-i 2π f n / fs) pour des fréquences f (Hz)
        quelconques, ce qui revient à un « numéro de bin » k = f·N/fs réel."""
        x = np.asarray(x, dtype=np.float64)
        freqs_hz = np.asarray(freqs_hz, dtype=np.float64)
        phase = np.outer(freqs_hz / self.fs, self._n_idx)     # f·n / fs
        return np.exp(-2j * np.pi * phase) @ x
