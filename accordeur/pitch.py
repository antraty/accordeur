"""
Détection de la fréquence fondamentale (f₀) et conversion en note.

Chaîne de traitement (détaillée dans le rapport) :

    signal brut (ex. 44,1 kHz)
      → filtre passe-bas + décimation   (ex. 5,5 kHz : moins de points à traiter)
      → retrait de la composante continue + fenêtre de Hann
      → DFT (bins 55 … 1600 Hz)         → spectre d'amplitude
      → pic principal, puis correction d'octave (le pic le plus fort est
        parfois un harmonique, pas la fondamentale)
      → « zoom » : DFT évaluée finement autour du pic + interpolation
        parabolique → f₀ précise (< 1 cent)
      → note la plus proche + écart en cents
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .dft import BandDFT

NOTES_EN = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTES_FR = ["Do", "Do#", "Ré", "Ré#", "Mi", "Fa", "Fa#", "Sol", "Sol#", "La", "La#", "Si"]


# --------------------------------------------------------------------------
# Notes
# --------------------------------------------------------------------------
def freq_to_midi(f: float, a4: float = 440.0) -> float:
    """Numéro de note MIDI (réel). La4 = 69."""
    return 69.0 + 12.0 * math.log2(f / a4)


def midi_to_freq(m: float, a4: float = 440.0) -> float:
    return a4 * 2.0 ** ((m - 69.0) / 12.0)


def note_name(midi_int: int, notation: str = "fr") -> tuple[str, int]:
    names = NOTES_FR if notation == "fr" else NOTES_EN
    return names[midi_int % 12], midi_int // 12 - 1


def nearest_note(f: float, a4: float = 440.0):
    """(midi entier, écart en cents entre f et la note la plus proche)."""
    m = freq_to_midi(f, a4)
    n = int(round(m))
    return n, 100.0 * (m - n)


# --------------------------------------------------------------------------
# Pré-traitement
# --------------------------------------------------------------------------
def hann(n: int) -> np.ndarray:
    """Fenêtre de Hann : w(n) = ½ (1 − cos(2π n / N))."""
    k = np.arange(n)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * k / n))


def make_lowpass(factor: int, taps_per_factor: int = 8) -> np.ndarray:
    """Filtre RIF passe-bas (sinc fenêtré de Hamming) avant décimation."""
    if factor <= 1:
        return np.array([1.0])
    m = taps_per_factor * factor + 1
    t = np.arange(m) - (m - 1) / 2.0
    fc = 0.45 / factor                      # fréquence de coupure (cycles/échantillon)
    h = np.sinc(2.0 * fc * t) * np.hamming(m)
    return h / h.sum()


def decimate(raw: np.ndarray, h: np.ndarray, factor: int) -> np.ndarray:
    """Filtre passe-bas puis garde 1 échantillon sur `factor`."""
    if factor <= 1:
        return raw.astype(np.float64)
    y = np.convolve(raw.astype(np.float64), h, mode="valid")
    return y[::factor]


# --------------------------------------------------------------------------
# Détecteur
# --------------------------------------------------------------------------
@dataclass
class Analysis:
    voiced: bool = False          # une note est-elle détectée ?
    f0: float = 0.0               # fréquence fondamentale (Hz)
    midi: int = 0                 # note la plus proche (MIDI)
    cents: float = 0.0            # écart à la note (−50 … +50)
    rms: float = 0.0              # niveau du signal
    freqs: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mag_db: np.ndarray = field(default_factory=lambda: np.zeros(0))   # dBFS
    f_peak: float = 0.0           # pic le plus fort avant correction d'octave


class PitchDetector:
    def __init__(
        self,
        fs: float,
        n: int = 4096,
        f_min: float = 55.0,
        f_max: float = 1600.0,
        a4: float = 440.0,
        rms_gate: float = 0.004,
        snr_gate: float = 6.0,
        sub_ratio: float = 0.20,
    ):
        self.fs = float(fs)
        self.n = int(n)
        self.f_min, self.f_max = f_min, f_max
        self.a4 = a4
        self.rms_gate = rms_gate          # seuil de niveau (amplitude RMS)
        self.snr_gate = snr_gate          # pic / médiane du spectre
        self.sub_ratio = sub_ratio        # seuil du sous-harmonique (÷ pic)
        self.dft = BandDFT(self.n, self.fs, f_min, f_max)
        self.window = hann(self.n)
        self._amp_norm = 2.0 / self.window.sum()   # |X| → amplitude crête

    # ----------------------------------------------------------------------
    def _peak_near(self, mag: np.ndarray, f: float):
        """Indice d'un maximum local du spectre autour de f, ou None."""
        fr = self.dft.freqs
        tol = max(2.0 * self.dft.df, 0.02 * f)
        lo = int(np.searchsorted(fr, f - tol))
        hi = int(np.searchsorted(fr, f + tol))
        if lo < 1 or hi >= len(fr) - 1 or hi <= lo:
            return None
        j = lo + int(np.argmax(mag[lo:hi]))
        if mag[j] >= mag[j - 1] and mag[j] >= mag[j + 1]:
            return j
        return None

    def _refine(self, xw: np.ndarray, f_coarse: float) -> float:
        """Zoom : la formule de la DFT évaluée sur une grille fine (pas de
        0,025 bin) autour de f_coarse, puis interpolation parabolique."""
        span = 1.5 * self.dft.df
        grid = f_coarse + np.linspace(-span, span, 121)
        m = np.abs(self.dft.at(xw, grid))
        j = int(np.argmax(m))
        if 0 < j < len(grid) - 1:
            a, b, c = m[j - 1], m[j], m[j + 1]
            den = a - 2.0 * b + c
            delta = 0.5 * (a - c) / den if den != 0 else 0.0
            return float(grid[j] + delta * (grid[1] - grid[0]))
        return float(grid[j])

    # ----------------------------------------------------------------------
    def analyze(self, frame: np.ndarray) -> Analysis:
        x = np.asarray(frame, dtype=np.float64)
        if len(x) != self.n:
            raise ValueError(f"trame de {len(x)} échantillons, {self.n} attendus")
        x = x - x.mean()                           # retire la composante continue
        rms = float(np.sqrt(np.mean(x ** 2)))
        xw = x * self.window

        X = self.dft.transform(xw)
        mag = np.abs(X) * self._amp_norm           # amplitude crête estimée
        mag_db = 20.0 * np.log10(np.maximum(mag, 1e-9))
        out = Analysis(rms=rms, freqs=self.dft.freqs, mag_db=mag_db)

        i0 = int(np.argmax(mag))
        floor = float(np.median(mag)) + 1e-12
        if rms < self.rms_gate or mag[i0] / floor < self.snr_gate:
            return out                              # silence / bruit

        f_peak = float(self.dft.freqs[i0])
        out.f_peak = f_peak

        # Correction d'octave : si un pic net existe à f/2, f/3 ou f/4,
        # le pic le plus fort était un harmonique -> on descend.
        f_cur = f_peak
        for _ in range(4):
            cands = []
            for m in (2, 3, 4):
                f_sub = f_cur / m
                if f_sub < self.f_min:
                    continue
                j = self._peak_near(mag, f_sub)
                if (j is not None and mag[j] >= self.sub_ratio * mag[i0]
                        and mag[j] / floor >= self.snr_gate):
                    cands.append(float(self.dft.freqs[j]))
            if not cands:
                break
            f_cur = min(cands)

        f0 = self._refine(xw, f_cur)
        out.voiced = True
        out.f0 = f0
        out.midi, out.cents = nearest_note(f0, self.a4)
        return out


# --------------------------------------------------------------------------
# Moteur : décimation + détecteur + lissage temporel
# --------------------------------------------------------------------------
class TunerEngine:
    N = 4096

    def __init__(self, sample_rate: float, a4: float = 440.0,
                 f_min: float = 55.0, f_max: float = 1600.0):
        self.sample_rate = float(sample_rate)
        self.factor = max(1, int(round(self.sample_rate / 5500.0)))
        self.fs = self.sample_rate / self.factor          # fréquence d'éch. effective
        self.h = make_lowpass(self.factor)
        self.detector = PitchDetector(self.fs, self.N, f_min, f_max, a4)
        self.raw_len = self.N * self.factor + len(self.h) - 1
        self._hist: deque[float] = deque(maxlen=5)
        self._miss = 0

    @property
    def a4(self) -> float:
        return self.detector.a4

    @a4.setter
    def a4(self, v: float) -> None:
        self.detector.a4 = float(v)

    @property
    def resolution(self) -> float:
        return self.detector.dft.df

    def process(self, raw: np.ndarray) -> Analysis:
        frame = decimate(raw, self.h, self.factor)[: self.N]
        res = self.detector.analyze(frame)
        if res.voiced:
            self._miss = 0
            # un saut d'octave brutal vide l'historique (changement de note)
            if self._hist and abs(math.log2(res.f0 / self._hist[-1])) > 0.06:
                self._hist.clear()
            self._hist.append(res.f0)
            res.f0 = float(np.median(self._hist))       # médiane : anti-aberrations
            res.midi, res.cents = nearest_note(res.f0, self.a4)
        else:
            self._miss += 1
            if self._miss > 4:
                self._hist.clear()
        return res
