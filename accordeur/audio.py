"""
Sources audio. Toutes exposent :
    .sample_rate        fréquence d'échantillonnage (Hz)
    .label              texte affiché dans la fenêtre
    .read(n)            les n derniers échantillons (float, −1 … +1)
    .start() / .stop()

* MicSource   : micro réel (bibliothèque `sounddevice`)
* WavSource   : lit un fichier .wav en boucle, au rythme du temps réel
* SynthSource : signal de démonstration (corde pincée simulée) — permet de
                tester l'accordeur sans micro, et de produire les captures
                d'écran du rapport.
"""

from __future__ import annotations

import threading
import time
import wave

import numpy as np

from .pitch import midi_to_freq


# --------------------------------------------------------------------------
class MicSource:
    def __init__(self, device=None, sample_rate: float | None = None,
                 seconds: float = 2.0):
        import sounddevice as sd            # import tardif : dépendance optionnelle
        self._sd = sd
        info = sd.query_devices(device, "input")
        self.sample_rate = float(sample_rate or info["default_samplerate"])
        self.label = f"Micro : {info['name']}"
        self._device = device
        self._buf = np.zeros(int(self.sample_rate * seconds), dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        x = indata[:, 0]
        with self._lock:
            self._buf = np.roll(self._buf, -len(x))
            self._buf[-len(x):] = x

    def start(self):
        self._stream = self._sd.InputStream(
            device=self._device, channels=1, samplerate=self.sample_rate,
            callback=self._callback, blocksize=0)
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()

    def read(self, n: int) -> np.ndarray:
        with self._lock:
            return self._buf[-n:].astype(np.float64)


# --------------------------------------------------------------------------
class WavSource:
    def __init__(self, path: str, clock=time.monotonic):
        with wave.open(path, "rb") as w:
            sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
            raw = w.readframes(w.getnframes())
        dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
        x = np.frombuffer(raw, dtype=dtype).astype(np.float64)
        if width == 1:
            x = (x - 128.0) / 128.0
        else:
            x /= float(2 ** (8 * width - 1))
        self._x = x.reshape(-1, ch).mean(axis=1)
        self.sample_rate = float(sr)
        self.label = f"Fichier : {path.split('/')[-1]}"
        self._clock = clock
        self._t0 = clock()

    def start(self):
        self._t0 = self._clock()

    def stop(self):
        pass

    def read(self, n: int) -> np.ndarray:
        end = int((self._clock() - self._t0) * self.sample_rate)
        idx = (np.arange(end - n, end)) % len(self._x)
        out = self._x[idx]
        out[(np.arange(end - n, end)) < 0] = 0.0
        return out


# --------------------------------------------------------------------------
class SynthSource:
    """Corde pincée simulée : harmoniques, enveloppe décroissante, bruit,
    et une justesse qui converge vers 0 cent (comme quand on tourne la
    cheville). Le signal est une fonction du temps : la fenêtre demandée
    est calculée à la volée."""

    # (numéro MIDI, amplitudes des harmoniques 1,2,3,…, écart initial en cents)
    # Les cordes graves ont une fondamentale faible, comme sur un vrai micro.
    SEQUENCE = [
        (40, [0.25, 1.00, 0.60, 0.30, 0.15], -34.0),   # Mi2
        (45, [0.35, 1.00, 0.55, 0.30, 0.12], +26.0),   # La2
        (50, [0.60, 1.00, 0.50, 0.25, 0.10], -22.0),   # Ré3
        (55, [1.00, 0.60, 0.40, 0.20, 0.10], +30.0),   # Sol3
        (59, [1.00, 0.50, 0.30, 0.15, 0.08], -28.0),   # Si3
        (64, [1.00, 0.45, 0.25, 0.12, 0.06], +20.0),   # Mi4
    ]
    SEG = 5.0          # durée d'une note (s)

    def __init__(self, sample_rate: float = 44100.0, a4: float = 440.0,
                 clock=time.monotonic, noise: float = 0.004):
        self.sample_rate = float(sample_rate)
        self.a4 = a4
        self.noise = noise
        self.label = "Démo : signal simulé (corde pincée)"
        self._clock = clock
        self._t0 = clock()
        self._rng = np.random.default_rng(1)

    def start(self):
        self._t0 = self._clock()

    def stop(self):
        pass

    # justesse (cents) en fonction du temps écoulé depuis le pincement
    @staticmethod
    def _cents(c0, s):
        # converge vers 0 entre s = 0,5 et s = 3,0 s, puis reste juste
        u = np.clip((s - 0.5) / 2.5, 0.0, 1.0)
        return c0 * (1.0 - u) ** 2

    def read(self, n: int) -> np.ndarray:
        now = self._clock() - self._t0
        t = now - (n - 1 - np.arange(n)) / self.sample_rate
        seg = np.floor(np.maximum(t, 0) / self.SEG).astype(int) % len(self.SEQUENCE)
        s = np.maximum(t, 0) % self.SEG
        f_inst = np.empty(n)
        amp = np.zeros(n)
        harm = np.zeros((n, 5))
        for i, (midi, a, c0) in enumerate(self.SEQUENCE):
            m = seg == i
            if not m.any():
                continue
            f_inst[m] = midi_to_freq(midi + self._cents(c0, s[m]) / 100.0, self.a4)
            harm[m] = a
        env = np.where(t < 0, 0.0, 0.5 * np.exp(-0.25 * s))       # pincement + decroissance
        phase = 2.0 * np.pi * np.cumsum(f_inst) / self.sample_rate
        y = np.zeros(n)
        for h in range(5):
            y += harm[:, h] * np.sin((h + 1) * phase)
        y *= env / 1.6
        y += self.noise * self._rng.standard_normal(n)
        return y
