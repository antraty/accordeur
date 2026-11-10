"""Tests : python -m unittest discover -s tests -v   (depuis la racine du projet)"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from accordeur.dft import BandDFT, dft_naive                       # noqa: E402
from accordeur.pitch import (TunerEngine, midi_to_freq, nearest_note,  # noqa: E402
                             note_name)

SR = 44100.0


def tone(f, harmonics=(1.0,), dur=1.0, noise=0.0, sr=SR, seed=0):
    t = np.arange(int(dur * sr)) / sr
    y = sum(a * np.sin(2 * np.pi * (h + 1) * f * t + 0.7 * h) for h, a in enumerate(harmonics))
    y = 0.4 * y / max(1e-9, sum(harmonics))
    if noise:
        y = y + noise * np.random.default_rng(seed).standard_normal(len(t))
    return y


def detect(sig, sr=SR):
    eng = TunerEngine(sr)
    return eng.process(sig[-eng.raw_len:])


class TestDFT(unittest.TestCase):
    def test_exemple_du_cours(self):
        x = [1, 6, 8, -4, 9]
        X = dft_naive(x)
        self.assertAlmostEqual(X[0].real, 20.0, places=9)        # X(1) = 20 dans le cours
        np.testing.assert_allclose(X, np.fft.fft(x), atol=1e-9)

    def test_naive_vs_fft(self):
        rng = np.random.default_rng(3)
        x = rng.standard_normal(64)
        np.testing.assert_allclose(dft_naive(x), np.fft.fft(x), atol=1e-9)

    def test_bande_vs_fft(self):
        rng = np.random.default_rng(4)
        n = 512
        x = rng.standard_normal(n)
        b = BandDFT(n, 5000.0, 100.0, 1500.0)
        np.testing.assert_allclose(b.transform(x), np.fft.fft(x)[b.k], rtol=1e-3, atol=1e-3)

    def test_zoom_coincide_avec_bin(self):
        rng = np.random.default_rng(5)
        n = 512
        x = rng.standard_normal(n)
        b = BandDFT(n, 5000.0, 100.0, 1500.0)
        np.testing.assert_allclose(b.at(x, b.freqs[:5]), np.fft.fft(x)[b.k[:5]], atol=1e-8)


class TestNotes(unittest.TestCase):
    def test_la4(self):
        n, c = nearest_note(440.0)
        self.assertEqual(note_name(n, "fr"), ("La", 4))
        self.assertAlmostEqual(c, 0.0, places=6)

    def test_do4_mi2(self):
        self.assertEqual(note_name(nearest_note(261.6256)[0], "fr"), ("Do", 4))
        self.assertEqual(note_name(nearest_note(82.4069)[0], "en"), ("E", 2))

    def test_cents(self):
        _, c = nearest_note(440.0 * 2 ** (13 / 1200))
        self.assertAlmostEqual(c, 13.0, places=6)


class TestDetection(unittest.TestCase):
    def test_precision_sinus(self):
        worst = 0.0
        for f in [55, 82.41, 110, 146.83, 196, 246.94, 329.63, 440, 523.25, 880, 1000, 1318.5]:
            for cents in (-30, 0, 17):
                ft = f * 2 ** (cents / 1200)
                r = detect(tone(ft))
                self.assertTrue(r.voiced, f"{ft} Hz non détecté")
                err = 1200 * math.log2(r.f0 / ft)
                worst = max(worst, abs(err))
                self.assertLess(abs(err), 1.0, f"{ft:.2f} Hz : erreur {err:.2f} cents")
        print(f"\n  pire erreur (sinus purs) : {worst:.3f} cents")

    def test_fondamentale_faible(self):
        # Mi2 de guitare : 2e harmonique 4× plus forte que la fondamentale
        for f in (82.41, 110.0, 146.83):
            r = detect(tone(f, harmonics=(0.25, 1.0, 0.6, 0.3, 0.15)))
            self.assertTrue(r.voiced)
            self.assertLess(abs(1200 * math.log2(r.f0 / f)), 1.5, f"f={f} -> {r.f0}")

    def test_bruit(self):
        for f in (110.0, 329.63, 440.0):
            r = detect(tone(f, harmonics=(1, 0.5, 0.3), noise=0.03, seed=2))
            self.assertTrue(r.voiced)
            self.assertLess(abs(1200 * math.log2(r.f0 / f)), 2.0)

    def test_silence_et_bruit_seul(self):
        self.assertFalse(detect(np.zeros(44100)).voiced)
        self.assertFalse(detect(0.002 * np.random.default_rng(1).standard_normal(44100)).voiced)
        self.assertFalse(detect(0.1 * np.random.default_rng(1).standard_normal(44100)).voiced)

    def test_autres_frequences_echantillonnage(self):
        for sr in (48000.0, 16000.0, 22050.0):
            ft = 196.0
            r = detect(tone(ft, harmonics=(1, 0.4), sr=sr), sr)
            self.assertTrue(r.voiced)
            self.assertLess(abs(1200 * math.log2(r.f0 / ft)), 1.0)


class TestInterface(unittest.TestCase):
    def test_application_hors_ecran(self):
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        import pygame
        from accordeur.audio import SynthSource
        from accordeur.gui import H, W, TunerApp
        pygame.init()
        screen = pygame.display.set_mode((W, H))
        t = [0.0]
        clk = lambda: t[0]                                  # noqa: E731
        src = SynthSource(clock=clk)
        app = TunerApp(src, 440.0, "fr", clock=clk)
        src.start()
        while t[0] < 4.6:
            t[0] += 1 / 30
            app.update(t[0])
        app.draw(screen)
        self.assertTrue(app.res.voiced)
        self.assertAlmostEqual(app.res.f0, 82.41, delta=0.1)
        for key in (pygame.K_UP, pygame.K_DOWN, pygame.K_n, pygame.K_SPACE):
            self.assertTrue(app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=key)))
        self.assertFalse(app.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE)))
        pygame.quit()


if __name__ == "__main__":
    unittest.main(verbosity=2)
