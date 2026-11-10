#!/usr/bin/env python3
"""Accordeur chromatique temps réel (DFT maison).

Exemples :
    python main.py                 # micro par défaut
    python main.py --list-devices  # liste des entrées audio
    python main.py --device 2      # choisir le micro
    python main.py --demo          # signal simulé, sans micro
    python main.py --wav note.wav  # analyse un fichier .wav en boucle
"""

import argparse
import sys

from accordeur.audio import SynthSource, WavSource
from accordeur.gui import run


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="signal simulé (aucun micro requis)")
    p.add_argument("--wav", metavar="FICHIER", help="fichier .wav à analyser en boucle")
    p.add_argument("--device", type=int, help="numéro du périphérique d'entrée")
    p.add_argument("--list-devices", action="store_true", help="affiche les entrées audio et quitte")
    p.add_argument("--a4", type=float, default=440.0, help="fréquence du La4 (défaut 440 Hz)")
    p.add_argument("--en", action="store_true", help="noms de notes anglais (C D E…)")
    a = p.parse_args()

    if a.list_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    if a.demo:
        src = SynthSource(a4=a.a4)
    elif a.wav:
        src = WavSource(a.wav)
    else:
        try:
            from accordeur.audio import MicSource
            src = MicSource(device=a.device)
        except Exception as e:                      # pas de micro / PortAudio absent
            print(f"[!] Micro indisponible ({e}).\n    Passage en mode démo.", file=sys.stderr)
            src = SynthSource(a4=a.a4)
            src.label = "Micro indisponible : démo (signal simulé)"
    run(src, a.a4, "en" if a.en else "fr")
    return 0


if __name__ == "__main__":
    sys.exit(main())
