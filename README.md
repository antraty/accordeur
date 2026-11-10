# Accordeur chromatique temps réel (DFT maison)

    pip install -r requirements.txt
    python main.py --demo      # sans micro
    python main.py             # micro par défaut
    python -m unittest discover -s tests

La DFT (formule du cours, O(N²)) est dans `accordeur/dft.py`. Le rapport est dans `rapport/Rapport_Accordeur.pdf`.
