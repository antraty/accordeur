"""
Fenêtre de l'accordeur (pygame) : aiguille d'accord + spectre en direct.

La classe TunerApp sépare :
    update(now)  → lit l'audio, lance l'analyse, anime l'aiguille
    draw(surface)→ dessine l'interface
afin de pouvoir aussi produire des captures d'écran sans ouvrir de fenêtre.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pygame
import pygame.gfxdraw as gfx

from .pitch import Analysis, TunerEngine, midi_to_freq, note_name

# --- palette ---------------------------------------------------------------
BG = (15, 17, 23)
CARD = (24, 27, 36)
LINE = (46, 51, 64)
TEXT = (234, 236, 242)
MUTED = (134, 142, 160)
GREEN = (66, 211, 146)
AMBER = (242, 183, 62)
RED = (242, 98, 98)
ACCENT = (98, 164, 255)

W, H = 1000, 744
FONT_STACK = "inter,segoeui,helveticaneue,helvetica,arial,dejavusans"


def status_color(cents: float) -> tuple[int, int, int]:
    a = abs(cents)
    return GREEN if a <= 5 else AMBER if a <= 15 else RED


def mix(c1, c2, t):
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


class Button:
    def __init__(self, rect, label_fn, action):
        self.rect = pygame.Rect(rect)
        self.label_fn = label_fn
        self.action = action


class TunerApp:
    HOP = 0.08            # secondes entre deux analyses

    def __init__(self, source, a4: float = 440.0, notation: str = "fr",
                 clock=time.monotonic):
        self.source = source
        self.engine = TunerEngine(source.sample_rate, a4)
        self.notation = notation
        self.paused = False
        self.clock = clock
        self.res = Analysis()
        self.needle = 0.0               # position affichée (cents)
        self.presence = 0.0             # 0 = pas de note, 1 = note (fondu)
        self.spec = None                # spectre lissé pour l'affichage
        self._last_analysis = -1e9
        self._last_update = None
        self.last_ms = 0.0
        self.fonts: dict = {}
        self.buttons = [
            Button((40, 664, 110, 36), lambda: "A4 −", lambda: self.add_a4(-1)),
            Button((158, 664, 110, 36), lambda: "A4 +", lambda: self.add_a4(+1)),
            Button((276, 664, 170, 36),
                   lambda: "Notation : " + ("Do Ré Mi" if self.notation == "fr" else "C D E"),
                   self.toggle_notation),
            Button((454, 664, 120, 36),
                   lambda: "Reprendre" if self.paused else "Pause", self.toggle_pause),
        ]

    # ------------------------------------------------------------------ état
    def add_a4(self, d):
        self.engine.a4 = min(466.0, max(415.0, self.engine.a4 + d))

    def toggle_notation(self):
        self.notation = "en" if self.notation == "fr" else "fr"

    def toggle_pause(self):
        self.paused = not self.paused

    def handle_event(self, e) -> bool:
        """Renvoie False pour quitter."""
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if e.key == pygame.K_UP:
                self.add_a4(+1)
            elif e.key == pygame.K_DOWN:
                self.add_a4(-1)
            elif e.key == pygame.K_n:
                self.toggle_notation()
            elif e.key == pygame.K_SPACE:
                self.toggle_pause()
        if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            for b in self.buttons:
                if b.rect.collidepoint(e.pos):
                    b.action()
        return True

    # ---------------------------------------------------------------- calcul
    def update(self, now: float | None = None):
        now = self.clock() if now is None else now
        dt = 0.033 if self._last_update is None else max(0.0, now - self._last_update)
        self._last_update = now

        if not self.paused and now - self._last_analysis >= self.HOP:
            self._last_analysis = now
            t0 = time.perf_counter()
            raw = self.source.read(self.engine.raw_len)
            self.res = self.engine.process(raw)
            self.last_ms = (time.perf_counter() - t0) * 1000.0
            if len(self.res.mag_db):
                if self.spec is None or len(self.spec) != len(self.res.mag_db):
                    self.spec = self.res.mag_db.copy()
                else:
                    self.spec = 0.45 * self.spec + 0.55 * self.res.mag_db

        # animation : l'aiguille rejoint la mesure avec une petite inertie
        target = self.res.cents if self.res.voiced else 0.0
        target = max(-50.0, min(50.0, target))
        k = 1.0 - math.exp(-dt / 0.07)
        self.needle += (target - self.needle) * k
        self.presence += ((1.0 if self.res.voiced else 0.0) - self.presence) * (1.0 - math.exp(-dt / 0.12))

    # --------------------------------------------------------------- dessin
    def font(self, size, bold=False):
        key = (size, bold)
        if key not in self.fonts:
            self.fonts[key] = pygame.font.SysFont(FONT_STACK, size, bold=bold)
        return self.fonts[key]

    def text(self, s, txt, size, color, pos, anchor="topleft", bold=False):
        img = self.font(size, bold).render(txt, True, color)
        r = img.get_rect()
        setattr(r, anchor, pos)
        s.blit(img, r)
        return r

    @staticmethod
    def card(s, rect, radius=14):
        pygame.draw.rect(s, CARD, rect, border_radius=radius)
        pygame.draw.rect(s, LINE, rect, width=1, border_radius=radius)

    def draw(self, s: pygame.Surface):
        s.fill(BG)
        self.text(s, "Accordeur chromatique", 26, TEXT, (40, 26), bold=True)
        self.text(s, self.source.label, 15, MUTED, (W - 40, 34), "topright")
        self.draw_gauge(s, pygame.Rect(40, 76, 590, 330))
        self.draw_readout(s, pygame.Rect(648, 76, 312, 330))
        self.draw_spectrum(s, pygame.Rect(40, 424, 920, 222))
        self.draw_footer(s)

    # ---- aiguille ----------------------------------------------------------
    def draw_gauge(self, s, rect):
        self.card(s, rect)
        cx, cy = rect.centerx, rect.bottom - 50
        R = min(rect.w // 2 - 36, rect.h - 100)
        span = 62.0                              # demi-ouverture en degrés (±50 cents)

        def pt(c, r):
            a = math.radians(c / 50.0 * span)
            return (cx + r * math.sin(a), cy - r * math.cos(a))

        # bandes colorées (zones de justesse)
        r_out, r_in = R, R - 16
        steps = 100
        for i in range(steps):
            c0 = -50 + 100 * i / steps
            c1 = -50 + 100 * (i + 1) / steps
            col = mix(CARD, status_color((c0 + c1) / 2), 0.55)
            poly = [pt(c0, r_out), pt(c1, r_out), pt(c1, r_in), pt(c0, r_in)]
            poly = [(int(round(x)), int(round(y))) for x, y in poly]
            gfx.filled_polygon(s, poly, col)
            gfx.aapolygon(s, poly, col)

        # graduations
        for c in range(-50, 51, 5):
            major = c % 10 == 0
            r0 = r_in - (16 if major else 8)
            p0, p1 = pt(c, r0), pt(c, r_in - 3)
            pygame.draw.aaline(s, TEXT if c == 0 else MUTED, p0, p1)
        for c in (-50, -25, 0, 25, 50):
            lab = "0" if c == 0 else f"{c:+d}"
            x, y = pt(c, r_in - 38)
            self.text(s, lab, 15, MUTED, (x, y), "center")
        self.text(s, "bas", 14, MUTED, pt(-50, R + 4), "midright")
        self.text(s, "haut", 14, MUTED, pt(+50, R + 4), "midleft")
        self.text(s, "cents", 14, MUTED, (cx, cy - R * 0.42), "center")

        # aiguille
        col = status_color(self.needle) if self.presence > 0.5 else MUTED
        col = mix(MUTED, col, min(1.0, self.presence * 1.5))
        tip = pt(self.needle, R - 6)
        a = math.radians(self.needle / 50.0 * span)
        nx, ny = math.cos(a), math.sin(a)             # normale à l'aiguille
        base1 = (cx - 6 * nx, cy - 6 * ny)
        base2 = (cx + 6 * nx, cy + 6 * ny)
        poly = [(int(base1[0]), int(base1[1])), (int(tip[0]), int(tip[1])),
                (int(base2[0]), int(base2[1]))]
        gfx.filled_polygon(s, poly, col)
        gfx.aapolygon(s, poly, col)
        gfx.filled_circle(s, cx, cy, 13, col)
        gfx.aacircle(s, cx, cy, 13, col)
        gfx.filled_circle(s, cx, cy, 5, CARD)

    # ---- lecture numérique -------------------------------------------------
    def draw_readout(self, s, rect):
        self.card(s, rect)
        r = self.res
        cx = rect.centerx
        if r.voiced:
            name, octv = note_name(r.midi, self.notation)
            col = status_color(r.cents)
            big = self.text(s, name, 104, col, (cx - 16, rect.top + 100), "center", bold=True)
            self.text(s, str(octv), 36, MUTED, (big.right + 4, big.bottom - 14), "midleft")
            self.text(s, f"{r.f0:7.2f} Hz", 30, TEXT, (cx, rect.top + 190), "center")
            self.text(s, f"{r.cents:+.1f} cents", 24, col, (cx, rect.top + 228), "center")
            if abs(r.cents) <= 5:
                msg, mc = "Accordé", GREEN
            elif r.cents < 0:
                msg, mc = "Trop grave : tendre la corde", AMBER if abs(r.cents) <= 15 else RED
            else:
                msg, mc = "Trop aigu : détendre la corde", AMBER if abs(r.cents) <= 15 else RED
            self.text(s, msg, 17, mc, (cx, rect.top + 272), "center", bold=True)
            tgt = midi_to_freq(r.midi, self.engine.a4)
            self.text(s, f"Cible : {tgt:.2f} Hz", 15, MUTED, (cx, rect.top + 300), "center")
        else:
            self.text(s, "—", 104, LINE, (cx, rect.top + 100), "center", bold=True)
            self.text(s, "Jouez une note…", 24, MUTED, (cx, rect.top + 200), "center")
            self.text(s, "Aucune fréquence détectée", 15, MUTED, (cx, rect.top + 240), "center")

    # ---- spectre -----------------------------------------------------------
    def draw_spectrum(self, s, rect):
        self.card(s, rect)
        self.text(s, "Spectre d'amplitude (DFT) — niveau en dBFS", 16, TEXT, (rect.left + 18, rect.top + 12), bold=True)
        res_txt = (f"N = {self.engine.N}   fs = {self.engine.fs:.0f} Hz   "
                   f"résolution = {self.engine.resolution:.2f} Hz/bin   "
                   f"calcul : {self.last_ms:.0f} ms")
        self.text(s, res_txt, 13, MUTED, (rect.right - 18, rect.top + 14), "topright")

        plot = pygame.Rect(rect.left + 58, rect.top + 44, rect.w - 80, rect.h - 82)
        fmax = self.engine.detector.f_max
        db_lo, db_hi = -100.0, 0.0

        def X(f):
            return plot.left + f / fmax * plot.w

        def Y(db):
            return plot.bottom - (min(db_hi, max(db_lo, db)) - db_lo) / (db_hi - db_lo) * plot.h

        for db in range(-100, 1, 20):                       # grille horizontale
            y = int(Y(db))
            pygame.draw.line(s, LINE, (plot.left, y), (plot.right, y))
            self.text(s, f"{db}", 12, MUTED, (plot.left - 8, y), "midright")
        for f in range(0, int(fmax) + 1, 200):              # grille verticale
            x = int(X(f))
            pygame.draw.line(s, LINE, (x, plot.top), (x, plot.bottom))
            self.text(s, f"{f}", 12, MUTED, (x, plot.bottom + 6), "midtop")
        self.text(s, "Hz", 12, MUTED, (plot.right, plot.bottom + 20), "topright")
        
        r = self.res
        if self.spec is not None and len(self.spec):
            fr = self.engine.detector.dft.freqs
            pts = [(X(f), Y(d)) for f, d in zip(fr, self.spec)]
            area = [(pts[0][0], plot.bottom)] + pts + [(pts[-1][0], plot.bottom)]
            layer = pygame.Surface(plot.size, pygame.SRCALPHA)
            pygame.draw.polygon(layer, (*ACCENT, 46),
                                [(x - plot.left, y - plot.top) for x, y in area])
            s.blit(layer, plot.topleft)
            pygame.draw.lines(s, ACCENT, False, pts, 2)

            if r.voiced:                                     # f0 et harmoniques
                col = status_color(r.cents)
                x0 = int(X(r.f0))
                pygame.draw.line(s, col, (x0, plot.top), (x0, plot.bottom), 2)
                self.text(s, f"f0 = {r.f0:.2f} Hz", 14, col,
                          (min(x0 + 8, plot.right - 110), plot.top + 4), "topleft", bold=True)
                h = 2
                while r.f0 * h <= fmax and h <= 6:
                    xh = int(X(r.f0 * h))
                    pygame.draw.line(s, MUTED, (xh, plot.top + 26), (xh, plot.bottom), 1)
                    self.text(s, f"{h}f0", 12, MUTED, (xh + 4, plot.top + 28), "topleft")
                    h += 1

    # ---- pied de page ------------------------------------------------------
    def draw_footer(self, s):
        mouse = pygame.mouse.get_pos()
        for b in self.buttons:
            hover = b.rect.collidepoint(mouse)
            pygame.draw.rect(s, mix(CARD, ACCENT, 0.25) if hover else CARD, b.rect, border_radius=9)
            pygame.draw.rect(s, LINE, b.rect, 1, border_radius=9)
            self.text(s, b.label_fn(), 15, TEXT, b.rect.center, "center")
        self.text(s, f"La4 = {self.engine.a4:.0f} Hz", 16, TEXT, (600, 682), "midleft", bold=True)
        self.text(s, "Raccourcis :  Haut / Bas = La4    N = notation    Espace = pause    Échap = quitter",
                  13, MUTED, (W // 2, 722), "center")


# ----------------------------------------------------------------------------
def run(source, a4=440.0, notation="fr"):
    pygame.init()
    pygame.display.set_caption("Accordeur chromatique — DFT")
    screen = pygame.display.set_mode((W, H))
    app = TunerApp(source, a4, notation)
    source.start()
    clock = pygame.time.Clock()
    running = True
    try:
        while running:
            for e in pygame.event.get():
                running = app.handle_event(e) and running
            app.update()
            app.draw(screen)
            pygame.display.flip()
            clock.tick(30)
    finally:
        source.stop()
        pygame.quit()
