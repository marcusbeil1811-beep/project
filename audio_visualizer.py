"""
Mood-Based 3D Audio Visualizer
=================================================================

Run:
    python audio_visualizer.py path/to/song.mp3

Controls while running:
    SPACE  - pause / resume
    ESC    - quit
    F      - toggle fullscreen
"""

import sys
import math
import colorsys
import numpy as np

import pygame
from pygame.locals import DOUBLEBUF, OPENGL, FULLSCREEN

from OpenGL.GL import *
from OpenGL.GLU import *

import librosa


# --------------------------------------------------------------------------
# 1. AUDIO ANALYSIS
# --------------------------------------------------------------------------
class AudioAnalyzer:
    """
    Loads an audio file once and pre-computes a per-video-frame feature
    track (bass / mid / treble energy, overall RMS, onset strength, and
    a global tempo estimate). The visualizer then just looks up the
    feature values for "now" based on playback position.
    """

    def __init__(self, filepath, fps=60):
        self.fps = fps
        print(f"Loading audio: {filepath}")
        self.y, self.sr = librosa.load(filepath, sr=None, mono=True)
        self.duration = librosa.get_duration(y=self.y, sr=self.sr)

        self.hop_length = max(1, int(self.sr / fps))

        print("Analyzing frequency bands...")
        self._analyze_bands()

        print("Detecting tempo & beats...")
        self._analyze_tempo()

        print(f"Analysis complete. Duration: {self.duration:.1f}s, "
              f"Tempo: {self.tempo:.1f} BPM")

    def _analyze_bands(self):
        S = np.abs(librosa.stft(self.y, hop_length=self.hop_length, n_fft=2048))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=2048)

        def band_energy(lo, hi):
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            if len(idx) == 0:
                return np.zeros(S.shape[1])
            return S[idx, :].mean(axis=0)

        bass = band_energy(20, 250)
        mid = band_energy(250, 2000)
        treble = band_energy(2000, 8000)

        def norm(x):
            peak = np.percentile(x, 95) + 1e-6
            return np.clip(x / peak, 0, 1)

        self.bass = norm(bass)
        self.mid = norm(mid)
        self.treble = norm(treble)

        rms = librosa.feature.rms(y=self.y, hop_length=self.hop_length)[0]
        self.rms = norm(rms)

        onset_env = librosa.onset.onset_strength(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        self.onset = norm(onset_env)

        self.n_frames = len(self.bass)

    def _analyze_tempo(self):
        tempo, _ = librosa.beat.beat_track(
            y=self.y, sr=self.sr, hop_length=self.hop_length
        )
        self.tempo = float(np.atleast_1d(tempo)[0]) if tempo else 120.0

    def features_at(self, t_seconds):
        """Return (bass, mid, treble, rms, onset) at playback time t."""
        idx = int(t_seconds * self.fps)
        idx = min(max(idx, 0), self.n_frames - 1)
        return (
            float(self.bass[idx]),
            float(self.mid[idx]),
            float(self.treble[idx]),
            float(self.rms[idx]),
            float(self.onset[idx]),
        )


# --------------------------------------------------------------------------
# 2. GEOMETRY
# --------------------------------------------------------------------------
def make_cube():
    verts = [
        (-1, -1, -1), (1, -1, -1), (1, 1, -1), (-1, 1, -1),
        (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),
    ]
    verts = [np.array(v, dtype=np.float64) for v in verts]

    faces = [
        (0, 1, 2), (0, 2, 3),  # back
        (4, 6, 5), (4, 7, 6),  # front
        (0, 4, 5), (0, 5, 1),  # bottom
        (3, 2, 6), (3, 6, 7),  # top
        (0, 3, 7), (0, 7, 4),  # left
        (1, 5, 6), (1, 6, 2),  # right
    ]
    return verts, faces


# --------------------------------------------------------------------------
# 3. VISUALIZER
# --------------------------------------------------------------------------
class Visualizer:
    def __init__(self, analyzer: AudioAnalyzer, audio_path: str, width=1000, height=720):
        self.analyzer = analyzer
        self.width, self.height = width, height
        self.paused = False
        self.fullscreen = False

        pygame.init()
        pygame.mixer.init(frequency=analyzer.sr)
        self._set_display_mode()

        pygame.display.set_caption("Mood-Based 3D Audio Visualizer")

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self.base_verts, self.faces = make_cube()

        self.smooth_bass = 0.0
        self.smooth_mid = 0.0
        self.smooth_treble = 0.0
        self.smooth_rms = 0.0
        self.pulse = 0.0

        # Slow-blending color state, so hue/saturation/brightness drift
        # gently rather than snapping with every audio frame.
        self.smooth_hue = 0.55
        self.smooth_sat = 0.55
        self.smooth_val = 0.55

        self.rotation = [0.0, 0.0, 0.0]

        pygame.mixer.music.load(audio_path)

    def _set_display_mode(self):
        flags = DOUBLEBUF | OPENGL | (FULLSCREEN if self.fullscreen else 0)
        pygame.display.set_mode((self.width, self.height), flags)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, self.width / self.height, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    def run(self):
        clock = pygame.time.Clock()
        pygame.mixer.music.play()

        running = True
        started = False
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self._toggle_pause()
                    elif event.key == pygame.K_f:
                        self.fullscreen = not self.fullscreen
                        self._set_display_mode()

            if pygame.mixer.music.get_busy():
                started = True

            if started and not pygame.mixer.music.get_busy() and not self.paused:
                running = False
                continue

            t = pygame.mixer.music.get_pos() / 1000.0
            if not self.paused:
                self._update(t)
                self._render()

            clock.tick(self.analyzer.fps)

        pygame.quit()

    def _toggle_pause(self):
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.music.pause()
        else:
            pygame.mixer.music.unpause()

    # ---- feature smoothing -------------------------------------------------
    def _update(self, t):
        bass, mid, treble, rms, onset = self.analyzer.features_at(t)

        smoothing = 0.35
        self.smooth_bass += (bass - self.smooth_bass) * smoothing
        self.smooth_mid += (mid - self.smooth_mid) * smoothing
        self.smooth_treble += (treble - self.smooth_treble) * smoothing
        self.smooth_rms += (rms - self.smooth_rms) * smoothing

        if onset > 0.55:
            self.pulse = min(1.0, self.pulse + onset)
        self.pulse *= 0.85

        base_speed = self.analyzer.tempo / 120.0
        speed_boost = 1.0 + self.smooth_treble * 0.7 + self.pulse * 0.9
        dt = 1.0 / self.analyzer.fps
        self.rotation[0] += 9 * base_speed * speed_boost * dt
        self.rotation[1] += 14 * base_speed * speed_boost * dt
        self.rotation[2] += 5 * base_speed * speed_boost * dt

    # ---- drawing ------------------------------------------------------------
    def _render(self):
        # Clean white background
        glClearColor(1.0, 1.0, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        gluLookAt(0, 0, 6, 0, 0, 0, 0, 1, 0)

        glRotatef(self.rotation[0], 1, 0, 0)
        glRotatef(self.rotation[1], 0, 1, 0)
        glRotatef(self.rotation[2], 0, 0, 1)

        # COLOR: hue drifts based on bass/treble balance, blended slowly.
        # Kept moderately saturated/darker so it stays visible against
        # the white background (very light colors would wash out).
        balance = (self.smooth_treble - self.smooth_bass + 1.0) / 2.0  # 0..1
        target_hue = (0.62 - balance * 0.5) % 1.0
        target_sat = 0.55 + self.smooth_mid * 0.3
        target_val = 0.45 + self.smooth_rms * 0.3 + self.pulse * 0.15

        color_smoothing = 0.06
        self.smooth_hue += (target_hue - self.smooth_hue) * color_smoothing
        self.smooth_sat += (target_sat - self.smooth_sat) * color_smoothing
        self.smooth_val += (target_val - self.smooth_val) * color_smoothing

        r, g, b = colorsys.hsv_to_rgb(
            self.smooth_hue, min(self.smooth_sat, 1.0), min(self.smooth_val, 0.85)
        )

        scale = 1.0 + self.smooth_bass * 0.55 + self.pulse * 0.3

        self._draw_shape(scale, (r, g, b), alpha=0.9, wireframe=True)
        core_scale = scale * (0.35 + self.smooth_treble * 0.25)
        self._draw_shape(
            core_scale,
            (min(r + 0.15, 1), min(g + 0.15, 1), min(b + 0.15, 1)),
            alpha=1.0,
            wireframe=False,
        )

        pygame.display.flip()

    def _draw_shape(self, scale, color, alpha=0.9, wireframe=True):
        r, g, b = color
        glPushMatrix()
        glScalef(scale, scale, scale)

        if wireframe:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glLineWidth(2.5)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glColor4f(r, g, b, alpha)
        glBegin(GL_TRIANGLES)
        for face in self.faces:
            for idx in face:
                v = self.base_verts[idx]
                glVertex3f(*v)
        glEnd()

        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glPopMatrix()


# --------------------------------------------------------------------------
# 4. ENTRY POINT
# --------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_visualizer.py <path_to_audio_file>")
        sys.exit(1)

    audio_path = sys.argv[1]
    analyzer = AudioAnalyzer(audio_path, fps=60)
    viz = Visualizer(analyzer, audio_path)
    viz.run()


if __name__ == "__main__":
    main()
