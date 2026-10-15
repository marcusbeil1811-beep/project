# Mood-Based 3D Audio Visualizer

A Python app that plays any music file you give it and renders a rotating,
pulsing 3D shape in a window. The shape's **color**, **rotation speed**, and
**intensity** (size/brightness) react to the music's bass, mid, and treble
energy, overall loudness, and tempo (BPM).

- **Color** shifts blue → red depending on whether the music is bass-heavy
  (calmer, cooler colors) or treble-heavy (energetic, warmer colors).
- **Speed** of rotation scales with the song's detected tempo (BPM), and
  spikes briefly on strong beats/onsets.
- **Intensity** (shape scale + brightness) pulses with bass hits and overall
  loudness (RMS energy).

This is a **desktop app** — it opens a real OpenGL window, so it needs to run
on your own computer (Windows/Mac/Linux), not in a browser or sandbox.

## 1. Install Python

You need Python 3.9–3.12 installed. Check with:

```bash
python3 --version
```

If you don't have it, get it from https://www.python.org/downloads/

## 2. Install the required libraries

From the folder containing `requirements.txt`, run:

```bash
pip install -r requirements.txt
```

This installs:

| Library | What it's for |
|---|---|
| `numpy` | Numerical array processing |
| `librosa` | Audio analysis — frequency bands, RMS loudness, tempo/beat detection |
| `soundfile` | Backend librosa uses to read audio files (mp3/wav/flac/etc.) |
| `pygame` | Creates the window, handles audio playback, keyboard input |
| `PyOpenGL` / `PyOpenGL_accelerate` | 3D rendering (the actual OpenGL graphics) |

### Notes by OS

- **macOS**: If `librosa`/`soundfile` complain about missing `libsndfile`,
  install it via Homebrew: `brew install libsndfile`
- **Linux**: You may need `sudo apt install libsndfile1 python3-opengl` and,
  if pygame's window fails to open, `sudo apt install libgl1-mesa-glx`
- **Windows**: Everything above installs via pip with no extra system
  packages needed in most cases.
- **MP3 support**: `librosa`/`soundfile` sometimes need `ffmpeg` installed
  system-wide to decode MP3s reliably. If you hit a loading error, install
  ffmpeg (`brew install ffmpeg` / `sudo apt install ffmpeg` / download from
  ffmpeg.org for Windows and add it to PATH), or just use a `.wav` file.

## 3. Run it

```bash
python audio_visualizer.py path/to/your/song.mp3
```

Replace the path with any audio file — mp3, wav, flac, ogg, etc.

The first run will take a few seconds to a couple of minutes (depending on
song length) while it analyzes the full track's frequencies and tempo before
the window opens — this is normal, it's precomputing the reactive data.

### Controls
- `SPACE` — pause / resume
- `ESC` — quit
- `F` — toggle fullscreen

## How it works (brief)

1. **Analysis pass**: the whole song is loaded and run through a
   Short-Time Fourier Transform (STFT). Energy is summed into three bands
   (bass 20–250Hz, mid 250–2000Hz, treble 2000–8000Hz) for every video
   frame's worth of audio, plus overall RMS loudness and onset strength
   (used for beat pulses). `librosa.beat.beat_track` estimates the song's
   tempo in BPM.
2. **Playback**: `pygame.mixer` plays the actual audio.
3. **Render loop**: every frame, the app looks up "what are the bass/mid/
   treble/loudness values right now" based on the audio's current playback
   position, smooths them slightly to avoid jitter, and uses them to set
   the 3D shape's hue/saturation/brightness, rotation speed, and scale.

## Customizing

Some easy things to tweak in `audio_visualizer.py`:
- `smoothing` in `_update()` — how quickly visuals react (higher = snappier).
- The frequency band ranges in `_analyze_bands()` (e.g. widen "bass" to 20–300Hz).
- Swap `make_icosahedron()` for a different shape, or add more nested shapes.
- The hue mapping formula in `_render()` if you want a different color mood
  (e.g. always green-to-purple instead of blue-to-red).
