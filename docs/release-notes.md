# Flybrain Snake 1.0

3,189 simulated neurons from a fruit-fly connectome control a recreation of Nokia Snake II.

- Play the game with a Nokia-style LCD and handset, wraparound, five maze layouts, nine speed levels, bonus creatures and retro sound.
- Watch the real connectome subcircuit control it with `uv run python -m flysnake.play --agent fly --seed 10`.
- Reproduce the 31.4-second video and every recorded decision from the included source and compact derived data.
- Read the biological provenance and all visual/readout assumptions in the English documentation.

The attached MP4 is 1080 × 1350 at 30 fps, encoded as H.264/AAC. It shows seed 10 collecting four food items for 16 points, then crashing after 99 steps. The complete run is shown with English captions and original synthesized sound. It was selected from 12 seeds; selection results and full telemetry are included in the repository.

The recreation is independent of Nokia. Maze layouts, speed timings and bonus rules are approximations; firmware-exact behavior is not claimed.

Code and generated artwork are MIT licensed. The derived MaleCNS v1.0 connectome is CC BY 4.0 with attribution in THIRD_PARTY_DATA.md.
