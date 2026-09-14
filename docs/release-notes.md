# Flybrain Snake 1.0.1 — horizontal video

The demo now uses a native **1920 × 1080, 16:9** composition. The complete Nokia LCD sits on the left; scores, the spike raster, and motor activity sit on the right. Captions and the repository link remain inside the frame without cropping or stretching.

The complete recorded run is unchanged: seed 10, four food items, 16 points, and a collision after 99 steps. Runtime is 31.4 seconds at 30 fps with English captions and original synthesized sound, encoded as H.264/AAC.

Reproduce it with:

```bash
uv run python -m scripts.render_fly_episode --seed 10 --frames-per-step 8 --layout landscape
```

Landscape is now the default video layout. The earlier composition is available with `--layout portrait`; its original video remains attached to v1.0.0.

Code and generated artwork are MIT licensed. Derived MaleCNS v1.0 data is CC BY 4.0 with attribution in THIRD_PARTY_DATA.md.
