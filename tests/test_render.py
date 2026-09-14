"""Smoke tests for the Nokia renderer, HUD composer and video writer."""

import numpy as np

from flysnake.game import HEIGHT_PX, WIDTH_PX, SnakeGame
from flysnake.render import LCD_BG, LCD_FG, HudComposer, NokiaRenderer, hex_lattice_points
from flysnake.video import VideoWriter, save_png


def test_nokia_renderer_shape_and_colors():
    frame = SnakeGame(seed=0).observe()["frame"]
    r = NokiaRenderer(scale=10)
    img = r.render(frame)
    assert img.shape == (HEIGHT_PX * 10, WIDTH_PX * 10, 3)
    assert img.dtype == np.uint8
    # A dark LCD pixel's centre is the foreground colour, an off pixel's is (almost) background.
    ys, xs = np.nonzero(frame)
    y, x = ys[0], xs[0]
    assert tuple(img[y * 10 + 5, x * 10 + 5]) == LCD_FG
    # Grid lines are the pure background colour.
    assert tuple(img[0, 5]) == LCD_BG
    # Bezel grows the image.
    assert NokiaRenderer(scale=4, bezel=8).render(frame).shape == (HEIGHT_PX * 4 + 16, WIDTH_PX * 4 + 16, 3)


def test_hud_compose_full_and_empty():
    frame = SnakeGame(seed=0).observe()["frame"]
    hud = HudComposer(raster_window=50)
    pts = hex_lattice_points(rows=6, cols=8)
    act = np.linspace(0, 1, len(pts))
    raster = np.random.default_rng(0).random((120, 200)) < 0.05
    img = hud.compose(
        frame,
        eye_left=(pts, act),
        eye_right={"points": pts, "activity": act[::-1]},
        raster=raster,
        dn_bars=[("DNa02", 0.2, 0.9), ("DNp09", 0.5, 0.1)],
        stats={"Score": 3, "Neurons": 10842, "Synapses": 1230557, "Steps": 42},
        caption="test",
    )
    assert img.shape == (1920, 1080, 3) and img.dtype == np.uint8
    # Panels with no data must not crash.
    img2 = hud.compose(frame)
    assert img2.shape == (1920, 1080, 3)
    assert not np.array_equal(img, img2)


def test_video_writer_and_png(tmp_path):
    hud = HudComposer(width=540, height=960, screen_scale=5)
    frame = SnakeGame(seed=0).observe()["frame"]
    img = hud.compose(frame, caption="x")
    path = tmp_path / "t.mp4"
    with VideoWriter(path, fps=10, size=(540, 960)) as vw:
        for _ in range(5):
            vw.write(img)
    assert path.exists() and path.stat().st_size > 1000
    assert vw.frames_written == 5
    png = save_png(img, tmp_path / "t.png")
    assert png.exists()
