"""End-to-end rendering check: GreedyAgent plays, dummy HUD, 10 s 9:16 MP4.

Outputs ``out/demo_greedy.mp4`` and ``out/demo_still.png``.  The eye,
raster and descending-neuron panels are fed with random (but smoothly
varying) data so the whole pipeline can be validated before the brain
module is wired in.

    uv run python scripts/demo_baselines.py [--seconds 10] [--fps 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flysnake.baselines import GreedyAgent  # noqa: E402
from flysnake.game import SnakeGame  # noqa: E402
from flysnake.render import HudComposer, hex_lattice_points  # noqa: E402
from flysnake.video import VideoWriter, save_png  # noqa: E402

DN_LABELS = ["DNa02", "DNa01", "DNp09", "DNb01", "DNg13", "DNp02"]


class DummyBrainFeed:
    """Random but temporally smooth stand-in for the brain module's outputs."""

    def __init__(self, seed: int = 0, n_neurons: int = 400, window: int = 60) -> None:
        self.rng = np.random.default_rng(seed)
        self.points = hex_lattice_points(rows=14, cols=18)
        n = len(self.points)
        self.act_l = self.rng.random(n)
        self.act_r = self.rng.random(n)
        self.dn = self.rng.random((len(DN_LABELS), 2))
        self.raster = self.rng.random((n_neurons, window)) < 0.03

    def tick(self, heading: tuple[int, int]) -> None:
        # Drift the eye activity, with a bias so a turn "lights up" one eye.
        bias_l = 0.15 if heading[1] < 0 else 0.0
        bias_r = 0.15 if heading[1] > 0 else 0.0
        self.act_l = np.clip(0.85 * self.act_l + 0.15 * (self.rng.random(len(self.act_l)) + bias_l), 0, 1)
        self.act_r = np.clip(0.85 * self.act_r + 0.15 * (self.rng.random(len(self.act_r)) + bias_r), 0, 1)
        self.dn = np.clip(0.8 * self.dn + 0.2 * self.rng.random(self.dn.shape), 0, 1)
        new_col = (self.rng.random((self.raster.shape[0], 1)) < 0.03)
        self.raster = np.concatenate([self.raster[:, 1:], new_col], axis=1)

    def dn_bars(self) -> list[tuple[str, float, float]]:
        return [(lbl, float(l), float(r)) for lbl, (l, r) in zip(DN_LABELS, self.dn)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Render a 9:16 demo with a greedy agent and synthetic HUD.")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--steps-per-second", type=float, default=8.0, help="game steps per second")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("out"))
    args = p.parse_args(argv)

    game = SnakeGame(seed=args.seed)
    agent = GreedyAgent()
    feed = DummyBrainFeed(seed=args.seed)
    hud = HudComposer(raster_window=60)

    n_frames = int(args.seconds * args.fps)
    frames_per_step = max(1, int(round(args.fps / args.steps_per_second)))
    mp4 = args.out / "demo_greedy.mp4"
    png = args.out / "demo_still.png"
    still: np.ndarray | None = None

    obs = game.observe()
    t0 = time.perf_counter()
    with VideoWriter(mp4, fps=args.fps, size=(hud.width, hud.height)) as vw:
        for i in range(n_frames):
            if i % frames_per_step == 0:
                if not obs["alive"]:
                    obs = game.reset()
                obs = game.step(agent.act(obs))
                feed.tick(obs["heading"])
            stats = {"Score": obs["score"], "Neurons": 10_842, "Synapses": 1_230_557, "Steps": obs["t"]}
            frame = hud.compose(
                obs["frame"],
                eye_left=(feed.points, feed.act_l),
                eye_right=(feed.points, feed.act_r),
                raster=feed.raster,
                dn_bars=feed.dn_bars(),
                stats=stats,
                caption="GREEDY DEMO / synthetic neuron activity",
            )
            vw.write(frame)
            if still is None and obs["score"] >= 3:
                still = frame
    if still is None:
        still = frame
    save_png(still, png)
    dt = time.perf_counter() - t0
    print(f"{mp4} ({vw.frames_written} frames, {dt:.1f} s, {vw.frames_written / dt:.1f} frames/s)")
    print(f"{png}")


if __name__ == "__main__":
    main()
