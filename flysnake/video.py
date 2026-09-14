"""H.264 video writer (ffmpeg pipe) and a PNG helper.

Example::

    with VideoWriter("out/demo.mp4", fps=30, size=(1080, 1920)) as vw:
        for frame in frames:          # np.uint8 (H, W, 3)
            vw.write(frame)

The writer streams raw RGB frames to an ``ffmpeg`` subprocess and encodes
``libx264`` / ``yuv420p`` with ``+faststart`` so the file plays in browsers
and on LinkedIn/X.  ffmpeg is looked up in this order: explicit argument,
``FLYSNAKE_FFMPEG`` env var, ``/opt/homebrew/bin/ffmpeg``, ``PATH``, then
the binary bundled with ``imageio-ffmpeg``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

_DEFAULT_FFMPEG = "/opt/homebrew/bin/ffmpeg"


def find_ffmpeg(explicit: str | None = None) -> str:
    """Locate an ffmpeg executable (see module docstring for the order)."""
    candidates = [explicit, os.environ.get("FLYSNAKE_FFMPEG"), _DEFAULT_FFMPEG, shutil.which("ffmpeg")]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - only when nothing is installed
        raise RuntimeError("ffmpeg not found; install it or `uv add imageio-ffmpeg`") from exc


class VideoWriter:
    """Stream RGB frames into an H.264 MP4.

    Parameters
    ----------
    path:
        Output file (parent directories are created).
    fps:
        Frames per second.
    size:
        ``(width, height)``; both must be even for ``yuv420p``.
    crf:
        x264 quality (lower = better, 18 is visually lossless).
    preset:
        x264 speed preset.
    ffmpeg:
        Explicit path to the ffmpeg binary.
    """

    def __init__(
        self,
        path: str | Path,
        fps: int = 30,
        size: tuple[int, int] = (1080, 1920),
        crf: int = 18,
        preset: str = "medium",
        ffmpeg: str | None = None,
    ) -> None:
        w, h = size
        if w % 2 or h % 2:
            raise ValueError(f"size must be even for yuv420p, got {size}")
        self.path = Path(path)
        self.fps = fps
        self.size = (w, h)
        self.frames_written = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            find_ffmpeg(ffmpeg),
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            "-preset", preset,
            "-movflags", "+faststart",
            str(self.path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        """Append one ``(H, W, 3)`` uint8 RGB frame."""
        frame = np.ascontiguousarray(frame)
        w, h = self.size
        if frame.shape != (h, w, 3) or frame.dtype != np.uint8:
            raise ValueError(f"expected uint8 frame of shape {(h, w, 3)}, got {frame.shape} {frame.dtype}")
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame.tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(self._collect_error()) from exc
        self.frames_written += 1

    def close(self) -> None:
        """Flush and wait for ffmpeg; raises if encoding failed."""
        if self._proc.stdin is None or self._proc.stdin.closed:
            return
        self._proc.stdin.close()
        ret = self._proc.wait()
        if ret != 0:
            raise RuntimeError(f"ffmpeg exited with {ret}: {self._collect_error()}")

    def _collect_error(self) -> str:
        if self._proc.stderr is None:
            return ""
        return self._proc.stderr.read().decode(errors="replace").strip()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def save_png(frame: np.ndarray, path: str | Path) -> Path:
    """Write an RGB (or 2-D grayscale) uint8 array as PNG; returns the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = (arr.astype(np.uint8) * (255 if arr.max() <= 1 else 1)).astype(np.uint8)
    Image.fromarray(arr).save(path)
    return path
