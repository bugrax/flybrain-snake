"""Nokia 3310 LCD renderer and the 9:16 HUD composer.

Two public classes:

* :class:`NokiaRenderer` -- turns a 48x84 ``uint8`` frame (1 = dark pixel)
  into an RGB image with the LCD green-grey background, dark pixels, a
  subtle pixel grid and an optional bezel.
* :class:`HudComposer` -- lays out a 1080x1920 vertical frame: Nokia screen
  on top, two compound-eye panels, a spike raster, descending-neuron bars,
  a stats strip and a caption.  All labels are English.

Colours are RGB tuples; images are ``np.uint8`` arrays of shape (H, W, 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from flysnake.game import HEIGHT_PX, WIDTH_PX

RGB = tuple[int, int, int]

# --- palette -----------------------------------------------------------------
LCD_BG: RGB = (196, 211, 165)  # #C4D3A5
LCD_FG: RGB = (30, 43, 26)  # #1E2B1A
HUD_BG: RGB = (14, 15, 17)
HUD_PANEL: RGB = (22, 24, 27)
HUD_LINE: RGB = (48, 52, 58)
HUD_TEXT: RGB = (214, 218, 210)
HUD_MUTED: RGB = (128, 134, 128)
HUD_ACCENT: RGB = (196, 211, 165)  # echo of the LCD colour
EYE_DARK: RGB = (28, 32, 28)
EYE_BRIGHT: RGB = (205, 255, 90)
BAR_LEFT: RGB = (120, 190, 255)
BAR_RIGHT: RGB = (255, 170, 90)
SPIKE: RGB = (205, 255, 90)

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)


@lru_cache(maxsize=32)
def load_mono_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a monospace font, falling back to Pillow's built-in one."""
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _lerp(a: RGB, b: RGB, t: float) -> RGB:
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(int(round(a[i] + (b[i] - a[i]) * t)) for i in range(3))  # type: ignore[return-value]


def _lerp_array(a: RGB, b: RGB, t: np.ndarray) -> np.ndarray:
    """Vectorised colour interpolation; ``t`` any shape → (*t.shape, 3) uint8."""
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)[..., None]
    return (np.asarray(a, np.float32) + (np.asarray(b, np.float32) - np.asarray(a, np.float32)) * t).astype(
        np.uint8
    )


# ============================================================ Nokia renderer
class NokiaRenderer:
    """Render a monochrome 48x84 frame as a Nokia 3310 LCD.

    Parameters
    ----------
    scale:
        Screen pixels per LCD pixel.
    bg, fg:
        LCD background and dark-pixel colours.
    grid:
        Draw the 1-px pixel grid (the gaps between LCD pixels).
    ghost:
        Strength (0..1) of the faint "off pixel" tint that real LCDs show.
    bezel:
        Width in screen pixels of a dark rounded bezel around the screen;
        ``0`` disables it.
    """

    def __init__(
        self,
        scale: int = 10,
        bg: RGB = LCD_BG,
        fg: RGB = LCD_FG,
        grid: bool = True,
        ghost: float = 0.05,
        bezel: int = 0,
        bezel_color: RGB = (34, 36, 40),
    ) -> None:
        if scale < 2:
            raise ValueError("scale must be >= 2 for the pixel grid to exist")
        self.scale = scale
        self.bg = bg
        self.fg = fg
        self.grid = grid
        self.ghost = ghost
        self.bezel = bezel
        self.bezel_color = bezel_color

    @property
    def screen_size(self) -> tuple[int, int]:
        """(width, height) of the rendered screen without bezel."""
        return WIDTH_PX * self.scale, HEIGHT_PX * self.scale

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the full output including bezel."""
        w, h = self.screen_size
        return w + 2 * self.bezel, h + 2 * self.bezel

    def render(self, frame: np.ndarray) -> np.ndarray:
        """Return an RGB uint8 image of shape (H*scale [+bezel], W*scale [+bezel], 3)."""
        frame = np.asarray(frame)
        if frame.shape != (HEIGHT_PX, WIDTH_PX):
            raise ValueError(f"expected frame of shape {(HEIGHT_PX, WIDTH_PX)}, got {frame.shape}")
        s = self.scale
        on = frame.astype(bool)

        # Base: off pixels get a faint ghost tint, on pixels the dark colour.
        ghost_col = _lerp(self.bg, self.fg, self.ghost)
        img = np.empty((HEIGHT_PX, WIDTH_PX, 3), dtype=np.uint8)
        img[...] = ghost_col
        img[on] = self.fg
        img = np.repeat(np.repeat(img, s, axis=0), s, axis=1)

        if self.grid:
            # The 1-px gap between LCD pixels shows the bare background.
            img[::s, :, :] = self.bg
            img[:, ::s, :] = self.bg

        if self.bezel > 0:
            img = self._add_bezel(img)
        return img

    def _add_bezel(self, screen: np.ndarray) -> np.ndarray:
        b = self.bezel
        h, w = screen.shape[:2]
        canvas = Image.new("RGB", (w + 2 * b, h + 2 * b), self.bezel_color)
        draw = ImageDraw.Draw(canvas)
        # Slight inner lip so the screen reads as recessed.
        lip = _lerp(self.bezel_color, (0, 0, 0), 0.5)
        draw.rounded_rectangle((b - 3, b - 3, b + w + 2, b + h + 2), radius=6, fill=lip)
        canvas.paste(Image.fromarray(screen), (b, b))
        return np.asarray(canvas)


# ============================================================ HUD composer
@dataclass
class EyeData:
    """Compound-eye activity for one eye.

    ``points`` is an (N, 2) float array of (azimuth_deg, elevation_deg) per
    neuron and ``activity`` an (N,) array in 0..1.
    """

    points: np.ndarray
    activity: np.ndarray

    @classmethod
    def coerce(cls, value: "EyeData | tuple | dict | None") -> "EyeData | None":
        if value is None or isinstance(value, EyeData):
            return value
        if isinstance(value, dict):
            return cls(np.asarray(value["points"], float), np.asarray(value["activity"], float))
        points, activity = value
        return cls(np.asarray(points, float), np.asarray(activity, float))


def hex_lattice_points(rows: int = 14, cols: int = 18, az_span: float = 150.0) -> np.ndarray:
    """Synthetic ommatidia layout: a honeycomb lattice in (azimuth, elevation) degrees.

    Columns are spaced ``az_span / cols`` apart; rows use the honeycomb pitch
    (sqrt(3)/2 of that) with every other row offset by half a step.  Handy for
    demos and for testing the eye panels before real coordinates from the
    connectome are available.
    """
    dx = az_span / cols
    dy = dx * np.sqrt(3) / 2
    el_span = (rows - 1) * dy
    pts = []
    for r in range(rows):
        for c in range(cols):
            az = -az_span / 2 + (c + 0.5 * (r % 2)) * dx
            el = el_span / 2 - r * dy
            pts.append((az, el))
    return np.asarray(pts, dtype=float)


def _median_nn_distance(xy: np.ndarray) -> float:
    """Median nearest-neighbour distance (subsampled brute force, no scipy)."""
    n = len(xy)
    if n < 2:
        return 20.0
    idx = np.arange(n) if n <= 600 else np.random.default_rng(0).choice(n, 600, replace=False)
    sub = xy[idx]
    d = np.sqrt(((sub[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    d[np.arange(len(idx)), idx] = np.inf
    return float(np.median(d.min(axis=1)))


class HudComposer:
    """Compose the 1080x1920 vertical HUD frame.

    Layout (top to bottom): header, Nokia screen (hero), left/right compound
    eyes, spike raster, descending-neuron bars, stats strip, caption.

    Parameters
    ----------
    width, height:
        Output size; defaults to 9:16 1080x1920.
    title:
        Small header text above the screen.
    raster_window:
        If set, only the last ``raster_window`` columns of the raster are
        drawn (e.g. the number of simulation steps in one second).
    """

    def __init__(
        self,
        width: int = 1080,
        height: int = 1920,
        title: str = "FLYBRAIN / SNAKE II",
        raster_window: int | None = None,
        screen_scale: int = 11,
    ) -> None:
        self.width = width
        self.height = height
        self.title = title
        self.raster_window = raster_window
        self.nokia = NokiaRenderer(scale=screen_scale, bezel=28)
        self._layout()

    # ----------------------------------------------------------- geometry
    def _layout(self) -> None:
        W, H = self.width, self.height
        m = int(W * 0.04)  # outer margin
        self.margin = m
        sw, sh = self.nokia.size
        self.header_box = (m, int(H * 0.030), W - m, int(H * 0.030) + 40)
        top = self.header_box[3] + 12
        self.screen_pos = ((W - sw) // 2, top)
        y = top + sh + int(H * 0.02)
        eye_h = int(H * 0.20)
        gap = int(W * 0.02)
        half = (W - 2 * m - gap) // 2
        self.eye_left_box = (m, y, m + half, y + eye_h)
        self.eye_right_box = (m + half + gap, y, W - m, y + eye_h)
        y += eye_h + int(H * 0.015)
        raster_h = int(H * 0.13)
        self.raster_box = (m, y, W - m, y + raster_h)
        y += raster_h + int(H * 0.015)
        bars_h = int(H * 0.115)
        self.bars_box = (m, y, W - m, y + bars_h)
        y += bars_h + int(H * 0.015)
        stats_h = int(H * 0.06)
        self.stats_box = (m, y, W - m, y + stats_h)
        self.caption_box = (m, H - int(H * 0.055), W - m, H - int(H * 0.02))

    # ------------------------------------------------------------ helpers
    @staticmethod
    def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str | None = None) -> tuple[int, int, int, int]:
        """Draw a panel background + label; return the inner content box."""
        draw.rounded_rectangle(box, radius=10, fill=HUD_PANEL, outline=HUD_LINE, width=1)
        x0, y0, x1, y1 = box
        pad = 14
        if label:
            f = load_mono_font(20)
            draw.text((x0 + pad, y0 + 10), label, font=f, fill=HUD_MUTED)
            y0 += 36
        return (x0 + pad, y0 + pad // 2, x1 - pad, y1 - pad)

    def _draw_eye(self, img: Image.Image, draw: ImageDraw.ImageDraw, box, eye: EyeData | None, label: str) -> None:
        inner = self._panel(draw, box, label)
        if eye is None or len(eye.points) == 0:
            f = load_mono_font(18)
            draw.text(((inner[0] + inner[2]) // 2, (inner[1] + inner[3]) // 2), "no data", font=f, fill=HUD_MUTED, anchor="mm")
            return
        pts = np.asarray(eye.points, float)
        act = np.clip(np.asarray(eye.activity, float), 0, 1)
        x0, y0, x1, y1 = inner
        pad = 12
        az, el = pts[:, 0], pts[:, 1]
        az_min, az_max = float(az.min()), float(az.max())
        el_min, el_max = float(el.min()), float(el.max())
        az_span = max(az_max - az_min, 1e-6)
        el_span = max(el_max - el_min, 1e-6)
        # Uniform scale (deg → px) that fits the panel, keeping aspect ratio.
        k = min((x1 - x0 - 2 * pad) / az_span, (y1 - y0 - 2 * pad) / el_span)
        cx = (x0 + x1) / 2 - k * (az_min + az_max) / 2
        cy = (y0 + y1) / 2 + k * (el_min + el_max) / 2
        xs = cx + k * az
        ys = cy - k * el  # elevation up = screen up
        xy = np.stack([xs, ys], axis=1)
        r = max(3.0, _median_nn_distance(xy) * 0.56)
        colors = _lerp_array(EYE_DARK, EYE_BRIGHT, act)
        for (x, y), c in zip(xy, colors):
            draw.regular_polygon((float(x), float(y), r), n_sides=6, rotation=30, fill=tuple(int(v) for v in c))

    def _draw_raster(self, img: Image.Image, draw: ImageDraw.ImageDraw, raster: np.ndarray | None) -> None:
        inner = self._panel(draw, self.raster_box, "SPIKE RASTER / neurons x time / last 1 s")
        x0, y0, x1, y1 = inner
        w, h = x1 - x0, y1 - y0
        if raster is None or np.size(raster) == 0:
            draw.text(((x0 + x1) // 2, (y0 + y1) // 2), "no data", font=load_mono_font(18), fill=HUD_MUTED, anchor="mm")
            return
        raster = np.asarray(raster).astype(bool)
        if raster.ndim != 2:
            raise ValueError("raster must be 2-D (neurons x time)")
        if self.raster_window is not None:
            raster = raster[:, -self.raster_window :]
        n, t = raster.shape
        # Bin spikes into the panel resolution (max-pool style via bincount).
        rows = (np.arange(n) * h // max(n, 1)).astype(np.int64)
        cols = (np.arange(t) * w // max(t, 1)).astype(np.int64)
        ri, ci = np.nonzero(raster)
        counts = np.zeros((h, w), dtype=np.float32)
        if len(ri):
            np.add.at(counts, (rows[ri], cols[ci]), 1.0)
        # Normalise by how many source cells map to a panel pixel, then boost.
        density = np.clip(counts / max(1.0, (n / h) * (t / w)) * 3.0, 0, 1)
        block = _lerp_array(HUD_PANEL, SPIKE, density)
        img.paste(Image.fromarray(block), (x0, y0))
        # Time axis: a faint "now" marker at the right edge.
        draw.line((x1 - 1, y0, x1 - 1, y1), fill=HUD_ACCENT, width=2)

    def _draw_bars(self, draw: ImageDraw.ImageDraw, dn_bars: Sequence[tuple[str, float, float]] | None) -> None:
        inner = self._panel(draw, self.bars_box, "MOTOR OUTPUT / left | right")
        x0, y0, x1, y1 = inner
        if not dn_bars:
            draw.text(((x0 + x1) // 2, (y0 + y1) // 2), "no data", font=load_mono_font(18), fill=HUD_MUTED, anchor="mm")
            return
        rows = list(dn_bars)[:6]
        row_h = (y1 - y0) / len(rows)
        cx = (x0 + x1) // 2
        label_w = 150
        half = (x1 - x0) // 2 - label_w // 2 - 10
        f = load_mono_font(max(14, min(20, int(row_h * 0.55))))
        for i, (label, lv, rv) in enumerate(rows):
            yc = int(y0 + (i + 0.5) * row_h)
            bh = max(4, int(row_h * 0.45))
            draw.text((cx, yc), str(label)[:12], font=f, fill=HUD_TEXT, anchor="mm")
            lx1 = cx - label_w // 2
            rx0 = cx + label_w // 2
            draw.line((lx1 - half, yc, lx1, yc), fill=HUD_LINE, width=1)
            draw.line((rx0, yc, rx0 + half, yc), fill=HUD_LINE, width=1)
            lw = int(np.clip(lv, 0, 1) * half)
            rw = int(np.clip(rv, 0, 1) * half)
            if lw > 0:
                draw.rectangle((lx1 - lw, yc - bh // 2, lx1, yc + bh // 2), fill=BAR_LEFT)
            if rw > 0:
                draw.rectangle((rx0, yc - bh // 2, rx0 + rw, yc + bh // 2), fill=BAR_RIGHT)

    def _draw_stats(self, draw: ImageDraw.ImageDraw, stats: dict[str, object] | None) -> None:
        inner = self._panel(draw, self.stats_box)
        x0, y0, x1, y1 = inner
        if not stats:
            return
        items = list(stats.items())
        n = len(items)
        col_w = (x1 - x0) / n
        fl = load_mono_font(16)
        fv = load_mono_font(30)
        for i, (k, v) in enumerate(items):
            cx = int(x0 + (i + 0.5) * col_w)
            draw.text((cx, y0 + 6), str(k).upper(), font=fl, fill=HUD_MUTED, anchor="ma")
            draw.text((cx, y1 - 4), _fmt_value(v), font=fv, fill=HUD_ACCENT, anchor="md")

    # ------------------------------------------------------------ compose
    def compose(
        self,
        frame: np.ndarray,
        eye_left: "EyeData | tuple | dict | None" = None,
        eye_right: "EyeData | tuple | dict | None" = None,
        raster: np.ndarray | None = None,
        dn_bars: Sequence[tuple[str, float, float]] | None = None,
        stats: dict[str, object] | None = None,
        caption: str = "",
    ) -> np.ndarray:
        """Build one HUD frame; returns ``np.uint8`` of shape (height, width, 3).

        ``frame`` is the 48x84 game frame; ``eye_left``/``eye_right`` are
        ``(points, activity)`` pairs (see :class:`EyeData`); ``raster`` is a
        bool array (neurons x time); ``dn_bars`` is a list of
        ``(label, left_value, right_value)`` with values in 0..1; ``stats``
        maps English labels to values; ``caption`` is the bottom line.
        """
        img = Image.new("RGB", (self.width, self.height), HUD_BG)
        draw = ImageDraw.Draw(img)

        # Header
        hx0, hy0, hx1, _ = self.header_box
        draw.text((hx0, hy0), self.title, font=load_mono_font(26), fill=HUD_TEXT, anchor="la")
        draw.text((hx1, hy0 + 4), "84×48 · 1 bit", font=load_mono_font(18), fill=HUD_MUTED, anchor="ra")

        # Nokia screen (hero)
        screen = self.nokia.render(frame)
        img.paste(Image.fromarray(screen), self.screen_pos)

        # Eyes
        self._draw_eye(img, draw, self.eye_left_box, EyeData.coerce(eye_left), "LEFT EYE / LC spikes")
        self._draw_eye(img, draw, self.eye_right_box, EyeData.coerce(eye_right), "RIGHT EYE / LC spikes")

        # Raster, bars, stats
        self._draw_raster(img, draw, raster)
        self._draw_bars(draw, dn_bars)
        self._draw_stats(draw, stats)

        # Caption
        cx0, cy0, cx1, cy1 = self.caption_box
        draw.text(((cx0 + cx1) // 2, (cy0 + cy1) // 2), caption, font=load_mono_font(24), fill=HUD_TEXT, anchor="mm")
        return np.asarray(img)


def _fmt_value(v: object) -> str:
    """Compact English number formatting for the stats strip."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        n = int(v)
        if abs(n) >= 1_000_000:
            return f"{n / 1e6:.1f}M"
        if abs(n) >= 10_000:
            return f"{n / 1e3:.1f}k"
        return f"{n:,}"
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.2f}"
    return str(v)
