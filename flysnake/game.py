"""Nokia 3310 "Snake II"-style snake on an 84x48 monochrome canvas.

Geometry
--------
The canvas is 84x48 pixels, divided into 4x4-pixel cells: 21 columns by
12 rows.  The outermost ring of cells is the wall, drawn as a thin frame,
so the playable field is 19x10 cells (x in 1..19, y in 1..10).  Cell
coordinates are ``(x, y)`` with ``y`` growing downwards, like the screen.

Actions are *relative* to the current heading:

* ``0`` (:data:`STRAIGHT`)   keep going
* ``1`` (:data:`TURN_LEFT`)  rotate the heading 90 degrees counter-clockwise
* ``2`` (:data:`TURN_RIGHT`) rotate the heading 90 degrees clockwise

The engine is fully deterministic for a given ``seed`` and depends only
on numpy.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- canvas / grid constants -------------------------------------------------
WIDTH_PX = 84
HEIGHT_PX = 48
CELL_PX = 4
GRID_W = WIDTH_PX // CELL_PX  # 21 columns
GRID_H = HEIGHT_PX // CELL_PX  # 12 rows
# playable cells: x in [PLAY_X0, PLAY_X1], y in [PLAY_Y0, PLAY_Y1]
PLAY_X0, PLAY_X1 = 1, GRID_W - 2  # 1..19
PLAY_Y0, PLAY_Y1 = 1, GRID_H - 2  # 1..10

# --- actions -----------------------------------------------------------------
STRAIGHT = 0
TURN_LEFT = 1
TURN_RIGHT = 2
ACTIONS = (STRAIGHT, TURN_LEFT, TURN_RIGHT)

Cell = tuple[int, int]
Heading = tuple[int, int]

START_LENGTH = 3


def turn_left(heading: Heading) -> Heading:
    """Rotate a heading 90 degrees counter-clockwise (screen coordinates, y down)."""
    dx, dy = heading
    return (dy, -dx)


def turn_right(heading: Heading) -> Heading:
    """Rotate a heading 90 degrees clockwise (screen coordinates, y down)."""
    dx, dy = heading
    return (-dy, dx)


def apply_action(heading: Heading, action: int) -> Heading:
    """Return the new heading after applying a relative action."""
    if action == TURN_LEFT:
        return turn_left(heading)
    if action == TURN_RIGHT:
        return turn_right(heading)
    if action == STRAIGHT:
        return heading
    raise ValueError(f"unknown action {action!r}; expected 0, 1 or 2")


def relative_action(heading: Heading, desired: Heading) -> int | None:
    """Map an absolute desired direction to a relative action.

    Returns ``None`` when ``desired`` is the reverse of ``heading`` (a snake
    cannot turn back on itself in one step).
    """
    if desired == heading:
        return STRAIGHT
    if desired == turn_left(heading):
        return TURN_LEFT
    if desired == turn_right(heading):
        return TURN_RIGHT
    return None


def is_wall(cell: Cell) -> bool:
    """True when the cell lies on the 1-cell wall ring."""
    x, y = cell
    return x < PLAY_X0 or x > PLAY_X1 or y < PLAY_Y0 or y > PLAY_Y1


class SnakeGame:
    """Deterministic Snake II engine.

    Parameters
    ----------
    seed:
        Seed for food placement.  The same seed and the same action
        sequence always produce the same game.

    Public state after :meth:`reset` / :meth:`step` is exposed through the
    observation dict (see :meth:`observe`).
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self.body: list[Cell] = []  # head first
        self.heading: Heading = (1, 0)
        self.food: Cell | None = None
        self.score = 0
        self.alive = True
        self.won = False
        self.t = 0
        self.steps_since_food = 0
        self.reset(seed)

    # ------------------------------------------------------------------ setup
    def reset(self, seed: int | None = None) -> dict[str, Any]:
        """Start a new game and return the first observation.

        Passing ``seed`` re-seeds the food generator; ``None`` keeps the
        current generator running (so consecutive episodes differ but the
        whole sequence is still reproducible from the constructor seed).
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        cx = (PLAY_X0 + PLAY_X1) // 2  # 10
        cy = (PLAY_Y0 + PLAY_Y1) // 2  # 5
        self.heading = (1, 0)
        self.body = [(cx - i, cy) for i in range(START_LENGTH)]
        self.score = 0
        self.alive = True
        self.won = False
        self.t = 0
        self.steps_since_food = 0
        self.food = self._spawn_food()
        return self.observe()

    def _free_cells(self) -> list[Cell]:
        occupied = set(self.body)
        return [
            (x, y)
            for y in range(PLAY_Y0, PLAY_Y1 + 1)
            for x in range(PLAY_X0, PLAY_X1 + 1)
            if (x, y) not in occupied
        ]

    def _spawn_food(self) -> Cell | None:
        free = self._free_cells()
        if not free:
            return None
        idx = int(self._rng.integers(len(free)))
        return free[idx]

    # ------------------------------------------------------------------- play
    @property
    def head(self) -> Cell:
        return self.body[0]

    def step(self, action: int = STRAIGHT) -> dict[str, Any]:
        """Advance one tick with a relative action and return the observation.

        After death (or a full board) the game is frozen: further calls
        return the same observation without changing state.
        """
        if not self.alive:
            return self.observe()

        self.heading = apply_action(self.heading, action)
        hx, hy = self.head
        new_head = (hx + self.heading[0], hy + self.heading[1])
        self.t += 1

        eating = new_head == self.food
        # The tail cell frees up this tick unless we are growing into it.
        blocking = self.body if eating else self.body[:-1]
        if is_wall(new_head) or new_head in blocking:
            self.alive = False
            return self.observe()

        self.body.insert(0, new_head)
        if eating:
            self.score += 1
            self.steps_since_food = 0
            self.food = self._spawn_food()
            if self.food is None:
                self.won = True
                self.alive = False
        else:
            self.body.pop()
            self.steps_since_food += 1
        return self.observe()

    def observe(self) -> dict[str, Any]:
        """Build the observation dict for the current state."""
        return {
            "frame": self.render_frame(),
            "head": self.head,
            "heading": self.heading,
            "food": self.food,
            "score": self.score,
            "alive": self.alive,
            "won": self.won,
            "body": list(self.body),
            "t": self.t,
            "steps_since_food": self.steps_since_food,
        }

    # ----------------------------------------------------------------- render
    def render_frame(self) -> np.ndarray:
        """Rasterise the state to a 48x84 uint8 array (1 = dark pixel)."""
        f = np.zeros((HEIGHT_PX, WIDTH_PX), dtype=np.uint8)
        _draw_wall(f)
        for i, cell in enumerate(self.body):
            _draw_body_cell(f, cell)
            if i + 1 < len(self.body):
                _draw_joint(f, cell, self.body[i + 1])
        _draw_head(f, self.head, self.heading)
        if self.food is not None:
            _draw_food(f, self.food)
        return f


# --- drawing primitives (module level so tests can inspect them) ------------
def _cell_origin(cell: Cell) -> tuple[int, int]:
    return cell[0] * CELL_PX, cell[1] * CELL_PX


def _draw_wall(f: np.ndarray) -> None:
    """1-px frame on the inner edge of the wall ring, Nokia style."""
    x0, y0 = PLAY_X0 * CELL_PX - 1, PLAY_Y0 * CELL_PX - 1  # 3, 3
    x1, y1 = (PLAY_X1 + 1) * CELL_PX, (PLAY_Y1 + 1) * CELL_PX  # 80, 44
    f[y0, x0 : x1 + 1] = 1
    f[y1, x0 : x1 + 1] = 1
    f[y0 : y1 + 1, x0] = 1
    f[y0 : y1 + 1, x1] = 1


def _draw_body_cell(f: np.ndarray, cell: Cell) -> None:
    """Body segment: a 3x3 block in the top-left of its 4x4 cell."""
    px, py = _cell_origin(cell)
    f[py : py + 3, px : px + 3] = 1


def _draw_joint(f: np.ndarray, a: Cell, b: Cell) -> None:
    """Fill the 1-px gap between two consecutive segments so the snake is continuous."""
    ax, ay = _cell_origin(a)
    bx, by = _cell_origin(b)
    if a[1] == b[1]:  # horizontal neighbours: fill the column between them
        x = max(ax, bx) - 1
        f[ay : ay + 3, x] = 1
    else:  # vertical neighbours: fill the row between them
        y = max(ay, by) - 1
        f[y, ax : ax + 3] = 1


def _draw_head(f: np.ndarray, head: Cell, heading: Heading) -> None:
    """Head: same 3x3 block plus a nose pixel in the heading direction."""
    px, py = _cell_origin(head)
    dx, dy = heading
    # Nose sits just outside the 3x3 block, giving the head a pointed look.
    nx = px + 1 + dx * 2
    ny = py + 1 + dy * 2
    if 0 <= nx < WIDTH_PX and 0 <= ny < HEIGHT_PX:
        f[ny, nx] = 1
    # Eye: clear the front-left corner pixel of the 3x3 block (front = heading,
    # left = counter-clockwise of heading in screen coordinates).
    lx, ly = turn_left(heading)
    f[py + 1 + dy + ly, px + 1 + dx + lx] = 0


def _draw_food(f: np.ndarray, food: Cell) -> None:
    """Food: a 3x3 plus sign, clearly different from the solid body blocks."""
    px, py = _cell_origin(food)
    f[py + 1, px : px + 3] = 1
    f[py : py + 3, px + 1] = 1
