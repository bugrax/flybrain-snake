"""Snake II recreation: 84x48 LCD, wraparound, critters and five mazes.

Original implementation; no Nokia ROM or game assets. Maze geometry and
timings are approximations, documented in docs/snake-ii.md.
The older walled experiment remains in flysnake.game for reproducibility.
"""
from __future__ import annotations

import numpy as np

from .game import SnakeGame, STRAIGHT, apply_action

COLS, ROWS, HUD = 21, 10, 8
TICK_MS = (300, 260, 220, 180, 150, 120, 100, 80, 60)
BONUS_TICKS = 40
DIGITS = (
    ('111', '101', '101', '101', '111'),
    ('010', '110', '010', '010', '111'),
    ('111', '001', '111', '100', '111'),
    ('111', '001', '111', '001', '111'),
    ('101', '101', '111', '001', '001'),
    ('111', '100', '111', '001', '111'),
    ('111', '100', '111', '101', '111'),
    ('111', '001', '010', '010', '010'),
    ('111', '101', '111', '101', '111'),
    ('111', '101', '111', '001', '111'),
)


def maze_cells(maze: int) -> set[tuple[int, int]]:
    """Original layouts evoking the five classic labyrinth choices."""
    if maze == 0:
        return set()
    if maze == 1:
        return {(x, y) for x in range(COLS) for y in range(ROWS)
                if x in (0, COLS - 1) or y in (0, ROWS - 1)}
    if maze == 2:
        return {(x, y) for x in (5, 15) for y in range(2, 8)}
    if maze == 3:
        return {(x, y) for y in (2, 7) for x in range(4, 17)}
    if maze == 4:
        return {(x, y) for x in (4, 16) for y in range(2, 8)} | {
            (x, y) for y in (2, 7) for x in range(7, 14)}
    if maze == 5:
        return {(x, y) for x in (5, 15) for y in range(ROWS) if y not in (4, 5)} | {
            (x, y) for y in (2, 7) for x in range(8, 13)}
    raise ValueError('maze must be 0..5')


class NokiaSnakeGame(SnakeGame):
    def __init__(self, seed=None, level=4, maze=0):
        if not 1 <= level <= 9:
            raise ValueError('level must be 1..9')
        self.level, self.maze = level, maze
        self.walls = maze_cells(maze)
        super().__init__(seed)

    @property
    def tick_ms(self):
        return TICK_MS[self.level - 1]

    def reset(self, seed=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.body = [(10, 5), (9, 5), (8, 5)]
        self.heading = (1, 0)
        self.score = self.food_eaten = self.t = self.steps_since_food = 0
        self.alive, self.won, self.paused = True, False, False
        self.bonus = None
        self.bonus_left = 0
        self.digestion = []
        self.event = 'ready'
        self.food = self._spawn_food()
        return self.observe()

    def _free_cells(self):
        occupied = set(self.body) | self.walls | set(self.bonus or ())
        return [(x, y) for y in range(ROWS) for x in range(COLS) if (x, y) not in occupied]

    def _spawn_bonus(self):
        free = set(self._free_cells()) - {self.food}
        pairs = [(c, (c[0] + 1, c[1])) for c in sorted(free)
                 if c[0] < COLS - 1 and (c[0] + 1, c[1]) in free]
        if pairs:
            self.bonus = pairs[int(self._rng.integers(len(pairs)))]
            self.bonus_left = BONUS_TICKS

    def toggle_pause(self):
        if self.alive:
            self.paused = not self.paused
        return self.observe()

    def step(self, action=STRAIGHT):
        if not self.alive or self.paused:
            return self.observe()
        self.heading = apply_action(self.heading, action)
        new = ((self.head[0] + self.heading[0]) % COLS,
               (self.head[1] + self.heading[1]) % ROWS)
        self.t += 1
        eating = new == self.food
        bonus = new in (self.bonus or ())
        growing = eating or bonus
        if new in self.walls or new in (self.body if growing else self.body[:-1]):
            self.alive, self.event = False, 'game over'
            return self.observe()
        self.body.insert(0, new)
        self.event = 'move'
        self.digestion = [i + 1 for i in self.digestion if i + 1 < len(self.body)]
        if growing:
            self.steps_since_food = 0
            self.digestion.append(0)
            if eating:
                self.food_eaten += 1
                self.score += self.level
                self.event = 'food'
                self.food = self._spawn_food()
            else:
                self.score += self.level * self.bonus_left
                self.bonus = None
                self.bonus_left = 0
                self.event = 'bonus'
            if len(self.body) + len(self.walls) == COLS * ROWS:
                self.won, self.alive, self.event = True, False, 'board complete'
                self.food = None
        else:
            self.body.pop()
            self.steps_since_food += 1
        if self.bonus:
            self.bonus_left -= 1
            if self.bonus_left <= 0:
                self.bonus = None
        # A bonus can temporarily reserve the last cells on the board.
        if self.food is None and self.alive:
            self.food = self._spawn_food()
        if eating and self.food_eaten % 5 == 0 and self.bonus is None and self.alive:
            self._spawn_bonus()
        return self.observe()

    def observe(self):
        obs = super().observe()
        obs.update(bounds=(0, 0, COLS - 1, ROWS - 1), wrap=True,
                   walls=sorted(self.walls), level=self.level, maze=self.maze,
                   bonus=self.bonus, bonus_left=self.bonus_left,
                   food_eaten=self.food_eaten, paused=self.paused, event=self.event)
        return obs

    def render_frame(self):
        f = np.zeros((48, 84), dtype=np.uint8)
        for i, digit in enumerate(f'{self.score:05d}'[-5:]):
            for y, line in enumerate(DIGITS[int(digit)]):
                for x, v in enumerate(line):
                    f[1 + y, 1 + i * 4 + x] = int(v)
        for i in range(self.level):
            f[2:6, 82 - 2 * i] = 1
        f[7, :] = 1
        if self.bonus:
            f[2:5, 28:28 + int(25 * self.bonus_left / BONUS_TICKS)] = 1
        field = f[HUD:]
        for x, y in self.walls:
            field[y*4:y*4+4, x*4:x*4+4] = 1
            field[y*4+1, x*4+1] = 0
        for i, (x, y) in enumerate(self.body):
            px, py = x * 4, y * 4
            field[py:py+3, px:px+3] = 1
            if i:
                field[py+1, px+1] = 0
            if i in self.digestion:
                field[py:py+4, px:px+4] = 1
            if i + 1 < len(self.body):
                bx, by = self.body[i+1]
                if abs(x - bx) + abs(y - by) == 1:
                    if y == by:
                        field[py:py+3, max(x,bx)*4-1] = 1
                    else:
                        field[max(y,by)*4-1, px:px+3] = 1
        x, y = self.head
        dx, dy = self.heading
        field[y*4+1-dx, x*4+1+dy] = 0
        ahead = ((x + dx) % COLS, (y + dy) % ROWS)
        if ahead == self.food or ahead in (self.bonus or ()):
            field[y*4+1+dy, x*4+1+dx] = 0
        if self.food:
            x, y = self.food
            field[y*4:y*4+3, x*4:x*4+3] |= np.array([[0,1,0],[1,1,1],[0,1,0]], np.uint8)
        if self.bonus:
            x, y = self.bonus[0]
            field[y*4:y*4+4, x*4:x*4+8] |= np.array(
                [[0,1,0,1,1,0,1,0],[1,1,1,1,1,1,1,1],
                 [0,1,1,1,1,1,1,0],[1,0,1,0,0,1,0,1]], np.uint8)
        return f
