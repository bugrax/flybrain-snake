"""Baseline agents for :class:`flysnake.game.SnakeGame` and a comparison CLI.

All agents share one interface::

    action = agent.act(obs)   # obs is the dict returned by SnakeGame.step()

where ``action`` is a relative action (0 straight, 1 left, 2 right).

Run the comparison table (used in the video as "random / greedy / fly /
you")::

    uv run python -m flysnake.baselines --episodes 50
"""

from __future__ import annotations

import argparse
from collections import deque
from typing import Any, Protocol

import numpy as np

from flysnake.game import (
    ACTIONS,
    STRAIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    Cell,
    Heading,
    SnakeGame,
    apply_action,
    is_wall,
    relative_action,
)

Obs = dict[str, Any]


class Agent(Protocol):
    """Anything with ``act(obs) -> action``."""

    def act(self, obs: Obs) -> int: ...


# --------------------------------------------------------------------- random
class RandomAgent:
    """Uniformly random relative action every step (the "random" row)."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, obs: Obs) -> int:
        return int(self._rng.choice(ACTIONS))


# --------------------------------------------------------------------- greedy
def _neighbours(cell: Cell) -> list[tuple[Heading, Cell]]:
    x, y = cell
    return [(d, (x + d[0], y + d[1])) for d in ((1, 0), (-1, 0), (0, 1), (0, -1))]


def bfs_first_step(start: Cell, goal: Cell, blocked: set[Cell]) -> Heading | None:
    """Breadth-first shortest path on the playfield.

    Returns the absolute direction of the first step towards ``goal`` or
    ``None`` when ``goal`` is unreachable (or already reached).
    """
    if start == goal:
        return None
    parent: dict[Cell, Cell] = {start: start}
    queue: deque[Cell] = deque([start])
    while queue:
        cur = queue.popleft()
        for _, nxt in _neighbours(cur):
            if nxt in parent or is_wall(nxt) or nxt in blocked:
                continue
            parent[nxt] = cur
            if nxt == goal:
                node = nxt
                while parent[node] != start:
                    node = parent[node]
                return (node[0] - start[0], node[1] - start[1])
            queue.append(nxt)
    return None


def reachable_area(start: Cell, blocked: set[Cell]) -> int:
    """Number of playfield cells reachable from ``start`` (flood fill)."""
    seen = {start}
    queue: deque[Cell] = deque([start])
    while queue:
        cur = queue.popleft()
        for _, nxt in _neighbours(cur):
            if nxt in seen or is_wall(nxt) or nxt in blocked:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return len(seen) - 1


class GreedyAgent:
    """Shortest-path-to-food agent (the "greedy" row).

    Strategy: BFS from the head to the food, treating body cells (except
    the tail, which moves away) as walls.  If the food is unreachable, pick
    the safe move whose target has the largest reachable area; prefer going
    straight on ties.
    """

    def act(self, obs: Obs) -> int:
        head: Cell = obs["head"]
        heading: Heading = obs["heading"]
        body: list[Cell] = obs["body"]
        food: Cell | None = obs["food"]
        blocked = set(body[:-1])  # tail vacates this tick unless we eat

        if food is not None:
            step = bfs_first_step(head, food, blocked)
            action = relative_action(heading, step) if step is not None else None
            if action is not None:
                # Only chase the food if there is still room to survive afterwards.
                nxt = (head[0] + step[0], head[1] + step[1])
                after = (set(body) if nxt == food else blocked) | {nxt}
                if reachable_area(nxt, after) >= len(body):
                    return action
        return self._best_safe_move(obs)

    @staticmethod
    def _safe_moves(obs: Obs) -> list[tuple[int, Cell]]:
        head: Cell = obs["head"]
        heading: Heading = obs["heading"]
        blocked = set(obs["body"][:-1])
        out = []
        for action in (STRAIGHT, TURN_LEFT, TURN_RIGHT):
            d = apply_action(heading, action)
            nxt = (head[0] + d[0], head[1] + d[1])
            if not is_wall(nxt) and nxt not in blocked:
                out.append((action, nxt))
        return out

    def _best_safe_move(self, obs: Obs) -> int:
        moves = self._safe_moves(obs)
        if not moves:
            return STRAIGHT  # doomed either way
        blocked = set(obs["body"][:-1])
        best_action, best_area = STRAIGHT, -1
        for action, nxt in moves:  # STRAIGHT is first, so it wins ties
            area = reachable_area(nxt, blocked | {nxt})
            if area > best_area:
                best_action, best_area = action, area
        return best_action


# ---------------------------------------------------------------------- human
class HumanKeysAgent:
    """Stub for keyboard play: feed keys with :meth:`push_key`.

    The agent consumes one queued turn per step and goes straight when
    the queue is empty.  Wire it to whatever input loop the UI uses
    (``"left"``/``"right"`` or ``"a"``/``"d"``).
    """

    KEYMAP = {"left": TURN_LEFT, "a": TURN_LEFT, "right": TURN_RIGHT, "d": TURN_RIGHT}

    def __init__(self) -> None:
        self._queue: deque[int] = deque()

    def push_key(self, key: str) -> None:
        action = self.KEYMAP.get(key.lower())
        if action is not None:
            self._queue.append(action)

    def act(self, obs: Obs) -> int:
        return self._queue.popleft() if self._queue else STRAIGHT


# ---------------------------------------------------------------- episode run
def run_episode(
    game: SnakeGame,
    agent: Agent,
    max_steps: int = 2000,
    seed: int | None = None,
    starvation: int | None = None,
) -> dict[str, Any]:
    """Play one episode and return ``{"score", "steps", "alive", "won"}``.

    ``starvation`` (steps without food) ends the episode early so agents
    that loop forever do not hang the benchmark.
    """
    obs = game.reset(seed)
    steps = 0
    while obs["alive"] and steps < max_steps:
        obs = game.step(agent.act(obs))
        steps += 1
        if starvation is not None and obs["steps_since_food"] >= starvation:
            break
    return {"score": obs["score"], "steps": steps, "alive": obs["alive"], "won": obs["won"]}


def benchmark(
    agents: dict[str, Agent], episodes: int = 50, max_steps: int = 2000, seed: int = 0
) -> dict[str, dict[str, float]]:
    """Run each agent for ``episodes`` seeded games; return summary stats."""
    results: dict[str, dict[str, float]] = {}
    for name, agent in agents.items():
        game = SnakeGame(seed)
        scores, lengths = [], []
        for ep in range(episodes):
            r = run_episode(game, agent, max_steps=max_steps, seed=seed + ep, starvation=400)
            scores.append(r["score"])
            lengths.append(r["steps"])
        results[name] = {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "max": float(np.max(scores)),
            "mean_steps": float(np.mean(lengths)),
        }
    return results


def format_table(results: dict[str, dict[str, float]], placeholders: tuple[str, ...] = ()) -> str:
    """Render the English comparison table used in the video."""
    header = f"{'Agent':<12}{'Mean':>10}{'Std':>8}{'Best':>8}{'Steps':>8}"
    lines = [header, "-" * len(header)]
    for name, s in results.items():
        lines.append(
            f"{name:<12}{s['mean']:>10.2f}{s['std']:>8.2f}{s['max']:>8.0f}{s['mean_steps']:>8.0f}"
        )
    for name in placeholders:
        lines.append(f"{name:<12}{'—':>10}{'—':>8}{'—':>8}{'—':>8}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Compare baseline agents (random / greedy).")
    p.add_argument("--episodes", type=int, default=50, help="episodes per agent")
    p.add_argument("--max-steps", type=int, default=2000, help="maximum steps per episode")
    p.add_argument("--seed", type=int, default=0, help="first episode seed")
    args = p.parse_args(argv)

    agents: dict[str, Agent] = {
        "random": RandomAgent(args.seed),
        "greedy": GreedyAgent(),
    }
    results = benchmark(agents, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    print(f"{args.episodes} episodes, seed {args.seed}, at most {args.max_steps} steps\n")
    print(format_table(results, placeholders=("fly", "you")))


if __name__ == "__main__":
    main()
