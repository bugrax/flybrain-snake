"""Tests for flysnake.game and the baseline agents."""

import numpy as np
import pytest

from flysnake.baselines import GreedyAgent, RandomAgent, run_episode
from flysnake.game import (
    GRID_H,
    GRID_W,
    HEIGHT_PX,
    PLAY_X0,
    PLAY_X1,
    STRAIGHT,
    TURN_LEFT,
    TURN_RIGHT,
    WIDTH_PX,
    SnakeGame,
    turn_left,
    turn_right,
)


def test_initial_state():
    g = SnakeGame(seed=1)
    obs = g.observe()
    assert obs["frame"].shape == (HEIGHT_PX, WIDTH_PX)
    assert obs["frame"].dtype == np.uint8
    assert set(np.unique(obs["frame"])) <= {0, 1}
    assert len(obs["body"]) == 3
    assert obs["heading"] == (1, 0)
    assert obs["head"] == (10, 5)
    assert obs["body"] == [(10, 5), (9, 5), (8, 5)]
    assert obs["score"] == 0 and obs["alive"] and obs["t"] == 0
    assert obs["food"] not in obs["body"]
    assert GRID_W == 21 and GRID_H == 12


def test_movement_straight():
    g = SnakeGame(seed=1)
    obs = g.step(STRAIGHT)
    assert obs["head"] == (11, 5)
    assert obs["body"] == [(11, 5), (10, 5), (9, 5)]
    assert obs["t"] == 1
    assert len(obs["body"]) == 3


def test_relative_turns():
    assert turn_left((1, 0)) == (0, -1)  # right -> up
    assert turn_right((1, 0)) == (0, 1)  # right -> down
    assert turn_left((0, -1)) == (-1, 0)  # up -> left
    assert turn_right((0, -1)) == (1, 0)  # up -> right
    g = SnakeGame(seed=1)
    obs = g.step(TURN_LEFT)
    assert obs["heading"] == (0, -1) and obs["head"] == (10, 4)
    obs = g.step(TURN_RIGHT)
    assert obs["heading"] == (1, 0) and obs["head"] == (11, 4)
    obs = g.step(TURN_RIGHT)
    assert obs["heading"] == (0, 1) and obs["head"] == (11, 5)


def test_eating_grows_and_scores():
    g = SnakeGame(seed=1)
    # Plant the food directly in front of the head.
    g.food = (11, 5)
    obs = g.step(STRAIGHT)
    assert obs["score"] == 1
    assert len(obs["body"]) == 4
    assert obs["body"][0] == (11, 5)
    assert obs["food"] != (11, 5) and obs["food"] not in obs["body"]
    assert obs["steps_since_food"] == 0
    # Next step keeps the new length.
    obs = g.step(STRAIGHT)
    assert len(obs["body"]) == 4


def test_wall_death():
    g = SnakeGame(seed=1)
    steps_to_wall = PLAY_X1 - g.head[0]  # cells until the last playable column
    for _ in range(steps_to_wall):
        obs = g.step(STRAIGHT)
        assert obs["alive"]
    obs = g.step(STRAIGHT)
    assert not obs["alive"]
    assert obs["head"] == (PLAY_X1, 5)  # head stays on the last safe cell
    # Frozen after death.
    t = obs["t"]
    obs2 = g.step(TURN_LEFT)
    assert obs2["t"] == t and obs2["head"] == obs["head"]


def test_self_collision():
    g = SnakeGame(seed=1)
    # Grow to length 5 by planting food ahead twice.
    for _ in range(2):
        hx, hy = g.head
        g.food = (hx + 1, hy)
        g.step(STRAIGHT)
    assert len(g.body) == 5
    # A tight loop: left, left, left runs into the body.
    g.step(TURN_LEFT)
    g.step(TURN_LEFT)
    obs = g.step(TURN_LEFT)
    assert not obs["alive"]


def test_moving_into_tail_is_safe():
    g = SnakeGame(seed=1)
    hx, hy = g.head
    g.food = (hx + 1, hy)
    g.step(STRAIGHT)  # length 4: head (11,5) body (10,5),(9,5),(8,5)
    g.food = (PLAY_X0, 1)  # keep food far away
    g.step(TURN_LEFT)  # (11,4)
    g.step(TURN_LEFT)  # (10,4)
    obs = g.step(TURN_LEFT)  # (10,5): the tail vacates this cell this tick
    assert obs["alive"]
    assert obs["head"] == (10, 5)


def test_determinism_with_seed():
    def play(seed):
        g = SnakeGame(seed=seed)
        rng = np.random.default_rng(123)
        trace = []
        for _ in range(60):
            obs = g.step(int(rng.integers(3)))
            trace.append((obs["head"], obs["food"], obs["score"], obs["alive"]))
            if not obs["alive"]:
                break
        return trace

    assert play(7) == play(7)
    assert SnakeGame(seed=7).food == SnakeGame(seed=7).food
    # Different seeds should (almost surely) place food differently somewhere.
    foods_a = [SnakeGame(seed=s).food for s in range(10)]
    foods_b = [SnakeGame(seed=s + 100).food for s in range(10)]
    assert foods_a != foods_b


def test_reset_restores_state():
    g = SnakeGame(seed=3)
    g.step(TURN_LEFT)
    obs = g.reset(seed=3)
    assert obs["head"] == (10, 5) and obs["t"] == 0 and obs["score"] == 0
    assert obs["food"] == SnakeGame(seed=3).food


def test_frame_has_wall_frame_and_food_dot():
    g = SnakeGame(seed=1)
    f = g.observe()["frame"]
    # Inner edge of the wall ring is a dark 1-px rectangle.
    assert f[3, 3:81].all() and f[44, 3:81].all()
    assert f[3:45, 3].all() and f[3:45, 80].all()
    # Food is drawn as a 3x3 plus sign.
    fx, fy = g.food
    px, py = fx * 4, fy * 4
    assert f[py + 1, px : px + 3].all() and f[py : py + 3, px + 1].all()
    assert f[py, px] == 0 and f[py + 2, px + 2] == 0


def test_invalid_action_raises():
    g = SnakeGame(seed=1)
    with pytest.raises(ValueError):
        g.step(5)


def test_greedy_scores_positive_and_beats_random():
    greedy = GreedyAgent()
    scores = [run_episode(SnakeGame(seed=s), greedy, max_steps=1500, starvation=300)["score"] for s in range(5)]
    assert min(scores) > 0
    assert np.mean(scores) >= 10
    random_scores = [run_episode(SnakeGame(seed=s), RandomAgent(s), max_steps=1500)["score"] for s in range(5)]
    assert np.mean(scores) > np.mean(random_scores)


def test_random_agent_actions_valid():
    a = RandomAgent(0)
    g = SnakeGame(seed=0)
    for _ in range(50):
        assert a.act(g.observe()) in (STRAIGHT, TURN_LEFT, TURN_RIGHT)
