import numpy as np
import pytest
from flysnake.nokia import NokiaSnakeGame, COLS, ROWS
from flysnake.eye import EyeModel
from flysnake.brain import load_subgraph


def test_wraparound_and_no_seam_bridge():
    g = NokiaSnakeGame(0)
    g.body = [(20,5),(19,5),(18,5)]
    g.food = (10,2)
    g.step()
    assert g.head == (0,5) and g.alive
    assert not g.render_frame()[28:31, 40:45].any()


def test_maze_collision_and_spawns():
    g = NokiaSnakeGame(0, maze=1)
    g.body = [(19,5),(18,5),(17,5)]
    assert g.food not in g.walls
    assert not g.step()['alive']
    for maze in range(6):
        g = NokiaSnakeGame(4, maze=maze)
        assert not set(g.body) & g.walls
        assert g.food not in g.walls


def test_food_score_bonus_pickup_and_expiry():
    g = NokiaSnakeGame(0, level=4)
    g.food = (11,5)
    g.food_eaten = 4
    g.step()
    assert g.score == 4 and len(g.body) == 4 and g.bonus
    assert not set(g.bonus) & (set(g.body) | {g.food})
    g.bonus = ((12,5),(13,5)); g.bonus_left = 7
    g.step()
    assert g.score == 32 and g.bonus is None and len(g.body) == 5
    g.bonus = ((1,1),(2,1)); g.bonus_left = 1
    g.food = (10,2)
    g.step()
    assert g.bonus is None and g.bonus_left == 0


def test_pause_freezes_state_and_reset_clears_bonus():
    g = NokiaSnakeGame(7)
    obs = g.toggle_pause()
    assert g.step()['head'] == obs['head'] and g.t == 0
    g.toggle_pause(); g.step()
    assert g.t == 1
    g._spawn_bonus(); g.reset(7)
    assert g.bonus is None and not g.paused and g.score == 0


def test_self_collision_tail_and_full_board():
    g = NokiaSnakeGame(0)
    g.body = [(3,3),(3,4),(4,4),(4,3),(5,3)]
    assert not g.step()['alive']
    g.reset(0); g.body = [(3,3),(3,4),(4,4),(4,3)]
    g.food = (1,1)
    assert g.step()['alive']
    g.reset(0)
    g.body = [(0,0)] + [(x,y) for y in range(ROWS) for x in range(COLS) if (x,y) not in ((0,0),(1,0))]
    g.food = (1,0)
    obs = g.step()
    assert obs['won'] and not obs['alive'] and obs['food'] is None


def test_eye_does_not_hallucinate_wall_at_open_edge():
    eye = EyeModel(load_subgraph())
    g = NokiaSnakeGame(0)
    g.body = [(20,5)]
    g.food = (10,2)
    _, info = eye.drive(g.observe())
    assert info['loom_sum'] == 0
    obs = g.observe(); obs['wrap'] = False
    _, info = eye.drive(obs)
    assert info['loom_sum'] > 0
    obs['food'] = None
    assert np.isfinite(eye.drive(obs)[0]).all()


def test_speed_and_validation():
    assert NokiaSnakeGame(level=9).tick_ms < NokiaSnakeGame(level=1).tick_ms
    with pytest.raises(ValueError): NokiaSnakeGame(level=0)
    with pytest.raises(ValueError): NokiaSnakeGame(maze=6)
