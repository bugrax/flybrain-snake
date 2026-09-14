"""The published decision trace must reproduce from the included connectome."""
import json
from pathlib import Path
from flysnake.controller import FlyController
from flysnake.nokia import NokiaSnakeGame


def test_published_episode_is_reproducible():
    path = Path(__file__).resolve().parents[1] / 'media' / 'flybrain-snake-linkedin.json'
    data = json.loads(path.read_text())
    game = NokiaSnakeGame(data['seed'], data['level'], data['maze'])
    controller = FlyController()
    for entry in data['trace']:
        assert list(game.head) == entry['head']
        assert game.score == entry['score']
        action = controller.act(game.observe())
        assert action == entry['action']
        game.step(action)
    assert game.score == data['points']
    assert game.food_eaten == data['food']
    assert game.t == data['steps']
    assert game.alive == data['alive']
