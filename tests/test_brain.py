import numpy as np, pytest
from flysnake.brain import load_subgraph, LIFNet, V_REST
from flysnake.eye import EyeModel, _egocentric
from flysnake.controller import FlyController, STEER_TYPES, ESCAPE_TYPES

@pytest.fixture(scope='module')
def g(): return load_subgraph()

def test_subgraph_contents(g):
    types = set(np.asarray(g['type']))
    for t in ['LC10a', 'LC4', 'LPLC2', 'DNa02', 'DNp01', 'DNp02', 'DNp04', 'DNp11', 'TTMn']: assert t in types
    assert int(g['n']) > 3000 and int(g['n_edges']) > 100000

def test_rf_centres_exist_for_lc(g):
    typ = np.asarray(g['type']); rf = np.asarray(g['rf_hex'])
    for t in ['LC10a', 'LC4', 'LPLC2']:
        m = typ == t; assert (~np.isnan(rf[m, 0])).mean() > 0.85

def test_silent_without_input(g):
    net = LIFNet(g); n = sum(int(net.step().sum()) for _ in range(200)); assert n == 0

def test_lateralised_steering_pathway(g):
    net = LIFNet(g)
    for side, other in [('L', 'R'), ('R', 'L')]:
        net.reset(); d = np.zeros(net.n, np.float32); d[net.ids('LC10a', side)] = 0.8
        c = np.zeros(net.n, np.int64)
        for _ in range(250): c += net.step(d)
        same = sum(c[net.ids(t, side)].sum() for t in STEER_TYPES); opp = sum(c[net.ids(t, other)].sum() for t in STEER_TYPES)
        assert same > 30 and same > 5 * max(opp, 1)

def test_lateralised_escape_pathway(g):
    net = LIFNet(g); net.reset(); d = np.zeros(net.n, np.float32); d[net.ids('LC4', 'L')] = 0.8
    c = np.zeros(net.n, np.int64)
    for _ in range(250): c += net.step(d)
    assert sum(c[net.ids(t, 'L')].sum() for t in ESCAPE_TYPES) > 5 * max(1, sum(c[net.ids(t, 'R')].sum() for t in ESCAPE_TYPES))
    assert c[net.ids('DNp01')].sum() > 10   # giant fiber

def test_no_runaway_after_stimulus(g):
    net = LIFNet(g); net.reset(); d = np.zeros(net.n, np.float32); d[net.ids('LC10a', 'L')] = 1.0
    tot = []
    for t in range(800): tot.append(int(net.step(d if t < 200 else None).sum()))
    assert sum(tot[600:]) < 50

def test_egocentric():
    az, dist = _egocentric((5, 5), (1, 0), (5, 8)); assert az == pytest.approx(90) and dist == 3   # +y is right of heading +x
    az, _ = _egocentric((5, 5), (0, -1), (8, 5)); assert az == pytest.approx(90)                    # heading up, +x is right

def test_controller_decisions(g):
    c = FlyController(g)
    big = dict(bounds=(-50, -50, 50, 50), body=[(0, 0)])
    assert c.act(dict(head=(0, 0), heading=(1, 0), food=(4, -3), **big)) == 1   # food left → turn left
    assert c.act(dict(head=(0, 0), heading=(1, 0), food=(4, 3), **big)) == 2    # food right → turn right
    assert c.act(dict(head=(0, 0), heading=(1, 0), food=(5, 0), **big)) == 0    # food ahead → straight
    wall = dict(head=(0, 0), heading=(1, 0), food=(-30, 0), body=[(2, j) for j in (-2, -1, 0, 1, 2)] + [(0, 0)], bounds=(-50, -50, 50, 50))
    assert c.act(wall) in (1, 2)                                                  # obstacle ahead → turns
