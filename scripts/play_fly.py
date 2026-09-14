"""Fly connectome plays Nokia Snake: run episodes, print score table + decision log."""
import argparse, time, numpy as np, json
from flysnake.game import SnakeGame, PLAY_X0, PLAY_X1, PLAY_Y0, PLAY_Y1
from flysnake.controller import FlyController

def wrap(obs):
    o = dict(obs); o['bounds'] = (PLAY_X0, PLAY_Y0, PLAY_X1, PLAY_Y1); return o

def run(ctrl, seed, max_steps=1500, starvation=300, log=None):
    game = SnakeGame(seed=seed); obs = wrap(game.reset(seed=seed))
    ctrl.net.reset(); ctrl.prev_obs = None
    reasons = []
    for t in range(max_steps):
        a = ctrl.act(obs); L = ctrl.last
        reasons.append(L['reason'].split(':')[0])
        if log is not None: log.append(dict(t=t, a=a, reason=L['reason'], steer=L['steer'], escape=L['escape'], gf=L['gf'], score=obs['score']))
        obs = wrap(game.step(a))
        if not obs['alive'] or obs.get('steps_since_food', 0) > starvation: break
    import collections
    return obs['score'], t + 1, obs['alive'], collections.Counter(reasons)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--episodes', type=int, default=10); ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()
    ctrl = FlyController(); scores = []; t0 = time.time()
    for ep in range(args.episodes):
        log = [] if args.verbose else None
        s, n, alive, reasons = run(ctrl, seed=ep, log=log)
        scores.append(s)
        print(f"ep {ep:2d}: score {s:3d}  steps {n:4d}  {'step limit' if alive else 'game over'}  decisions {dict(reasons)}")
        if log:
            for r in log[:40]: print('   ', r)
    print(f"\nFLY: mean {np.mean(scores):.2f}  std {np.std(scores):.2f}  best {max(scores)}  ({args.episodes} games, {time.time()-t0:.1f}s)")
