"""Spontaneous activity: does a little membrane noise (spontaneous saccades) reduce starvation without hurting?"""
import numpy as np, sys
from flysnake.controller import FlyController
from scripts.play_fly import run
for noise in [0.0, 0.3, 0.6, 1.0]:
    ctrl = FlyController()
    orig = ctrl.net.step
    ctrl.net.step = lambda d=None, noise_mV=noise, _o=orig: _o(d, noise_mV=noise_mV)
    scores=[]; starved=0
    for ep in range(20):
        s,n,alive,_ = run(ctrl, seed=ep); scores.append(s); starved += int(alive)
    print(f"noise {noise:.1f} mV: mean {np.mean(scores):.2f} max {max(scores)} starved {starved}/20", flush=True)
