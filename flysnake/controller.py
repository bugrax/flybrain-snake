"""Closed loop: game state → eye model → connectome LIF (250 ms per game step) → descending-neuron readout → turn.

Readout (all population spike counts of real descending neurons, left vs right):
  steer  = DNa02 + DNa01 + DNa03 + DNa13 + DNa15 (ipsilateral steering DNs; LC10a→AOTU→DNa02 pathway)
           right > left → turn right (toward the object), and vice versa.
  escape = DNp02 + DNp03 + DNp04 + DNp11 (directional escape DNs driven by LC4)
           threat side fires → turn AWAY from that side (contralateral escape, Card & Dickinson 2008).
  GF     = DNp01 (giant fiber) — emergency: if it fires with an obstacle straight ahead, turn to the quieter side.
"""
from __future__ import annotations
import numpy as np
from .brain import LIFNet, load_subgraph
from .eye import EyeModel

STEER_TYPES = ['DNa02', 'DNa01', 'DNa03', 'DNa13', 'DNa15']
ESCAPE_TYPES = ['DNp02', 'DNp03', 'DNp04', 'DNp11']
GF = 'DNp01'

class FlyController:
    def __init__(self, g=None, ms_per_step=250, seed=0, steer_thr=6, escape_thr=8):
        self.g = g if g is not None else load_subgraph()
        self.net = LIFNet(self.g, seed=seed); self.eye = EyeModel(self.g)
        self.ms = ms_per_step; self.steer_thr = steer_thr; self.escape_thr = escape_thr
        self.raster = np.zeros((self.net.n, 0), bool)
        self._ids = {t: (self.net.ids(t, 'L'), self.net.ids(t, 'R')) for t in STEER_TYPES + ESCAPE_TYPES + [GF, 'TTMn', 'MDN']}
        self.prev_obs = None; self.last = {}
        self.gain = dict(steer=(1.0, 1.0), escape=(1.0, 1.0))
        self._calibrate()
    def _run(self, d):
        self.net.reset(); counts = np.zeros(self.net.n, np.int64); spikes = np.zeros((self.net.n, self.ms), bool)
        for t in range(self.ms):
            spk = self.net.step(d); counts += spk; spikes[:, t] = spk
        return counts, spikes
    def _lr(self, counts, types):
        L = sum(counts[self._ids[t][0]].sum() for t in types); R = sum(counts[self._ids[t][1]].sum() for t in types)
        return int(L), int(R)
    def _calibrate(self):
        """Left/right gain normalisation: the isolated male subcircuit is not mirror-symmetric (e.g. the frontal LC10a
        subset drives left steering DNs ~7x more than right). We present the SAME frontal stimulus to both eyes once
        and rescale each side so a symmetric view yields a symmetric readout. Documented in the README."""
        big = dict(bounds=(-50, -50, 50, 50))
        food = dict(head=(0, 0), heading=(1, 0), food=(5, 0), body=[(0, 0)], **big)
        wall = dict(head=(0, 0), heading=(1, 0), food=(-30, 0), body=[(2, -1), (2, 0), (2, 1), (2, -2), (2, 2), (0, 0)], **big)
        cS, _ = self._run(self.eye.drive(food)[0]); sL, sR = self._lr(cS, STEER_TYPES)
        cE, _ = self._run(self.eye.drive(wall)[0]); eL, eR = self._lr(cE, ESCAPE_TYPES)
        def g(L, R):
            m = 0.5 * (L + R)
            return (float(np.clip(m / max(L, 1), 0.33, 3.0)), float(np.clip(m / max(R, 1), 0.33, 3.0)))
        self.calib = dict(steer_raw=(sL, sR), escape_raw=(eL, eR), gains_if_applied=dict(steer=g(sL, sR), escape=g(eL, eR)))
        # gains are measured for the record but NOT applied: with the graded eye model (σ=25°) the frontal
        # response is already symmetric (see scripts/tuning_sweep.py), so the readout stays uncorrected.
    def act(self, obs: dict) -> int:
        d, info = self.eye.drive(obs, self.prev_obs); self.prev_obs = obs
        counts, spikes = self._run(d)
        lr = lambda types: self._lr(counts, types)
        sL0, sR0 = lr(STEER_TYPES); eL0, eR0 = lr(ESCAPE_TYPES); gL, gR = lr([GF]); tL, tR = lr(['TTMn'])
        sL, sR = sL0 * self.gain['steer'][0], sR0 * self.gain['steer'][1]
        eL, eR = eL0 * self.gain['escape'][0], eR0 * self.gain['escape'][1]
        action, reason = 0, 'straight'
        if max(eL, eR) >= self.escape_thr:
            if abs(eL - eR) >= 0.3 * (eL + eR):
                action = 2 if eL > eR else 1; reason = 'escape: left threat -> right' if eL > eR else 'escape: right threat -> left'
            elif abs(sL - sR) >= max(3, 0.4 * (sL + sR)):
                action = 2 if sR > sL else 1; reason = 'escape: front threat -> food side'
            else:
                action = 2 if eL >= eR else 1; reason = 'escape: front threat -> quieter side'
        elif (gL + gR) >= 4 and info.get('loom_sum', 0) > 0.5:
            action = 2 if eL >= eR else 1; reason = 'giant fiber: emergency turn'
        elif max(sL, sR) >= self.steer_thr and abs(sL - sR) >= max(3, 0.4 * (sL + sR)):
            action = 2 if sR > sL else 1; reason = 'food: turn right' if sR > sL else 'food: turn left'
        self.last = dict(steer=(round(sL), round(sR)), escape=(round(eL), round(eR)), steer_raw=(sL0, sR0), escape_raw=(eL0, eR0), gf=(gL, gR), ttmn=(tL, tR), counts=counts, spikes=spikes,
                         drive=d, info=info, action=action, reason=reason)
        return action
