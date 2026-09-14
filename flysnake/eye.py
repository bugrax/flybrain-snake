"""Explicit *eye model* (declared, not connectome): turns the Snake state into drive currents for the
lobula columnar neurons, using each neuron's receptive-field centre derived from the connectome.

Egocentric view: the compound eyes sit on the snake's head and look along its heading (a 'head camera';
this is stated openly). Two stimulus classes the fly actually has detectors for:
  * small dark object (the food)          → LC10a  (LC10a→AOTU→DNa02 steering pathway)
  * expanding / approaching edge (wall, own body directly ahead) → LC4 and LPLC2 (looming detectors)
Distance is encoded on the elevation axis (near = low in the visual field), azimuth relative to heading.
"""
from __future__ import annotations
import numpy as np
from .brain import hex_to_angles

AZ_SIGMA = 25.0     # deg, angular width of a stimulus on the LC array
EL_SIGMA = 45.0
FOOD_GAIN = 1.0     # mV/ms given to the best-matching LC10a neurons for a food cell in view
LOOM_GAIN = 1.1     # mV/ms given to the best-matching LC4/LPLC2 neurons for an obstacle 1 cell ahead
MODE = 'graded'    # 'graded': drive = gain * w (no per-side normalisation); 'topk': per-side top-K normalised
TOPK_FOOD = 60      # per eye: LC10a neurons receiving full drive for a food cell (RF-match normalised per side)
TOPK_LOOM = 30      # per eye: LC4/LPLC2 neurons receiving full drive for an obstacle
FOV_AZ = 150.0      # each eye covers ~150° to its side; a narrow rear blind spot remains
RF_AZ_MAX = 85.0    # RF centres derived from hex span ~5..85°; more lateral stimuli drive the most lateral RFs

def _egocentric(head, heading, cell):
    """Return (azimuth deg, distance cells) of a cell relative to head & heading. Left = negative azimuth."""
    dx, dy = cell[0] - head[0], cell[1] - head[1]
    hx, hy = heading
    # forward component and rightward component (screen y grows downward → right-hand rule adjusted)
    fwd = dx * hx + dy * hy
    right = dx * (-hy) + dy * hx
    dist = float(np.hypot(dx, dy))
    az = float(np.degrees(np.arctan2(right, fwd)))
    return az, dist

def _norm_per_side(w, side, topk):
    """Per eye, scale so that the top-k best RF matches get 1.0 (removes left/right population-size bias);
    an eye whose best match is poor (w < 0.05) stays silent."""
    out = np.zeros_like(w)
    for sd in ('L', 'R'):
        m = side == sd
        if m.sum() == 0: continue
        ws = w[m]; k = min(topk, m.sum())
        ref = np.sort(ws)[-k]
        if ref < 0.03: continue
        out[m] = np.clip(ws / (ref + 1e-6), 0, 1)
    return out

def _dist_to_el(dist):
    return float(np.clip(-40.0 + 12.0 * dist, -40.0, 30.0))   # 1 cell away → -28°, far → up to +30°

class EyeModel:
    def __init__(self, g: dict):
        self.type = np.asarray(g['type']); self.side = np.asarray(g['side'])
        self.rf = hex_to_angles(np.asarray(g['rf_hex']), self.side)
        self.n = len(self.type)
        self.food_ids = np.where((self.type == 'LC10a') & ~np.isnan(self.rf[:, 0]))[0]
        self.loom_ids = np.where(np.isin(self.type, ['LC4', 'LPLC2']) & ~np.isnan(self.rf[:, 0]))[0]
        self.food_side = self.side[self.food_ids]; self.loom_side = self.side[self.loom_ids]
        self.el_ref = float(np.nanmedian(self.rf[self.food_ids, 1]))
        self.last_loom = 0.0
    def drive(self, obs: dict, prev_obs: dict | None = None) -> tuple[np.ndarray, dict]:
        head, heading, body = tuple(obs['head']), tuple(obs['heading']), obs['body']
        food = tuple(obs['food']) if obs.get('food') is not None else head
        d = np.zeros(self.n, np.float32); info = {}
        # --- food: small dark object
        # Use the visible food position; the eye model has no knowledge of
        # shortest paths through a wraparound edge or of bonus point values.
        az, dist = _egocentric(head, heading, food)
        if abs(az) <= FOV_AZ and dist > 0:
            gain = FOOD_GAIN * (0.7 + 0.3 * np.exp(-dist / 10.0))
            az_m = float(np.clip(az, -RF_AZ_MAX, RF_AZ_MAX))
            rf = self.rf[self.food_ids]
            # 2D game → azimuth-only retinotopy; elevation weight kept broad around the RF median (documented)
            w = np.exp(-((rf[:, 0] - az_m) ** 2) / (2 * AZ_SIGMA ** 2) - ((rf[:, 1] - self.el_ref) ** 2) / (2 * (2 * EL_SIGMA) ** 2))
            if MODE == 'topk': w = _norm_per_side(w, self.food_side, TOPK_FOOD)
            d[self.food_ids] = gain * w
            info['food_az'] = az; info['food_dist'] = dist
        # --- looming: obstacles (walls / body) ahead; expansion ~ approach speed / distance²
        if 'bounds' in obs: xmin, ymin, xmax, ymax = obs['bounds']
        else: xmin, ymin, xmax, ymax = 0, 0, obs['grid_w'] - 1, obs['grid_h'] - 1
        obstacles = set(map(tuple, body[:-1])) if len(body) > 1 else set()
        obstacles.update(map(tuple, obs.get('walls', ())))
        loom_total = np.zeros(len(self.loom_ids), np.float32); loom_sum = 0.0
        rf = self.rf[self.loom_ids]
        cand = []
        # walls: sample wall cells along the field border within 6 cells
        for k in range(1, 7):
            cx, cy = head[0] + heading[0] * k, head[1] + heading[1] * k
            for lat in (-2, -1, 0, 1, 2):
                px, py = cx + (-heading[1]) * lat, cy + heading[0] * lat
                wall = not obs.get('wrap', False) and (px < xmin or py < ymin or px > xmax or py > ymax)
                cell = ((px - xmin) % (xmax - xmin + 1) + xmin,
                        (py - ymin) % (ymax - ymin + 1) + ymin) if obs.get('wrap', False) else (px, py)
                if wall or cell in obstacles:
                    cand.append((px, py))
        for c in set(cand):
            az_c, dist_c = _egocentric(head, heading, c)
            if dist_c <= 0 or abs(az_c) > 100: continue        # looming only matters for cells ahead/lateral
            expansion = 1.0 / dist_c ** 1.5                    # angular expansion ~ approach at 1 cell/step (softened)
            az_m = float(np.clip(az_c, -RF_AZ_MAX, RF_AZ_MAX))
            w = np.exp(-((rf[:, 0] - az_m) ** 2) / (2 * AZ_SIGMA ** 2) - ((rf[:, 1] - self.el_ref) ** 2) / (2 * (2 * EL_SIGMA) ** 2))
            if MODE == 'topk': w = _norm_per_side(w, self.loom_side, TOPK_LOOM)
            loom_total = np.maximum(loom_total, LOOM_GAIN * expansion * w)
            loom_sum += expansion
        d[self.loom_ids] = np.maximum(d[self.loom_ids], loom_total)
        info['loom_sum'] = loom_sum
        return d, info
