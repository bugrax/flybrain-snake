"""Connectome subcircuit extraction + leaky integrate-and-fire simulation (MaleCNS v1.0).

Honesty notes (these are stated in the README and in the video):
  * The retina→lamina→medulla stage is NOT simulated as LIF (a raw Shiu-type LIF cannot produce
    small-object / looming selectivity there). Instead an explicit *eye model* (flysnake.eye) drives the
    lobula columnar (LC) neurons directly, using receptive-field centres derived from each LC neuron's
    real presynaptic columns (assignedOlHex1/2 in the MaleCNS annotations).
  * From the LC populations downstream everything is the real connectome: every synapse between the
    included neurons, signed by the MaleCNS neurotransmitter prediction, integrated with the LIF
    parameters of Shiu et al. 2024 (Nature).
  * Electrical (gap-junction) synapses are absent from the chemical connectome; the LC4→GF and GF→TTMn
    electrical components are therefore NOT in the model (we do not add them by hand for Snake).
"""
from __future__ import annotations
import time, json
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / 'data' / 'malecns-v1.0'
CACHE = Path(__file__).resolve().parent.parent / 'data' / 'subgraph_snake.npz'

SEED_TYPES = ['LC10a', 'LC11', 'LC4', 'LPLC2', 'LC6', 'LC16']          # visual projection neurons we drive
READOUT_TYPES = ['DNa02', 'DNa01', 'DNa03', 'DNp01', 'DNp02', 'DNp03', 'DNp04', 'DNp06', 'DNp09', 'DNp11',
                 'DNp35', 'DNp103', 'DNa10', 'DNa13', 'DNa15', 'MDN', 'TTMn', 'DNpe017', 'DNp20']
NT_SIGN = {'acetylcholine': +1.0, 'gaba': -1.0, 'glutamate': -1.0, 'histamine': -1.0,
           'dopamine': +1.0, 'octopamine': +1.0, 'serotonin': +1.0}

# Shiu et al. 2024 LIF parameters
V_REST, V_RESET, V_TH = -52.0, -52.0, -45.0   # mV
TAU_M, TAU_SYN, T_REF, DELAY = 20.0, 5.0, 2.2, 1.8   # ms
W_SYN = 0.275                                  # mV per synapse (single free parameter)
# Spike-frequency adaptation (NOT in Shiu 2024; added because the isolated subcircuit lacks its inhibitory
# surround and otherwise reverberates indefinitely through recurrent DN/LC excitation — see README).
A_INC, TAU_A = 1.0, 150.0                      # mV per spike, ms


def build_subgraph(min_syn: int = 5, inter_min_in: int = 20, verbose=True) -> dict:
    """Extract: seeds (LC types) → 1-hop postsynaptic partners (≥ inter_min_in synapses from seeds)
    → 2-hop descending neurons; keep every edge (≥ min_syn) among the included set."""
    import pyarrow.feather as f, pandas as pd
    t0 = time.time()
    ann = f.read_feather(DATA / 'body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId')
    nt = f.read_feather(DATA / 'body-neurotransmitters-male-cns-v1.0.feather')
    W = f.read_feather(DATA / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
    Wfull = W
    W = W[W.weight >= min_syn]
    bodies = np.union1d(ann.index.values, np.union1d(Wfull.body_pre.values, Wfull.body_post.values))
    idx = pd.Series(np.arange(len(bodies)), index=bodies); N = len(bodies)
    A = sp.csr_matrix((W.weight.values.astype(np.float32), (idx[W.body_pre.values].values, idx[W.body_post.values].values)), shape=(N, N))
    typ = pd.Series('', index=bodies, dtype=object); typ.loc[ann.index] = ann['type'].fillna('').values
    sup = pd.Series('', index=bodies, dtype=object); sup.loc[ann.index] = ann['superclass'].fillna('').values
    side = pd.Series('', index=bodies, dtype=object); side.loc[ann.index] = ann['somaSide'].fillna('').values
    hex1 = pd.Series(np.nan, index=bodies); hex1.loc[ann.index] = ann['assignedOlHex1'].values
    hex2 = pd.Series(np.nan, index=bodies); hex2.loc[ann.index] = ann['assignedOlHex2'].values
    T, S = typ.values, side.values
    seeds = np.where(np.isin(T, SEED_TYPES))[0]
    v = np.zeros(N, np.float32); v[seeds] = 1
    in_from_seeds = A.T @ v
    hop1 = np.where((in_from_seeds >= inter_min_in))[0]
    v1 = np.zeros(N, np.float32); v1[hop1] = 1
    in_from_hop1 = A.T @ v1
    dn_mask = (sup.values == 'descending_neuron')
    hop2_dn = np.where(dn_mask & (in_from_hop1 >= inter_min_in))[0]
    readout = np.where(np.isin(T, READOUT_TYPES))[0]
    keep = np.unique(np.concatenate([seeds, hop1, hop2_dn, readout]))
    # neurotransmitter sign per body (consensus_nt if present else predicted)
    nt = nt.drop_duplicates('body').set_index('body')
    ntcol = 'consensus_nt' if 'consensus_nt' in nt.columns else 'predicted_nt'
    sign = np.zeros(len(keep), np.float32)
    ntname = np.array([''] * len(keep), dtype=object)
    for k, b in enumerate(bodies[keep]):
        if b in nt.index:
            name = str(nt.at[b, ntcol]).lower(); ntname[k] = name
            sign[k] = NT_SIGN.get(name, 0.0)
    unknown = (sign == 0).sum()
    sign[sign == 0] = +1.0  # unknown → treat as excitatory (Shiu: majority rule; documented)
    sub = A[keep][:, keep].tocoo()
    Wsigned = sp.csr_matrix((sub.data * sign[sub.row] * W_SYN, (sub.row, sub.col)), shape=(len(keep), len(keep)))
    # receptive-field centres for seeds from hex-tagged presynaptic columns (ALL synapses, no min_syn filter)
    hasHex = ~np.isnan(hex1.values)
    rf = np.full((len(keep), 2), np.nan, np.float32)
    Afull = sp.csr_matrix((Wfull.weight.values.astype(np.float32), (idx[Wfull.body_pre.values].values, idx[Wfull.body_post.values].values)), shape=(N, N))
    Acsc = Afull.tocsc()
    for k, i in enumerate(keep):
        if T[i] not in SEED_TYPES: continue
        col = Acsc[:, i]; pre, w = col.indices, col.data
        m = hasHex[pre]
        if m.sum() == 0: continue
        rf[k, 0] = np.average(hex1.values[pre][m], weights=w[m]); rf[k, 1] = np.average(hex2.values[pre][m], weights=w[m])
    out = dict(body=bodies[keep].astype(np.int64), type=T[keep].astype(str), side=S[keep].astype(str), superclass=sup.values[keep].astype(str),
               nt=ntname.astype(str), sign=sign, rf_hex=rf, W_indptr=Wsigned.indptr, W_indices=Wsigned.indices, W_data=Wsigned.data.astype(np.float32),
               n=len(keep), n_syn=float(sub.data.sum()), n_edges=int(sub.nnz))
    if verbose:
        print(f'subgraph: {len(keep)} neurons ({len(seeds)} LC seeds, {len(hop1)} hop-1, {len(hop2_dn)} hop-2 DNs), '
              f'{sub.nnz} edges, {int(sub.data.sum())} synapses, {unknown} unknown-NT (→ +), {time.time()-t0:.0f}s')
        import collections
        print('superclass:', dict(collections.Counter(out['superclass'])))
    np.savez_compressed(CACHE, **out)
    return out


def load_subgraph(rebuild=False) -> dict:
    if rebuild or not CACHE.exists():
        return build_subgraph()
    z = np.load(CACHE, allow_pickle=False)
    return {k: z[k] for k in z.files}


@dataclass
class LIFNet:
    """Push-mode LIF network on the signed CSR weight matrix. dt = 1 ms, exponential synapses, 2-ms delay bin."""
    g: dict
    dt: float = 1.0
    seed: int = 0
    def __post_init__(self):
        g = self.g; self.n = int(g['n'])
        self.W = sp.csr_matrix((g['W_data'], g['W_indices'], g['W_indptr']), shape=(self.n, self.n))
        self.type = np.asarray(g['type']); self.side = np.asarray(g['side'])
        self.v = np.full(self.n, V_REST, np.float32); self.i_syn = np.zeros(self.n, np.float32)
        self.ref = np.zeros(self.n, np.float32); self.a = np.zeros(self.n, np.float32)
        self.delay_steps = max(1, int(round(DELAY / self.dt)))
        self.queue = [np.zeros(self.n, np.float32) for _ in range(self.delay_steps)]
        self.rng = np.random.default_rng(self.seed)
        self.t = 0
        self._by_type = {}
    def ids(self, t: str, side: str | None = None) -> np.ndarray:
        key = (t, side)
        if key not in self._by_type:
            m = self.type == t
            if side: m &= self.side == side
            self._by_type[key] = np.where(m)[0]
        return self._by_type[key]
    def reset(self):
        self.v[:] = V_REST; self.i_syn[:] = 0; self.ref[:] = 0; self.a[:] = 0
        for q in self.queue: q[:] = 0
        self.t = 0
    def step(self, drive_mV: np.ndarray | None = None, noise_mV: float = 0.0) -> np.ndarray:
        """Advance 1 ms. drive_mV: external input in mV/ms added to membrane (from the eye model)."""
        dt = self.dt
        # synaptic current decays; incoming delayed spikes add W_SYN*count (already signed in W)
        arriving = self.queue.pop(0)
        self.i_syn += arriving
        self.i_syn *= np.float32(np.exp(-dt / TAU_SYN))
        self.a *= np.float32(np.exp(-dt / TAU_A))
        dv = (-(self.v - V_REST) + self.i_syn - self.a) * (dt / TAU_M)
        if drive_mV is not None: dv += drive_mV * dt
        if noise_mV > 0: dv += self.rng.normal(0, noise_mV * np.sqrt(dt), self.n).astype(np.float32)
        active = self.ref <= 0
        self.v[active] += dv[active]
        self.ref -= dt
        spk = active & (self.v >= V_TH)
        self.v[spk] = V_RESET; self.ref[spk] = T_REF; self.a[spk] += A_INC
        # push: postsynaptic input from spiking neurons, delivered after DELAY
        if spk.any():
            contrib = np.asarray(self.W[np.where(spk)[0]].sum(axis=0)).ravel().astype(np.float32)
        else:
            contrib = np.zeros(self.n, np.float32)
        self.queue.append(contrib)
        self.t += 1
        return spk


def hex_to_angles(rf_hex: np.ndarray, side: np.ndarray) -> np.ndarray:
    """Approximate map from optic-lobe hex column coordinates to (azimuth, elevation) in degrees.
    Hex axes (hex1, hex2) span ~1..36 / 1..39 per eye; ~5.1° inter-ommatidial angle. We use the
    two diagonal combinations as azimuth/elevation and mirror the left eye. Documented as approximate."""
    h1, h2 = rf_hex[:, 0], rf_hex[:, 1]
    az = (h1 - h2) * 5.1 * 0.87          # anterior↔posterior axis (0 ≈ frontal)
    el = ((h1 + h2) - 40.0) * 5.1 * 0.5   # dorsal↔ventral
    az = np.where(side == 'L', -np.abs(az) - 5, np.abs(az) + 5)   # left eye → negative azimuth
    return np.stack([az, el], axis=1).astype(np.float32)


if __name__ == '__main__':
    g = build_subgraph()
    import collections
    print('types (top 30):', collections.Counter(g['type']).most_common(30))
    net = LIFNet(g)
    t0 = time.time()
    for _ in range(1000): net.step(noise_mV=0.0)
    print(f'1000 steps silent: {time.time()-t0:.2f}s')
