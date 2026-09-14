"""Go/no-go: drive an LC population on one side; measure which descending neurons spike, and on which side.
This is the honest core claim of the project: connectome LC→(interneurons)→DN pathways carry lateralised signals."""
import numpy as np, time, collections
from flysnake.brain import load_subgraph, LIFNet
g = load_subgraph(); net = LIFNet(g)
types, sides = net.type, net.side
DN_WATCH = ['DNa02','DNa01','DNa03','DNa10','DNa13','DNa15','DNp01','DNp02','DNp03','DNp04','DNp06','DNp09','DNp11','DNp35','DNp103','MDN','TTMn']
def run(stim_type, stim_side, drive, ms=300, frac=1.0, noise=0.0):
    net.reset(); net.rng = np.random.default_rng(1)
    ids = net.ids(stim_type, stim_side)
    ids = ids[: int(len(ids)*frac)]
    d = np.zeros(net.n, np.float32); d[ids] = drive
    counts = np.zeros(net.n, np.int64)
    t0=time.time()
    for t in range(ms):
        spk = net.step(d if t < ms-50 else None, noise_mV=noise)
        counts += spk
    dt_ms = (time.time()-t0)/ms*1000
    out = {}
    for T in DN_WATCH:
        L = counts[net.ids(T,'L')].sum(); R = counts[net.ids(T,'R')].sum()
        if L or R: out[T]=(int(L),int(R))
    stim_rate = counts[ids].sum()/max(1,len(ids))/(ms-50)*1000
    total = counts.sum()
    return out, stim_rate, total, dt_ms
print(f"{'stim':10s} {'side':4s} {'drive':>5s} {'LCHz':>6s} {'spikes':>7s} {'ms/step':>7s}  DN counts (L,R)")
for stim in ['LC10a','LC11','LC4','LPLC2']:
    for side in ['L','R']:
        for drive in [0.3, 0.6, 1.0]:
            out, sr, total, dtm = run(stim, side, drive)
            print(f"{stim:10s} {side:4s} {drive:5.1f} {sr:6.0f} {total:7d} {dtm:7.2f}  {out}")
# noise-only baseline
out, sr, total, dtm = run('LC4','L',0.0, noise=0.5)
print('noise-only 0.5 mV:', total, out)
