"""Synthetic situations → does the closed loop decide sensibly? (no game needed)"""
import numpy as np, time
from flysnake.controller import FlyController
c = FlyController()
W, H = 19, 10
def obs(head, heading, food, body=None):
    return dict(head=head, heading=heading, food=food, body=body or [head], grid_w=W, grid_h=H)
cases = [
 ('food directly ahead',        obs((5,5),(1,0),(10,5))),
 ('food left (up)',  obs((5,5),(1,0),(7,2))),
 ('food right (down)',   obs((5,5),(1,0),(7,8))),
 ('food behind',          obs((5,5),(1,0),(1,5))),
 ("wall one cell ahead", obs((18,5),(1,0),(2,2))),
 ("wall two cells ahead", obs((17,5),(1,0),(2,2))),
 ("wall three cells ahead", obs((16,5),(1,0),(2,2))),
 ('body ahead, food left', obs((5,5),(1,0),(7,1), body=[(7,4),(7,5),(7,6),(6,5),(5,5)])),
 ('corner: walls ahead and right', obs((17,8),(1,0),(2,2))),
]
t0=time.time()
for name, o in cases:
    a = c.act(o); L=c.last
    print(f"{name:32s} → {['STRAIGHT','LEFT','RIGHT'][a]:4s} | steer L/R {L['steer']} escape L/R {L['escape']} GF {L['gf']} TTMn {L['ttmn']} | {L['reason']} | {L['info']}")
print(f"{len(cases)} decisions {time.time()-t0:.2f}s")
import numpy as np
rf=c.eye.rf; T=c.eye.type; S=c.eye.side
for t in ['LC10a','LC4','LPLC2']:
    for sd in ['L','R']:
        m=(T==t)&(S==sd)&~np.isnan(rf[:,0])
        print(f"RF {t} {sd}: n={m.sum()} az [{rf[m,0].min():.0f},{np.median(rf[m,0]):.0f},{rf[m,0].max():.0f}] el [{rf[m,1].min():.0f},{np.median(rf[m,1]):.0f},{rf[m,1].max():.0f}]")
