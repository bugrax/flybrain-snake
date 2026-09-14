"""Azimuth tuning of the steering readout (DNa02+DNa01+DNa03+DNa13+DNa15, L vs R) for food stimuli,
and of the escape readout for obstacles, under different eye-model settings. Pick a robust, monotonic setting."""
import numpy as np, itertools, sys
import flysnake.eye as eye
from flysnake.controller import FlyController, STEER_TYPES, ESCAPE_TYPES
c = FlyController.__new__(FlyController)   # bypass __init__ calibration
from flysnake.brain import load_subgraph, LIFNet
from flysnake.eye import EyeModel
c.g = load_subgraph(); c.net = LIFNet(c.g); c.ms = 250
c._ids = {t: (c.net.ids(t,'L'), c.net.ids(t,'R')) for t in STEER_TYPES+ESCAPE_TYPES+['DNp01','TTMn','MDN']}
def food_obs(az_deg, dist=5):
    a = np.radians(az_deg); fx, fy = dist*np.cos(a), dist*np.sin(a)   # heading (1,0): right = +y
    return dict(head=(0,0), heading=(1,0), food=(round(fx), round(fy)), body=[(0,0)], bounds=(-50,-50,50,50))
def wall_obs(az_deg, dist=2):
    a = np.radians(az_deg); cx, cy = round(dist*np.cos(a)), round(dist*np.sin(a))
    body = [(cx+i, cy+j) for i in (-1,0,1) for j in (-1,0,1)] + [(0,0)]
    return dict(head=(0,0), heading=(1,0), food=(-30,0), body=body, bounds=(-50,-50,50,50))
AZ = [-90,-60,-30,-15,0,15,30,60,90]
for mode, sigma, gain in [('graded',15,1.0),('graded',25,1.0),('graded',25,1.5),('graded',40,1.0),('graded',40,1.5),('topk',15,1.2),('topk',25,1.2)]:
    eye.MODE, eye.AZ_SIGMA, eye.FOOD_GAIN = mode, sigma, gain
    c.eye = EyeModel(c.g)
    row=[]
    for az in AZ:
        d,_ = c.eye.drive(food_obs(az)); counts,_ = c._run(d); L,R = c._lr(counts, STEER_TYPES)
        row.append(f"{L:3d}/{R:<3d}")
    print(f"FOOD  {mode:6s} σ={sigma:2d} g={gain:.1f} | " + ' '.join(f"{az:+3d}:{r}" for az,r in zip(AZ,row)))
for mode, sigma, gain in [('graded',15,1.1),('graded',25,1.1),('graded',25,1.6),('graded',40,1.1),('topk',15,1.1)]:
    eye.MODE, eye.AZ_SIGMA, eye.LOOM_GAIN = mode, sigma, gain
    c.eye = EyeModel(c.g)
    row=[]
    for az in [-60,-30,0,30,60]:
        d,_ = c.eye.drive(wall_obs(az)); counts,_ = c._run(d); L,R = c._lr(counts, ESCAPE_TYPES)
        row.append(f"{az:+3d}:{L:3d}/{R:<3d}")
    print(f"WALL  {mode:6s} σ={sigma:2d} g={gain:.1f} | " + ' '.join(row))
