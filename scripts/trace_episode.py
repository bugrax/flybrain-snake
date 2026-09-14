import numpy as np
from flysnake.game import SnakeGame, PLAY_X0, PLAY_X1, PLAY_Y0, PLAY_Y1
from flysnake.controller import FlyController
from flysnake.eye import _egocentric
ctrl=FlyController(); game=SnakeGame(seed=2); obs=dict(game.reset(seed=2)); obs['bounds']=(PLAY_X0,PLAY_Y0,PLAY_X1,PLAY_Y1)
fs=ctrl.eye.food_side; fid=ctrl.eye.food_ids
for t in range(60):
    a=ctrl.act(obs); L=ctrl.last; d=L['drive']
    az,dist=_egocentric(tuple(obs['head']),tuple(obs['heading']),tuple(obs['food']))
    dl=d[fid][fs=='L']; dr=d[fid][fs=='R']
    lc_l=L['counts'][fid][fs=='L'].sum(); lc_r=L['counts'][fid][fs=='R'].sum()
    print(f"t{t:2d} head {tuple(obs['head'])} hd {tuple(obs['heading'])} food {tuple(obs['food'])} az {az:5.0f} d {dist:4.1f} | drive>0.5 L {int((dl>0.5).sum()):3d} R {int((dr>0.5).sum()):3d} maxL {dl.max():.2f} maxR {dr.max():.2f} | LC10a spk L {lc_l:4d} R {lc_r:4d} | steer {L['steer']} esc {L['escape']} → {['STRAIGHT','LEFT','RIGHT'][a]:3s} {L['reason'][:22]}")
    obs=dict(game.step(a)); obs['bounds']=(PLAY_X0,PLAY_Y0,PLAY_X1,PLAY_Y1)
    if not obs['alive']: print('GAME OVER'); break
