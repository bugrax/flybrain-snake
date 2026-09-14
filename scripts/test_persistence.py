"""Does activity persist / run away after the stimulus stops? (recurrent LC10a↔LC10a, AOTU loops)"""
import numpy as np
from flysnake.brain import load_subgraph, LIFNet
g=load_subgraph(); net=LIFNet(g)
for stim in ['LC10a','LC4']:
    for side in ['L']:
        net.reset(); ids=net.ids(stim,side); d=np.zeros(net.n,np.float32); d[ids]=1.0
        tot=[]
        for t in range(1200):
            spk=net.step(d if t<200 else None); tot.append(int(spk.sum()))
        tot=np.array(tot)
        print(stim, side, 'spikes/100ms windows:', [int(tot[i:i+100].sum()) for i in range(0,1200,100)])
