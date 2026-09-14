import numpy as np, collections
from flysnake.brain import load_subgraph, LIFNet
g=load_subgraph(); net=LIFNet(g)
net.reset(); ids=net.ids('LC10a','L'); d=np.zeros(net.n,np.float32); d[ids]=1.0
late=np.zeros(net.n,np.int64)
for t in range(700):
    spk=net.step(d if t<200 else None)
    if t>=500: late+=spk
c=collections.Counter()
for i in np.where(late>0)[0]: c[f"{net.type[i]}{net.side[i]}"]+=int(late[i])
print('persistent spikes 500-700ms by type/side:',c.most_common(15))
print('n neurons active late:',int((late>0).sum()))
