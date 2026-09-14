"""Explore MaleCNS v1.0 for the Snake controller: small-object (LC10a/LC11), looming (LC4/LPLC2),
descending neurons downstream, and columnar retinotopy (assignedOlHex) to derive LC receptive fields."""
import pyarrow.feather as f, pandas as pd, numpy as np, scipy.sparse as sp, time, re
D='data/malecns-v1.0/'
t0=time.time()
ann=f.read_feather(D+'body-annotations-male-cns-v1.0-minconf-0.5.feather').set_index('bodyId')
W=f.read_feather(D+'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
print('weights',list(W.columns),len(W),f'{time.time()-t0:.0f}s')
bodies=np.union1d(ann.index.values,np.union1d(W.body_pre.values,W.body_post.values))
idx=pd.Series(np.arange(len(bodies)),index=bodies); N=len(bodies)
A=sp.csr_matrix((W.weight.values.astype(np.float32),(idx[W.body_pre.values].values,idx[W.body_post.values].values)),shape=(N,N))
typ=pd.Series('',index=bodies,dtype=object); typ.loc[ann.index]=ann['type'].fillna('').values
sup=pd.Series('',index=bodies,dtype=object); sup.loc[ann.index]=ann['superclass'].fillna('').values
side=pd.Series('',index=bodies,dtype=object); side.loc[ann.index]=ann['somaSide'].fillna('').values
hex1=pd.Series(np.nan,index=bodies); hex1.loc[ann.index]=ann['assignedOlHex1'].values
hex2=pd.Series(np.nan,index=bodies); hex2.loc[ann.index]=ann['assignedOlHex2'].values
T=typ.values; S=side.values
def ids(types): return np.where(np.isin(T,types))[0]
print('\n== LC / small-object / looming populations (count by side) ==')
for t in ['LC10a','LC10b','LC10c','LC10d','LC10a-1','LC10a-2','LC11','LC12','LC15','LC17','LC4','LPLC2','LPLC1','LC6','LC16','LC9','LC13','LC18','LC21','LC22','LC25','LC26','LC31','T2','T2a','T3','Tm1','Tm2','Tm3','Tm4','Tm9','Tm20','Mi1','L1','L2','L3','T4a','T4b','T4c','T4d','T5a','T5b','T5c','T5d']:
    i=ids([t]);
    if len(i): print(f'{t:8s} n={len(i):4d} L={np.sum(S[i]=="L"):4d} R={np.sum(S[i]=="R"):4d} hex={int(np.sum(~np.isnan(hex1.values[i]))):4d}')
lc10=[t for t in np.unique(T) if t.startswith('LC10')]; print('LC10 variants:',lc10)
print('\n== Downstream of LC10a and LC11: top postsynaptic types (synapse-weighted) ==')
for src in ['LC10a','LC11','LC4','LPLC2']:
    s=ids([src]);
    if not len(s): s=ids([t for t in np.unique(T) if t.startswith(src)])
    out=np.asarray(A[s].sum(axis=0)).ravel()
    ser=pd.Series(out,index=T).groupby(level=0).sum().sort_values(ascending=False)
    ser=ser[ser.index!='']
    print(f'{src}: total out {int(out.sum())} →', {k:int(v) for k,v in ser.head(14).items()})
    # descending neurons among targets
    dn=pd.Series(out,index=[f'{a}|{b}' for a,b in zip(T,sup)]).groupby(level=0).sum()
    dn=dn[[k.endswith('|descending_neuron') for k in dn.index]].sort_values(ascending=False)
    print('   DN targets:',{k.split('|')[0]:int(v) for k,v in dn.head(10).items() if v>0})
print('\n== Two-hop: LC10a/LC11 → interneurons → DNs (which DNs receive ≥ X synapses via 1 intermediate) ==')
DNmask=(sup.values=='descending_neuron')
for src in ['LC10a','LC11']:
    s=ids([src]); v=np.zeros(N,np.float32); v[s]=1
    h1=A.T@v                      # synapses onto each neuron from src
    inter=(h1>=5)&~DNmask
    w1=np.zeros(N,np.float32); w1[inter]=h1[inter]
    h2=A.T@(w1/ (w1.max()+1e-9))
    ser=pd.Series(h2[DNmask],index=T[DNmask]).groupby(level=0).sum().sort_values(ascending=False)
    print(src,'2-hop DN score:',{k:round(float(v),1) for k,v in ser.head(12).items()})
print('\n== Key DNs / MNs counts by side ==')
for t in ['DNa01','DNa02','DNa03','DNa04','DNb01','DNp01','DNp02','DNp03','DNp04','DNp06','DNp09','DNp11','DNp20','DNpe017','MDN','TTMn','PSI','DNae014','DNa11','DNge','DNg13']:
    i=ids([t]);
    if len(i): print(f'{t:8s} n={len(i)} sides={dict(pd.Series(S[i]).value_counts())}')
print('\n== Retinotopy: derive RF centers for LC4/LPLC2/LC11/LC10a from presynaptic columnar partners with hex ==')
hasHex=~np.isnan(hex1.values)
print('bodies with hex:',hasHex.sum(),'types with hex (top):',pd.Series(T[hasHex]).value_counts().head(20).to_dict())
for t in ['LC4','LPLC2','LC11','LC10a','LC6','LC16']:
    i=ids([t])
    if not len(i): continue
    sub=A[:,i].tocsc()  # inputs to each LC neuron
    ok=0; spreads=[]
    for k,col in enumerate(i):
        pre=sub[:,k].indices; w=sub[:,k].data
        m=hasHex[pre]
        if m.sum()==0: continue
        ww=w[m]; hx=hex1.values[pre][m]; hy=hex2.values[pre][m]
        cx=np.average(hx,weights=ww); cy=np.average(hy,weights=ww)
        spread=np.sqrt(np.average((hx-cx)**2+(hy-cy)**2,weights=ww))
        ok+=1; spreads.append(spread)
    print(f'{t:6s}: {ok}/{len(i)} neurons have hex-tagged inputs; median RF spread {np.median(spreads):.1f} columns' if spreads else f'{t}: none')
print('hex1 range',np.nanmin(hex1.values),np.nanmax(hex1.values),'hex2 range',np.nanmin(hex2.values),np.nanmax(hex2.values))
print(f'done {time.time()-t0:.0f}s')
