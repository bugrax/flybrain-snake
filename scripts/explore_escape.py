"""Explore MaleCNS v1.0: escape circuit size (photoreceptors -> LC4/LPLC2 -> GF/DNp) for a goalkeeper sim."""
import pyarrow.feather as f, pandas as pd, numpy as np, scipy.sparse as sp, time, sys
D='data/malecns-v1.0/'
t0=time.time()
ann=f.read_feather(D+'body-annotations-male-cns-v1.0-minconf-0.5.feather')
W=f.read_feather(D+'connectome-weights-male-cns-v1.0-minconf-0.5.feather')
print('weights cols',list(W.columns),'rows',len(W),f'{time.time()-t0:.1f}s')
print(W.head(3).to_string())
pre,post=[c for c in W.columns if 'pre' in c.lower()][0],[c for c in W.columns if 'post' in c.lower()][0]
wcol=[c for c in W.columns if c not in (pre,post)][0]
print('using',pre,post,wcol, 'weight stats', W[wcol].describe().to_string())
# index bodies
bodies=np.union1d(ann.bodyId.values, np.union1d(W[pre].values,W[post].values))
idx=pd.Series(np.arange(len(bodies)),index=bodies)
N=len(bodies); print('N bodies',N)
A=sp.csr_matrix((W[wcol].values.astype(np.float32),(idx[W[pre].values].values,idx[W[post].values].values)),shape=(N,N))
print('nnz',A.nnz, 'total syn',A.sum())
ann=ann.set_index('bodyId')
typ=pd.Series(index=bodies,dtype=object); typ.loc[ann.index]=ann['type'].values
sup=pd.Series(index=bodies,dtype=object); sup.loc[ann.index]=ann['superclass'].values
side=pd.Series(index=bodies,dtype=object); side.loc[ann.index]=ann['somaSide'].values
def ids(types=None, superclass=None):
    m=np.ones(N,bool)
    if types is not None: m&=typ.isin(types).values
    if superclass is not None: m&=(sup==superclass).values
    return np.where(m)[0]
PR=ids(superclass='ol_sensory'); print('photoreceptors (ol_sensory)',len(PR), pd.Series(typ.values[PR]).value_counts().head(10).to_dict())
LC=ids(['LC4','LPLC2','LC6','LC16','LPLC1']); print('LC4/LPLC2/LC6/LC16/LPLC1',len(LC))
DN=ids(['DNp01','DNp02','DNp04','DNp11']); print('escape DNs',len(DN), [(typ.values[i],side.values[i]) for i in DN])
MN=ids(['TTMn','DLMn a, b','DLMn c-f','PSI']); print('TTMn/DLMn/PSI',len(MN))
# direct LC -> DN synapses
sub=A[LC][:,DN].toarray();
for j,d in enumerate(DN):
    s=pd.Series(sub[:,j],index=typ.values[LC]).groupby(level=0).sum()
    print(f'  {typ.values[d]:6s}{side.values[d]}: inputs from', {k:int(v) for k,v in s.items() if v>0})
# top inputs to LC4 and LPLC2 by type
for T in ['LC4','LPLC2']:
    tgt=ids([T]); inp=np.asarray(A[:,tgt].sum(axis=1)).ravel()
    s=pd.Series(inp,index=typ.values).groupby(level=0).sum().sort_values(ascending=False)
    print(f'top inputs to {T}:',{k:int(v) for k,v in s.head(12).items()})
# hop-limited subgraph: forward reach from PR within k, backward reach from DN within k, weight>=thr
def reach(Mat, seeds, k, thr):
    B=(Mat>=thr).astype(np.float32); B.data[:]=1
    v=np.zeros(N,np.float32); v[seeds]=1; seen=v.copy()
    for _ in range(k):
        v=(B.T@v>0).astype(np.float32); seen=np.maximum(seen,v)
    return seen>0
for thr in [3,5,10]:
    for k in [3,4,5]:
        fw=reach(A,PR,k,thr); bw=reach(A.T.tocsr(),DN,k,thr)
        core=fw&bw
        n=core.sum(); nnz=A[core][:,core].nnz
        print(f'thr>={thr:2d} hops={k}: neurons={n:6d} edges={nnz:8d}', pd.Series(sup.values[core]).value_counts().head(6).to_dict())
print(f'done {time.time()-t0:.1f}s')
