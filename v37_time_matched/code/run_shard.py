import sys,argparse,json
from pathlib import Path
import pandas as pd,numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root))
import calendar_matched_experiment as e
p=argparse.ArgumentParser()
for v in ['mode','building']:p.add_argument('--'+v,required=True)
for v in ['start','end']:p.add_argument('--'+v,type=int,required=True)
a=p.parse_args()
pc=json.loads((root/'protocol_calendar_matched.json').read_text())
assert e.filehash(root/'protocol_calendar_matched.json')==(root/'PROTOCOL_LOCK_SHA256.txt').read_text().split()[0]
(root/'schema_gate').mkdir(exist_ok=True);(root/'shards').mkdir(exist_ok=True)
data,_,features=e.L.load_and_gate(e.V21/'raw',root/'schema_gate')
X,y,g,t,dup,keys=e.build_source(a.building,data[a.building],features)
pool,pairs,us=e.pool_candidates(y,g,dup,keys,a.mode)
valid=[u for u in us if all(any(a1!=u and b1!=u for a1,b1,n in pairs[ci]) for ci in range(3))]
targets=valid[a.start:a.end]
assert targets
cells=[];donors=[];diag=[];tridx={};pred={}
for u in targets:
 for r in range(pc['repetitions']):
  c,d,z,ix,yp=e.one_run(a.building,a.mode,u,r,X,y,g,t,dup,keys,pool,pairs,pc)
  cells+=c;donors+=d;diag+=z;tridx.update(ix);pred.update(yp)
 print('TARGET',a.mode,a.building,u,flush=True)
s=f'{a.mode}_{a.building}_{a.start:02d}-{a.end:02d}'
pd.DataFrame(cells).to_csv(root/'shards'/f'{s}_cells.csv',index=False)
pd.DataFrame(donors).to_csv(root/'shards'/f'{s}_donors.csv',index=False)
pd.DataFrame(diag).to_csv(root/'shards'/f'{s}_timing.csv',index=False)
np.savez_compressed(root/'shards'/f'{s}_indices.npz',**tridx)
np.savez_compressed(root/'shards'/f'{s}_predictions.npz',**pred)
e.jsonwrite({'mode':a.mode,'building':a.building,'targets':targets,'fits':len(pred),'cells':len(cells),'protocol_sha256':e.filehash(root/'protocol_calendar_matched.json'),'eligible_target_names':valid},root/'shards'/f'{s}_qa.json')
print('SHARD_COMPLETE',s,len(pred),flush=True)
