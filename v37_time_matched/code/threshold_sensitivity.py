"""Post-hoc descriptive support-threshold diagnostics (fixed support grid inherited from V25)."""
from pathlib import Path
import numpy as np,pandas as pd
r=Path(__file__).resolve().parent;o=r/'results'
d=pd.concat([pd.read_csv(o/f'{k}_cell_counts.csv') for k in ('exact_hour','month_hour')],ignore_index=True)
rows=[]
for (kind,b,model),x in d.groupby(['kind','building','model']):
 for th in [1,10,30,100,300]:
  v=x.loc[x.n_test>=th];q={};su={}
  for arm,g in v.groupby('arm'):
   g=g.copy(); classmeans=g.groupby(['class','target']).recall.mean().groupby('class').mean()
   for c in ('Normal','RATSF','SFF'):su[c]=int(g.loc[g['class']==c,'target'].nunique())
   q[arm]=float(classmeans.mean()) if len(classmeans)==3 else np.nan
  supported=(len(q)==2 and all(np.isfinite(vv) for vv in q.values()))
  rows.append({'kind':kind,'building':b,'model':model,'support_threshold':th,'status':'DEFINED' if supported else 'UNDEFINED_MISSING_CLASS',
   'Q_k1_pct':100*q.get('k1',np.nan),'Q_k2_pct':100*q.get('k2',np.nan),'delta_pp':100*(q.get('k2',np.nan)-q.get('k1',np.nan)),
   **{f'targets_{c}':su.get(c,0) for c in ('Normal','RATSF','SFF')}})
pd.DataFrame(rows).to_csv(o/'THRESHOLD_SENSITIVITY_DESCRIPTIVE.csv',index=False)
wide=d.groupby(['kind','building','model','target','class','arm'])['recall'].mean().unstack('arm')
wide['delta_pp']=100*(wide['k2']-wide['k1'])
w=wide.reset_index().sort_values(['kind','building','model','target','class'])
w.to_csv(o/'PER_AHU_CLASS_CONTRASTS.csv',index=False)
print('SUPPORT GRID')
print(pd.DataFrame(rows).query('support_threshold in [1,30,300]')[['kind','building','model','support_threshold','status','delta_pp','targets_RATSF']].to_string(index=False))
print('NEGATIVE CLASS-TARGET CELLS',(wide.delta_pp<0).sum(),'of',len(wide.dropna(subset=['delta_pp'])))
