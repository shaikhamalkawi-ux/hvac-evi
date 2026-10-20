from pathlib import Path
import json,hashlib,sys
import pandas as pd,numpy as np
root=Path(__file__).resolve().parent
sys.path.insert(0,str(root));import calendar_matched_experiment as m
shards=root/'shards';o=root/'results';o.mkdir(exist_ok=True)
for kind in ('exact_hour','month_hour'):
 for suffix,target in [('cells',f'{kind}_cell_counts.csv'),('donors',f'{kind}_donor_log.csv'),('timing',f'{kind}_timing_audit.csv')]:
  files=sorted(shards.glob(f'{kind}_*_{suffix}.csv'))
  assert len(files)==6,(kind,suffix,len(files),files)
  z=pd.concat([pd.read_csv(f) for f in files],ignore_index=True)
  z=z.sort_values(['building','target','repeat']+(['arm','model','class'] if suffix=='cells' else ['class'] if suffix=='donors' else ['arm'])).reset_index(drop=True)
  z.to_csv(o/target,index=False)
  print(kind,suffix,'rows',len(z))
 score=pd.read_csv(o/f'{kind}_cell_counts.csv')
 assert len(score)==(40 if kind=='exact_hour' else 41)*10*4*3
 for (b,t),x in score.groupby(['building','target']):
  assert len(x)==10*4*3,(b,t,len(x))
  assert x.n_test.nunique()<=3

cells=pd.concat([pd.read_csv(o/f'{kind}_cell_counts.csv') for kind in ('exact_hour','month_hour')],ignore_index=True)
feas=pd.concat([pd.read_csv(o/f'{kind}_feasibility.csv') for kind in ('exact_hour','month_hour')],ignore_index=True)
s,g,c,l=m.summarize(cells,feas)
s.to_csv(o/'SCORE_Q_BY_BUILDING.csv',index=False)
g.to_csv(o/'MATCHED_SOURCE_DIVERSITY_GAINS.csv',index=False)
c.to_csv(o/'Q_CLASS_COMPONENTS.csv',index=False)
l.to_csv(o/'LEAVE_ONE_TARGET_AHU_OUT.csv',index=False)
print('\nDELTA Q (pp)\n',g.to_string(index=False))
print('\nSCORES\n',s.to_string(index=False))
print('\nCLASSES\n',c.to_string(index=False))
print('\nTARGET LEAVE ONE OUT\n',l.groupby(['kind','building','model']).delta_pp_without_target.agg(['min','max']).to_string())
