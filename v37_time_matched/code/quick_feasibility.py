import sys,json,hashlib
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
R=Path('/mnt/data/HVAC_EVI_CalendarMatchedProspective_R1/work/HVAC_EVI_V25_Robustness_Research_20261011')
V21=R/'inputs/HVAC_EVI_V21_DataAllocation_Evidence'
sys.path.insert(0,str(V21/'code'))
import source_loader_A4 as L
for b,df in L.load_and_gate(V21/'raw',Path('/mnt/data/HVAC_EVI_CalendarMatchedProspective_R1/schema_gate'))[0].items():
 print('===',b,'N',len(df),flush=True)
 if b=='hospital':t=pd.to_datetime(df.DATE.astype(str)+' '+df.Time.astype(str),format='%d/%m/%Y %H:%M')
 else:t=pd.to_datetime(df.Time.astype(str),format='%Y-%m-%d %H')
 df=df.copy();df['t']=t; df['lab']=df.__label_canonical__
 dup=df.duplicated(['__ahu_canonical__','t'],keep=False);print('dup',dup.sum())
 df=df.loc[~dup]
 for c in ['Normal','RATSF','SFF']:
  sub=df[df.lab==c]
  by={u:g for u,g in sub.groupby('__ahu_canonical__') if len(g)>=128}
  keys={u:{'exact_hour':g.t.dt.strftime('%Y%m%d%H').tolist(),'day':g.t.dt.strftime('%Y%m%d').tolist(),'month_hour':g.t.dt.strftime('%Y%m%H').tolist(),'month':g.t.dt.strftime('%Y%m').tolist(),'season_hour':g.t.dt.strftime('%Y%m%H').tolist(),'hour':g.t.dt.strftime('%H').tolist()} for u,g in by.items()}
  print(c,'eligible',len(by),'rows counts',sorted((u,len(g)) for u,g in by.items())[:20])
  for kind in ['exact_hour','day','month_hour','month','hour']:
   poss=[];maxsum=[]
   for a,ka in keys.items():
    for d,kb in keys.items():
     if a==d:continue
     aa=Counter(ka[kind]);bb=Counter(kb[kind]); cc=sum(min(n,bb.get(s,0)) for s,n in aa.items())
     maxsum.append(cc)
     if cc>=64:poss.append((a,d,cc))
   print(' ',kind,'pair64',len(poss),'pairs128',sum(x>=128 for x in maxsum),'max overlap',max(maxsum,default=0),'min pairs per target',(min((sum(a!=u and d!=u for a,d,n in poss) for u in df.__ahu_canonical__.unique()),default=0)),'sample',poss[:3])
