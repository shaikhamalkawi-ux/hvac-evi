#!/usr/bin/env python3
"""Retrospective, internally time-key-locked donor-exchange experiment.
Controls selected training time-key histograms BY CONSTRUCTION, not equipment-day episodes.
Zero target information except heldout identity is consulted when acquiring training data.
No scientific revision, external preregistration or repair confirmation.
"""
from __future__ import annotations
import argparse, hashlib, itertools, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

R=Path(__file__).resolve().parent
SOURCE=R/'work/HVAC_EVI_V25_Robustness_Research_20261011'
V21=SOURCE/'inputs/HVAC_EVI_V21_DataAllocation_Evidence'
sys.path.insert(0,str(V21/'code'))
import source_loader_A4 as L
LABELS=('Normal','RATSF','SFF')


def digest_bytes(buf):return hashlib.sha256(buf).hexdigest()
def filehash(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
 return h.hexdigest()
def seed(*args):return int.from_bytes(hashlib.sha256('|'.join(map(str,args)).encode()).digest()[:8],"little")
def jsonwrite(obj,path):Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,default=str)+'\n',encoding='utf8')

def stamp(df,b):
 if b=='hospital':t=pd.to_datetime(df['DATE'].astype(str)+' '+df['Time'].astype(str),format='%d/%m/%Y %H:%M',errors='coerce')
 else:t=pd.to_datetime(df['Time'].astype(str),format='%Y-%m-%d %H',errors='coerce')
 assert t.notna().all()
 return t

def build_source(b,d,features):
 d=d.reset_index(drop=True)
 with np.load(V21/'results'/f'{b}_canonical_matrix.npz') as z:
  X=z['X'].copy();y=z['y'].copy();g=z['groups'].astype(str).copy()
 assert np.array_equal(d['__ahu_canonical__'].astype(str).to_numpy(),g),b
 assert np.array_equal(pd.Categorical(d['__label_canonical__'],categories=list(LABELS)).codes.astype(np.int8),y),b
 xraw=d[list(features)].to_numpy(float)
 assert X.shape==xraw.shape
 assert np.allclose(X,xraw,equal_nan=True,rtol=0,atol=0),b
 t=stamp(d,b)
 time=t.to_numpy(dtype='datetime64[ns]')
 dup=pd.DataFrame({'group':g,'time':time}).duplicated(['group','time'],keep=False).to_numpy()
 assert b=='hospital' or not dup.any(),b
 if b=='hospital':assert int(dup.sum())==322
 keys={
 'exact_hour':t.dt.strftime('%Y-%m-%d %H').to_numpy(dtype='U13'),
 'month_hour':t.dt.strftime('%Y-%m %H').to_numpy(dtype='U10')
 }
 assert len(time)==len(X)==len(y)==len(g)
 return X,y,g,time,dup,keys

def pool_candidates(y,g,dup,keys,kind):
 """Reference calendar stratum index based only on source rows and historical labels."""
 pool={}; all_ahu=sorted(set(g))
 for ci,c in enumerate(LABELS):
  for a in all_ahu:
   ix=np.flatnonzero((g==a)&(y==ci)&~dup)
   if len(ix)<128:continue
   d=defaultdict(list)
   for i in ix:d[str(keys[kind][i])].append(int(i))
   pool[(ci,a)]={s:np.asarray(rows,dtype=np.int32) for s,rows in d.items()}
 pairs={}
 for ci,c in enumerate(LABELS):
  eligible=sorted(a for (j,a) in pool if j==ci)
  ps=[]
  for a,b in itertools.permutations(eligible,2):
   ka=pool[ci,a];kb=pool[ci,b]
   n=sum(min(len(v),len(kb[s])) for s,v in ka.items() if s in kb)
   if n>=64:ps.append((a,b,int(n)))
  pairs[ci]=ps
 return pool,pairs,all_ahu

def choose_pair(pool,ci,a,b,rng):
 ma=pool[ci,a];mb=pool[ci,b]
 # Construct all disjoint one-to-one same-time-key pairs, then sample 64 of these.
 left=[];right=[]
 for s in sorted(set(ma)&set(mb)):
  aa=ma[s][rng.permutation(len(ma[s]))]
  bb=mb[s][rng.permutation(len(mb[s]))]
  n=min(len(aa),len(bb))
  if n:
   left.extend(aa[:n]);right.extend(bb[:n])
 assert len(left)>=64
 pick=rng.permutation(len(left))[:64]
 A=np.asarray(left,dtype=np.int32)[pick]
 B=np.asarray(right,dtype=np.int32)[pick]
 eligible_A=np.concatenate([v for v in ma.values()])
 rest=np.setdiff1d(eligible_A,A,assume_unique=True)
 assert len(rest)>=64
 stay=rest[rng.permutation(len(rest))[:64]]
 return A,B,stay

def model(name,pconf):
 return Pipeline([('impute',SimpleImputer(**pconf['preprocessing']['SimpleImputer'])),('classifier',(
  DecisionTreeClassifier if name=='DecisionTree' else RandomForestClassifier)(**pconf['fitted_models'][name]))])

def one_run(b,kind,u,r,X,y,g,t,dup,keys,pool,pairs,pc):
 train={'k1':[],'k2':[]}; donors=[];classes=[]
 for ci,c in enumerate(LABELS):
  possible=[(a,b,n) for a,b,n in pairs[ci] if a!=u and b!=u]
  assert possible,(b,kind,u,c)
  rng=np.random.default_rng(seed('HVAC-calendar-exchange',pc['date'],kind,b,u,r,c))
  a,partner,overlap=possible[int(rng.integers(len(possible)))]
  matchedA,matchedB,stayA=choose_pair(pool,ci,a,partner,rng)
  arm1=np.concatenate([matchedA,stayA]);arm2=np.concatenate([matchedB,stayA])
  assert len(arm1)==len(arm2)==128
  assert Counter(keys[kind][arm1])==Counter(keys[kind][arm2])
  assert len(set(arm1))==128 and len(set(arm2))==128
  assert set(g[arm1])=={a} and set(g[arm2])=={a,partner}
  assert np.all(y[arm1]==ci) and np.all(y[arm2]==ci)
  assert not np.any(dup[arm1]) and not np.any(dup[arm2])
  train['k1'].append(arm1);train['k2'].append(arm2)
  donors.append({'building':b,'kind':kind,'target':u,'repeat':r,'class':c,'anchor':a,'partner':partner,'candidate_overlap':overlap,'candidate_feasible_pairs':len(possible),
   'matched_class_rows':64,'retained_class_rows':64,'matching_exact_key_count':len(set(keys[kind][matchedA]))})
 for arm in train:
  train[arm]=np.sort(np.concatenate(train[arm])).astype(np.int32)
  idx=train[arm]
  assert len(idx)==384 and len(set(idx.tolist()))==384
  assert tuple(np.bincount(y[idx],minlength=3))==(128,128,128)
  assert (g[idx]!=u).all() and not dup[idx].any()
 for ci in range(3):
  assert Counter(keys[kind][train['k1'][y[train['k1']]==ci]])==Counter(keys[kind][train['k2'][y[train['k2']]==ci]])
 te=np.flatnonzero((g==u)&~dup)
 assert len(te)>0
 out=[]; pred={}; archive={}; diag=[]
 for arm,idx in train.items():
  cal_days=len(np.unique(t[idx].astype('datetime64[D]')))
  ahu_days=len(set(zip(g[idx],t[idx].astype('datetime64[D]'))))
  diag.append({'building':b,'kind':kind,'target':u,'repeat':r,'arm':arm,'unique_calendar_days':cal_days,'AHU_calendar_days':ahu_days,
    'distinct_source_AHUs':len(set(g[idx])),'index_sha256':digest_bytes(idx.astype('<i4').tobytes()),'n_test':len(te)})
  archive[f'{u}__r{r:02d}__{arm}']=idx
  for name in pc['fitted_models']:
   mdl=model(name,pc)
   mdl.fit(X[idx],y[idx]); yp=mdl.predict(X[te]).astype(np.int8)
   cm=np.bincount(y[te]*3+yp,minlength=9).reshape(3,3)
   key=f'{u}__r{r:02d}__{arm}__{name}'
   pred[key]=yp
   for ci,c in enumerate(LABELS):
    out.append({'building':b,'kind':kind,'target':u,'repeat':r,'arm':arm,'model':name,'class':c,
      'n_test':int(cm[ci].sum()),'correct':int(cm[ci,ci]),'recall':float(cm[ci,ci]/cm[ci].sum()) if cm[ci].sum() else float('nan'),
      'false_positives':int(cm[:,ci].sum()-cm[ci,ci]),'true_class_support':int(cm[ci].sum())})
 return out,donors,diag,archive,pred

def summarize(df,feas):
 score=[];class_scores=[];looc=[]
 for (kind,b,m,arm),g0 in df.groupby(['kind','building','model','arm'],sort=True):
  represented=g0[g0.n_test>0]
  valid=represented.groupby(['class','target','repeat'],sort=True)['recall'].mean().reset_index()
  cls=valid.groupby(['class'])['recall'].mean()
  assert len(cls)==3
  Q=float(cls.mean())
  score.append({'kind':kind,'building':b,'model':m,'arm':arm,'Q':Q,'Q_pct':100*Q,'eligible_targets':int(g0.target.nunique()),'targets_Normal':int(represented[represented['class']=='Normal'].target.nunique()),'targets_RATSF':int(represented[represented['class']=='RATSF'].target.nunique()),'targets_SFF':int(represented[represented['class']=='SFF'].target.nunique())})
  for c,v in cls.items():class_scores.append({'kind':kind,'building':b,'model':m,'arm':arm,'class':c,'mean_target_AHU_recall':float(v),'target_AHUs':int(represented[represented['class']==c].target.nunique())})
 s=pd.DataFrame(score)
 wide=s.pivot(index=['kind','building','model'],columns='arm',values='Q')
 gains=(wide.k2-wide.k1).rename('delta_Q').reset_index()
 gains['delta_pp']=100*gains.delta_Q
 for (kind,b,m),x in df.groupby(['kind','building','model']):
  for excluded in sorted(x.target.unique()):
   remain=x[(x.target!=excluded)&(x.n_test>0)]
   rr=remain.groupby(['arm','class','target'])['recall'].mean().groupby(['arm','class']).mean().groupby('arm').mean()
   assert {'k1','k2'}<=set(rr.index)
   looc.append({'kind':kind,'building':b,'model':m,'omitted_target':excluded,'delta_pp_without_target':100*float(rr['k2']-rr['k1'])})
 return s,gains,pd.DataFrame(class_scores),pd.DataFrame(looc)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['gate','exact_hour','month_hour','summary'],required=True);parser.add_argument('--jobs',type=int,default=4);args=parser.parse_args()
 pc=json.loads((R/'protocol_calendar_matched.json').read_text())
 assert filehash(R/'protocol_calendar_matched.json')==(R/'PROTOCOL_LOCK_SHA256.txt').read_text().split()[0]
 rawmanifest=json.loads((V21/'raw/RAW_DATA_MANIFEST_SHA256.json').read_text())
 for f,rec in rawmanifest.items():assert filehash(V21/'raw'/f)==rec['sha256'],f
 (R/'schema_gate').mkdir(exist_ok=True)
 data,schemas,feat=L.load_and_gate(V21/'raw',R/'schema_gate')
 (R/'results').mkdir(exist_ok=True)
 modes=['exact_hour','month_hour'] if args.mode=='gate' else ([args.mode] if args.mode in ('exact_hour','month_hour') else [])
 if args.mode=='summary':
  d=pd.concat([pd.read_csv(R/'results'/f'{kind}_cell_counts.csv') for kind in ('exact_hour','month_hour')],ignore_index=True)
  feas=pd.concat([pd.read_csv(R/'results'/f'{kind}_feasibility.csv') for kind in ('exact_hour','month_hour')])
  s,g,c,l=summarize(d,feas)
  s.to_csv(R/'results/SCORE_Q_BY_BUILDING.csv',index=False);g.to_csv(R/'results/MATCHED_SOURCE_DIVERSITY_GAINS.csv',index=False);c.to_csv(R/'results/Q_CLASS_COMPONENTS.csv',index=False);l.to_csv(R/'results/LEAVE_ONE_TARGET_AHU_OUT.csv',index=False)
  print('RESULTS',g.to_string(index=False));print('SOURCE FROZEN',filehash(R/'protocol_calendar_matched.json'))
  return
 gate=[]
 for kind in modes:
  pool_records=[];full_records=[]
  for b,d in data.items():
   X,y,grp,t,dup,keys=build_source(b,d,feat)
   pool,pairs,all_ahu=pool_candidates(y,grp,dup,keys,kind)
   for u in all_ahu:
    entry={'kind':kind,'building':b,'target':u,'duplicate_source_rows_dropped':int(dup.sum()),'test_rows':int(((grp==u)&~dup).sum())}
    for ci,c in enumerate(LABELS):
     good=[(a,p,z) for a,p,z in pairs[ci] if a!=u and p!=u]
     entry[f'{c}_eligible_ordered_pairs']=len(good)
     entry[f'{c}_max_matched_pairs']=max([z for _,_,z in good],default=0)
    entry['eligible_all_classes']=all(entry[f'{c}_eligible_ordered_pairs']>0 for c in LABELS)
    pool_records.append(entry)
   print('GATE',kind,b,'eligible',sum(q['eligible_all_classes'] for q in pool_records if q['building']==b),'total',len(all_ahu),flush=True)
   if args.mode=='gate':continue
   targets=[q['target'] for q in pool_records if q['building']==b and q['eligible_all_classes']]
   if not targets:continue
   # Parallel is at held-out target × repeat (four fixed model fits per job).
   results=Parallel(n_jobs=args.jobs,backend='threading')(
    delayed(one_run)(b,kind,u,r,X,y,grp,t,dup,keys,pool,pairs,pc)
    for u in targets for r in range(pc['repetitions']))
   cells=[];donors=[];diags=[];training={};preds={}
   for cr,dr,gg,aa,pp in results:
    cells.extend(cr);donors.extend(dr);diags.extend(gg);training.update(aa);preds.update(pp)
   assert len(cells)==len(targets)*pc['repetitions']*2*len(pc['fitted_models'])*3
   # Re-audit stored predictions against saved matrices and target class truth, never against fit objects.
   for k,v in preds.items():
    u,rr,arm,name=k.split('__');te=np.flatnonzero((grp==u)&~dup)
    assert len(v)==len(te)
    counts=np.bincount(y[te]*3+v,minlength=9).reshape(3,3)
    crows=[z for z in cells if z['target']==u and z['repeat']==int(rr[1:]) and z['arm']==arm and z['model']==name]
    assert len(crows)==3 and all(z['correct']==counts[ci,ci] and z['n_test']==counts[ci].sum() for ci,z in enumerate(crows))
   np.savez_compressed(R/'results'/f'{kind}_{b}_training_indices.npz',**training)
   np.savez_compressed(R/'results'/f'{kind}_{b}_predictions.npz',**preds)
   full_records+=cells
   pd.DataFrame(donors).to_csv(R/'results'/f'{kind}_{b}_donor_log.csv',index=False)
   pd.DataFrame(diags).to_csv(R/'results'/f'{kind}_{b}_timing_audit.csv',index=False)
   print('FITS',kind,b,'targets',len(targets),'reps',pc['repetitions'],'models',len(pc['fitted_models']),'fits',len(preds),'cell_rows',len(cells),flush=True)
  pd.DataFrame(pool_records).to_csv(R/'results'/f'{kind}_feasibility.csv',index=False)
  if args.mode!='gate':
   df=pd.DataFrame(full_records)
   df.to_csv(R/'results'/f'{kind}_cell_counts.csv',index=False)
   assert (df.groupby(['building','target','repeat','arm','model']).size()==3).all()
   log={'mode':kind,'source_manifest_sha256':{f:rec['sha256'] for f,rec in rawmanifest.items()},'model_fits':len(df)//3,'feasible_target_AHUs':len(pd.DataFrame(pool_records).query('eligible_all_classes==True')),'class_target_cells':len(df),'protocol_sha256':filehash(R/'protocol_calendar_matched.json'),'source_V25_zip_sha256':filehash(Path('/mnt/data/HVAC_EVI_V25_Executed_Robustness_Evidence_20261011.zip'))}
   jsonwrite(log,R/'results'/f'{kind}_execution_manifest.json')
   print('FINISHED',kind,log['model_fits'],flush=True)

if __name__=='__main__':main()
