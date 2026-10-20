"""Independent source/label/time key/prediction audit of all frozen-R1 fit outputs."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
import numpy as np,pandas as pd
r=Path(__file__).resolve().parent
sys.path.insert(0,str(r));import calendar_matched_experiment as e
P=r/'shards';O=r/'results';(r/'qa').mkdir(exist_ok=True)
pc=json.loads((r/'protocol_calendar_matched.json').read_text())
assert e.filehash(r/'protocol_calendar_matched.json')==(r/'PROTOCOL_LOCK_SHA256.txt').read_text().split()[0]
manifest=json.loads((e.V21/'raw/RAW_DATA_MANIFEST_SHA256.json').read_text())
for fn,x in manifest.items():assert e.filehash(e.V21/'raw'/fn)==x['sha256']
data,_,features=e.L.load_and_gate(e.V21/'raw',r/'schema_gate')
cnt=Counter(); rebuilt=[];shard_info=[];test_count=0;refit_count=0;total_eligible=0
for kind in ('exact_hour','month_hour'):
 for b,df in data.items():
  X,y,g,t,dup,keys=e.build_source(b,df,features)
  folders=sorted(P.glob(f'{kind}_{b}_*_qa.json'))
  assert len(folders)>=1
  done=set()
  for f in folders:
   base=f.name[:-8]
   q=json.loads(f.read_text());assert q['protocol_sha256']==e.filehash(r/'protocol_calendar_matched.json')
   assert set(q['targets']).isdisjoint(done)
   done|=set(q['targets'])
   targetrows=pd.read_csv(P/f'{base}_cells.csv')
   trainarr=np.load(P/f'{base}_indices.npz',allow_pickle=False)
   preds=np.load(P/f'{base}_predictions.npz',allow_pickle=False)
   assert len(trainarr.files)==len(q['targets'])*pc['repetitions']*2
   assert len(preds.files)==len(q['targets'])*pc['repetitions']*2*len(pc['fitted_models'])
   for u in q['targets']:
    te=np.flatnonzero((g==u)&~dup)
    test_truth=y[te]
    for rep in range(pc['repetitions']):
     i1=trainarr[f'{u}__r{rep:02d}__k1'];i2=trainarr[f'{u}__r{rep:02d}__k2']
     for arm,ii in [('k1',i1),('k2',i2)]:
      assert ii.dtype==np.int32 and len(ii)==384 and len(set(ii.tolist()))==384
      assert tuple(np.bincount(y[ii],minlength=3))==(128,128,128)
      assert not (g[ii]==u).any() and not dup[ii].any()
      for ci in range(3):
       expected=(1 if arm=='k1' else 2)
       assert len(set(g[ii[y[ii]==ci]]))==expected
       cnt['source_class_units_checked']+=1
     for ci in range(3):
      assert Counter(keys[kind][i1[y[i1]==ci]])==Counter(keys[kind][i2[y[i2]==ci]])
      cnt['matched_label_time_histograms_checked']+=1
     if kind=='exact_hour':assert set(t[i1])==set(t[i2]), (kind,b,u,rep)
     cnt['paired_allocations_checked']+=1
     for arm,ii in [('k1',i1),('k2',i2)]:
      for model in pc['fitted_models']:
       tag=f'{u}__r{rep:02d}__{arm}__{model}'
       yp=preds[tag];assert len(yp)==len(te)
       cm=np.bincount(test_truth*3+yp,minlength=9).reshape(3,3)
       rows=targetrows[(targetrows.target==u)&(targetrows.repeat==rep)&(targetrows.arm==arm)&(targetrows.model==model)]
       assert len(rows)==3
       for ci,c in enumerate(e.LABELS):
        v=rows.loc[rows['class']==c].iloc[0]
        assert int(v.n_test)==int(cm[ci].sum()) and int(v.correct)==int(cm[ci,ci])
        assert int(v.false_positives)==int(cm[:,ci].sum()-cm[ci,ci])
        rebuilt.append({'kind':kind,'building':b,'model':model,'arm':arm,'target':u,'repeat':rep,'class':c,
          'n_test':int(cm[ci].sum()),'correct':int(cm[ci,ci]),'recall':float(cm[ci,ci]/cm[ci].sum()) if cm[ci].sum()>0 else np.nan})
        cnt['saved_prediction_cell_checks']+=1
       # Independent model replay (first target and first repeat per building and design).
       if u==q['targets'][0] and rep==0:
        m=e.model(model,pc);m.fit(X[ii],y[ii]);assert np.array_equal(m.predict(X[te]),yp)
        cnt['actual_independent_retrained_fits']+=1
   shard_info.append({'shard':base,'targets':q['targets'],'n_preds':len(preds.files),'n_training_indices':len(trainarr.files),'cell_rows':len(targetrows)})
   preds.close();trainarr.close()
  expected=13 if b=='auditorium' else (7 if kind=='exact_hour' else 8) if b=='hospital' else 20
  assert len(done)==expected,(b,kind,len(done),expected)
  cnt['feasible_targets_'+kind]+=len(done)
  print('VERIFIED',kind,b,'targets',len(done),flush=True)

D=pd.DataFrame(rebuilt)
assert len(D)==9720
S=[]
for (kind,b,m,arm),df in D.groupby(['kind','building','model','arm'],sort=True):
 cl=df[df.n_test>0].groupby(['class','target'])['recall'].mean().groupby('class').mean()
 assert len(cl)==3
 S.append({'kind':kind,'building':b,'model':m,'arm':arm,'Q_rebuilt_pct':100*cl.mean()})
S=pd.DataFrame(S).sort_values(['kind','building','model','arm']).reset_index(drop=True)
T=pd.read_csv(O/'SCORE_Q_BY_BUILDING.csv').sort_values(['kind','building','model','arm']).reset_index(drop=True)
assert (S[['kind','building','model','arm']]==T[['kind','building','model','arm']]).all().all()
assert np.allclose(S.Q_rebuilt_pct,T.Q_pct,atol=1e-11,rtol=0)
S.to_csv(O/'INDEPENDENT_SCORE_VERIFICATION.csv',index=False)
assert cnt['paired_allocations_checked']==810
assert cnt['matched_label_time_histograms_checked']==2430
assert cnt['saved_prediction_cell_checks']==9720
assert cnt['actual_independent_retrained_fits']==48
# Baseline original V36R2 is a distinct design; preserve all 12 Table 3 values verbatim.
prev=pd.read_csv('/mnt/data/HVAC_EVI_PostRejection_TargetDayGate_R1_20261020/Q_REPLAY_V36R2_TABLE3_CHECK.csv')
assert len(prev)==12 and (prev.abs_delta_pp<0.005).all()
qa={'stage':'fully verified retrospective time-key-matched donor-exchange study, manuscript untouched',
    'independent_replay_Q_entries':len(S), 'independent_replay_Q_max_abs_difference_pct':float(np.max(np.abs(S.Q_rebuilt_pct-T.Q_pct))),
    'source_wang_raw_SHA256_all_match':True,'frozen_protocol_SHA256':e.filehash(r/'protocol_calendar_matched.json'),
    'source_V25_SHA256':e.filehash(Path('/mnt/data/HVAC_EVI_V25_Executed_Robustness_Evidence_20261011.zip')),
    'archives':shard_info,'counters':dict(cnt), 'original_V36R2_table3_12of12_locked_within_display_rounding':True,
    'scope_warning':'No causal equipment-identity inference, no hospital N06 strict-hour evidence, no external validation'}
e.jsonwrite(qa,r/'qa/INDEPENDENT_VERIFIER_QA.json')
print('QA PASS',json.dumps(dict(cnt),sort_keys=True));print('Q MAX DIFF',qa['independent_replay_Q_max_abs_difference_pct'])
