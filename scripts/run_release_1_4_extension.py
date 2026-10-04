from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
IN=ROOT/'core_inputs'; OUT=ROOT/'generated'; OUT.mkdir(exist_ok=True)

events=pd.read_csv(IN/'predist_admitted_event_vectors.csv')
features=[c for c in events.columns if '__' in c]
operational=[c for c in features if c.endswith('__robust_shift') or c.endswith('__iqr_ratio')]
quality=[c for c in features if c.endswith('__event_missing') or c.endswith('__train_missing')]

def calc(train,target,cols,loq,hiq):
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    tr=imp.fit_transform(train[cols]); te=imp.transform(target[cols])
    lo=np.quantile(tr,loq,axis=0) if loq>0 else np.min(tr,axis=0)
    hi=np.quantile(tr,hiq,axis=0) if hiq<1 else np.max(tr,axis=0)
    inside=(te>=lo)&(te<=hi); v=(~inside).sum(axis=1)
    return {'N':int(len(te)),'Supported':int((v==0).sum()),'n0':int((v==0).sum()),'n1':int((v==1).sum()),'n2':int((v==2).sum()),'n3_5':int(((v>=3)&(v<=5)).sum()),'nGT5':int((v>5).sum()),'median_violations':float(np.median(v)),'mean_violations':float(np.mean(v)),'q25_violations':float(np.quantile(v,.25)),'q75_violations':float(np.quantile(v,.75)),'feature_failure_counts':{cols[j]:int((~inside[:,j]).sum()) for j in range(len(cols))}}

rows=[]; details={}
for a,b in [(1,2),(2,1)]:
    tr=events[events.manufacturer==a]; te=events[events.manufacturer==b]
    for bn,lo,hi in [('minmax',0,1),('p01_p99',.01,.99),('p05_p95',.05,.95)]:
        for fn,cols in [('full_28',features),('operational_14',operational),('quality_14',quality)]:
            r=calc(tr,te,cols,lo,hi); rows.append({'Direction':f'M{a}->M{b}','Bounds':bn,'FeatureFamily':fn,**{k:v for k,v in r.items() if k!='feature_failure_counts'}}); details[f'M{a}_to_M{b}/{bn}/{fn}']=r
pd.DataFrame(rows).to_csv(OUT/'predist_support_violation_decomposition.csv',index=False)
(OUT/'predist_support_violation_decomposition.json').write_text(json.dumps(details,indent=2),encoding='utf-8')

ornl=pd.read_csv(IN/'ornl_matched_class_metrics.csv')
o=(ornl[(ornl.model=='DecisionTree') & (ornl.design.isin(['row_matched','group_matched']))].pivot(index='class_label',columns='design',values='recall').reset_index())
o['delta_day_minus_row']=o['group_matched']-o['row_matched']; o.sort_values('delta_day_minus_row').to_csv(OUT/'ornl_classwise_recall_primary.csv',index=False)

boiler=pd.read_csv(IN/'boiler_matched_class_metrics.csv')
b=(boiler[(boiler.model=='DecisionTree') & (boiler.design.isin(['final_quarter_all','final_quarter_basic']))].pivot(index='class_label',columns='design',values='recall').reset_index())
b['delta_basic_minus_all']=b['final_quarter_basic']-b['final_quarter_all']; b.sort_values('delta_basic_minus_all').to_csv(OUT/'boiler_classwise_recall_primary.csv',index=False)

cards=pd.read_csv(IN/'HVAC_EVI_Release_1_1_Active_100_Cards.csv'); t=cards[cards.card_type=='support_qualified_transfer'].copy(); t['target_macro_f1']=t.auxiliary_metrics_json.apply(lambda s: json.loads(s)['target_macro_f1']); t['strict_support']=t.support_value.astype(float); t['direction']=t['target'].str.replace('Manufacturer ','M',regex=False).str.replace(' to Manufacturer ',' to M',regex=False); t.to_csv(OUT/'predist_transfer_score_support_points.csv',index=False)

# compact support-decomposition figure
D=pd.DataFrame(rows); fig,axes=plt.subplots(2,1,figsize=(8.4,6.2))
for ax,(direction,sub) in zip(axes,D.groupby('Direction')):
    bounds=['minmax','p01_p99','p05_p95']; fams=['full_28','operational_14','quality_14']; x=np.arange(3); w=.23
    for i,fam in enumerate(fams):
        vals=[int(sub[(sub.Bounds==bb)&(sub.FeatureFamily==fam)].Supported.iloc[0]) for bb in bounds]
        ax.bar(x+(i-1)*w,vals,w,label={'full_28':'Full 28','operational_14':'Operational 14','quality_14':'Quality 14'}[fam])
    ax.set_xticks(x); ax.set_xticklabels(['Min-max','1st-99th','5th-95th']); ax.set_ylabel('Supported target events'); ax.set_title(direction); ax.grid(axis='y',alpha=.2)
axes[0].legend(frameon=False,ncol=3); axes[-1].set_xlabel('Declared support bounds'); fig.suptitle('PreDist support failure changes by feature family and bound definition'); fig.tight_layout(rect=[0,0,1,.96]); fig.savefig(OUT/'Figure_4_PreDist_Support_Decomposition.png',dpi=220,bbox_inches='tight')
