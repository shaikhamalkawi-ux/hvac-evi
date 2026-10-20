#!/usr/bin/env python3
"""Public aggregate consistency + SHA-256 verifier for the V37 repository folder.

This script is NOT a full refit, source-data validation, or statistical replication.
All tests concern consistency of already-published numeric summaries.
"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def close(a, b, label):
    check(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=1e-9, rel_tol=0), label)


def load_csv(path):
    with (ROOT/path).open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def key(r):
    return (r['kind'], r['building'], r['model'])


def main():
    # Verify that the archived public table files exactly match this release manifest.
    lines=(ROOT/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines()
    observed=set()
    for line in lines:
        sha,path=line.split('  ',1)
        check(path not in observed, f'duplicate manifest item: {path}')
        observed.add(path)
        raw=(ROOT/path).read_bytes()
        check(hashlib.sha256(raw).hexdigest()==sha, f'checksum mismatch: {path}')
    files={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()}
    check(files-observed=={'SHA256SUMS.txt'}, 'unmanifested files (or removed items)')

    q=load_csv('results/SCORE_Q_BY_BUILDING.csv')
    q2=load_csv('results/INDEPENDENT_SCORE_VERIFICATION.csv')
    component=load_csv('results/Q_CLASS_COMPONENTS.csv')
    gain=load_csv('results/MATCHED_SOURCE_DIVERSITY_GAINS.csv')
    f1=load_csv('results/POOLED_MACROF1_BY_DESIGN.csv')
    f1diff=load_csv('results/POOLED_MACROF1_DELTA_BY_DESIGN.csv')
    thr=load_csv('results/THRESHOLD_SENSITIVITY_DESCRIPTIVE.csv')
    omitted=load_csv('results/LEAVE_ONE_TARGET_AHU_OUT.csv')
    qa=json.loads((ROOT/'qa/INDEPENDENT_VERIFIER_QA.json').read_text(encoding='utf-8'))
    proto=json.loads((ROOT/'protocol_calendar_matched.json').read_text(encoding='utf-8'))

    check((len(q),len(q2),len(component),len(gain),len(f1),len(f1diff),len(thr),len(omitted))
          == (24,24,72,12,24,12,60,162),'unexpected row counts')
    Q={(key(r),r['arm']):r for r in q}
    Q2={(key(r),r['arm']):r for r in q2}
    check(len(Q)==len(Q2)==24,'nonunique Q rows')
    for k,r in Q.items():
        close(float(r['Q'])*100,float(r['Q_pct']),'Q proportion-to-percent')
        close(float(r['Q_pct']),float(Q2[k]['Q_rebuilt_pct']),'independent-score table vs Q')
    comps={}
    for r in component:
        comps.setdefault((key(r),r['arm']),[]).append(float(r['mean_target_AHU_recall']))
    check(len(comps)==24 and all(len(v)==3 for v in comps.values()),'class component count')
    for k,v in comps.items():
        close(sum(v)/3*100,float(Q[k]['Q_pct']),'Q vs mean across class components')

    F={(key(r),r['arm']):r for r in f1}
    check(len(F)==24,'F1 key duplicates')
    for k,r in F.items():
        close(float(r['mean'])*100,float(r['F1_pct']),'F1 proportion-to-percent')
    for r in gain:
        k=key(r)
        d=float(Q[(k,'k2')]['Q_pct'])-float(Q[(k,'k1')]['Q_pct'])
        close(d,float(r['delta_pp']),'Q arm difference')
        close(float(r['delta_Q'])*100,d,'gain delta in proportion')
        check(d>0,f'expected positive aggregate gain missing for {k}')
    for r in f1diff:
        k=key(r)
        f_k1=float(F[(k,'k1')]['mean'])
        f_k2=float(F[(k,'k2')]['mean'])
        close(f_k1,float(r['k1']),'pooled macro-F1 first arm')
        close(f_k2,float(r['k2']),'pooled macro-F1 second arm')
        close((f_k2-f_k1)*100,float(r['delta_F1_pp']),'pooled macro-F1 difference')

    # Threshold=1 uses the same target-class population as primary Q; others change estimand.
    base_threshold=[r for r in thr if int(r['support_threshold'])==1]
    check(len(base_threshold)==12,'threshold=1 count')
    for r in base_threshold:
        k=key(r)
        check(r['status']=='DEFINED','undefined threshold=1')
        close(float(r['Q_k1_pct']),float(Q[(k,'k1')]['Q_pct']),'threshold 1 first arm')
        close(float(r['Q_k2_pct']),float(Q[(k,'k2')]['Q_pct']),'threshold 1 second arm')
        close(float(r['delta_pp']),float(Q[(k,'k2')]['Q_pct'])-float(Q[(k,'k1')]['Q_pct']),'threshold delta')

    target_count={'auditorium':13,'hospital':8,'office':20}
    for r in q:
        n=target_count[r['building']] - (1 if r['building']=='hospital' and r['kind']=='exact_hour' else 0)
        check(int(r['eligible_targets'])==n, f'eligible target mismatch: {r}')
    check(len({(r['kind'],r['building'],r['model'],r['omitted_target']) for r in omitted})==162,
          'leave-one-target-AHU duplicated rows')
    counters=qa['counters']
    for field,value in {'paired_allocations_checked':810,'matched_label_time_histograms_checked':2430,
                        'saved_prediction_cell_checks':9720,'actual_independent_retrained_fits':48,
                        'feasible_targets_exact_hour':40,'feasible_targets_month_hour':41}.items():
        check(counters[field]==value,f'recorded QA counter {field}')
    check(proto['source_training_rows_per_class']==128 and proto['repetitions']==10,'protocol invariant')
    print('PASS: manifest hashes, 12 aggregate gains, 24 Q rows, 72 class components, 12 pooled F1 deltas,')
    print('      12 threshold=1 contrasts, all eligibility counts, and archived QA/protocol metadata.')
    print('LIMIT: this is a public-table consistency check, not a 3,240-fit raw-data replay.')


if __name__=='__main__':
    main()
