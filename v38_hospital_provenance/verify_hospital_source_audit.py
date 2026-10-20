#!/usr/bin/env python3
"""Reconstruct Hospital date/duplicate audit from the exact public Figshare v3 source.
Usage: python verify_hospital_source_audit.py /path/to/hosptial_scientific_data.csv
Requires pandas. Does not alter input or write observations; prints aggregate JSON only.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import pandas as pd
P=argparse.ArgumentParser();P.add_argument("csv",type=Path);args=P.parse_args()
raw=args.csv.read_bytes()
sha=hashlib.sha256(raw).hexdigest()
assert sha=="66feab7fe66d77ca1c632315a61a7a93412ff00dc2fb0cd9b2485394cecca9f7", "Source version SHA mismatch: do not substitute another file"
df=pd.read_csv(args.csv,low_memory=False)
assert set(["AHU name","DATE","Time","labeling"]).issubset(df.columns)
ts=pd.to_datetime(df["DATE"].str.strip()+" "+df["Time"].str.strip(),format="%d/%m/%Y %H:%M",errors="raise")
assert ts.notna().all()
keys=["AHU name","DATE","Time"]
def duplicates(t):
    g=t.groupby(keys,dropna=False)["labeling"].agg(["size","nunique"])
    multi=g[g["size"]>1]
    assert (multi["size"]==2).all(), "Duplicate multiplicity unexpected"
    return len(multi),int(multi["size"].sum()),int((multi["nunique"]>1).sum())
subset=df[df["labeling"].isin(["Normal condition","Return air temperature fault","Supply fan fault"])]
date_arr=pd.to_datetime(df["DATE"],format="%d/%m/%Y")
result={"source_sha256":sha,"source_filename":args.csv.name,"raw_rows":len(df),"retained_three_class_rows":len(subset),"other_label_rows":len(df)-len(subset),"ahu_count":df["AHU name"].nunique(),"min_timestamp":str(ts.min()),"max_timestamp":str(ts.max()),"year_counts":{str(int(k)):int(v) for k,v in date_arr.dt.year.value_counts().items()},"unique_days":date_arr.nunique(),"raw_duplicate_keys_rows_conflicting":duplicates(df),"three_class_duplicate_keys_rows_conflicting":duplicates(subset)}
assert result["raw_rows"]==66048 and result["retained_three_class_rows"]==64513 and result["unique_days"]==344
assert result["raw_duplicate_keys_rows_conflicting"][:2]==(184,368)
assert result["three_class_duplicate_keys_rows_conflicting"]==(161,322,33)
assert result["min_timestamp"].startswith("2023-01-01") and result["max_timestamp"].startswith("2023-12-21")
print(json.dumps({"status":"PASS",**result},indent=2))
