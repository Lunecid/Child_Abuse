#!/usr/bin/env python3
"""Descriptive allocation benchmarks and threshold sample composition.

Reads aggregate counts by default. --raw-zips regenerates threshold composition
from authorized source ZIPs; no individual records or narratives are exported.
"""
from pathlib import Path
import argparse,csv,json,zipfile,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis/contract_outputs'
KO=['방임','정서학대','신체학대','성학대']; EN=['Neglect','Emotional abuse','Physical abuse','Sexual abuse']

def uniform():
 j=pd.read_csv(ROOT/'analysis/sensitivity_outputs/combination_record_matrix.csv'); rows=[]
 assert j.profile_n.sum()==3236
 for scope,kmin in [('all',1),('single',1),('multi',2)]:
  q=j[j.n_domains.ge(kmin)]
  if scope=='single':q=q[q.n_domains.eq(1)]
  for restricted in [False,True]:
   for d,e in zip(KO,EN):
    a=q[q.profile.str.split('+').apply(lambda x:d in x)]
    count=a.profile_n-a['해당 없음'] if restricted else a.profile_n
    n=int(count.sum());match=int(a[d].sum());expected=float((count/a.n_domains).sum())
    rows.append(dict(scope=scope,exclude_none=restricted,domain=e,n=n,same=match,nonrepresentation=n-match,observed_percent=100*(n-match)/n,expected_matches=expected,expected_nonrepresentation_percent=100*(n-expected)/n))
 out=pd.DataFrame(rows);out.to_csv(OUT/'uniform_choice.csv',index=False)
 a=out[(out.scope=='multi')&~out.exclude_none]
 assert a.n.tolist()==[350,648,487,92]
 assert a.nonrepresentation.tolist()==[248,525,122,11]
 assert np.allclose(a.expected_nonrepresentation_percent,[57.0,54.1,55.3,54.2],atol=.05)
 b=out[(out.scope=='multi')&out.exclude_none]
 assert b.loc[b.domain.eq('Emotional abuse'),['n','nonrepresentation']].values.tolist()==[[616,493]]
 print(a.to_string(index=False));print(b.to_string(index=False))
 return out

def threshold(zips):
 scores=[];labels=[];hashes={}
 for path in zips:
  path=Path(path);hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
  with zipfile.ZipFile(path) as z:
   for name in sorted(z.namelist()):
    if not name.endswith('.json'):continue
    obj=json.loads(z.read(name).decode('utf-8-sig'))
    value=obj['info']['학대의심'].strip().replace('(','').replace(')','')
    labels.append(KO.index(value) if value in KO else 4)
    found={x['항목']:x['점수'] for s in obj['list'] if s.get('문항')=='학대여부' for x in s['list'] if x.get('항목') in KO}
    scores.append([found[d] for d in KO])
 s=np.asarray(scores);labels=np.asarray(labels);assert s.shape==(3236,4)
 baseline=s>=np.array([4,5,5,5])
 maxima=baseline & (s==np.where(baseline,s,-1).max(axis=1,keepdims=True))
 tie_n=maxima.sum(axis=1)
 credit=maxima/np.maximum(tie_n[:,None],1)
 benchmark=[]
 for scope,mask in [('all',baseline.sum(axis=1)>=1),('single',baseline.sum(axis=1)==1),('multi',baseline.sum(axis=1)>=2)]:
  for d,e in enumerate(EN):
   included=mask&baseline[:,d];n=int(included.sum());expected=float(credit[included,d].sum())
   benchmark.append(dict(scope=scope,domain=e,n=n,expected_matches=expected,
       expected_nonrepresentation_percent=100*(n-expected)/n))
 highest=pd.DataFrame(benchmark);highest.to_csv(OUT/'highest_score_choice.csv',index=False)
 assert highest[(highest.scope=='multi')].n.tolist()==[350,648,487,92]
 print(highest.to_string(index=False))
 rows=[]
 for scheme,cut in [('baseline',[4,5,5,5]),('common6',[6]*4),('common7',[7]*4)]:
  y=s>=cut;multi=y.sum(axis=1)>=2
  for d,e in enumerate(EN):
   a=y[:,d];b=labels==d
   sn,tn=int(a.sum()),int(b.sum());sm,tm=int((a&multi).sum()),int((b&multi).sum())
   rows.append(dict(scheme=scheme,domain=e,score_n=sn,type_n=tn,shared=int((a&b).sum()),score_multi=sm,type_multi=tm,score_multi_percent=100*sm/sn,type_multi_percent=100*tm/tn,difference_pp=100*(sm/sn-tm/tn)))
 out=pd.DataFrame(rows);out.to_csv(OUT/'threshold_composition.csv',index=False)
 (OUT/'source_hashes.json').write_text(json.dumps(hashes,indent=2))
 assert out[out.scheme.eq('baseline')].shared.tolist()==[228,339,500,264]
 print(out.to_string(index=False))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--raw-zips',nargs=2);args=p.parse_args();OUT.mkdir(exist_ok=True)
 uniform()
 if args.raw_zips:threshold(args.raw_zips)
